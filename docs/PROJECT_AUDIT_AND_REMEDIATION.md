# 项目问题账本与逐步修复计划

> 建立日期：2026-08-20  
> 范围：当前 `GlobexAgentLearning` 代码、运行配置、评测产物与架构/迁移文档。  
> 工作规则：同时只处理一个最高优先级问题；未进入当前步骤的问题只记录，不顺手扩改。

## 1. 已定决策

### D-001 正式多 Agent 架构

正式架构保留异构 `search_agent` / `trade_agent`，主 Agent 持全量工具并默认单干，复杂且可并行的子任务才通过 `task_dispatch` 派发。课程中的同质 fork 仅保留为教学对比和后续实验参考，不再作为当前项目的迁移目标，也不把现有异构实现视为缺陷。

### D-002 商品价格语义

商品级 `price_cny` 只表示无需选择规格即可确定的商品价格。多规格多价格商品不得把最低价冒充成交价；选择规格后，订单必须使用该规格的价格和可售状态快照。

### D-003 修复顺序

本轮只处理 P0-001“运行时检索链与已评测主链不一致”。P0-002 规格/下单问题已经完成根因定位和统一契约草案，但本轮不修改其代码或数据。

### D-004 LangGraph Redis Checkpoint 主线

正式本地主线是 FastAPI 直接执行 LangGraph、进程内任务管理和一个共享官方
`langgraph-checkpoint-redis.AsyncRedisSaver`。Redis Stream 独立 worker 仍是可选部署能力，默认
`QUEUE_ENABLED=0`，不与 checkpoint 混用。官方 saver 管理内部 Redis key；应用侧只为 child done/result
marker 使用 `globex:{env}:v1:*` 前缀。

Main thread ID 使用 shopping session ID 的 HMAC 摘要；child ID 使用父 ID、agent role 和稳定
`tool_call_id`/dispatch ID；顶层 `checkpoint_ns` 固定为空字符串。用户重试同一 thread 恢复，未完成子任务重入同一
child thread；已完成子任务先读独立 marker，避免工具重放二次执行。同一进程同一 session 使用 asyncio 串行锁，
不同 session 可并行。

当前 0.5.2 官方 saver 没有 serializer 注入参数。本地不设 key 可以运行；配置 `LANGGRAPH_AES_KEY` 或在非本地
环境启动会诚实 fail-fast，不宣称已加密，待官方支持/升级。旧 JSON/SQL SessionStore bridge 保留兼容但不自动迁移，
正式 composition/runtime 不再使用。

## 2. 当前运行时检索链与文档设计

### 2.1 修复前实际链路

```text
原 Query
  → BGE-M3 Query embedding
  → 分区 Faiss HNSW/IP 全局合并 Top-8
  → 只读取这 8 个 StandardItem
  → BGE Reranker 只重排这 8 个
  → availability / ship_to / price cap 硬过滤
  → requested Top-K（默认 5）
```

证据是 `src/globex_agent/application/usecases/catalog_search.py` 原来的 `_RECALL_TOP_N = 8`。因此第 9～100 位的相关商品在进入 reranker 前已经丢失，reranker 不可能复现离线评测中把深位候选救回 Top-10 的收益；过滤发生在 Top-8 之后还会继续缩短结果，且无法补位。

向量链不可用时会扫描全商品库做关键词降级。返回值沿用 `keyword_bm25` 名称，但当前实现是空白/CJK bigram 的词项重叠打分，并非真正 BM25；该命名问题单独列为 P2-005。

### 2.2 文档和评测的目标链路

`ENGINEERING_PLAN.md`、`README.md`、`docs/experiments/phase5_bge_m3_retrieval_2026-08-15.md` 和 `docs/data/multiplatform_catalog.md` 的商品主线都是：

```text
BGE-M3 Query/Item ANN Top-100
  → BGE-Reranker-v2-m3 对候选池联合打分
  → Top-10 计算 Recall/MRR/NDCG
```

Top-100 是粗召回候选深度；Top-10 是离线评测截断位置，不表示 reranker 只看 10 条。应用运行时应该让 reranker 看最多 100 条，再做确定性硬约束并返回调用方请求的 Top-K。当前 `ProductSearchSpec` 默认 `top_k=5`，尚未统一最大值约束，记入 P1-007。

### 2.3 本轮修复后的链路

```text
原 Query
  → BGE-M3 Query embedding
  → 分区 Faiss HNSW/IP 全局合并 Top-100
  → 读取新增 StandardItem → ConstraintEvaluator 硬过滤
  → 合格数不足 top_k 时扩到 Top-200 → 400 → 最大 500
  → BGE Reranker 重排完整合格候选池
  → requested Top-K
```

每轮只处理新 `item_id`，达到数量、索引耗尽或 500 即停止；不足时返回部分结果及
`requested_top_k / ann_recall_depth / ann_candidate_count / eligible_count / filtered_count /
returned_count / is_partial / filtered_count_by_reason / exhaustion_reason`。Reranker 永远只接收
合格候选，不能通过精排把违规商品带回结果。

`ConstraintEvaluator` 只处理可靠事实：`is_available`、明确提供的
`attributes.ships_to`、商品级明确价格上限、以及调用方明确给出的平台、locale、品牌和
类目。locale/platform 推导出的配送范围只用于展示；当 `ship_to` 没有明确事实时严格以
`ship_to_unknown` 过滤。裸 `attributes.material` 和裸 variant dict 尚无规范化 typed schema，
不参与本轮硬过滤。

索引与 runtime reranker 改为共用 SearchDocument：`title → brand → category_path → searchable
attributes → variant option name/value → description`。它排除价格、库存、SKU/variant ID、配送、
URL、时间戳和 provenance；attributes 的任意嵌套层也执行递归清除。版本升至
`item-text-v4-recursive-search-fields-only`，启动会明确跳过旧
manifest 的 Faiss 索引并要求重建，避免静默的向量空间不匹配。

## 3. 淘宝商品 67.9% 无法下单的结论

### 3.1 数据事实

当前淘宝规范化库共有 23,421 个商品：

- 7,513 个商品只有一个明确观测价格，允许设置商品级 `price_cny`。
- 15,896 个商品存在多个规格价格，另有 12 个没有有效价格；合计 15,908 个商品的商品级 `price_cny` 为 `null`，占 67.9%。
- 23,421 个商品都保留了 `variants`；共 239,925 条规格记录。
- 当前真实规格字段使用 `variant_id / option_name / value / price_cny / is_available`；`spec` 和 `name` 在这些记录中均不存在。

这个 67.9% 主要不是脏数据，而是有意避免把最低规格价伪装为最终成交价，设计依据见 `docs/data/multiplatform_catalog.md` 的“价格处理”。

### 3.2 真正故障点

问题属于“正确的数据语义 + 不完整的领域契约和订单代码”：

1. `StandardItem.variants` 仍是 `list[dict[str, Any]]`，没有统一、可校验的规格模型，字段漂移无法在入库时失败。
2. `catalog_search._variant_cards()` 读取 `spec/name`，而淘宝数据写的是 `option_name/value`，所以规格展示为空。
3. `order_usecases._item_price()` 只读商品级 `price_cny`，没有根据用户选择的 `variant_id` 取规格价，因而拒绝全部 15,908 个多价格商品。
4. `_resolved_variant_id()` 只检查 ID 是否存在，没有检查该规格的 `is_available`。
5. `_line_title()` 同样读取 `spec/name`，订单行无法稳定展示所选规格。

结论：不应把 15,908 个商品的最低规格价回填到商品级价格；应修统一规格契约和订单解析代码。

### 3.3 下一步统一字段草案（P0-002）

```text
StandardItemVariant
  variant_id: str
  options: list[VariantOption]
    - name: str
      value: str
  display_name: str
  price_cny: Decimal | null
  currency: Currency
  is_available: bool
```

统一规则：

- 各来源的 `option_name/value`、`spec`、`name` 等别名只允许在 ingestion adapter 中转换。
- 进入 `StandardItem` 后只允许上述 canonical 字段，检索、前端和订单代码不得继续兼容多套别名。
- 无规格商品由订单层生成明确的 default variant，不能和真实 SKU 混用。
- 下单先解析并校验 `variant_id`，再快照该规格的价格、币种、标题和可售状态。
- 商品卡可另提供 `price_range_cny` 供展示，但它不是订单成交价。

## 4. 问题清单

状态说明：`已修复`、`待处理`、`观察项`、`已定案`。

| ID | 优先级 | 状态 | 问题 | 主要证据/影响 |
|---|---|---|---|---|
| P0-001 | P0 | 已修复并验证 | runtime Top-8 与 ANN 主链不一致，过滤后不能补候选 | `catalog_search.py` 现执行 100→200→400→500 渐进 ANN，前置事实型硬过滤、后置 rerank；正式四分区 v5 Faiss 已重建并通过结构/真实模型冒烟。 |
| P0-002 | P0 | 已实现并验证 | typed catalog-schema-v2 已统一规格、材质、三态 availability、商品卡和订单快照；淘宝 staging 保留 23,421 商品 / 239,925 源规格、239,913 规范 variant | 迁移 hash/重复 ID/未知价格见 `docs/data/catalog_schema_v2_migration_20260820.md`；正式索引已切换为 `item-text-v5-catalog-schema-v2`。 |
| P0-003 | P0 | 已修复并验证 | 写操作确认边界依赖 LLM/提示词，没有不可绕过的确定性门禁 | `OrderConfirmationService` 实现 PrepareOrder/ConfirmOrder、PrepareCancel/ConfirmCancel；token 只存 hash，默认 TTL 300 秒，session/user/action/payload SHA256/事实重读/一次性原子 claim 均在应用层校验；LLM 仅能准备，可信 API 才能确认。 |
| P0-004 | P0 | 已实现并通过真实 Redis 验收 | LangGraph checkpoint 仍依赖进程内 saver/旧 SessionStore bridge | 共享官方 `AsyncRedisSaver`、节点级 `durability=sync`、7 天 TTL、稳定匿名 thread、子任务 done marker、同 session 进程内锁；Redis 8.2.8 真实集成测试通过。当前 0.5.2 无 serializer 注入，配置加密 key 仍按设计诚实 fail-fast。 |
| P1-001 | P1 | 待处理 | Redis Stream 声称支持 pending 重投，但消费循环只读取 `>` 新消息 | worker 异常后 pending 消息不会自动被其他消费者 claim；需 `XAUTOCLAIM`/`XCLAIM`、重试次数和 DLQ 集成测试。 |
| P1-002 | P1 | 部分修复，仍待处理 | 会话串行化、全局模型限流和子任务并发边界不完整 | 当前单 FastAPI 进程按匿名 session 使用 asyncio 串行锁；多进程分布式锁、全局模型限流和并发预算仍未完成。 |
| P1-003 | P1 | 待处理 | 熔断器可能把业务校验错误当系统故障，half-open 竞争控制不足 | 需要只统计超时/连接/5xx 等可重试故障，并保证同一依赖只有一个 half-open probe。 |
| P1-004 | P1 | 待处理 | 上下文压缩只做粗粒度截断，配置项未真正接线 | `tool_result_limit`、`reply_token_budget` 已配置但未形成统一 token budget；长工具结果和历史仍可能挤爆上下文。 |
| P1-005 | P1 | 待处理 | 语义缓存 namespace 硬编码 `prompt-v1`，缺少数据/规则指纹 | prompt、模型、商品索引、汇率、税费规则改变后可能复用陈旧答案。 |
| P1-006 | P1 | 待处理 | Agent 评测样本和判分不足以支撑“系统可用”结论 | 固定少量 case；真实 flow 完整通过率偏低；judge 曾接受未由工具证明的门店事实；需扩大冻结集并做事实来源门禁。 |
| P1-007 | P1 | 待处理 | 商品检索外部 Top-K 契约不统一 | `SearchRequest` 最大 50，而 `ProductSearchSpec` 仅要求正数；需统一默认值、最大值和过滤后补位语义。 |
| P1-008 | P1 | 待处理 | 跨平台/locale 检索路由未进入应用主链 | 已有 router、fusion、canonical 等基础设施，但 composition 的 `CatalogSearchUseCase` 只使用分区 Faiss 合并，没有按 platform/locale 约束查询。 |
| P1-009 | P1 | 部分修复，仍待处理 | Docker/部署产物与本机运行条件不闭合 | compose 已切到固定 Redis 8.2.8 并补 checkpoint 配置；`.dockerignore` 排除 `output`、OpenSearch 初始化和模型/索引镜像闭合仍待处理。 |
| P1-010 | P1 | 待处理 | 测试依赖被忽略的大型本地数据，fresh clone 可复现性不足 | 部分测试/运行路径依赖 `data/processed` 与本地模型/索引；需小型提交夹具和 integration marker。 |
| P1-011 | P1 | 待处理 | 订单存储和 API/worker 组合存在一致性风险 | 默认文件模式订单仓储可能只驻内存；多进程间不可共享；SQL `count + 1` 型 ID 需数据库唯一序列/重试保证。 |
| P1-012 | P1 | 待处理 | 商品 runtime reranker 没有使用已配置的独立 Python/GPU worker | `Settings.bge_reranker_python` 已存在，但 composition 固定创建 CPU `BgeReranker`；真实 Top-100 单查询耗时 62.096 秒。需接入常驻子进程、超时和性能回归。 |
| P2-001 | P2 | 待处理 | WebSocket 只有实时订阅，无断线重放和有界背压 | 断线期间事件丢失；订阅队列无大小上限；前端还应以 HTTP 最终响应兜底。 |
| P2-002 | P2 | 待处理 | streaming 对消息块类型过滤不足 | 当前从 message chunk 读取所有字符串 content，需确认只发布 AI 文本，避免 tool/system 内容混入 token 流。 |
| P2-003 | P2 | 待处理 | PII、会话订阅和输出安全治理不足 | 工具事件/对话日志可能带地址电话；WebSocket session 订阅缺身份绑定；缺 prompt-injection/敏感输出门禁与脱敏策略。 |
| P2-004 | P2 | 待处理 | DDD 洋葱边界并不完全成立 | application 层直接依赖部分 infrastructure 实现；应逐步经端口注入，但不在业务修复中顺手大重构。 |
| P2-005 | P2 | 待处理 | `keyword_bm25` 诊断名称不准确 | runtime 降级是 token overlap + CJK bigram，不是 BM25；要么接真实 BM25/FTS，要么改名并保持可观测性诚实。 |
| P2-006 | P2 | 待处理 | 文档权威顺序和描述存在冲突/过期 | `AGENT_SYNC.md`、迁移 handoff、合并方案、工程计划曾对同质/异构、旧工具去留和 runtime 检索链给出不同说法。 |
| P2-007 | P2 | 观察项 | `create_react_agent` 已被上游标记 deprecated | 当前为忠实课程/现有 LangGraph 路径可暂留；升级前补等价回归测试，不在当前问题中迁移框架。 |
| P2-008 | P2 | 待处理 | 长期记忆缺少更新、删除、冲突和相关性治理 | 现有 append/list 能力不足，且写入内容可能受 prompt injection 污染；需要结构化记忆与 provenance。 |
| P2-009 | P2 | 待处理 | 健康检查不能证明检索产物完整可用 | 需要校验索引 manifest、商品库 hash、分区数量、模型文本版本和 OpenSearch 文档数，而不仅是组件对象存在。 |

## 5. 推荐执行顺序

1. P0-001（已完成）：runtime Top-100 → reranker 对齐、文本一致、契约测试、真实模型单查询回归。
2. P0-002：建立 typed variant 契约，迁移淘宝数据适配，修商品卡与订单，增加 67.9% 场景测试。
3. P0-003（已完成）：把交易确认变成不可绕过的确定性安全门。
4. P1-001～P1-005：可靠性、并发、上下文和缓存正确性。
5. P1-006～P1-011：评测、检索路由、部署与存储可复现性。
6. P2：体验、安全、架构洁净度和文档治理。

每完成一项，都在本文件更新状态、实际执行命令、结果和未覆盖风险；不因代码生成而标记完成。

## 6. P0-001 验证记录

实际执行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_application_usecases.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

结果：该段为早期基线记录；本次 Redis checkpoint 最终验收结果见下方第 8 节。

真实模型单查询使用本地四分区 Faiss 索引、BGE-M3 和 BGE Reranker，查询“适合地铁通勤的头戴式主动降噪耳机”：

```json
{
  "recall_strategy": "embedding_rerank",
  "rerank_applied": true,
  "returned_hits": 10,
  "total_candidates_after_filters": 100,
  "first_item_id": "taobao:cn:908889716378",
  "elapsed_seconds": 62.096
}
```

这证明 runtime 已真实执行 Top-100 候选精排；它不是完整冻结评测集的指标复跑，因此不更新 Recall/MRR/NDCG。62.096 秒的 CPU 延迟作为 P1-012 单独处理。

## 7. P0-003 验证记录

实际写入口盘点：`TradeAgentFactory` 只注册 `prepare_order_tool`、
`prepare_cancel_order_tool` 和 `query_order_tool`；旧 `create_order_tool`/
`cancel_order_tool` 兼容名只返回禁用错误。FastAPI 可信入口为：

- `POST /commerce/orders/prepare` → `POST /commerce/orders/confirm`
- `POST /commerce/orders/{order_id}/cancel` → `POST /commerce/orders/cancel/confirm`

`PlaceOrderUseCase.execute` 与 `CancelOrderUseCase.execute` 也会直接拒绝，只有
`OrderConfirmationService` 的内部 confirmed 路径可执行写入。确认意图保存
`confirmation_id`、token SHA-256、session/user/action、canonical payload、payload SHA-256、
preview、状态、TTL、幂等键和 consumed_at；默认 TTL 为 300 秒。原始 token 只在可信 prepare
API 响应返回，不进入 LLM tool 返回、事件或普通回复。

定向与全量验证覆盖错误 token、跨 session、过期、payload 篡改、价格/可售状态变化、一次性
重放、并发双击、正常下单/取消以及旧直调拒绝；全量 `pytest` 为 `172 passed, 1 warning`。
Ruff、compileall、`git diff --check` 全通过。当前原子门禁为进程内 `asyncio.Lock`；多进程/重启
持久化确认意图和订单 ID 一致性仍属于 P1-011，不在本次顺手扩展。

## 8. Redis Checkpoint 迁移最终验收记录

实现与静态检查：

```powershell
uv --cache-dir .uv-cache run --no-sync python -m compileall -q src tests
uv --cache-dir .uv-cache run --no-sync ruff check src tests
docker compose -f docker/docker-compose.yaml config --quiet
git diff --check
```

以上四项均通过。核心迁移单测：

```powershell
uv --cache-dir .uv-cache run --no-sync pytest -q tests/unit/test_checkpoint_migration.py tests/unit/test_application_agents.py tests/unit/test_application_tools.py
```

结果：`26 passed in 9.31s`。

真实 Redis 集成测试使用 `redis.asyncio`，强制连接 `redis://127.0.0.1:6379/0`，未使用 fakeredis。官方 saver 的 RediSearch 索引只能建立在 DB 0，测试用唯一 thread ID 隔离：

```powershell
$env:GLOBEX_REQUIRE_REAL_REDIS='1'
$env:GLOBEX_REAL_REDIS_URL='redis://127.0.0.1:6379/0'
uv --cache-dir .uv-cache run --no-sync pytest -q tests/integration/test_redis_checkpoint_real.py
```

结果：`6 passed, 8 warnings in 4.17s`。Redis 容器为官方 `redis:8.2.8-bookworm`，状态为
`healthy`；JSON.SET/GET、FT.CREATE/SEARCH 命令验证通过。6 项覆盖节点级 history、重建 saver/graph
恢复、TTL/refresh、thread 隔离、真实 child marker、pending/interrupt 重入和 Redis 不可用 startup fail-fast。
中断用例采用官方静态 `interrupt_before`，因为当前 Python 3.10 下动态 `interrupt()` 异步上下文传播不受
当前 LangGraph 支持。

完整测试：

```powershell
uv --cache-dir .uv-cache run --no-sync pytest -q
```

结果：`192 passed, 9 warnings in 17.24s`（真实 Redis 已启用）。server 测试已通过测试专用
容器注入，生产 `build_container` 仍保持 Redis fail-fast。此前 C 盘 worktree 缺 Amazon catalog SQLite 的
2 项失败与中文路径导致的 1 项 Faiss 写入失败，已在 `D:\PycharmProjects\GlobexAgentLearning` 的真实数据、
ASCII 路径环境中单独复验为 `3 passed in 0.45s`。本轮新增回归为 0，真实 Redis 集成不再跳过。
