# Globex 电商搜索 Agent：整体设计与实现说明

> 文档日期：2026-08-21
> 分支：`codex/migrate-langgraph-ddd`  
> Python：3.10.20  
> 用途：让新读者通过本文了解项目当前的全部设计与实现，并能提出改进建议。

## 1. 项目定位

GlobexAgentLearning 是一个用于学习、运行和逐步组装电商搜索 Agent 的项目。它不是大型电商系统，而是把课程中的关键能力做成一个可本地运行、可评测、可继续演进的完整工程。

当前核心能力（已实现与待验收项分开理解）：

- 用 LangGraph 的 ReAct Agent 完成“理解需求 → 调工具 → 最终回复”的闭环
- 主 Agent 默认单干，复杂任务通过 `task_dispatch` 并行调用 `search_agent` / `trade_agent`
- 商品召回使用 BGE-M3 + Faiss + BGE Reranker
- 品类洞察使用 OpenSearch Hybrid + `CategoryCard`
- 支持订单创建、查询、取消
- 有基础长期买家偏好存取；CRUD、相关性、冲突、provenance、确认和安全治理仍未完成
- 支持多轮会话持久化、事件流、语义缓存、工具熔断、模型回退；上下文当前主要是消息数截断，四层上下文、L0-L4、统一 token budget 和 Cache Breakpoint 尚未完成
- 提供 FastAPI + WebSocket + React 前端

本机 4GB GPU 验收 profile（2026-08-21）：Query Embedding 使用本地
`D:/models/bge-m3` 的 CPU FP32 常驻 worker；Reranker 使用本地
`D:/models/bge-reranker-v2-m3` 的 CUDA FP16 常驻 worker，`batch_size=16`。
商品主链仍是 Faiss ANN Top-100 → BGE Reranker → Top-10；只切换运行设备，
不改变模型、索引或召回逻辑。启动时预热 CategoryInsight CPU encoder、Embedding
worker 和 Reranker worker；请求取消会主动终止当前子进程，避免后台线程长期占用
请求锁。真实基准中 CPU/GPU Query Top-100 重叠为 3/3=1.0；GPU FP16 Reranker
batch 4/8/16 均未 OOM，中位分别为 1788.322/1695.567/1620.227 ms，选择 16；
CPU Query → Faiss Top-100 → GPU Reranker 组合三次中位为 1820.656 ms。完整原始结果在
`output/eval/local_4gb_profile.json`，不能把该本机延迟外推为生产 SLA。

项目边界：

- 商品领域模型统一使用 `StandardItem`，不引入参考仓库的 `Product/Sku`
- 不引入 AgentScope、Qdrant、vLLM、K8s 等重型组件
- 默认本地文件持久化，`DATABASE_URL=file`
- 真实密钥只放本地 `.env`，仓库只提交 `.env.example`

## 2. 总体架构

### 2.1 洋葱架构

项目采用 DDD 风格的洋葱架构，依赖方向从外向内：

```text
presentation / composition
        ↓
application（usecases / tools / agents / orchestrator）
        ↓
domain（模型、规则、端口）
        ↑
infrastructure（实现 domain 端口）
```

- `domain` 只放纯 Python 规则、数据模型和抽象端口，不依赖 LangChain、FastAPI、OpenSearch。
- `application` 负责用例、工具、Agent 图、事件发布和编排。
- `infrastructure` 实现领域端口：召回、向量、持久化、缓存、队列、LLM 封装。
- `presentation` 负责 HTTP / WebSocket / DTO。
- `composition.py` 是组合根，负责把所有实现装配成 `Container`。

### 2.2 一次请求的主链路

```text
POST /commerce/intents
  → MainAgentOrchestrator.handle_intent
    → SessionRegistry.get_or_create（恢复/新建 LangGraph checkpointer）
    → 可选 context.compressed
    → MainAgent.reply
      → LangGraph ReAct Agent
        → 模型决定调用 product_search_tool / category_insight_tool /
          order_tools / remember_preference_tool / task_dispatch
        → 工具结果返回模型
      → final.result
    → conversation_store 记录对话和事件
```

商品事实采用单向证据链：

```text
StandardItem → ProductFactSnapshot → ProductCard → LLM 最终回答
```

- `ProductFactSnapshot` 是内部、版本化、可审计的证据 DTO，保存商品/变体精确事实、价格与变体绑定、店铺/库存、provenance、hash 及 `exposed_facts`。
- `ProductCard` 是给 Search Agent/LLM 的紧凑结构化投影，只增加 `evidence_id` 作为引用，不承载整份原始商品或审计状态。
- 最终回答只能使用 Snapshot 明确暴露给 Card 的 facts；Fact Guard 不因 Snapshot 内部存在某字段就放行未暴露事实。

商品工具在同一次构造中生成 Card 与 Snapshot。模型只收到 Card 和 evidence ref；审计/评测事件保存 schema-aware Snapshot 白名单，避免通用审计深度限制把 `variants`/`highlights` 截成 `[omitted]`。

### 2.2.1 在线生产链与离线质量链

两条链必须分开理解。在线生产链负责完成一次用户请求；Fact Guard、P0 门禁和
LLM Judge 不在在线返回路径中。

在线生产链：

```text
Query
  → Orchestrator / context / cache
  → Main LLM 工具规划
  → 可选 CategoryInsight（品类 aggregate/reference）
  → ProductSearch
  → StandardItem
  → ProductFactSnapshot + ProductCard
  → LLM final answer
  → audit / ConversationStore 持久化
```

CategoryInsight 是由 LLM 选择的可选工具，不直接参与商品硬过滤或商品排序；商品硬约束
和商品 rerank 在 `CatalogSearchUseCase` 内完成。`ProductFactSnapshot` 从
`StandardItem` 确定性生成，不是 LLM 生成的证据。

离线质量链：

```text
persisted conversation / audit
  → exposed-evidence parser
  → deterministic Fact Guard
  → P0=0 评测门禁
  → stable evidence catalog
  → rubric generator
  → LLM Judge
  → local schema/ID validation
  → validated report
```

当前“P0=0 后才运行 Judge”是评测执行规范，不是在线生产状态机，也尚未被普通评测命令
实现为不可绕过的脚本门禁。

### 2.2.2 未实现的下一阶段可信回答设计

以下流程是下一阶段设计，当前未实现，不应描述为在线已有能力：

```text
ProductFactSnapshot / CategoryInsight
  → LLM draft answer
  → 在线 Fact Guard
      ├─ 通过 → 返回
      └─ 失败 → grounded rewrite
                   → 再次 Fact Guard
                       ├─ 通过 → 返回
                       └─ 失败 → deterministic fallback
```

当前实现只有离线 `validate_final_response`；没有在线 final-answer Fact Guard、grounded
rewrite 或确定性 fallback。

### 2.3 事件模型

事件总线是 `TradeEventBus`，按 `shopping_session_id` 路由。事件类型固定为 12 类：

```text
agent.dispatch / tool.invoke / tool.result / token.delta / plan.update
context.compressed / model.fallback / cache.hit / task.queued / task.started
final.result / error
```

事件既被 WebSocket 推给前端，也会在启用了 Redis 背板时跨进程广播。

### 2.4 并发与上下文

- 使用 `contextvars` 的 `ShoppingContext` 保存当前任务的 `shopping_session_id / buyer_id / locale / currency`
- 每个 asyncio Task 上下文隔离，多用户并发不会串台
- FastAPI lifespan 创建一个共享的官方 `AsyncRedisSaver`，Main/Search/Trade 图共享它；不按请求创建连接
- `SessionRegistry` 缓存 Agent，并按匿名 session 使用进程内 `asyncio` 锁串行执行；不同 session 可并行
- Redis checkpoint 在 LangGraph 节点级写入，服务重启后用户重试同一稳定 `thread_id` 即恢复；不做启动扫描
- checkpoint 不是逻辑锁：Redis 持久化解决恢复，锁解决同一进程内并发状态/写操作交错

## 3. 目录结构

```text
src/globex_agent/
├── domain/
│   ├── catalog/          # StandardItem、Money、汇率、检索端口
│   ├── shipping/         # 运费/关税规则
│   ├── order/            # Address、OrderLine、Order、OrderRepository
│   ├── buyer/            # 长期偏好
│   ├── session/          # 会话/对话存储端口
│   └── queue/            # 任务队列端口
├── application/
│   ├── usecases/         # catalog_search、order_usecases
│   ├── tools/            # 业务工具与韧性包装
│   ├── agents/           # Main/Search/Trade Agent、checkpoint、编排器
│   └── prompts/          # globex.yml
├── infrastructure/
│   ├── llm.py            # OpenAI-compatible 模型与回退
│   ├── eventbus.py       # 事件总线
│   ├── resilience.py     # 超时与熔断
│   ├── context.py        # ContextVar 会话上下文
│   ├── persistence/      # JSON 文件、SQLite
│   ├── cache/            # 语义缓存
│   ├── queue/            # Redis Stream
│   ├── recall/           # 商品/知识卡检索
│   ├── embedding/        # BGE-M3 异步适配器
│   ├── vector/           # Faiss 异步适配器
│   └── rerank/           # BGE Reranker 异步适配器
├── category_insight/     # CategoryCard、知识卡服务、精排
├── presentation/         # FastAPI、WebSocket、DTO
├── composition.py        # 组合根
└── worker.py             # Redis 队列 worker
```

## 4. 领域层

### 4.1 `StandardItem`

文件：`src/globex_agent/domain/catalog/models.py`

`StandardItem` 是商品规范模型，字段包括：

```text
item_id / same_group_id / platform / locale / language
title / description / brand / category_path
price_cny / original_price_cny / currency_raw / price_source
rating / review_count / attributes / variants
availability / provenance / source_updated_at / ingested_at
```

关键约束：

- `item_id` 必须以 `platform:` 开头
- 有 `price_cny` 时不能标 `price_source=unavailable`
- 缺失 `price_cny` 时必须显式 `price_source=unavailable`
- `original_price_cny` 不能低于 `price_cny`
- provenance 必须带时区

商品检索输出契约保留课程中的 `Candidate → PricePoint → LandedCost → PickedItem` 概念，但实际工具输出使用的是 `ProductCard`。

### 4.2 `Money` / `ExchangeRateTable` / `TariffSchedule`

文件：

- `src/globex_agent/domain/catalog/money.py`
- `src/globex_agent/domain/catalog/exchange_rate.py`
- `src/globex_agent/domain/shipping/tariff_schedule.py`

`Money` 使用最小货币单位（分）存储，避免浮点误差。

`ExchangeRateTable` 是静态汇率快照，生产可替换为实时汇率服务。

`TariffSchedule` 是纯规则函数：

```text
landed = subtotal + freight + tariff
tariff = max(0, subtotal - de_minimis) * rate
```

不同目的国使用不同基础运费、免税额度、品类税率。寄往中国时：

- 单件基础运费 25 元
- 个人物品免税额度内关税为 0
- 续件按首件 60% 递增

### 4.3 订单聚合

文件：

- `src/globex_agent/domain/order/order.py`
- `src/globex_agent/domain/order/order_line.py`
- `src/globex_agent/domain/order/address.py`

订单状态机：

```text
DRAFT → CONFIRMED → CANCELLED
```

订单创建仍在领域层进入 `CONFIRMED`，取消只能从 `CONFIRMED` 到 `CANCELLED` 并必须带 reason；
但用户写操作必须经过 `OrderConfirmationService` 的 Prepare → 可信 Confirm 两阶段门禁。

订单行使用 `item_id + variant_id`，并对下单时价格做快照。金额只通过 `Order.snapshot()` 输出，不允许 Agent 自行计算。

### 4.4 买家偏好

文件：`src/globex_agent/domain/buyer/preference.py`

`BuyerPreference` 支持 `like` / `dislike`，偏好按 `buyer_id` 持久化，同 buyer 同 statement 幂等去重。

### 4.5 会话与队列端口

- `SessionStore`：legacy AgentState bridge，仅为兼容旧代码保留；正式 checkpoint 由官方 Redis saver 管理，composition 不装配它
- `ConversationStore`：保存业务可读对话流水和过程事件
- `TaskQueue`：任务队列抽象，当前实现 Redis Stream

领域层只定义端口，基础设施提供文件、SQLite 或 Redis 实现。ConversationStore、订单仓储、偏好/长期记忆与 Agent
checkpoint 各自负责不同数据，不互相替代。

## 5. 应用层

### 5.1 `CatalogSearchUseCase`

文件：`src/globex_agent/application/usecases/catalog_search.py`

执行流程：

```text
ProductSearchSpec
  → vector recall（BGE-M3 + Faiss，100 → 200 → 400 → 最大 500）
  → 批量读取新增 StandardItem
  → ConstraintEvaluator 硬约束过滤
  → rerank（BGE Reranker，仅重排合格候选池）
  → requested Top-K ProductCard 列表
```

初始 Top-100 是运行时粗召回深度；过滤后的合格候选不足 `top_k` 时才逐级补到
200、400、500。每轮只处理新 `item_id`；达到数量、索引耗尽或 500 即停止。评测文档
中的 Top-10 是 Recall/MRR/NDCG 的截断口径。达到上限仍不足时返回部分结果和完整诊断。

召回策略：

- `embedding_rerank`
- `embedding_only`
- `bm25_fallback_*`（仅 ANN 故障或未配置时）

硬约束由可复用的 `ConstraintEvaluator` 执行：只接受三态 `availability`、明确的
`attributes.ships_to`、商品级明确价格上限、以及调用方明确给出的平台/locale/品牌/类目。
平台或 locale 推导出的配送范围仅作展示，不能证明 `ship_to`，因而会以
`ship_to_unknown` 诊断过滤。材质和规格价格/可售状态等待统一 typed schema 后再加入。

`SearchDocument` 是建索引和 runtime reranker 的唯一构造器，只含 title、brand、类目、
description、可搜索 attributes 和规格 option 名称/值；不含价格、库存、SKU/variant ID、
配送、URL、时间戳或 provenance（attributes 任意嵌套层同样递归清除）。其
`item-text-v5-catalog-schema-v2` 版本进入 manifest；旧 v4 索引拒绝加载。
应用启动会拒用 text-format 不匹配的持久化 Faiss 索引。

### 5.2 订单用例

文件：`src/globex_agent/application/usecases/order_usecases.py`

- `PlaceOrderUseCase`：只允许确认服务调用内部 confirmed 执行路径；公开 `execute` 直接拒绝绕过确认
- `QueryOrderUseCase`：按订单号查询
- `CancelOrderUseCase`：只允许确认服务调用内部 confirmed 执行路径；公开 `execute` 直接拒绝绕过确认
- `confirmation_usecases.py`：生成一次性随机 token（只存 SHA-256）、canonical payload 摘要、5 分钟
  默认 TTL、session/user/action 校验、价格/规格/配送/availability 重读和原子 claim/consume。

订单工具层只暴露 prepare 订单/取消和查询；ConfirmOrder/ConfirmCancel 只在 FastAPI 可信结构化入口，
不注册为 LLM tool。工具事件和返回预览不包含原始 token。

### 5.3 工具

工具目录：`src/globex_agent/application/tools/`

| 工具 | 作用 |
|---|---|
| `product_search_tool` | 标准化 query → 商品卡 |
| `category_insight_tool` | 品类知识卡 → 选购口径 |
| `web_search_tool` | Tavily 联网搜索，未配置 key 不注册 |
| `prepare_order_tool` | 读取事实并生成不可变订单预览 |
| `query_order_tool` | 查询订单 |
| `prepare_cancel_order_tool` | 生成取消预览 |
| `remember_preference_tool` | 写入长期偏好 |
| `task_dispatch` | 批量并行调度子 Agent |

所有工具都发布事件，异常统一返回 `[error] ...`。

`task_dispatch` 使用批量接口：

```json
{"dispatches": [{"subagent_type": "search_agent", "demands": "..."}]}
```

每个 dispatch 使用独立 `thread_id`，通过 `asyncio.gather` 并行执行，并发布 `agent.dispatch` 与 `tool.result` 时间区间。

### 5.4 Agent

- `main_agent.py`：主 Agent，默认单干，持有全部业务工具和 `task_dispatch`
- `search_agent.py`：商品检索子 Agent
- `trade_agent.py`：订单交易子 Agent
- `base.py`：`LangGraphAgent` 包装 `astream` / 原生 checkpoint / 上下文压缩
- `identity.py`：稳定匿名 main/child thread ID 与根 `checkpoint_ns`
- `session_lock.py`：按匿名 session 的进程内 asyncio 串行锁
- `orchestrator.py`：`MainAgentOrchestrator`，处理缓存、偏好注入、事件发布、对话存储

Agent 图使用 `langgraph.prebuilt.create_react_agent`，checkpointer 使用共享的官方
`langgraph-checkpoint-redis.AsyncRedisSaver`。`src/globex_agent/infrastructure/checkpoint.py`
负责 startup/asetup、严格 fail-fast、7 天 TTL、dispatch done/result marker 和 shutdown。

### 5.5 Checkpoint、身份与恢复

- Main thread ID：`globex:{version}:{env}:main:{HMAC(shopping_session_id)}`；不包含敏感原文。
- Child thread ID：由父 thread、`search_agent`/`trade_agent` role 和稳定 dispatch ID 派生。优先使用
  LangGraph 注入的 `tool_call_id`，重试/重建 child Agent 对象也复用同一 ID。
- 顶层 `checkpoint_ns` 固定为 `""`；Main/Search/Trade/并发子任务通过不同 thread ID 隔离，应用不耦合 saver 的内部 namespace。
- `task_dispatch` 的完成结果是应用侧独立 marker，不是 checkpoint。父工具重放时先读 marker，已完成子任务不二次执行。
- 当前恢复触发是用户重试/同 thread 再调用；不引入启动扫描，不把 Redis Stream worker 当作 checkpoint 恢复机制。
- 本地 checkpoint 默认保留 10080 分钟（7 天），读取时刷新。Redis 不可用、未配置 URL 或不满足加密策略时启动失败，禁止静默回退 `InMemorySaver`。
- `AsyncRedisSaver` 0.5.2 的真实 API 支持直接构造、`asetup()` 和异步上下文 `__aexit__()`；当前版本没有公开 serializer 注入。
  因此本地不设 `LANGGRAPH_AES_KEY` 可运行；配置 key 或非本地强制加密会拒绝启动，不能宣称 checkpoint 已加密，待官方支持/升级。
- `data/sessions/*.json` 与 SQL `SessionStore` bridge 默认明确不兼容、不自动迁移；需要旧数据时另行评估一次性迁移，不混入正式启动路径。

### 5.6 提示词

文件：`src/globex_agent/application/prompts/globex.yml`

提示词约定包括：

- 默认单干，复杂任务才派发
- 价格/库存/运费必须来自工具返回
- 下单/取消前必须先出确认卡；用户通过 `/commerce/orders/confirm` 或
  `/commerce/orders/cancel/confirm` 结构化入口提交 token，LLM 不能自我确认
- 禁止点名 filtered_out 的库外商品
- 默认只推荐 1 个最匹配商品，除非用户明确要求多个或对比

## 6. 基础设施层

### 6.1 LLM

文件：`src/globex_agent/infrastructure/llm.py`

所有模型请求走 OpenAI-compatible Chat Completions：

- `ChatOpenAI` 作为主模型
- 可选备用模型
- `GatewayThrottle` 控制并发与最小请求间隔
- `_FallbackRunnable` 只对瞬时错误触发 `model.fallback`

### 6.2 事件总线

文件：`src/globex_agent/infrastructure/eventbus.py`

`TradeEventBus` 是进程内发布订阅，支持：

- 按会话订阅
- 同步本地投递
- 可选 Redis 背板跨进程广播

### 6.3 韧性

文件：

- `src/globex_agent/infrastructure/resilience.py`
- `src/globex_agent/application/tools/resilient.py`

每个工具通过 `ToolResilienceMiddleware` 执行：

- 超时
- 连续失败熔断
- 半开探测
- 失败后返回 `[error]`
- 发布 `tool.result` 的 circuit 状态

默认超时示例：`product_search_tool` 15s，订单工具 10s，`task_dispatch` 180s。

### 6.4 上下文

文件：`src/globex_agent/infrastructure/context.py`

`ShoppingContext` 用 `ContextVar` 保存会话快照，工具和子 Agent 无需层层传参。

### 6.5 持久化

文件：

- `src/globex_agent/infrastructure/persistence/json_file_stores.py`
- `src/globex_agent/infrastructure/persistence/sql/`

`DATABASE_URL=file` 时：

- 偏好：`data/preferences/{buyer_id}.json`
- 会话：`data/sessions/{session_id}.json`
- 对话：`data/conversations/{session_id}.jsonl`

配置 SQLite 时，订单、偏好、会话、对话使用 SQL 实现。

### 6.6 语义缓存

文件：`src/globex_agent/infrastructure/cache/semantic_cache.py`

语义缓存把“买家问句 → 最终回复”按向量相似度复用：

- 默认相似度阈值 0.95
- 写操作、上下文依赖问句不缓存
- 按 buyer 分桶
- namespace 包含模型名与提示词指纹
- 失败回复不进缓存

跑真实模型评测时应关闭，否则评的是缓存而不是 Agent。

### 6.7 队列与 Worker

文件：

- `src/globex_agent/infrastructure/queue/redis_stream_queue.py`
- `src/globex_agent/worker.py`

`QUEUE_ENABLED=1` 时：

- `POST /commerce/intents/async` 入队
- worker 消费 Redis Stream
- `task.queued` / `task.started` / `final.result` 事件通过背板回到 API

这是可选的 Redis Stream 多用户部署能力，不是 LangGraph checkpoint。默认 `QUEUE_ENABLED=0`，本地主线由
FastAPI 直接执行 LangGraph；embedding/reranker GPU 常驻 worker 仍是另一条独立链路。

### 6.8 召回与向量

文件：

- `src/globex_agent/infrastructure/recall/`
- `src/globex_agent/infrastructure/embedding/bge_m3_embedding.py`
- `src/globex_agent/infrastructure/vector/faiss_product_index.py`
- `src/globex_agent/infrastructure/rerank/bge_reranker.py`

商品召回：

```text
BGE-M3 Query/Item Embedding
  → Faiss HNSW + Inner Product
  → BGE Reranker Top-K
```

支持分区商品库（Amazon US/ES/JP、Taobao CN）。

### 6.9 CategoryInsight

文件：

- `src/globex_agent/category_insight/models.py`
- `src/globex_agent/category_insight/service.py`
- `src/globex_agent/category_insight/reranking.py`

知识卡类型：

```text
bestseller / attribute / price_range
```

检索链路：

- OpenSearch `ik_max_word` 建索引
- Hybrid：KNN + BM25，按 query type 动态权重
- 可选 BGE Reranker，Top-30 → Top-8/15
- 首分 >= 0.92 时跳过精排
- 向量失败 → BM25；OpenSearch 失败 → 空结构 `confidence=0`
- 当前淘宝中文数据已按 category-cards-taobao-zh-v3 重建 48 张卡；bestseller 是目录高频款型代理，attribute 是目录样本出现率，price_range 是品类参考区间
- CategoryEvidenceRef、工具 source boundary、审计有界 evidence refs 和确定性 fact_guard 已落地
- 历史模块记录曾完成 50 条评测；其中“8 条基础 Flow 事实门禁仍未通过”属于旧批次描述，
  已被当前稳定 ID 权威基线取代。当前基线见 `output/eval/flow-rubric-luna-final-20260822.md`。

## 7. 表现层与前端

### 7.1 FastAPI

文件：`src/globex_agent/presentation/server.py`

接口：

```text
POST /commerce/intents
POST /commerce/intents/async
GET  /commerce/tasks/{task_id}
WS   /commerce/events
GET  /commerce/orders/{order_id}
POST /commerce/orders/{order_id}/cancel
GET  /health
```

`/health` 返回模型、数据库、Redis、语义缓存、队列状态。

### 7.2 WebSocket

`ConnectionManager` 按 `shopping_session_id` 订阅事件，并把事件推给前端。

### 7.3 前端

目录：`frontend/`

- React + Vite + TypeScript
- 左侧对话区、商品卡、事件时间线
- WebSocket 订阅事件流，断线自动重连
- 会话 ID / buyer ID 存 localStorage

### 7.4 一键启动

脚本：`scripts/start_dev.ps1`

- 自动启动后端并等待 `/health`
- 自动安装前端依赖并启动 Vite
- 按 `Ctrl+C` 统一停止

```powershell
.\scripts\start_dev.ps1
.\scripts\start_dev.ps1 -SkipFrontend
```

## 8. 评测体系与当前结果

### 8.1 自动化测试（历史阶段记录）

```text
历史 Redis checkpoint 验收曾为 192 passed；本轮知识卡聚焦测试为 50 passed，完整 pytest 为 199 passed, 9 warnings。
本轮 Ruff 结果为 All checks passed，不能将历史完整 pytest 结果冒充本轮 v3 数据验收。
```

Redis checkpoint 集成测试为 `6 passed`，使用真实 `redis:8.2.8-bookworm`，覆盖节点级 history、saver/graph
重建恢复、TTL 刷新、Main/Search/Trade/session 隔离、child done marker、pending/interrupt 重入和 Redis
故障 fail-fast。此前在 C 盘 worktree 缺 Amazon catalog SQLite 与中文路径导致的 3 个环境失败，已在具有
真实数据且路径为 ASCII 的 D 盘项目复验通过；完整测试无失败。

覆盖：领域模型、订单状态机、召回、知识卡、事件、熔断、模型回退、会话恢复、并发隔离、表现层。

### 8.2 基础 13 条真实模型回归（历史阶段记录）

文件：

- `scripts/eval_regression.py`
- `eval/cases.yaml`

结果：`13/13 PASS`，平均分 `1.000`

报告：`eval/report-20260819-223705.md`

### 8.3 Flow Query 全流程运行（历史阶段记录）

脚本：`scripts/eval_flow_queries.py`

把已有 RAG/召回 query 直接喂给完整 Agent 服务。当前 9 条：

- 3 条中文 CategoryInsight query
- 3 条中文商品召回 query
- 3 条英文 ESCI query

结果：`9/9` 正常完成，平均耗时约 20.6 秒/条。

报告：`eval/flow-report-20260819-232003.md`

历史上曾跑过真实 OpenSearch + CategoryInsight RAG 的 8 条 query：

- `8/8` 正常完成
- 平均耗时约 33.5 秒/条
- 报告：`eval/flow-report-20260820-021422.md`

该结果属于历史数据/代码状态。新的本机 4GB profile 已完成 CPU Query/GPU
Reranker 真实基准，契约和结果见本文开头及 `output/eval/local_4gb_profile.json`。
此前批次的 18 项确定性违规（各 Flow 原始计数 `3/0/3/1/4/2/2/3`）经按
`fact_violations` 原始条目重新对账，均属于评测假阳性，不是已证实的真实幻觉。根因是
旧审计深度截断、变体别名/选项映射不足、范围端点与跨商品绑定不足，以及预算/非商品金额
和重叠正则命中未过滤去重。新的 Snapshot/exposed-facts 链路已完成重测：8/8 Flow 正常，
确定性 P0 为 0；外部 HTTP 500 是历史排障过程，当前稳定 ID 权威 Judge 结果统一见
`output/eval/flow-rubric-luna-final-20260822.md` 和对应 JSON。

串联脚本：`scripts/eval_flow_with_rubric.py`

### 8.4 Per-case LLM Rubric（历史中间结果）

脚本：`scripts/eval_flow_rubric.py`

流程：

1. 读取 `data/conversations/flow-*.jsonl`
2. 用 LLM 为每条 query 单独生成 P0/P1/P2 rubric
3. 用 LLM Judge 结合商品库事实表、系统规则、rubric 打分
4. PASS 条件：P0 全过且总分 >= 0.7

结果：`4/9 PASS`，平均分 `0.628`

报告：`eval/flow-rubric-20260820-003923.md`

真实 OpenSearch + CategoryInsight RAG 串联批次：

- `3/8 PASS`，平均分 `0.667`
- 报告：`eval/flow-rubric-20260820-023633.md`

本轮已加入 ProductFactSnapshot、evidence refs 与确定性 P0 fact guard。旧批次的 18 项
原始 `fact_violations` 均已归类为评测假阳性：CategoryInsight 深层审计截断、变体别名
匹配、范围/跨商品绑定、预算与重叠价格解析四类问题造成了误报；不是对最终回答真实错误
的确认。新批次确定性 P0=0；HTTP 500 属于历史失效评测过程，当前有效 Judge 评分和
样本口径以稳定 ID 权威报告为准。

以下通过/失败列表属于旧 rubric 口径，仅用于复盘，不能作为当前质量结论。通过：羽毛球包（1.0）、乳胶枕（0.775）、颈椎按摩器（1.0）。

失败：汽车氛围灯（0.2）、儿童学习椅（0.7）、手机直播补光灯（0.487）、户外电源（0.7）、平板电脑支架（0.475）。

主要失败模式：

- 省略或错误处理运费/关税规则
- 对“价格不可用”的商品给出价格范围
- 添加商品库中不存在的店铺、库存、尺码
- 无匹配时仍声称“存在但超预算”
- 未按 rubric 要求列出全部符合条件商品

### 8.4.1 Judge exposed-evidence 修复（2026-08-22）

旧的 LLM Judge `0/7` 结果因 facts 渲染遗漏 Snapshot 变体和 CategoryInsight 暴露字段，
并被“事实表未列出即不存在”的提示语放大，不能作为 Agent 真实问题结论。本轮只修复
评测链路，不改 Agent 生产检索流程：

- `scripts/eval_flow_rubric.py` 构造有界 `judge-evidence-v1`，rubric generator 与
  `call_judge` 接收同一对象；实际 exposed 的 `variants`、变体价格/库存、店铺、商品库存、
  `evidence_id/content_hash/schema_version` 均可被引用。
- CategoryInsight 仅保留实际暴露的 `price_tiers`、`attributes`、`bestsellers` 和
  category/reference scope；未知不等于不存在，CategoryInsight 不证明具体 SKU。
- rubric criteria 记录稳定 id、source_scope、evidence_fields、applies_if；Judge 输出
  `pass|fail|not_evaluable`，本地校验阻止引用不存在的 evidence 字段，`not_evaluable`
  不计作 P0 fail；历史布尔 `pass` 结果仍可解析。

验收：专用临时目录完整 pytest `235 passed, 9 warnings`；相关证据链测试在专用目录
`17 passed`；Ruff 和 `git diff --check` 通过。已有 8 Flow 确定性门禁为 P0=0，未重跑
Flow。重新发送 Judge 前，运行环境因第三方数据出站安全策略拦截了本轮首个外部请求，
未获得 HTTP 响应，故本轮没有有效 Judge 报告、评分或平均分；这些项目保持 N/A。
FastAPI/模型 worker 无残留，Redis/OpenSearch 可保留。

### 8.5 召回与知识卡评测

- `scripts/eval/run_recall_eval.py --prepare`
- `scripts/eval/run_retrieval_comparison.py`
- `scripts/eval/run_category_recall.py`

当前商品召回基线、向量召回、精排、CategoryInsight 都有冻结数据与评测入口。

本轮 CategoryInsight test split（50 条、@10）：

| 方案 | Recall | MRR | NDCG | AllTypes |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 1.0000 | 1.0000 | 0.8723 | 1.0000 |
| KNN / BGE-M3 | 1.0000 | 1.0000 | 0.8633 | 1.0000 |
| Hybrid | 1.0000 | 1.0000 | 0.8651 | 1.0000 |
| Hybrid + BGE Reranker（0.92 跳过） | 1.0000 | 1.0000 | 0.8744 | 1.0000 |

四路均完成 50/50 查询；Reranker 实际执行 12 条、旁路 38 条。相对重建前报告，
Recall/MRR 未回退，但 KNN/Hybrid NDCG 有小幅下降，故默认仍关闭 Category Reranker。

## 9. 部署形态

文件：`docker/docker-compose.yaml`

服务：

- app：FastAPI + Agent
- worker：Redis Stream 消费者
- redis：缓存/队列/背板
- opensearch：`globex/opensearch-ik:2.19.1`
- frontend：Node 22 + Vite

模型以只读 bind mount 挂载：

```text
D:/models/bge-m3
D:/models/bge-reranker-v2-m3
```

当前状态：

- Docker daemon 可用
- `docker-app`、`docker-worker`、`globex/opensearch-ik:2.19.1` 镜像已构建
- 完整 compose 未作为日常评测环境使用；评测直接使用已有 `globex-category-opensearch` 容器
- 已新增 `.dockerignore`，避免构建上下文把 `.venv`、前端、output 等大目录打进镜像

如需完整 compose 部署，仍可用：

```powershell
docker compose -f docker\docker-compose.yaml up -d --wait
```

## 10. 关键设计取舍与失败模式

### 10.1 为什么用 LangGraph + 工具，而不是纯规则管道

课程要求 Agent 能理解自然语言并动态编排工具。LangGraph 提供消息循环、checkpointer、流式输出，便于多轮与并行子 Agent。

### 10.2 为什么领域层用端口抽象

换 SQLite、Redis、OpenSearch 时，application/domain 不感知实现变化。

### 10.3 为什么硬约束放代码而不是模型

预算、可售性、运费、关税是事实规则。模型只负责意图理解和表达，不能自行计算金额。

### 10.4 失败模式

- 模型可能编造商品/价格 → 需要工具返回 + 评测事实表约束
- 工具冷启动超时 → 首次商品搜索可能触发 15s 超时后重试
- 语义缓存若开启会覆盖 Agent 行为 → 评测必须关闭
- 文件持久化无并发控制 → 多 worker 时需要 SQL/Redis
- 自动审批/联网限制只影响外部 LLM 运行，不影响本地单测

## 12. 2026-08-22 评测权威基线

稳定 evidence ID 契约的最新 8 Flow 汇总保存在
`output/eval/flow-rubric-luna-final-20260822.md` 和对应 JSON。确定性 Fact Guard 为
8/8、P0=0；LLM Judge 8/8 有效，平均分 0.99125，适用 P0 criterion 20/20 通过，P0
fail Flow 数为 0。Flow 1、2、7 没有适用 P0 criterion，因此不把它们写成 P0 pass。

旧 0/7 及旧 18 项违规结论来自不完整审计证据、字段/正则解析和提示词误导，已判定为
无效评测假阳性，不代表当前 Agent 真实幻觉。4GB 本机最终配置与基准只引用
`output/eval/local_4gb_profile.json`：CPU FP32 Query Embedding、GPU FP16 Reranker
batch16，Top-100 overlap 1.0，组合链路 median 1820.656 ms。

## 11. 建议输入点

欢迎从以下方向提出建议：

1. **评测事实完整性**：flow query 的 Judge 输入是否应加入 `tool.result`，如何避免“无事实可核验”误判
2. **Agent 防幻觉**：如何让模型在价格不可用、无匹配、filtered_out 场景下更稳定
3. **检索排序**：embedding + rerank 后的候选排序是否还有更好的结构化加权
4. **会话与多 worker**：文件持久化在并发下的风险，以及 SQLite/Redis 迁移策略
5. **韧性参数**：工具超时、熔断阈值、缓存阈值是否合理
6. **部署**：完整 compose 与已有 OpenSearch 容器的切换策略、模型挂载、OpenSearch IK 初始化
7. **前端**：WebSocket 重连、事件时间线、商品卡展示是否需要改进
