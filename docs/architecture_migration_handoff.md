# Globex LangGraph + DDD 架构迁移执行方案

> 用途：交给 DeepSeek V4 flash 执行。本文件是当前唯一执行依据，若与历史方案冲突，以本文件为准。
> 状态：已由项目主人确认，等待执行。

> P0-003 更新（2026-08-21）：订单/取消写操作已完成 Prepare → 可信 Confirm 门禁。
> Agent 只注册准备工具；确认 token 只经 FastAPI 结构化入口提交，应用层保存 token hash、
> canonical payload SHA-256、session/user/action、TTL 和一次性状态。多进程持久化仍列为 P1-011。

## 0. 目标

把 `GlobexAgentLearning` 从旧的 9 工具课程复现，迁移成面向简历/面试的
LangGraph + DDD 洋葱架构：

- 主 Agent 默认单干。
- 需要时通过批量 `task_dispatch` 并行调用两个异构专家子 Agent：
  `search_agent` 和 `trade_agent`。
- 商品召回继续使用 BGE-M3 + Faiss。
- 品类洞察继续使用 OpenSearch Hybrid + `CategoryCard`。
- 韧性、队列、事件、持久化、FastAPI 和前端参考 `globex-agent-main`，
  但 AgentScope / Qdrant / Product-Sku 领域模型不进入本项目。

## 1. 不变原则

1. 课程资料只读，不复制课程 Markdown 或图片进入本仓库。
2. 商品数据以现有 `StandardItem` 为规范，不引入参考仓库的 `Product/Sku` 作为领域模型。
3. RAG 知识卡以现有 `CategoryCard` 为规范，不把参考仓库 `knowledge/*.md` 原样当 RAG 文档。
4. Python 保持 3.10.20，不升级 Python 版本。
5. 所有模型请求走 OpenAI-compatible Chat Completions 格式。
6. 默认持久化使用 JSON 文件仓储，`DATABASE_URL=file`。
7. 旧 9 工具、旧 `agent/`、旧 `pipeline/` 删除；Git 历史保留旧版本。
8. 每个阶段必须实际运行并验证，不能只报告“代码已生成”。
9. 不做一次性整体 rename；按阶段提交，每阶段 `pytest` 全绿后再进入下一阶段。

## 2. 当前基线

执行前先记录基线，完成后可对比：

```powershell
git status --short
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src scripts examples tests
```

当前已确认：工作区干净，Python 3.10.20，现有召回和 CategoryInsight 测试已通过。

## 3. 目标目录

最终以 `src/globex_agent/` 为根：

```text
src/globex_agent/
├── domain/
│   ├── catalog/
│   │   ├── models.py
│   │   ├── retrieval.py
│   │   └── money.py
│   ├── shipping/
│   │   └── tariff_schedule.py
│   ├── order/
│   │   ├── address.py
│   │   ├── order_line.py
│   │   ├── order.py
│   │   └── ports/
│   ├── buyer/
│   │   └── preference.py
│   ├── session/
│   │   └── ports/
│   └── queue/
│       └── ports/
├── application/
│   ├── usecases/
│   │   ├── catalog_search.py
│   │   └── order_usecases.py
│   ├── tools/
│   │   ├── product_search_tool.py
│   │   ├── category_insight_tool.py
│   │   ├── web_search_tool.py
│   │   ├── order_tools.py
│   │   ├── remember_preference_tool.py
│   │   └── task_dispatch_tool.py
│   ├── agents/
│   │   ├── main_agent.py
│   │   ├── search_agent.py
│   │   ├── trade_agent.py
│   │   └── orchestrator.py
│   └── prompts/
│       └── globex.yml
├── infrastructure/
│   ├── recall/              # 由现有 recall/ 移动而来
│   ├── embedding/
│   ├── vector/
│   ├── rerank/
│   ├── cache/
│   ├── queue/
│   ├── persistence/
│   ├── eventbus.py
│   ├── settings.py
│   ├── throttle.py
│   ├── transient.py
│   ├── context.py
│   └── resilience.py
├── category_insight/       # 现有 CategoryInsight 保留
├── presentation/
│   ├── server.py
│   ├── connection.py
│   └── dto.py
├── composition.py
└── worker.py
```

## 4. 领域模型选择

### 4.1 商品

继续使用 `StandardItem`，现有字段保留：

```text
item_id / same_group_id / platform / locale / language
title / description / brand / category_path
price_cny / original_price_cny / currency_raw / price_source
rating / review_count / attributes / variants
is_available / provenance / source_updated_at / ingested_at
```

订单行引用 `item_id + variant_id`，不使用参考仓库的 `product_id + sku_id`。

### 4.2 商品检索输出

继续使用现有契约：

```text
Candidate -> PricePoint -> LandedCost -> PickedItem
```

新 `product_search_tool` 对外输出可以保留这些字段，但必须增加：

- `landed_price`：到手价三分项。
- `recall_strategy`：`embedding_rerank / embedding_only / keyword_bm25` 等。
- `filtered_out`：被硬约束挡掉的候选，带 `reason`。

### 4.3 知识卡

继续使用 `CategoryCard`：

```text
card_id / category / card_type / summary / raw_evidence / last_updated / confidence
```

`card_type` 只允许 `bestseller / attribute / price_range`。

## 5. 关键适配点

### 5.1 订单与库存

现有 `StandardItem.variants[].is_available` 可以决定是否可售，但没有库存数量。
第一阶段先采用确定性库存视图：

- `is_available=true` 默认视为库存充足。
- `is_available=false` 不可下单。
- 不伪造平台真实库存。
- 后续如需演示扣库存，再增加明确的派生库存字段和 provenance。

### 5.2 原产国与可送国家

现有 `StandardItem` 没有 `origin_country` 和 `ships_to`。

第一阶段使用确定性的本地规则适配：

- `origin_country` 从 `platform/locale` 或 provenance 推导，无法推导时为空。
- `ships_to` 从平台和 locale 推导，或由固定规则表提供。
- 所有推导字段必须在文档中标注为 `derived`，不得伪装成平台真实字段。

### 5.3 同步召回内核适配

现有 BGE-M3 / Faiss / Reranker 是同步接口，新 usecase 需要异步接口。
在 `infrastructure/embedding`、`vector`、`rerank` 中写异步适配器，内部调用现有同步代码。

## 6. 执行阶段

每阶段都要跑测试。Git 提交信息建议使用：

```text
feat(migration): phase 0 baseline
feat(migration): phase 1 domain
feat(migration): phase 2 infrastructure
feat(migration): phase 3 application
feat(migration): phase 4 presentation
feat(migration): phase 5 eval and docs
```

### Phase 0：基线

1. 创建分支 `codex/migrate-langgraph-ddd`。
2. 记录当前测试结果。
3. 只更新依赖和配置模板，不删除业务代码。

依赖新增：

```text
fastapi
uvicorn[standard]
httpx
aiosqlite
sqlalchemy[asyncio]
redis
```

`.env.example` 统一为：

```text
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=
LLM_FALLBACK_MODEL=
LLM_JUDGE=
DATABASE_URL=file
REDIS_URL=
CHECKPOINT_REDIS_URL=
GLOBEX_ENV=local
CHECKPOINT_TTL_MINUTES=10080
CHECKPOINT_REFRESH_ON_READ=1
CHECKPOINT_REQUIRE_ENCRYPTION=0
LANGGRAPH_AES_KEY=
THREAD_ID_HMAC_KEY=
QUEUE_ENABLED=0
SEMANTIC_CACHE_ENABLED=1
```

当前实现补充：FastAPI 主线使用共享官方 `AsyncRedisSaver`，Redis checkpoint 与 Redis Stream
worker 分离；旧 JSON/SQL `SessionStore` bridge 不自动迁移，非本地加密配置在当前
`langgraph-checkpoint-redis==0.5.2` serializer 不可注入时会 fail-fast。

验收：

- 当前测试仍通过。
- 新依赖可安装。

### Phase 1：领域层

从参考仓库搬入纯 Python 领域规则，但从 `Product/Sku` 改为 `StandardItem`。

新建：

- `domain/catalog/money.py`
- `domain/shipping/tariff_schedule.py`
- `domain/order/address.py`
- `domain/order/order_line.py`
- `domain/order/order.py`
- `domain/order/ports/order_repository.py`
- `domain/buyer/preference.py`
- `domain/session/ports/session_store.py`
- `domain/session/ports/conversation_store.py`
- `domain/queue/ports/task_queue.py`

保留并整理：

- `domain/models.py` 迁成 `domain/catalog/models.py`
- `domain/retrieval.py` 迁成 `domain/catalog/retrieval.py`

适配内容：

- `OrderLine` 使用 `item_id / variant_id`，不再使用 `product_id / sku_id`。
- `Order` 状态机保留 `DRAFT -> CONFIRMED -> CANCELLED`。
- `Money` 仍使用最小货币单位。
- `TariffSchedule` 仍负责运费、关税和到手价。

验收：

- 领域相关单测全绿。
- 现有召回和 CategoryInsight 单测不因领域模型移动而失败。

### Phase 2：基础设施层

把现有 `recall/` 物理移动到 `infrastructure/recall/`。

同时搬入参考仓库的纯 Python 基础设施：

- `eventbus.py`
- `settings.py`
- `throttle.py`
- `transient.py`
- `context.py`
- `resilience.py`
- `cache/`
- `queue/`
- `persistence/`

新建异步适配器：

- `infrastructure/embedding/bge_m3_embedding.py`
- `infrastructure/vector/faiss_product_index.py`
- `infrastructure/rerank/bge_reranker.py`

这些适配器实现领域端口，内部调用现有 BGE-M3、Faiss、Reranker。

不搬：

- `qdrant_product_index.py`
- `openai_embedding_client.py`
- `http_reranker.py`
- `rag/category_knowledge.py`

验收：

- 召回评测脚本路径更新后仍能运行。
- 熔断、限流、缓存、队列、JSON 仓储相关单测通过。
- OpenSearch CategoryInsight 路径保持可用。

### Phase 3：应用层

按 `StandardItem` 重写参考仓库的应用用例，而不是照搬 `Product/Sku`。

新建：

- `application/usecases/catalog_search.py`
- `application/usecases/order_usecases.py`
- `application/tools/product_search_tool.py`
- `application/tools/category_insight_tool.py`
- `application/tools/web_search_tool.py`
- `application/tools/order_tools.py`
- `application/tools/remember_preference_tool.py`
- `application/tools/task_dispatch_tool.py`
- `application/agents/main_agent.py`
- `application/agents/search_agent.py`
- `application/agents/trade_agent.py`
- `application/agents/orchestrator.py`
- `application/prompts/globex.yml`

工具语义：

- `product_search_tool`：输入标准化 query 和槽位，返回商品卡、`landed_price`、`recall_strategy`、`filtered_out`。
- `category_insight_tool`：调用现有 `CategoryInsightService`。
- `order_tools`：只向 Agent 暴露准备订单/准备取消和查询；创建/取消必须经
  `OrderConfirmationService` 的可信 Confirm API，订单行使用 `item_id + variant_id`。
- `remember_preference_tool`：写入长期偏好。
- `web_search_tool`：无 `TAVILY_API_KEY` 时不注册。
- `task_dispatch_tool`：批量接口。

### `task_dispatch` 接口

使用批量接口：

```python
task_dispatch(
    dispatches: list[dict],
) -> str
```

每个元素：

```json
{
  "subagent_type": "search_agent | trade_agent",
  "demands": "自包含子任务指令"
}
```

实现要求：

1. 对每个 dispatch 分配独立 `thread_id`。
2. 使用 `asyncio.gather` 并行调用对应的 LangGraph 子图。
3. 发布 `agent.dispatch`，开始时间和结束时间必须可观察。
4. 发布 `tool.result`，包含 `started_at / finished_at / elapsed_ms`。
5. 任意子 Agent 失败时返回 `[error]`，主 Agent 必须能继续并降级。

验收：

- 主 Agent 单干路径跑通。
- 主 Agent 派发 search_agent 路径跑通。
- 主 Agent 派发 trade_agent 路径跑通。
- 并行派发多个 search_agent 时，事件时间区间存在重叠。

### Phase 4：表现层

新建：

- `presentation/dto.py`
- `presentation/connection.py`
- `presentation/server.py`
- `composition.py`
- `worker.py`
- `frontend/`

API：

- `POST /commerce/intents`
- `POST /commerce/intents/async`
- `GET /commerce/tasks/{task_id}`
- `WS /commerce/events`
- `GET /commerce/orders/{order_id}`
- `POST /commerce/orders/{order_id}/cancel`
- `GET /health`

事件保持 12 类：

```text
agent.dispatch / tool.invoke / tool.result / token.delta / plan.update
context.compressed / model.fallback / cache.hit / task.queued / task.started
final.result / error
```

验收：

- FastAPI 服务可启动。
- WebSocket 能收到事件。
- `/health` 返回正确状态。
- 前端能渲染对话流、商品卡和事件时间线。

### Phase 5：评测、部署与文档

1. 把参考仓库的 `eval/cases.yaml` 复制为应用级端到端 case。
2. 把 case 中的商品事实改成 `StandardItem` 种子商品。
3. 实现 LLM Judge，使用 `LLM_JUDGE`。
4. 保留现有召回门禁：
   - 商品召回 Recall@10 / MRR / NDCG。
   - CategoryInsight Recall@K / MRR / NDCG。
5. 更新 compose，把 Qdrant 换成现有 OpenSearch。
6. 更新 README、工程计划、同步提示文档。

验收：

- 全部测试通过。
- `smoke_e2e.py` 通过。
- 13 条端到端 case 有报告。
- 现有召回门禁仍能复跑。

## 7. 删除范围

以下内容在新版本中删除：

- `src/globex_agent/tools/`
- `src/globex_agent/pipeline/`
- `src/globex_agent/agent/`
- `src/globex_agent/prompt/prompts.yml`
- `examples/02_deterministic_pipeline.py`
- `examples/03_single_agent.py`
- 只服务旧 9 工具和旧 AgentLoop 的测试。

以下内容必须保留：

- 现有召回实现。
- 现有 CategoryInsight 实现。
- 现有召回评测和数据。
- `examples/04_semantic_retrieval.py` 和 `examples/05_category_insight.py`，除非它们只依赖被删除的旧 Agent 工具。

## 8. 最终验证命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src scripts examples tests
.\.venv\Scripts\python.exe scripts\eval\run_recall_eval.py --prepare
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --data-dir data\category_insight\taobao_zh --index-name globex_category_kb_taobao_zh_v1 --cards-filename category_cards_taobao_zh.jsonl --cases-filename category_recall_cases_taobao_zh.jsonl --manifest-filename category_recall_manifest_taobao_zh.json --embedding-model D:\models\bge-m3 --reranker-model D:\models\bge-reranker-v2-m3 --reranker-python C:\Anaconda\envs\blog_04\python.exe --local-files-only
```

应用服务验证：

```powershell
.\.venv\Scripts\python.exe -m uvicorn globex_agent.presentation.server:app --port 8000
.\.venv\Scripts\python.exe scripts\smoke_e2e.py
.\.venv\Scripts\python.exe scripts\eval_regression.py
```

## 9. 禁止事项

- 不引入 AgentScope。
- 不引入 Qdrant。
- 不把参考仓库的 `Product/Sku` 当作领域模型。
- 不把参考仓库的 `knowledge/*.md` 原样作为 RAG 文档。
- 不删除现有召回数据、评测集和 CategoryInsight 数据。
- 不升级 Python 版本。
- 不把真实密钥写入仓库。
- 不虚构运行指标。
