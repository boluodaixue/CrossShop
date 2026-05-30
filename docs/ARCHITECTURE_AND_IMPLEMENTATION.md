# Globex 电商搜索 Agent：整体设计与实现说明

> 文档日期：2026-08-20  
> 分支：`codex/migrate-langgraph-ddd`  
> Python：3.10.20  
> 用途：让新读者通过本文了解项目当前的全部设计与实现，并能提出改进建议。

## 1. 项目定位

GlobexAgentLearning 是一个用于学习、运行和逐步组装电商搜索 Agent 的项目。它不是大型电商系统，而是把课程中的关键能力做成一个可本地运行、可评测、可继续演进的完整工程。

当前核心能力：

- 用 LangGraph 的 ReAct Agent 完成“理解需求 → 调工具 → 最终回复”的闭环
- 主 Agent 默认单干，复杂任务通过 `task_dispatch` 并行调用 `search_agent` / `trade_agent`
- 商品召回使用 BGE-M3 + Faiss + BGE Reranker
- 品类洞察使用 OpenSearch Hybrid + `CategoryCard`
- 支持订单创建、查询、取消
- 支持长期买家偏好
- 支持多轮会话持久化、上下文压缩、事件流、语义缓存、工具熔断、模型回退
- 提供 FastAPI + WebSocket + React 前端

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
- 每个 `shopping_session_id` 有独立的 LangGraph `InMemorySaver`
- `SessionRegistry` 缓存 Agent，并按会话序列化 checkpoint 到 `SessionStore`
- 服务重启后，从 `SessionStore` 恢复 checkpoint

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
is_available / provenance / source_updated_at / ingested_at
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

下单即进入 `CONFIRMED`，取消只能从 `CONFIRMED` 到 `CANCELLED` 并必须带 reason。

订单行使用 `item_id + variant_id`，并对下单时价格做快照。金额只通过 `Order.snapshot()` 输出，不允许 Agent 自行计算。

### 4.4 买家偏好

文件：`src/globex_agent/domain/buyer/preference.py`

`BuyerPreference` 支持 `like` / `dislike`，偏好按 `buyer_id` 持久化，同 buyer 同 statement 幂等去重。

### 4.5 会话与队列端口

- `SessionStore`：保存/读取 LangGraph checkpoint
- `ConversationStore`：保存业务可读对话流水和过程事件
- `TaskQueue`：任务队列抽象，当前实现 Redis Stream

领域层只定义端口，基础设施提供文件、SQLite 或 Redis 实现。

## 5. 应用层

### 5.1 `CatalogSearchUseCase`

文件：`src/globex_agent/application/usecases/catalog_search.py`

执行流程：

```text
ProductSearchSpec
  → vector recall（BGE-M3 + Faiss，Top-8）
  → rerank（BGE Reranker）
  → 硬约束过滤（availability / ship_to / price cap）
  → ProductCard 列表
```

召回策略：

- `embedding_rerank`
- `embedding_only`
- `keyword_bm25`

硬约束过滤使用结构化逻辑，不交给模型。`filtered_out` 记录被硬约束挡掉的候选及原因。

### 5.2 订单用例

文件：`src/globex_agent/application/usecases/order_usecases.py`

- `PlaceOrderUseCase`：校验商品存在、可售、变体存在，创建订单
- `QueryOrderUseCase`：按订单号查询
- `CancelOrderUseCase`：校验订单状态后取消

订单工具层再包一层，用于发布 `tool.invoke` / `tool.result` 事件并把异常转为 `[error]`。

### 5.3 工具

工具目录：`src/globex_agent/application/tools/`

| 工具 | 作用 |
|---|---|
| `product_search_tool` | 标准化 query → 商品卡 |
| `category_insight_tool` | 品类知识卡 → 选购口径 |
| `web_search_tool` | Tavily 联网搜索，未配置 key 不注册 |
| `create_order_tool` | 创建订单 |
| `query_order_tool` | 查询订单 |
| `cancel_order_tool` | 取消订单 |
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
- `base.py`：`LangGraphAgent` 包装 `astream` / checkpoint / 上下文压缩
- `checkpoint.py`：把 `InMemorySaver` checkpoint 序列化为 JSON，支持重启恢复
- `orchestrator.py`：`MainAgentOrchestrator`，处理缓存、偏好注入、事件发布、会话持久化

Agent 图使用 `langgraph.prebuilt.create_react_agent`，checkpointer 使用 `InMemorySaver`。

### 5.5 提示词

文件：`src/globex_agent/application/prompts/globex.yml`

提示词约定包括：

- 默认单干，复杂任务才派发
- 价格/库存/运费必须来自工具返回
- 下单/取消前必须先出确认卡
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

### 8.1 自动化测试

```text
pytest：162 passed
ruff：All checks passed
```

覆盖：领域模型、订单状态机、召回、知识卡、事件、熔断、模型回退、会话恢复、并发隔离、表现层。

### 8.2 基础 13 条真实模型回归

文件：

- `scripts/eval_regression.py`
- `eval/cases.yaml`

结果：`13/13 PASS`，平均分 `1.000`

报告：`eval/report-20260819-223705.md`

### 8.3 Flow Query 全流程运行

脚本：`scripts/eval_flow_queries.py`

把已有 RAG/召回 query 直接喂给完整 Agent 服务。当前 9 条：

- 3 条中文 CategoryInsight query
- 3 条中文商品召回 query
- 3 条英文 ESCI query

结果：`9/9` 正常完成，平均耗时约 20.6 秒/条。

报告：`eval/flow-report-20260819-232003.md`

另外已跑真实 OpenSearch + CategoryInsight RAG 的 8 条 query：

- `8/8` 正常完成
- 平均耗时约 33.5 秒/条
- 报告：`eval/flow-report-20260820-021422.md`

串联脚本：`scripts/eval_flow_with_rubric.py`

### 8.4 Per-case LLM Rubric

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

通过：羽毛球包（1.0）、乳胶枕（0.775）、颈椎按摩器（1.0）。

失败：汽车氛围灯（0.2）、儿童学习椅（0.7）、手机直播补光灯（0.487）、户外电源（0.7）、平板电脑支架（0.475）。

主要失败模式：

- 省略或错误处理运费/关税规则
- 对“价格不可用”的商品给出价格范围
- 添加商品库中不存在的店铺、库存、尺码
- 无匹配时仍声称“存在但超预算”
- 未按 rubric 要求列出全部符合条件商品

### 8.5 召回与知识卡评测

- `scripts/eval/run_recall_eval.py --prepare`
- `scripts/eval/run_retrieval_comparison.py`
- `scripts/eval/run_category_recall.py`

当前商品召回基线、向量召回、精排、CategoryInsight 都有冻结数据与评测入口。

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

## 11. 建议输入点

欢迎从以下方向提出建议：

1. **评测事实完整性**：flow query 的 Judge 输入是否应加入 `tool.result`，如何避免“无事实可核验”误判
2. **Agent 防幻觉**：如何让模型在价格不可用、无匹配、filtered_out 场景下更稳定
3. **检索排序**：embedding + rerank 后的候选排序是否还有更好的结构化加权
4. **会话与多 worker**：文件持久化在并发下的风险，以及 SQLite/Redis 迁移策略
5. **韧性参数**：工具超时、熔断阈值、缓存阈值是否合理
6. **部署**：完整 compose 与已有 OpenSearch 容器的切换策略、模型挂载、OpenSearch IK 初始化
7. **前端**：WebSocket 重连、事件时间线、商品卡展示是否需要改进
