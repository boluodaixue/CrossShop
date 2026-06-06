# Globex 对比分析与合并方案

> 定位：简历 / 面试项目。
> 本文档记录「课程文档」与「globex-agent 参考仓库」的对比结论，以及据此确定的最终合并方案、逐文件来源与迁移顺序。
> 生成时间：2026-08-15（依会话进行中时间）。

---

## 0. 资料来源

| 角色 | 来源 | 说明 |
| --- | --- | --- |
| 课程文档（规范设计） | `D:\刘诗恩\Obsidian\电商搜索Agent资料` | 作者「会敲代码的泡」，小红书《电商搜索 Agent》课程，LangGraph/LangChain 主线 |
| 对比仓库 A（最初给的） | `github.com/PastWestCoast/globex-agent` | 私有仓库，对外 404，无法访问 |
| 对比仓库 B（最终用） | `github.com/boluodaixue/globex-agent`，本地下载 `D:\PycharmProjects\globex-agent-main` | AgentScope 2.0 + DDD 洋葱架构的「换框架重实现」 |
| 本仓库（现有实现） | `D:\PycharmProjects\GlobexAgentLearning` | LangGraph 课程复现，已有 9 工具 + Faiss/OpenSearch Hybrid + BGE-M3 + 召回评测 |

> 仓库 B 的 README 自述「对齐参考实现与教程口径」，`docs/设计演进记录.md` 反复引用「教程正文」——它是**同一套 Globex 概念在 AgentScope 2.0 + DDD 下的换框架重实现**，共享课程概念骨架，但框架/分层/召回/多 Agent/工具/基础设施做出不同取舍。

---

## 1. 对比结论总览

一句话：**仓库 B 是「同概念、换框架、做减法」的重实现。**

| 层 | 结论 |
| --- | --- |
| **同**（对齐口径） | 三件事判断、主 Agent 单干优先、二阶段召回+降级、长期记忆跨会话、事件流+WebSocket、工具熔断/超时、Token 预算/模型回退、Rubric P0/P1/P2、下单确认卡 |
| **换**（框架/基础设施） | LangGraph→AgentScope 2.0；扁平 `app/`→DDD 洋葱；三塔→embedding+Qdrant；AGUI 七事件→自有 12 事件；vLLM/OpenSearch/LangFuse/K8s→网关/Qdrant/SQLite/OTLP |
| **减**（丢弃的课程内核） | ①同质 fork（改固定 2 异构专家）②三塔个性化+训练全链路 ③9+1 工具流水线（比价运费内联、收尾工具交模型）④跨平台（4 平台变单库） |

---

## 2. 相同点（仓库确实对齐了教程口径）

1. **三件事判断几乎逐字保留**：课程「能并行 / 上下文隔离 / 链深≥3」→ 仓库 `task_dispatch_tool.py` docstring 与 main_agent prompt 写「可并行 / 上下文隔离 / 调用链深」，且都强调「多个独立子任务同一轮一次性发起」。
2. **主 Agent 单干优先**：默认自己直接调业务工具，仅确有必要才派发。
3. **二阶段召回 + 降级链**：`embedding_rerank → embedding_only → keyword_2gram`，`recall_strategy` 如实标注。
4. **长期记忆跨会话**：写路径工具化、读路径注入 system/hint（`<buyer-preferences>`）。
5. **事件流 + WebSocket 实时可见**：都按会话 id 路由事件。
6. **工具熔断 + 超时**：分级超时 + closed→open→half_open。
7. **Token 预算 + 模型降级**：`ReplyBudgetControlMiddleware` + `GatewayThrottle` 回退 `LLM_FALLBACK_MODEL`。
8. **Rubric P0/P1/P2 评测**：P0 数字/安全底线、P1 行为命中、P2 表达。
9. **下单前确认卡**：写操作前必须先出确认卡、用户确认后再执行。
10. **「修 Harness 不修 Prompt」精神**：仓库 13 条踩坑档案（如「硬约束过滤导致误导性回答」「语义缓存把评测弄成评缓存」）正是课程 17-1 方法论的实战注释。

---

## 3. 不同点（逐模块）

### 3.1 多 Agent：同质 fork → 异构专家（最大分歧）

| 维度 | 课程（同质 fork） | 仓库 B（异构专家） |
| --- | --- | --- |
| 子 Agent 工具集 | 与主 Agent 完全相同（`FULL_TOOL_SET` 共享） | 各持子集（search 3 工具 / trade 3 工具） |
| 子 Agent 数量 | 动态（按需 1~10 个） | 固定 2 类（search_agent / trade_agent） |
| 递归能力 | 支持（子可再 fork，`MAX_FORK_DEPTH=2`） | 不支持（子 Agent 是叶子） |
| 派发实现 | `dispatch_tool(demands)` + `parallel_dispatch_tool(demands_list)` | `task_dispatch(subagent_type, demands)`，`is_concurrency_safe=True` 并发批执行 |

> 补充：仓库的「异构」是**中间态**——主 Agent 持全部业务工具（superset，可单干），子 Agent 是子集专家。但子 Agent 数量固定 2、工具集各异、不可递归，仍落在课程 03-0 §3.2 反对的「传统异构方案」一列。

### 3.2 召回：三塔 → 单塔 OpenAI embedding

| 环节 | 课程 | 仓库 B |
| --- | --- | --- |
| 编码模型 | 三塔（User/Query/Item），`请求向量=f(User,Query)`，`ANN top_k=100`，双通道 `α·语义+β·个性化` | OpenAI `text-embedding-v4`(1024) 单塔，只 embed query 一个向量，无 User 塔 |
| 向量库 | 召回层 Faiss(HNSW+IP)→Milvus；应用层 OpenSearch(Hybrid) | Qdrant(COSINE)，本地嵌入/服务端双形态 |
| Embedding 训练 | 三阶段 CPT(1e-5)→SFT(5e-6)→DPO(1e-6)，InfoNCE 四改进 | 无训练 |
| Reranker | BGE-Reranker-v2-m3(567M)+LoRA，三阶段 BCE→Margin→NDCG | HTTP reranker（Qwen3-Reranker /rerank），未部署降级按向量分 |
| 质量门禁 | Recall@100≥0.85 / MRR@10≥0.45 / NDCG@10≥0.55 | 无训练门禁，仅 13 case LLM judge |

### 3.3 工具集：9+1 → 6 文件 + 内联到手价

- 课程 `FULL_TOOL_SET`：`planner / chat_fallback / web_search / category_insight / item_search / item_picker / price_compare / shipping_calc / shopping_summary` + `dispatch_tool`（单向类型化数据流）。
- 仓库 6 工具文件（8 工具函数 + 内置 Task 四件套）：`product_search`（检索+内联 `landed_price`）/ `category_insight` / `web_search` / 订单三工具 / `remember_preference` / `task_dispatch` + `TaskCreate/TaskUpdate/TaskList/TaskGet`。
- 仓库把 `price_compare + shipping_calc` 合并「内联到手价」，`item_picker/shopping_summary/planner/chat_fallback` 交给模型/内置 Task 计划；多出订单闭环与记忆工具。

### 3.4 基础设施与部署

| 维度 | 课程 | 仓库 B |
| --- | --- | --- |
| 推理 | vLLM GPU（35B MoE） | OpenAI 兼容网关（qwen3-max→qwen-plus） |
| 向量/知识库 | OpenSearch + Faiss | Qdrant |
| 持久化 | Redis/Postgres Store | SQLite(默认)/JSON 双实现 |
| 削峰 | active_tasks 字典 | Redis Stream + worker 进程 + 幂等键 |
| 可观测 | LangFuse 全链路 Trace | OTLP Trace（TracingMiddleware） |
| 编排 | K8s 灰度 | Docker Compose（app+worker+qdrant+redis+frontend） |

### 3.5 事件协议、上下文压缩、评测训练、业务范围

- **事件协议**：课程 AGUI 七事件（`session_created/assistant_call/tool_start/tool_end/task_result/task_cancelled/error`）→ 仓库 12 事件（`agent.dispatch/tool.invoke/tool.result/token.delta/plan.update/context.compressed/model.fallback/cache.hit/task.queued/task.started/final.result/error`）。语义可映射，仓库多出工程可观测事件。
- **上下文压缩**：课程 Cache Breakpoint（保 Prompt Cache 命中率，L0–L4）→ 仓库 `ContextConfig(trigger_ratio=0.75, reserve_ratio=0.15)`。
- **评测训练**：课程 Rubric→SFT→Agentic RL(GSPO) 数据飞轮 → 仓库 13 case LLM judge，无 SFT/RL。
- **业务范围**：课程不下单 → 仓库多出订单闭环（Order 状态机 + 确认卡）。

---

## 4. 最终选用方案（9 项决策）

| # | 维度 | 定案 | 来源 |
| --- | --- | --- | --- |
| 1 | 框架 | **LangGraph**（`create_react_agent`） | 课程 / 本仓库现有 |
| 2 | 多 Agent | **异构子 Agent**（search_agent / trade_agent），主 Agent 持全量工具「单干优先」 | 仓库 B |
| 3 | 召回与向量/知识库 | **BGE-M3 双塔（query-item）+ Faiss（商品召回）+ OpenSearch Hybrid（品类 KB）** | 本仓库现有 |
| 4 | 工具集 | **6 工具形式**（`product_search` 内联 `landed_price` + 订单三工具 + 记忆 + 派发） | 仓库 B |
| 5 | 分层结构 | **DDD 洋葱四层 + 装配根（composition root）** | 仓库 B |
| 6 | 韧性工程 | 熔断 / 限流 / 语义缓存 / 队列削峰 / 幂等 | 仓库 B |
| 7 | 事件总线 / 前后端 | **12 事件总线 + FastAPI + React**（`EventTimeline` 前端） | 仓库 B |
| 8 | 评测 / 训练 | 端到端 **cases.yaml + LLM judge P0/P1/P2**（仓库）+ 模块级**召回门禁 Recall@K/MRR/NDCG**（现有）；SFT/RL 不落地 | 仓库 B + 现有 |
| 9 | 部署 | 仓库 compose 精简结构，**qdrant 换成 opensearch**（本机 Docker 2.19.1 + IK 已自部署） | 仓库 B + 现有 |

### 4.1 当前证据与本机推理配置补充

- 商品事实不再由 ProductCard 与审计事件分别拼装。`StandardItem → ProductFactSnapshot → ProductCard → LLM` 是唯一方向；ProductCard 只保留紧凑模型字段与 `evidence_id`，Snapshot 保存版本、hash、provenance 和 `exposed_facts`。
- Fact Guard/eval parser 只消费 Snapshot 明确暴露的事实；变体价格必须和同一变体选项绑定。审计仍使用全局 PII 脱敏和载荷上限，但 Snapshot 使用 schema-aware 白名单，避免 variants/highlights 被通用 depth 截断。
- 本机 4GB 仅改变设备分配：Query Embedding 为 CPU FP32 常驻 worker，Reranker 为 GPU FP16 常驻 worker；生产仍支持 Embedding GPU + Reranker GPU 的分离部署。模型、Faiss 四分区、schema-v2、Top-100→Top-10 主链不变。
- 旧 Flow 的 P0 条目因证据截断和解析问题属于历史失效结果；当前稳定 ID 权威基线已完成
  8/8 Flow、确定性 P0=0 和 Judge 8/8 有效。P0=0 后运行 Judge 是评测执行规范，不是在线生产链门禁。

---

## 5. 目标架构

```text
框架：LangGraph（create_react_agent 表达 Agent；子 Agent 用独立子图，不用
      boluodaixue 的 "FunctionTool 包 Agent" hack）
分层：DDD 洋葱
├─ domain/           Money / ExchangeRateTable / TariffSchedule / Order(状态机) /
│                    Product / BuyerPreference / 各类端口          ← 仓库（纯搬）
├─ application/
│   ├─ agents/       主 Agent(CommerceConcierge) + SearchAgent + TradeAgent
│   │                —— 主 Agent 持全量工具"单干优先"，专家子 Agent 持子集"按需派发"
│   ├─ tools/        6 工具形式                                    ← 仓库（LangChain 重写外壳）
│   └─ usecases/     CatalogSearch / PlaceOrder / QueryOrder / CancelOrder
├─ infrastructure/   BGE-M3 双塔 + Faiss(商品) + OpenSearch Hybrid(品类KB) + Reranker  ← 现有
│                    熔断/限流/语义缓存/Redis Stream 队列/幂等/SQLite    ← 仓库
├─ presentation/     FastAPI + WS + React 前端                        ← 仓库
└─ composition.py    装配根（API/worker 共用）                        ← 仓库
```

**叙事主线（面试一句话）**：主 Agent 持全量工具默认单干，只有子任务满足「可并行/需上下文隔离/调用链深」时才派发到 Search/Trade 两个异构专家且同轮真并发；检索用 BGE-M3 双塔 + Faiss/OpenSearch Hybrid 二阶段召回，价格/关税在检索链路内联算好到手价并带 `filtered_out` 自证边界；工程用 DDD 洋葱 + 装配根，配熔断/限流/语义缓存/队列削峰/幂等。

---

## 6. AgentScope 依赖清单 + LangGraph 对应替换

### 6.1 强依赖 AgentScope 的生产文件（16 个，4 组，必须重写）

| 组 | 文件 | 依赖的 AgentScope 对象 |
| --- | --- | --- |
| ① agent 编排 | `application/agents/` 6 个（main_agent / search_agent / trade_agent / orchestrator / context_policy / permissions） | `Agent` / `ReActConfig` / `AgentState` / `Toolkit` / `FunctionTool` / `Task*` / `ContextConfig` / `Permission*` / `reply_stream` 事件 |
| ② 工具外壳 | `application/tools/` 6 个 | `ToolChunk` / `TextBlock` / `ToolResultState` / `KnowledgeBase` |
| ③ infra 薄层 | `infrastructure/llm.py`、`resilience.py`、`tracing.py` | `OpenAIChatModel` / `OpenAICredential` / `ToolMiddlewareBase` / `TracingMiddleware` / `ReplyBudgetControlMiddleware` |
| ④ RAG | `infrastructure/rag/category_knowledge.py` | `KnowledgeBase` / `QdrantStore` / `TextParser` / `ApproxTokenChunker` |

### 6.2 纯 Python / 零 AgentScope 依赖（直接搬）

- **`domain/**` 全部**：Money / Order / TariffSchedule / ExchangeRateTable / Product / Preference / 所有端口（纯 dataclass + ABC）。
- **`application/usecases/`**：`catalog_search.py`、`order_usecases.py`（只依赖 domain 端口，无 agentscope import）。
- **`infrastructure/`**：`eventbus.py`、`settings.py`、`throttle.py`、`transient.py`、`context.py`、`cache/*`、`embedding/openai_embedding_client.py`、`rerank/http_reranker.py`、`vector/qdrant_product_index.py`、`persistence/*`、`queue/redis_stream_queue.py`。
- **`presentation/*`**：`server.py` / `connection.py` / `dto.py`（纯 FastAPI）。
- **`composition.py` / `worker.py`**（模式）、`docker/`、`eval/cases.yaml`、`frontend/`、`knowledge/`、`scripts/`。

> 关键：韧性工程的核心逻辑（`CircuitBreakerRegistry`、`GatewayThrottle`、`SemanticCache`、`RedisStreamTaskQueue`、幂等键）全是纯 Python，**只有"接进框架"的薄壳（`ToolResilienceMiddleware(ToolMiddlewareBase)` 等）才碰 AgentScope**。

### 6.3 LangGraph 对应替换表

| AgentScope 概念 | LangGraph / LangChain 等价物 |
| --- | --- |
| `Agent` + `ReActConfig` | `create_react_agent`（langgraph.prebuilt） |
| `AgentState` | LangGraph `State(TypedDict)` + `MemorySaver` / checkpointer |
| `Toolkit` / `FunctionTool` | `@tool`（langchain_core.tools）+ `ToolNode` |
| `TaskCreate/Update/List/Get`（内置计划） | 自定义 task 管理工具，或 LangGraph `Send`/子图 + task state |
| `ContextConfig`（上下文压缩） | 自定义压缩节点 / `post_model_hook`，摘要写入 state |
| `PermissionBehavior/Rule` | 工具白名单 + 显式 allowlist 包装（LangGraph 无权限系统） |
| `reply_stream`（类型化事件流） | `astream_events(version="v2")` + 自定义事件映射 |
| `ToolChunk` / `ToolResultState` / `TextBlock` | 工具返回 `str`/`dict`，错误用异常或 `[error]` 字符串 |
| `OpenAIChatModel` / `OpenAICredential` | `ChatOpenAI`（langchain_openai）或 `init_chat_model` |
| `ToolMiddlewareBase` | `@tool` 装饰器包装（timeout/熔断）或 `RunnableConfig` + callback |
| `TracingMiddleware` | **`LangfuseCallbackHandler`**（`config={"callbacks":[handler]}`，可挂 Rubric 分数） |
| `ReplyBudgetControlMiddleware` | 自定义 callback 统计 token + 超预算注入 system reminder |
| `KnowledgeBase` / `QdrantStore` / `TextParser` / `ApproxTokenChunker` | 现有 `recall/category_kb.py`（OpenSearch Hybrid） |
| `OpenAIEmbeddingModel` | 现有 BGE-M3（sentence-transformers）+ `recall/embedding.py` |

> `TracingMiddleware` 本身不是仓库作者写的——它是 `agentscope.middleware.TracingMiddleware` 内置件。仓库作者只写了纯 OTel 的 `setup_tracing()`（建全局 TracerProvider + OTLP exporter）。迁移后删掉 OTel + 该中间件，换成 LangFuse 官方 `CallbackHandler`，更简单且能给 trace 挂 Rubric 分数（OTel 做不到）。

---

## 7. 逐文件来源与迁移顺序

> 目标目录（DDD 重构后，仍在 `src/globex_agent/` 下）：`domain/ application/ infrastructure/ presentation/ composition.py worker.py`。
> 标注：**搬** = 从仓库 B 搬入；**保留** = 现有实现保留/改造；**新写** = 用 LangGraph 重写。

### 阶段 1：domain/（纯搬仓库 + 现有合并）

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `domain/catalog/money.py`、`exchange_rate.py`、`product.py`、`sku.py`、`product_search_spec.py`、`ports/*` | 搬 | 分单位 Money 防浮点 |
| `domain/order/order.py`、`order_line.py`、`address.py`、`ports/*` | 搬 | Order 状态机 DRAFT→CONFIRMED→CANCELLED |
| `domain/shipping/tariff_schedule.py` | 搬 | 关税/运费规则内核（`landed_price` 前置依赖） |
| `domain/buyer/preference.py`、`ports/*`；`domain/session/ports/*`；`domain/queue/ports/*` | 搬 | PreferenceStore/SessionStore/TaskQueue 端口 |
| `domain/models.py`、`retrieval.py`（现有） | 保留改造 | 并入上述领域模型与端口 |

**验收标准**：`pytest tests/unit` 领域相关用例全绿；Money/Tariff/Order 状态机可独立单测（仓库 `test_domain.py`、`test_pricing.py` 搬入适配）。

### 阶段 2：infrastructure/（现有召回内核保留 + 仓库韧性搬入）

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `infrastructure/recall/embedding.py`（BGE-M3）、`index.py`（Faiss）、`category_kb.py`（OpenSearch Hybrid）、`keyword.py`、`sqlite_fts.py`、`fusion.py`、`router.py`、`reranker.py` | 保留 | 召回内核不动 |
| `infrastructure/eventbus.py`、`settings.py`、`throttle.py`、`transient.py`、`context.py` | 搬 | 纯 Python |
| `infrastructure/cache/*`（redis_cache / semantic_cache / cached_embedding_client）、`queue/redis_stream_queue.py`、`persistence/*` | 搬 | 语义缓存 + 队列 + SQLite/JSON 仓储 |
| `infrastructure/resilience.py` | 搬 + 新写 | `CircuitBreakerRegistry` 纯部分搬；`ToolResilienceMiddleware` 外壳换成 LangChain 包装 |
| `infrastructure/ports/`（EmbeddingClient / ProductVectorIndex / Reranker） | 保留 + 新写 | 把 BGE-M3/Faiss/rerank 实现成 `catalog_search.py` 的三个端口 |
| `infrastructure/rag/category_knowledge.py` | **不搬** | 由现有 OpenSearch Hybrid 取代 |

**验收标准**：`scripts/eval/run_category_recall.py` / `run_recall_eval.py` 召回门禁通过；熔断/限流/缓存/队列单测（仓库 `test_phase4_*.py` 搬入适配）全绿。

### 阶段 3：application/（agents + tools 用 LangGraph 重写，usecases 搬）

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `application/usecases/catalog_search.py` | 搬 | 五步流程 + `recall_strategy` + `filtered_out`，内部接 BGE-M3/Faiss 端口 |
| `application/usecases/order_usecases.py` | 搬 | PlaceOrder / QueryOrder / CancelOrder |
| `application/tools/product_search_tool.py` | 新写（外壳仓库） | 签名照搬仓库（`normalized_query, category, ship_to, top_k, price_max_major, target_currency`），返回 `landed_price + recall_strategy + filtered_out`，LangChain `@tool` 返回 dict |
| `application/tools/category_insight_tool.py` | 新写（外壳仓库 + 现有内核） | 仓库签名，内部走现有 OpenSearch Hybrid |
| `application/tools/web_search_tool.py`、`order_tools.py`、`remember_preference_tool.py` | 新写 | 仓库语义，LangChain 返回 |
| `application/tools/task_dispatch_tool.py` | 新写 | 用 LangGraph 子图实现，**不再用 FunctionTool 包 Agent** |
| `application/agents/main_agent.py`、`search_agent.py`、`trade_agent.py` | 新写 | `create_react_agent`；主 Agent 挂全量工具，子 Agent 挂子集 |
| `application/agents/orchestrator.py` | 新写 | 消费 `astream_events` 映射到 12 事件 + 记忆注入 + 语义缓存 + 有界重试 |
| `application/prompts/globex.yml` | 搬 | 仓库 prompt 文本直接复用 |
| 现有 `tools/`（item_search / price_compare / shipping_calc / item_picker / shopping_summary / planner / chat_fallback） | 归档 | 不进主线，留作「课程 9 工具 vs 仓库 6 工具」对照与面试谈资 |

**验收标准**：`scripts/smoke_e2e.py` 通过；`eval/cases.yaml` 前 5 条 case 通过；主 Agent 单干 + 派发两条路径都跑通。

### 阶段 4：presentation/ + composition/ + worker/（仓库搬）

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `presentation/server.py`、`connection.py`、`dto.py` | 搬 | FastAPI（同步/异步/任务查询/订单/health） |
| `frontend/*` | 搬 | React 18 + Vite + TS（对话流 + 商品卡 + 事件时间线） |
| `composition.py` | 搬 | 装配根（API/worker 共用），改为装配 LangGraph 工厂 |
| `worker.py` | 搬 | Redis Stream 消费进程 |

**验收标准**：`docker compose` 起 app+worker+opensearch+redis+frontend；WS 收到 12 事件且前端渲染；`/health` 报告 database/redis/queue 状态。

### 阶段 5：评测/ 与部署/（仓库 + 现有）

| 文件 | 来源 | 说明 |
| --- | --- | --- |
| `eval/cases.yaml` + `scripts/eval_regression.py` | 搬 | 13 case，LLM judge P0/P1/P2 |
| `scripts/eval/run_category_recall.py`、`run_recall_eval.py` | 保留 | 模块级召回门禁（Recall@K / MRR / NDCG） |
| `docker/docker-compose.yaml` | 搬 + 改 | 仓库结构，qdrant 服务换成 opensearch |
| `infra/opensearch/` | 保留 | 本机 Docker 单节点 2.19.1 + IK |
| SFT / Agentic RL | 不落地 | README 记「演进方向」 |

**验收标准**：13 case 回归报告出分；召回门禁 Recall@10≥0.75 / MRR≥0.65 / NDCG@10≥0.70；compose 全链路起且评测可复跑（评测前关语义缓存，防「评缓存」）。

---

## 8. 迁移总原则

1. 每阶段结束跑 `pytest` 全绿再进下一阶段，不做一次性整体 rename。
2. 韧性/缓存/队列/幂等的「纯 Python 核心」直接搬，「框架薄壳」用 LangGraph 对应物重写（见 §6.3）。
3. 现有 9 工具与召回评测不删，归档为对照/门禁，避免推倒重来丢掉已验资产。
4. 事件名、prompt、case 与仓库 B 保持一致，降低迁移时的对照成本。
