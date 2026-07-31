# CrossShop Agent

CrossShop Agent 是一个基于 AgentScope 2.x 的跨境购物助手参考实现。它把商品检索、品类知识解释、跨轮对话上下文、偏好记忆和受控下单流程组合在一起；订单流程只用于演示和测试，不连接真实支付，也不对价格、库存、税费、配送或平台履约作实时保证。

## 核心能力

- 按平台检索规范化商品事实，支持商品属性、价格和多平台结果组织。
- 使用 CategoryInsight 知识检索回答品类选择、属性和风险问题，并保留来源元数据。
- 通过 AgentScope 的 `AgentState` 维护本轮及跨轮上下文，支持上下文压缩、恢复和偏好选择。
- 提供受约束的购物车/订单确认与取消流程，以及会话事件和前端展示接口。
- 可选接入 Redis 做缓存、任务削峰、跨进程事件转发和共享熔断；依赖不可用时会按配置退化。

## 架构与数据流

一次请求的大致路径是：`HTTP/WebSocket → MainAgent → SearchAgent/TradeAgent → 商品/知识检索与工具 → SQLite 持久化 → 事件返回前端`。

### 商品检索

商品事实来自只读的规范化 JSONL 商品事实仓（由 `JsonlProductRepository` 读取）。离线脚本把商品写入 OpenSearch Product Index；在线检索使用同一商品索引的 ANN 向量召回与 BM25 关键词召回，再用 RRF（Reciprocal Rank Fusion）合并，必要时调用独立 reranker。商品索引由脚本离线构建，应用在线只读，不把商品目录复制进 SQLite。

### CategoryInsight RAG

品类知识使用独立的 OpenSearch Knowledge Index。CategoryInsight 文档先切块、向量化并离线发布，运行时从该索引召回，再由 Agent 结合问题和来源元数据生成解释。Markdown 是构建输入，不是线上检索的数据源；构建和发布入口见 `scripts/index/build_knowledge_opensearch.py`。

### 会话、订单与缓存

- SQLite 保存会话、对话消息、事件、订单、订单明细、买家偏好，以及每个 session 最新的一份完整 `AgentState`。每轮结束时该 session 的快照覆盖更新；它不是版本库，项目不提供历史状态回滚。
- 订单表是交易状态的事实来源；`AgentState` 快照只负责恢复对话上下文，不作为订单账本。
- `DATABASE_URL` 未设置时默认使用 `DATA_DIR/crossshop.db`；设置为 `file` 时切换到本地 JSON 文件实现（适合本地排查，不提供数据库级并发控制）。
- Redis 是可选组件：可用于 embedding/语义缓存、幂等键、Redis Streams 任务队列、Pub/Sub 事件背板，以及开启 `BREAKER_SHARED=1` 时的跨实例共享熔断状态。Redis 不保存 `AgentState` checkpoint；checkpoint 只由会话存储负责。

## 数据边界

仓库内的 `examples/data/products.jsonl` 是合成商品示例，只服务于本地测试和演示，不代表真实市场商品。`knowledge/public-demo-v1/` 共包含 13 篇可公开演示的 CategoryInsight 文档：8 篇目录提炼示例（属性、款型代理、历史价格区间和选购/避坑各 2 篇），以及 5 篇带来源信息的官方页面中文概括。后者不是官方原文，也不是会自动更新的实时规则；具体来源、查阅日期和适用范围保留在 provenance 元数据中。公开发布流程见 [docs/RAG_DATA.md](docs/RAG_DATA.md)。

私有环境中的完整 72 篇语料（64 篇历史目录知识卡与 8 篇官方来源概括）及其 CategoryInsight 发布物不随公共仓库分发。请把私有语料放在仓库外，使用独立 source root 和索引 alias 构建；不要把私有文档、真实商品目录、生成索引或运行产物提交到 Git。

## 目录结构

```text
app/                 领域模型、Agent 编排、用例、基础设施和 FastAPI 接口
scripts/index/       商品与 CategoryInsight OpenSearch 索引构建/审计脚本
scripts/eval/        召回、上下文和 rubric 评测脚本
examples/data/       合成公共商品示例
knowledge/           公开知识与 CategoryInsight 示例文档
eval/                评测案例、标注和数据契约
tests/               离线单元/契约测试及可选集成测试
frontend/            React + Vite 前端
docker/              Docker Compose 全栈编排
docs/                数据边界与设计演进记录
```

## 配置与密钥安全

复制 `.env.example` 为 `.env` 后按需修改。常用配置包括：

- LLM：`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`；服务使用 OpenAI-compatible API。
- 商品检索：`PRODUCT_CATALOG_ROOT`、`PRODUCT_EMBEDDING_BASE_URL`、`PRODUCT_EMBEDDING_MODEL`、`PRODUCT_EMBEDDING_DIM`、`OPENSEARCH_ENDPOINT`。
- 知识检索：`CATEGORY_KB_BACKEND`、`CATEGORY_KNOWLEDGE_INDEX`、`CATEGORY_KB_COLLECTION`。
- 存储与运行：`DATA_DIR`、`DATABASE_URL`、`REDIS_URL`、`QUEUE_ENABLED`、`BREAKER_SHARED`、`CORS_ORIGINS`。
- 可选能力：`RERANKER_BASE_URL`、`TAVILY_API_KEY`、LangFuse/OTLP 配置。

环境变量优先于 `.env`。不要提交 `.env`、API key、私有语料、真实目录、索引、数据库或评测产物；Docker Compose 对必需的 LLM 和商品 embedding 配置使用启动时校验，避免把密钥写入镜像或 Compose 文件。

## 快速开始

最小本地启动（Python 3.11–3.13、`uv`、Node.js 18+）：

```powershell
Copy-Item .env.example .env
# 在 .env 或当前终端中设置可用的 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
uv sync
uv run uvicorn app.presentation.server:app --reload --port 8000
```

服务启动后可访问 `http://127.0.0.1:8000/health`。要运行完整商品/知识检索，还需准备相应的 JSONL 事实仓、embedding/reranker 服务和 OpenSearch 索引；索引脚本的实际参数以以下帮助为准：

```powershell
uv run python scripts/index/build_product_opensearch.py --help
uv run python scripts/index/build_knowledge_opensearch.py --help
```

前端开发服务器：

```powershell
cd frontend
npm ci
npm run dev
```

也可以使用 Docker Compose 启动 app、worker、OpenSearch、Redis、Qdrant 和前端；先在宿主机环境中设置 Compose 文件要求的 LLM 与商品 embedding 配置，再运行：

```powershell
docker compose -f docker/docker-compose.yaml up -d --build
```

## 测试与 CI

默认测试不依赖私有语料和在线服务，使用 fake/in-memory 实现覆盖主要契约：

```powershell
uv run ruff check .
uv run pytest
```

OpenSearch、embedding、reranker、真实 LLM、Redis 和私有目录相关检查均为显式 opt-in 的集成/评测任务。GitHub Actions 的 offline 检查运行公共离线套件，frontend 检查执行 TypeScript/Vite 构建；评测报告和运行产物不会提交。

## 许可证与第三方数据

项目代码采用 Apache License 2.0，详见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。第三方依赖遵循各自许可证；商品目录、来源资料、模型和生成索引的使用与再分发边界见 [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md)。
