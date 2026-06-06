# Globex Agent Learning

Globex Agent Learning 是用于学习、运行和逐步组装电商搜索 Agent 的正式项目。课程原文唯一来源是 D:\刘诗恩\Obsidian\电商搜索Agent资料，课程文件不复制进仓库；项目代码、测试和运行记录只保存在本仓库。

## 当前状态

当前主线已进入 LangGraph + DDD 组合项目：商品检索、CategoryInsight 品类参考、订单两阶段门禁、FastAPI/WebSocket、Redis checkpoint、Faiss/BGE 检索、证据快照、确定性 Fact Guard 和稳定 ID Judge 评测链均有代码与测试。

权威质量基线：

- 确定性 Fact Guard：8/8 Flow，P0=0。
- 稳定 ID Judge：8/8 有效，分数 1, 1, 0.93, 1, 1, 1, 1, 1，平均 0.99125，适用 P0 20/20 全通过。
- 完整 pytest 历史基线：238 passed, 9 warnings；本轮修复后的最终命令结果以提交前验收记录为准。
- 本机 4GB 权威配置：CPU FP32 Query Embedding + GPU FP16 Reranker batch 16；结果见 output/eval/local_4gb_profile.json，不重建索引、不改 Top-100→Top-10 契约。

当前明确未实现或非在线门禁：最终回答在线 Fact Guard、grounded rewrite、确定性 fallback；P0=0 是离线评测执行门槛，不是生产请求不可绕过的状态机。CategoryInsight 是可选的品类参考工具，不直接参与商品硬过滤或排序。

## 两条真实流程

在线生产链：

    Query → Orchestrator / context / cache / checkpoint
          → LLM 工具规划 → (可选) CategoryInsight 品类参考
          → ProductSearch → StandardItem
          → ProductFactSnapshot + ProductCard → LLM final answer
          → audit / conversation persistence

离线质量链：

    persisted conversation + audit → exposed evidence parser
      → deterministic Fact Guard → P0=0 评测门槛
      → stable evidence catalog → rubric → Judge → validated report

在线链目前没有把 P0/Judge 接入生产请求。下一阶段设计为：Snapshot/CategoryInsight → draft → online Fact Guard → pass return；失败则 grounded rewrite → Fact Guard，仍失败才走 deterministic fallback。该设计是后续增强，并非当前基线中的已实现能力。

## 关键边界

- CategoryInsight：检索品类知识卡并返回有范围标识的属性、价格档位、参考款型和 provenance。它证明品类级参考事实，不证明某个 SKU 的精确价格、库存、店铺或订单事实。
- ProductFactSnapshot：由 StandardItem 一次性生成的版本化内部证据 DTO，含内容 hash、evidence ID、变体绑定、来源和实际 exposed facts，是审计与评测证据边界。
- ProductCard：从同一个 Snapshot 投影出的紧凑商品结构，只服务 Search Agent/LLM；不携带完整审计状态。
- 最终回答：LLM 面向用户的自然语言结果。当前由离线 Fact Guard/评测验证，生产运行时尚未接入最终回答重写与 fallback。

商品硬事实链保持单向关系：StandardItem → ProductFactSnapshot → ProductCard → LLM。商品检索固定为 BGE-M3 Query/Item → Faiss Top-100 → BGE-Reranker Top-10；CategoryInsight 的 OpenSearch/RAG 路径与商品 Faiss 主链分离。

## 目录与入口

    src/globex_agent/domain/              领域模型与业务规则
    src/globex_agent/application/         编排、usecase、工具、证据模型
    src/globex_agent/infrastructure/      Faiss/OpenSearch、模型、缓存、仓储、韧性组件
    src/globex_agent/presentation/        FastAPI 与 WebSocket
    tests/                                单元、契约、集成测试
    scripts/                              启动、索引、基准、评测脚本
    docs/                                 当前架构、数据契约、实验和问题账本
    frontend/                             React 对话界面
    data/                                 小规模正式演示数据与本地运行数据

主要代码入口：

- src/globex_agent/composition.py：生产容器、模型设备和服务组装。
- src/globex_agent/application/agents/orchestrator.py：工具编排、审计 sanitizer 和会话持久化边界。
- src/globex_agent/application/usecases/catalog_search.py：商品召回、Snapshot 构建和 Card 投影。
- src/globex_agent/application/tools/product_search_tool.py：LLM 工具输出与审计 evidence 关联。
- src/globex_agent/application/evidence.py：ProductFactSnapshot、稳定 hash/evidence ID 和有界持久化。
- src/globex_agent/eval/fact_guard.py：确定性事实与变体/范围/价格绑定检查。
- scripts/eval_flow_rubric.py：exposed evidence、稳定 ID catalog、rubric 和 Judge 解析。

## 环境与快速开始

项目使用 Python >=3.10,<3.11，依赖和命令以 pyproject.toml 为准。密钥只放本地 .env，不要提交 .env。

    uv sync
    $env:PYTHONPATH = "."
    uv run pytest -q --basetemp D:\PycharmProjects\GlobexAgentLearning\.pytest_tmp-review
    uv run ruff check --no-cache src scripts examples tests
    .\scripts\start_dev.ps1 -SkipFrontend
    Invoke-RestMethod http://127.0.0.1:8000/health

不依赖外部模型的示例：

    $env:PYTHONPATH = "."
    uv run python examples\00_smoke.py
    uv run python examples\03_single_agent.py --case 1

scripts/start_dev.ps1 会等待 FastAPI /health；退出脚本会清理本轮启动的后端和前端进程。Redis/OpenSearch 按各自 compose 文档启动，未启动时不要把离线测试结果写成在线服务已通过。

## 本机 4GB 与生产部署

本机模板 .env.example 使用 CPU FP32 Query Embedding 常驻 worker、GPU FP16 Reranker 常驻 worker（实际 CUDA Python 路径按机器填写）。生产仍支持 Query Embedding GPU + Reranker GPU 的分离配置；本机设备分配不是生产语义变更。

契约固定为 query 1024 维、归一化、max sequence length 512，商品候选保持 Top-100，Reranker 输出 Top-10。CategoryInsight encoder 保持 CPU 预热。最终 benchmark 已写入 output/eval/local_4gb_profile.json，只引用该权威文件，不在文档治理阶段重跑。

## 测试与评测

代码变更至少运行相关 pytest、完整 pytest、Ruff 和 git diff --check。Windows pytest 临时目录清理若出现 WinError 5，必须将测试主体通过与清理权限问题分开报告。

离线评测只消费已持久化的 conversation/audit exposed evidence；stable catalog 使用业务稳定 ID，Judge criterion 必须完整覆盖且只能引用 catalog ID。当前权威报告为 output/eval/flow-rubric-luna-final-20260822.md 及同名 JSON。

## 文档导航与项目主线

- 权威文档导航：docs/README.md，每份文档标注 current / historical / design / proposal。
- 架构与实现：docs/ARCHITECTURE_AND_IMPLEMENTATION.md。
- 项目状态与路线图：docs/PROJECT_STATUS_AND_ROADMAP.md。
- 问题账本：docs/PROJECT_AUDIT_AND_REMEDIATION.md。
- 检索与数据契约：docs/data/catalog_faiss_schema_v2_20260820.md、docs/data/category_card_data_contract.md。
- 8 Flow 评测记录：docs/experiments/flow_query_eval_20260820.md。

推荐当前项目主线：在稳定证据链之上做一个有 feature flag 的在线 grounded answer 安全闭环；先实现 shadow mode、单次 bounded rewrite 和确定性 fallback，再决定是否切换生产默认。详细入口和验收见 docs/PROJECT_STATUS_AND_ROADMAP.md。
