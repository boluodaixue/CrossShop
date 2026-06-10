# Globex Agent Learning

Globex Agent Learning 是用于学习、运行和逐步组装电商搜索 Agent 的正式项目。课程原文唯一来源是 D:\刘诗恩\Obsidian\电商搜索Agent资料，课程文件不复制进仓库；项目代码、测试和运行记录只保存在本仓库。

## 当前状态

当前主线已进入 LangGraph + DDD 组合项目：商品检索、CategoryInsight 品类参考、订单两阶段门禁、FastAPI/WebSocket、Redis checkpoint、Faiss/BGE 检索、完整商品事实快照、在线证据验证和独立 Final Judge 均有代码与测试。

权威质量基线：

- 8 Flow 是保留的历史基线，不被本轮 14 Flow 覆盖。
- 本轮完整 pytest：256 passed, 9 warnings；最终增量定向测试 25 passed；目标文件 Ruff、git diff --check、Docker 配置检查和前端 production build 均通过。
- 本机 4GB 权威配置：CPU FP32 Query Embedding + GPU FP16 Reranker batch 16；结果见 output/eval/local_4gb_profile.json，不重建索引、不改 Top-100 粗召回契约。

当前在线链已接入：生成模型只产生最小 RecommendationDraft；最后一次成功且非空 ProductSearch 冻结权威 Top-K，后端用完整冻结事实执行 Fact Guard，并把 selections 对应的商品/选中 variant 投影为 SemanticEvidenceBundle 交给一次整段 Evidence Judge。验证失败或不可用时固定保守回答并清空商品卡。离线只保留独立 Final LLM-as-Judge；缺证据、门禁不可用或 Schema 无效均为 inconclusive。CategoryInsight 是可选的品类参考工具，不直接参与商品硬过滤或排序。

2026-08-22 扩展 RAG Flow 回归的 14 条 query 已全部发起，但配置的第三方模型 endpoint 返回 `ConnectError`，有效样本为 `0/14`；确定性报告将 14 条标为外部阻塞，未运行 Judge，因此这批结果是候选实验记录，不替代旧 8 Flow 权威基线。数据契约见 `data/eval/flow_regression_v2.jsonl`，报告见 `docs/experiments/flow_rag_regression_20260822.md`、`output/eval/flow-report-extend14-20260822.md` 和 `output/eval/deterministic-p0-extend14-20260822.md`。

同日 retry 已在可出站环境完整完成 14/14 Flow，有效样本 `14/14`；确定性 Fact Guard 为 `10/14 PASS`，共 12 个 P0 具体价格证据违规，因此按门禁未运行 Judge，也不形成权威 14 Flow 基线。逐项结果和新增 6 条的 card/item exact 命中核对见 `docs/experiments/flow_rag_regression_retry_20260822.md`；新 Flow 与 Fact Guard 报告为 `output/eval/flow-report-extend14-retry-20260822-060515.md`、`output/eval/deterministic-p0-extend14-retry-20260822-060515.md`。

同日完整商品快照复跑使用前缀 `flowext14full-20260822-1918`，14/14 Flow 完成；Agent 实际看到的完整商品卡和 SKU 变体与 exposed evidence 对等，四个边界商品已核对。确定性 Fact Guard 为 `P0=0`，但 5 条因 claim-binding 无唯一证据而 inconclusive；Judge 仅 `8/14` 有效，6 条 inconclusive，有效样本平均 `1.0`，因此仍不标权威新基线。逐项结果和绝对报告路径见 `docs/experiments/flow_rag_regression_complete_snapshot_20260822.md`。

Langfuse 只作为后续可选的可观测性旁路：可承载 trace/span、generation、工具耗时、token、cost 和 score；它不替代 ConversationStore、ProductFactSnapshot、订单状态或本地权威证据。本项目本轮没有安装、启动或接入 Langfuse。

2026-08-23 本轮新 14 Flow 已全部有界完成 REST 请求，但外部主模型端点反复返回 `openai.APIConnectionError: Connection error.`，底层为 `httpx.ConnectError: All connection attempts failed`。因此 14/14 均返回保守回答、在线门禁为 unavailable、Final Judge 14/14 inconclusive；不形成权威新基线。逐项结果见 `output/eval/flow-online-20260823-0309.json`、`output/eval/final-judge-online-20260823-0309.md`，失败证据见 `output/eval/external-failure-20260823-0309.md`。

2026-08-23 04:05 在获准出站环境重新完整运行 14 Flow：14/14 REST 请求在单条 90 秒上限内返回，0/14 有效；请求实际到达 `https://opencode.ai/zen/go/v1/chat/completions`，日志同时出现 HTTP 200/401，WS 未捕获到可用工具事件，因此 Final Judge 14/14 依缺证据标为 inconclusive。新报告为 `output/eval/flow-online-20260823-0405.json`、`output/eval/final-judge-online-20260823-0405.md`，精确失败证据为 `output/eval/external-failure-20260823-0405.md`；仍不替代旧 8 Flow 权威基线。

2026-08-23 04:46 使用进程级 `ONLINE_EVIDENCE_JUDGE_MODEL=deepseek-v4-flash` 与 `EVAL_FINAL_JUDGE_MODEL=deepseek-v4-flash`（未修改 `.env`）重新完整运行 14 Flow：14/14 REST 返回，14/14 有真实 `tool.invoke/tool.result` 事件，2/14 另捕获到 `evidence.verify/final.result`；其余 12 条因缺少终止事件被明确标为 `event_capture_error`，不能计为有效。REST 返回状态为 `supported` 的 14 条不等于 14 条有效样本；独立 Final Judge 完成 2/14、12/14 inconclusive，因此仍不形成权威新基线。报告见 `output/eval/flow-online-20260823-044639.json`、`output/eval/final-judge-online-20260823-045359.json`。

## 最新 2026-08-23 最终复跑

`semantic-0823budgetfix` 是本轮最终可出站复跑，保留此前 `045947/050551` 作为历史诊断，不覆盖它们：

- 在线报告：`output/eval/flow-online-20260823-semantic-0823budgetfix.json/.md`；14/14 REST HTTP 200、14/14 捕获 `final.result`、14/14 Flow valid/supported，平均耗时 33.151 秒。
- Final Judge：`output/eval/final-judge-online-20260823-semantic-0823budgetfix.json/.md`；14/14 completed，13/14 quality PASS，平均分 0.975。
- 唯一未通过为 `legacy-flow-05`：P0/P2 通过，P1 因回答推荐的商品不满足用户“防水”约束而失败，score 0.65。它不是在线证据或连接失败，而是用户需求满足度失败。
- 这批结果仍是 candidate/诊断记录：虽然在线链 14/14 有效且门禁 supported，但 Final Judge 仅 13/14 PASS，不能替代旧 8 Flow 权威基线。

本轮最终实现和验收记录保存在 `docs/experiments/flow_online_evidence_verification_20260823.md`。FastAPI 与本轮模型 worker 已关闭，Redis/OpenSearch 保留运行。

## 两条真实流程

在线生产链：

    Query → (按需) CategoryInsight / ProductSearch
          → 硬过滤 → BGE Reranker → 冻结 Top-K（默认 5）
          → 最小 RecommendationDraft
          → Fact Guard + 整段 Evidence Judge
          → 后端按 item_id/variant_id hydrate recommended_cards
          → final.result / REST / WS（text + cards + status）

离线质量链：

    actual online final answer + final cards + complete tool outputs
      + online verification status → independent Final LLM-as-Judge
      → P0/P1/P2 or inconclusive

## 关键边界

- CategoryInsight：检索品类知识卡并返回有范围标识的属性、价格档位、参考款型和 provenance。它证明品类级参考事实，不证明某个 SKU 的精确价格、库存、店铺或订单事实。
- ProductFactSnapshot：由 StandardItem 一次性生成的版本化内部证据 DTO，保留完整 highlights 与 variants；审计保存模型实际收到的完整工具输出，不构造字段级 evidence catalog 或 bounded evidence。
- ProductCard：从同一个 Snapshot 投影出的候选卡；最终 recommended_cards 只能由后端从冻结 Top-K 按 item_id/variant_id 组装，模型不能填写卡片事实。
- SemanticEvidenceBundle：后端从完整冻结事实确定性投影，只包含 answer_text 实际选择/比较的商品、选中 variant、可说出的报价事实和 CategoryInsight source_boundary；它服务于整段语义 Judge，不替代完整审计快照，也不拆 claim 或生成 evidence ID。
- 最终回答：模型只产生 answer_text + selections；在线验证不可用或三次生成均未通过时固定保守回答并返回空卡。
- 订单：prepare/confirm 使用同一 TariffSchedule 按具体 SKU、数量和目的国复算 merchandise/freight/tariff/landed，交易授权仍只由可信 Confirm API 完成。

商品硬事实链保持单向关系：StandardItem → ProductFactSnapshot → ProductCard → LLM。商品检索固定为 BGE-M3 Query/Item → Faiss Top-100 → BGE-Reranker → 请求 Top-K（在线默认 K=5）；CategoryInsight 的 OpenSearch/RAG 路径与商品 Faiss 主链分离。

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
- src/globex_agent/application/evidence.py：ProductFactSnapshot、完整 highlights/variants 和审计投影。
- src/globex_agent/application/evidence_verification.py：最小 RecommendationDraft、在线 Fact Guard、整段 Evidence Judge 和后端卡片 hydrate。
- scripts/eval_flow_rubric.py：只读取真实在线结果并运行独立 Final LLM-as-Judge；缺证据或异常为 inconclusive。

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

契约固定为 query 1024 维、归一化、max sequence length 512，商品候选保持 Top-100，Reranker 后按请求返回 Top-K（在线默认 K=5）。CategoryInsight encoder 保持 CPU 预热。最终 benchmark 已写入 output/eval/local_4gb_profile.json，只引用该权威文件，不在文档治理阶段重跑。

## 测试与评测

代码变更至少运行相关 pytest、完整 pytest、Ruff 和 git diff --check。Windows pytest 临时目录清理若出现 WinError 5，必须将测试主体通过与清理权限问题分开报告。

离线评测先只向 `EVAL_RUBRIC_GENERATOR_MODEL` 发送 query 与执行前契约生成 P0/P1/P2 Rubric；Final Judge 只接收 query、Rubric、用户可见回答/紧凑卡片、执行摘要和在线门禁，不接收完整工具输出。P0/P1/P2 归一化后按 0.30/0.30/0.40 聚合为 0～1 的 `final_score`；P0 触发一票否决，`EVAL_PASS_THRESHOLD` 默认 0.85。缺证据、门禁不可用、Rubric/Final Judge 异常或 Schema 无效为 `inconclusive`，不计通过或失败；`supported/unsupported` 只属于在线 Evidence Judge。8 Flow 历史基线保留；新 14 Flow 只有 14/14 有效、在线门禁可信且 14/14 质量 PASS 时才可成为权威。

2026-08-23 两阶段离线复跑使用 `output/eval/flow-online-20260823-semantic-0823budgetfix.json`（SHA-256 为 `f737d6f07a0a98413d8c5c2dbd310f52993781fb80a11aea2b6900dbcfda4995`）：14 条中 11 条 completed、3 条因 Rubric Schema 不合格 inconclusive；completed 平均 `final_score=0.8411`，7 条 quality PASS。该结果是新协议的候选/诊断记录，不替代旧 8 Flow 权威基线。报告见 `output/eval/offline-llm-judge-20260823-222909.md/.json`。
随后修正了 Rubric 的在线门禁语义：`require_verified_final` 仅表示门禁要求，成功状态固定为 `supported`，Fact Guard 固定为 `true`；225633 重跑中 14 条为 11 条 completed、3 条 inconclusive，completed 平均 `final_score=0.9257`，8 条 quality PASS。该报告 supersede 222909，但仍是候选/诊断记录，不替代旧 8 Flow 权威基线。报告见 `output/eval/offline-llm-judge-20260823-225633.md/.json`。

当前 14 Flow 两阶段离线现状基线标识为 `baseline-v1`，固定记录、失败项与后续改进见 `docs/experiments/offline_eval_baseline_v1_20260823.md`。该基线是可复现现状，不等于质量全通过；`inconclusive` 永不计为 FAIL。

## 文档导航与项目主线

- 权威文档导航：docs/README.md，每份文档标注 current / historical / design / proposal。
- 架构与实现：docs/ARCHITECTURE_AND_IMPLEMENTATION.md。
- 项目状态与路线图：docs/PROJECT_STATUS_AND_ROADMAP.md。
- 问题账本：docs/PROJECT_AUDIT_AND_REMEDIATION.md。
- 检索与数据契约：docs/data/catalog_faiss_schema_v2_20260820.md、docs/data/category_card_data_contract.md。
- 8 Flow 评测记录：docs/experiments/flow_query_eval_20260820.md。

推荐当前项目主线仍是课程级可交付运行闭环；本轮只完成在线证据/卡片/订单一致性边界，不把它改写成下一主阶段。详细入口和验收见 docs/PROJECT_STATUS_AND_ROADMAP.md。
