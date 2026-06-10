# 项目状态与路线图（current）

本文档重新按“课程目标是否形成可交付项目”审计。证据以当前代码、仓库文档、实际测试和课程原文为准；课程原文只读。

## 一级模块审计

| 一级模块 | 已完成 | 部分完成/缺失 | 证据 |
| --- | --- | --- | --- |
| 面向用户的产品体验 | React 对话框、会话 ID、事件时间线、商品卡片和最终回答展示已存在 | 前端没有课程目标中的完整任务控制、取消、文件/图片入口；订单 prepare/confirm 没有完整 UI；当前提交接口同步等待结果，不能完整体现课程的“启动任务后实时消费”体验 | frontend/src/App.tsx、frontend/src/components/EventTimeline.tsx、frontend/src/components/ProductCards.tsx；src/globex_agent/presentation/server.py；课程部署观测篇/15 FastAPI接口与前后端闭环.md |
| Agent 业务闭环 | Main/Search/Trade Agent、商品检索、比价/到手价、CategoryInsight、prepare/confirm/cancel 订单链路和 checkpoint 已进入 src | 没有真实支付或外部履约；购物车不是当前课程工具目标，不能把它伪装成缺陷；跨平台 fork 的运行体验与课程 AGUI 目标仍需端到端交付验证 | src/globex_agent/application/agents/、application/tools/、application/usecases/、domain/order/；课程项目配置篇/09 Globex项目总览与工程初始化.md、工具设计篇 11–13 |
| 后端 API | /health、/commerce/intents、异步任务查询、WebSocket 事件、订单查询/准备/确认/取消 API 已存在 | 与课程示例的 /api/task、/ws/{thread_id}、取消和文件接口不是同一完整协议；API 与前端的重连、任务状态和错误恢复尚未形成验收闭环 | src/globex_agent/presentation/server.py、presentation/connection.py、frontend/src/App.tsx；课程部署观测篇/15 FastAPI接口与前后端闭环.md |
| 数据与索引生命周期 | StandardItem/schema-v2、数据迁移、Faiss 四分区索引、CategoryInsight OpenSearch 构建脚本和 hash/manifest 契约已存在 | 生产级增量导入、原子切换、回滚、版本观测和索引重建后的服务切换没有形成完整运行流程 | scripts/data/、scripts/index/、src/globex_agent/infrastructure/recall/；docs/data/catalog_faiss_schema_v2_20260820.md |
| 服务部署编排 | Dockerfile、docker/docker-compose.yaml、OpenSearch compose、本机启动脚本、Redis checkpoint 和本机模型 worker 已存在 | Compose 没有课程目标中的 vLLM、Embedding、GPU Reranker 独立服务；模型路径依赖宿主机挂载，ready/health 只覆盖基础服务；生产 profile、secret、服务切换和完整启动验收尚未闭环 | Dockerfile、docker/docker-compose.yaml、infra/opensearch/、scripts/start_dev.ps1；课程部署观测篇/16-1 Docker Compose 全栈编排与环境锁定.md、16-2 vLLM推理服务与GPU部署.md |
| 可观测性与运维 | logging、audit/conversation、事件总线、/health、checkpoint、Redis Stream 可选队列、韧性组件已存在 | Langfuse/Trace、指标看板、工具 P99 告警、完整 token/cost 观测、灰度回滚和 K8s graceful shutdown 尚未交付；Langfuse 规划仅为后续可观测旁路，不替代 ConversationStore、ProductFactSnapshot、订单状态或本地权威证据 | src/globex_agent/infrastructure/resilience.py、application/agents/orchestrator.py、presentation/server.py；课程部署观测篇/16-3 至 16-6 |
| 质量与安全 | pytest/Ruff、8 Flow 历史基线、完整 ProductFactSnapshot、PII/token sanitizer、在线 Fact Guard + SemanticEvidenceBundle + 整段 Evidence Judge、两阶段 Rubric Generator/Final Judge 协议已完成 | 新评分方案待外部评测复跑后确认；当前 `EVAL_RUBRIC_GENERATOR_MODEL` 与 `EVAL_FINAL_JUDGE_MODEL` 默认均为 `deepseek-v4-flash`，P0/P1/P2 归一化后按 0.30/0.30/0.40 聚合，阈值 0.85；旧 14 Flow 报告不覆盖；CategoryInsight 调用状态机、semantic cache 证据刷新和生产 prompt-injection/output/security 清单仍需后续运维阶段落地 | src/globex_agent/application/evidence.py、application/evidence_verification.py、scripts/eval_flow_rubric.py、tests/；课程部署观测篇/16-6 安全护栏与K8s生产化.md |

结论：项目的 Agent 核心业务和离线可信评测已经形成，真正阻止它成为当前课程目标下“可交付完整项目”的最大一级缺口是课程级的端到端交付与运行化：前端/API 协议闭环、可复现的全栈模型服务编排，以及基本观测/运维验收尚未统一完成。它不是单个回答校验器缺失。

## 非阻塞细节增强

下列项目必须保留为后续增强，不能作为下一主线：

- CategoryInsight 调用顺序或状态机强制化。
- semantic cache 的 evidence/version 刷新策略。
- Judge 对 CategoryInsight components 的持续覆盖核对。

2026-08-22 扩展 RAG Flow 回归是独立的质量验证记录，不改变推荐的“课程级可交付运行闭环”主阶段。14 条 query 已完整发起，但外部模型 endpoint `https://opencode.ai/zen/go/v1` 连接失败；有效样本 `0/14`，确定性 Fact Guard 和 Judge 均未形成可判定新基线。详见 `docs/experiments/flow_rag_regression_20260822.md`。

同日 retry 已完整运行 14/14，外部阻塞为 0；确定性 Fact Guard 为 `10/14 PASS`，12 个具体价格证据问题使 P0 门禁未通过，Judge 按规则未运行。因此 retry 仍是候选/阻塞实验记录，不替代旧 8 Flow 权威基线，也不改变推荐的主阶段。详见 `docs/experiments/flow_rag_regression_retry_20260822.md`。

随后完成的完整商品快照复跑仍是候选记录：14/14 Flow 有效，Agent 实际看到的完整 SKU 与 exposed evidence 对等；确定性 P0 总数为 `0`，但 5 条 claim-binding inconclusive。Judge 有效 `8/14`、有效平均 `1.0`，6 条 inconclusive，故不标权威 14 Flow 基线。该局部评测不改变课程级可交付运行闭环的下一主阶段，详见 `docs/experiments/flow_rag_regression_complete_snapshot_20260822.md`。

2026-08-23 按新在线协议完成 14/14 有界 Flow 请求；API 与 Redis/OpenSearch、CPU Query Embedding、GPU Reranker 均就绪，但主模型反复出现 `openai.APIConnectionError: Connection error.` / `httpx.ConnectError: All connection attempts failed`。14 条均为 `verification_status=unavailable`，Final Judge 14/14 inconclusive，不替代旧 8 Flow 基线。逐项 JSON、Final Judge 报告和失败证据保留在 `output/eval/flow-online-20260823-0309.json`、`output/eval/final-judge-online-20260823-0309.md`、`output/eval/external-failure-20260823-0309.md`。

2026-08-23 04:05 获准出站后再次完整运行 14/14；每条 REST 均在 90 秒上限内返回，但 0/14 有效，全部 `verification_status=unavailable`。实际端点日志出现 HTTP 200/401，WS 未形成模型可见工具证据，故 Final Judge 14/14 inconclusive；新报告保留在 `output/eval/flow-online-20260823-0405.json`、`output/eval/final-judge-online-20260823-0405.md`、`output/eval/external-failure-20260823-0405.md`，仍不提升为权威 14 Flow 基线。

2026-08-23 04:46 使用两个独立 Judge 配置变量的进程级覆盖完成当时的 14/14 REST 诊断；WS 捕获到 14/14 的真实 `tool.invoke/tool.result`，其中 2/14 还包含 `evidence.verify/final.result`，另 12 条因终止事件缺失标记 `event_capture_error`。独立 Final Judge 报告为 2/14 completed、12/14 inconclusive；该结果不替代旧 8 Flow 权威基线。报告：`output/eval/flow-online-20260823-044639.json`、`output/eval/flow-online-20260823-044639.md`、`output/eval/final-judge-online-20260823-045359.json`、`output/eval/final-judge-online-20260823-045359.md`。

最终 budgetfix 复跑：`output/eval/flow-online-20260823-semantic-0823budgetfix.json/.md` 为 14/14 REST HTTP 200、14/14 WS `final.result`、14/14 valid/supported，平均 33.151 秒；`output/eval/final-judge-online-20260823-semantic-0823budgetfix.json/.md` 为 14/14 completed、13/14 quality PASS、mean 0.975。唯一失败 `legacy-flow-05` 的 P0/P2 通过、P1 因未满足“防水”失败，score 0.65。该结果验证了在线协议和评分门禁，但因未达到 14/14 quality PASS，仍是 candidate/诊断记录，不替代旧 8 Flow。

它们改善可信度和评测严谨性，但不阻止当前 Agent、前端基础交互、API、订单门禁和离线质量链继续运行。

2026-08-23 新两阶段离线评测已在可出站环境完成：固定在线报告 SHA-256 为
`f737d6f07a0a98413d8c5c2dbd310f52993781fb80a11aea2b6900dbcfda4995`；首条 Rubric
输入 1167 bytes、schema 有效，耗时约 77.9 秒。14 条中 11 条 completed、3 条因
Rubric Schema 不合格 inconclusive；completed 平均 `final_score=0.8411`，7 条
quality PASS。该结果仅作为新协议候选/诊断记录，不替代旧 8 Flow 权威基线。报告：
`output/eval/offline-llm-judge-20260823-222909.md/.json`。

随后修正 Rubric 门禁语义：`require_verified_final` 仅表示需要在线门禁，成功值固定为
`verification_status=supported`、`evidence_judge_status=supported`、
`fact_guard_passed=true`，并增加逐条 flush 进度。新报告 225633 为 14 条中 11 条
completed、3 条 inconclusive，completed 平均 `final_score=0.9257`，8 条 quality
PASS；它 supersede 222909，但仍是候选/诊断记录，不替代旧 8 Flow 权威基线。报告：
`output/eval/offline-llm-judge-20260823-225633.md/.json`。

当前 14 Flow 两阶段离线现状已登记为 `baseline-v1`：该基线固定输入、协议、计分和
11 completed/3 inconclusive/8 PASS/平均 0.9257，并记录 3 条真实 FAIL 与 3 条
inconclusive。它是可复现现状，不等于质量全通过；`inconclusive` 不计为 FAIL，也不替代
旧 8 Flow 历史基线。详见 `docs/experiments/offline_eval_baseline_v1_20260823.md`。

## 推荐唯一下一主阶段：课程级可交付运行闭环

### 范围

1. 对齐前端与 API 的任务协议：启动任务、任务状态、取消、WebSocket 会话重连、最终商品卡/事件展示；按课程目标决定是否补齐文件/图片入口。
2. 固化一个可复现 Compose profile：FastAPI、React、Redis、OpenSearch 和模型服务边界可启动、可探活、可停止；本机 4GB profile 保持 CPU Query Embedding + GPU Reranker，生产 profile 保留 GPU embedding + GPU reranker/vLLM 的独立配置。
3. 为模型服务、索引服务和应用增加 readiness/依赖失败说明；启动文档必须能从空服务状态得到明确诊断。
4. 建立最小运行观测：请求/任务/会话关联 ID、工具耗时和错误事件可查；先以仓库内可复现日志/health 为基线，再为 Langfuse 可观测旁路/指标平台保留适配边界。
5. 用一个浏览器到商品回答、事件流、订单 prepare/confirm 的端到端验收场景收尾。

### 为什么优先

课程第 15 章把“浏览器输入 → AGUI 事件 → 商品清单”定义为工程最终闭环；第 16-1 至 16-6 继续要求全栈编排、模型服务、观测、成本/韧性、安全和发布能力。当前代码已经有可复用的 Agent、API、React、Docker、Redis、OpenSearch 和模型 worker 基础，最短路径是把这些边界交付成一条可重复启动和验收的系统链。

### 依赖

- src/globex_agent/presentation/server.py、presentation/connection.py
- frontend/src/App.tsx、EventTimeline.tsx、ProductCards.tsx
- src/globex_agent/composition.py
- docker/docker-compose.yaml、Dockerfile
- scripts/start_dev.ps1
- 现有 Redis/OpenSearch、4GB profile 和权威测试

### 非目标

不在本阶段修改商品 schema-v2、重建既有 Faiss 索引、改变 Top-100→Top-10、加入真实支付/外部履约或改变订单确认门禁；本轮在线证据验证、卡片 hydrate 和报价一致性属于当前任务的局部实现，不改写课程主阶段边界。

### 可验收标准

1. 新环境按一条文档命令启动 Compose profile，FastAPI、前端、Redis、OpenSearch 和声明的模型边界均有明确 healthy/blocked 结果。
2. 浏览器提交购物 query 后能看到实时事件、商品卡和最终回答；断线可重连，用户可查询/取消任务，失败状态可见。
3. 商品搜索、CategoryInsight、订单 prepare→confirm 的一次完整用户场景通过；不声称已实现真实支付。
4. 记录 query、session/task、tool elapsed、error 和 final result 的关联日志；至少 /health 和模型/索引依赖状态可诊断。
5. 运行 frontend build、相关集成测试、完整 pytest、Ruff、diff check；现有确定性 8 Flow P0=0 不回退。

## 后续两级路线图

### 第二级：生产运维与安全

在可交付运行闭环稳定后，补齐 Langfuse 可观测旁路/指标告警、token budget 与路由降级、工具熔断/优先级队列、prompt-injection/output guard、日志治理、K8s graceful shutdown、灰度与回滚。此阶段才处理课程 16-3 至 16-6 的生产化深度。

### 第三级：数据与业务扩展

再处理增量数据导入、索引原子切换/回滚、证据和 cache 版本生命周期，以及有明确业务需求时的购物车、真实支付/履约适配。
