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
| 可观测性与运维 | logging、audit/conversation、事件总线、/health、checkpoint、Redis Stream 可选队列、韧性组件已存在 | 没有 LangFuse/Trace、指标看板、工具 P99 告警、完整 token/cost 观测、灰度回滚和 K8s graceful shutdown 交付 | src/globex_agent/infrastructure/resilience.py、application/agents/orchestrator.py、presentation/server.py；课程部署观测篇/16-3 至 16-6 |
| 质量与安全 | pytest/Ruff、8 Flow 确定性 Fact Guard、Snapshot、PII/payload sanitizer、stable-ID Judge 和权威评测已完成 | 在线 Fact Guard/rewrite/fallback、P0 脚本硬门禁、CategoryInsight 调用状态机、semantic cache 证据刷新属于细节增强；生产 prompt-injection/output/security 清单仍需后续运维阶段落地 | src/globex_agent/application/evidence.py、eval/fact_guard.py、scripts/eval_flow_rubric.py、tests/；课程部署观测篇/16-6 安全护栏与K8s生产化.md |

结论：项目的 Agent 核心业务和离线可信评测已经形成，真正阻止它成为当前课程目标下“可交付完整项目”的最大一级缺口是课程级的端到端交付与运行化：前端/API 协议闭环、可复现的全栈模型服务编排，以及基本观测/运维验收尚未统一完成。它不是单个回答校验器缺失。

## 非阻塞细节增强

下列项目必须保留为后续增强，不能作为下一主线：

- 在线 final-answer Fact Guard、grounded rewrite、deterministic fallback。
- P0=0 的脚本不可绕过状态机；P0/Judge 当前属于离线质量链。
- CategoryInsight 调用顺序或状态机强制化。
- semantic cache 的 evidence/version 刷新策略。
- Judge 对 CategoryInsight components 的持续覆盖核对。

它们改善可信度和评测严谨性，但不阻止当前 Agent、前端基础交互、API、订单门禁和离线质量链继续运行。

## 推荐唯一下一主阶段：课程级可交付运行闭环

### 范围

1. 对齐前端与 API 的任务协议：启动任务、任务状态、取消、WebSocket 会话重连、最终商品卡/事件展示；按课程目标决定是否补齐文件/图片入口。
2. 固化一个可复现 Compose profile：FastAPI、React、Redis、OpenSearch 和模型服务边界可启动、可探活、可停止；本机 4GB profile 保持 CPU Query Embedding + GPU Reranker，生产 profile 保留 GPU embedding + GPU reranker/vLLM 的独立配置。
3. 为模型服务、索引服务和应用增加 readiness/依赖失败说明；启动文档必须能从空服务状态得到明确诊断。
4. 建立最小运行观测：请求/任务/会话关联 ID、工具耗时和错误事件可查；先以仓库内可复现日志/health 为基线，再为 LangFuse/指标平台保留适配边界。
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

不在本阶段修改商品 schema-v2、重建既有 Faiss 索引、改变 Top-100→Top-10、加入真实支付/外部履约、改变订单确认门禁，也不把在线 Fact Guard/rewrite/fallback、P0 硬门禁、CategoryInsight 状态机或 cache 版本刷新混入主线。

### 可验收标准

1. 新环境按一条文档命令启动 Compose profile，FastAPI、前端、Redis、OpenSearch 和声明的模型边界均有明确 healthy/blocked 结果。
2. 浏览器提交购物 query 后能看到实时事件、商品卡和最终回答；断线可重连，用户可查询/取消任务，失败状态可见。
3. 商品搜索、CategoryInsight、订单 prepare→confirm 的一次完整用户场景通过；不声称已实现真实支付。
4. 记录 query、session/task、tool elapsed、error 和 final result 的关联日志；至少 /health 和模型/索引依赖状态可诊断。
5. 运行 frontend build、相关集成测试、完整 pytest、Ruff、diff check；现有确定性 8 Flow P0=0 不回退。

## 后续两级路线图

### 第二级：生产运维与安全

在可交付运行闭环稳定后，补齐 LangFuse/指标告警、token budget 与路由降级、工具熔断/优先级队列、prompt-injection/output guard、日志治理、K8s graceful shutdown、灰度与回滚。此阶段才处理课程 16-3 至 16-6 的生产化深度。

### 第三级：数据与业务扩展

再处理增量数据导入、索引原子切换/回滚、证据和 cache 版本生命周期，以及有明确业务需求时的购物车、真实支付/履约适配。在线 Fact Guard/rewrite/fallback 和 P0 硬门禁可作为质量增强插入，但不改变这一级主干顺序。
