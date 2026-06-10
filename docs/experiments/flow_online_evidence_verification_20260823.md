# 2026-08-23 在线证据验证与 14 Flow 运行记录

状态：candidate / diagnostic。保留旧 8 Flow 历史基线；本记录不覆盖、不升级权威性。前面的失败运行保留为历史排障证据，最终 budgetfix 结果见文末。

## 运行口径

- 数据集：`data/eval/flow_regression_v2.jsonl`，14 条，未修改原 8 Flow 历史记录。
- 主链：FastAPI → MainAgent → ProductSearch/CategoryInsight（按模型决策）→ 最小 RecommendationDraft → 在线 Fact Guard + 整段 Evidence Judge。
- 每条 REST 等待上限 90 秒；模型 SDK 与在线门禁均只在各自边界内有限重试。
- 保存：`output/eval/flow-online-20260823-0309.json`、`output/eval/final-judge-online-20260823-0309.json`。

## 结果

14/14 REST 请求在约 48.5–49.3 秒返回，但均为固定保守回答：`verification_status=unavailable`，`recommended_cards=[]`。在线首轮生成未拿到可验证草稿，因此 Evidence Judge 没有可判定输入；独立 Final Judge 按缺证据规则将 14/14 标记为 `inconclusive`。

外部失败精确证据：主模型日志记录 `openai.APIConnectionError: Connection error.`，底层为 `httpx.ConnectError: All connection attempts failed`。详情见 `output/eval/external-failure-20260823-0309.md` 与 `output/backend.stderr.log`。本次未向外部发送密钥、PII、订单确认 token 或底层目录数据。

## 权威性结论

本次不是 14/14 有效运行，不能成为权威新基线，也不把保守回答计为通过。Redis/OpenSearch 保留运行；本轮 API、CPU Query Embedding 和 GPU Reranker 已在收尾时关闭。

## 04:05 出站环境复跑（最新）

为响应 VPN/出站环境修正，使用新文件名重新完整运行同一 14 条数据，未覆盖 03:09 报告：

- `output/eval/flow-online-20260823-0405.json`：14/14 REST 请求在单条 90 秒上限内返回，0/14 有效；均为 `verification_status=unavailable` 保守回答。
- `output/eval/final-judge-online-20260823-0405.json`：14/14 `inconclusive`，原因是缺少在线事件、模型输出和可信门禁状态；没有把无效样本算作通过。
- `output/eval/external-failure-20260823-0405.md`：记录实际出站端点的 HTTP 200/401 日志状态。401 response body 未记录，因此不推断具体认证原因。

本次虽确认请求可以出站并收到上游响应，但在线证据未形成，仍是候选/诊断结果，不替代旧 8 Flow 历史基线；本轮 API 已在收尾时关闭。

## 04:46 修正 WS 捕获后的出站复跑（历史诊断）

本轮未修改 `.env`，只在进程级覆盖 `ONLINE_EVIDENCE_JUDGE_MODEL=deepseek-v4-flash` 与
`EVAL_FINAL_JUDGE_MODEL=deepseek-v4-flash`，用于覆盖同一已授权端点；两个配置在代码和模板中仍保持独立。

- `output/eval/flow-online-20260823-044639.json` / `.md`：14/14 REST 在每条 90 秒上限内返回；14/14 捕获真实 `tool.invoke/tool.result`，且完整 `model_output` 保留。只有 2/14 捕获到完整 `evidence.verify/final.result`；12/14 记录 `event_capture_error=RuntimeError: websocket capture stopped before final.result`，因此有效流程为 2/14。
- `output/eval/final-judge-online-20260823-045359.json` / `.md`：独立 Final Judge 2/14 completed、12/14 inconclusive。缺少终止事件或在线门禁证据的样本没有被重构、猜测或计为通过。
- 日志显示主模型请求存在 HTTP 200；本报告不据此推断此前 401 的具体认证原因。外发 Final Judge 只接收 query、最小 draft 和模型实际可见工具输出，PII/token 已在发送边界脱敏，审计快照仍只保留本地。

权威性结论：本次 14 条 REST 已完整发起，但只有 2/14 形成完整在线事件链，Final Judge 12 条 inconclusive，仍为候选/诊断记录，不替代旧 8 Flow 基线。

## 最终复跑（04:59 Flow / 05:05 Final Judge）

保留 04:46 记录作为历史诊断；以下为已落盘的最终复跑结果：

- `output/eval/flow-online-20260823-045947.json/.md`：14/14 REST 有界返回，12/14 Flow valid；REST 状态为 `13 supported / 1 unavailable`。
- 两条无效样本分别为缺少 `final.result` 的 WS capture error，以及 Evidence Judge unavailable/单条 90 秒路径。
- `output/eval/final-judge-online-20260823-050551.json/.md`：9/14 completed、5/14 inconclusive。

本轮仍为 candidate，不替代旧 8 Flow 历史基线。Final Judge 报告在停止指令到达前已经完成写入，因此不将其描述为中断或未完成。

## 最终 budgetfix 复跑（2026-08-23）

本次在可出站环境运行最终代码和有界预算，用新文件名保存，未覆盖前述历史诊断：

- 在线报告：`output/eval/flow-online-20260823-semantic-0823budgetfix.json/.md`。
- Final Judge 报告：`output/eval/final-judge-online-20260823-semantic-0823budgetfix.json/.md`。
- 14/14 REST HTTP 200，14/14 捕获 `final.result`，14/14 Flow valid，14/14 `verification_status=supported`，平均耗时 33.151 秒。
- 本历史报告使用旧的一阶段 Final Judge 协议，评分为 P0 50%、P1 35%、P2 15%，不得与新两阶段 Rubric/Final Judge 报告直接比较；新协议改为 P0/P1/P2 归一化后 0.30/0.30/0.40，P0 触发一票否决，阈值 0.85。
- 唯一失败是 `legacy-flow-05`：P0/P2 通过，P1 因用户要求“防水”而实际推荐未满足防水约束，score 0.65。该失败属于用户需求满足度，不是在线 Evidence Judge、WS 或外部连接失败。

## 本次实现和验收边界

- 最后一次成功且非空 ProductSearch 冻结权威 Top-K；第 2/3 次生成禁止工具调用。
- Fact Guard 读取完整冻结事实，检查 draft、selection、variant、可售性、报价上下文和金额算术；后端从同一冻结事实 hydrate 卡片。
- Evidence Judge 只接收后端确定性生成的 SemanticEvidenceBundle：answer_text 实际涉及的商品、选中 variant、可说出的报价事实和 CategoryInsight `source_boundary`。完整 SKU、variants、highlights、所有搜索输出和快照仍保留在审计，不构造 Claim Ledger、字段级 evidence catalog 或 evidence ID。
- Judge 单次与累计预算、整轮预算、WS terminal grace 和错误类型已纳入实现；Judge 异常只重试 Judge，不消耗生成次数。
- 离线只保留独立 Final LLM-as-Judge；缺证据、在线门禁 unavailable、Schema/score 无效或 Judge 异常均为 inconclusive。

## 代码与验证

关键代码：`src/globex_agent/application/agents/orchestrator.py`、`src/globex_agent/application/evidence_verification.py`、`scripts/eval_flow_queries.py`、`scripts/eval_flow_rubric.py`、`frontend/src/components/ProductCards.tsx`。

最终验收记录：完整 pytest 256 passed、最终增量定向测试 25 passed；目标文件 Ruff、`git diff --check`、Docker Compose 配置检查和 frontend production build 通过。FastAPI 与本轮模型 worker 已关闭，Redis/OpenSearch 保留运行。由于只有 13/14 quality PASS，本报告仍为候选/诊断结果，不替代旧 8 Flow 权威基线。

## 两阶段离线 Rubric/Final Judge 复跑（2026-08-23 22:29）

- 输入沿用 `output/eval/flow-online-20260823-semantic-0823budgetfix.json`，SHA-256：`f737d6f07a0a98413d8c5c2dbd310f52993781fb80a11aea2b6900dbcfda4995`；未重跑在线 Flow。
- 首条 Rubric 真实请求：输入 1167 bytes，模型 `deepseek-v4-flash`，`trust_env=False`，schema 有效，耗时约 77.9 秒；随后才启动 14 条离线两阶段评测。
- 结果：14 条中 `completed=11`、`inconclusive=3`（Rubric Schema 不合格），`quality_pass=true` 为 7 条；completed 平均 `final_score=0.8411`。
- 逐条结果、P0/P1/P2、final_score 与失败原因保存在 `output/eval/offline-llm-judge-20260823-222909.md` 和 `.json`。3 条 Schema 失败按规则计为 inconclusive，不计 PASS/FAIL。
- 该结果为新协议候选/诊断记录，不替代旧 8 Flow 权威基线；完整原始在线 JSON 与事件证据继续保留在本地。

## Rubric 门禁语义修正与重跑（2026-08-23 22:56）

- 修正 `require_verified_final` 的含义：它是执行前门禁要求，不是字面状态值；成功契约固定为 `verification_status=supported`、`evidence_judge_status=supported`、`fact_guard_passed=true`。增加确定性校验，明确拒绝字段要求 `verified`/`passed` 等不存在成功值。
- 重新使用同一固定输入（SHA-256 `f737d6f07a0a98413d8c5c2dbd310f52993781fb80a11aea2b6900dbcfda4995`），单条预检输入 1332 bytes、schema 有效、耗时约 36.3 秒；随后运行 14 条两阶段评测。每条完成立即 flush 脱敏进度行。
- 新结果：`completed=11/14`、`inconclusive=3/14`、`quality_pass=8/14`；completed 平均 `final_score=0.9257`。inconclusive 为 Rubric Schema/网络失败，按规则不计 PASS/FAIL。
- 报告：`output/eval/offline-llm-judge-20260823-225633.md/.json`。它 supersede `offline-llm-judge-20260823-222909.*`，但仍是候选/诊断记录，不替代旧 8 Flow 权威基线。
