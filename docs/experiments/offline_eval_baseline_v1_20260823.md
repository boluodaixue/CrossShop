# 14 Flow 两阶段离线评测现状基线（baseline-v1）

状态：当前 14 Flow 两阶段离线现状基线。它用于复现和比较当前实现，不表示质量全通过，也不取代旧 8 Flow 历史基线。

## 固定输入与协议

- 在线输入：`output/eval/flow-online-20260823-semantic-0823budgetfix.json`
- 输入 SHA-256：`f737d6f07a0a98413d8c5c2dbd310f52993781fb80a11aea2b6900dbcfda4995`
- 模型：`deepseek-v4-flash`
- Rubric Generator 只接收 query 与执行前契约；Final Judge 只接收紧凑用户可见证据、执行摘要和在线门禁。
- 评分权重：P0/P1/P2 = `0.30/0.30/0.40`；阈值：`0.85`。
- `inconclusive` 不计为 FAIL，也不计为 PASS。

## 当前结果

- 报告：`output/eval/offline-llm-judge-20260823-225633.md/.json`
- `completed=11/14`，`inconclusive=3/14`，`quality_pass=8/14`
- completed 平均 `final_score=0.9257`
- 225633 supersedes `output/eval/offline-llm-judge-20260823-222909.md/.json`；222909 及更早诊断报告保留，不覆盖。

## completed 但未通过

- `legacy-flow-05`：`0.840`。防水与免接线硬约束只部分满足，需求覆盖度不足（P2）。
- `legacy-flow-08`：`0.840`。品类概览维度和选项比较不足（P2）。
- `product-only-filter-mesh`：`0.797`。卡片 `highlights` 中混入 `source_tag`、`source_attributes`、`shop_name` 等字段表达，触发 P0；后续需区分用户可见商家名与内部 metadata 并清洗展示字段。

## inconclusive

- `legacy-flow-03`：Rubric P1 混入需求满足度。
- `legacy-flow-04`：Rubric P2 使用错误证据域。
- `no-evidence-quantum-hoverboard`：Rubric Generator `ReadTimeout`。

## 待改进项

- 对瞬时网络错误/超时使用有限、有界重试。
- 对 Rubric 语义越界保留一次定向修复重试，但 Schema/语义仍不合格时必须 inconclusive。
- 评测脚本已对每条输出 flush 进度行；子 Agent 转发进度可能受消息边界限制。

本基线不包含完整 6MB 在线 JSON 的复制品；完整在线 JSON、事件和工具证据仍保留在 `output/eval/` 本地审计目录。
