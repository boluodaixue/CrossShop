# 扩展 RAG Flow 回归：完整商品快照候选记录（2026-08-22）

状态：候选实验记录，未成为权威基线。旧 8 Flow 权威基线和前两轮 14 Flow 失败/阻塞记录均保留。

## 执行口径

- 数据集：`data/eval/flow_regression_v2.jsonl`，保留原 8 条并新增 6 条，共 14 条；本轮 14/14 完整运行。
- 本轮前缀：`flowext14full-20260822-1918`。Agent 实际收到的商品卡、完整 SKU 变体和 CategoryInsight 结果，按同源数据写入 exposed evidence snapshot；审计层没有前 20 截断，也没有从底层目录补入 Agent 未见数据。
- 发送范围：按用户授权仅发送 14 条 query、运行时实际可见的商品/RAG 结果，以及 Judge 阶段的单条 claim 和本会话有界 evidence；没有发送密钥或 Agent 未见的底层目录数据。
- Fact Guard 只执行 stable evidence ID、商品/SKU 归属、结构化数值、RAG 边界和跨会话引用等确定性检查；语义等价交给逐 claim Judge。Fact Guard 的 claim 无法唯一绑定时保持 inconclusive，不降级为通过。
- Langfuse 未安装、未启动、未接入，仍只规划为后续 trace/span、generation、工具耗时、token、cost、score 的可观测性旁路。

## 四个边界商品核对

本轮 Agent 实际返回并暴露的完整变体与商品工具结果一致：`taobao:cn:877431065474` 为 2 个 SKU 且均为 218 元；`taobao:cn:891723786306` 为 6 个 SKU，价格为 4/5 元；`taobao:cn:717765429133` 为 40 个 SKU，包含 188、335、416、330 元；`taobao:cn:650019148657` 为 26 个 SKU，包含 13、14、18、22、24、25、27、30 元。底层目录中存在但本次 Agent 未看到的 SKU 不会进入本次 snapshot，也不能成为通过依据。

## 门禁结果

- Flow：14/14 完成，未覆盖旧报告。
- Deterministic Fact Guard：14/14 有效，P0 总数 `0`；9/14 完成，5/14 因 claim-binding 无唯一当前会话 evidence ID 而 inconclusive。这里的 0 是“没有确定性 P0 违规”，不是把 5 条写成通过。
- LLM Judge：在确定性 P0=0 后运行；8/14 有效，完成项 8/8 为 `1.0`，有效样本平均 `1.0`；6/14 inconclusive，外部阻塞最终为 0。运行期间观察到上游 HTTP 500/503 重试，未将其伪装成通过。
- 由于 Judge 不是 14/14 有效，本轮不标权威新基线。

## 14 条逐项结果

| # | Case | RAG / 商品实际边界 | 工具调用 | 回答行为 | Deterministic P0 | Judge |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `cizh-badminton-bag-attribute_constraint-dev-01` | RAG 命中；商品 5 条 | category×1, product×1 | 按大容量、单肩推荐，给商品变体价格 | PASS | 有效 1.0 |
| 2 | `cizh-badminton-bag-colloquial-dev-01` | RAG 命中；商品含 `877431065474` | category×1, product×1 | 口语化“靠谱省心”推荐，给 199 元商品事实 | PASS | 有效 1.0 |
| 3 | `cizh-badminton-bag-noun-dev-01` | RAG 命中；商品含 `877431065474` | category×1, product×1 | 品类指南与代表商品分段回答 | PASS | 有效 1.0 |
| 4 | `cizh-badminton-bag-style-dev-01` | RAG 命中；商品 5 条 | category×1, product×1 | 按外观简洁筛选，给具体商品价格 | PASS | 有效 1.0 |
| 5 | `cizh-car-ambient-light-attribute_constraint-train-01` | RAG 命中；商品含 `891723786306` | category×1, product×2 | 明确防水证据不足，只确认免接线商品 | PASS | 有效 1.0 |
| 6 | `cizh-car-ambient-light-colloquial-test-02` | RAG 命中；商品 5 条 | category×1, product×1 | 预算内推荐，给 18–21 元变体范围 | PASS | 有效 1.0 |
| 7 | `cizh-car-ambient-light-colloquial-train-01` | RAG 命中；商品 5 条 | category×1, product×1 | 推荐免接线款并保留后续澄清 | PASS | inconclusive |
| 8 | `cizh-car-ambient-light-noun-dev-02` | RAG 命中；商品 5 条 | category×1, product×1 | 类型概览与商品推荐 | INCONCLUSIVE，0 P0 | inconclusive |
| 9 | `flow-no-evidence-moon-soil-sampler-v2` | RAG 未命中；目标商品未命中，仅有无关候选 | category×1, product×1 | 诚实说明无证据，提出园艺/科普澄清方向 | PASS | 有效 1.0 |
| 10 | `flow-no-evidence-quantum-hoverboard-v2` | RAG 未命中；目标商品未命中，仅有无关候选 | product×1 | 不用普通滑板替代，明确无法推荐 | PASS | inconclusive |
| 11 | `flow-product-only-led-floodlight-v2` | RAG 未命中；商品命中 `717765429133`，40 SKU | category×1, product×2 | 只用商品事实，给 300W/188 元并列出已见变体 | INCONCLUSIVE，0 P0 | inconclusive |
| 12 | `flow-product-only-soy-milk-filter-v2` | RAG 未命中；商品命中 `650019148657`，26 SKU | category×1, product×3 | 只用商品规格和价格，未写品类统计趋势 | INCONCLUSIVE，0 P0 | 有效 1.0 |
| 13 | `flow-rag-hit-mobile-fill-light-v2` | RAG 命中 `手机直播补光灯`；商品命中 `862877808381` | category×2, product×1 | 先分开品类形态/档位，再给具体商品 78/98 元等事实 | INCONCLUSIVE，0 P0 | inconclusive |
| 14 | `flow-rag-hit-tablet-stand-v2` | RAG 命中 `平板电脑支架`；商品命中 `730200126394` | category×1, product×2 | 先品类参考，再给折叠、金属、升降商品及变体价格 | INCONCLUSIVE，0 P0 | inconclusive |

## 原始证据与风险

- Flow 报告：`D:\PycharmProjects\GlobexAgentLearning\output\eval\flow-report-extend14-full-20260822-1918.md`
- Deterministic 报告：`D:\PycharmProjects\GlobexAgentLearning\output\eval\deterministic-p0-extend14-full-20260822-1918-rerun.md`
- Judge 报告：`D:\PycharmProjects\GlobexAgentLearning\output\eval\flow-rubric-extend14-full-20260822-1918.md`
- Judge progress：`D:\PycharmProjects\GlobexAgentLearning\output\eval\flow-rubric-progress-extend14-full-20260822-1918.json`
- 原始会话/审计：`D:\PycharmProjects\GlobexAgentLearning\data\conversations\flowext14full-20260822-1918-*.jsonl`，共 14 份。

当前主要风险不是底层 SKU 未暴露，而是自然语言回答把多个 SKU、品类档位和商品端点混在同一 claim 中，Fact Guard 无法安全唯一绑定；Judge 按保守契约将这类 claim 判为 inconclusive。后续应优先让生成阶段产出高风险 claim 的结构化 evidence bindings，再改善 Judge 候选选择，而不是继续增加中文正则。
