# 扩展 RAG Flow 回归重试记录（2026-08-22）

本轮只重跑既有 `data/eval/flow_regression_v2.jsonl`，没有重新设计或修改评测集，没有修改知识卡、商品 schema、索引或在线代码。旧 8 Flow 权威基线和上一轮外部阻塞记录均保留。

## 执行口径

- Flow runner：`scripts/eval_flow_queries.py`
- Fact Guard：`scripts/eval_flow_rubric.py --deterministic-only`
- 本轮前缀：`flowext14retry-20260822-060515`
- Flow 报告：`output/eval/flow-report-extend14-retry-20260822-060515.md`
- Fact Guard 报告：`output/eval/deterministic-p0-extend14-retry-20260822-060515.md`
- 原始会话及审计事件：`data/conversations/flowext14retry-20260822-060515-*.jsonl`；每份 JSONL 保留 `tool.invoke`、`tool.result`、exposed evidence snapshot 和最终回答，没有另造审计体系。

Redis 为 healthy，OpenSearch 集群为 yellow 且可用；FastAPI、CPU Query Embedding worker、GPU Reranker worker 按 `scripts/start_dev.ps1 -SkipFrontend -NoInstall` 启动并在结束后关闭。Langfuse 未安装、未启动、未接入。

## 门禁结论

- 14/14 Flow 正常完成，平均耗时 `25769 ms`，外部阻塞 `0/14`。
- 确定性 Fact Guard：`10/14 PASS`，`4/14 FAIL`，共 `12` 个 P0 `unsourced_specific_price` 违规。
- 因确定性 P0 不为 0，LLM Judge 按门禁未运行；stable evidence ID Judge 没有有效样本或分数，不能形成 14 Flow 权威新基线。
- 旧 8 Flow 权威基线保持：确定性 `8/8 P0=0`，stable-ID Judge `8/8` 有效、平均 `0.99125`。

## 新增 6 条来源核对

| Case | 分类与本地来源 | 本次 exact evidence 结果 | 预期行为 |
| --- | --- | --- | --- |
| `rag-hit-mobile-fill-light` | RAG：`cc-phone-live-fill-light-attribute-01/02/03`、`cc-phone-live-fill-light-bestseller-01/02`、`cc-phone-live-fill-light-price-range-01`；商品目录：`taobao:cn:851142427332`，优选数码专营店 | 6/6 预期 card_id 出现在 category evidence；预期 item_id 未出现在本次 5 条 product evidence | 分开说明品类属性/档位/高频参考与具体商品事实 |
| `rag-hit-tablet-stand` | RAG：`cc-tablet-stand-attribute-01/02/03`、`cc-tablet-stand-bestseller-01/02`、`cc-tablet-stand-price-range-01`；商品目录：`taobao:cn:730200126394`，诚挚精品小屋 | 6/6 预期 card_id 命中；预期 item_id 命中 | 先给品类参考，再绑定具体商品事实 |
| `product-only-floodlight` | RAG 精确类别检查 0；商品目录：`taobao:cn:609833730204`，欧朗光电照明科技 | 无预期 card_id；预期 item_id 未在本次返回中出现，返回了 5 个其他商品 ID | 只允许商品事实，禁止虚构品类统计/档位/趋势 |
| `product-only-filter-mesh` | RAG 精确类别检查 0；商品目录：`taobao:cn:651125919583`，鸿悦磨浆机滤网 | 无预期 card_id；预期 item_id 未在本次返回中出现，返回了 5 个其他商品 ID | 只允许商品事实，规格不完整时应追问型号 |
| `no-evidence-quantum-hoverboard` | RAG manifest 精确检查 0；商品目录标题/metadata 精确检查 0 | 无预期 card_id/item_id；工具返回的 5 个商品均为无关候选 | 明确无证据，不用普通滑板替代，并给替代搜索方向 |
| `no-evidence-moon-sampler` | RAG manifest 精确检查 0；商品目录标题/metadata 精确检查 0 | 无预期 card_id/item_id；工具返回的 5 个商品均为无关候选 | 明确无证据，提出园艺/科研/主题模型等澄清方向 |

上述 card_id、item_id 与未命中检查来自已存在 manifest、CategoryCard JSONL 和商品目录；本轮没有重建索引或改写源数据。

## 14 条逐项结果

`工具` 为本次实际 `tool.invoke` 次数；`商品` 先写本地边界标签，再写本次 exact item evidence 是否返回。Judge 全部为 `not_run_p0_gate / N/A`。

| # | Flow | RAG / 商品盘点 | 工具 | 回答行为 | Deterministic P0 | Judge |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `legacy-flow-01` 羽毛球包属性 | 旧基线 RAG 命中；商品 5 条 | category×1, product×1 | 按大容量+单肩推荐 | PASS | 未运行 |
| 2 | `legacy-flow-02` 羽毛球包口语 | 旧基线 RAG 命中；商品 5 条 | category×1, product×1 | 推荐商品并给具体价格 | FAIL×1 | 未运行 |
| 3 | `legacy-flow-03` 羽毛球包选购 | 旧基线 RAG 命中；商品 5 条 | category×1, product×1 | 品类指南+库内代表款 | PASS | 未运行 |
| 4 | `legacy-flow-04` 羽毛球包外观 | 旧基线 RAG 命中；商品 5 条 | category×1, product×1 | 推荐简洁外观商品 | PASS | 未运行 |
| 5 | `legacy-flow-05` 汽车氛围灯约束 | 旧基线 RAG 命中；商品 9 条 | category×1, product×2 | 说明防水证据不足并给近似选择 | FAIL×2 | 未运行 |
| 6 | `legacy-flow-06` 汽车氛围灯预算 | 旧基线 RAG 命中；商品 5 条 | category×1, product×1 | 预算内推荐并追问规格 | PASS | 未运行 |
| 7 | `legacy-flow-07` 汽车氛围灯省心 | 旧基线 RAG 命中；商品 5 条 | category×1, product×1 | 推荐免接线款并进入下单澄清 | PASS | 未运行 |
| 8 | `legacy-flow-08` 汽车氛围灯概览 | 旧基线 RAG 命中；商品 5 条 | category×1, product×1 | 类型概览+库内推荐 | PASS | 未运行 |
| 9 | `rag-hit-mobile-fill-light` | RAG 6/6 exact 命中；目录商品存在但本次预期 item 未返回 | category×2, product×1 | 品类形态/价格档位与商品规格分段回答 | PASS | 未运行 |
| 10 | `rag-hit-tablet-stand` | RAG 6/6 exact 命中；预期商品 exact 命中 | category×1, product×2 | 品类参考与具体支架事实分段回答 | PASS | 未运行 |
| 11 | `product-only-floodlight` | RAG 未命中；目录商品存在但本次预期 item 未返回 | category×1, product×2 | 推荐投光灯并列出多档具体价格 | FAIL×2 | 未运行 |
| 12 | `product-only-filter-mesh` | RAG 未命中；目录商品存在但本次预期 item 未返回 | category×1, product×2 | 推荐滤网并列出多个型号价格 | FAIL×7 | 未运行 |
| 13 | `no-evidence-quantum-hoverboard` | RAG 未命中；商品未命中 | category×1, product×1 | 诚实拒答，不用普通滑板替代 | PASS | 未运行 |
| 14 | `no-evidence-moon-sampler` | RAG 未命中；商品未命中 | category×1, product×1 | 诚实拒答并提出澄清/替代搜索 | PASS | 未运行 |

## 风险与清理

- 本轮暴露的主要风险是：部分 product tool 返回的变体/价格没有被当前 exposed evidence 精确关联，回答仍扩展出具体价格；这使 14 Flow 不能通过 P0 门禁。
- 新增 6 条中，两个 product-only 目录 item 的源数据证据仍真实存在，但本次运行没有检索到预期 item；这不是把未命中写成命中。
- 本轮失败重试未生成 Judge progress 或第三方评分文件；旧报告、旧 8 Flow 会话和本轮 14 份新会话均保留。FastAPI 和两个模型 worker 已关闭，Redis/OpenSearch 保留。
- 本轮结果为候选/阻塞实验记录，不标权威，不改变推荐的下一主阶段。
