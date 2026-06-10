# 扩展 RAG Flow 回归评测（2026-08-22）

状态：候选实验记录，未成为权威基线。旧 8 Flow 权威报告仍保留并继续有效。

## 评测边界

本轮沿用 `scripts/eval_flow_queries.py` 的 JSONL query runner 和 `scripts/eval_flow_rubric.py` 的 exposed evidence / stable-ID 解析，数据契约为 `data/eval/flow_regression_v2.jsonl`。没有安装或接入 Langfuse，也没有修改知识卡、商品 schema、索引或 Top-100 → Reranker → Top-10 契约。

RAG 只读来源：

- `data/category_insight/taobao_zh/category_card_manifest_taobao_zh.json`：8 个品类、48 张卡。
- `data/category_insight/taobao_zh/category_cards_taobao_zh.jsonl`：卡片 `card_id`。
- `data/processed/catalogs/taobao/cn/items.jsonl`：23,421 条商品目录 item/variant。

## 新增 6 条来源与边界

| Case | 分类 | RAG 证据 | 商品证据/未命中检查 |
| --- | --- | --- | --- |
| `rag-hit-mobile-fill-light` | RAG 命中 | `cc-phone-live-fill-light-attribute-01/02/03`、`bestseller-01/02`、`price-range-01` | `taobao:cn:851142427332`，variants `5814211587618` 等，店铺 `优选数码专营店` |
| `rag-hit-tablet-stand` | RAG 命中 | `cc-tablet-stand-attribute-01/02/03`、`bestseller-01/02`、`price-range-01` | `taobao:cn:730200126394`，variants `5649516048197` 等，店铺 `诚挚精品小屋` |
| `product-only-floodlight` | RAG 未命中、商品命中 | card manifest 对 `LED投光灯` 精确品类命中数为 0 | `taobao:cn:609833730204`，300W variants `4286240901396` / `4534735108539` 等，店铺 `欧朗光电照明科技` |
| `product-only-filter-mesh` | RAG 未命中、商品命中 | card manifest 对 `豆浆机滤网` 精确品类命中数为 0 | `taobao:cn:651125919583`，尼龙/涤纶 100 目 10 个装 variants `5380673284415` / `4933440428348` |
| `no-evidence-quantum-hoverboard` | RAG 与商品均未命中 | `量子悬浮滑板` 在卡 manifest 精确匹配数为 0 | 商品目录 title/description/category/attributes 全量精确短语扫描为 0 |
| `no-evidence-moon-sampler` | RAG 与商品均未命中 | `月球土壤采样器` 在卡 manifest 精确匹配数为 0 | 商品目录 title/description/category/attributes 全量精确短语扫描为 0 |

## 运行结果

启动前置：Redis `127.0.0.1:6379` 可达，OpenSearch `127.0.0.1:9200` 集群 health 为 yellow；按 `scripts/start_dev.ps1 -SkipFrontend` 启动 FastAPI，并在进程环境中补充本地 Redis URL（未修改 `.env`）。`/health` 返回 `status=ok`、`redis=ok`、`semantic_cache=false`。CPU Query Embedding 与配置的 GPU Reranker worker 已出现于本轮进程。

Flow 运行结果：

- 14/14 query 均已发起，没有抽样；原始会话全部保存在 `data/conversations/flowext14-*.jsonl`。
- 14/14 返回 `[error] Connection error.`，有效样本 `0/14`。
- 后端日志记录上游 `httpx.ConnectError: All connection attempts failed`；端点为 `https://opencode.ai/zen/go/v1`，涉及范围仅为本轮 query 请求及运行时应暴露的 bounded evidence，连接失败前未产生有效工具证据。

确定性 Fact Guard：`output/eval/deterministic-p0-extend14-20260822.md`。结果为有效样本 `0/14`、外部阻塞 `14`、PASS `0/0`；14 条均为 `N/A`，不能写作确定性 P0=0。

LLM Judge：未运行。原因是只有在 14 Flow 有效且确定性 P0=0 后才允许进入 Judge；本轮不满足门禁。空 rubric、空 Judge、stable evidence ID 不一致规则未被绕过。

逐项结果如下。工具调用栏为“无有效调用”，因为模型在工具规划前即连接失败；Judge 均为 `not_run_external_blocked / N/A`。

| Flow | Case/query | RAG | 商品 | 工具调用 | 回答行为 | Deterministic P0 | Judge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `cizh-badminton-bag-attribute_constraint-dev-01` | 旧基线羽毛球包 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 2 | `cizh-badminton-bag-colloquial-dev-01` | 旧基线羽毛球包 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 3 | `cizh-badminton-bag-noun-dev-01` | 旧基线羽毛球包 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 4 | `cizh-badminton-bag-style-dev-01` | 旧基线羽毛球包 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 5 | `cizh-car-ambient-light-attribute_constraint-train-01` | 旧基线汽车氛围灯 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 6 | `cizh-car-ambient-light-colloquial-test-02` | 旧基线汽车氛围灯 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 7 | `cizh-car-ambient-light-colloquial-train-01` | 旧基线汽车氛围灯 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 8 | `cizh-car-ambient-light-noun-dev-02` | 旧基线汽车氛围灯 | 旧基线商品证据未生成 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 9 | `flow-rag-hit-mobile-fill-light-v2` | 命中 6 张卡 | `taobao:cn:851142427332` | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 10 | `flow-rag-hit-tablet-stand-v2` | 命中 6 张卡 | `taobao:cn:730200126394` | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 11 | `flow-product-only-led-floodlight-v2` | 未命中，精确检查 0 | `taobao:cn:609833730204` | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 12 | `flow-product-only-soy-milk-filter-v2` | 未命中，精确检查 0 | `taobao:cn:651125919583` | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 13 | `flow-no-evidence-quantum-hoverboard-v2` | 未命中，精确检查 0 | 未命中，精确检查 0 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |
| 14 | `flow-no-evidence-moon-soil-sampler-v2` | 未命中，精确检查 0 | 未命中，精确检查 0 | 无有效调用 | `[error] Connection error.` | N/A 外部阻塞 | 未运行 |

## 报告、变更与清理

- Agent Flow 原始报告：`output/eval/flow-report-extend14-20260822.md`。
- Deterministic 报告：`output/eval/deterministic-p0-extend14-20260822.md`。
- 原始 conversation/audit JSONL：`data/conversations/flowext14-*.jsonl`；旧 8 Flow 会话和报告未覆盖。
- 代码/数据变更：新增 14 条 query 契约和 4 条契约测试；修正 deterministic-only 对外部错误会话的统计，避免把无回答样本计为 P0 PASS；未提交、未 push。
- 本轮启动的 FastAPI 和模型 worker 在最终验收后关闭；Redis/OpenSearch 保留。
- 本轮不标权威新基线，旧 8 Flow 权威文件保持不变。

Langfuse 仍只规划为后续可观测旁路：可记录 trace/span、generation、工具耗时、token、cost、score；不替代 ConversationStore、ProductFactSnapshot、订单状态或本地权威证据。本轮未安装、未启动、未接入。
