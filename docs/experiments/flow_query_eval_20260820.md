# Flow Query 全流程真实模型评测阶段总结

> 日期：2026-08-20
> 分支：`codex/migrate-langgraph-ddd`
> 状态：基础 13 条回归已完成；flow query 流程运行已完成；per-case LLM rubric 评测已完成。

## 背景

基础 13 条应用级 case 用于验收固定业务场景，覆盖预算检索、到手价、订单闭环、记忆、无货不编造等。为了扩大真实模型评测覆盖面，本次把仓库中已有的 RAG/召回 query 直接作为完整 Agent 流程输入，不再只反复跑 13 条手工 case。

## 已完成

### 1. 基础 13 条真实模型回归

- 完整 13 条全部 PASS，平均分 1.000
- 报告：`eval/report-20260819-223705.md`
- 使用现有 `scripts/eval_regression.py` + `eval/cases.yaml`

### 2. Flow query 全流程运行

新增脚本：`scripts/eval_flow_queries.py`

它读取现有 query 文件，把每条 query 作为真实用户输入发送到 `/commerce/intents`，记录 Agent 最终回复、耗时和错误，并生成 Markdown 报告。

本次实际运行 9 条：

| 来源 | 条数 |
|---|---|
| `data/category_insight/taobao_zh/category_recall_cases_taobao_zh.jsonl` | 3 |
| `data/processed/eval/shopsimulator_retrieval_v1/queries.jsonl` | 3 |
| `data/processed/eval/amazon_esci/us/queries.jsonl` | 3 |

结果：`9/9` 正常完成，平均耗时约 20.6 秒/条。

报告：`eval/flow-report-20260819-232003.md`

### 3. 通用 rubric 的 LLM Judge 初测

新增脚本：`scripts/eval_rubric_flow_report.py`

该脚本只基于 flow report 中的 query + 最终回复做通用 P0/P1/P2 打分。

结果：`3/9` P0 全过，平均分 `0.531`

报告：`eval/rubric-report-20260819-233756.md`

该结果偏低的主要原因是 Judge 只看到最终回复，没有商品库事实表、系统规则或工具返回，因此把无法核验的具体商品/价格判为疑似编造。此版本只作为中间诊断，不代表最终口径。

### 4. Per-case LLM rubric 评测脚本

新增脚本：`scripts/eval_flow_rubric.py`

该脚本已对齐基础 13 条的评测口径：

1. 读取 `data/conversations/flow-*.jsonl`，还原 query、最终回复和 `product_search_tool` 返回的商品事实
2. 用 LLM 为每条 query 单独生成 P0/P1/P2 rubric
3. 用 LLM Judge 结合商品库事实表、系统运费/关税规则、该 case rubric 打分
4. PASS 条件与基础 13 条一致：P0 全过且总分 >= 0.7
5. 支持 `--resume`，中途断网可续跑

脚本已完成本地解析自检：9 条 flow 会话全部能正确还原。

### 5. Per-case LLM rubric 正式结果

运行命令：

```powershell
.\.venv\Scripts\python.exe scripts\eval_flow_rubric.py
```

结果：`4/9 PASS`，平均分 `0.628`

报告：`eval/flow-rubric-20260820-003923.md`

| query | 结果 | 分数 |
|---|---|---|
| category 羽毛球包：大容量单肩 | PASS | 0.883 |
| category 羽毛球包：靠谱省心 | PASS | 0.812 |
| category 羽毛球包：选购指南 | PASS | 0.825 |
| esci：toddler turtleneck girls purple | FAIL | 0.502 |
| esci：weird items | FAIL | 0.698 |
| esci：woman ski pants | FAIL | 0.300 |
| shopsim：乳头保护罩 | FAIL | 0.623 |
| shopsim：篮球近视护目镜 | FAIL | 0.225 |
| shopsim：运动鞋 | PASS | 0.788 |

主要失败模式：

- 省略或错误处理系统运费/关税规则
- 对“价格不可用”的商品给出价格范围或具体数字
- 添加商品库事实表中不存在的店铺、库存、尺码建议等信息
- 商品库无匹配结果时仍声称“存在但超预算”
- 只展示 1 个候选，未按生成的 rubric 要求列出全部符合条件商品

### 6. 真实 OpenSearch + CategoryInsight RAG 完整链路

已有 OpenSearch 容器 `globex-category-opensearch` 在 `127.0.0.1:9200` 运行，知识卡索引 `globex_category_kb_taobao_zh_v1` 存在且包含 48 张卡。

新增串联脚本：`scripts/eval_flow_with_rubric.py`

流程：

1. 从 `data/category_insight/taobao_zh/category_recall_cases_taobao_zh.jsonl` 自动挑选各品类 query
2. 通过真实 Agent 服务跑完整流程，生成会话和 flow report
3. 对同一次会话执行 per-case LLM rubric

本次运行 8 条 CategoryInsight RAG query：

- flow 阶段：`8/8` 正常完成，平均耗时 33.5 秒/条
- rubric 阶段：`3/8 PASS`，平均分 `0.667`

报告：

- `eval/flow-report-20260820-021422.md`
- `eval/flow-rubric-20260820-023633.md`

通过：

- 羽毛球包：大容量单肩（1.0）
- 乳胶枕：泰国天然乳胶 + 护颈椎（0.775）
- 颈椎按摩器（1.0）

失败：

- 汽车氛围灯（0.2）：把警示爆闪灯说成防水氛围灯
- 儿童学习椅（0.7）：对价格不可用商品给出 408-488 元
- 手机直播补光灯（0.487）：虚构价格区间与店铺
- 户外电源（0.7）：对价格不可用商品给出 655-855 元
- 平板电脑支架（0.475）：遗漏候选、虚构价格

该批次确认了真实 OpenSearch 知识卡检索已接入完整 Agent 流程；失败项仍集中在价格不可用处理、候选完整性和属性准确识别。

## 附带修复

- `scripts/eval_regression.py`：修复 `--cases` 指定自定义 case 文件时，商品库事实表仍读取默认 `cases.yaml` 的问题
- `src/globex_agent/application/prompts/globex.yml`：强化约束，禁止最终回复点名 filtered_out 的库外商品、品牌或价格

## 后续

1. 按失败模式修正 Agent 提示词或工具输出，再重跑 flow rubric
2. 对比基础 13 条与 flow query 的分数口径，确认两套评测是否可比
3. 如需更严格核验，可把 `tool.result` 也作为 Judge 输入（当前基础 13 条也未包含）
