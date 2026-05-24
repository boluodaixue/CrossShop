# Flow Query 全流程真实模型评测阶段总结

> 日期：2026-08-20
> 分支：`codex/migrate-langgraph-ddd`
> 状态：基础 13 条回归已完成；flow query 流程运行已完成；per-case LLM rubric 评测脚本已就绪，尚待一次联网运行。

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

## 待运行

`scripts/eval_flow_rubric.py` 尚未完成正式联网运行，原因是 Codex 自动审批层连续返回 `guardian assessment was not valid JSON`，导致需要访问外部 LLM API 的命令被拦下。

待执行命令：

```powershell
.\.venv\Scripts\python.exe scripts\eval_flow_rubric.py
```

如果中途断网：

```powershell
.\.venv\Scripts\python.exe scripts\eval_flow_rubric.py --resume
```

预期产物：`eval/flow-rubric-YYYYMMDD-HHMMSS.md`，包含每条 query 的生成 rubric、Judge 判定、分数和 PASS/FAIL。

## 附带修复

- `scripts/eval_regression.py`：修复 `--cases` 指定自定义 case 文件时，商品库事实表仍读取默认 `cases.yaml` 的问题
- `src/globex_agent/application/prompts/globex.yml`：强化约束，禁止最终回复点名 filtered_out 的库外商品、品牌或价格

## 后续

1. 完成 `eval_flow_rubric.py` 正式运行并记录 per-case rubric 结果
2. 对比基础 13 条与 flow query 的分数口径，确认两套评测是否可比
3. 如需更严格核验，可把 `tool.result` 也作为 Judge 输入（当前基础 13 条也未包含）
