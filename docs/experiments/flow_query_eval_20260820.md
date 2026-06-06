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

2026-08-21 证据链复测已完成。旧批次原始 `fact_violations` 计数为
`3/0/3/1/4/2/2/3=18`，18 项均为评测假阳性，不是已证实的真实幻觉。四类根因：
（1）通用审计 sanitizer 深度截断 CategoryInsight `price_tiers`；（2）变体型号/品牌前缀、
方括号代码和数量后缀未做同一 `evidence_id/item_id` 内的保守别名映射；（3）范围端点没有
明确变化语境和同一 Snapshot 的精确 min/max 绑定；（4）预算/运费等非商品金额及重叠正则
命中未过滤去重。

修复后 `ProductFactSnapshot` 从 `StandardItem` 一次生成，ProductCard、审计 Snapshot 和
eval parser 通过 `evidence_id/content_hash` 关联；CategoryInsight 使用 schema-aware 白名单
保留实际暴露的 `price_tiers`；Fact Guard 只消费 `exposed_facts`，并对变体/范围/预算语义
执行保守校验。

### 历史中间验收（已被稳定 ID 权威基线取代）

实际验收：

- 8/8 Flow 正常，确定性 P0 为 0；报告：`output/eval/flow-report-luna-p0-20260821-2212.md`。
- 现有 Snapshot 的 evidence_id/hash 均完整，CategoryInsight 每次保留 3 条 tiers，未出现
  `[omitted]`；F3 为纯品类洞察回答，因此没有 product snapshot。
- 基于同一批报告执行有界 Judge（单次 HTTP 45 秒、1 次尝试、总时限 540 秒）。首个 rubric
  请求收到 `HTTPStatusError: 500 Internal Server Error`，目标为
  `https://opencode.ai/zen/go/v1/chat/completions`；立即停止，Judge 完成 `0/8`，无有效分数，
  其余 7 项标记为外部阻塞未运行。报告：
  该历史外部阻塞报告已清理；当前权威汇总见
  `output/eval/flow-rubric-luna-final-20260822.md`。
- 完整 pytest 曾使用专用 base temp（整理时已清理）：收集 230，
  `229 passed, 10 warnings, 1 error`。唯一错误为 Windows `WinError 5` 的 pytest 临时目录
  清理权限问题，非断言失败；相关定向测试 16 passed，ProductFactSnapshot 测试 5 passed。
- Ruff `All checks passed`；未重跑 Flow；FastAPI 和模型 worker 已关闭，Redis/OpenSearch 保留。

### 2026-08-22 Judge 重试（deepseek-v4-flash，历史阻塞记录）

本轮仅基于既有 `luna-p0-20260821-2212-*` 会话执行 Judge，未重跑 8 条 Flow。新进程确认：

- model：`deepseek-v4-flash`
- base URL：`https://opencode.ai/zen/go/v1`
- 代理：仅本轮进程清除大小写 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`，未修改 `.env`
- 单请求超时 45 秒、最多 2 次尝试、总时限 840 秒

结果：前 3/8 项 rubric 生成和评分完成，但均未通过 Judge（P0 通过 `0/3`），分数分别为
`0.787 / 0.225 / 0.571`，完成项平均分 `0.528`。第 4 项在 rubric 生成阶段两次请求后
`ReadTimeout`，随后 4 项标记为 `not_run_external_blocked`；本轮 Judge 完成 `3/8`，
不能将未运行项计为通过，也不能将 `0.528` 作为 8 项平均分。

该历史 flash 重试报告已清理；当前权威汇总见
`output/eval/flow-rubric-luna-final-20260822.md`。

FastAPI/模型 worker 未启动；Redis/OpenSearch 保留。剩余风险为 Judge provider/model 路由
仍有间歇性超时，8 项完整评分尚未形成。

### 2026-08-22 Flow 4–8 独立 Judge 补测（历史中间结果）

为避免单条超时阻塞后续条目，Flow 4–8 分别以独立进程运行，均使用
`deepseek-v4-flash`、单请求 45 秒、最多 2 次尝试；未重跑 Flow 主流程。

| Flow | Rubric | Score | P0 通过 | 状态/主要 P0 原因 |
|---|---|---:|---:|---|
| 4 羽毛球包·外观简洁 | 完成 | 0.658 | 否 | 虚构 145 元商品价格；事实表价格不可用 |
| 5 汽车氛围灯·防水免接线 | 完成 | 0.667 | 否 | 虚构 11 元/个价格；事实表价格不可用 |
| 6 汽车氛围灯·预算实用 | 完成 | 0.312 | 否 | 虚构 18–21 元价格、库外 USB 触控车载呼吸灯，并将其称为便宜档/预算内 |
| 7 汽车氛围灯·靠谱省心 | 完成 | N/A | N/A | rubric 第 2 次请求 `ReadTimeout`；未进入评分 |
| 8 汽车氛围灯·主要选择 | 完成 | 0.320 | 否 | 虚构 5–30 元价格及库外规格/安装方式/投影迎宾灯等事实 |

前 3 项与本轮 4–8 合并后：Judge 完成 `7/8`，P0 通过 `0/7`，完成项平均分
`0.506`（样本数 7）；Flow 7 明确为 N/A，未计入平均分。合并报告：
上述历史合并报告已清理；当前权威汇总见
`output/eval/flow-rubric-luna-final-20260822.md`。

上述独立 flash 报告已清理，结果仅保留在本轮权威汇总及必要的稳定 ID 原始 JSON 中。

### 2026-08-22 Judge 证据链与提示词修复（本轮实现）

本轮修复范围限定为 `scripts/eval_flow_rubric.py`、对应单元测试和评测记录，
没有重跑 Agent 8 Flow，也没有修改商品检索生产流程。rubric generator 与 Judge
共用同一个有界 `judge-evidence-v1`，保留 ProductFactSnapshot 暴露的变体价格、
店铺、库存、evidence_id/hash/schema，以及 CategoryInsight 实际暴露的
`price_tiers`、`attributes`、`bestsellers` 和 category/reference scope。
`price_major=null` 不再隐藏变体价格；未知字段不再推断为“不存在”。

rubric criteria 现在带稳定 id、source_scope、evidence_fields、applies_if；Judge
支持 `pass|fail|not_evaluable`，旧 `pass` 格式仍可解析，证据缺失的
`not_evaluable` 不计为 P0 失败，也不伪装为通过；新增 `--source` 用于单 Flow
隔离评测。

验证：专用临时目录完整 pytest `235 passed, 9 warnings`；相关证据链测试默认
复现 Windows `WinError 5` 后在专用目录复核 `17 passed`；Ruff 与
`git diff --check` 通过。既有 8 Flow 确定性门禁仍为 P0=0。

外部 Judge 重评未形成新的有效评分报告：首个单会话调用在发送既有会话/商品证据到
配置的第三方 LLM endpoint 前被运行环境出站数据安全审查拦截，未到达 HTTP 层。
因此本轮 8 项 Judge、平均分和样本数均为 N/A，不能将本轮视为通过；旧
`0/7` 结果也不被本轮修复后的契约重新确认。

### 2026-08-22 Judge 重试结果（r3，已被稳定 ID 基线取代）

用户明确授权后，使用 `deepseek-v4-flash` 和
`https://opencode.ai/zen/go/v1` 对既有 8 条会话逐条隔离评测；单请求45秒、最多2次，
未重跑 Agent Flow。8/8 独立调用均完成或返回可诊断结果，HTTP 外部错误 0。

| Flow | 状态 | 原始分数 | 有效口径 | P0 |
|---|---|---:|---:|---|
| 1 羽毛球包 attribute_constraint | completed | 0.5 | 0.5 | fail：防水设计无证据 |
| 2 羽毛球包 colloquial | inconclusive | N/A | N/A | rubric 字段路径校验失败 |
| 3 羽毛球包 noun | inconclusive | N/A | N/A | rubric 字段路径校验失败 |
| 4 羽毛球包 style | 空 Judge 结果 | 1.0 | N/A | 不计为通过 |
| 5 汽车氛围灯 attribute_constraint | 空 Judge 结果 | 1.0 | N/A | 不计为通过 |
| 6 汽车氛围灯 colloquial-test | inconclusive | N/A | N/A | rubric 字段路径校验失败 |
| 7 汽车氛围灯 colloquial-train | completed | 1.0 | 1.0 | pass |
| 8 汽车氛围灯 noun | inconclusive | N/A | N/A | rubric 字段路径校验失败 |

有效可评分样本为 2/8，平均分 `0.75`（Flow 1、7），有效 P0 通过 `1/2`。
`inconclusive` 项不是 Agent Flow 失败；Flow 4/5 的原始 1.0 是 scorer 对空判定数组的
默认值，已在合并报告中标记为不可计分。主要剩余风险是 provider 生成的 CategoryInsight
嵌套字段路径与本地白名单命名不一致，以及空 Judge JSON 未被本地 schema 拒绝。

上述 r3 报告已清理；其历史结论不作为当前基线。

### 2026-08-22 稳定 evidence ID 契约重评（r4，已被 Flow1 r5 权威汇总取代）

本轮实施稳定 ID 契约：本地从 exposed evidence 生成有界 deterministic catalog，
rubric 只能引用 catalog evidence_id；Judge 必须逐等级覆盖 rubric 全部 criterion_id，
缺失、重复、多出 ID、空 rubric 或空 Judge 均判 inconclusive，不再按空 ratio 计 1.0。

验证：完整 pytest \`238 passed, 9 warnings\`；稳定 ID 相关测试 \`20 passed\`；全仓 Ruff
通过，\`git diff --check\` 通过。

仅重评此前无效的 Flow 2/3/4/5/6/8，未重跑 Agent Flow；每项独立进程、单请求45秒、最多2次。
6/6 完成，HTTP 外部失败 0。Flow 1/7 复用此前有效结果。

| Flow | 状态 | 分数 | P0 |
|---|---|---:|---|
| 1 羽毛球包 attribute_constraint | completed | 0.500 | fail：防水设计无证据 |
| 2 羽毛球包 colloquial | completed | 1.000 | 无适用 P0 criterion |
| 3 羽毛球包 noun | completed | 0.930 | pass |
| 4 羽毛球包 style | completed | 1.000 | pass |
| 5 汽车氛围灯 attribute_constraint | completed | 1.000 | pass |
| 6 汽车氛围灯 colloquial-test | completed | 1.000 | pass |
| 7 汽车氛围灯 colloquial-train | completed | 1.000 | pass |
| 8 汽车氛围灯 noun | completed | 1.000 | pass |

合并有效样本 `8/8`，平均分 `0.929`，Judge P0 通过 `7/8`；确定性 Fact Guard 仍为既有
报告的 P0=0。Flow 1 的防水设计是当前唯一被 Judge 判定的真实 Agent 事实问题。

报告：`output/eval/flow-rubric-luna-final-20260822.md`；结构化汇总：
`output/eval/flow-rubric-luna-final-20260822.json`。

### 2026-08-22 权威基线（稳定 ID r5）

最新 Flow 1 r5 与稳定 ID Flow 2/3/4/5/6/8、此前有效 Flow 7 合并后，Judge 有效
`8/8`，分数为 `1.000/1.000/0.930/1.000/1.000/1.000/1.000/1.000`，平均
`0.99125`（样本 8/8）。适用 P0 criterion 为 `20`，通过 `20`，失败 `0`；Flow 1、2、7
无适用 P0 criterion，不能写作 P0 pass。确定性 Fact Guard 为 `8/8`、P0=0。
完整 pytest 为 `238 passed, 9 warnings`，相关测试 `20 passed`，Ruff 与
`git diff --check` 通过。旧 `0/7` 与旧 18 项事实违规是评测假阳性/无效结果，已不再作为基线。
