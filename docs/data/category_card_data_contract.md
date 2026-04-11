# CategoryInsight 数据契约（ESCI 离线版）

## 目标与边界

本数据集用于离线跑通课程第 13、13-1 章的知识卡 RAG，不表示真实 Amazon
商品库、实时价格、历史成交价或销量榜。

商品事实的唯一真实底座是阶段五已经使用的 Amazon Shopping Queries ESCI
英文子集：720 个商品、35 个完整标注 Query 候选池，Apache-2.0。ESCI 只提供
商品文本和相关性标签，不提供价格、销量或官方商品类目。

因此首版数据严格区分：

| 字段 | 性质 | 可作何种解释 |
| --- | --- | --- |
| `product_id`、标题、正文、ESCI 标签 | 公开数据集原始/清洗字段 | 真实公开商品文本和相关性标注 |
| 标准品类与别名 | 人工维护 ground truth | 本项目离线分类口径 |
| 属性分布 | 从商品标题与正文确定性抽取、聚合 | 仅代表当前有效样本，不代表市场份额 |
| `typical_price_cny` | 固定种子生成 | 仅用于验证价格档提炼，不是真实价格 |
| `order_count_90d` | 固定种子生成并受 ESCI 标签权重影响 | 仅用于离线 bestseller 排序，不是真实销量 |

所有生成字段均写明 `generated_offline_not_observed`，并记录固定种子
`20260815` 和规则版本。禁止在报告中把它们描述成 Amazon 销量或成交价。

## 三种卡片的意义

`bestseller`、`attribute`、`price_range` 是三种卡片类型，并非每个品类只有
三张卡。本版 12 个品类各生成 2 张 bestseller、3 张 attribute、1 张
price_range，共 72 张。

- `bestseller`：提供代表性商品形态和最多 6 条商品证据。当前排序依据为明确
  标记的离线生成热度；`raw_evidence` 严格使用 `name | price | reason`。
- `attribute`：只统计标题/正文命中规则的有效样本，分母与计数写入 provenance；
  没有命中属性的商品不进入该属性分母。
- `price_range`：根据生成的 CNY 价格事实按三分位切成有限的 budget、mid、
  premium 区间，避免课程示例 `400+` 与 `tuple[float, float]` 解析不一致。

当前 12 个品类都是普通商品品类，不是套装品类。bestseller `summary` 中的三个
短语用于检索和拆分商品形态；后续提炼时必须读取 provenance 中的
`category_kind=ordinary`，不得把这些短语误写入“套装组件”结果。

## 品类归并

人工维护表位于 `data/category_insight/category_taxonomy.json`。它把 35 个 ESCI
Query group 归并到 12 个标准品类，并用商品标题 `required_terms/excluded_terms` 做
第二道包含/排除门禁。

Query `52717` 的 Exact 结果同时包含图书、仿真植物和标语商品，无法形成可信的
单一品类，因此显式排除。其余相关标签只允许 Exact、Substitute、Complement；
Irrelevant 不参加知识事实聚合。标题门禁共拒绝 37 条关系，其中抽审后新增排除 7
条明显噪声：标签耗材、go-kart 配件、通用车库座椅和自行车专用工具。

这个归并只服务于知识卡数据生产。它不能作为无偏线上类目分类器，也不能把同一
批 ESCI Query 直接当作阶段六最终测试集；知识卡检索评测需要单独构造、完整判断
并冻结 train/dev/test。

## 文件契约

课程索引文档只读取 `category_cards.jsonl`，每行严格只有七个字段：

```text
card_id, category, card_type, summary, raw_evidence, last_updated, confidence
```

其余信息放在 sidecar，避免污染课程 Schema：

| 文件 | 用途 |
| --- | --- |
| `category_taxonomy.json` | 标准品类、别名、Query 映射、属性规则和生成价格边界 |
| `category_item_facts.jsonl` | 商品—品类事实、文本属性、生成字段及来源 |
| `category_cards.jsonl` | 允许写入知识库的课程 Schema 卡片 |
| `category_card_provenance.jsonl` | 每张卡的商品 ID、样本量、字段来源和生成声明 |
| `rejected_memberships.jsonl` | 未通过商品文本品类门禁的关系 |
| `rejected_cards.jsonl` | 未通过 Schema/置信度/格式门禁的卡片 |
| `audit_queue.jsonl` | 稳定哈希抽取的不低于 10% 人工审核队列 |
| `audit_results.jsonl` | 逐卡审核检查项、结论、修正说明和所审核卡片语料哈希 |
| `category_card_manifest.json` | 数量、规则、输入输出 SHA-256 和真值边界 |

## 知识卡检索评测契约

`eval_query_annotations.jsonl` 人工编写并冻结了 50 条英文知识卡 Query，覆盖名词型、
属性约束型、气质/风格型和完全口语型。它们没有逐字复用阶段五的 ESCI Query。

生成后的 `category_recall_cases.jsonl` 使用全局 72 张卡作为每条 Query 的封闭候选
池。每条 Query 恰有 5 张相关卡，按重要性获得 5、4、3、2、1 的 NDCG gain；其余
67 张卡均显式写为 `not_relevant`。因此 test split 中不存在未标注卡片，也不会把
“召回后才出现的未知卡片”临时当作负例。

固定拆分为 train/dev/test=30/10/10，三个 split 都覆盖四种 Query 类型。权重、
阈值和规则只能使用 train/dev；配置冻结后 test 只报告一次。当前标注是人为构造的
课程离线真值，只能评测这 72 张知识卡，不能外推线上市场效果。

### v2 修订契约

v1 文件继续保留作为历史实验，不覆盖。v2 使用以下版本化文件：

| 文件 | 用途 |
| --- | --- |
| `eval_query_annotations_v2.jsonl` | 40 条原 train/dev 标注与 10 条新 test 标注 |
| `category_recall_cases_v2.jsonl` | 50 Query×72 卡的完整判断矩阵 |
| `category_recall_manifest_v2.json` | v2 数量、口径和输入/输出 SHA-256 |
| `category_recall_v2_freeze.json` | 在读取新 test 之前冻结的检索与精排配置 |

v1 的 10 条 test 只作为历史诊断，不进入 v2；v2 test 的 Query ID 与文本均为新建，
构建脚本会把历史 test 复用视为错误。train/dev 仍为原来的 30/10 条及原 split，
其中 3 条名词型 Query 增加明确的 `price ranges/price tiers` 意图，用于在非 test
数据上验证价格卡语义。

每条 Query 仍恰有 5 张正例，并必须包含本品类价格卡，但价格卡不是固定同一顺位：

- 明确价格、预算、档位意图：`strong`，gain 4～5；
- 宽泛品类洞察：`medium`，gain 2～3；
- 属性、风格和使用场景：`coverage_weak`，gain 1。

其余四张正例按 Query 与卡片实际内容选择，不使用固定卡片类型配额。评测额外报告
`BestsellerCoverage@K`、`AttributeCoverage@K`、`PriceRangeCoverage@K` 和
`AllCardTypesCoverage@K`；这些是观测指标，不参与强制插卡或结果配额。

知识卡七字段 Schema 不变。索引时派生统一英文检索文本，不写回卡片本体：

```text
Category: {category}. Knowledge type: {natural-language type}. Summary: {English-normalized summary}.
```

该文本同时供 BGE-M3、BM25 和 contextual Reranker 使用。价格摘要会从中文档位模板
转换为带 `price tiers / Budget tier / mid-range tier / premium tier / CNY` 的英文文本，
使只包含数字区间的卡片也具有可检索语义。`raw_evidence` 不进入检索文本。

## 生成与验证

```powershell
.\.venv\Scripts\python.exe scripts\data\build_category_cards.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval.py
.\.venv\Scripts\python.exe -m pytest tests\unit\test_category_card_dataset.py -q
```

当前固定产出：370 条品类—商品事实、369 个唯一商品、72 张通过门禁的卡片、
0 张拒收卡片。8 张稳定哈希样本已全部完成实质复核；首次抽审发现 `No Drain
Holes` 否定短语误判及 7 条跨品类噪声，修正并重建数据后全部通过。只有重新运行后
哈希、审核语料哈希与测试一致，才视为可复现。
