# CategoryInsight 数据契约（淘宝中文版）

## 目标与边界

本数据集用于离线跑通课程第 13、13-1 章的中文 CategoryInsight 知识卡 RAG，
商品底座是阶段五已经使用的真实 ShopSimulator 淘宝中文目录：
`data/processed/catalogs/taobao/cn/items.jsonl`，共 23,421 件。

当前版本只做 8 个普通商品品类，不使用食品、黄金、农资、宠物食品、仿生植物等
不适合知识卡 RAG 的品类。淘宝目录提供中文 `category_path`、中文标题、
`source_attributes`、`price_cny` 和 `variants[].price_cny`，但不提供真实销量。

| 字段 | 性质 | 可作何种解释 |
| --- | --- | --- |
| `item_id`、标题、`category_path`、`source_attributes` | ShopSimulator 真实公开字段 | 淘宝商品中文事实 |
| `price_cny` / `variants[].price_cny` | 真实观测挂牌/规格价 | 可计算品类价格档位，但不是历史成交价 |
| bestseller 排序 | 由同质商品数与属性覆盖度生成的离线代理 | 不是真实销量榜 |
| 标准品类、别名、热门形态、属性规则 | 人工维护 ground truth | 本项目离线分类口径 |

所有生成字段明确标记。当前唯一生成字段是 `bestseller_proxy_rank`，来源标记为
`generated_offline_proxy`。任何报告都不能把淘宝挂牌价描述成成交价，也不能把
代理排序描述成真实销量榜。

## 首期 8 个品类

| 标准品类 | 卡片 slug | 淘宝叶子类目 |
| --- | --- | --- |
| 乳胶枕 | `latex-pillow` | 乳胶枕 |
| 儿童学习椅 | `children-study-chair` | 儿童学习椅 |
| 手机直播补光灯 | `phone-live-fill-light` | 手机直播补光灯 |
| 汽车氛围灯 | `car-ambient-light` | 汽车氛围灯/装饰灯/日行灯 |
| 平板电脑支架 | `tablet-stand` | 平板电脑支架 |
| 颈椎按摩器 | `neck-massager` | 颈椎按摩器/枕 |
| 羽毛球包 | `badminton-bag` | 羽毛球包 |
| 户外电源 | `portable-power-station` | 户外电源/移动电站 |

标准品类名取有语义的中文叶子名。`汽车氛围灯`、`颈椎按摩器`、`户外电源` 是
淘宝组合叶子名的收敛写法，完整 `category_path` 保留在 facts/provenance 中。

## 三种卡片的意义

每类各生成 2 张 `bestseller`、3 张 `attribute`、1 张 `price_range`，8 类共 48
张。每张卡片严格只有课程七字段：

```text
card_id, category, card_type, summary, raw_evidence, last_updated, confidence
```

- `bestseller`：`summary` 使用 `category：形式 / 形式 / 形式`，证据严格使用
  `title | price | reason`。排序使用 `homogeneous_item_count` 和
  `attribute_coverage` 的离线代理，不生成销量。
- `attribute`：从真实 `source_attributes` 按人工维护的三条属性规则聚合，取前
  两个主值，其余合并为“其他”，百分比按有效样本归一为 100%。
- `price_range`：真实观测价格做三分位。单规格商品取顶层 `price_cny`，多规格
  商品取全部真实 `variants[].price_cny`；卡片只表示挂牌/规格价，不是成交价。

## 标题门禁与入库门禁

品类归属采用两层：淘宝叶子类目必须命中 `source_leaf_names`，且中文标题必须
命中该品类的 `required_terms`。本版标题门禁拒绝 121 条关系，主要原因是叶子
类目相同但标题缺少标准品类词，避免跨品类噪声进入知识事实。

卡片入库继续沿用课程门禁：Schema 严格校验、`confidence >= 0.5`、summary 长度
不超过 200、evidence 数量 1–3 且每条不超过 80，三类 summary 格式分别校验。
48 张卡片拒收 0 张；稳定哈希抽审 5 张，比例不低于 10%。

## 文件契约

目录：`data/category_insight/taobao_zh/`

| 文件 | 用途 |
| --- | --- |
| `category_taxonomy_taobao_zh.json` | 8 品类、中文别名、slug、叶子类目、标题门禁、热门形态、属性规则 |
| `category_item_facts_taobao_zh.jsonl` | 815 条品类—商品事实、真实价格观测、属性命中和代理字段 |
| `category_cards_taobao_zh.jsonl` | 48 张允许写入知识库的课程七字段卡片 |
| `category_card_provenance_taobao_zh.jsonl` | 每张卡的来源商品、样本量、价格口径和代理声明 |
| `rejected_memberships_taobao_zh.jsonl` | 121 条未通过中文标题门禁的关系 |
| `rejected_cards_taobao_zh.jsonl` | 未通过卡片门禁的草卡 |
| `audit_queue_taobao_zh.jsonl` | 稳定哈希抽取的人工抽审队列 |
| `category_retrieval_texts_taobao_zh.jsonl` | 每张卡的中文派生检索文本 |
| `category_card_manifest_taobao_zh.json` | 数量、规则、输入输出 SHA-256 和真值边界 |

## 知识卡检索评测契约

50 条中文 Query 覆盖名词型、属性约束型、气质/风格型和口语型。Query 文本由
课程口径的确定性模板生成，不复用历史 ESCI Query。每条 Query 对全局 48 张卡
全部显式判断，恰有 5 张相关卡，按顺序获得 5、4、3、2、1 的 NDCG gain。

固定拆分：

```text
train/dev/test = 30/10/10
noun/attribute_constraint/style/colloquial = 14/12/12/12
```

每个 split 都覆盖四种 Query 类型。权重、阈值和规则只在 train/dev 选择，test
只报告一次。相关卡集合按 Query 类型选择：

- 名词型：价格卡、2 张 bestseller、2 张 attribute。
- 属性约束型：2 张 attribute、2 张 bestseller、价格卡。
- 气质型：2 张 attribute、2 张 bestseller、价格卡。
- 口语型：价格卡、2 张 bestseller、2 张 attribute。

## OpenSearch 索引

索引名：`globex_category_kb_taobao_zh_v1`

```text
analyzer      = ik_max_word
search_analyzer = ik_smart
dimension     = 1024
embedding     = BAAI/bge-m3，本地 D:\models\bge-m3
reranker      = BAAI/bge-reranker-v2-m3，本地 D:\models\bge-reranker-v2-m3
```

知识卡七字段不扩展。中文派生检索文本以 sidecar 形式写入 `retrieval_text` 和
`retrieval_text_zh`，同时供 BM25、BGE-M3 和 contextual Reranker 使用。

## 生成与验证

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe scripts\data\build_category_cards_taobao_zh.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval_taobao_zh.py
```

当前固定产出：815 条品类—商品事实、48 张通过门禁的卡片、0 张拒收卡片、5 张
稳定抽审样本、50 条中文 Query 完整判断集。

OpenSearch 索引与四路召回复现：

```powershell
docker compose -f infra\opensearch\docker-compose.yml up -d --wait
.\.venv\Scripts\python.exe scripts\index\build_category_kb_taobao_zh.py `
  --embedding-model D:\models\bge-m3 --device cpu --batch-size 8 `
  --max-seq-length 512 --local-files-only --recreate

.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py `
  --data-dir data\category_insight\taobao_zh `
  --index-name globex_category_kb_taobao_zh_v1 `
  --cards-filename category_cards_taobao_zh.jsonl `
  --cases-filename category_recall_cases_taobao_zh.jsonl `
  --manifest-filename category_recall_manifest_taobao_zh.json `
  --embedding-model D:\models\bge-m3 --device cpu --max-seq-length 512 `
  --reranker-model D:\models\bge-reranker-v2-m3 `
  --reranker-python C:\Anaconda\envs\blog_04\python.exe `
  --reranker-device cuda:0 --reranker-batch-size 4 `
  --reranker-max-length 256 --local-files-only `
  --coarse-k 30 --top-k 10 --splits train dev test `
  --reranker-modes contextual `
  --output output\eval\category_recall_taobao_zh.json
```

独立 test split 实测（v1 基线，按课程口径取 @8）：

| 方案 | Recall@8 | MRR@8 | NDCG@8 | AllCardTypes@8 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 1.0000 | 1.0000 | 0.8723 | 1.0000 |
| KNN / BGE-M3 | 1.0000 | 1.0000 | 0.8695 | 1.0000 |
| Hybrid | 1.0000 | 1.0000 | 0.8709 | 1.0000 |
| Hybrid + BGE Reranker（0.92 跳过） | 1.0000 | 1.0000 | 0.8531 | 1.0000 |
| Hybrid + BGE Reranker（全量精排） | 0.9000 | 0.9500 | 0.7915 | 0.7000 |

全量精排的 BGE Reranker 在 v1 基线 48 卡封闭池上没有提升，反而把 Recall@8
从 1.0 拉到 0.90、NDCG 从 0.8531 拉到 0.7915；0.92 首分跳过的默认策略既省
延迟，也避免把粗排正确卡片挤出 Top-10。

## 历史英文 ESCI 版

旧英文 ESCI 演示版文件仍保留在 `data/category_insight/` 根目录，作为历史实验，
不覆盖、不混入淘宝中文版：

- `category_cards.jsonl`、`category_item_facts.jsonl`、`category_card_manifest.json`
- `category_recall_cases.jsonl`、`category_recall_manifest.json`
- `docs/experiments/phase6_category_insight_2026-08-15.md`
- `docs/experiments/phase6_category_insight_v2_2026-08-15.md`
- `docs/experiments/phase6_bilingual_v3_2026-08-15.md`

英文版使用 ESCI 商品文本、standard 分词、固定种子合成价格和销量。它不再作为
当前 CategoryInsight 数据契约。
