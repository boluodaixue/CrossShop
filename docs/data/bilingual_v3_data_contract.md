# 双语商品与知识卡 v3 数据契约

> **已废弃（历史记录）**：本 v3 商品侧的「英文机器翻译成中文充当商品」方案已被真实
> ShopSimulator 淘宝商品取代（商品召回见 docs/data/shopsimulator_retrieval_v1.md）。
> 本文档仅保留作历史记录，不再作为当前商品数据契约。

## 1. 目标与边界

v3 用于离线验证两条互不混用的链路：

1. 商品召回：纯 Query/Item ANN Top-100 → BGE Reranker Top-10；
2. 知识卡 RAG：BM25、BGE-M3 KNN、动态 Hybrid Top-30 → Reranker Top-10。

本版不构造训练集，不执行对比训练、Hard Negative 训练或 Reranker 微调，也没有
User 塔。`max_seq_length=512` 只约束推理编码输入；它不是“训练阶段 256 token
截断”。

真实数据与生成数据必须分开解释：

| `source_kind` | 含义 | 可否当作真实市场事实 |
| --- | --- | --- |
| `real_esci` | 370 条已接受的 ESCI 英文品类—商品事实，底层 369 个唯一商品 | 只能把原英文商品文本当公开数据事实 |
| `synthetic_llm_template` | 模型编写双语品类规格后按固定种子展开的商品事实 | 不可以 |
| `model_authored_synthetic` | 英文评测 Query | 不可以 |
| `synthetic_translation_pair` | 与英文 intent 成对的机器生成中文 Query | 不可以，且未经人工审核 |

价格、90 天订单数和 bestseller 排序始终是离线生成字段，不是 Amazon 价格、销量或
真实榜单。中文标题、属性和 Query 是机器生成的双语夹具，不宣称人工翻译质量。

## 2. 规模与品类

v3 保留原 12 个品类，并增加 8 个电子产品品类：真无线蓝牙耳机、便携蓝牙音箱、
机械键盘、无线鼠标、USB-C 扩展坞、充电宝、网络摄像头、无线路由器。

| 数据 | 规模 | 约束 |
| --- | ---: | --- |
| 标准品类 | 20 | 英文 canonical category + 中文别名 |
| 品类—商品事实 | 1356 | 每品类至少 60 条；370 real + 986 synthetic |
| 可检索商品 v3 文档 | 1356 | `fact_id` 唯一；与原 720 个 ESCI Item 文件并存 |
| 知识卡 | 200 | 每品类 10 张 |
| 卡片组成 | 40 / 140 / 20 | bestseller / attribute / price_range |
| 知识卡开发 Query | 480 行 / 240 intents | train/dev/test 各 160 行 |
| 知识卡最终 holdout | 320 行 / 160 intents | 每品类四类型各 2 intents，中英文成对 |
| 商品合成诊断 Query | 160 行 / 80 intents | 每品类 4 个属性意图，中英文成对 |

每个品类固定生成 2 张 bestseller、7 张 attribute、1 张 price_range。这里的 10 张
是本项目 v3 的离线覆盖选择，不是课程原文规定的固定张数。

## 3. 知识卡与检索文本

`category_cards_v3.jsonl` 仍严格只有课程七字段：

```text
card_id, category, card_type, summary, raw_evidence, last_updated, confidence
```

来源、中文文本和检索文本均放在 sidecar，不污染 Schema：

| 文件 | 用途 |
| --- | --- |
| `category_item_facts_v3.jsonl` | 双语事实、属性、真实/生成来源边界 |
| `category_cards_v3.jsonl` | 200 张七字段卡片 |
| `category_card_provenance_v3.jsonl` | 逐卡来源商品和生成声明 |
| `category_retrieval_texts_v3.jsonl` | 英文、中文、双语三份派生检索文本 |
| `category_generation_specs_v3.json` | 20 品类的双语形态、7 组属性与价格边界 |
| `category_taxonomy_v3.json` | 英文 canonical category 与中文别名 |
| `audit_queue_v3.jsonl` | 稳定抽取的 10% 待人工复核队列；不得写成已人工通过 |
| `category_card_manifest_v3.json` | 数量、策略、真值边界和 SHA-256 |

派生文本包含品类、知识类型、受控商品形态和摘要。英文 Query 精排英文文本，中文
Query 精排中文文本；用于向量和 BM25 的 `retrieval_text` 是两者拼接。OpenSearch
建索引使用 `ik_max_word`，查询使用 `ik_smart`。

## 4. 完整标注规则

知识卡评测的候选池是全部 200 张卡。每个 Query 恰有 5 张正例，gain 为
5、4、3、2、1；其余 195 张均显式标为 0。中英文行共享 `intent_id`、split、候选、
正例、gain 和判断。价格卡按意图分为 strong、medium、coverage_weak，不强制固定
同一顺位。

首次 160 行 test 曾用于诊断双语文本，因此只保留为开发证据。当前实现冻结后另建
`category_recall_final_test_v3.jsonl`：160 个新 intent、320 行 Query，与开发 Query
文本零重复。最终报告只能引用这批 holdout；不得继续依据其结果调参后仍称独立 test。

商品合成诊断的每条 Query 使用 120 个封闭候选、5 个 Exact。候选内所有 Item 都有
Exact/Substitute/Irrelevant 标签；Recall/MRR 只把 Exact 当正例，Substitute 仅以
0.01 进入 NDCG。额外满足全部约束的商品在候选池冻结前排除，不能改标成负例。

## 5. 版本化文件与复现

v1/v2 文件不覆盖。v3 生成命令：

```bat
.\.venv\Scripts\python.exe scripts\data\build_category_cards_v3.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval_v3.py
.\.venv\Scripts\python.exe scripts\data\build_product_recall_v3.py
docker compose -f infra\opensearch\docker-compose.yml build
docker compose -f infra\opensearch\docker-compose.yml up -d --wait
.\.venv\Scripts\python.exe scripts\index\build_category_kb.py --local-files-only --analyzer ik_max_word --search-analyzer ik_smart --recreate
```

数据门禁由 `test_category_card_dataset.py` 和 `test_recall_dataset.py` 固化。生成成功不等
于人工审核成功；当前 `audit_queue_v3.jsonl` 仍是待审核状态。
