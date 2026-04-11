# 商品召回与知识卡 RAG v4 Challenge 数据契约

> **已废弃（历史记录）**：本 v4 与 v3 同属「英文机器翻译成中文充当商品」方案族，
> 已被真实 ShopSimulator 淘宝商品取代。本文档仅保留作历史记录。

## 1. 定位

v3 原样保留为 easy/smoke set，用于检查链路能否运行；v4 是新的 challenge set，
用于暴露真实的排序差异。v4 没有训练集，也不执行对比训练、Hard Negative 训练或
Reranker 微调。`max_seq_length=512` 是本次离线推理编码上限，不是训练阶段的
256-token 截断。

两条链路必须分开解释：

1. 商品召回主线：纯 Query/Item BGE-M3 ANN Top-100 → BGE Reranker Top-10。
2. 知识卡 RAG：BM25、BGE-M3 KNN、课程动态 Hybrid Top-30 → Reranker Top-10。

BM25 在商品侧只作为基线；Hybrid 只允许作为扩展消融，不能替代课程主线的 ANN 候选。

## 2. 数据规模与来源边界

| 数据 | v4 规模 | 约束 |
| --- | ---: | --- |
| 品类 | 20 | 12 个原品类 + 8 个电子产品品类 |
| 商品事实 | 2,356 | 每品类至少 110，最多 220 |
| 商品事实来源 | 370 / 986 / 1,000 | `real_esci` / v3 synthetic / v4 challenge synthetic |
| 知识卡 | 300 | 每品类 15 张 |
| 卡片组成 | 40 / 240 / 20 | bestseller / attribute / price_range |
| 知识卡评测 | 160 intents / 320 Query | 每品类四种 Query 类型各 2 个，中英成对 |
| 商品评测 | 80 intents / 160 Query | 每品类四种挑战类型各 1 个，中英成对 |

真实与合成边界：

- `real_esci` 只表示保留的 ESCI 英文商品文本事实。
- 中文标题、中文属性、价格、90 天订单数和 bestseller 排序均为离线生成，不是真实市场事实。
- `synthetic_llm_template` 与 `synthetic_challenge_v4` 均为未人工复核的合成商品。
- 所有 v4 Query 都是模型编写的离线评测夹具；中文 Query 是未人工复核的成对翻译。
- v4 可以用于学习和链路诊断，不能被描述为商业效果、真实销售趋势或真实用户分布。

## 3. 知识卡契约

`category_cards_v4.jsonl` 继续严格保持课程七字段：

```text
card_id, category, card_type, summary, raw_evidence,
last_updated, confidence
```

每品类 15 张卡由以下内容构成：

- 2 张 bestseller：离线生成热度排序，仅表示知识卡结构演示。
- 7 张全品类 attribute：对应七个结构化属性分布。
- 5 张形态条件 attribute：描述某一商品形态下的属性分布，增加同品类区分难度。
- 1 张 price_range：按离线生成价格的三分位数形成入门/中端/高端档。

v3 曾把一个品类的全部商品形态复制到该品类每一张卡的检索文本里，导致 BM25 只要识别
品类就能取回全部 10 张卡。v4 的 retrieval sidecar 只包含“品类 + 卡类型 + 本卡摘要”，
不再做全形态扩写。每品类有 15 张卡，而最终只取 Top-10，所以“识别品类”不再等于
“自动召回全部正例”。

知识卡候选池为全局 300 张卡。每条 Query 恰有 5 张正例，顺序 gain 为 5、4、3、2、1；
其余 295 张显式标 0。中英文共享 `intent_id`、候选、正例、gain 和判断，不存在未标注卡。

价格卡按意图决定重要性：明确预算为强正例，宽泛品类洞察为中等正例，纯场景/风格可不
标价格卡；不会强制每条 Query 都包含价格卡。

## 4. 商品召回契约

每条商品 Query 使用 500 个封闭候选，其中恰有 5 个 Exact：

- 先加入最多 95 个同品类非 Exact 困难负例；
- 再加入电子、园艺、出行工具、服饰等近邻品类负例；
- 最后用其他跨品类商品补足 500；
- 满足全部 Exact 条件但未被选入五个正例的商品，在冻结候选池前排除，绝不改标为负例。

四种挑战 Query 为：

1. `lexical_constraints`：明确属性约束；
2. `scenario_paraphrase`：场景表达加属性偏好；
3. `negative_constraint`：包含“要 A、不要 B”的否定约束；
4. `colloquial_scenario`：口语场景，可省略标准品类名。

Recall/MRR 只把 Exact 当正例；Substitute 仅以 0.01 gain 进入 NDCG。每条候选池的
500 个 Item 都有 Exact/Substitute/Irrelevant 标签，中英 Query 共享全部标签。商品文本
包含一个标题和完整双语属性正文，不会再次把标题拼进 body。

## 5. 冻结与防泄漏

`challenge_freeze_v4.json` 在首次评测前写入数据哈希与固定配置：

- OpenSearch：`ik_max_word` 建索引、`ik_smart` 查询；
- 向量模型：`BAAI/bge-m3`，512 token；
- 知识卡动态权重：名词 0.5/0.5，属性 0.7/0.3，风格 0.9/0.1，口语 1.0/0.0；
- 知识卡：coarse Top-30、final Top-10、contextual rerank；
- 商品：Faiss HNSW + IP，ANN Top-100、Reranker Top-10；
- Reranker：`BAAI/bge-reranker-v2-m3`，`blog_04` CUDA FP16。

v4 Query 文本与 v3 知识卡 final holdout、v3 商品 Query 均为零重复。首次结果完成后，
不得再根据 v4 test 指标修改 Query、标签、权重或阈值并继续称其为独立 test。

## 6. 文件与复现

```powershell
.\.venv\Scripts\python.exe scripts\data\build_challenge_v4.py
.\.venv\Scripts\python.exe -m pytest tests\unit\test_challenge_v4_dataset.py -q
```

关键文件：

- `data/category_insight/category_item_facts_v4.jsonl`
- `data/category_insight/category_cards_v4.jsonl`
- `data/category_insight/category_card_provenance_v4.jsonl`
- `data/category_insight/category_retrieval_texts_v4.jsonl`
- `data/category_insight/category_recall_challenge_v4.jsonl`
- `data/category_insight/category_recall_challenge_manifest_v4.json`
- `data/category_insight/challenge_freeze_v4.json`
- `data/eval/product_v4/recall_items.jsonl`
- `data/eval/product_v4/recall_cases.jsonl`
- `data/eval/product_v4/recall_manifest.json`

