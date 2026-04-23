# 阶段 6：CategoryInsight 中文淘宝知识卡 RAG 整改记录

## 结果摘要

把 CategoryInsight 知识卡底座从英文 ESCI 演示版切到真实 ShopSimulator 淘宝中文目录，
首期冻结 8 个普通品类、48 张中文七字段卡片、50 条中文 Query。

- 卡片：8 品类 × 6 卡 = 48 张。
- 品类—商品事实：815 条。
- 标题门禁拒绝：121 条。
- 卡片门禁拒绝：0 张。
- 稳定抽审：5 张。
- Query：50 条，train/dev/test=30/10/10。
- Query 类型：noun 14、attribute_constraint 12、style 12、colloquial 12。
- OpenSearch 索引：`globex_category_kb_taobao_zh_v1`。
- 分词：`ik_max_word` / `ik_smart`。

## 数据口径

- 价格：单规格商品取 `price_cny`，多规格商品取全部真实 `variants[].price_cny`。
- bestseller：同质商品数 + `source_attributes` 覆盖度代理，
  标记 `generated_offline_proxy`，不生成销量。
- attribute：真实 `source_attributes` 按人工维护规则聚合，top 2 + other。
- 完整数据契约见 `docs/data/category_card_data_contract.md`。

## 实际命令

```powershell
$env:PYTHONIOENCODING = "utf-8"

.\.venv\Scripts\python.exe scripts\data\build_category_cards_taobao_zh.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval_taobao_zh.py

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

## v1 基线（当前）

独立 test split（0.92 跳过 vs 全量精排，按课程口径取 @8）：

| 方案 | Recall@8 | MRR@8 | NDCG@8 | AllCardTypes@8 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 1.0000 | 1.0000 | 0.8723 | 1.0000 |
| KNN / BGE-M3 | 1.0000 | 1.0000 | 0.8695 | 1.0000 |
| Hybrid | 1.0000 | 1.0000 | 0.8709 | 1.0000 |
| Hybrid + BGE Reranker（0.92 跳过） | 1.0000 | 1.0000 | 0.8531 | 1.0000 |
| Hybrid + BGE Reranker（全量精排） | 0.9000 | 0.9500 | 0.7915 | 0.7000 |

BM25/KNN/Hybrid 与最初缓存版一致；Reranker 在当前机器重跑为 0.8531（最初
缓存版 0.8709），GPU fp16 输出不是逐位稳定，但全量精排不升反降的结论一致。
评测脚本已为四路方案统一输出 `*_quick_top8`，主表直接读 Recall@8。

## 口语自然化实验（已回退，保留记录）

以下 v2 实验按用户要求回退，当前数据集回到 v1 基线，本节只保留实验记录。
按课程口语 Query 的语义要求，当时把 12 条 colloquial 从「品类名 + 口语套话」改为
不出现标准品类名的自然口语短句，例如「露营充电用的」对应户外电源、「能装球拍的
包」对应羽毛球包。相关卡由固定 5 张改为 1–5 张可变，数据集版本升级为
`category-card-recall-taobao-zh-v2`。

```powershell
$env:PYTHONIOENCODING = "utf-8"

.\.venv\Scripts\python.exe scripts\data\build_category_eval_taobao_zh.py

# 默认保留 0.92 首分跳过
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
  --output output\eval\category_recall_taobao_zh_v2_bypass.json

# 全量精排对照
同参数加 --rerank-all，输出到 output\eval\category_recall_taobao_zh_v2_rerank_all.json
```

独立 test split：

| 方案 | Recall@10 | MRR@10 | NDCG@10 | AllCardTypes@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.8500 | 0.9000 | 0.7707 | 0.7000 |
| KNN / BGE-M3 | 1.0000 | 0.9500 | 0.8272 | 0.7000 |
| Hybrid | 1.0000 | 0.9500 | 0.8286 | 0.7000 |
| Hybrid + BGE Reranker（0.92 跳过） | 0.8667 | 0.8200 | 0.7053 | 0.7000 |
| Hybrid + BGE Reranker（全量精排） | 0.7867 | 0.7700 | 0.6496 | 0.4000 |

口语自然化后 BM25 的 Recall@10 从 1.0 降到 0.85，KNN/Hybrid 仍为 1.0，口语
Query 主要拉开词法召回与语义召回的差距。全量精排没有提升，反而把 Reranker
的 Recall@10 从 0.8667 拉到 0.7867、NDCG 从 0.7053 拉到 0.6496；0.92 首分
跳过仍保留为生产默认。规则分类器对口语 Query 的兜底准确率由 1.0 降到 0.76，
正式指标仍使用 Planner
标注的 Query 类型。

## 遇到的问题

- 启动 Docker Desktop 时，沙箱内的 Docker 命令无法访问命名管道；经用户授权后从
  沙箱外启动 Docker Desktop，OpenSearch 容器恢复。
- `blog_04` Reranker worker 首次运行提示无法写 HuggingFace 缓存；因传入
  `--local-files-only` 和本地模型路径，模型已加载并完成评测，提示不影响指标。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_category_kb.py `
  tests\unit\test_category_insight_service.py `
  tests\unit\test_category_reranking.py `
  tests\unit\test_category_card_dataset.py `
  tests\unit\test_category_recall_dataset.py -q
.\.venv\Scripts\ruff.exe check src scripts examples tests
```
