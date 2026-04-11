# 阶段 6：CategoryInsight 知识卡检索 v2 实验记录

日期：2026-08-15  
状态：v2 数据契约、索引、train/dev 选择和一次性独立 test 均已实际运行

## 1. 修改目标

v1 已证明课程链路可以运行，但存在三个相互关联的问题：价格卡检索文本主要是中文
档位和数字、Reranker 只看缺少品类上下文的 `summary`、旧 test 来自同一批人工模板。
v2 不改变七字段 CategoryCard Schema、三种卡片含义、课程动态 Hybrid 权重或模型，
只修正文本构造、标注口径和独立 test。

统一派生检索文本为：

```text
Category: {category}. Knowledge type: {natural-language type}. Summary: {English-normalized summary}.
```

它用于 BGE-M3 编码、BM25 的 `retrieval_text` 字段和 contextual Reranker。
`summary_only` 精排仍保留为课程字面基线；`raw_evidence` 不进入检索文本。

## 2. v2 数据契约

- 仍为 72 张卡、50 条 Query、每条 Query 全量判断 72 张卡且恰有 5 张正例。
- 原 30 train + 10 dev 保持 split；v1 的 10 条 test 全部排除，另建 10 条新 test。
- 每条 Query 都包含本品类价格卡，因为 CategoryInsight 输出需要价格档，但重要性按
  意图区分：strong 5 条、medium 8 条、coverage_weak 37 条。
- strong 的价格卡 gain 为 4～5，medium 为 2～3，coverage_weak 为 1；构建脚本硬校验。
- 三种卡片覆盖只作为指标，不做 Top-K 类型配额。

v2 case SHA-256：
`ed5467498b9fd261bf2d7207b265a5ed66404c32a9a9801a1b70b606510a0e1d`。

## 3. train/dev 配置选择

仅在 train/dev 同时比较课程字面的 `Query × summary` 与
`Query × retrieval_text`，新 test 此时未运行。Top-10 结果：

| 精排文本 | Train NDCG | Dev NDCG | Train AllTypes | Dev AllTypes |
| --- | ---: | ---: | ---: | ---: |
| summary-only | 0.7868 | 0.8189 | 0.5667 | 0.8000 |
| contextual | 0.8607 | 0.8348 | 0.8667 | 1.0000 |

因此冻结 contextual 为 v2 主精排；冻结证据写入
`data/category_insight/category_recall_v2_freeze.json`。动态 Hybrid 本身在 dev 的
NDCG@10 为 0.8493，仍略高于 contextual 的 0.8348，所以冻结结论只是“contextual
优于 summary-only”，并不提前宣称精排一定优于粗排。

## 4. 一次性独立 test

| 方案 | Recall@10 | MRR@10 | NDCG@10 | Bestseller | Attribute | Price | AllTypes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.8800 | 0.9143 | 0.7617 | 1.0000 | 1.0000 | 0.7000 | 0.7000 |
| BGE-M3 KNN | 0.9800 | 0.9000 | 0.8255 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 动态 Hybrid | 0.9800 | 0.9000 | 0.8259 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Hybrid + contextual BGE Reranker | 0.9400 | 1.0000 | 0.8655 | 1.0000 | 1.0000 | 0.9000 | 0.9000 |

contextual 精排把重要卡片和首个正例排得更靠前，NDCG 比 Hybrid 提高 0.0397，MRR
提高 0.10；但 Recall 降低 0.04，并在 1 条 Query 上把相关价格卡排出 Top-10，因而
Price/AllTypes Coverage 从 1.00 降到 0.90。这个结果支持保留精排用于重排序质量，
但不支持声称它全面优于未精排 Hybrid。

10 条 test 中精排实际执行 4 条，另 6 条因粗排最高分≥0.92 按课程旁路。规则
Query 类型分类器只命中 7/10；上表为隔离召回能力而使用冻结标注中的 Planner 类型，
因此实际工具在 Planner 未显式传类型时仍存在分类误差风险，不能把 0.8655 直接当作
端到端工具指标。

test 报告 SHA-256：
`7d3c313ff8c1004b27710265fdfcb2a8529a2d25dbeffbc0c7f3f95c485e4480`。

## 5. 实际配置与复现

| 模块 | 冻结配置 |
| --- | --- |
| 向量模型 | `BAAI/bge-m3`，主 `.venv` CPU，1024 维，max length 512 |
| 索引 | OpenSearch 2.19.1，Faiss HNSW cosine，M=32，efConstruction=200 |
| BM25 | `category^2 + retrieval_text`，standard analyzer |
| Hybrid 权重 | noun 0.5/0.5；attribute 0.7/0.3；style 0.9/0.1；colloquial 1.0/0.0 |
| 粗排/精排 | Top-30 → Top-10，旁路阈值 0.92 |
| 精排模型 | `BAAI/bge-reranker-v2-m3`，`blog_04` CUDA FP16 常驻子进程 |

```bat
.\.venv\Scripts\python.exe scripts\data\build_category_eval.py --annotations data\category_insight\eval_query_annotations_v2.jsonl --dataset-version category-card-recall-en-v2 --cases-filename category_recall_cases_v2.jsonl --manifest-filename category_recall_manifest_v2.json --forbid-test-reuse-from data\category_insight\category_recall_cases.jsonl
.\.venv\Scripts\python.exe scripts\index\build_category_kb.py --local-files-only --recreate
.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --local-files-only --cases-filename category_recall_cases_v2.jsonl --manifest-filename category_recall_manifest_v2.json --splits train dev --reranker-modes summary_only contextual --reranker-python C:\Anaconda\envs\blog_04\python.exe --output output\eval\category_recall_v2_selection.json
.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --local-files-only --cases-filename category_recall_cases_v2.jsonl --manifest-filename category_recall_manifest_v2.json --splits test --reranker-modes contextual --reranker-python C:\Anaconda\envs\blog_04\python.exe --output output\eval\category_recall_v2_test.json
```

详细逐 Query 排名保存在本地 `output/eval/category_recall_v2_selection.json` 与
`output/eval/category_recall_v2_test.json`；v1 数据和 `category_recall.json` 均保留。
