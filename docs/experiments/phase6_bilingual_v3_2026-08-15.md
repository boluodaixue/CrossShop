# 双语商品召回与知识卡 v3 实验记录

日期：2026-08-15  
状态：数据、索引、封闭评测和独立 holdout 已实际运行

## 1. 实现结果

- 数据扩为 20 品类、1356 条品类商品事实、200 张知识卡；v1/v2 原文件保留。
- 370 条事实保留 `real_esci`，986 条标为 `synthetic_llm_template`；中文均明确为
  machine-generated/unreviewed。
- 知识卡 Query 按 `intent_id` 成对，中英文共享全部标签；最终 holdout 为 160 个
  intent、320 行 Query、每条全量判断 200 张卡。
- OpenSearch 2.19.1 安装 analysis-ik 2.19.1；使用 `ik_max_word` 建索引、
  `ik_smart` 查询。旧 `globex_category_kb` 72 文档保留，新
  `globex_category_kb_v3` 为 200 文档。
- BGE-M3 在主 `.venv` CPU 编码；BGE-Reranker-v2-m3 在 `blog_04` CUDA FP16
  常驻进程运行。知识卡索引为 OpenSearch Faiss HNSW cosine；商品索引为本地
  Faiss HNSW Inner Product，M=32、efConstruction=200、efSearch=128。

## 2. 知识卡最终 holdout

以下是实现冻结后第一次运行的新 holdout，不包含此前用于诊断的 Query 文本：

| 方案 | Recall@10 | MRR@10 | NDCG@10 | AllTypes@10 | P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.9856 | 0.9922 | 0.8722 | 0.9594 | 10.44 ms |
| BGE-M3 KNN | 0.9688 | 0.9786 | 0.8360 | 0.9719 | 172.90 ms |
| 动态 Hybrid | 0.9750 | 0.9885 | 0.8643 | 0.9781 | 176.11 ms |
| Hybrid + BGE Reranker | 0.9481 | 0.9651 | 0.8230 | 0.9250 | 190.75 ms |

语言切片：

| 方案 | EN Recall@10 | ZH Recall@10 | EN NDCG@10 | ZH NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.9775 | 0.9938 | 0.8691 | 0.8754 |
| BGE-M3 KNN | 0.9650 | 0.9725 | 0.8422 | 0.8297 |
| 动态 Hybrid | 0.9675 | 0.9825 | 0.8612 | 0.8673 |
| Hybrid + BGE Reranker | 0.9850 | 0.9113 | 0.8730 | 0.7729 |

粗排的中英文差很小；Reranker 的中文下降是当前主要缺口。动态 Hybrid 使用标注中的
Planner 类型来隔离召回能力；规则 fallback 分类器仅 113/320=35.3%，不能作为已完成
的端到端分类能力。精排共执行 110 条，210 条因粗排最高分 ≥0.92 旁路。

Reranker 低于未精排 Hybrid 不是候选丢失：它在中文和口语预算型 Query 上把跨品类
价格卡提到前面。当前正确结论是“保留为实验步骤，但不默认宣称增益”，而不是通过
强制插卡、改标签或查看 holdout 后调阈值来制造提升。

## 3. 商品召回 synthetic bilingual 诊断

真实 ESCI 35 Query/720 Item 报告继续是公开真实数据主报告。本节只回答扩品类和中文
是否能跑通，不与 ESCI 指标合并。

数据为 1356 个 Item、80 个 intent/160 条 Query；每条有 120 个完整标注候选和 5 个
Exact。课程主链没有接入 BM25 或 Hybrid 候选：ANN Top-100 → Reranker Top-10。

| 方案 | Recall@10 | MRR@10 | NDCG@10 | Candidate Recall@100 | P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 基线 | 0.8900 | 0.9824 | 0.8828 | 1.0000 | 0.54 ms |
| BGE-M3 ANN | 0.5700 | 0.7968 | 0.5688 | 1.0000 | 113.58 ms |
| ANN + BGE Reranker | 0.8000 | 0.8883 | 0.7915 | 1.0000 | 1767.57 ms |

| 方案 | EN Recall@10 | ZH Recall@10 | EN NDCG@10 | ZH NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.9000 | 0.8800 | 0.8866 | 0.8791 |
| BGE-M3 ANN | 0.5600 | 0.5800 | 0.5492 | 0.5885 |
| ANN + BGE Reranker | 0.9275 | 0.6725 | 0.9262 | 0.6568 |

ANN Top-100 的候选 Recall=1.0，说明课程主链候选没有丢 Exact。Reranker 对英文增益很
大，对中文增益有限。BM25 得分高是因为合成 Query 和商品共享明确属性词，属于数据
构造偏置，不能据此否定语义召回，也不能与真实 ESCI 横向合并。

## 4. 复现命令与证据

```bat
.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --local-files-only --index-name globex_category_kb_v3 --cards-filename category_cards_v3.jsonl --cases-filename category_recall_final_test_v3.jsonl --manifest-filename category_recall_final_test_manifest_v3.json --splits final_test --reranker-modes contextual --reranker-python C:\Anaconda\envs\blog_04\python.exe --reranker-batch-size 2 --output output\eval\category_recall_v3_final_holdout.json
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --data-dir data\eval\product_v3 --local-files-only --index output\index\product-v3-bge-m3-hnsw-ip.faiss --reranker-python C:\Anaconda\envs\blog_04\python.exe --reranker-batch-size 2 --output output\eval\product_recall_synthetic_bilingual_v3.json
```

| 证据文件 | SHA-256 |
| --- | --- |
| `category_recall_v3_final_holdout.json` | `e25e6ab2559735033ddb1439544e4eb24dd20f1aac62242dbff1eb6a467ac5a1` |
| `product_recall_synthetic_bilingual_v3.json` | `e8d6adfbfcbe0cdc3ad2f0b66ce37088ea0ebaf27f9444cf40f30a4cb8d9cf38` |
| `index_manifest_v3.json` | `5f5b21d5464e32e6a7322e352798a77967da6258b02309e9fdef679cffe3da26` |
| `product-v3-bge-m3-hnsw-ip.manifest.json` | `1a6e59d00c41fbc61cadbec3c844c9e17ecd174931ed8e612cb24f916cbb051e` |

报告含本机离线延迟，因此重新运行会改变报告文件哈希，即便排序指标相同。稳定数据
哈希以各自 `data/**/manifest*.json` 为准。
