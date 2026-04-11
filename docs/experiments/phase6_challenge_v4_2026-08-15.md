# 商品召回与知识卡 RAG v4 Challenge 实验记录

日期：2026-08-15  
状态：数据已冻结，知识卡与商品首次评测均已真实运行并保存

## 1. 为什么需要 v4

v3 的高分主要来自构造偏置，而不是系统已经接近满分：知识卡每品类只有 10 张且评测取
Top-10，每张卡还复制了全部商品形态；商品则是 120 个候选取 Top-100，Query 直接复用
合成商品属性。v4 保留 v3 作为 smoke set，另建 challenge set，不覆盖任何历史结果。

v4 的关键变化：

- 商品事实从 1,356 扩到 2,356，每品类至少 110；
- 知识卡从每品类 10 张扩到 15 张，总计 300 张；
- 知识卡 retrieval text 不再给每张卡复制全品类商品形态；
- 商品候选池从 120 扩到 500，包含 60～95 个同品类负例和近邻品类负例；
- 加入独立改写、场景、口语和否定约束 Query；
- 所有候选仍全部标注，每条 Query 仍恰有 5 个强正例；
- 配置和文件哈希在首次评测前写入 `challenge_freeze_v4.json`。

## 2. 知识卡 RAG 首次冻结结果

候选为全局 300 张卡，粗排 Top-30，最终 Top-10。动态 Hybrid 使用课程的 Query 类型
权重；评测检索能力时使用标注的 Planner 类型，规则分类器只作为单独诊断。

| 方案 | Recall@10 | MRR@10 | NDCG@10 | AllTypes@10 | P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.4713 | 0.6371 | 0.4528 | 0.0625 | 8.56 ms |
| BGE-M3 KNN | 0.6400 | 0.6903 | 0.5898 | 0.2406 | 153.50 ms |
| 动态 Hybrid | 0.6500 | 0.7138 | 0.6036 | 0.2313 | 155.64 ms |
| Hybrid + BGE Reranker | 0.6106 | 0.6745 | 0.5523 | 0.2062 | 177.53 ms |

动态 Hybrid 的语言切片：EN Recall@10=0.6250、ZH=0.6750；EN NDCG@10=0.5778、
ZH=0.6294。按 Query 类型：

| 类型 | BM25 Recall@10 | KNN Recall@10 | Hybrid Recall@10 | Reranker Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| noun | 0.6125 | 0.8450 | 0.8550 | 0.8250 |
| attribute_constraint | 0.7450 | 0.7600 | 0.7825 | 0.7625 |
| style | 0.2675 | 0.5025 | 0.5100 | 0.5000 |
| colloquial | 0.2600 | 0.4525 | 0.4525 | 0.3550 |

结论：BM25 不再接近满分；BGE-M3 在风格和口语型 Query 上产生了明确增益，动态融合整体
最好。Reranker 仍低于未精排 Hybrid，主要下降发生在中文和口语 Query：Reranker EN
Recall=0.6738、ZH=0.5475，colloquial Recall 从 0.4525 降到 0.3550。当前应把 Reranker
保留为真实实验步骤，但不能默认宣称增益。

规则 fallback 分类器仅 90/320=28.1% 正确；这不影响上述“分类正确假设下”的检索结果，
但说明端到端 Planner 分类能力仍未完成。contextual 门控执行精排 146 条，旁路 174 条。

## 3. 商品召回首次冻结结果

主结果严格采用课程口径：纯 Query/Item ANN Top-100 → Reranker Top-10。BM25 只作为
基线，没有把 BM25/Hybrid 候选送入主 Reranker。

| 方案 | Recall@10 | MRR@10 | NDCG@10 | Candidate Recall@100 | P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 基线 | 0.7137 | 0.6346 | 0.6274 | 1.0000 | 1.47 ms |
| BGE-M3 ANN | 0.3150 | 0.3469 | 0.2716 | 0.9938 | 135.20 ms |
| ANN + BGE Reranker | 0.5475 | 0.5459 | 0.5013 | 0.9938 | 2487.87 ms |

语言切片：

| 方案 | EN Recall@10 | ZH Recall@10 | EN NDCG@10 | ZH NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.6475 | 0.7800 | 0.5635 | 0.6913 |
| BGE-M3 ANN | 0.3000 | 0.3300 | 0.2543 | 0.2890 |
| ANN + BGE Reranker | 0.7550 | 0.3400 | 0.6948 | 0.3077 |

ANN Recall@100=0.9938，表示粗排候选几乎包含全部 Exact；ANN Top-10 低只是正例在前
100 内排序不够靠前。Reranker 把 Recall@10 从 0.3150 提到 0.5475，证明精排有真实
作用，但中文几乎没有增益，整体仍低于 BM25。

## 4. 为什么 v4 的 BM25 仍有 0.7137

只读偏差审计复现了相同 BM25 总分，并给出 Query 类型切片：

| 商品 Query 类型 | BM25 Recall@10 |
| --- | ---: |
| lexical_constraints | 0.8300 |
| scenario_paraphrase | 0.8250 |
| colloquial_scenario | 0.6350 |
| negative_constraint | 0.5650 |

因此 BM25 高并不意味着粗排/精排无用，而是 v4 仍保留了可解释的结构化词法信号：

- lexical 和 scenario_paraphrase Query 仍明确写出目标属性值；
- 中文商品和 Query 来自同一套机器生成词表，中文 bigram 重叠尤其高；
- 本地 BM25 对标题词给予 3 倍权重；
- 标签仍从同一份品类属性规格生成，并非独立人工购物判断。

否定约束的 BM25 Recall 仅 0.5650，说明词法匹配会同时命中“不要”的属性；口语场景也
明显低于显式属性 Query。v4 已经比 v3 更有诊断价值，但仍是合成封闭集，不能替代真实
用户 Query、人工相关性判断和跨域 Item 文本。

## 5. 设备与真实性证据

- BGE-M3 在项目主 `.venv` 使用 CPU 编码；商品 2,356 Item 的首次离线建库约 16 分钟。
- 商品索引为 Faiss HNSW + Inner Product，M=32、efConstruction=200、efSearch=128。
- BGE-Reranker-v2-m3 由 `C:\Anaconda\envs\blog_04\python.exe` 常驻子进程运行。
- 商品精排实测使用 `cuda:0 + FP16 + batch size 8`；RTX 3050 峰值观察为约 98% GPU、
  3,643/4,096 MiB 显存、73°C，没有 OOM。
- 本次延迟是笔记本离线实验数据，不代表商业在线延迟。

## 6. 复现命令与证据哈希

```powershell
.\.venv\Scripts\python.exe scripts\index\build_category_kb.py --cards data\category_insight\category_cards_v4.jsonl --retrieval-texts data\category_insight\category_retrieval_texts_v4.jsonl --index-name globex_category_kb_v4 --analyzer ik_max_word --search-analyzer ik_smart --embedding-model BAAI/bge-m3 --device cpu --batch-size 8 --max-seq-length 512 --local-files-only --recreate --manifest output\category_kb\index_manifest_v4.json

.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --data-dir data\category_insight --index-name globex_category_kb_v4 --cards-filename category_cards_v4.jsonl --cases-filename category_recall_challenge_v4.jsonl --manifest-filename category_recall_challenge_manifest_v4.json --embedding-model BAAI/bge-m3 --device cpu --max-seq-length 512 --reranker-model BAAI/bge-reranker-v2-m3 --reranker-python C:\Anaconda\envs\blog_04\python.exe --reranker-device cuda:0 --reranker-batch-size 8 --reranker-max-length 512 --local-files-only --coarse-k 30 --top-k 10 --splits final_test --reranker-modes contextual --output output\eval\category_recall_challenge_v4.json

.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --data-dir data\eval\product_v4 --embedding-model BAAI/bge-m3 --reranker-model BAAI/bge-reranker-v2-m3 --device cpu --reranker-python C:\Anaconda\envs\blog_04\python.exe --reranker-device cuda:0 --reranker-batch-size 8 --top-k 10 --candidate-k 100 --max-seq-length 512 --reranker-max-length 512 --local-files-only --rebuild-index --index-backend faiss-hnsw --hnsw-m 32 --ef-construction 200 --ef-search 128 --index output\index\product-v4-bge-m3-hnsw-ip.faiss --output output\eval\product_recall_challenge_v4.json

.\.venv\Scripts\python.exe scripts\eval\audit_challenge_v4.py
```

| 证据文件 | SHA-256 |
| --- | --- |
| `category_recall_challenge_v4.json` | `d21f8d046dc0243c8b057a7cfcf6533fb7c1a740a1fc00305733a4ea0910e244` |
| `product_recall_challenge_v4.json` | `fc770d5b91ab713c35b4e944ca8b1bd7ec8fbe18b4545395820020fcbf01bdd2` |
| `index_manifest_v4.json` | `43278708434c9c325cea9b5d6a16a46c9e009d99a500f8c13a7f4e7a700efce4` |
| `product-v4-bge-m3-hnsw-ip.manifest.json` | `65cda9e53dcd383310618bf2c8e645fe3a4b3a9317085349ba1d8dbb64d999c7` |
| `challenge_freeze_v4.json` | `b1deadb74fecbc0a5559f741c42dfe6730700c62af0c42989d96885729ebc519` |

报告包含本机延迟，重跑会改变报告文件哈希；稳定的数据输入哈希以各 manifest 和 freeze
文件为准。
