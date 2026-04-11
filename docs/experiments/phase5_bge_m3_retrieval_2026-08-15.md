# 阶段 5：BGE-M3 商品召回实验记录

- 日期：2026-08-15
- 状态：已真实运行并通过回归
- 主链：`BGE-M3 Query/Item → Faiss HNSW/IP Top-100 → BGE-Reranker-v2-m3 Top-10`
- 边界：无 User 塔；BM25 只作基线；Hybrid 不进入主链；本轮只做推理评测，不做训练或微调

## 实验环境

| 组件 | 环境与配置 |
| --- | --- |
| Query/Item 编码 | 项目 `.venv`，Python 3.10.20，PyTorch 2.13.0 CPU，`BAAI/bge-m3`，FP32 |
| ANN 索引 | `faiss-cpu` 1.15.0，HNSW + Inner Product，L2 归一化 |
| Reranker | Conda `blog_04`，Python 3.9.16，PyTorch 2.0.1+cu118，RTX 3050 Ti Laptop GPU |
| 精排精度 | 加载官方原始 `BAAI/bge-reranker-v2-m3` 权重，运行时 FP16；未使用第三方 FP16 重打包 |
| 精排参数 | `batch_size=1`，`max_length=256`，常驻 JSONL 子进程 |

## 数据与评测口径

- 数据版本：`amazon-esci-task1-en-us-closed-recall-v2`。
- 35 个英文 Query、720 个去重 Item，`train/dev/test=21/6/8`。
- 每个 Query 的候选池全部有 ESCI 标注，未标注候选禁止进入结果，judged coverage 为 1.0。
- Recall/MRR 只把 `Exact` 当正例，每个 Query 有 2～5 个 Exact 正例，平均 4.6 个。
- NDCG gain：`Exact=1.0`、`Complement=0.1`、`Substitute=0.01`、`Irrelevant=0.0`。
- 候选池为 8～51 个 Item，平均 24.5429。代码请求 Top-100，但不能表述为每个 Query 实际精排了 100 个候选。
- test split 是准确率主结果；35-query overall 只作诊断，不与 test 指标混报。
- 数据哈希：
  - cases：`b4435bb77358e432fb8eebdd5f43de2593948ac333a1756e3c4cced63781041b`
  - items：`f3e745c310a08ea880a705bd34dbb1707c7da49ebcaba10b7cb730f42d66fd19`

## 索引配置

- 模型：`BAAI/bge-m3`。
- Item 文本格式：`item-text-v2`，删除重复 title 和字面量 `None`，title 优先。
- 推理窗口：512 token。
- 720 个 Item，1024 维归一化向量。
- Faiss：`IndexHNSWFlat + METRIC_INNER_PRODUCT`。
- 参数：`M=32`、`efConstruction=200`、`efSearch=128`。
- 本地索引大小：3,144,354 bytes。
- 索引 SHA-256：`fd746de3d7677e598b61bb7361f0afacebdf1565f2e937e1fee29486071d5454`。

## 独立 test split 主结果

| 方案 | Recall@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| BM25 基线 | 0.633333 | 0.713889 | 0.564619 |
| BGE-M3 + Faiss HNSW/IP | 0.758333 | 0.713542 | 0.644830 |
| ANN Top-100 → BGE Reranker Top-10 | **0.875000** | **0.875000** | **0.788628** |

精排相对 ANN 粗排：Recall@10 提升 0.116667，NDCG@10 提升 0.143798。

## 35-query overall 诊断

| 方案 | Recall@10 | MRR@10 | NDCG@10 | Recall@100 | P50 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 基线 | 0.531429 | 0.650079 | 0.484503 | 0.813810 | 0.1024 ms | 0.2350 ms |
| BGE-M3 + Faiss HNSW/IP | 0.613810 | 0.638639 | 0.534425 | 1.000000 | 86.6778 ms | 113.6263 ms |
| ANN → BGE Reranker | 0.633810 | 0.724875 | 0.592388 | 1.000000 | 528.3160 ms | 1041.0986 ms |

Reranker 报告中的 3 个 fallback 是候选池本身不超过 Top-10，因此直接保留全部粗排候选；不是模型、CUDA 或子进程失败。

## 实际运行命令

```bat
.\.venv\Scripts\python.exe scripts\index\build_item_index.py --batch-size 4
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe examples\04_semantic_retrieval.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts examples tests
```

验证结果：pytest `68 passed`；Ruff `All checks passed`；GPU 精排 smoke test 和 ItemSearch 注入示例均成功。

## 结论与限制

1. BGE-M3 明显优于当前 BM25 词法基线，BGE Reranker 又显著改善 Top-10 的召回与排序质量，课程商品主链已达到可运行、可复现状态。
2. 当前数据只有英文，不能据此声称中文或中英跨语言效果。
3. 封闭评测池保证没有未标注结果，但候选池最大只有 51，Recall@100 不是全量商品库指标。
4. 当前延迟是笔记本离线评测数据，不是商业服务 SLA；本阶段目标是跑通和验证效果。
5. 本轮没有进行 BGE-M3 监督训练、Hard Negative Mining 或 Reranker 微调，不能把结果描述成领域训练效果。

## 本地产物

- 原始报告：`output/eval/retrieval_comparison.json`（运行产物，Git 忽略）。
- Faiss 索引：`output/index/esci-bge-m3-hnsw-ip.faiss`（运行产物，Git 忽略）。
- 可提交实验快照：本文件。
