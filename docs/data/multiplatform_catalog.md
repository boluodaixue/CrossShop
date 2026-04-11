# Amazon + Taobao 多平台商品数据

## 已实现口径

当前数据层有两个逻辑平台库、四个检索分区：

```text
Amazon catalog.sqlite3
├── amazon:us（英文）
├── amazon:es（西班牙语）
└── amazon:jp（日语）

Taobao catalog.sqlite3
└── taobao:cn（中文）
```

SQLite 按平台分库，避免业务字段和平台规则互相污染；FTS5、BGE-M3 和评测按
`platform:locale` 分区，避免不同站点的候选池混在一起。

## 数据规模

本次实际构建结果：

| 数据 | 商品 | Query | qrel / 任务 |
|---|---:|---:|---:|
| Amazon US | 5,638 | 300 | ESCI 完整 Query 组的一部分 |
| Amazon ES | 7,856 | 300 | ESCI 完整 Query 组的一部分 |
| Amazon JP | 8,323 | 300 | ESCI 完整 Query 组的一部分 |
| Amazon 合计 | 21,817 | 900 | 22,064 条 ESCI 判断 |
| Taobao CN | 23,421 | 23,421 | 23,421 条任务和单目标 qrel |

Amazon 默认每个 locale 选择 240 个上游 train Query 作为 train，继续选择 30 个
上游 train Query 作为 dev，再选择 30 个上游 test Query 作为 test。选择顺序由
`SHA-256(locale:source_split:query_id)` 决定。同一 Query 的全部商品候选和 ESCI 标签
必须一起保留，不能按商品行随机切分。

Taobao 原始商品恰好一件对应一条 instruction。原始 `train` 标签映射到 train，原始
`eval` 标签映射到 test，实际为 21,962 / 1,459 条。

## 数据与任务分层

```text
data/raw/                              # 原始下载，只读、Git 忽略
data/processed/catalogs/              # 统一 StandardItem JSONL
data/processed/databases/             # 两个 SQLite 平台商品库
data/processed/eval/                   # Query 与 qrel
data/processed/tasks/shopsimulator/    # Agent 任务、目标规格和约束
data/processed/manifests/              # 数量、哈希、策略和许可提示
output/index/catalog/                  # 四个 BGE-M3/Faiss 分区
output/eval/                           # 可复现评测报告
```

商品、检索标注和 Agent 任务不能合成一张表：商品是可检索事实，qrel 是离线检索评测
标签，ShoppingTask 才包含目标规格和 Agentic RL 环境所需约束。

## 价格处理

ESCI 没有价格字段，统一记为 `price_source=unavailable`，不伪造汇率或商品价。

ShopSimulator 中 7,513 件商品只有一个明确观测价格，可以直接用于比价；15,896 件有
多个规格价格，另有 12 件没有有效价格。这 15,908 件在选择具体 SKU 前都记为价格
不可用，同时完整保留 `variants` 和价格区间，防止把最低价误当成最终成交价。

## Query 语言策略

冻结的 `BAAI/bge-m3` 直接编码用户原 Query，语义召回不翻译；多语言 reranker 也使用
原 Query。只有词法召回允许提供目标 locale 的本地化 Query，然后与原 Query 的语义
召回做融合。代码中的 `LocalizedFusionSearchBackend` 对这条边界有单元测试。

ESCI 原生评测只能证明本地语言 Query 到本地站点商品的检索效果。若要评测“中文 Query
到 Amazon US/ES/JP”，必须额外提供 `query_id` 对齐的中文 Query，并复用原 qrel；不能
把原生 ESCI 指标改名为跨语言指标。

## 重建命令

```powershell
.\.venv\Scripts\python.exe scripts\data\prepare_shopsimulator_catalog.py
.\.venv\Scripts\python.exe scripts\data\prepare_esci_multilingual_catalog.py --download
.\.venv\Scripts\python.exe scripts\data\build_catalog_databases.py
C:\Anaconda\envs\blog_04\python.exe scripts\index\encode_catalog_items_cuda.py --local-files-only --batch-size 4
.\.venv\Scripts\python.exe scripts\index\build_catalog_indexes.py `
  --local-files-only --precomputed-root output\embeddings\catalog
```

CUDA worker 只使用既有环境里的 PyTorch、Transformers、NumPy 和官方本地模型缓存，
按 SentenceTransformers 配置执行 CLS pooling、L2 normalize；主项目仍负责校验文档 ID、
构建 Faiss 和写 manifest。没有 CUDA 时可省略预编码步骤，直接运行
`build_catalog_indexes.py --local-files-only`，但 CPU 全量编码明显更慢。

单分区词法评测示例：

```powershell
.\.venv\Scripts\python.exe scripts\eval\run_catalog_retrieval_eval.py `
  --platform amazon --locale us --backend fts --split test --top-k 10
```

`run_catalog_retrieval_eval.py` 也支持 `semantic` 和 `hybrid`。跨语言实验可通过
`--query-overrides` 提供单独构造的 Query；`--lexical-query-overrides` 仅作用于 hybrid
的 BM25/FTS 分支。

## 当前检索基线

| 分区 | 后端 | Test Query | Recall@10 | MRR@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| Amazon US | FTS | 30 | 0.594151 | 0.668704 | 0.606670 |
| Amazon US | BGE-M3 | 30 | 0.560776 | 0.679762 | 0.586919 |
| Amazon US | Hybrid 0.7/0.3 | 30 | 0.602866 | 0.690873 | 0.615287 |
| Amazon US | Reranker | 30 | 0.678137 | 0.798466 | 0.686447 |
| Amazon ES | FTS | 30 | 0.557551 | 0.651839 | 0.568015 |
| Amazon ES | BGE-M3 | 30 | 0.501177 | 0.726481 | 0.566690 |
| Amazon ES | Hybrid 0.7/0.3 | 30 | 0.542097 | 0.757870 | 0.616676 |
| Amazon ES | Reranker | 30 | 0.570283 | 0.801481 | 0.667909 |
| Amazon JP | FTS | 30 | 0.459341 | 0.700317 | 0.541401 |
| Amazon JP | BGE-M3 | 30 | 0.496820 | 0.784444 | 0.605482 |
| Amazon JP | Hybrid 0.7/0.3 | 30 | 0.535386 | 0.827778 | 0.651538 |
| Amazon JP | Reranker | 30 | 0.580533 | 0.862222 | 0.718822 |

原始 1,459 条 ShopSimulator eval 是单目标 Agent 任务，不能与上表的多商品检索指标
同名。其目标商品可达性诊断为：

| 分区 | 后端 | Test 任务 | GoldTargetHit@10 | GoldTargetMRR@10 | GoldTargetNDCG@10 |
|---|---|---:|---:|---:|---:|
| Taobao CN | FTS | 1,459 | 0.549006 | 0.346708 | 0.394871 |
| Taobao CN | BGE-M3 | 1,459 | 0.805346 | 0.531329 | 0.597463 |
| Taobao CN | Hybrid 0.7/0.3 | 1,459 | 0.851268 | 0.597320 | 0.658799 |

淘宝多商品检索集 shopsimulator_retrieval_v1（100 条 test，正例=Exact+Substitute，
多系统 pool + 模型标注 + 门禁冻结）的实际结果，这才是普通 Recall/MRR/NDCG 的应用口径：

| 后端 | Test Query | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| FTS（BM25） | 100 | 0.210708 | 0.299817 | 0.290661 |
| BGE-M3 | 100 | 0.518752 | 0.596440 | 0.676776 |
| Hybrid 0.7/0.3 | 100 | 0.542868 | 0.646369 | 0.694781 |
| Reranker | 100 | 0.554712 | 0.726095 | 0.634743 |

这里的 Hybrid 权重是固定基线，不声称已经调优。US/ES 的 BGE-M3 单路 Recall 没有超过
FTS，不能只报告有利分区；JP 的多语言语义召回有明显价值，三个 Amazon 分区的
Hybrid 结果最好或接近最好。Taobao 表只说明指定目标能否进入 Top10，不证明多个合理
商品的召回覆盖率。

ESCI 的 qrel 来自完整保留的选中 Query 组；在合并后的 locale 商品库中，其他 Query 带入
的商品对当前 Query 没有判断，因此报告同时给出 judged coverage。原 ShopSimulator
每条 Query 只有一个目标商品，judged coverage 天然很低，它不等于检索错误率；普通
Recall/MRR/NDCG 使用另建的多商品检索集。

Reranker 为 BGE-M3 ANN Top-100 → BGE-reranker-v2-m3 Top-10，是 Amazon 三个 locale 的
Recall/MRR/NDCG 全最优，也是淘宝集 Recall/MRR 最优。注意 Amazon 与淘宝不可直接横向比：
Amazon 30 query/语言、正例仅 Exact、ESCI 封闭池 judged coverage 约 0.70–0.86；淘宝
100 query、正例 Exact+Substitute，FTS/BGE/Hybrid 覆盖率 1.0，而 Reranker 覆盖率 0.702
（top-100 重排带出约 30% 未标注项，按 unjudged=非相关保守计，NDCG 被低估）。

## 当前本地存储

不含模型和向量索引时，本次实际数据约 1.82 GB：Amazon 原始 parquet 约 1.16 GB，
ShopSimulator 原始压缩包约 19.9 MB，规范化文本约 142 MB，两个 SQLite 库合计约
500 MB。预计算向量约 189 MB，四个 Faiss 索引约 198 MB；加上这些检索产物仍只有
约 2.21 GB（不含模型缓存）。所有大型数据与运行产物均已被 `.gitignore` 排除。

ShopSimulator 中文普通检索集的 Query 选择、Top10 judgment pool、多级相关性与抽审
口径见 `docs/data/shopsimulator_retrieval_v1.md`。原始单目标任务和新多商品 qrels 必须
继续分开报告。
