# catalog-schema-v2 正式 Faiss 索引（2026-08-20）

正式路径为 `output/index/catalog/<platform>/<locale>/`，四个分区均使用
`item-text-v5-catalog-schema-v2`。旧 `item-text-v2` 索引已在验收通过后删除；运行时
通过 manifest 与 position mapping 加载，旧文本版本仍会被拒绝。

| 分区 | 商品/向量数 | 维度 | 距离 | 数据 hash |
| --- | ---: | ---: | --- | --- |
| amazon:us | 5,638 | 1,024 | normalized inner product | manifest (`c741be32…efb5c6`) |
| amazon:es | 7,856 | 1,024 | normalized inner product | manifest (`1e607e5e…42716`) |
| amazon:jp | 8,323 | 1,024 | normalized inner product | manifest (`126430b5…ec3a2a`) |
| taobao:cn | 23,421 | 1,024 | normalized inner product | manifest (`aeca1872…fcb4e1`) |
| 合计 | 45,238 | — | — | — |

模型为本地 `D:/models/bge-m3`，CUDA 构建环境为 `C:\Anaconda\envs\blog_04`，GPU 为
NVIDIA GeForce RTX 3050 Ti Laptop GPU；item 编码使用 float16 推理、CLS pooling、L2
normalization，输出 float32 向量。每个 index 旁保存 `bge-m3-hnsw-ip.mapping.json`，
映射按 index position 对应 item_id；manifest 记录模型、维度、距离、text 版本、构建参数、
数据 hash、index hash 和 mapping hash。

表中 hash 为规范化 JSONL 的 `items_sha256`；原始 Taobao 分区 hash 为
`d06ea6f54e2316d420239178ee79a8e50af4276d61b5f5533894882b3b75569f`。

## 验收

- 四个分区向量、mapping、SQLite item 数一致；item_id 无重复/缺失。
- 全量向量 finite，维度 1024，L2 范数误差小于 `2e-7`。
- v5 SearchDocument 全量递归检查通过：只包含标题、品牌、类目、描述、typed 技术属性、
  材质和 options；价格、可售、库存、ID、配送、URL、时间和 provenance 不进入语义字段。
- 真实 Faiss ANN + SQLite 水合 + 硬约束 + BGE Reranker 冒烟成功，返回
  `embedding_rerank`；配送未知查询验证了 100→200→400→500 渐进补候选，ANN 故障验证了
  `bm25_fallback_after_ann_failure`。
- 当前仓库没有可用的冻结 ShopSimulator qrels 文件，不能据此声称新索引的 Recall@100 或
  Top-K 质量已全面证明；本次报告只记录可复现的结构和固定查询链路验收。

复现构建使用 `scripts/index/encode_catalog_items_cuda.py` 生成真实 BGE-M3 item vectors，
再由 `scripts/index/build_catalog_indexes.py --precomputed-root ...` 构建 HNSW/IP index。
商品主链保持纯 ANN → 水合 → 硬过滤/补候选 → Reranker；BM25 仅作为 ANN 初始化或运行失败
时的明确降级，不引入 Hybrid。

## Agent 查询编码运行时

正式 Agent 启动入口为 `src/globex_agent/presentation/server.py` 的 FastAPI lifespan，
通过 `composition.build_container()` 创建查询编码器。运行时使用 `BGE_M3_DEVICE=cuda:0`；
当主项目 Python 没有 CUDA-enabled torch 时，由 `BGE_M3_PYTHON` 指定的 CUDA Python
环境启动 `scripts/embedding/bge_m3_query_worker.py`。worker 常驻加载 BGE-M3，使用
512 token、FP16、CLS pooling、L2 normalization、1024 维输出，连续查询复用同一进程和模型。
启动时会校验四个 Faiss manifest 的模型、维度、text 版本、pooling、精度和 normalized 契约；
CUDA 不可用或契约不一致会直接失败，不会静默切 CPU。2026-08-20 本机实测使用
`C:\Anaconda\envs\blog_04\python.exe`、`cuda:0`、PID 34348，连续两批输出
`(2, 1024)`/`(1, 1024)`，有限且范数约 1；真实两次 ANN 查询均为 `embedding_only`，未走 BM25，
同一 worker 的 `model_reuse_count=2`。
