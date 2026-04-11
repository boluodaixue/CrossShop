# 阶段 6：CategoryInsight 商品知识卡 RAG 实验记录

日期：2026-08-15  
状态：本地课程链路已实际运行；仅用于离线学习和评测

## 1. 本阶段口径

本阶段实现的是商品知识卡 RAG，不是商品召回。商品召回仍保持阶段五的
`Query/Item ANN Top-100 → Reranker Top-10`；知识卡链路独立使用课程第 13、13-1
章的 `OpenSearch BM25 + KNN Hybrid Top-30 → BGE Reranker → quick Top-8 / deep
Top-15`。

三类卡片的语义如下：

- `bestseller`：该品类中具有代表性的商品形态及商品级证据；当前只能解释为离线
  生成的流行度，不是真实销量榜。
- `attribute`：从品类商品文本中确定性抽取的关键属性分布。
- `price_range`：该品类的便宜款、中档、高端三个有限区间；当前价格为离线增强，
  不是挂牌价或成交价。

## 2. 数据与真值边界

知识卡和阶段五共用 Amazon Shopping Queries ESCI 英文子集。人工维护 Query→标准
品类映射，再通过商品标题词门禁剔除混类商品；Query group `52717` 因不能形成可信
单一品类而整体排除。

| 项目 | 实际数量 |
| --- | ---: |
| 原商品候选 | 720 |
| 标准品类 | 12 |
| 品类—商品事实 | 370 |
| 唯一事实商品 | 369 |
| 被门禁拒绝的品类成员关系 | 37 |
| `bestseller` 卡 | 24 |
| `attribute` 卡 | 36 |
| `price_range` 卡 | 12 |
| 合计卡片 | 72 |
| 入库拒绝卡片 | 0 |
| 10% 抽审队列 | 8 |

观测字段包括商品 ID、title、body 和 ESCI relevance label。`typical_price_cny` 与
`order_count_90d` 因 ESCI 不提供而使用固定种子 `20260815` 确定性生成，均在事实表
和 provenance sidecar 中标记为 `generated_offline_not_observed`。知识卡本体严格保持
课程七字段，不在 Schema 中偷偷加入来源字段。

卡片文件 SHA-256：
`1b14a2ce6d8d929ae48aceadee65a2fc30ebd990f589b31dec30072481b27d20`。

8 张稳定哈希样本已逐张检查品类语义、摘要—证据一致性、来源追溯和生成字段声明。
首次抽审发现并修复 `No Drain Holes` 的否定短语误判，并排除 7 条标签耗材、
go-kart 配件、通用车库座椅和自行车专用工具；修正后 8/8 通过，逐卡结论保存在
`data/category_insight/audit_results.jsonl`。

完整数据契约见 `docs/data/category_card_data_contract.md`。

## 3. 评测契约

评测集包含 50 条人工编写的英文知识需求，冻结为 train/dev/test=30/10/10；名词型、
属性约束型、风格型和完全口语型在每个 split 都有覆盖。每个 Query 都对全局 72 张
知识卡显式判断，固定 5 张正例，按重要性赋 gain 5、4、3、2、1。未标注候选在所有
split（包括 test）中均被禁止，不复用 ESCI 原 Query 文本。

评测 case SHA-256：
`ab55a0c980de05f38ab24f2ca9436f2d1c2c45709c5aeae8aa884e8d882f8c06`。

## 4. 实现配置

| 模块 | 实际配置 |
| --- | --- |
| OpenSearch | 官方镜像 2.19.1，本地 Docker 单节点 |
| 文本分析器 | `standard`；当前语料是英文，课程的 `ik_max_word` 是中文方案 |
| 向量模型 | 官方原始 `BAAI/bge-m3`，主环境 CPU，1024 维，max length 512 |
| 向量索引 | OpenSearch `knn_vector`，Faiss HNSW，cosine，M=32，efConstruction=200 |
| 入向量文本 | `category + card_type + summary` |
| 粗召回 | 动态 Hybrid Top-30 |
| 精排模型 | 官方原始 `BAAI/bge-reranker-v2-m3`，`blog_04` CUDA FP16 常驻进程 |
| 精排输入 | `Query × CategoryCard.summary`，与课程一致 |
| 精排输出 | quick Top-8；deep Top-15 |
| 精排旁路 | 粗排最高分≥0.92，或候选数≤最终 K |

动态权重严格按课程表选择：名词型 0.5/0.5、属性约束型 0.7/0.3、风格型
0.9/0.1、完全口语型 1.0/0.0（前者为 KNN，后者为 BM25）。完全口语型请求不构造
BM25 分支。当前 `CategoryInsight` 对外契约按课程只暴露 `category/depth`，因此实际
工具运行使用内部规则分类器；其 50-query 诊断准确率为 0.96，test 为 1.0。离线召回
指标使用冻结标注中的 Query 类型，避免把分类错误和召回错误混为一个指标；报告中
同时单列规则分类准确率。

## 5. 独立 test 结果

指标以每个 Query 的 5 张显式正例计算；延迟是模型加载和预热完成后的 50-query
overall 诊断，不代表商业线上 SLA。

| 方案 | Test Recall@10 | Test MRR@10 | Test NDCG@10 | Overall P50 | Overall P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.6000 | 0.7500 | 0.5685 | 13.01 ms | 33.26 ms |
| BGE-M3 KNN | 0.9600 | 1.0000 | 0.8802 | 121.15 ms | 157.13 ms |
| 课程动态 Hybrid | 0.9600 | 1.0000 | 0.8725 | 119.22 ms | 140.97 ms |
| Hybrid + BGE Reranker Top-10 | 0.8000 | 0.9000 | 0.7721 | 135.78 ms | 588.77 ms |
| quick Top-8 | 0.7800 | 0.9000 | 0.7693 | 135.78 ms | 588.77 ms |

课程对 Top-10 的门槛为 Recall≥0.75、MRR≥0.65、NDCG≥0.70；当前精排 test
结果三项均通过。50 条 Query 中精排实际执行 22 次，因最高粗排分≥0.92 旁路 28 次。

需要如实保留的 bad case：精排虽然通过门槛，却把 test NDCG 从 Hybrid 的 0.8725
降到 0.7721。原因不是候选漏召，主要是课程规定只给 cross-encoder 看 `summary`；
当前自动生成的 “Material / Color / Installation” 等属性摘要缺少品类名，不同品类的
摘要非常相似，通用 Reranker 会把跨品类通用属性抬高。这里没有根据 test 标签修改
卡片或规则。后续只可在 train/dev 上比较“摘要中携带品类”或领域 Reranker，再一次性
报告新 test。

## 6. CategoryInsight 输出与降级

`CategoryInsightService` 已实现：

- 品类别名归一；普通品类的 `components=[]`，避免把普通商品误拆成套装组件。
- quick 输出 bestseller 与价格层；deep 额外输出属性分布。
- 只从最终解析出的同品类卡片提炼，防止跨品类候选污染结构化答案。
- 向量塔异常时降级 BM25；OpenSearch 同时失败时返回空结构和 confidence=0；
  Reranker 异常时保留粗排顺序。
- 当前项目还没有 WebSearch 工具，因此课程的低置信度 WebSearch 补充只保留为后续
  集成项，不伪装为已实现。

本机 `bathroom fan` 的 quick/deep 示例已实际运行：归一到 `home ventilation fans`，
普通品类 `components=[]`，得到 5 个 bestseller、3 个价格层；deep 额外得到 3 个属性
分布，输出 confidence=0.68。

## 7. 复现命令

```bat
.\.venv\Scripts\python.exe scripts\data\build_category_cards.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval.py
docker compose -f infra\opensearch\docker-compose.yml up -d
.\.venv\Scripts\python.exe scripts\index\build_category_kb.py --local-files-only --recreate
.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe examples\05_category_insight.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
```

详细逐 Query 排名和 bad cases 在本地 `output/eval/category_recall.json`；该运行产物不
提交仓库。OpenSearch 容器和索引默认保留，便于离线复跑。
