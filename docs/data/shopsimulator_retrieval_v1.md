# ShopSimulator 中文普通商品检索集 v1

## 定位

这套数据与 ShopSimulator 原始 Agent 任务并存，但用途不同：

| 数据 | 判断对象 | 指标 |
|---|---|---|
| 原始 1,459 条 eval 任务 | 是否找到并购买指定目标商品/SKU | GoldTargetHit、SKU 准确率、购买成功率 |
| `shopsimulator-retrieval-zh-v1` | 一个中文 Query 下的多个相关商品 | Recall、MRR、NDCG |

普通检索集不把原目标商品自动当唯一正例，也不把其他商品自动当负例。

## Query 选择

- 只从原始 `eval` 切分选择，避免与后续 Agent 训练任务混用。
- 优先使用上游 `instruction_simple` 对应的 `simple_query`；缺失时才清洗并回退完整
  `instruction`。
- 要求目标商品所在类目至少存在两件共享两个源属性的同类商品，以降低“天然只有一个
  答案”的概率。
- 稳定哈希先保证类目覆盖，再补充高属性支持 Query；候选阶段为 30 dev + 150 test。
- 完成相关性门禁后冻结 15 dev + 100 test，其余有效 Query 作为扩容备用。

`source_target_item_id` 只用于数据血缘和确保候选池没有遗漏原任务商品。它不会出现在
标注 prompt 中，也不是正例门禁条件。

## 候选池

每条 Query 的 judgment pool 是以下结果的并集：

- SQLite FTS5 检索深度 100，纳入 Top10；
- BGE-M3/Faiss 检索深度 100，纳入 Top10；
- 对上述两路各自 min-max 后按 0.7/0.3 融合，纳入 Hybrid Top10；
- 同叶子类目下的属性近邻 Top15；
- 同叶子类目稳定哈希样本 3 件；
- 原任务目标商品 1 件，仅保证数据血缘覆盖。

这样当前 FTS、BGE-M3、Hybrid 三条基线的 Top10 都在标注池内。以后换新模型或新召回
策略时，若 Top10 出现未标注商品，必须扩充 pool 并补标，不能直接把未标注商品当负例。

## 相关性等级

| label | gain | 含义 |
|---|---:|---|
| Exact | 3 | 类目正确，且可见字段明确满足硬约束和主要偏好 |
| Substitute | 2 | 核心意图和硬约束可接受，仅软偏好或非关键规格有差异 |
| Partial | 1 | 商品相关，但关键条件缺失、无法确认或存在冲突 |
| Irrelevant | 0 | 类目/用途错误，或明确违反核心条件 |

Recall/MRR 把 `gain >= 2` 视为正例；NDCG 使用完整 0–3 gain。每条正式 Query 必须至少有
两个 Exact/Substitute，否则进入 `review_queue.jsonl`，不进入冻结评测集。

## 标注与审计

- 批量初标使用 `gpt-5.6-luna`，只读取 Query 和商品可见字段。
- 三个 worker 写互斥 shard；导入脚本验证候选数量、顺序、grade 范围和 pool fingerprint。
- 合并器拒绝重复冲突和旧候选池标签。
- 正式 Query 另取稳定 10% 独立盲审；机器标签在人工复核前只能称为 silver qrels。
- 简历或报告必须注明“模型初标 + 规则门禁 + 抽样复核”，不能写成人工全量标注。

## 产物

```text
data/processed/eval/shopsimulator_retrieval_v1/
├── selected_tasks.jsonl      # 候选 Query 与源任务血缘
├── candidate_pools.jsonl     # 多系统 judgment pool
├── annotations.jsonl         # 合并后的逐候选标签
├── queries.jsonl             # 冻结 dev/test Query
├── qrels.jsonl               # 冻结检索判断
├── review_queue.jsonl        # 多正例门禁未通过
├── valid_reserve.jsonl       # 可用于后续扩容的有效 Query
├── audit_sample.jsonl        # 稳定抽审样本
└── manifest.json             # 数量、策略与文件哈希
```

这些文件属于本地大型/派生数据，受 `.gitignore` 保护，不提交仓库。仓库只提交生成代码、
数据契约和可复现命令。

## 实测检索指标（2026-08-17 冻结后）

标注完成 180/180：gpt-5.6-luna 110 条 + deepseek-v4-flash 70 条（混合标注源已逐条
记录 annotator_model）。正例门禁通过 133 条、进 review_queue 47 条（0 正例 12、1 正例
35，抽查确认是“具体颜色/图案/款式在目录无多匹配”，非建池漏召回）；冻结 15 dev + 100 test
共 115 条、3,763 qrel。

冻结 test（100 条，top-k=10，Recall/MRR 正例=Exact+Substitute）：

| 后端 | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|
| FTS（BM25） | 0.210708 | 0.299817 | 0.290661 |
| BGE-M3 | 0.518752 | 0.596440 | 0.676776 |
| Hybrid 0.7/0.3 | 0.542868 | 0.646369 | 0.694781 |
| Reranker | 0.554712 | 0.726095 | 0.634743 |

Reranker 为 BGE-M3 ANN Top-100 → BGE-reranker-v2-m3 Top-10，Recall/MRR 最高；其 NDCG
低于 Hybrid 是因为 judged coverage 仅 0.702（top-100 重排会带出约 30% 未标注项，按
unjudged=非相关保守计），真实 NDCG 应更高。FTS/BGE/Hybrid 的 judged coverage 均为 1.0。

## 复现

```powershell
# 1. 生成 Query 与候选池
.\.venv\Scripts\python.exe scripts\data\prepare_shopsimulator_retrieval_benchmark.py `
  --stage pool --local-files-only --batch-size 16

# 2. 导入离线标注分片并合并
.\.venv\Scripts\python.exe scripts\data\merge_shopsimulator_retrieval_annotations.py `
  --shard-dir data\processed\eval\shopsimulator_retrieval_v1\luna_shards_top10

# 3. 冻结 queries/qrels/manifest
.\.venv\Scripts\python.exe scripts\data\prepare_shopsimulator_retrieval_benchmark.py `
  --stage finalize

# 4. 运行普通检索评测
.\.venv\Scripts\python.exe scripts\eval\run_catalog_retrieval_eval.py `
  --platform taobao --locale cn --dataset shopsimulator_retrieval_v1 `
  --backend hybrid --split test --top-k 10 --local-files-only
```
