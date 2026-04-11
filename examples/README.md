# 章节示例

## 00 工程导入检查

命令：

```bat
.\.venv\Scripts\python.exe examples\00_smoke.py
```

输入：无。实际输出：`globex_agent 0.1.0 imported successfully`。

## 01 本地商品目录加载

对应课程：第 09 章工程初始化与第 09-1 章多平台商品数据底座。

```bat
.\.venv\Scripts\python.exe examples\01_load_catalog.py
```

输入：`data/demo/products.jsonl`。2026-08-12 实际输出摘要：

```text
input records: 36
accepted records: 36
standard items: 36
same-product groups: 12
platforms: aliexpress, amazon, shopee
example: Aurora QuietPro 头戴式降噪耳机
same_group_id: hp-aurora-quietpro
platform items: aliexpress:aqp-3001=CNY 1219.00, amazon:aqp-1001=CNY 1299.00, shopee:aqp-2001=CNY 1269.00
```

## 02 不依赖 LLM 的确定性推荐链路

对应课程：第 11 章 ItemSearch、第 12 章 PriceCompare/ShippingCalc、第 14 章 ItemPicker/ShoppingSummary。CategoryInsight 和 AgentLoop 尚未接入。

```bat
.\.venv\Scripts\python.exe examples\02_deterministic_pipeline.py --case 1
```

输入：`data/demo/queries.jsonl` 第 1 条——“预算 1500 元的通勤降噪耳机，不要入耳式”，并加载 `demo-user-001` 用户画像供后置 ItemPicker 使用；ItemSearch 不读取画像。2026-08-13 实际输出摘要：

```text
ItemSearch[aliexpress]: 3 candidates / 7 total_recall / truncated=True
ItemSearch[amazon]: 3 candidates / 7 total_recall / truncated=True
ItemSearch[shopee]: 3 candidates / 7 total_recall / truncated=True
PriceCompare: 9 ranked PricePoint rows
ShippingCalc: 9 LandedCost rows
ItemPicker: 3 picked / 0 rejected
Nimbus Lite（shopee）：item_id=shopee:nl-2001，到手价 CNY 800.94
Sonic Commute X2（shopee）：item_id=shopee:scx2-2001，到手价 CNY 1076.54
Aurora QuietPro（shopee）：item_id=shopee:aqp-2001，到手价 CNY 1405.14
```

可用 `--case 1` 到 `--case 6` 运行全部固定场景。其中第 3、5、6 条会因到手价超预算或属性不满足而得到结构化空结果，这是硬约束生效的预期行为，不是运行错误。

## 03 单 AgentLoop 最小闭环

对应课程：第 01 章 AgentLoop 概念、第 02 章多轮工具调用、第 10 章模型与提示词配置、第 14 章主 Loop 组装。本阶段明确不接 fork。

默认使用脚本化模型，不访问网络、不需要 `.env`：

```bat
.\.venv\Scripts\python.exe examples\03_single_agent.py --case 1
```

2026-08-13 实际调用轨迹：

```text
[human] 预算 1500 元的通勤降噪耳机，不要入耳式
[ai] tool_call: planner
[tool] planner
[ai] tool_call: item_search
[tool] item_search
[ai] tool_call: item_search
[tool] item_search
[ai] tool_call: item_search
[tool] item_search
[ai] tool_call: price_compare
[tool] price_compare
[ai] tool_call: shipping_calc
[tool] shipping_calc
[ai] tool_call: item_picker
[tool] item_picker
[ai] tool_call: shopping_summary
[tool] shopping_summary
[ai] final answer
terminal_tool: shopping_summary
```

这不是直接调用阶段二 pipeline：每个 `ai tool_call`、`ToolMessage` 追加、下一轮模型调用和终止判断都由 LangGraph 执行。脚本化模型只代替远程 LLM 产生可复现的工具选择。

真实模型模式：

```bat
.\.venv\Scripts\python.exe examples\03_single_agent.py --case 1 --real
```

固定 `--case` 仍使用 `queries.jsonl` 中预先校验的结构化请求。验证真正的自然语言 Planner：

```bat
.\.venv\Scripts\python.exe examples\03_single_agent.py --real --query "500元以内防泼水、不要真皮的通勤背包"
```

2026-08-13 实际结果：DeepSeek 输出 `budget=500`、`water_resistant=yes`、`excluded_material=真皮`、`usage=通勤` 和三个平台，随后完成 ItemSearch、PriceCompare、ShippingCalc、ItemPicker 并进入 `shopping_summary`，进程退出码为 0。`--query` 必须和 `--real` 一起使用；可选 `--user-id demo-user-001` 注入一个演示画像。

## 阶段 04 离线检索评测

```bat
.\.venv\Scripts\python.exe scripts\eval\run_recall_eval.py --prepare
```

`--prepare` 会复用仓库中的 ESCI 封闭评测子集；只有显式增加 `--refresh` 才联网重新抽取。2026-08-15 实际输出为 Recall@10=0.531429、MRR@10=0.650079、NDCG@10=0.484503、空召回率=0.028571。

## 04 Query/Item 向量召回与精排

首次运行会下载官方 BGE-M3 和 BGE-Reranker-v2-m3，并为 ESCI 的720个清洗后商品构建1024维 Faiss HNSW/IP Item 索引。BGE-M3 留在项目主环境CPU；Reranker 由 `blog_04` 的CUDA 11.8常驻子进程以FP16运行：

```bat
.\.venv\Scripts\python.exe scripts\index\build_item_index.py --batch-size 4
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --reranker-python C:\Anaconda\envs\blog_04\python.exe
```

模型缓存齐全后再增加 `--local-files-only`，即可完全离线复跑。

默认脚本只输出商品召回课程主链和 BM25 基线，不再把知识卡 RAG 的 Hybrid 配置混进来。2026-08-15 本机实测中，准确率取独立 test split；Recall@100 和 P50 是 35-query 整体诊断：

```text
backend              Test Recall  Test MRR  Test NDCG  Overall Recall@100  Overall P50ms
bm25_baseline        0.6333       0.7139    0.5646     0.8138              0.10
ann_recall           0.7583       0.7135    0.6448     1.0000              86.68
ann_reranker         0.8750       0.8750    0.7886     1.0000              528.32
```

中文演示目录接回 `ItemSearch`：

```bat
.\.venv\Scripts\python.exe examples\04_semantic_retrieval.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
```

固定 query 下依次打印 BM25 基线、BGE-M3 + Faiss HNSW/IP、ANN Top-100 + BGE Reranker 的排序。runtime `search_items(...)` 默认注入 ANN 后先做 canonical 去重并直接返回 Top-K；增加 `--runtime-reranker` 才显式启用精排，失败时保留 canonical ANN 顺序。当前封闭候选池最多51条，因此请求深度是100，但本数据无法让每个 query 实际返回100条。

Hybrid 仅作单独扩展消融，必须显式启用并给出权重：

```bat
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe --include-hybrid-extension --semantic-weight 0.4 --lexical-weight 0.6 --output output\eval\retrieval_hybrid_ablation.json
```

## 05 CategoryInsight 商品知识卡 RAG

先基于阶段五的 ESCI 商品底座重建知识卡和全判断评测集，再启动本地 OpenSearch、
构建 1024 维 BGE-M3 卡片索引：

```bat
.\.venv\Scripts\python.exe scripts\data\build_category_cards.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval.py
docker compose -f infra\opensearch\docker-compose.yml up -d
.\.venv\Scripts\python.exe scripts\index\build_category_kb.py --local-files-only --recreate
```

运行 50-query 离线评测和 quick/deep 工具示例：

```bat
.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe examples\05_category_insight.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
```

2026-08-15 独立 test 实测：动态 Hybrid Recall@10=0.9600、NDCG@10=0.8725；
Hybrid + BGE Reranker Recall@10=0.8000、MRR@10=0.9000、NDCG@10=0.7721。
因此工具 runtime 默认不启用 Reranker；示例传 `--reranker-python` 或
`--enable-reranker` 属于显式实验路径。
`bathroom fan` 示例归一到 `home ventilation fans`；quick 返回 5 个代表商品和 3 个
价格层，deep 额外返回 3 个属性分布，confidence=0.68。价格和流行度是明确标记的
固定种子离线增强，不是真实价格或销量。

## 本轮报错与解决

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `uv` 缓存目录拒绝访问 | Codex 沙箱不能读取用户级 uv 缓存 | 验收和 VS Code 运行直接调用 `.venv` 中的 Python/Ruff |
| pytest 默认临时目录拒绝访问 | 中文 Windows 用户名与临时目录 ACL 组合问题 | 测试夹具改为在既有测试目录创建独立临时文件并自动清理 |
| CMD 中误用 `$env:PYTHONUTF8 = "1"` | `$env:` 是 PowerShell 语法 | CMD 使用 `set PYTHONUTF8=1`；当前示例通常无需额外设置 |
| GBK 终端不能输出 `¥` | Windows 默认控制台编码不支持该字符 | 价格单位固定显示为 `CNY` |
| GBK 终端不能输出模型生成的 Emoji | 真实模型可能返回当前代码页无法编码的字符 | 示例保留当前终端编码并将不可表示字符安全替换，避免最后 `print` 使成功链路报错 |
| DeepSeek Thinking 模式拒绝 Planner 的强制 `tool_choice` | `function_calling` 结构化输出会强制选择结构工具 | Planner 改用 JSON 模式，仍由 Pydantic 校验课程规定的输出 Schema |
| DeepSeek 把空字符串/空列表返回成 `[]`/`{}` | JSON 模式不保证服务端严格执行 Schema | 只归一化空容器漂移，其他类型错误继续由 Pydantic 拒绝 |
| 原实现出现 `Offer/offer_id/canonical_product_id` | 这些是偏离课程原文的自定义结构 | 回调为平台级 `StandardItem.item_id`，跨平台只用 `same_group_id` 关联 |
| pytest 报 `create_react_agent` 弃用警告 | 课程 API 在 LangGraph 1.x 中已进入迁移期 | 阶段三为贴近原文保留；后续单独做 `create_agent` 迁移，不混入本章 |
| Hugging Face 提示 Windows 不支持缓存软链接 | 未启用开发者模式，模型缓存退化为普通文件 | 不影响运行，只会多占磁盘；本阶段不要求管理员权限或修改系统设置 |
