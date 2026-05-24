# Globex Agent Learning

用于学习、运行并逐步组装“电商搜索 Agent”课程中的关键模块。

完整实施路线见 [ENGINEERING_PLAN.md](ENGINEERING_PLAN.md)。

## 当前架构

项目已迁移为 **LangGraph + DDD 洋葱架构**：

- `src/globex_agent/domain/`：`StandardItem`、`CategoryCard`、Money、订单状态机、关税运费规则与端口。
- `src/globex_agent/application/`：`catalog_search` / `order_usecases`、业务工具、Main/Search/Trade Agent 与编排器。
- `src/globex_agent/infrastructure/`：召回（BGE-M3 + Faiss）、CategoryInsight OpenSearch 路径、事件总线、缓存、队列、JSON/SQL 仓储与韧性组件。
- `src/globex_agent/presentation/`：FastAPI + WebSocket；`frontend/` 为 React 对话界面。
- 主 Agent 默认单干，需要时通过批量 `task_dispatch(dispatches)` 并行调用 `search_agent` / `trade_agent`。

Amazon US/ES/JP 与 Taobao CN 数据库、Query/qrel 和分区检索的实际构建记录见
[docs/data/multiplatform_catalog.md](docs/data/multiplatform_catalog.md)。

## 资料与代码分离

- 课程资料：`D:\刘诗恩\Obsidian\电商搜索Agent资料`
- 项目代码：`D:\PycharmProjects\GlobexAgentLearning`

课程资料通过 Codex 本地项目的辅助文件夹提供上下文，不复制到本仓库。

## 学习流程

每个章节按以下顺序推进：

1. 阅读对应课程文档和必要的前置章节。
2. 说明本章解决的问题与核心设计。
3. 在 `examples/` 中运行关键代码。
4. 将可复用部分逐步整理进 `src/globex_agent/`。
5. 在 `tests/` 中验证当前实现。
6. 把运行结果和面试解释记录到 Obsidian 的“我的学习记录”。

## 阶段规划

1. AgentLoop 与多轮工具调用
2. 多 Agent 按需 fork
3. 商品搜索、比价与运费工具
4. Query/Item 向量召回与 Reranker（BM25/Hybrid 为基线与扩展消融）
5. 上下文压缩与长期记忆
6. AGUI、WebSocket 与 FastAPI
7. Globex 主链路组装
8. 评测、部署思想与面试复盘

## 当前状态

- [x] LangGraph + DDD 架构迁移 Phase 0-4 完成，Phase 5 评测/部署/文档已完成首轮落地
- [x] 主 Agent 单干、search/trade 派发与并行时间重叠均有自动化验收
- [x] FastAPI `/health`、WebSocket 事件、前端构建验收通过
- [x] 运行时已接入 SQLite 商品库 + Faiss + BGE Reranker，实测 `embedding_rerank`
- [x] 最新完整真实模型回归 `13/13 PASS`（平均分 `1.000`），报告见 [eval/report-20260819-223705.md](eval/report-20260819-223705.md)
- [x] 新增 flow query 真实模型评测：9 条 RAG/召回/ESCI query 已跑完整流程；per-case LLM rubric 脚本已就绪，待联网运行后补结果，详见 [docs/experiments/flow_query_eval_20260820.md](docs/experiments/flow_query_eval_20260820.md)
- [x] 真实 LLM smoke 通过；补充 `model.fallback`、`plan.update`、`context.compressed`、熔断、重启恢复与并发会话隔离测试，pytest `159 passed`
- [x] 前端 `npm run build` 通过，`package-lock.json` 已由当前 `package.json` 重新生成；Docker daemon 尚未启动，`docker compose config` 已通过

- [x] 初始化独立项目目录
- [x] 建立 Codex 项目规则
- [x] 关联课程资料路径
- [x] 阅读课程资料至第 15 章
- [x] 建立完整工程计划
- [x] 完成工程阶段 0：环境与测试基线
- [x] 完成工程阶段 1：领域模型与小型商品数据
- [x] 完成工程阶段 2：确定性工具链 MVP
- [x] 完成工程阶段 3：脚本化与真实模型单 AgentLoop 闭环
- [x] 完成工程阶段 4：BM25 检索基线、ESCI 子集与离线指标
- [x] 淘宝多商品检索集 shopsimulator_retrieval_v1 已冻结（115 query）并跑出 FTS/BGE-M3/Hybrid/Reranker 四路指标
- [x] 完成工程阶段 5：商品召回（淘宝检索集冻结 + 两库四路指标 + 双语翻译方案归档）
- [x] 工程阶段 6 中文化整改：知识卡 RAG 已切到真实淘宝中文底座与 ik 分词
- [x] 使用本地 `.env` 完成真实模型固定 case 和自由文本 smoke test

## 快速开始

当前 VS Code 项目已经有 `.venv`。在 VS Code 的 CMD 终端中直接执行：

```bat
.\.venv\Scripts\python.exe examples\00_smoke.py
.\.venv\Scripts\python.exe examples\01_load_catalog.py
.\.venv\Scripts\python.exe scripts\eval\run_recall_eval.py --prepare
.\.venv\Scripts\python.exe -m uvicorn globex_agent.presentation.server:app --port 8000
.\.venv\Scripts\python.exe scripts\smoke_e2e.py
.\.venv\Scripts\python.exe scripts\eval_regression.py
.\.venv\Scripts\python.exe scripts\index\build_item_index.py --batch-size 4
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe examples\04_semantic_retrieval.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe scripts\data\build_category_cards_taobao_zh.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval_taobao_zh.py
docker compose -f infra\opensearch\docker-compose.yml build
docker compose -f infra\opensearch\docker-compose.yml up -d --wait
.\.venv\Scripts\python.exe scripts\index\build_category_kb_taobao_zh.py --embedding-model D:\models\bge-m3 --local-files-only --recreate
.\.venv\Scripts\python.exe scripts\eval\run_category_recall.py --data-dir data\category_insight\taobao_zh --index-name globex_category_kb_taobao_zh_v1 --cards-filename category_cards_taobao_zh.jsonl --cases-filename category_recall_cases_taobao_zh.jsonl --manifest-filename category_recall_manifest_taobao_zh.json --embedding-model D:\models\bge-m3 --reranker-model D:\models\bge-reranker-v2-m3 --reranker-python C:\Anaconda\envs\blog_04\python.exe --local-files-only
.\.venv\Scripts\python.exe examples\05_category_insight.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src scripts examples tests
```

如果 CMD 中中文乱码，先执行 `set PYTHONUTF8=1`；PowerShell 对应写法才是 `$env:PYTHONUTF8 = "1"`。需要重建环境时再使用 `python -m uv sync`。

在 VS Code 中阅读代码，建议按这条路径：

阶段一、二历史代码已删除，可查看 `git show main:src/globex_agent/tools/item_search.py` 等旧版本。

迁移后主链：`application/prompts/globex.yml` → `application/agents/main_agent.py` → `application/agents/search_agent.py` / `trade_agent.py` → `application/tools/*` → `application/agents/orchestrator.py`。

阶段四：`infrastructure/recall/base.py` → `infrastructure/recall/keyword.py` → `src/globex_agent/eval/recall_metrics.py` → `scripts/data/prepare_esci_subset.py` → `scripts/eval/run_recall_eval.py` → `data/eval/recall_manifest.json`

阶段五主链：`infrastructure/recall/embedding.py` → `infrastructure/recall/index.py` → `scripts/index/build_item_index.py` → `infrastructure/recall/reranker.py` → `scripts/eval/run_retrieval_comparison.py` → `examples/04_semantic_retrieval.py`。

阶段六知识卡：`data/category_insight/taobao_zh/category_taxonomy_taobao_zh.json` → `scripts/data/build_category_cards_taobao_zh.py` → `scripts/data/build_category_eval_taobao_zh.py` → `infrastructure/recall/category_kb.py` → `scripts/index/build_category_kb_taobao_zh.py` → `category_insight/reranking.py` → `category_insight/service.py` → `application/tools/category_insight_tool.py` → `examples/05_category_insight.py`。

## 已实现模块

- `LocalCatalog`：逐行校验平台级 `StandardItem`；同平台、同 canonical、同语言的重复行保留评分更高/更新的一条，EN/ZH parallel listing 保持分离。
- 演示数据：36 条平台级 `StandardItem`，通过 12 个 `same_group_id` 关联跨平台同款，覆盖 3 个品类和 3 个平台。
- 确定性工具链：`ItemSearch → PriceCompare → ShippingCalc → ItemPicker → ShoppingSummary`。
- 单 AgentLoop：LangGraph 实际执行 `human → ai(tool_call) → tool → ... → ai(final)` 消息循环；固定 case 可用脚本模型，`--query --real` 使用 DeepSeek 解析自由文本并编排工具。
- 检索基线：可替换 `SearchBackend`、无外部依赖的 BM25、35 个 ESCI 封闭查询组和 Recall/MRR/NDCG/空召回率报告；每个返回商品都有显式标注。
- 双塔检索：课程主线保留 BGE-M3 ANN Top-100 → 可选 BGE Reranker Top-10；runtime 当前默认采用 ANN overfetch → `same_group_id` canonical 去重 → Top-K，展示语言只在相关性排序完成后选择。BM25 只作基线，Hybrid 只作显式扩展消融。
- 商品知识卡 RAG：OpenSearch 使用独立的动态 Hybrid 配置；runtime 默认不启用当前会降分的通用 Reranker，只有显式 opt-in 时才运行，失败时保留 Hybrid 顺序。知识卡配置不混入商品召回主链。
- 验证结果：Python 3.10.20；Ruff 与 pytest 通过；固定 case、真实自由文本、阶段四基线、阶段五商品召回和阶段六 CategoryInsight 均已实际运行成功。

阶段 2 的计算口径：

- ItemSearch 按原文一次检索一个平台，每个平台返回一个 `ItemSearchOutput`；本地实现委托给 BM25 `SearchBackend`，召回不读取用户画像。
- PriceCompare 直接接收多平台合流的 `list[Candidate]`，使用 `Decimal` 和版本化演示汇率，按标价排序并截断到 Top-12。
- ShippingCalc 只接收 `PriceCompare.ranked`，按第 12 章的 0.5 kg 占位重量、运费表和税率表估算 `landed_cny`。
- ItemPicker 先执行预算、材质和属性硬约束，再综合到手价、评分、时效、税档和偏好打分；打分后才按 `same_group_id` 去掉同款重复推荐。
- ShoppingSummary 使用固定模板收敛，不调用模型，并明确披露模拟数据与估算规则。

当前依赖保持最小：

| 依赖 | 用途 | 可选替代 |
| --- | --- | --- |
| `pydantic` | 定义 `StandardItem`、`Candidate`、`PricePoint`、`LandedCost`、`PickedItem` 等稳定契约 | 标准库 dataclass + 手写校验，但错误定位和 JSON 解析成本更高 |
| `langchain` / `langchain-openai` | 第 10 章统一模型入口、消息和工具抽象 | 纯 SDK 手写消息协议，但偏离课程 |
| `langgraph` | 第 2 章 AgentLoop、检查点和循环限制 | 纯 Python `while`，但缺少课程状态图语义 |
| `PyYAML` | 从 `prompts.yml` 加载版本化提示词 | Python 字符串常量，但不符合第 10 章配置方式 |
| `python-dotenv` | 真实模型模式读取本地 `.env` | 由终端手动设置环境变量 |
| `numpy` | 向量归一化、精确检索基线与索引测试 | 纯 Faiss，但不便保留可解释的正确性基线 |
| `pyarrow` | 流式读取 Amazon ESCI 官方 parquet，避免把百万行一次性载入内存 | Hugging Face Viewer 小批量下载，但全量分组筛选更慢 |
| `faiss-cpu` | 构建 HNSW + Inner Product 商品 ANN 索引 | Milvus，适合更大规模和服务化 |
| `sentence-transformers` | 在主环境加载 BGE-M3 Query/Item 编码器 | 直接使用 Transformers，但要手写池化和批处理 |
| `pytest` | 参数化测试、夹具和清晰失败报告 | 标准库 `unittest` |
| `ruff` | 一次完成导入排序与静态规范检查 | Flake8 + isort 等组合 |

## 阶段 2 面试解释

| 模块 | 为什么这样设计 | 替代方案 | 主要失败模式与代码位置 |
| --- | --- | --- | --- |
| ItemSearch | 先建立可复现关键词基线，后续才能证明向量召回是否改进 | BM25、Embedding、混合召回 | 同义词和跨语言召回弱；见 `src/globex_agent/tools/item_search.py` |
| PriceCompare | 接收合流候选、币种归一并做 Top-N 剪枝，金额使用 Decimal 避免浮点误差 | 实时汇率服务 | 演示汇率不代表实时值；见 `src/globex_agent/tools/price_compare.py` |
| ShippingCalc | 与标价比较分离，允许标价 Top-N 后再估算到手价 | HS Code + 原产地税费服务 | 税率和时效是简化规则；见 `src/globex_agent/tools/shipping_calc.py` |
| ItemPicker | 硬约束先过滤，偏好只能影响排序，不能覆盖预算和黑名单 | Learning-to-Rank 或 LLM Reranker | 未知硬约束会安全停止；见 `src/globex_agent/tools/item_picker.py` |
| ShoppingSummary | 模板输出稳定、可测、零模型成本 | LLM 生成摘要 | 表达较固定；见 `src/globex_agent/tools/shopping_summary.py` |

## 阶段 3：单 AgentLoop

默认运行不需要模型密钥：

```bat
.\.venv\Scripts\python.exe examples\03_single_agent.py --case 1
```

消息顺序应为：

```text
planner
→ item_search × 3（单 Loop 顺序调用三个平台）
→ price_compare
→ shipping_calc
→ item_picker
→ shopping_summary（终结）
```

真实模型模式需要把 `.env.example` 复制为 `.env`，填写 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`LLM_MAIN`，然后运行：

```bat
.\.venv\Scripts\python.exe examples\03_single_agent.py --case 1 --real
```

固定 `--case` 用 `queries.jsonl` 中预先校验的结构化请求；要验证真正的自然语言 Planner，运行：

```bat
.\.venv\Scripts\python.exe examples\03_single_agent.py --real --query "500元以内防泼水、不要真皮的通勤背包"
```

这时 Planner 使用同一个真实模型输出课程规定的结构化字段，并把预算、平台和硬约束更新到本轮 `runtime.request`；商品数据和后续业务计算仍是本地演示实现。

课程原文使用 `langgraph.prebuilt.create_react_agent`；LangGraph 1.x 已将它标记为弃用并推荐 `langchain.agents.create_agent`。为了让当前代码与原文可逐行对应，阶段三暂时保留原 API，不在本阶段擅自迁移。

### 阶段 3 面试解释

| 模块 | 为什么这样设计 | 替代方案 | 主要失败模式与代码位置 |
| --- | --- | --- | --- |
| `main_agent` | 忠实使用课程的 `create_react_agent + tools + prompt + checkpointer` 组成单 AgentLoop，并加迭代上限和超时 | 手写 `while` 循环或迁移到 `create_agent` | 模型反复调用工具会触发最大迭代，外部调用过慢会超时；见 `src/globex_agent/agent/main_agent.py` |
| 工具适配层 | 只把阶段二纯函数包装成 LangChain Tool，业务规则仍由确定性代码负责 | 让模型直接完成检索、计算和筛选 | 工具输入输出契约漂移会中断链路；见 `src/globex_agent/tools/agent_tools.py` |
| YAML 提示词与工具注册表 | 对应课程第 10、14 章，集中约束调用顺序、终结工具和可见工具集 | 在 Python 中散落字符串和工具列表 | 提示词只能引导、不能保证模型服从，因此终结判断和运行上限仍由代码兜底；见 `src/globex_agent/prompt/prompts.yml`、`src/globex_agent/agent/tool_registry.py` |
| `ScriptedShoppingModel` | 在没有密钥时仍真实走 LangGraph 消息、ToolNode、检查点和终结链路，测试结果可复现 | 测试时调用真实 LLM | 它验证编排和契约，不验证自然语言规划质量；见 `src/globex_agent/agent/scripted_model.py` |

当前已为 CategoryInsight 在本机 Docker 引入单节点 OpenSearch；尚未引入 fork、vLLM、AGUI 或 K8s。同质子 AgentLoop fork 留到后续多 Agent 阶段；商品召回层继续使用 Faiss，不依赖知识卡 OpenSearch。

## 阶段 4：离线检索基线

阶段四不调用 LLM，也不构建向量模型。`ItemSearch` 只依赖稳定的 `SearchBackend` 接口；本阶段先用 `keyword-bm25-v1` 建立基线，阶段五再在不修改模型可见工具参数的前提下注入 Query/Item 向量召回与精排。按本项目约定取消 User 塔：画像不编码、不建索引，也不是 `SearchBackend` 的输入；在 query 和平台相同的前提下，换用户不会改变召回。画像仍可作为 Agent 上下文，并在后置 `ItemPicker` 中处理黑名单和软偏好。

公开评测子集来自 Amazon Shopping Queries ESCI 数据集（Apache-2.0），通过已连接商品文本的 `tasksource/esci` 镜像按 query_id 下载小样本。最终保存 35 个互不交叉的查询组和 720 个去重商品，拆分为 train/dev/test=21/6/8；完整多 GB 数据不进入仓库。

评测采用逐 query 封闭候选池，候选必须 100% 有 ESCI 标注。Recall/MRR 只把 Exact 当正例，每个 query 保留 2～5 个 Exact；Complement/Substitute 仅保留给分级 NDCG。当前数据仅含英文，因此不能据此声称中文或中英跨语言效果。此处不制作训练样本，也不执行训练截断、正负采样、Hard Negative 或微调。

离线复现命令：

```bat
.\.venv\Scripts\python.exe scripts\eval\run_recall_eval.py --prepare
```

2026-08-15 按新口径重跑的 BM25 实际结果：Recall@10=0.531429、MRR@10=0.650079、NDCG@10=0.484503、空召回率=0.028571；Recall@100=0.813810。报告本地生成到 `output/eval/`，该目录不提交。候选池最多只有 51 条，因此 Recall@100 是封闭池内的深度诊断，不是全商品库召回指标。

## 阶段 5：Query/Item 向量召回与精排

本阶段没有 User 塔。Query 在线编码，Item 离线编码；画像继续只供 Planner 和后置 ItemPicker 使用。课程主实现使用官方原始 `BAAI/bge-m3` 在项目主环境 CPU 编码，使用 `BAAI/bge-reranker-v2-m3` 在 Conda `blog_04` 的 CUDA 11.8 常驻子进程中运行时转 FP16；没有下载第三方 FP16 重打包。Item 文本先删除重复 title 和字面量 `None`，再以 title 优先的字段格式编码；Embedding 窗口为 512，Reranker 按课程 Query/Doc 预算使用 256 token。

`blog_04` 已有的 CUDA 版 PyTorch 保持不变；精排进程只需补齐固定版本的 Transformers 运行依赖，不需要在主项目环境安装 CUDA 包：

```bat
C:\Anaconda\envs\blog_04\python.exe -m pip install -r scripts\reranker\requirements-blog-04.txt
```

首次运行需要联网构建索引并让 GPU 子进程下载一次官方精排权重：

```bat
.\.venv\Scripts\python.exe scripts\index\build_item_index.py --batch-size 4
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --reranker-python C:\Anaconda\envs\blog_04\python.exe
```

两套模型缓存齐全后可完全离线运行：

```bat
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
.\.venv\Scripts\python.exe examples\04_semantic_retrieval.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe
```

商品召回的课程主结果固定为 `BGE-M3 Query/Item → Faiss HNSW + IP Top-100 → BGE-Reranker-v2-m3 Top-10`。720 个 Item 已真实编码为1024维归一化向量，索引参数为 `M=32、efConstruction=200、efSearch=128`。BM25 只作词法基线，Hybrid 不进入主链，也不再复用知识卡 RAG 的动态权重或 Top-30/Top-8 配置。

2026-08-15 本机重跑结果如下。准确率取独立 test split（8 个 query）；候选 Recall@100 和延迟是 35 个 query 的整体诊断，因此列名显式区分口径。

| 角色/方案 | Test Recall@10 | Test MRR@10 | Test NDCG@10 | Overall Recall@100 | Overall P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 基线 | 0.6333 | 0.7139 | 0.5646 | 0.8138 | 0.10 ms |
| BGE-M3 + Faiss HNSW/IP | 0.7583 | 0.7135 | 0.6448 | 1.0000 | 86.68 ms |
| ANN Top-100 → BGE Reranker Top-10（课程主结果） | 0.8750 | 0.8750 | 0.7886 | 1.0000 | 528.32 ms |

三路结果的 judged coverage 均为 1.0。BGE Reranker 把独立 test 的 Recall@10 从 0.7583 提升到 0.8750，NDCG@10 从 0.6448 提升到 0.7886。35-query overall 的主链 Recall@10=0.633810、NDCG@10=0.592388，只作诊断，不与 test 主结果混报。当前封闭候选池只有8～51条，所以代码请求 Top-100、候选 Recall@100=1.0，但不能声称每个 query 实际精排了100条。完整结果和 bad case 位于本地 `output/eval/retrieval_comparison.json`。

可提交、可长期追溯的实验环境、数据哈希、完整指标与复现命令记录在 `docs/experiments/phase5_bge_m3_retrieval_2026-08-15.md`。

Hybrid 仅在明确要求扩展消融时启用，而且必须显式给出静态权重；脚本不会自动套用知识卡 RAG 的动态分类权重。例如复现历史的 0.4/0.6 诊断可另存报告：

```bat
.\.venv\Scripts\python.exe scripts\eval\run_retrieval_comparison.py --local-files-only --reranker-python C:\Anaconda\envs\blog_04\python.exe --include-hybrid-extension --semantic-weight 0.4 --lexical-weight 0.6 --output output\eval\retrieval_hybrid_ablation.json
```

## 阶段 6：CategoryInsight 商品知识卡 RAG（中文化整改完成）

当前知识卡底座是阶段五使用的真实 ShopSimulator 淘宝中文目录。首期冻结 8 个普通
品类：乳胶枕、儿童学习椅、手机直播补光灯、汽车氛围灯、平板电脑支架、颈椎按摩器、
羽毛球包、户外电源。每个品类生成 2 张 `bestseller`、3 张 `attribute`、1 张
`price_range`，共 48 张课程七字段 `CategoryCard`。

淘宝目录没有真实销量，bestseller 使用同质商品数和 `source_attributes` 覆盖度作
离线代理，并显式标记 `generated_offline_proxy`，不伪造销量。价格来自真实
`price_cny` 或 `variants[].price_cny`，切三分位生成价格卡；卡片与 provenance
会明确标注“挂牌/规格价，不是历史成交价”。完整口径见
`docs/data/category_card_data_contract.md`。

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe scripts\data\build_category_cards_taobao_zh.py
.\.venv\Scripts\python.exe scripts\data\build_category_eval_taobao_zh.py
```

中文检索评测集冻结为 50 条 Query，覆盖名词型、属性约束型、气质/风格型和口语
型，split 为 train/dev/test=30/10/10。每条 Query 对全局 48 张卡全部显式判断，
恰有 5 张相关卡，按顺序获得 5、4、3、2、1 gain。

OpenSearch 使用 `analysis-ik`，索引 `globex_category_kb_taobao_zh_v1` 以
`ik_max_word` 建索引、`ik_smart` 查询；中文检索文本从 sidecar 写入
`retrieval_text` 和 `retrieval_text_zh`。BGE-M3 和 BGE Reranker 分别使用本地
`D:\models\bge-m3` 与 `D:\models\bge-reranker-v2-m3`。

独立 test split 的四路实测（v1 基线，按课程口径取 @8）：

| 方案 | Recall@8 | MRR@8 | NDCG@8 | AllCardTypes@8 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 1.0000 | 1.0000 | 0.8723 | 1.0000 |
| KNN / BGE-M3 | 1.0000 | 1.0000 | 0.8695 | 1.0000 |
| Hybrid | 1.0000 | 1.0000 | 0.8709 | 1.0000 |
| Hybrid + BGE Reranker（0.92 跳过） | 1.0000 | 1.0000 | 0.8531 | 1.0000 |
| Hybrid + BGE Reranker（全量精排） | 0.9000 | 0.9500 | 0.7915 | 0.7000 |

全量精排在 v1 基线 48 卡封闭池上不升反降，把 Reranker 的 Recall@8 从 1.0
拉到 0.90、NDCG 从 0.8531 拉到 0.7915，因此生产默认仍保留 0.92 首分跳过。

旧英文 ESCI 演示版仍保留在 `data/category_insight/` 根目录，作为历史实验，不覆盖：

- 12 个英文 ESCI 品类、72 张卡，standard 分词。
- 50 条英文 Query 与固定种子价格/销量合成字段。
- 历史实验记录见 `docs/experiments/phase6_category_insight_2026-08-15.md`、
  `docs/experiments/phase6_category_insight_v2_2026-08-15.md` 和
  `docs/experiments/phase6_bilingual_v3_2026-08-15.md`。

`CategoryInsightService` 仍保留 quick/deep 提炼、品类归一、普通品类空组件、
向量失败降级 BM25、OpenSearch 失败返回 confidence=0，以及 Reranker 失败保留粗排。
当前尚无 WebSearch 工具，低置信度 WebSearch 补充仍是后续项。
