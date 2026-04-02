# Globex 电商搜索 Agent 完整工程计划

> 状态：阶段 0～2 已实现并验证，下一步进入阶段 3
> 当前阅读进度：已读完第 14 章
> 当前代码状态：确定性五工具链、36 条模拟报价、34 个测试和 3 个示例已运行
> 实施原则：离线优先、纵向切片、接口与实现解耦、每阶段必须实际运行并验收

## 1. 项目目标

实现一个可独立演示、可评测、可继续演进的电商搜索 Agent。用户用自然语言描述预算、用途和偏好后，系统能够：

1. 理解并结构化购物需求。
2. 从本地商品数据中召回候选商品。
3. 结合语义相关性、用户偏好、价格、运费和约束进行筛选。
4. 在任务复杂时按需 fork 同质子 AgentLoop 并行处理。
5. 输出结构化推荐清单、选择理由、风险与数据来源。
6. 保存可控的长期偏好，并在后续会话中使用。
7. 通过评测脚本展示真实、可复现的效果指标。
8. 通过 FastAPI 和 WebSocket 暴露服务，并能被后续客服 Agent 调用。

第一版不是复刻大型电商平台，也不追求生产规模。第一版的目标是证明完整工程链路，并让每一项简历描述都有代码、测试和运行证据。

## 2. 项目范围

### 2.1 核心交付范围

- 小规模、可检查的本地商品目录。
- 统一商品 Schema 和数据校验。
- 9 个业务工具中的核心离线实现。
- 单 AgentLoop 的 Think → Act → Observe → Reflect 闭环。
- Query / User 双通道召回、融合与 Reranker。
- CategoryInsight 商品知识 RAG。
- 主 AgentLoop + 按需 fork 的同质子 AgentLoop。
- Cache Breakpoint、工具结果压缩和长期偏好 Store。
- 检索评测、工具评测、Agent 轨迹评测与回归测试。
- FastAPI、任务取消、WebSocket/AGUI 事件流。
- 面向客服 Agent 的稳定推荐能力接口。

### 2.2 暂不进入第一版

- 京东、淘宝等真实线上平台 API。
- 对平台页面进行爬取。
- OpenSearch、Milvus、Redis 等重型基础设施。
- vLLM、GPU 服务和 K8s。
- 从零训练三塔 Embedding、SFT 或 Agentic RL。
- 完整 React 商业前端。
- 自动修改 Prompt 或自动训练的全自动飞轮。

这些能力只在本地闭环稳定且有评测基线后再决定是否引入。

## 3. 最终主链路

```mermaid
flowchart LR
    U["用户购物需求"] --> P["Planner 结构化需求"]
    P --> CI["CategoryInsight 可选品类洞察"]
    P --> IS["ItemSearch 商品召回"]
    CI --> IS
    IS --> PC["PriceCompare 比价"]
    PC --> SC["ShippingCalc 到手价"]
    SC --> IP["ItemPicker 约束与偏好精挑"]
    IP --> SS["ShoppingSummary 推荐结果"]

    M["长期偏好 Store"] --> P
    M --> IS
    M --> IP
    SS --> M

    IS -. "复杂任务" .-> D["dispatch_tool"]
    D --> S1["同质子 AgentLoop A"]
    D --> S2["同质子 AgentLoop B"]
    S1 --> IS
    S2 --> IS
```

完整工具集包括：

| 工具 | 第一版定位 |
| --- | --- |
| `Planner` | 把自然语言需求转换成预算、品类、偏好、硬约束 |
| `ChatFallback` | 非购物问题、信息不足和需要澄清时的终结性兜底 |
| `WebSearch` | 第一版默认关闭；后续作为外部信息兜底 |
| `CategoryInsight` | 从本地品类知识卡中召回选购知识 |
| `ItemSearch` | 对统一商品目录执行关键词、语义和个性化召回 |
| `ItemPicker` | 在候选集中执行硬约束过滤和综合排序 |
| `PriceCompare` | 对同一标准商品的多平台报价进行比较 |
| `ShippingCalc` | 计算运费、税费和到手价估算 |
| `ShoppingSummary` | 输出最终推荐清单，是终结性工具 |
| `dispatch_tool` | 元工具；按需 fork 同质子 AgentLoop，不属于业务工具 |

## 4. 工程原则与关键取舍

1. **先确定性后智能化**：先用普通 Python 串通工具，再让 LLM 决定调用顺序。
2. **先纵向闭环后横向扩展**：先让一条查询从输入走到推荐结果，再丰富所有工具。
3. **接口不绑定后端**：`ItemSearch` 依赖 `SearchBackend` 抽象；本地关键词、向量索引和未来平台 API 都是可替换实现。
4. **本地优先**：先用 JSONL、小型内存索引和本地文件 Store，不提前引入分布式组件。
5. **数据来源透明**：真实字段、派生字段和模拟字段必须可区分。
6. **硬规则交给代码**：预算、币种、黑名单、到手价等约束不能只依赖模型判断。
7. **LLM 输出必须校验**：所有工具入参和关键输出使用 Schema 校验。
8. **失败可以降级**：工具超时、空召回和子 Agent 失败要成为可处理结果，不能拖垮整条任务。
9. **指标不得虚构**：文档中只记录实际跑出的数据集规模、准确率、延迟和成功率。
10. **每阶段可回滚**：每个阶段独立测试、独立示例、独立 Git 提交。

## 5. 目标目录

课程中的 `app/` 目录在本仓库映射为 `src/globex_agent/`，不复制课程文档。

```text
GlobexAgentLearning/
├── src/globex_agent/
│   ├── domain/                 # 核心 Schema、枚举和接口
│   ├── config/                 # 环境和运行配置
│   ├── catalog/                # 商品加载、清洗和标准化
│   ├── tools/                  # 9 个业务工具
│   ├── recall/                 # 关键词、Embedding、User 通道、融合、精排
│   ├── rag/                    # 品类知识卡、索引和检索
│   ├── agent/                  # AgentLoop、工具注册、fork 和中间件
│   ├── memory/                 # 长期偏好 Store
│   ├── context/                # Cache Breakpoint 与上下文压缩
│   ├── api/                    # FastAPI、WebSocket、AGUI 事件
│   ├── eval/                   # 数据集、指标、Rubric 和报告
│   ├── observability/          # 结构化日志、Trace 和运行统计
│   ├── resilience/             # 超时、重试、熔断和幂等
│   └── security/               # 工具白名单、内容过滤和脱敏
├── examples/                   # 各阶段独立可运行示例
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── data/
│   ├── demo/                   # 可提交的小型演示数据
│   ├── processed/              # 可复现生成的小型处理结果
│   └── eval/                   # 固定评测集
├── scripts/
│   ├── data/
│   ├── index/
│   └── eval/
├── output/                     # 本地运行产物，不提交
├── .env.example
├── pyproject.toml
└── ENGINEERING_PLAN.md
```

目录只随阶段逐步创建，不一次性生成空模块。

## 6. 分阶段实施计划

### 阶段 0：工程基线

**目标**

建立最小、可重复的 Python 开发与测试环境。

**任务**

- [x] 确认 Python 版本和虚拟环境。
- [x] 选择依赖管理方式并生成锁文件。
- [x] 只加入当前需要的轻量依赖，并说明每个依赖的用途和替代方案。
- [x] 配置 `pytest`、格式检查和最小导入测试。
- [x] 补全 `.gitignore`，排除 `.env`、原始数据、模型、索引和运行产物。
- [x] 建立首个可运行示例和测试。
- [x] 创建工程基线 Git 提交。

**验收**

- 新环境可以按 README 中的一组命令完成安装。
- `globex_agent` 可以导入。
- 测试命令执行成功，至少包含一个 smoke test。
- 仓库中不存在真实密钥和大型文件。

**阶段产物**

- `pyproject.toml`
- 锁文件
- `tests/test_smoke.py`
- 更新后的 `README.md` 和 `.env.example`

---

### 阶段 1：领域模型与小型商品数据

**目标**

让后续所有工具共享一套稳定的数据契约。

**数据路线**

1. 先制作 30～50 条受控演示商品，覆盖 2～3 个品类和 2～3 个模拟平台。
2. 同一商品的跨平台报价通过 `canonical_product_id` 关联。
3. 再从真实公开数据集中按 `query_id` 抽取子集，避免只抽商品造成查询与标注失联。
4. 价格、平台、运费等公开数据中不存在的字段可以补充，但必须标记为 `synthetic` 或 `derived`。

**核心 Schema**

- `StandardItem`：标准商品。
- `Offer`：平台报价、币种、库存和链接。
- `UserProfile`：偏好、黑名单和历史反馈。
- `SearchRequest` / `SearchCandidate` / `SearchResult`。
- `PriceComparison` / `ShippingQuote`。
- `PickedItem` / `ShoppingRecommendation`。
- `DataProvenance`：字段来源与生成方式。

**任务**

- [x] 定义字段、枚举、必填项和校验规则。
- [x] 编写 JSONL 加载器与错误行报告。
- [x] 建立去重、缺失值和异常价格处理。
- [x] 建立演示商品、用户和查询数据。
- [x] 记录数据来源、许可、抽样规则和模拟字段。

**验收**

- 合法数据全部加载成功。
- 非法币种、负价格、空商品 ID 等错误会被拒绝。
- 同一商品的多平台报价能够正确归组。
- 测试覆盖成功加载和主要失败分支。

**阶段产物**

- `src/globex_agent/domain/models.py`
- `src/globex_agent/catalog/local_catalog.py`
- `data/demo/products.jsonl`
- `data/demo/users.jsonl`
- `data/README.md`
- `tests/unit/test_models.py`
- `tests/unit/test_local_catalog.py`
- `examples/01_load_catalog.py`

---

### 阶段 2：确定性工具链 MVP

**目标**

不接 LLM，先证明业务工具本身能够从查询走到推荐结果。

**首批工具**

1. `ItemSearch`：关键词匹配、过滤和 Top-K。
2. `PriceCompare`：按标准商品归组并比较报价。
3. `ShippingCalc`：对 Top-N 估算运费、税费和到手价。
4. `ItemPicker`：执行预算、黑名单等硬约束，再综合排序。
5. `ShoppingSummary`：生成稳定的结构化结果和可读文本。

**任务**

- [x] 为每个工具定义清晰的输入输出 Schema。
- [x] 将纯业务逻辑和未来 Agent Tool 包装分开。
- [x] 固定工具顺序写一个离线 pipeline 示例。
- [x] 明确价格、运费、税费的计算假设和精度。
- [x] 对无结果、缺字段、未知平台和超预算实现明确返回。

**验收**

- 固定输入能够稳定得到相同输出。
- 最终推荐不得包含违反硬约束的商品。
- 比价和到手价可以用手算样例核对。
- 任一工具失败不会返回含糊的自由文本异常。

**阶段产物**

- `src/globex_agent/tools/item_search.py`
- `src/globex_agent/tools/price_compare.py`
- `src/globex_agent/tools/shipping_calc.py`
- `src/globex_agent/tools/item_picker.py`
- `src/globex_agent/tools/shopping_summary.py`
- `examples/02_deterministic_pipeline.py`
- 对应单元测试和一条集成测试

**第一个可演示里程碑**

运行一个命令，输入“预算 1500 元、通勤降噪耳机、不要入耳式”，输出 3 个以内的合规推荐、到手价和选择理由。

---

### 阶段 3：单 AgentLoop 最小闭环

**目标**

让 LLM 负责意图理解和工具编排，但仍只运行一个 AgentLoop。

**任务**

- [ ] 定义 `Planner` 和 `ChatFallback`。
- [ ] 建立统一模型客户端和配置入口。
- [ ] 建立 Prompt 文件与加载器。
- [ ] 注册当前已实现工具。
- [ ] 实现工具调用循环、最大轮数、总超时和终结性工具。
- [ ] 使用假的/脚本化模型测试路由，真实模型只做 smoke test。
- [ ] 记录每一轮模型消息、工具名、参数、结果摘要和结束原因。

**验收**

- 购物请求最终进入 `ShoppingSummary`。
- 非购物请求进入 `ChatFallback`。
- 参数不合法时能修正、澄清或安全结束。
- 达到最大轮数和超时时能降级结束。
- 自动化测试不依赖真实模型 API。

**阶段产物**

- `src/globex_agent/agent/llm.py`
- `src/globex_agent/agent/prompts.py`
- `src/globex_agent/agent/tool_registry.py`
- `src/globex_agent/agent/main_agent.py`
- `src/globex_agent/tools/planner.py`
- `src/globex_agent/tools/chat_fallback.py`
- `examples/03_single_agent.py`

---

### 阶段 4：离线检索基线与评测集

**目标**

在引入向量模型前建立可比较的检索基线，防止后续“感觉变好”。

**任务**

- [ ] 定义 `SearchBackend` 接口。
- [ ] 实现关键词/BM25 风格的本地基线。
- [ ] 从公开电商搜索数据抽取小规模查询—商品相关性集合。
- [ ] 按查询组划分 train/dev/test，防止相同查询泄漏。
- [ ] 实现 Recall@K、MRR@K、NDCG@K 和空召回率。
- [ ] 固定评测配置、随机种子和数据版本。

**验收**

- 一条命令可以从数据准备跑到评测报告。
- 重复运行得到一致结果。
- 报告中记录数据规模、切分方式、配置和真实指标。
- 后续召回实现可以在不改 `ItemSearch` 工具签名的情况下替换。

**阶段产物**

- `src/globex_agent/recall/base.py`
- `src/globex_agent/recall/keyword.py`
- `src/globex_agent/eval/recall_metrics.py`
- `scripts/data/prepare_esci_subset.py`
- `scripts/eval/run_recall_eval.py`
- `data/eval/recall_cases.jsonl`

---

### 阶段 5：Embedding 双通道召回、融合与 Reranker

**目标**

实现课程三塔思想的可运行离线版本，不从零训练大模型。

**第一版解释**

- Item 塔：离线编码商品标题、属性和类目。
- Query 塔：在线编码用户当前查询。
- User 塔：把长期偏好或历史正反馈编码成用户向量。
- 语义通道：Query 向量检索 Item 向量。
- 个性化通道：User 向量检索 Item 向量。
- 两路结果去重融合后，Reranker 对较小候选集精排。

**任务**

- [ ] 选择轻量预训练 Embedding，并记录选择理由。
- [ ] 构建和持久化小型本地向量索引。
- [ ] 实现 Query 和 Item 编码。
- [ ] 实现 User Profile 到用户向量的确定性构造。
- [ ] 实现双通道权重融合、去重和冷启动退化。
- [ ] 实现一个可替换的 Reranker 接口和轻量基线。
- [ ] 对比关键词、向量、融合、融合+精排四组结果。
- [ ] 记录准确率与延迟，不只记录最终一组。

**验收**

- 无用户历史时自动退化为语义召回。
- 用户偏好不能覆盖预算和黑名单等硬约束。
- 向量模型或索引不可用时可以回退到关键词基线。
- 评测报告包含消融对比和至少一个 bad case。

**阶段产物**

- `src/globex_agent/recall/embedding.py`
- `src/globex_agent/recall/user_profile.py`
- `src/globex_agent/recall/index.py`
- `src/globex_agent/recall/fusion.py`
- `src/globex_agent/recall/reranker.py`
- `scripts/index/build_item_index.py`
- `examples/04_retrieval_comparison.py`
- `output/eval/recall_report.json`（本地生成）

**暂缓项**

三塔监督训练、Hard Negative Mining、InfoNCE 改造和 Reranker 微调只保留为后续实验，不作为 MVP 阻塞项。

---

### 阶段 6：CategoryInsight 与本地 RAG

**目标**

让 Agent 在模糊品类需求下先获得可靠的选购知识，而不是直接盲搜商品。

**知识卡**

- 品类爆款与典型用途。
- 关键属性及属性之间的权衡。
- 价格带和常见陷阱。
- 适用人群、禁忌和更新时间。
- 数据来源和可信度。

**任务**

- [ ] 定义知识卡 Schema。
- [ ] 制作小规模品类知识卡。
- [ ] 建立字段标准化、切分、入库门禁。
- [ ] 实现关键词 + 向量混合召回。
- [ ] 对召回结果进行精排和答案压缩。
- [ ] 建立 RAG 标注集与 Recall@K、MRR、NDCG 评测。
- [ ] 实现空召回、低置信度和过期卡片提示。

**验收**

- 回答中的关键结论能追溯到知识卡。
- RAG 只提供选购知识，不伪造实时价格与库存。
- 加入 RAG 前后的检索或任务结果可对比。
- 知识卡更新不需要修改 AgentLoop。

**阶段产物**

- `src/globex_agent/rag/models.py`
- `src/globex_agent/rag/retriever.py`
- `src/globex_agent/tools/category_insight.py`
- `data/demo/category_cards.jsonl`
- `scripts/eval/run_category_recall.py`
- `examples/05_category_insight.py`

---

### 阶段 7：同质子 AgentLoop 与 fork 协同

**目标**

在单 AgentLoop 已稳定的前提下，加入课程中的按需 fork。

**fork 判断**

只在满足以下至少一项时 fork：

1. 子任务彼此独立且并行能缩短延迟。
2. 子任务会产生大量中间上下文，需要隔离。
3. 子任务内部预计需要至少三步工具调用。

单平台、单关键词、一步工具可完成的任务不 fork。

**任务**

- [ ] 实现 `dispatch_tool(demands)` 元工具。
- [ ] 主/子 Agent 共享同一模型、Prompt 和完整工具集。
- [ ] 子任务拥有独立 `thread_id` 和 checkpoint。
- [ ] 使用 `ContextVar` 传递父任务会话目录等必要上下文。
- [ ] 并行执行独立子任务并结构化合流。
- [ ] 加入 fork 深度、并发数、超时、最大轮数和取消传播。
- [ ] 加入工具结果截断和重复调用检测。
- [ ] 比较不开 fork、顺序 fork、并行 fork 的结果和延迟。

**验收**

- 跨平台或多子需求场景可以并发。
- 子 Agent 的消息历史不污染主 Agent。
- 子 Agent 超时或拒绝 fork 时主 Agent 能继续并降级。
- 禁止无限递归和无边界并发。
- 有自动化测试验证上下文隔离和异常合流。

**阶段产物**

- `src/globex_agent/agent/dispatch_tool.py`
- `src/globex_agent/agent/fork_guard.py`
- `src/globex_agent/agent/middleware.py`
- `tests/integration/test_fork_isolation.py`
- `examples/06_multi_agent_fork.py`

---

### 阶段 8：上下文治理与长期记忆

**目标**

让多轮会话不无限增长，并让稳定偏好可以跨会话复用。

**上下文策略**

- 稳定前缀保持不变以利用 Prompt Cache。
- Cache Breakpoint 之前是稳定历史，之后是当前动态后缀。
- 优先缩短冗长工具结果，再压缩更老的历史。
- 最近若干轮原文保留，摘要不能覆盖关键硬约束。

**记忆策略**

- 只保存稳定、可复用的用户偏好。
- 临时需求不写入长期 Store。
- 记忆包含来源、置信度、创建时间和更新时间。
- 用户可以查看、修改和删除记忆。
- 长期记忆负责偏好表达，User 塔负责把偏好转成召回信号，两者不混为一层。

**任务**

- [ ] 实现文件或 SQLite 版偏好 Store。
- [ ] 实现相关记忆检索和 Prompt 注入。
- [ ] 实现候选新偏好提取、确认和写回。
- [ ] 实现 Cache Breakpoint 和分层压缩。
- [ ] 对摘要遗漏、记忆冲突和错误偏好建立测试。
- [ ] 对长对话记录 token/字符规模变化。

**验收**

- 新会话能够读取已确认偏好。
- 临时预算不会被错误固化为长期偏好。
- 用户新声明与旧记忆冲突时，以当前显式输入优先。
- 长对话压缩后仍保留预算、禁忌和已选候选。
- Store 或压缩模型失败时能够继续完成当前任务。

**阶段产物**

- `src/globex_agent/memory/models.py`
- `src/globex_agent/memory/store.py`
- `src/globex_agent/memory/injector.py`
- `src/globex_agent/context/breakpoint.py`
- `src/globex_agent/context/compressor.py`
- `examples/07_memory_and_context.py`

---

### 阶段 9：完整评测与质量门禁

**目标**

把“项目可以运行”升级为“项目效果可解释、改动可回归”。

**四层评测**

| 层 | 关注点 | 主要指标 |
| --- | --- | --- |
| 数据 | 完整性、合法性、来源 | 错误率、缺失率、重复率 |
| 检索 | 相关商品能否进入候选 | Recall@K、MRR@K、NDCG@K、空召回率 |
| 工具 | 业务计算和约束是否正确 | 单元通过率、约束违例率、工具错误率 |
| Agent | 工具选择、顺序和最终答案 | 任务成功率、终结率、无效调用数、fork 合理性、Rubric 分数 |

**任务**

- [ ] 建立固定离线案例集，覆盖正常、边界和失败场景。
- [ ] 实现确定性规则评分。
- [ ] 实现工具轨迹校验。
- [ ] 在规则评分后增加可选 LLM Judge，并保留输入输出。
- [ ] 记录延迟、工具调用次数、token 和成本。
- [ ] 建立 baseline 与候选版本对比。
- [ ] 失败案例落到 bad-case 文件，不自动进入训练集。
- [ ] 设定回归门禁；阈值来自基线和实际目标，不照抄课程示例数字。

**最少场景**

- 单品类明确需求。
- 多平台比价。
- 模糊需求先做品类洞察。
- 有稳定偏好的个性化搜索。
- 预算冲突和黑名单。
- 空召回和低置信度。
- 子 Agent 超时。
- 非购物请求。
- Prompt Injection/恶意工具指令。

**验收**

- 一条命令生成机器可读 JSON 和可读 Markdown 报告。
- 报告包含数据版本、代码版本、配置和失败案例。
- 修改召回或 Prompt 后可以与基线自动对比。
- 简历中的所有数字都能从报告重现。

**阶段产物**

- `src/globex_agent/eval/cases.py`
- `src/globex_agent/eval/agent_metrics.py`
- `src/globex_agent/eval/rubric.py`
- `src/globex_agent/eval/report.py`
- `data/eval/agent_cases.jsonl`
- `scripts/eval/run_agent_eval.py`

---

### 阶段 10：FastAPI、WebSocket 与 AGUI

**目标**

把本地 Python 能力变成可调用服务，并让长任务过程可见、可取消。

**接口**

- `POST /api/tasks`：提交任务，立即返回 `thread_id`。
- `GET /api/tasks/{thread_id}`：查询状态与最终结果。
- `DELETE /api/tasks/{thread_id}`：取消任务。
- `WS /ws/{thread_id}`：推送标准事件。
- `GET /health`：健康检查。
- `POST /api/recommendations`：面向其他 Agent 的结构化推荐接口。

**事件**

- `task_started`
- `thinking`
- `tool_started` / `tool_finished`
- `fork_started` / `fork_finished`
- `warning` / `error`
- `task_finished` / `task_cancelled`

**任务**

- [ ] 实现请求/响应 Schema。
- [ ] 实现 `thread_id`、会话目录和 `active_tasks`。
- [ ] 实现 WebSocket 连接管理和断线重连。
- [ ] 实现后台任务、取消传播和清理。
- [ ] 将 Agent 事件映射为稳定协议。
- [ ] 编写一个最小命令行或静态页面客户端。
- [ ] 编写 API 与 WebSocket 集成测试。

**验收**

- HTTP 请求不会阻塞等待整个 Agent 完成。
- 两个并发用户的事件、记忆和输出不串台。
- 取消主任务会传播到子任务。
- WebSocket 断线不会导致 Agent 崩溃。
- API 文档可以直接调用完整推荐链路。

**阶段产物**

- `src/globex_agent/api/schemas.py`
- `src/globex_agent/api/context.py`
- `src/globex_agent/api/connection.py`
- `src/globex_agent/api/events.py`
- `src/globex_agent/api/server.py`
- `tests/integration/test_api.py`
- `tests/integration/test_websocket.py`

**第二个可演示里程碑**

客户端提交任务后实时看到“需求拆解 → 检索 → fork → 比价 → 运费 → 精挑 → 总结”，最后收到结构化商品推荐。

---

### 阶段 11：稳定性、可观测性与安全

**目标**

选择第 16～17 章中对作品集最有价值、且本地可验证的生产化能力。

**优先实现**

- [ ] 结构化日志和每次任务的 Trace ID。
- [ ] LLM、工具、fork 的耗时和错误统计。
- [ ] 请求级 token 预算、最大工具次数和模型降级。
- [ ] 工具超时、有限重试和熔断器。
- [ ] 请求幂等和重复提交保护。
- [ ] 工具白名单、角色边界和工具结果内容过滤。
- [ ] 日志中的密钥、用户标识和敏感内容脱敏。
- [ ] Middleware / Hook / Pipeline 的统一生命周期。
- [ ] 单步验证：关键阶段进入下一步前检查状态不变量。

**暂缓实现**

- LangFuse 自部署。
- Redis 共享熔断状态。
- 请求优先级队列。
- 多实例部署。
- K8s、灰度发布和 GPU 推理。

**验收**

- 故意让一个工具超时，任务仍能给出明确降级结果。
- 超出预算会终止继续搜索并收敛回答。
- 非法工具名和工具返回中的恶意指令会被拒绝。
- 单次任务可以从日志还原关键决策链。

---

### 阶段 12：容器化与可复现演示

**目标**

在本地服务稳定后提供轻量、可复现的交付方式。

**任务**

- [ ] 为 API 构建单服务 Dockerfile。
- [ ] 配置健康检查、非 root 用户和环境变量注入。
- [ ] 如确有第二个本地服务，再加入 Docker Compose。
- [ ] 写一组从构建、启动到验证的 SOP。
- [ ] 建立最小演示脚本和截图/录屏素材。

**验收**

- 新机器按文档可以启动服务并执行 smoke test。
- 镜像不包含 `.env`、原始数据和本地输出。
- 容器停止时能完成任务取消与资源清理。

---

### 阶段 13：与客服 Agent 联动

**目标**

让另一个客服 Agent 在用户询问“能否推荐商品”时调用本项目，而不是复制检索逻辑。

**边界**

- Globex 负责检索、比价、运费、精挑和推荐解释。
- 客服 Agent 负责订单、物流、退款、售后和对话路由。
- 两个项目只通过稳定接口通信，不共享内部 Agent 状态。

**推荐请求**

```json
{
  "query": "预算 1500 元的通勤降噪耳机",
  "user_id": "demo-user-001",
  "constraints": {
    "budget": 1500,
    "currency": "CNY"
  },
  "top_k": 3
}
```

**推荐响应**

```json
{
  "status": "ok",
  "recommendations": [],
  "assumptions": [],
  "warnings": [],
  "trace_id": "..."
}
```

**任务**

- [ ] 先做 Python 函数调用示例。
- [ ] 再通过 `POST /api/recommendations` 集成。
- [ ] 如面试展示需要，再增加 MCP Tool 包装。
- [ ] 设置超时、最大步数、错误码和降级话术。
- [ ] 编写消费者契约测试。

**验收**

- 客服 Agent 不需要理解 Globex 内部工具即可调用推荐能力。
- Globex 超时时客服 Agent 能继续处理其他客服问题。
- 响应中明确区分数据事实、估算和模型生成理由。

---

### 阶段 14：第 17～19 章增强路线

这一阶段不阻塞简历版本，只在前面的评测能够指出真实问题后实施。

- Harness：动态工具权限、对话阶段状态机、Hook 和 Silent Drift 检测。
- Bad-case 飞轮：自动采集、去重、分级和人工审核。
- Prompt 治理：稳定前缀、版本号、变更记录和 A/B 对比。
- 记忆治理：冲突解决、过期策略和成功偏好沉淀。
- Skill：仅在工具数量和 Prompt 体积确实造成问题时做渐进加载。
- 模型训练：只有高质量轨迹达到足够规模后，才评估 SFT/RL。

任何“自进化”在第一版都必须经过人工审核，不能让 Agent 自动修改生产 Prompt 或训练数据。

## 7. 实施顺序与依赖

```mermaid
flowchart TD
    P0["0 工程基线"] --> P1["1 Schema 与数据"]
    P1 --> P2["2 确定性工具链"]
    P2 --> P3["3 单 AgentLoop"]
    P1 --> P4["4 检索基线"]
    P4 --> P5["5 向量召回与精排"]
    P3 --> P6["6 CategoryInsight RAG"]
    P5 --> P6
    P3 --> P7["7 同质 fork"]
    P5 --> P7
    P7 --> P8["8 上下文与记忆"]
    P6 --> P9["9 完整评测"]
    P8 --> P9
    P9 --> P10["10 API 与事件流"]
    P10 --> P11["11 稳定性与安全"]
    P11 --> P12["12 容器化"]
    P10 --> P13["13 客服 Agent 联动"]
    P9 --> P14["14 Harness / 自进化 / Prompt"]
```

不能为了“看起来完整”跳过前置阶段：

- 没有阶段 2，就无法判断 Agent 错还是工具错。
- 没有阶段 4，就无法证明 Embedding 和 Reranker 是否改进。
- 没有阶段 9，就不应在简历中写效果提升百分比。
- 没有本地稳定 API，就不应先做 Docker、K8s 或跨项目集成。

## 8. 测试策略

### 8.1 单元测试

- Schema 校验。
- 数据清洗和去重。
- 价格、币种、运费、税费计算。
- 硬约束过滤和排序。
- 融合、去重和指标计算。
- fork 深度、循环检测和结果截断。
- 记忆冲突与上下文压缩。

### 8.2 集成测试

- 本地目录 → 检索 → 比价 → 精挑 → 总结。
- AgentLoop → 多工具调用。
- 主 Agent → 子 Agent → 合流。
- 记忆读取 → Prompt 注入 → 写回。
- API → 后台任务 → WebSocket 事件。

### 8.3 端到端测试

- 使用固定用户、固定数据和可控模型响应跑完整链路。
- 真实模型只作为单独 smoke/evaluation profile，避免普通测试产生费用和随机失败。

### 8.4 故障注入

- 空数据。
- 非法模型输出。
- 工具超时。
- 子 Agent 超时。
- 向量索引不存在。
- Store 不可用。
- WebSocket 断开。
- 超出 token 与工具调用预算。

## 9. 每阶段 Definition of Done

一个阶段只有同时满足以下条件才标记完成：

1. 能用一句话说明它解决的问题。
2. 对应代码已经实际运行。
3. 相关自动化测试通过。
4. 有独立示例及运行命令。
5. 记录了输入、输出、错误和解决办法。
6. 至少记录一个设计取舍和一个失败模式。
7. 指标来自可复现脚本。
8. 更新 README 当前状态。
9. 按用户要求再更新 Obsidian 阅读进度与学习日志。
10. 整理面试解释：为什么这样做、替代方案、失败模式、代码位置。
11. 创建清晰的 Git 提交。

## 10. 简历版本的最低完成线

达到以下条件即可形成第一版可投递项目，不需要等待第 16～19 章全部完成：

- [ ] 完成阶段 0～10。
- [ ] 至少有一份真实检索消融报告。
- [ ] 至少有一份 Agent 端到端评测报告。
- [ ] 至少演示一个多 Agent fork 场景。
- [ ] 至少演示一个长期偏好影响召回/精挑的场景。
- [ ] API 和 WebSocket 可本地运行。
- [ ] 有清晰 README、架构图、演示命令和限制说明。
- [ ] 所有简历数字均能从脚本重现。

可用于简历的能力表述方向，最终数字必须在项目完成后填写：

> 设计并实现离线电商搜索 Agent，构建主 AgentLoop 与按需 fork 的同质子 Agent 协同机制；完成 Query/User 双通道召回、候选融合与 Reranker 精排，并通过固定评测集对 Recall@K、MRR、NDCG 和任务成功率进行回归；基于 FastAPI 与 WebSocket 实现异步任务、实时事件流、取消和上下文隔离。

## 11. 推荐节奏

不按日历硬赶，按可验证里程碑推进：

| 里程碑 | 包含阶段 | 结果 |
| --- | --- | --- |
| M1 可运行离线工具链 | 0～2 | 不用模型也能完成推荐 |
| M2 单 Agent MVP | 3 | 模型能够正确编排工具并收敛 |
| M3 搜索质量版 | 4～6 | 有向量召回、精排、RAG 和对比指标 |
| M4 多 Agent 完整版 | 7～9 | 有 fork、记忆、上下文治理和评测 |
| M5 可展示服务版 | 10～12 | 有 API、事件流、稳定性和可复现部署 |
| M6 双项目联动版 | 13 | 客服 Agent 可调用推荐能力 |
| M7 进阶研究版 | 14 | 按真实 bad case 增加 Harness/Prompt/训练能力 |

## 12. 下一步工作

阶段 0～2 已完成实际验收。下一步只进入阶段 3：在已经验证的五工具业务链之上实现单 AgentLoop、Planner、ChatFallback 和脚本化模型测试；仍不提前引入同质子 Agent fork、向量库或 API。
