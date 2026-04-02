# Globex Agent Learning

用于学习、运行并逐步组装“电商搜索 Agent”课程中的关键模块。

完整实施路线见 [ENGINEERING_PLAN.md](ENGINEERING_PLAN.md)。

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
4. Embedding、混合召回与 Reranker
5. 上下文压缩与长期记忆
6. AGUI、WebSocket 与 FastAPI
7. Globex 主链路组装
8. 评测、部署思想与面试复盘

## 当前状态

- [x] 初始化独立项目目录
- [x] 建立 Codex 项目规则
- [x] 关联课程资料路径
- [x] 阅读课程资料至第 14 章
- [x] 建立完整工程计划
- [x] 完成工程阶段 0：环境与测试基线
- [x] 完成工程阶段 1：领域模型与小型商品数据
- [x] 完成工程阶段 2：确定性工具链 MVP
- [ ] 运行第一个 AgentLoop 示例

## 快速开始

项目固定使用 Python 3.10.20 和 uv。Windows PowerShell 下执行：

```powershell
python -m pip install --user uv
python -m uv python install 3.10.20
python -m uv sync
$env:PYTHONUTF8 = "1"
python -m uv run python examples/00_smoke.py
python -m uv run python examples/01_load_catalog.py
python -m uv run python examples/02_deterministic_pipeline.py --case 1
python -m uv run pytest
python -m uv run ruff check src tests examples
```

若当前终端还没有刷新 `uv` 的 PATH，使用上面的 `python -m uv` 即可。受限执行环境若无法访问用户级 uv 缓存，可直接调用已同步的虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests examples
```

## 已实现模块

- `LocalCatalog`：逐行校验 JSONL，报告错误，拒绝重复报价、主数据冲突和价格异常值。
- 演示数据：36 条模拟平台报价，归组为 12 个标准商品，覆盖耳机、背包、机械键盘及 3 个模拟平台。
- 确定性工具链：`ItemSearch → PriceCompare → ShippingCalc → ItemPicker → ShoppingSummary`。
- 验证结果：Python 3.10.20；Ruff 通过；34 个 pytest 测试通过；3 个独立示例实际运行成功。

阶段 2 的计算口径：

- ItemSearch 使用本地字符词项、结构化属性和可选用户画像打分，随后稳定 Top-K。
- PriceCompare 使用 `Decimal` 和版本化演示汇率，仅比较商品标价。
- ShippingCalc 使用商品报价中的模拟运费，并按平台演示税率估算 `到手价 = 商品价 + 运费 + 税费`。
- ItemPicker 先执行预算、材质、平台和属性硬约束，再按相关性、到手价、评分与偏好排序。
- ShoppingSummary 使用固定模板收敛，不调用模型，并明确披露模拟数据与估算规则。

当前依赖保持最小：

| 依赖 | 用途 | 可选替代 |
| --- | --- | --- |
| `pydantic` | 统一商品、报价、用户、检索、比价、运费和推荐数据契约 | 标准库 dataclass + 手写校验，但错误定位和 JSON 解析成本更高 |
| `pytest` | 参数化测试、夹具和清晰失败报告 | 标准库 `unittest` |
| `ruff` | 一次完成导入排序与静态规范检查 | Flake8 + isort 等组合 |

## 阶段 2 面试解释

| 模块 | 为什么这样设计 | 替代方案 | 主要失败模式与代码位置 |
| --- | --- | --- | --- |
| ItemSearch | 先建立可复现关键词基线，后续才能证明向量召回是否改进 | BM25、Embedding、混合召回 | 同义词和跨语言召回弱；见 `src/globex_agent/tools/item_search.py` |
| PriceCompare | 同款报价归组，金额统一用 Decimal，避免浮点误差 | 实时汇率服务 | 演示汇率不代表实时值；见 `src/globex_agent/tools/price_compare.py` |
| ShippingCalc | 与标价比较分离，允许标价 Top-N 后再估算到手价 | HS Code + 原产地税费服务 | 税率和时效是简化规则；见 `src/globex_agent/tools/shipping_calc.py` |
| ItemPicker | 硬约束先过滤，偏好只能影响排序，不能覆盖预算和黑名单 | Learning-to-Rank 或 LLM Reranker | 未知硬约束会安全停止；见 `src/globex_agent/tools/item_picker.py` |
| ShoppingSummary | 模板输出稳定、可测、零模型成本 | LLM 生成摘要 | 表达较固定；见 `src/globex_agent/tools/shopping_summary.py` |

当前不需要模型密钥，也没有引入 LangChain、OpenSearch、Faiss、vLLM 或 K8s。下一步是阶段 3：单 AgentLoop 最小闭环。
