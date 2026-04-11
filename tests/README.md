# 测试目录

测试与当前已经学习并实现的模块同步增加，不提前为未实现功能搭建空框架。

当前覆盖：

- 包导入 smoke test。
- 金额、币种、ID、时区和到手价等领域约束。
- 36 条演示商品、3 个演示用户与 6 条演示查询的完整加载。
- 跨平台 `same_group_id` 只关联不合并，以及同平台同款清洗。
- 非法行、重复 `item_id` 和价格异常值的错误报告。
- 每个平台独立 `ItemSearchOutput` 的 Top-K 和结构化空召回。
- PriceCompare 合流、Top-N 剪枝、固定汇率换算和最低标价。
- ShippingCalc 的手算到手价、标价赢家变化和未知目的地。
- ItemPicker 的预算、材质、属性硬约束和未知约束安全停止。
- ShoppingSummary 的稳定输出、空结果解释和完整管线回归。
- 阶段三工具注册、模型不可见依赖、YAML 提示词和终结工具规则。
- 自由文本 Planner 的结构化输出、`runtime.request` 更新、平台默认值和空容器漂移归一化。
- 脚本化单 AgentLoop 的 18 条消息、8 次工具调用、非购物兜底和同 thread 多轮检查点。
- `SearchBackend` 的稳定排序、空召回和 BM25 可复现性。
- Recall@K、MRR@K、NDCG@K 与空召回率的手算结果。
- ESCI 封闭评测子集的规模、全量已标注候选、每 query 2～5 个 Exact 正例和 21/6/8 切分。
- BGE-M3 与 mE5 的模型专属输入前缀、精确余弦基线、Faiss HNSW/IP 持久化和封闭池子索引。
- 项目主环境与外部CUDA Python之间的常驻 JSONL Reranker 进程协议。
- 作为扩展消融的 BM25/语义 min-max 加权融合、去重和语义异常回退。
- 商品主链请求 Query/Item 向量 Top-100 后执行可替换 Reranker Top-10，失败时保留向量粗排顺序。
- runtime canonical fallback 会 overfetch，并保证重复 canonical 不占 Top-K、parallel 只展示一次、en_only 不因语言偏好丢失，展示语言不改变相关性分数或顺序。
- CategoryInsight 默认停在动态 Hybrid；Reranker 只有显式注入时运行，失败时保留 Hybrid 粗排。

```powershell
python -m uv run pytest
```

2026-08-15 实际结果：68 个测试全部通过。课程使用的 `create_react_agent` 在 LangGraph 1.x 下属于已知弃用 API；测试配置仅过滤这一条已知警告，不隐藏其他警告。
