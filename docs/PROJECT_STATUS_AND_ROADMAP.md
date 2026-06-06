# 项目状态与路线图（current）

本文档以当前代码、测试和权威评测文件为准，不把课程设计或历史实验当作已实现能力。

## 审计结论

| 一级模块 | 当前状态 | 证据入口 |
| --- | --- | --- |
| 用户端产品能力 | partial | FastAPI/WebSocket、前端、搜索和订单接口已接入；在线最终回答安全闭环尚未接入 |
| Agent 业务闭环 | current/partial | application/agents、orchestrator、工具和 checkpoint 已组成主链；调用顺序仍受 LLM 工具规划影响 |
| 商品搜索 | current | Faiss 四分区 schema-v2、Query/Item BGE-M3、Top-100、Reranker Top-10 |
| CategoryInsight | current/partial | OpenSearch/RAG、CPU encoder、价格档位/属性/bestseller 有界暴露；是可选参考工具，不是商品硬过滤/排序器 |
| 订单链路 | current | Prepare → trusted Confirm 两阶段门禁，结构化 API 才能提交 token |
| 前后端/API | current/partial | FastAPI /health、commerce intents、WebSocket 和订单路由已存在；完整生产部署编排仍不完整 |
| 数据与索引生命周期 | current/partial | StandardItem/schema-v2、Faiss 和 CategoryInsight 资料有契约；自动增量重建与生产生命周期治理仍有限 |
| 部署与服务编排 | partial | 本机启动脚本、Redis/OpenSearch compose、独立模型 worker 已有；课程部署章节中的 vLLM/K8s 全套不是当前实现 |
| 可观测性与故障恢复 | partial | audit/conversation、缓存、checkpoint、韧性组件存在；完整 LangFuse/队列恢复和端到端告警未闭环 |
| 质量与安全 | current/partial | Snapshot、PII/载荷边界、Fact Guard、稳定 ID Judge 和 8 Flow 基线已完成；在线 final-answer gate 仍缺失 |

examples 目录是可独立运行的课程/实验示例，不等同于 src 中的完整运行时；当前项目主链以 src/globex_agent、scripts 和 tests 为准。

## 已完成

- StandardItem → ProductFactSnapshot → ProductCard 的单向证据链；Snapshot 有 schema version、hash、evidence ID、provenance 和 exposed facts。
- ProductSearch 在同一次构造中生成 Card 与 Snapshot；audit 仅保留白名单、有界且脱敏的 CategoryInsight/Snapshot 信息。
- CategoryInsight 品类知识卡检索与范围标识；品类价格档位不会被误当作 SKU 精确价格。
- Fact Guard 的变体价格绑定、别名保守匹配、范围语境、预算/运费排除和重叠 claim 去重。
- stable evidence catalog、criterion ID 双向覆盖校验、pass/fail/not_evaluable 语义和旧格式读取兼容。
- 本机 4GB 配置：CPU FP32 query embedding、GPU FP16 reranker batch 16；生产仍支持 embedding GPU + reranker GPU。
- 权威评测：确定性 Fact Guard 8/8、P0=0；Judge 8/8、平均 0.99125、适用 P0 20/20；权威文件在 output/eval/flow-rubric-luna-final-20260822.md 与同名 JSON。

## 部分完成与非阻塞技术债

- 没有在线 final answer Fact Guard、grounded rewrite 或 deterministic fallback。
- P0=0 是离线评测执行规范，不是在线生产状态机不可绕过的硬门禁。
- CategoryInsight 的调用顺序依赖 LLM prompt；代码没有把它强制成每次商品检索前置步骤。
- semantic cache 可能复用旧结果，绕过当轮证据刷新；需要完整版本指纹/失效策略。
- CategoryInsight components 在线/审计存在，但 Judge 覆盖仍需持续核对 exposed evidence 是否完整。
- 课程部署/观测章节中的 vLLM、LangFuse、K8s、完整 token/circuit/queue 运维能力不应写成当前已交付。

这些问题不阻塞当前证据链基线，也不是本轮提交的理由之外的功能扩展。

## 未实现与非目标

未实现：在线 rewrite/fallback 闭环、自动化增量索引生命周期、完整生产级模型服务编排、完整外部观测平台接入。非目标：修改商品 schema-v2、重建既有 Faiss 索引、改变 Top-100 或 Top-10、量化模型、替换 Redis checkpoint/订单门禁、修改课程原文。

## 依赖关系与推荐唯一下一主阶段

推荐下一主阶段：在线 grounded answer 安全闭环（先 shadow mode，后决定默认开启）。

理由是商品证据链、exposed facts 和离线 Guard 已稳定，在线回答目前仍是唯一明显的可信度断点；它可以在不改变召回、商品 schema、订单门禁和生产默认设备分配的前提下独立验收。该阶段不是当前基线的前置依赖，而是证据链完成后的后续增强。

依赖图：

    ProductFactSnapshot + CategoryInsight exposed facts
      → final-answer context contract
      → draft answer
      → online Fact Guard
      → pass return
      → fail: one bounded grounded rewrite
      → re-check
      → fail again: deterministic fallback

主要入口：

- 证据：src/globex_agent/application/evidence.py
- 编排与回答落点：src/globex_agent/application/agents/orchestrator.py
- 商品工具：src/globex_agent/application/tools/product_search_tool.py
- 确定性规则：src/globex_agent/eval/fact_guard.py
- 评测参考：scripts/eval_flow_rubric.py
- 服务组装：src/globex_agent/composition.py

## 下一阶段可验收标准

1. 用 feature flag 接入 final-answer guard，默认 shadow mode，不改变现有生产返回语义。
2. Guard 只消费当前 Snapshot exposed facts 和 CategoryInsight 明确 scope 的事实；价格必须同一 item/evidence/variant 绑定。
3. 失败最多执行一次 grounded rewrite；rewrite 后再次运行同一 Guard，禁止无限循环。
4. 二次失败返回 deterministic fallback，并在 audit/conversation 中记录 guard、rewrite 和 fallback 的原因及 evidence IDs。
5. 增加无证据、错误变体价、CategoryInsight 越界、PII、rewrite 仍失败的单测/集成测；历史 8 Flow 的确定性 P0 仍为 0。
6. 运行完整 pytest、Ruff、diff check 和不依赖外部 Judge 的 8 Flow 回归；再以明确授权决定是否重跑 Judge。

后续候选但不应抢占下一主阶段：semantic cache freshness、队列/故障恢复与完整观测、生产模型服务编排、索引增量治理。
