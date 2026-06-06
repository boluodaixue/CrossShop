# 文档导航（current）

本文档是仓库文档入口。状态含义：current=当前实现/契约；historical=保留实验或排障证据；design=已记录但未完整实现的设计；proposal=候选方案，不能当作已实现能力。

## 当前权威文档

| 文档 | 状态 | 用途 |
| --- | --- | --- |
| ARCHITECTURE_AND_IMPLEMENTATION.md | current | 在线生产链、离线质量链、模块边界与实现说明 |
| PROJECT_STATUS_AND_ROADMAP.md | current | 项目完整性审计、依赖关系和下一主阶段验收 |
| PROJECT_AUDIT_AND_REMEDIATION.md | current + historical sections | 问题修复、历史失效评测和非阻塞技术债 |
| experiments/flow_query_eval_20260820.md | current + historical sections | 8 Flow 权威评测及历史中间结果边界 |
| data/catalog_faiss_schema_v2_20260820.md | current | 商品索引 schema-v2、分区和运行契约 |
| data/catalog_schema_v2_migration_20260820.md | current | schema-v2 迁移记录与验证 |
| data/category_card_data_contract.md | current | CategoryInsight 卡片、provenance 和价格范围口径 |
| data/multiplatform_catalog.md | current | 多平台商品目录与构建边界 |
| data/bilingual_v3_data_contract.md | current | 双语数据契约和字段边界 |
| data/challenge_v4_data_contract.md | current | challenge 数据契约 |

## 架构、同步和方案记录

| 文档 | 状态 | 何时阅读 |
| --- | --- | --- |
| AGENT_SYNC.md | current | 需要与主 Agent/课程进度同步时 |
| architecture_migration_handoff.md | design | 复核 LangGraph + DDD 迁移决策和交接约束时 |
| globex对比分析与合并方案.md | historical/design | 了解合并取舍和未采纳方案时；不作为当前实现入口 |

## 实验和历史资料

| 文档 | 状态 | 用途 |
| --- | --- | --- |
| data/category_card_data_contract.md | current | 当前 CategoryInsight 数据事实边界 |
| data/shopsimulator_retrieval_v1.md | historical | ShopSimulator 检索实验数据来源 |
| data/shopsimulator_variant_identity_audit_20260820.md | historical | 变体身份排查证据 |
| experiments/phase5_bge_m3_retrieval_2026-08-15.md | historical | 阶段五召回/精排实验 |
| experiments/phase6_bilingual_v3_2026-08-15.md | historical | 双语知识卡实验 |
| experiments/phase6_category_insight_2026-08-15.md | historical | CategoryInsight 初版实验 |
| experiments/phase6_category_insight_v2_2026-08-15.md | historical | CategoryInsight v2 实验 |
| experiments/phase6_challenge_v4_2026-08-15.md | historical | challenge v4 实验 |
| experiments/phase6_taobao_zh_2026-08-18.md | historical | 淘宝中文底座实验 |

历史文档保留是为了复现取舍和失败模式；若与 current 文档或权威 output 报告冲突，以 current 文档、代码和可复现实测为准。外部 Judge 的历史 HTTP 500、旧 0/7 与旧 18 项 P0 不得作为当前质量结论。

## 推荐阅读顺序

1. ARCHITECTURE_AND_IMPLEMENTATION.md：先建立在线/离线边界。
2. PROJECT_STATUS_AND_ROADMAP.md：了解哪些模块已进入 src、哪些仍是设计。
3. data/catalog_faiss_schema_v2_20260820.md 和 data/category_card_data_contract.md：核对商品与品类事实契约。
4. experiments/flow_query_eval_20260820.md：读取当前 8 Flow 权威基线。
5. historical 文档：需要追溯实验或排障时再读。

当前项目主干缺口是课程级可交付运行闭环：前端/API 任务协议、全栈模型服务编排和最小运行观测尚未统一验收。在线 Fact Guard/rewrite/fallback、P0 硬门禁、CategoryInsight 状态机和 cache evidence 刷新均是非阻塞细节增强。下一主阶段及验收以 PROJECT_STATUS_AND_ROADMAP.md 为准。
