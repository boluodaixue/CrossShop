# 给新 Agent 的同步提示词

> 用途：新开一个 Agent 会话时，把下面代码块整段粘贴给它，即可快速同步本项目进度与已定方案。
> 权威方案文档：`docs/globex对比分析与合并方案.md`（本文档是「索引」，那份是「正文」）。

---

```text
你正在协助我推进「Globex 电商搜索 Agent」这个简历/面试项目。请先按下面的顺序读文件，快速同步我的进度和已定方案，再开始任何工作。读完先复述你对现状和方案的理解，等我确认后再动手写代码。

## 一、必读文件（按顺序）

1. `D:\PycharmProjects\GlobexAgentLearning\AGENTS.md` —— 项目规则（目录约定、实现边界、验证要求），必须遵守。
2. `D:\PycharmProjects\GlobexAgentLearning\docs\globex对比分析与合并方案.md` —— 本项目的「对比分析 + 合并方案」定稿文档，是当前最权威的方案说明，务必精读。
3. 需要看具体代码时再读：
   - 课程文档（只读，规范设计来源）：`D:\刘诗恩\Obsidian\电商搜索Agent资料`
   - 参考仓库（AgentScope 2.0 实现，工程模式素材）：`D:\PycharmProjects\globex-agent-main`
   - 本仓库现有实现（LangGraph 课程复现）：`D:\PycharmProjects\GlobexAgentLearning\src\globex_agent\`

## 二、背景速览（三份来源的关系）

- 「Globex」是一套跨境电商对话式购物 Agent，课程作者「会敲代码的泡」用 LangGraph/LangChain 教学。
- `globex-agent-main` 是另一个人用 **AgentScope 2.0 + DDD 洋葱架构**做的「同概念、换框架、做减法」的重实现，README 自述「对齐教程口径」。它共享课程的概念骨架，但在框架/分层/召回/多 Agent/工具上做了不同取舍。
- 我（项目主人）已有一个 LangGraph 课程复现仓库 `GlobexAgentLearning`，已有 9 工具 + BGE-M3 双塔 + Faiss + OpenSearch Hybrid + 召回评测。
- 目标是把两者合并成**我自己的简历/面试项目**，方案已定稿在 `docs/globex对比分析与合并方案.md`。

## 三、已定方案（9 项决策，不可随意更改）

1. 框架：**LangGraph**（`create_react_agent`）
2. 多 Agent：**异构子 Agent**（search_agent / trade_agent），主 Agent 持全量工具「单干优先」
3. 召回与向量/知识库：**BGE-M3 双塔 + Faiss（商品召回）+ OpenSearch Hybrid（品类 KB）**，保留现有实现
4. 工具集：**仓库的 6 工具形式**（product_search 内联 landed_price + 订单三工具 + remember_preference + task_dispatch）
5. 分层结构：**DDD 洋葱四层 + 装配根**（以参考仓库为主）
6. 韧性工程：熔断/限流/语义缓存/队列削峰/幂等（以参考仓库为主）
7. 事件总线/前后端：**参考仓库的 12 事件 + FastAPI + React**
8. 评测：端到端 **cases.yaml + LLM judge P0/P1/P2**（参考仓库）+ 模块级**召回门禁 Recall@K/MRR/NDCG**（现有）；SFT/RL 不落地
9. 部署：参考仓库 compose 结构，**qdrant 换成 opensearch**（本机 Docker 2.19.1 + IK 已自部署）

## 四、关键认知（避免误判）

- 参考仓库里约 80% 是「纯 Python / FastAPI」，只有 agent 编排 + 工具外壳 + 中间件那一层强依赖 AgentScope（详见方案文档 §6）。迁移时：**纯 Python 核心直接搬，框架薄壳用 LangGraph 对应物重写**。
- 现有 9 工具（item_search/price_compare/shipping_calc/item_picker/shopping_summary/planner/chat_fallback）**不进主线**，归档为「9 工具 vs 6 工具」对照与面试谈资，不要删。
- 现有召回内核（BGE-M3/Faiss/OpenSearch Hybrid/召回评测）是**要保留的核心资产**，迁移时只换外壳，不动内核。

## 五、迁移顺序与当前进度

目标目录（仍在 `src/globex_agent/` 下）：`domain/ application/ infrastructure/ presentation/ composition.py worker.py`

顺序：**domain → infrastructure → application → presentation/composition → 评测/部署**（详见方案文档 §7，含每个文件的来源标注「搬/保留/新写」和验收标准）。

**当前进度：代码迁移已完成首轮，Phase 0-5 均有提交；search-budget / no-fabrication / long-context-memory 回归已通过，order-full-cycle 已修复并单独重跑通过；事件流、熔断、重启恢复与并发会话隔离测试已补齐。Docker daemon 恢复后还需完成容器 build/up 验证。**

## 六、工作纪律

1. 每阶段结束跑 `pytest` 全绿再进下一阶段，不做一次性整体 rename。
2. 课程文档只读，不复制课程 Markdown/图片进本仓库。
3. 修改 Python 后至少跑语法检查或相关测试，报告实际执行的命令和结果，不把「代码已生成」当「已跑通」。
4. 忠实执行已定方案，不擅自重构、不替换框架、不扩大范围；有不同意见先提出来讨论，不要直接改方案。

## 七、你的任务

（在此填入具体任务；若为空，则：读完上述文件后，用你自己的话复述一遍「现状 + 方案 + 下一步」，并指出方案文档里你发现的任何不一致或风险，等我确认。）
```
