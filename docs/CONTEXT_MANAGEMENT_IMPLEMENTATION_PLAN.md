# Globex 上下文管理实现方案（design）

> 日期：2026-08-24  
> 状态：design，尚未实现  
> 依据：当前代码、课程第 05 章和《05 Cache Breakpoint 文档冲突与 Prompt 结构》；出现冲突时以本文确认的实现口径为准。

## 1. 已确认的最终口径

上下文管理分成三件正交的事：

1. 四层上下文决定信息是什么、以什么形式进入模型。
2. L0-L4 决定体积治理和状态维护职责。
3. Cache Breakpoint 决定最终 Prompt 中哪一段保持稳定、哪一段允许变化。

L0-L4 的职责固定为：

| 层级 | 职责 | 触发方式 |
| --- | --- | --- |
| L0 | 限制工具入口体积 | 每次工具执行 |
| L1 | 检查总 Token 预算，只做测量和调度 | 每次调用模型前 |
| L2 | 生命周期驱动，整理动态区中已经 settled 的内容 | 工具交互完成或一轮结束后 |
| L3 | 总上下文达到阈值时，压缩旧冻结历史 | L1 判定总输入超出软阈值时 |
| L4 | 持续维护结构化会话状态 | 请求开始和一轮结束时增量更新 |

L1 不是压缩；L4 也不要求本身产生摘要。L2 不再使用 Token 阈值触发，Token 阈值只决定是否需要 L3。

## 2. 四层上下文与 Prompt 物理顺序

### 2.1 信息结构

| 层 | 内容 | 生命周期 |
| --- | --- | --- |
| 固定层 | System Prompt、工具定义、安全规则 | 版本变更时更新 |
| 状态层 | L4 的当前请求、已执行搜索条件、推荐结果、订单状态 | 每轮增量覆盖 |
| 历史层 | 已冻结的精简交互、阶段摘要、必要的近期对话 | settled 后冻结；L3 可压缩旧段 |
| 当前层 | 当前请求、尚未消费的工具结果、未完成工具调用 | 本轮完成后转为 settled |

完整 Trace、完整工具原始结果和审计记录属于冷数据，不是第五个 Prompt 层，默认不进入模型。

### 2.2 Prompt 顺序

信息分层不等于 Prompt 排列顺序。状态层会频繁变化，因此必须放在 Cache Breakpoint 后：

```text
固定 System Prompt + 工具定义
已冻结阶段摘要 + 已冻结精简历史
================ Cache Breakpoint ================
L4 结构化状态快照
当前仍 active 的消息/工具交互
当前用户请求
```

当前模型网关未接入显式 `cache_control`，第一版先实现逻辑断点和稳定前缀；只有拿到供应商真实 cache usage 后，才宣称缓存命中能力。

## 3. 当前代码现状

### 已有基础

- LangGraph Redis Checkpointer 持久化完整 `messages`，支持主 Agent 和子 Agent 恢复。
- `SubmitIntentInput` 已有 `raw_query / locale / currency / buyer_id / session_id`。
- 所有主要工具通过 `tool.invoke / tool.result` 发布结构化参数和结果。
- `ConversationStore` 已保存用户/Agent 对话和一轮内的过程事件，可作为审计与冷数据来源。
- 商品搜索默认返回 Top-K=5，已经具备部分 L0 意义上的输出控制。

### 当前缺口或冲突

- `LangGraphAgent.compress_context()` 只按消息数量保留首条和尾部消息；可能切断 Assistant tool-call 与 Tool result 的协议对。
- `CONTEXT_SIZE` 被粗略换算成 `context_size // 1000` 条消息，不是真实 Token 预算。
- `TOOL_RESULT_LIMIT`、`REPLY_TOKEN_BUDGET` 已配置但没有形成统一预算链路。
- 尚无 L4 状态模型、生命周期分类、冻结游标、阶段摘要和 Prompt 组装器。
- EventBus 的 `context.compressed` 只有事件名，没有对应的完整上下文治理实现。

旧的按消息数截断应在新方案稳定后删除，不能与新 L2/L3 并存。

## 4. L4：只归集 Agent 链路已有信息

L4 第一版不新增意图提取模型，也不与长期记忆、在线哨兵或 Langfuse 耦合。它只把请求和 Agent 已经产生的结构化事件投影成会话快照。

### 4.1 建议状态结构

```json
{
  "schema_version": "session-context-v1",
  "revision": 12,
  "session": {
    "shopping_session_id": "...",
    "buyer_id": "...",
    "locale": "zh-CN",
    "currency": "CNY"
  },
  "request": {
    "session_origin_raw_query": "预算500元，寄到美国",
    "current_raw_query": "第二个商品怎么样"
  },
  "last_search": {
    "tool_call_id": "product-search-...",
    "args": {
      "normalized_query": "...",
      "ship_to": "US",
      "price_max_major": 500,
      "target_currency": "CNY",
      "top_k": 5
    },
    "status": "succeeded",
    "returned_count": 5,
    "item_refs": ["item-1", "item-2"]
  },
  "last_recommendation": {
    "verification_status": "supported",
    "items": []
  },
  "order": null,
  "updated_at": "..."
}
```

字段按实际事件出现再写入，不为缺失信息造默认事实。

### 4.2 数据映射

| 链路已有信息 | 写入 L4 |
| --- | --- |
| `SubmitIntentInput` | 会话字段、`session_origin_raw_query`、`current_raw_query` |
| `product_search_tool.tool.invoke` | 完整替换 `last_search.args`，表示 Agent 实际执行条件 |
| 对应 `tool.result` | 成败、返回数量、商品引用 |
| `final.result` | 验证状态和后端最终推荐的精简商品卡 |
| `prepare_order_tool` 参数/结果 | 当前订单行、配送国家、confirmation 状态 |
| `query_order_tool` 结果 | 当前订单状态 |

完整 Top-K、完整地址、完整工具 JSON 和 Trace 不复制进 L4；保留在现有审计/冷数据中。L4 只保留当前决策需要的有界字段和引用。

### 4.3 更新时机与规则

1. 请求开始：先写入原始 `raw_query`、locale、currency；首个 query 同时写入 `session_origin_raw_query`。
2. 本轮运行：active 工具结果仍由当前层消息承载，不在 Prompt 中重复展开 L4。
3. 一轮结束：按捕获到的 `tool.invoke/tool.result/final.result` 顺序运行纯 Reducer，一次提交新 revision。
4. 同会话已有串行锁；并行子 Agent 的搜索按 `tool_call_id` 配对，最后一次成功搜索成为 `last_search`。
5. 失败或进程中断时不提交半份快照；原始 checkpoint/messages 仍负责恢复。

因此 L4 保存的是“链路已发生的事实”，不是再次猜测用户意图。原始 query 同时保留，避免只剩模型解释后的工具参数。

## 5. L2：生命周期整理与 Breakpoint 前移

### 5.1 交互单元

不能按单条消息压缩，必须按协议完整分组：

```text
用户消息
Assistant tool-call + 对应 Tool result（可能多组）
Assistant 最终回复
```

失败、重试和并行工具也必须按 `tool_call_id` 配对，禁止留下孤立 ToolMessage。

### 5.2 settled 判定

一批历史只有同时满足以下条件才可冻结：

- 属于已经结束的完整轮次，当前图没有 pending task/interrupt。
- 所有 tool-call 都已有成功或失败结果。
- 当前模型已经消费过这些结果，不是刚返回、仍用于下一次 Think 的 observation。
- 后续仍需保持的字段已经进入 L4；完整结果已在 checkpoint/audit 中可恢复。
- 它位于当前动态区最前面的连续区间；冻结游标只允许向前移动。

第一版采用最稳妥的轮次边界：新一轮用户消息到来时，上一完整轮次成为 settled。不会为了控制体积去压缩正在使用的 active 内容。

### 5.3 冻结前整理

L2 对 settled 轮次只执行一次确定性整理：

- 保留原始用户话语和最终答复的必要语义。
- 工具交互改写成紧凑结构：工具名、call id、实际参数、状态、结果数量、关键引用。
- 已进入 L4 的长字段不在历史里重复展开。
- 第一版不直接改写原始 `messages`，模型先使用单独的 `llm_input_messages` 视图，避免误删协议消息。

整理结果持久化为 frozen segment，记录源消息范围、context revision 和内容 hash。持久化成功后才移动逻辑 Breakpoint，之后不得反复改写该段。

这里要区分 Prompt 压缩和存储保留：L2/L3 替换的是模型可见表示。等完整交互已可靠写入 ConversationStore/冷数据并完成恢复测试后，才可以按完整交互单元清理 checkpoint 中对应的旧原文，防止 Redis 状态无限增长；不能在同一步里先删后存。

## 6. L0、L1 和 L3 的协作

### L0：工具入口体积合同

- 为每个工具定义模型可见字段、Top-K、单项长度和结果上限。
- 超长原始结果先写冷数据，ToolMessage 只返回 `artifact_ref + 最小充分结果`。
- ProductSearch 保留当前精排后的 Top-K；不会把召回 Top-100 注入主 Agent。
- 当前仍 active 的结果如果过大，只能回到 L0 缩小或重新查询，不能由 L3 有损压缩。

### L1：总预算检查

每次模型调用前计算：

```text
预计输入 = 固定层 + 冻结历史 + L4 状态 + active/current
可用输入 = 模型窗口 - 回复预留 - 安全余量
```

L1 本身不删除任何内容：

- 未超限：直接调用模型。
- 超限且存在旧冻结历史：触发 L3，然后重新计数。
- active/current 自身已超限：返回 L0 缩小结果或明确失败；不得压缩正在使用的信息。

固定 Prompt、工具定义、L4 和历史分别分配多少 Token，先通过真实运行测量决定，不在方案阶段拍固定比例。

### L3：只压缩旧冻结历史

L3 的唯一输入是已冻结历史，不处理 L4、当前 query 或 active 工具结果：

```text
旧 frozen segments
→ 生成结构化阶段摘要
→ 摘要替代这些 segment 的模型可见表示
→ 原始 segment 仍在 checkpoint/audit
→ 摘要冻结并成为新的稳定前缀
```

触发阈值是 L1 计算出的“总 Prompt 输入软阈值”，不是动态区阈值。具体阈值待基线测量后写入配置。

## 7. 建议代码结构

```text
src/globex_agent/application/context/
├── models.py          # L4、frozen segment、budget report
├── reducer.py         # SubmitIntent + TradeEvent → L4
├── lifecycle.py       # 工具配对、active/settled/frozen 判定
├── assembler.py       # 四层 Prompt / llm_input_messages
├── budget.py          # L1 计数与决策
└── compactor.py       # L2 确定性整理、L3 阶段摘要
```

接线位置：

- `orchestrator.py`：请求开始和一轮结束时提交 L4 revision。
- `main_agent.py`：自定义 State Schema，并用 `pre_model_hook` 返回 `llm_input_messages`。
- Redis Checkpointer：保存 L4、frozen segments、freeze cursor；继续保存原始 messages。
- `ConversationStore`：继续作为业务审计，不改造成 checkpoint 或 L4 Store。
- `base.py`：新链路验收后删除旧 `compress_context()` 消息数截断。

第一阶段只治理主 Agent；Search/Trade 子 Agent 保持短任务上下文。它们的结构化工具事件仍可归集到主会话 L4。

## 8. 分阶段实施顺序

### Phase A：基线和数据模型

- 增加分层 Token 统计，不改变 Prompt。
- 记录固定 Prompt、工具 schema、L4 候选字段、历史和各工具结果的真实 Token 分布。
- 建立 L4/frozen/budget 数据模型和 Reducer 单测。

### Phase B：L4 独立落地

- 从现有请求与事件生成 L4，写入 LangGraph checkpoint。
- 暂不注入 Prompt，不接长期记忆、哨兵或 Langfuse。
- 验证重启恢复、并行工具配对和 revision 幂等。

### Phase C：L2 + 四层 Prompt

- 增加生命周期分组、settled 判定、frozen segment 和单调 freeze cursor。
- 用 `pre_model_hook/llm_input_messages` 组装四层上下文，不破坏 checkpoint 原始消息。
- 把 L4 注入 Breakpoint 后；移除旧消息数截断。
- 原文归档与恢复测试通过后，再单独启用 checkpoint 旧 settled 单元清理开关。

### Phase D：L0 + L1

- 为现有工具补齐模型可见输出合同和超长结果冷数据引用。
- 启用总 Token 预算检查和 overflow 决策；先观测后确定配置值。

### Phase E：L3

- 只对旧 frozen segments 做阶段摘要。
- 验证摘要前后约束保持、任务成功率和恢复能力，再启用自动触发。

### Phase F：缓存与整体评测

- 若当前模型供应商暴露 cache usage，再实现供应商适配和真实命中率统计。
- 跑长会话、Redis 恢复、现有在线证据门禁与离线 Flow 回归。

## 9. 验收标准

- L4 中的每个字段都能追溯到 SubmitIntent 或具体 tool/final 事件。
- 原始 query、当前实际搜索条件、最终推荐和订单状态在多轮后不丢失。
- active 内容无论是否触发总阈值都不被 L2/L3 压缩。
- L2 只整理完整 settled 单元，冻结游标单调前进，不产生孤立工具消息。
- L3 只替换冻结历史的模型可见表示，原始记录仍可审计和恢复。
- Prompt 中不出现召回 Top-100；主 Agent 只看到工具交付的精排 Top-K 或其后续精简表示。
- Token 阈值来自真实统计和配置，不再用消息条数近似。
- Redis 重启恢复后，L4 revision、冻结历史和当前 active 交互一致。

## 10. 未来可能打通的链路（仅保存，不在当前阶段接入）

| 链路 | 未来接口方向 | 当前边界 |
| --- | --- | --- |
| 长期记忆 Store | 从 L4 中经用户确认的稳定偏好生成候选记忆；召回结果按需进入状态层 | L4 不直接写长期记忆，也不复制 PreferenceStore |
| 在线哨兵 | 读取 raw query、L4 实际执行条件和最终结果做一致性检查 | 哨兵不是 L4 的数据源，不参与第一版 Reducer |
| Rubric/离线评测 | 用 L4 快照检查预算、国家等约束是否跨轮保留以及压缩前后是否回退 | 仅消费审计快照，不改变在线状态 |
| Langfuse | Trace 中记录 context revision、各层 Token、breakpoint、L2/L3 动作和 cache usage | Langfuse 只做旁路观测，不作为状态存储 |
| Semantic Cache | 将与答案有关的 L4 约束版本纳入 cache key，并处理证据版本 | 当前有历史时仍按现有策略绕过语义缓存 |
| RAG/商品检索 | 以后可由 L4 提供已执行约束或跟进查询上下文 | 当前工具参数仍由 Agent 正常生成 |
| Checkpoint 恢复 | 发生异常时从原始 messages/events 重建 L4，并校验 revision | 第一版只做正常提交和 Redis 状态恢复 |
| 任务边界 | 区分“同一购物任务继续”和“会话内开启新任务”，重置 origin/state | 当前没有可靠 task-boundary 机制，先不猜测 |
| User 塔 | 将经确认的长期偏好或行为信号用于个性化召回 | 不把 L4 的临时任务约束直接训练成用户偏好 |

这些链路必须在各自阶段单独设计和验收，不能为了预留接口提前改变本方案的数据权威边界。
