"""task_dispatch 工具

SubAgent as Tool 的调度工具——MainAgent 调它意味着
"派一个专家子 Agent 去执行这段 demands"。
2.0 库级没有 subagent 原语，用 FunctionTool 包装子 Agent 实现同等语义。

子 Agent 每次调度新建实例（独立 AgentState = 上下文隔离），
只把最终结论回传给 MainAgent，
中间的工具调用过程由业务工具自身通过 EventBus 直接上报前端。

真并行：本工具注册为 is_concurrency_safe，主 Agent 同一轮发起的多个 task_dispatch
会被 2.0 批量 asyncio.gather 并发执行；agent.dispatch 事件带 started_at，
完成时另发 tool.result 带 finished_at/elapsed_ms，可从事件流直接判定时间重叠。

H5 偏好边界：历史偏好只供 MainAgent 在检索结果产生后做最终挑选，不能进入
SearchAgent 的 normalized_query、OpenSearch 或 Reranker。因此本工具只转发当前轮
明确需求，绝不读取或注入 PreferenceStore。

注意：本模块不能用 `from __future__ import annotations`
（AgentScope schema 生成依赖运行时注解）。
"""

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from agentscope.message import TextBlock, ToolResultState, UserMsg
from agentscope.tool import ToolChunk

from app.application.agents.search_agent import SearchAgentFactory
from app.application.agents.trade_agent import TradeAgentFactory
from app.infrastructure.context import SearchDispatchContext, ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def build_task_dispatch_tool(
    search_factory: SearchAgentFactory,
    trade_factory: TradeAgentFactory,
    bus: TradeEventBus,
):
    async def task_dispatch(
        subagent_type: Literal["search_agent", "trade_agent"],
        demands: str,
        platform: Literal["globex_reference", "taobao", "amazon"] | None = None,
        site_locale: Literal["us", "es", "jp"] | None = None,
    ) -> ToolChunk:
        """调度专家子代理执行子任务，返回子代理的结论（JSON 字符串）。

        仅当子任务满足"可并行 / 需要上下文隔离 / 内部调用链较深"任一条件时使用；
        简单的单步工具调用应自己直接调业务工具完成。
        多个彼此独立的子任务请在同一轮一次性发起多个本工具调用，系统会并发执行。

        Args:
            subagent_type (`str`):
                子代理类型："search_agent"（跨境商品检索专家）或
                "trade_agent"（下单交易专家）。
            demands (`str`):
                自包含的自然语言指令，必须包含子代理完成任务所需的全部上下文
                （当前轮预算、product_id/sku_id、收货地址等），子代理看不到主对话历史。
                不得包含长期 like/dislike 或其他历史偏好。
            platform (`str | None`):
                派发 search_agent 时明确固定的平台；trade_agent 不得传。
            site_locale (`str | None`):
                仅在 platform="amazon" 且买家明确指定 Amazon 站点时传入。
        """
        if subagent_type == "trade_agent" and (platform or site_locale):
            return _dispatch_error("trade_agent 不接受 platform 或 site_locale")
        if subagent_type == "search_agent" and platform is None:
            return _dispatch_error("派发 search_agent 必须明确指定 platform")
        if site_locale is not None and platform != "amazon":
            return _dispatch_error("site_locale 仅可用于 amazon 检索任务")

        session_id = ShoppingContext.current_session_id()
        dispatch_correlation_id = (
            uuid4().hex if subagent_type == "search_agent" else None
        )
        started_at = datetime.now(UTC).isoformat()
        started_monotonic = time.monotonic()
        bus.publish(
            session_id,
            "agent.dispatch",
            {
                "agent": subagent_type,
                "demands": demands,
                "platform": platform,
                "site_locale": site_locale,
                "dispatch_correlation_id": dispatch_correlation_id,
                "started_at": started_at,
            },
        )

        if subagent_type == "search_agent":
            worker = search_factory.build(
                platform=platform,
                site_locale=site_locale,
            )
        elif subagent_type == "trade_agent":
            worker = trade_factory.build()
        else:
            return ToolChunk(
                content=[
                    TextBlock(
                        type="text", text=f"[error] 未知 subagent_type：{subagent_type}"
                    )
                ],
                state=ToolResultState.ERROR,
            )

        search_events = (
            bus.subscribe(session_id) if subagent_type == "search_agent" else None
        )
        dispatch_token = (
            SearchDispatchContext.set(dispatch_correlation_id)
            if dispatch_correlation_id is not None
            else None
        )
        try:
            reply = await worker.reply(UserMsg("commerce_concierge", demands))
        finally:
            captured = []
            if search_events is not None:
                while not search_events.empty():
                    captured.append(search_events.get_nowait())
                bus.unsubscribe(session_id, search_events)
            if dispatch_token is not None:
                SearchDispatchContext.reset(dispatch_token)
        output = reply.get_text_content() or ""
        if subagent_type == "search_agent":
            search_result = _latest_search_result(
                captured,
                dispatch_correlation_id=dispatch_correlation_id,
                platform=platform,
                site_locale=site_locale,
            )
            output = json.dumps(
                {
                    "agent": "search_agent",
                    "platform": platform,
                    "site_locale": site_locale,
                    "search_result_available": bool(search_result),
                    "search_args": search_result.get("args", {}),
                    "hits": search_result.get("hits", []),
                    "filtered_out": search_result.get("filtered_out", []),
                    "recall_strategy": search_result.get("recall_strategy"),
                    "agent_output": output,
                },
                ensure_ascii=False,
            )
        bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "task_dispatch",
                "agent": subagent_type,
                "platform": platform,
                "site_locale": site_locale,
                "dispatch_correlation_id": dispatch_correlation_id,
                "started_at": started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "elapsed_ms": round((time.monotonic() - started_monotonic) * 1000),
            },
        )
        return ToolChunk(
            content=[TextBlock(type="text", text=output)],
            state=ToolResultState.SUCCESS,
        )

    return task_dispatch


def _dispatch_error(message: str) -> ToolChunk:
    return ToolChunk(
        content=[TextBlock(type="text", text=f"[error] {message}")],
        state=ToolResultState.ERROR,
    )


def _latest_search_result(
    events,
    *,
    dispatch_correlation_id: str | None,
    platform: str | None,
    site_locale: str | None,
) -> dict:
    """Return the latest exact product-tool event for this fixed SearchAgent.

    Each concurrent dispatch has its own local EventBus queue. Queues receive
    all events for the shopping session, so an internal correlation ID selects
    the owning dispatch; platform/site are verified as a second boundary. The
    SearchAgent's prose is never inspected or parsed.
    """
    latest: dict = {}
    for event in events:
        if getattr(event, "type", "") != "tool.result":
            continue
        payload = getattr(event, "payload", None)
        if not isinstance(payload, Mapping):
            continue
        if payload.get("tool") != "product_search_tool" or payload.get("error"):
            continue
        if payload.get("dispatch_correlation_id") != dispatch_correlation_id:
            continue
        if payload.get("platform") != platform:
            continue
        if payload.get("site_locale") != site_locale:
            continue
        if not isinstance(payload.get("hits"), list):
            continue
        latest = {
            "args": dict(payload.get("args") or {}),
            "hits": list(payload["hits"]),
            "filtered_out": list(payload.get("filtered_out") or []),
            "recall_strategy": payload.get("recall_strategy"),
        }
    return latest
