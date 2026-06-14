"""Wrap LangChain tools with timeout and circuit-breaker resilience."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.resilience import ToolResilienceMiddleware


def wrap_tool(
    tool: StructuredTool,
    registry: Any,
    bus: TradeEventBus,
) -> StructuredTool:
    """Return a new tool that delegates to ``tool`` through the middleware."""

    async def _resilient_call(**kwargs: Any) -> Any:
        async def operation() -> Any:
            if tool.coroutine is None:
                raise TypeError(f"resilience wrapper requires an async tool: {tool.name}")
            # ToolNode already validated the outer call and injected hidden
            # arguments. Preserve them when composing wrappers.
            return await tool.coroutine(**kwargs)

        return await ToolResilienceMiddleware(registry, bus).run(
            tool.name,
            operation,
        )

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_resilient_call,
        return_direct=tool.return_direct,
    )
