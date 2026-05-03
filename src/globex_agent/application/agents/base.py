"""Small LangGraph agent wrapper used by all agent factories."""

from __future__ import annotations

from uuid import uuid4


class LangGraphAgent:
    def __init__(
        self,
        name: str,
        graph,
        *,
        thread_id: str | None = None,
    ) -> None:
        self.name = name
        self._graph = graph
        self._thread_id = thread_id

    async def reply(self, query: str, *, thread_id: str | None = None) -> str:
        actual_thread_id = (
            thread_id
            or self._thread_id
            or f"{self.name}-{uuid4().hex[:8]}"
        )
        state = await self._graph.ainvoke(
            {"messages": [("user", query)]},
            config={
                "configurable": {"thread_id": actual_thread_id},
                "recursion_limit": 30,
            },
        )
        messages = state.get("messages", [])
        if not messages:
            return ""
        final = messages[-1]
        if isinstance(final, dict):
            return str(final.get("content", ""))
        content = getattr(final, "content", "")
        if isinstance(content, str):
            return content
        return str(content)
