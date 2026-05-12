"""Small LangGraph agent wrapper used by all agent factories."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

from globex_agent.application.agents.checkpoint import (
    compress_checkpoint,
    restore_checkpoint,
    serialize_checkpoint,
)


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

    async def reply(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        event_sink: Callable[[str, dict], Awaitable[None] | None] | None = None,
    ) -> str:
        actual_thread_id = (
            thread_id
            or self._thread_id
            or f"{self.name}-{uuid4().hex[:8]}"
        )
        config = {
            "configurable": {"thread_id": actual_thread_id},
            "recursion_limit": 30,
        }
        if event_sink is not None:
            return await self._reply_with_stream(query, config, event_sink)

        state = await self._graph.ainvoke(
            {"messages": [("user", query)]},
            config=config,
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

    async def _reply_with_stream(
        self,
        query: str,
        config: dict,
        event_sink: Callable[[str, dict], Awaitable[None] | None],
    ) -> str:
        assistant_text: list[str] = []
        async for message_chunk, _metadata in self._graph.astream(
            {"messages": [("user", query)]},
            config=config,
            stream_mode="messages",
        ):
            content = getattr(message_chunk, "content", "")
            if isinstance(content, str) and content:
                assistant_text.append(content)
                result = event_sink("token.delta", {"token": content})
                if result is not None:
                    await result

        state = await self._graph.aget_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            return "".join(assistant_text)
        final = messages[-1]
        if isinstance(final, dict):
            return str(final.get("content", ""))
        content = getattr(final, "content", "")
        if isinstance(content, str):
            return content
        if not content and assistant_text:
            return "".join(assistant_text)
        return str(content)

    def checkpoint_state(self, thread_id: str) -> str:
        """Return a JSON snapshot of the latest graph checkpoint."""

        checkpointer = getattr(self._graph, "checkpointer", None)
        if checkpointer is None:
            return "{}"
        return serialize_checkpoint(checkpointer, thread_id)

    def restore_checkpoint_state(self, state_json: str, thread_id: str) -> None:
        """Restore a graph checkpoint from a previous process run."""

        checkpointer = getattr(self._graph, "checkpointer", None)
        if checkpointer is None:
            return
        restore_checkpoint(checkpointer, state_json, thread_id)

    def compress_context(
        self,
        thread_id: str,
        max_messages: int,
    ) -> dict[str, int] | None:
        """Trim old graph messages when the conversation becomes too long."""

        checkpointer = getattr(self._graph, "checkpointer", None)
        if checkpointer is None:
            return None
        return compress_checkpoint(checkpointer, thread_id, max_messages)
