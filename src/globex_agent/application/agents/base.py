"""Small LangGraph agent wrapper used by all agent factories."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from globex_agent.application.agents.identity import graph_config


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
        actual_thread_id = thread_id or self._thread_id
        if not actual_thread_id:
            raise ValueError("thread_id is required for a persistent LangGraph agent")
        config = graph_config(actual_thread_id)
        graph_input = await self._input_for_reply(query, config)
        if event_sink is not None:
            return await self._reply_with_stream(graph_input, config, event_sink)

        state = await self._graph.ainvoke(
            graph_input,
            config=config,
            durability="sync",
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

    async def _input_for_reply(self, query: str, config: dict) -> dict | None:
        """Continue a durable graph step instead of appending a duplicate turn."""

        snapshot = await self._graph.aget_state(config)
        if snapshot.next or snapshot.tasks:
            return None
        return {"messages": [("user", query)]}

    async def _reply_with_stream(
        self,
        graph_input: dict | None,
        config: dict,
        event_sink: Callable[[str, dict], Awaitable[None] | None],
    ) -> str:
        assistant_text: list[str] = []
        async for message_chunk, _metadata in self._graph.astream(
            graph_input,
            config=config,
            stream_mode="messages",
            durability="sync",
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

    async def compress_context(
        self,
        thread_id: str,
        max_messages: int,
    ) -> dict[str, int] | None:
        """Trim old graph messages when the conversation becomes too long."""

        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        config = graph_config(thread_id)
        state = await self._graph.aget_state(config)
        messages = state.values.get("messages", [])
        if not isinstance(messages, list) or len(messages) <= max_messages:
            return None

        keep_head = [messages[0]] if getattr(messages[0], "type", None) == "system" else []
        tail_size = max(1, max_messages - len(keep_head))
        kept = [*keep_head, *messages[-tail_size:]]
        await self._graph.aupdate_state(
            config,
            {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]},
        )
        return {
            "removed": len(messages) - len(kept),
            "kept": len(kept),
        }
