"""Small LangGraph agent wrapper used by all agent factories."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from globex_agent.application.agents.identity import graph_config
from globex_agent.application.context.models import L4Context
from globex_agent.application.context.reducer import reduce_l4


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
        session_context: L4Context | Mapping[str, Any] | None = None,
    ) -> str:
        actual_thread_id = thread_id or self._thread_id
        if not actual_thread_id:
            raise ValueError("thread_id is required for a persistent LangGraph agent")
        config = graph_config(actual_thread_id)
        graph_input = await self._input_for_reply(query, config, session_context)
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

    async def _input_for_reply(
        self,
        query: str,
        config: dict,
        session_context: L4Context | Mapping[str, Any] | None = None,
    ) -> dict | None:
        """Continue a durable graph step instead of appending a duplicate turn."""

        snapshot = await self._graph.aget_state(config)
        if snapshot.next or snapshot.tasks:
            return None
        graph_input: dict[str, Any] = {"messages": [("user", query)]}
        if isinstance(session_context, L4Context):
            graph_input["session_context"] = session_context.to_dict()
        elif isinstance(session_context, Mapping):
            graph_input["session_context"] = dict(session_context)
        return graph_input

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

    async def get_session_context(
        self,
        thread_id: str | None = None,
    ) -> L4Context:
        """Load the L4 value through the graph's configured checkpointer."""

        actual_thread_id = thread_id or self._thread_id
        if not actual_thread_id:
            raise ValueError("thread_id is required for a persistent LangGraph agent")
        snapshot = await self._graph.aget_state(graph_config(actual_thread_id))
        raw = snapshot.values.get("session_context")
        return L4Context.from_dict(raw if isinstance(raw, Mapping) else None)

    async def reduce_session_context(
        self,
        intent: Any,
        events: Iterable[Any] = (),
        *,
        thread_id: str | None = None,
    ) -> L4Context:
        """Idempotently reduce and checkpoint L4 without changing messages."""

        actual_thread_id = thread_id or self._thread_id
        if not actual_thread_id:
            raise ValueError("thread_id is required for a persistent LangGraph agent")
        config = graph_config(actual_thread_id)
        snapshot = await self._graph.aget_state(config)
        raw = snapshot.values.get("session_context")
        previous = L4Context.from_dict(raw if isinstance(raw, Mapping) else None)
        updated = reduce_l4(intent, events, previous)
        if updated == previous:
            return previous
        await self._graph.aupdate_state(
            config,
            {"session_context": updated.to_dict()},
            as_node="agent",
        )
        return updated

    async def prepare_session_context(
        self,
        intent: Any,
        *,
        thread_id: str | None = None,
    ) -> L4Context:
        """Reduce request-start facts for the next graph input checkpoint."""

        previous = await self.get_session_context(thread_id)
        return reduce_l4(intent, (), previous)
