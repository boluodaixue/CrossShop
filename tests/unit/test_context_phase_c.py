from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import PrivateAttr

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.agents.main_agent import MainAgentState
from globex_agent.application.context import (
    CACHE_BREAKPOINT_TEXT,
    FrozenSegment,
    assemble_llm_input_messages,
    context_pre_model_hook,
    freeze_settled_units,
    group_interaction_units,
    settled_prefix_units,
)


def _call(name: str, call_id: str, args: dict) -> dict:
    return {"name": name, "id": call_id, "args": args, "type": "tool_call"}


def _parallel_history():
    return [
        HumanMessage(content="旧问题"),
        AIMessage(
            content="",
            tool_calls=[
                _call("product_search_tool", "call-a", {"ship_to": "US"}),
                _call("category_insight_tool", "call-b", {"category": "bag"}),
            ],
        ),
        ToolMessage(
            content=json.dumps({"insights": {}, "count": 1}),
            tool_call_id="call-b",
            name="category_insight_tool",
        ),
        ToolMessage(
            content=json.dumps({"hits": [{"item_id": "item-a"}]}),
            tool_call_id="call-a",
            name="product_search_tool",
        ),
        AIMessage(content="旧回答"),
        HumanMessage(content="当前问题"),
    ]


def test_groups_parallel_tools_by_call_id_and_only_settles_previous_turn() -> None:
    messages = _parallel_history()
    units = group_interaction_units(messages)

    assert len(units) == 2
    assert units[0].complete is True
    assert [tool.tool_call_id for tool in units[0].tools] == ["call-a", "call-b"]
    assert [tool.result_message_index for tool in units[0].tools] == [3, 2]
    settled = settled_prefix_units(messages)
    assert [(unit.start, unit.end) for unit in settled] == [(0, 5)]
    assert units[1].complete is False


def test_failure_and_retry_remain_one_complete_interaction() -> None:
    messages = [
        HumanMessage(content="查商品"),
        AIMessage(
            content="",
            tool_calls=[_call("product_search_tool", "first", {"top_k": 5})],
        ),
        ToolMessage(
            content='{"error":"timeout"}',
            tool_call_id="first",
            name="product_search_tool",
            status="error",
        ),
        AIMessage(
            content="重试",
            tool_calls=[_call("product_search_tool", "retry", {"top_k": 3})],
        ),
        ToolMessage(
            content='{"hits":[{"item_id":"i1"}]}',
            tool_call_id="retry",
            name="product_search_tool",
        ),
        AIMessage(content="完成"),
        HumanMessage(content="下一轮"),
    ]
    unit = settled_prefix_units(messages)[0]
    segments, cursor = freeze_settled_units(
        messages,
        [unit],
        context_revision=4,
    )
    compacted = json.loads(segments[0].content)

    assert cursor == 6
    assert [tool["status"] for tool in compacted["tools"]] == ["error", "success"]
    assert compacted["tools"][1]["returned_count"] == 1
    assert compacted["tools"][1]["refs"] == ["i1"]


def test_missing_or_orphan_result_never_freezes() -> None:
    missing = [
        HumanMessage(content="旧问题"),
        AIMessage(content="", tool_calls=[_call("search", "missing", {})]),
        AIMessage(content="不完整"),
        HumanMessage(content="当前"),
    ]
    orphan = [
        HumanMessage(content="旧问题"),
        ToolMessage(content="{}", tool_call_id="orphan", name="search"),
        AIMessage(content="回答"),
        HumanMessage(content="当前"),
    ]

    assert settled_prefix_units(missing) == []
    assert "missing_tool_result" in group_interaction_units(missing)[0].incomplete_reason
    assert settled_prefix_units(orphan) == []
    assert "orphan_tool_result" in group_interaction_units(orphan)[0].incomplete_reason


def test_pending_or_interrupt_blocks_freezing() -> None:
    messages = _parallel_history()
    assert settled_prefix_units(messages, has_pending=True) == []
    assert settled_prefix_units(messages, has_interrupt=True) == []


def test_freeze_cursor_and_segment_content_are_immutable() -> None:
    messages = _parallel_history()
    units = settled_prefix_units(messages)
    first, first_cursor = freeze_settled_units(
        messages,
        units,
        context_revision=3,
    )
    replayed, replayed_cursor = freeze_settled_units(
        messages,
        units,
        existing=first,
        freeze_cursor=first_cursor,
        context_revision=99,
    )

    assert replayed_cursor == first_cursor == 5
    assert replayed == first
    assert replayed[0].content_hash == first[0].content_hash
    assert replayed[0].context_revision == 3


def test_assembler_orders_frozen_breakpoint_l4_then_active_without_old_raw() -> None:
    messages = _parallel_history()
    units = settled_prefix_units(messages)
    frozen, cursor = freeze_settled_units(messages, units, context_revision=2)
    assembled = assemble_llm_input_messages(
        messages=messages,
        frozen_segments=frozen,
        freeze_cursor=cursor,
        session_context={"revision": 2, "request": {"current_raw_query": "当前问题"}},
    )
    contents = [str(message.content) for message in assembled]

    assert "<frozen-interaction" in contents[0]
    assert contents[1] == CACHE_BREAKPOINT_TEXT
    assert contents[2].startswith("<session-context>")
    assert isinstance(assembled[3], HumanMessage)
    assert assembled[3].content == "当前问题"
    assert all(not isinstance(message, ToolMessage) for message in assembled[3:])
    assert sum("旧问题" in content for content in contents) == 1


def test_pre_model_hook_persists_frozen_metadata_without_deleting_messages() -> None:
    messages = _parallel_history()
    update = context_pre_model_hook(
        {
            "messages": messages,
            "session_context": {"revision": 7},
            "frozen_segments": [],
            "freeze_cursor": 0,
        }
    )
    original_ids = [message.id for message in messages]

    assert "messages" not in update
    assert update["freeze_cursor"] == 5
    assert update["frozen_segments"][0]["context_revision"] == 7
    assert [message.id for message in messages] == original_ids
    replayed = context_pre_model_hook(
        {**update, "messages": messages, "session_context": {"revision": 8}}
    )
    assert replayed["frozen_segments"] == update["frozen_segments"]
    assert replayed["freeze_cursor"] == update["freeze_cursor"]


def test_existing_frozen_segment_deserializes_for_hook() -> None:
    segment = FrozenSegment(
        segment_id="fixed",
        content="{}",
        source_message_start=0,
        source_message_end=1,
    )
    update = context_pre_model_hook(
        {
            "messages": [HumanMessage(content="active")],
            "frozen_segments": [segment.to_dict()],
            "freeze_cursor": 1,
        }
    )
    assert update["frozen_segments"] == [segment.to_dict()]


class _InterruptToolModel(BaseChatModel):
    _bound: bool = PrivateAttr(default=False)

    @property
    def _llm_type(self) -> str:
        return "phase-c-interrupt-model"

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: str | None = None,
        **kwargs,
    ) -> Runnable:
        del tools, tool_choice, kwargs
        self._bound = True
        return self

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        last_human = max(
            index for index, message in enumerate(messages) if isinstance(message, HumanMessage)
        )
        if any(isinstance(message, ToolMessage) for message in messages[last_human + 1 :]):
            response = AIMessage(content="tool consumed")
        else:
            response = AIMessage(
                content="",
                tool_calls=[_call("phase_c_echo", "interrupt-call", {"text": "ok"})],
            )
        return ChatResult(generations=[ChatGeneration(message=response)])


@tool
def phase_c_echo(text: str) -> str:
    """Return a small deterministic tool result."""

    return json.dumps({"result": text})


def _interrupt_graph(saver: InMemorySaver):
    return create_react_agent(
        _InterruptToolModel(),
        tools=[phase_c_echo],
        prompt="fixed prompt",
        pre_model_hook=context_pre_model_hook,
        state_schema=MainAgentState,
        checkpointer=saver,
        interrupt_before=["tools"],
    )


async def test_pending_resume_keeps_active_tool_protocol_and_freezes_only_next_turn() -> None:
    saver = InMemorySaver()
    thread_id = "phase-c-interrupt"
    first = LangGraphAgent("main", _interrupt_graph(saver), thread_id=thread_id)
    await first.reply("first")
    pending = await first._graph.aget_state(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    assert pending.next
    assert pending.values.get("freeze_cursor", 0) == 0
    assert isinstance(pending.values["messages"][-1], AIMessage)
    assert pending.values["messages"][-1].tool_calls

    rebuilt = LangGraphAgent("main", _interrupt_graph(saver), thread_id=thread_id)
    assert await rebuilt.reply("first") == "tool consumed"
    resumed = await rebuilt._graph.aget_state(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    assert not resumed.next
    assert [type(message) for message in resumed.values["messages"]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert resumed.values.get("freeze_cursor", 0) == 0

    await rebuilt.reply("second")
    next_pending = await rebuilt._graph.aget_state(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    assert next_pending.next
    assert next_pending.values["freeze_cursor"] == 4
    assert len(next_pending.values["frozen_segments"]) == 1
    assert len(next_pending.values["messages"]) == 6
