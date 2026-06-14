from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import PrivateAttr

from globex_agent.application.agents.main_agent import MainAgentState
from globex_agent.application.context import (
    ContextBudgetExceeded,
    ContextBudgetPolicy,
    FrozenSegment,
    StageSummary,
    assemble_llm_input_messages,
    build_context_pre_model_hook,
    context_pre_model_hook,
    summarize_frozen_segments,
)


class _ExactCounter:
    def get_num_tokens(self, text: str) -> int:
        return len(text)

    def get_num_tokens_from_messages(self, messages: list[BaseMessage]) -> int:
        return sum(len(str(message.content)) + 1 for message in messages)


class _ExactAnswerModel(BaseChatModel):
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "phase-e-exact-model"

    def bind_tools(self, tools, **kwargs: Any) -> Runnable:
        del tools, kwargs
        return self

    def get_num_tokens(self, text: str) -> int:
        return len(text)

    def get_num_tokens_from_messages(self, messages: list[BaseMessage]) -> int:
        return sum(len(str(message.content)) + 1 for message in messages)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self._calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="done"))])


def _segment(index: int, *, repeated: bool = True) -> FrozenSegment:
    tool = {
        "tool": "product_search_tool",
        "tool_call_id": "shared-call" if repeated else f"call-{index}",
        "args": {
            "normalized_query": "轻量通勤双肩包",
            "ship_to": "US",
            "price_max_major": 500,
            "target_currency": "CNY",
            "constraints": "防泼水且不要真皮" * 80,
        },
        "status": "success",
        "returned_count": 5,
        "refs": ["item-a", "item-b"],
    }
    return FrozenSegment(
        segment_id=f"seg-{index}",
        content=json.dumps(
            {
                "user": f"第{index}轮：预算500元，寄到美国，必须防泼水且不要真皮。",
                "tools": [tool],
                "assistant": "已按全部约束完成检索并保留候选。" * 100,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        source_message_start=index * 4,
        source_message_end=index * 4 + 4,
        context_revision=index + 1,
    )


def _probe(
    segments: list[FrozenSegment],
    *,
    stage_summary: dict[str, Any] | None = None,
    soft_limit: int = 1,
    mode: str = "observe_only",
    keep_recent: int = 1,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="当前请求保持原样")],
        "frozen_segments": [segment.to_dict() for segment in segments],
        "freeze_cursor": 0,
        "session_context": {
            "revision": 9,
            "request": {"current_raw_query": "当前请求保持原样"},
        },
    }
    if stage_summary is not None:
        state["stage_summary"] = stage_summary
    return context_pre_model_hook(
        state,
        model=_ExactCounter(),
        fixed_prompt="fixed",
        tool_schemas=[{"name": "product_search_tool"}],
        budget_policy=ContextBudgetPolicy(
            model_context_tokens=100_000,
            soft_limit_tokens=soft_limit,
            mode=mode,
            l3_keep_recent_frozen_segments=keep_recent,
        ),
    )


def test_stage_summary_preserves_requests_and_tool_facts_deduplicates_and_is_idempotent() -> None:
    segments = [_segment(index, repeated=False) for index in range(3)]
    summary = summarize_frozen_segments(segments)
    replay = summarize_frozen_segments(segments, existing=summary)

    assert summary == replay
    assert summary.source_segment_ids == ("seg-0", "seg-1", "seg-2")
    assert len(summary.user_requests) == 3
    assert all("预算500元" in request for request in summary.user_requests)
    assert all("不要真皮" in request for request in summary.user_requests)
    expected_fact = json.loads(segments[0].content)["tools"][0]
    expected_fact.pop("tool_call_id")
    expected_fact["tool_call_ids"] = ["call-0", "call-1", "call-2"]
    assert summary.tool_facts == (expected_fact,)
    assert summary.tool_facts[0]["args"]["ship_to"] == "US"
    assert summary.tool_facts[0]["status"] == "success"
    assert summary.tool_facts[0]["returned_count"] == 5
    assert summary.tool_facts[0]["refs"] == ["item-a", "item-b"]
    assert len(summary.assistant_outcomes) == 1
    assert len(summary.assistant_outcomes[0]) == 512
    assert summary.source_message_start == 0
    assert summary.source_message_end == 12
    assert summary.context_revision == 3
    assert summary.combined_source_hash
    assert summary.summary_hash
    assert StageSummary.from_dict(summary.to_dict()) == summary


def test_assembler_replaces_covered_segments_but_keeps_sources_immutable() -> None:
    segments = [_segment(index) for index in range(3)]
    original = [segment.to_dict() for segment in segments]
    summary = summarize_frozen_segments(segments[:2])
    view = assemble_llm_input_messages(
        messages=[HumanMessage(content="active")],
        frozen_segments=segments,
        freeze_cursor=0,
        session_context={"revision": 3},
        stage_summary=summary,
    )
    text = "\n".join(str(message.content) for message in view)

    assert text.count("<stage-summary>") == 1
    assert 'id="seg-0"' not in text
    assert 'id="seg-1"' not in text
    assert text.count('id="seg-2"') == 1
    assert "active" in text
    assert [segment.to_dict() for segment in segments] == original


def test_l1_automatically_runs_l3_recounts_to_ok_and_preserves_l4_active() -> None:
    segments = [_segment(index) for index in range(4)]
    before = _probe(segments)
    candidate = summarize_frozen_segments(segments[:3])
    after = _probe(segments, stage_summary=candidate.to_dict())
    before_tokens = before["budget_report"]["total_input_tokens"]
    after_tokens = after["budget_report"]["total_input_tokens"]
    assert after_tokens < before_tokens
    soft_limit = (before_tokens + after_tokens) // 2
    active = HumanMessage(content="当前请求保持原样")
    l4 = {"revision": 9, "request": {"current_raw_query": "当前请求保持原样"}}
    state = {
        "messages": [active],
        "frozen_segments": [segment.to_dict() for segment in segments],
        "freeze_cursor": 0,
        "session_context": copy.deepcopy(l4),
    }
    update = context_pre_model_hook(
        state,
        model=_ExactCounter(),
        fixed_prompt="fixed",
        tool_schemas=[{"name": "product_search_tool"}],
        budget_policy=ContextBudgetPolicy(
            model_context_tokens=100_000,
            soft_limit_tokens=soft_limit,
            mode="enforce",
            l3_keep_recent_frozen_segments=1,
        ),
    )

    assert update["budget_decision"] == "ok"
    assert update["context_compression"]["before_tokens"] == before_tokens
    assert update["context_compression"]["after_tokens"] == after_tokens
    assert update["context_compression"]["after_tokens"] < before_tokens
    assert update["stage_summary"]["source_segment_ids"] == ["seg-0", "seg-1", "seg-2"]
    assert update["frozen_segments"] == [segment.to_dict() for segment in segments]
    assert "messages" not in update
    assert state["messages"] == [active]
    assert state["session_context"] == l4
    visible = "\n".join(str(item.content) for item in update["llm_input_messages"])
    assert visible.count('id="seg-3"') == 1
    assert 'id="seg-0"' not in visible
    assert "<session-context>" in visible
    assert "当前请求保持原样" in visible


def test_l3_replay_new_segment_append_and_hash_conflict_behavior() -> None:
    segments = [_segment(index) for index in range(4)]
    first = summarize_frozen_segments(segments[:2])
    replay = summarize_frozen_segments(segments[:2], existing=first)
    extended = summarize_frozen_segments([segments[2]], existing=replay)
    assert replay.summary_hash == first.summary_hash
    assert extended.source_segment_ids == ("seg-0", "seg-1", "seg-2")
    assert extended.summary_hash != first.summary_hash
    view = assemble_llm_input_messages(
        messages=[],
        frozen_segments=segments,
        freeze_cursor=0,
        session_context=None,
        stage_summary=extended,
    )
    text = "".join(str(item.content) for item in view)
    assert text.count('id="seg-3"') == 1
    assert 'id="seg-2"' not in text

    changed = FrozenSegment(
        segment_id="seg-0",
        content='{"user":"changed","tools":[],"assistant":"x"}',
    )
    with pytest.raises(ValueError, match="hash changed"):
        summarize_frozen_segments([changed], existing=first)
    with pytest.raises(ValueError, match="source unavailable or changed"):
        assemble_llm_input_messages(
            messages=[],
            frozen_segments=segments[1:],
            freeze_cursor=0,
            session_context=None,
            stage_summary=first,
        )


def test_estimated_observe_only_and_no_eligible_history_never_auto_compress() -> None:
    segments = [_segment(0)]
    observed = _probe(segments, mode="observe_only", keep_recent=0)
    assert observed["budget_decision"] == "needs_l3"
    assert "stage_summary" not in observed
    estimated = context_pre_model_hook(
        {
            "messages": [HumanMessage(content="active")],
            "frozen_segments": [segment.to_dict() for segment in segments],
        },
        budget_policy=ContextBudgetPolicy(
            model_context_tokens=100_000,
            soft_limit_tokens=1,
            mode="enforce",
            l3_keep_recent_frozen_segments=0,
        ),
    )
    assert estimated["budget_report"]["estimated"] is True
    assert "stage_summary" not in estimated
    no_eligible = _probe(segments, mode="enforce", keep_recent=1)
    assert no_eligible["budget_decision"] == "needs_l3_no_eligible"
    assert (
        no_eligible["budget_report"]["total_input_tokens"]
        <= no_eligible["budget_report"]["available_input_tokens"]
    )
    assert "stage_summary" not in no_eligible


@pytest.mark.parametrize("segment_count,keep_recent", [(1, 1), (4, 1)])
def test_exact_enforcement_rejects_hard_overflow_after_l3_limits(
    segment_count: int,
    keep_recent: int,
) -> None:
    segments = [_segment(index) for index in range(segment_count)]
    before = _probe(segments)
    if segment_count == 1:
        hard_limit = before["budget_report"]["total_input_tokens"] - 1
    else:
        summarized = summarize_frozen_segments(segments[:-keep_recent])
        after = _probe(segments, stage_summary=summarized.to_dict())
        hard_limit = after["budget_report"]["total_input_tokens"] - 1
    state = {
        "messages": [HumanMessage(content="当前请求保持原样")],
        "frozen_segments": [segment.to_dict() for segment in segments],
        "freeze_cursor": 0,
        "session_context": {
            "revision": 9,
            "request": {"current_raw_query": "当前请求保持原样"},
        },
    }

    with pytest.raises(ContextBudgetExceeded) as raised:
        context_pre_model_hook(
            state,
            model=_ExactCounter(),
            fixed_prompt="fixed",
            tool_schemas=[{"name": "product_search_tool"}],
            budget_policy=ContextBudgetPolicy(
                model_context_tokens=hard_limit,
                soft_limit_tokens=1,
                mode="enforce",
                l3_keep_recent_frozen_segments=keep_recent,
            ),
        )

    assert raised.value.decision == "needs_l3_hard_overflow"
    assert raised.value.report.total_input_tokens > hard_limit
    assert state["messages"] == [HumanMessage(content="当前请求保持原样")]
    assert "stage_summary" not in state


def test_l3_failure_returns_no_partial_summary_or_source_mutation() -> None:
    broken = FrozenSegment(segment_id="broken", content="not-json")
    state = {
        "messages": [HumanMessage(content="active")],
        "frozen_segments": [broken.to_dict(), _segment(1).to_dict()],
        "freeze_cursor": 0,
    }
    original = copy.deepcopy(state)
    with pytest.raises(ValueError, match="invalid frozen segment JSON"):
        context_pre_model_hook(
            state,
            model=_ExactCounter(),
            budget_policy=ContextBudgetPolicy(
                model_context_tokens=100_000,
                soft_limit_tokens=1,
                mode="enforce",
                l3_keep_recent_frozen_segments=1,
            ),
        )
    assert state == original


async def test_l3_checkpoint_roundtrip_keeps_raw_messages_and_frozen_sources() -> None:
    segments = [_segment(index) for index in range(4)]
    before = _probe(segments)
    candidate = summarize_frozen_segments(segments[:3])
    after = _probe(segments, stage_summary=candidate.to_dict())
    soft_limit = (
        before["budget_report"]["total_input_tokens"]
        + after["budget_report"]["total_input_tokens"]
    ) // 2
    saver = InMemorySaver()
    model = _ExactAnswerModel()
    hook = build_context_pre_model_hook(
        model=model,
        fixed_prompt="fixed",
        tool_schemas=[],
        budget_policy=ContextBudgetPolicy(
            model_context_tokens=100_000,
            soft_limit_tokens=soft_limit,
            mode="enforce",
            l3_keep_recent_frozen_segments=1,
        ),
    )
    graph = create_react_agent(
        model,
        tools=[],
        prompt="fixed",
        pre_model_hook=hook,
        state_schema=MainAgentState,
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "phase-e-roundtrip"}}
    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="active")],
            "frozen_segments": [segment.to_dict() for segment in segments],
            "freeze_cursor": 0,
        },
        config=config,
    )
    first = await graph.aget_state(config)
    rebuilt = create_react_agent(
        _ExactAnswerModel(),
        tools=[],
        prompt="fixed",
        pre_model_hook=hook,
        state_schema=MainAgentState,
        checkpointer=saver,
    )
    restored = await rebuilt.aget_state(config)

    assert restored.values["stage_summary"] == first.values["stage_summary"]
    assert restored.values["frozen_segments"] == [segment.to_dict() for segment in segments]
    assert [message.content for message in restored.values["messages"]] == ["active", "done"]
    assert restored.values["context_compression"]["after_tokens"] < restored.values[
        "context_compression"
    ]["before_tokens"]
