from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import PrivateAttr

from globex_agent.application.agents.main_agent import MainAgentState
from globex_agent.application.context import (
    TOOL_OUTPUT_CONTRACTS,
    ContextBudgetExceeded,
    ContextBudgetPolicy,
    FrozenSegment,
    ToolArtifactStore,
    build_context_pre_model_hook,
    context_pre_model_hook,
    format_tool_output,
)


class _ExactCounter:
    def get_num_tokens(self, text: str) -> int:
        return len(text)

    def get_num_tokens_from_messages(self, messages: list[BaseMessage]) -> int:
        return sum(len(str(message.content)) + 1 for message in messages)


class _RecordingExactModel(BaseChatModel):
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "phase-d-exact-test-model"

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
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="bounded-answer"))]
        )


@pytest.fixture
def artifact_store() -> ToolArtifactStore:
    # Keep this path relative: the repository is under a non-ASCII Windows
    # profile and the available Python 3.10 runtime mangles resolved paths.
    return ToolArtifactStore(Path("data/runtime/pytest-artifacts"))


def test_all_model_visible_tools_have_explicit_l0_contracts() -> None:
    assert set(TOOL_OUTPUT_CONTRACTS) == {
        "product_search_tool",
        "category_insight_tool",
        "web_search_tool",
        "prepare_order_tool",
        "query_order_tool",
        "prepare_cancel_order_tool",
        "task_dispatch",
        "remember_preference_tool",
    }
    assert TOOL_OUTPUT_CONTRACTS["product_search_tool"].list_limits["hits"] == 5
    assert TOOL_OUTPUT_CONTRACTS["task_dispatch"].list_limits["dispatches"] == 8


def test_product_top_100_is_bounded_json_and_exact_raw_roundtrips(
    artifact_store: ToolArtifactStore,
) -> None:
    original = {
        "hits": [
            {
                "item_id": f"item-{index}",
                "title": "商品" + ("很长" * 500),
                "availability": "available",
                "variants": [{"variant_id": f"v-{row}"} for row in range(30)],
            }
            for index in range(100)
        ],
        "returned_count": 100,
        "recall_strategy": "embedding_rerank",
        "internal_recall_top_100": [f"raw-{index}" for index in range(100)],
    }
    raw = json.dumps(original, ensure_ascii=False)
    text = format_tool_output(
        "product_search_tool",
        raw,
        artifact_store=artifact_store,
        configured_limit=20_000,
    )
    visible = json.loads(text)

    assert len(visible["hits"]) == 5
    assert len(visible["hits"][0]["variants"]) == 12
    assert len(visible["hits"][0]["title"]) == 512
    assert "internal_recall_top_100" not in visible
    assert len(text) <= 16_000
    reference = visible["artifact_ref"]
    assert set(reference) == {"scheme", "sha256", "bytes", "media_type"}
    assert reference == artifact_store.put(raw)
    assert artifact_store.read(reference) == raw


@pytest.mark.parametrize(
    ("tool_name", "field", "limit"),
    [
        ("web_search_tool", "results", 5),
        ("task_dispatch", "dispatches", 8),
        ("prepare_order_tool", "items", 10),
    ],
)
def test_l0_list_contracts_preserve_valid_json_and_archive_omissions(
    artifact_store: ToolArtifactStore,
    tool_name: str,
    field: str,
    limit: int,
) -> None:
    original = {field: [{"id": index, "text": "x"} for index in range(limit + 5)]}
    raw = json.dumps(original)
    visible = json.loads(
        format_tool_output(
            tool_name,
            raw,
            artifact_store=artifact_store,
            configured_limit=20_000,
        )
    )
    assert len(visible[field]) == limit
    assert artifact_store.read(visible["artifact_ref"]) == raw


def test_l1_counts_disjoint_layers_and_keeps_report_out_of_prompt() -> None:
    segment = FrozenSegment(segment_id="old", content='{"user":"旧问题"}')
    messages: list[BaseMessage] = [
        HumanMessage(content="当前问题"),
        AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c1"}]),
        ToolMessage(
            content='{"hits":[{"item_id":"i1"}]}',
            tool_call_id="c1",
            name="product_search_tool",
        ),
    ]
    hook = build_context_pre_model_hook(
        model=_ExactCounter(),
        fixed_prompt="fixed",
        tool_schemas=[{"type": "function", "function": {"name": "x"}}],
        budget_policy=ContextBudgetPolicy(
            model_context_tokens=100_000,
            soft_limit_tokens=100_000,
            mode="enforce",
        ),
    )
    update = hook(
        {
            "messages": messages,
            "frozen_segments": [segment.to_dict()],
            "freeze_cursor": 0,
            "session_context": {"revision": 2, "request": {"current_raw_query": "当前问题"}},
        }
    )
    report = update["budget_report"]

    assert report["estimated"] is False
    assert report["decision"] == "ok"
    assert set(report["layers"]) == {
        "fixed_prompt",
        "tool_schemas",
        "l4_candidate",
        "history",
        "active",
        "current",
    }
    assert list(report["tool_results"]) == ["product_search_tool:c1:2"]
    assert "budget_report" not in "".join(
        str(message.content) for message in update["llm_input_messages"]
    )
    assert "messages" not in update
    assert messages[2].content == '{"hits":[{"item_id":"i1"}]}'


def test_l1_needs_l3_with_only_recent_segment_does_not_rewrite_history() -> None:
    segment = FrozenSegment(segment_id="old", content="frozen-history")
    update = context_pre_model_hook(
        {
            "messages": [HumanMessage(content="current")],
            "frozen_segments": [segment.to_dict()],
            "freeze_cursor": 0,
        },
        model=_ExactCounter(),
        budget_policy=ContextBudgetPolicy(
            model_context_tokens=100_000,
            soft_limit_tokens=1,
            mode="enforce",
        ),
    )
    assert update["budget_decision"] == "needs_l3_no_eligible"
    assert update["frozen_segments"] == [segment.to_dict()]
    assert any("frozen-history" in str(item.content) for item in update["llm_input_messages"])


def test_l1_active_overflow_raises_without_lossy_active_compression() -> None:
    message = HumanMessage(content="active-current-is-authoritative")
    with pytest.raises(ContextBudgetExceeded) as raised:
        context_pre_model_hook(
            {"messages": [message]},
            model=_ExactCounter(),
            budget_policy=ContextBudgetPolicy(
                model_context_tokens=10,
                mode="enforce",
            ),
        )
    assert raised.value.decision == "active_overflow"
    assert message.content == "active-current-is-authoritative"


def test_l1_fixed_and_l4_overflow_is_distinct_and_estimates_stay_observe_only() -> None:
    with pytest.raises(ContextBudgetExceeded) as raised:
        context_pre_model_hook(
            {"messages": [HumanMessage(content="x")], "session_context": {"large": "L" * 20}},
            model=_ExactCounter(),
            fixed_prompt="fixed",
            budget_policy=ContextBudgetPolicy(model_context_tokens=10, mode="enforce"),
        )
    assert raised.value.decision == "fixed_overflow"

    observed = context_pre_model_hook(
        {"messages": [HumanMessage(content="x" * 10_000)]},
        budget_policy=ContextBudgetPolicy(model_context_tokens=10, mode="observe_only"),
    )
    assert observed["budget_report"]["estimated"] is True
    assert observed["budget_decision"] == "active_overflow"


async def test_budget_report_checkpoint_roundtrip_and_model_view_unchanged() -> None:
    saver = InMemorySaver()
    model = _RecordingExactModel()
    hook = build_context_pre_model_hook(
        model=model,
        fixed_prompt="fixed-system",
        tool_schemas=[],
        budget_policy=ContextBudgetPolicy(
            model_context_tokens=10_000,
            soft_limit_tokens=9_000,
            mode="enforce",
        ),
    )
    graph = create_react_agent(
        model,
        tools=[],
        prompt="fixed-system",
        pre_model_hook=hook,
        state_schema=MainAgentState,
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "phase-d-budget"}}
    await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, config=config)
    first = await graph.aget_state(config)

    rebuilt = create_react_agent(
        _RecordingExactModel(),
        tools=[],
        prompt="fixed-system",
        pre_model_hook=hook,
        state_schema=MainAgentState,
        checkpointer=saver,
    )
    restored = await rebuilt.aget_state(config)
    assert restored.values["budget_report"] == first.values["budget_report"]
    assert restored.values["budget_decision"] == "ok"
    assert [message.content for message in restored.values["messages"]] == [
        "hello",
        "bounded-answer",
    ]
    received = "".join(str(message.content) for message in model._calls[0])
    assert "budget_report" not in received
    assert "hello" in received
