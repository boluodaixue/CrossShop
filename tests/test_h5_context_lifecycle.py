"""H5 custom-context migration: AgentScope rewrites of Phase A-F semantics."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentscope.message import (
    AssistantMsg,
    SystemMsg,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)
from agentscope.state import AgentState
from agentscope.tool import ToolResponse

from app.application.context import (
    CACHE_BREAKPOINT_TEXT,
    ContextBudgetPolicy,
    FrozenSegment,
    L4Context,
    StageSummary,
    ToolArtifactStore,
    advance_context_state,
    format_tool_output,
    freeze_settled_units,
    group_interaction_units,
    reduce_l4,
    settled_prefix_units,
    summarize_frozen_segments,
)
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_middleware import (
    CONTEXT_NAMESPACE,
    ContextLifecycleMiddleware,
    ContextOwnershipError,
)


def _turn(
    query: str,
    answer: str,
    *,
    call_id: str = "call-1",
    result_state: ToolResultState = ToolResultState.SUCCESS,
) -> list:
    return [
        UserMsg(name="buyer", content=query),
        AssistantMsg(
            name="Main",
            content=[
                ToolCallBlock(
                    id=call_id,
                    name="product_search_tool",
                    input=json.dumps({"normalized_query": query, "top_k": 5}),
                    state=ToolCallState.FINISHED,
                ),
                ToolResultBlock(
                    id=call_id,
                    name="product_search_tool",
                    output=json.dumps(
                        {"hits": [{"product_id": "P-1", "skus": [{"sku_id": "S-1"}]}]},
                    ),
                    state=result_state,
                ),
                TextBlock(type="text", text=answer),
            ],
        ),
    ]


_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / ".test-context-artifacts"
)


def _middleware() -> ContextLifecycleMiddleware:
    return ContextLifecycleMiddleware(
        artifact_root=_ARTIFACT_ROOT,
        model_context_tokens=8_000,
        tool_result_limit=2_000,
    )


def test_phase_a_l4_is_idempotent_and_ignores_orphan_and_failed_results() -> None:
    intent = {
        "shopping_session_id": "session-1",
        "buyer_id": "buyer-1",
        "locale": "zh-CN",
        "currency": "CNY",
        "raw_query": "轻便背包",
    }
    events = [
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "ok",
                "args": {
                    "normalized_query": "轻便背包",
                    "platform": "taobao",
                    "top_k": 5,
                },
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "orphan",
                "raw_output": {"hits": [{"product_id": "WRONG"}]},
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "ok",
                "raw_output": {"hits": [{"product_id": "P-1"}]},
                "error": True,
            },
        },
    ]
    first = reduce_l4(intent, events, now="2026-08-28T00:00:00+00:00")
    assert first.revision == 1
    assert first.last_search is None
    replay = reduce_l4(intent, events, first, now="later")
    assert replay == first

    success = reduce_l4(
        intent,
        [events[0], {**events[2], "payload": {**events[2]["payload"], "error": False}}],
        first,
        now="2026-08-28T00:01:00+00:00",
    )
    assert success.last_search == {
        "tool_call_id": "ok",
        "args": {"normalized_query": "轻便背包", "platform": "taobao", "top_k": 5},
        "status": "succeeded",
        "returned_count": 1,
        "product_refs": ["P-1"],
    }


def test_l4_accepts_verified_structured_selection_without_parsing_prose() -> None:
    intent = {
        "shopping_session_id": "session-selection",
        "buyer_id": "buyer-selection",
        "locale": "zh-CN",
        "currency": "CNY",
        "raw_query": "找一个背包",
    }
    events = [
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-1",
                "args": {
                    "normalized_query": "背包",
                    "platform": "taobao",
                    "top_k": 5,
                },
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-1",
                "raw_output": {
                    "hits": [
                        {"product_id": "P-1", "title": "完整原标题一"},
                        {
                            "product_id": "P-2",
                            "title": "完整原标题二",
                            "highlights": [f"亮点-{index}" for index in range(12)],
                            "skus": [
                                {
                                    "sku_id": f"SKU-{index}",
                                    "spec": f"规格-{index}",
                                    "price_major": index,
                                    "currency": "CNY",
                                    "stock": 1,
                                    "internal": "drop",
                                }
                                for index in range(9)
                            ],
                            "internal_blob": "drop",
                        },
                    ],
                },
                "error": False,
            },
        },
        {
            "type": "final.result",
            "payload": {
                "text": "这段自然语言故意不含商品名",
                "structured_output": {
                    "final_text": "推荐 P-2 完整原标题二",
                    "selected_products": [
                        {
                            "platform": "taobao",
                            "product_id": "P-2",
                        },
                        {
                            "platform": "taobao",
                            "product_id": "UNKNOWN",
                        },
                    ],
                },
            },
        },
    ]

    state = reduce_l4(intent, events)

    assert state.last_recommendation is not None
    assert state.last_recommendation["product_refs"] == ["P-2"]
    assert state.last_recommendation["displayed_products"][0]["card"]["title"] == (
        "完整原标题二"
    )
    card = state.last_recommendation["displayed_products"][0]["card"]
    assert len(card["highlights"]) == 8
    assert len(card["skus"]) == 5
    assert "internal_blob" not in card
    assert "internal" not in card["skus"][0]
    assert (
        state.last_recommendation["displayed_history"]
        == (state.last_recommendation["displayed_products"])
    )


def test_l4_keeps_verified_display_history_across_category_switches() -> None:
    first = reduce_l4(
        {"shopping_session_id": "session-1", "raw_query": "找露营灯"},
        [
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "search-1",
                    "args": {
                        "normalized_query": "露营灯",
                        "category": "户外照明",
                        "platform": "globex_reference",
                    },
                },
            },
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "search-1",
                    "raw_output": {
                        "hits": [{"product_id": "P1008", "title": "露营灯"}],
                    },
                },
            },
            {
                "type": "final.result",
                "payload": {
                    "structured_output": {
                        "final_text": "推荐露营灯",
                        "selected_products": [
                            {
                                "platform": "globex_reference",
                                "product_id": "P1008",
                            },
                        ],
                    },
                },
            },
        ],
    )
    switched = reduce_l4(
        {"shopping_session_id": "session-1", "raw_query": "改找毛巾"},
        [
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "search-2",
                    "args": {
                        "normalized_query": "速干毛巾",
                        "category": "旅行装备",
                        "platform": "globex_reference",
                    },
                },
            },
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "search-2",
                    "raw_output": {
                        "hits": [{"product_id": "P1007", "title": "速干毛巾"}],
                    },
                },
            },
            {
                "type": "final.result",
                "payload": {
                    "structured_output": {
                        "final_text": "推荐毛巾",
                        "selected_products": [
                            {
                                "platform": "globex_reference",
                                "product_id": "P1007",
                            },
                        ],
                    },
                },
            },
        ],
        first,
    )

    assert switched.last_recommendation is not None
    assert switched.last_recommendation["product_refs"] == ["P1007"]
    assert [
        item["card"]["product_id"]
        for item in switched.last_recommendation["displayed_history"]
    ] == ["P1007", "P1008"]


@pytest.mark.asyncio
async def test_phase_b_request_start_persists_on_model_failure() -> None:
    middleware = _middleware()
    agent = SimpleNamespace(state=AgentState(session_id="session-1"), name="Main")
    token = ShoppingContext.set(
        ShoppingContextSnapshot("session-1", "buyer-1", "zh-CN", "CNY"),
    )

    async def fail(**_kwargs):
        raise RuntimeError("model unavailable")
        yield  # pragma: no cover

    try:
        with pytest.raises(RuntimeError, match="model unavailable"):
            async for _ in middleware.on_reply(
                agent,
                {"inputs": [UserMsg(name="buyer-1", content="找一个背包")]},
                fail,
            ):
                pass
    finally:
        ShoppingContext.reset(token)

    context = agent.state.middle_context[CONTEXT_NAMESPACE]["session_context"]
    assert context["request"]["current_raw_query"] == "找一个背包"
    assert context["last_recommendation"] is None


@pytest.mark.asyncio
async def test_phase_b_session_owner_survives_agent_state_restore() -> None:
    middleware = _middleware()
    state = AgentState(session_id="session-1")
    state.middle_context[CONTEXT_NAMESPACE] = {
        "session_context": L4Context(session={"buyer_id": "buyer-1"}).to_dict(),
    }
    restored = AgentState.model_validate_json(state.model_dump_json())
    agent = SimpleNamespace(state=restored, name="Main")
    token = ShoppingContext.set(
        ShoppingContextSnapshot("session-1", "buyer-2", "zh-CN", "CNY"),
    )

    async def unused(**_kwargs):
        yield AssistantMsg(name="Main", content="should not run")

    try:
        with pytest.raises(ContextOwnershipError):
            async for _ in middleware.on_reply(
                agent,
                {"inputs": UserMsg(name="buyer-2", content="你好")},
                unused,
            ):
                pass
    finally:
        ShoppingContext.reset(token)


def test_phase_c_call_id_pairing_and_settled_prefix_guards() -> None:
    complete = _turn("背包", "这是结果")
    units = group_interaction_units(
        [*complete, UserMsg(name="buyer", content="再看看")]
    )
    assert units[0].complete
    assert units[0].tools[0].tool_call_id == "call-1"
    assert (
        len(settled_prefix_units([*complete, UserMsg(name="buyer", content="再看看")]))
        == 1
    )
    assert not settled_prefix_units(
        [*complete, UserMsg(name="buyer", content="再看看")],
        has_pending=True,
    )

    missing = complete[1].model_copy(
        update={
            "content": [complete[1].content[0], TextBlock(type="text", text="done")]
        },
    )
    assert (
        "missing_tool_result"
        in group_interaction_units([complete[0], missing])[0].incomplete_reason
    )
    failed = _turn("背包", "失败", result_state=ToolResultState.ERROR)
    assert "failed_tool_result" in group_interaction_units(failed)[0].incomplete_reason


def test_phase_c_parallel_tool_results_pair_by_call_id() -> None:
    assistant = AssistantMsg(
        name="Main",
        content=[
            ToolCallBlock(
                id="call-a",
                name="product_search_tool",
                input='{"normalized_query":"背包"}',
                state=ToolCallState.FINISHED,
            ),
            ToolCallBlock(
                id="call-b",
                name="category_insight_tool",
                input='{"question":"背包怎么选"}',
                state=ToolCallState.FINISHED,
            ),
            ToolResultBlock(
                id="call-b",
                name="category_insight_tool",
                output='{"insights":[]}',
                state=ToolResultState.SUCCESS,
            ),
            ToolResultBlock(
                id="call-a",
                name="product_search_tool",
                output='{"hits":[]}',
                state=ToolResultState.SUCCESS,
            ),
            TextBlock(type="text", text="完成"),
        ],
    )
    messages = [
        UserMsg(name="buyer", content="找背包并告诉我怎么选"),
        assistant,
        UserMsg(name="buyer", content="继续"),
    ]
    unit = settled_prefix_units(messages)[0]
    assert unit.complete
    assert [tool.tool_call_id for tool in unit.tools] == ["call-a", "call-b"]
    assert [tool.result.id for tool in unit.tools] == ["call-a", "call-b"]


def test_phase_c_freeze_is_monotonic_and_restart_safe() -> None:
    messages = [*_turn("第一轮", "一"), UserMsg(name="buyer", content="第二轮")]
    units = settled_prefix_units(messages)
    frozen, cursor = freeze_settled_units(messages, units, context_revision=2)
    replayed = [FrozenSegment.from_dict(item.to_dict()) for item in frozen]
    again, next_cursor = freeze_settled_units(
        messages,
        units,
        existing=replayed,
        freeze_cursor=cursor,
        context_revision=2,
    )
    assert again == frozen
    assert next_cursor == cursor == 2


def test_phase_d_l0_contract_preserves_exact_artifact() -> None:
    store = ToolArtifactStore(_ARTIFACT_ROOT)
    raw = json.dumps(
        {
            "hits": [
                {"product_id": f"P-{index}", "skus": [{"sku_id": f"S-{index}"}]}
                for index in range(8)
            ],
            "unknown_internal_field": "must not enter prompt",
        },
        ensure_ascii=False,
    )
    rendered = json.loads(
        format_tool_output(
            "product_search_tool",
            raw,
            artifact_store=store,
            configured_limit=16_000,
        ),
    )
    assert len(rendered["hits"]) == 5
    assert "unknown_internal_field" not in rendered
    assert store.read(rendered["artifact_ref"]) == raw


@pytest.mark.asyncio
async def test_phase_d_on_acting_pairs_event_and_artifact() -> None:
    middleware = _middleware()
    agent = SimpleNamespace(state=AgentState(session_id="session-1"), name="Main")
    call = ToolCallBlock(
        id="call-1",
        name="product_search_tool",
        input='{"normalized_query":"背包","top_k":5}',
        state=ToolCallState.ALLOWED,
    )
    raw = '{"hits":[{"product_id":"P-1","skus":[{"sku_id":"S-1"}]}]}'

    async def handler(**_kwargs):
        yield ToolResponse(content=[TextBlock(type="text", text=raw)])

    output = []
    async for chunk in middleware.on_acting(agent, {"tool_call": call}, handler):
        output.append(chunk)
    namespace = agent.state.middle_context[CONTEXT_NAMESPACE]
    assert [event["type"] for event in namespace["current_turn_events"]] == [
        "tool.invoke",
        "tool.result",
    ]
    assert namespace["tool_artifacts"][0]["tool_call_id"] == "call-1"
    assert output[0].metadata["globex_artifact_ref"]["sha256"]


def test_phase_e_l3_is_deterministic_and_hash_verified() -> None:
    segment = FrozenSegment(
        segment_id="frozen-0-2",
        content='{"assistant":"ok","tools":[],"user":"背包"}',
        source_message_start=0,
        source_message_end=2,
        context_revision=1,
    )
    first = summarize_frozen_segments([segment])
    replay = summarize_frozen_segments([segment], existing=first.to_dict())
    assert replay == first
    tampered = first.to_dict()
    tampered["content"]["user_requests"] = ["已篡改"]
    with pytest.raises(ValueError, match="content hash mismatch"):
        StageSummary.from_dict(tampered)


def test_phase_f_assembler_is_non_destructive_and_has_logical_breakpoint() -> None:
    messages = [*_turn("第一轮", "一"), UserMsg(name="buyer", content="第二轮")]
    snapshot = [message.model_dump_json() for message in messages]
    state, model_view = advance_context_state(
        context_messages=messages,
        namespace={
            "session_context": L4Context(
                request={"current_raw_query": "第二轮"}
            ).to_dict()
        },
        fixed_messages=[],
        tool_schemas=[],
        policy=ContextBudgetPolicy(model_context_tokens=8_000, soft_limit_tokens=7_000),
        has_pending=False,
        has_interrupt=False,
    )
    assert [message.model_dump_json() for message in messages] == snapshot
    assert state["freeze_cursor"] == 2
    assert any(
        CACHE_BREAKPOINT_TEXT[:-3] in (message.get_text_content() or "")
        for message in model_view
    )
    assert model_view[-1].get_text_content() == "第二轮"


@pytest.mark.asyncio
async def test_phase_f_builtin_compression_does_not_delete_raw_context() -> None:
    middleware = _middleware()
    state = AgentState(session_id="session-1", context=_turn("背包", "结果"))
    agent = SimpleNamespace(state=state, name="Main")
    before = state.model_dump_json()

    async def destructive(**_kwargs):  # pragma: no cover - must not be called
        state.context.clear()

    await middleware.on_compress_context(agent, {}, destructive)
    assert state.context
    assert json.loads(before)["context"] == state.model_dump(mode="json")["context"]
    assert state.middle_context[CONTEXT_NAMESPACE]["builtin_compression_skipped"] == 1


@pytest.mark.asyncio
async def test_phase_f_on_model_call_uses_view_and_persists_middle_context() -> None:
    middleware = _middleware()
    state = AgentState(
        session_id="session-1",
        context=[*_turn("第一轮", "一"), UserMsg(name="buyer", content="第二轮")],
    )
    state.middle_context[CONTEXT_NAMESPACE] = {
        "session_context": L4Context(
            request={"current_raw_query": "第二轮"},
        ).to_dict(),
    }
    agent = SimpleNamespace(state=state, name="Main")
    before = [message.model_dump_json() for message in state.context]
    captured = {}

    class _Model:
        async def count_tokens(self, messages, tools):
            del tools
            return len(messages)

    async def handler(**kwargs):
        captured.update(kwargs)
        return "model-result"

    result = await middleware.on_model_call(
        agent,
        {
            "messages": [SystemMsg(name="system", content="prompt"), *state.context],
            "tools": [],
            "tool_choice": None,
            "current_model": _Model(),
        },
        handler,
    )
    assert result == "model-result"
    assert [message.model_dump_json() for message in state.context] == before
    assert any(
        CACHE_BREAKPOINT_TEXT[:-3] in (message.get_text_content() or "")
        for message in captured["messages"]
    )
    restored = AgentState.model_validate_json(state.model_dump_json())
    assert restored.middle_context[CONTEXT_NAMESPACE]["freeze_cursor"] == 2


def test_phase_f_soft_threshold_builds_l3_but_keeps_recent_l2() -> None:
    messages = [
        *_turn("第一轮", "一" * 2_000, call_id="call-1"),
        *_turn("第二轮", "二", call_id="call-2"),
        UserMsg(name="buyer", content="第三轮"),
    ]
    state, model_view = advance_context_state(
        context_messages=messages,
        namespace={"session_context": L4Context(revision=2).to_dict()},
        fixed_messages=[],
        tool_schemas=[],
        policy=ContextBudgetPolicy(
            model_context_tokens=8_000,
            soft_limit_tokens=1,
            l3_keep_recent_frozen_segments=1,
        ),
        has_pending=False,
        has_interrupt=False,
    )
    assert state["stage_summary"]["source_segment_ids"] == ["frozen-0-2"]
    assert state["frozen_segments"][1]["segment_id"] == "frozen-2-4"
    assert any(
        "<stage-summary>" in (message.get_text_content() or "")
        for message in model_view
    )
    assert any(
        "frozen-2-4" in (message.get_text_content() or "") for message in model_view
    )


def test_phase_f_preference_hints_do_not_break_multiturn_l2_l3() -> None:
    first = _turn("第一轮真实需求", "一" * 2_000, call_id="call-1")
    second = _turn("第二轮真实需求", "二", call_id="call-2")
    messages = [
        UserMsg(
            name="memory_hint",
            content="<buyer-preferences>偏好 A</buyer-preferences>",
        ),
        *first,
        UserMsg(
            name="memory_hint",
            content="<buyer-preferences>偏好 B</buyer-preferences>",
        ),
        *second,
        UserMsg(
            name="memory_hint",
            content="<buyer-preferences>偏好 C</buyer-preferences>",
        ),
        UserMsg(name="buyer", content="第三轮真实需求"),
    ]

    units = group_interaction_units(messages)
    assert len(units) == 3
    assert [unit.human.get_text_content() for unit in units] == [
        "第一轮真实需求",
        "第二轮真实需求",
        "第三轮真实需求",
    ]
    assert [unit.start for unit in units] == [0, 3, 6]
    assert all(unit.human.name != "memory_hint" for unit in units)

    state, model_view = advance_context_state(
        context_messages=messages,
        namespace={"session_context": L4Context(revision=2).to_dict()},
        fixed_messages=[],
        tool_schemas=[],
        policy=ContextBudgetPolicy(
            model_context_tokens=8_000,
            soft_limit_tokens=1,
            l3_keep_recent_frozen_segments=1,
        ),
        has_pending=False,
        has_interrupt=False,
    )
    assert state["freeze_cursor"] == 6
    assert [item["segment_id"] for item in state["frozen_segments"]] == [
        "frozen-0-3",
        "frozen-3-6",
    ]
    assert state["stage_summary"]["source_segment_ids"] == ["frozen-0-3"]
    # Current hint and buyer request keep their original order in the model view.
    assert [message.name for message in model_view[-2:]] == ["memory_hint", "buyer"]
    assert model_view[-1].get_text_content() == "第三轮真实需求"
