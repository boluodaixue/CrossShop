"""H5 custom-context migration: AgentScope rewrites of Phase A-F semantics."""

from __future__ import annotations

import json
from contextlib import contextmanager
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

from app.application.agents.orchestrator import MainAgentOrchestrator
from app.application.context import (
    CACHE_BREAKPOINT_TEXT,
    ContextBudgetPolicy,
    FrozenSegment,
    L4Context,
    StageSummary,
    ToolArtifactStore,
    advance_context_state,
    assemble_llm_input_messages,
    format_tool_output,
    freeze_settled_units,
    group_interaction_units,
    reduce_l4,
    settled_prefix_units,
    summarize_frozen_segments,
)
from app.application.context.assembler import _summary_rebuild_inputs
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_middleware import (
    CONTEXT_NAMESPACE,
    ContextLifecycleMiddleware,
    ContextOwnershipError,
)
from app.infrastructure.eventbus import TradeEventBus


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


def _structured_only_turn(
    query: str,
    final_text: str,
    *,
    call_id: str = "structured-1",
    result_state: ToolResultState = ToolResultState.SUCCESS,
) -> list:
    return [
        UserMsg(name="buyer", content=query),
        AssistantMsg(
            name="Main",
            content=[
                ToolCallBlock(
                    id=call_id,
                    name="GenerateStructuredOutput",
                    input=json.dumps(
                        {"final_text": final_text, "selected_products": []},
                        ensure_ascii=False,
                    ),
                    state=ToolCallState.FINISHED,
                ),
                ToolResultBlock(
                    id=call_id,
                    name="GenerateStructuredOutput",
                    output="Structured output generated successfully.",
                    state=result_state,
                ),
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


def test_custom_l3_transition_publishes_one_privacy_safe_event() -> None:
    bus = TradeEventBus()
    queue = bus.subscribe("session-1")
    orchestrator = object.__new__(MainAgentOrchestrator)
    orchestrator._bus = bus  # noqa: SLF001
    agent = SimpleNamespace(
        state=SimpleNamespace(
            summary=None,
            context=["raw-1", "raw-2"],
            middle_context={
                CONTEXT_NAMESPACE: {
                    "stage_summary": {
                        "source_segment_ids": ["frozen-0-2"],
                        "summary_hash": "a" * 64,
                    },
                    "context_compression": {
                        "action": "l3_stage_summary",
                        "before_tokens": 120,
                        "after_tokens": 80,
                    },
                },
            },
        ),
    )

    orchestrator._publish_compression(  # noqa: SLF001
        "session-1",
        agent,
        None,
        None,
    )

    event = queue.get_nowait()
    assert event.type == "context.compressed"
    assert event.payload == {
        "action": "l3_stage_summary",
        "before_tokens": 120,
        "after_tokens": 80,
        "source_segment_count": 1,
        "summary_hash": "a" * 64,
    }

    orchestrator._publish_compression(  # noqa: SLF001
        "session-1",
        agent,
        None,
        "a" * 64,
    )
    assert queue.empty()


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
                    "platform": "reference_seed",
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
        "args": {"normalized_query": "轻便背包", "platform": "reference_seed", "top_k": 5},
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
                    "platform": "reference_seed",
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
                            "platform": "reference_seed",
                            "product_id": "P-2",
                        },
                        {
                            "platform": "reference_seed",
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


def test_l4_keeps_bounded_order_lines_and_created_at() -> None:
    state = reduce_l4(
        {"shopping_session_id": "session-order", "raw_query": "查订单"},
        [
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": "query_order_tool",
                    "tool_call_id": "query-1",
                    "args": {"order_id": "GBX-1"},
                },
            },
            {
                "type": "tool.result",
                "payload": {
                    "tool": "query_order_tool",
                    "tool_call_id": "query-1",
                    "raw_output": {
                        "order_id": "GBX-1",
                        "status": "CONFIRMED",
                        "currency": "CNY",
                        "total_amount_major": 89.0,
                        "created_at": "2026-08-31T00:00:00+00:00",
                        "shipping_address": "must not enter L4",
                        "lines": [
                            {
                                "product_id": "P1008",
                                "sku_id": "P1008-S1",
                                "title": "LumenGo 便携露营灯 可充电",
                                "unit_price_major": 89.0,
                                "quantity": 1,
                            }
                        ],
                    },
                },
            },
        ],
    )

    assert state.order == {
        "order_id": "GBX-1",
        "status": "CONFIRMED",
        "currency": "CNY",
        "total_amount_major": 89.0,
        "created_at": "2026-08-31T00:00:00+00:00",
        "items": [
            {
                "product_id": "P1008",
                "sku_id": "P1008-S1",
                "title": "LumenGo 便携露营灯 可充电",
                "unit_price_major": 89.0,
                "quantity": 1,
            }
        ],
    }


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
                        "platform": "crossshop_reference",
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
                                "platform": "crossshop_reference",
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
                        "platform": "crossshop_reference",
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
                                "platform": "crossshop_reference",
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


def test_l4_evicts_oldest_after_twenty_and_refreshes_same_product_fact() -> None:
    state = L4Context()
    for index in range(22):
        product_id = f"P-{index:02d}"
        state = reduce_l4(
            {
                "shopping_session_id": "session-1",
                "raw_query": f"查找商品 {product_id}",
            },
            [
                {
                    "type": "tool.invoke",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": f"search-{index}",
                        "args": {
                            "normalized_query": product_id,
                            "platform": "crossshop_reference",
                        },
                    },
                },
                {
                    "type": "tool.result",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": f"search-{index}",
                        "raw_output": {
                            "hits": [
                                {
                                    "product_id": product_id,
                                    "title": f"商品 {index}",
                                    "price_major": index + 1,
                                    "currency": "CNY",
                                },
                            ],
                        },
                    },
                },
                {
                    "type": "final.result",
                    "payload": {
                        "structured_output": {
                            "final_text": f"推荐 {product_id}",
                            "selected_products": [
                                {
                                    "platform": "crossshop_reference",
                                    "product_id": product_id,
                                },
                            ],
                        },
                    },
                },
            ],
            state,
        )

    recommendation = state.last_recommendation or {}
    history = recommendation["displayed_history"]
    history_ids = [item["card"]["product_id"] for item in history]
    assert recommendation["product_refs"] == ["P-21"]
    assert len(history_ids) == 20
    assert history_ids == [f"P-{index:02d}" for index in range(21, 1, -1)]

    refreshed = reduce_l4(
        {"shopping_session_id": "session-1", "raw_query": "重新查看 P-10"},
        [
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "search-refresh",
                    "args": {
                        "normalized_query": "P-10",
                        "platform": "crossshop_reference",
                    },
                },
            },
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "search-refresh",
                    "raw_output": {
                        "hits": [
                            {
                                "product_id": "P-10",
                                "title": "商品 10 新快照",
                                "price_major": 999,
                                "currency": "CNY",
                            },
                        ],
                    },
                },
            },
            {
                "type": "final.result",
                "payload": {
                    "structured_output": {
                        "final_text": "重新展示 P-10",
                        "selected_products": [
                            {
                                "platform": "crossshop_reference",
                                "product_id": "P-10",
                            },
                        ],
                    },
                },
            },
        ],
        state,
    )
    refreshed_history = (refreshed.last_recommendation or {})["displayed_history"]
    assert refreshed_history[0]["card"]["product_id"] == "P-10"
    assert refreshed_history[0]["card"]["price_major"] == 999
    assert sum(item["card"]["product_id"] == "P-10" for item in refreshed_history) == 1


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
    failed_unit = group_interaction_units(failed)[0]
    assert failed_unit.complete
    assert failed_unit.tools[0].result.state == ToolResultState.ERROR


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


def test_phase_d_order_contract_keeps_domain_lines() -> None:
    store = ToolArtifactStore(_ARTIFACT_ROOT)
    rendered = json.loads(
        format_tool_output(
            "query_order_tool",
            {
                "order_id": "GBX-1",
                "status": "CONFIRMED",
                "created_at": "2026-08-31T00:00:00+00:00",
                "lines": [
                    {
                        "product_id": "P1008",
                        "sku_id": "P1008-S1",
                        "title": "LumenGo 便携露营灯 可充电",
                        "unit_price_major": 89.0,
                        "quantity": 1,
                    }
                ],
            },
            artifact_store=store,
            configured_limit=8_000,
        ),
    )

    assert rendered["lines"][0]["product_id"] == "P1008"
    assert rendered["lines"][0]["quantity"] == 1
    assert rendered["created_at"] == "2026-08-31T00:00:00+00:00"


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
    assert output[0].metadata["crossshop_artifact_ref"]["sha256"]


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
    assert len(state["budget_report"]["stable_prefix_sha256"]) == 64
    assert state["budget_report"]["stable_prefix_estimated_tokens"] > 0
    assert any(
        CACHE_BREAKPOINT_TEXT[:-3] in (message.get_text_content() or "")
        for message in model_view
    )
    assert model_view[-1].get_text_content() == "第二轮"


def test_phase_f_breakpoint_hash_ignores_agentscope_runtime_ids() -> None:
    frozen = FrozenSegment(
        segment_id="frozen-0-2",
        content='{"assistant":"回答","tools":[],"user":"问题"}',
        source_message_start=0,
        source_message_end=2,
    )
    kwargs = {
        "fixed_messages": [SystemMsg(name="system", content="固定提示")],
        "active_messages": [UserMsg(name="buyer", content="当前问题")],
        "frozen_segments": [frozen],
        "session_context": L4Context(revision=1).to_dict(),
    }

    first = assemble_llm_input_messages(**kwargs)
    second = assemble_llm_input_messages(**kwargs)
    first_breakpoint = first[2].get_text_content()
    second_breakpoint = second[2].get_text_content()

    assert first_breakpoint == second_breakpoint
    assert "prefix_sha256=" in first_breakpoint


def test_structured_output_only_turn_is_complete_without_mutating_raw_state() -> None:
    messages = [
        *_structured_only_turn("第一轮", "结构化最终回答"),
        UserMsg(name="buyer", content="第二轮"),
    ]
    snapshot = [message.model_dump_json() for message in messages]

    units = settled_prefix_units(messages)

    assert len(units) == 1
    assert units[0].complete is True
    assert units[0].final_answer is not None
    assert units[0].final_answer.get_text_content() == "结构化最终回答"
    assert [message.model_dump_json() for message in messages] == snapshot


@pytest.mark.parametrize(
    ("final_text", "result_state"),
    [
        ("", ToolResultState.SUCCESS),
        ("不会被接受", ToolResultState.ERROR),
    ],
)
def test_structured_output_projection_rejects_empty_or_failed_results(
    final_text: str,
    result_state: ToolResultState,
) -> None:
    messages = [
        *_structured_only_turn(
            "第一轮",
            final_text,
            result_state=result_state,
        ),
        UserMsg(name="buyer", content="第二轮"),
    ]

    assert settled_prefix_units(messages) == []


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
    # Complete previous interaction freezes; current buyer turn stays hot.
    assert restored.middle_context[CONTEXT_NAMESPACE]["freeze_cursor"] == 2


@pytest.mark.asyncio
async def test_phase_f_model_call_exports_privacy_safe_prompt_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    observed: dict[str, object] = {}

    @contextmanager
    def _capture_span(name, attributes):
        observed["name"] = name
        observed["attributes"] = dict(attributes)
        yield SimpleNamespace()

    monkeypatch.setattr(
        "app.infrastructure.context_middleware.trace_span",
        _capture_span,
    )

    class _Model:
        async def count_tokens(self, messages, tools):
            del tools
            return len(messages)

    async def handler(**_kwargs):
        return "model-result"

    await middleware.on_model_call(
        agent,
        {
            "messages": [
                SystemMsg(name="system", content="private prompt"),
                *state.context,
            ],
            "tools": [{"name": "private_tool", "description": "private schema"}],
            "tool_choice": None,
            "current_model": _Model(),
        },
        handler,
    )

    assert observed["name"] == "crossshop.prompt.breakdown"
    attributes = observed["attributes"]
    assert attributes["crossshop.prompt.agent"] == "Main"
    for key in (
        "crossshop.prompt.system_tokens",
        "crossshop.prompt.tool_schema_tokens",
        "crossshop.prompt.recent_history_tokens",
        "crossshop.prompt.frozen_context_tokens",
        "crossshop.prompt.tool_result_tokens",
        "crossshop.prompt.total_estimated_tokens",
        "crossshop.prompt.stable_prefix_estimated_tokens",
    ):
        assert isinstance(attributes[key], int)
        assert attributes[key] >= 0
    assert len(attributes["crossshop.prompt.stable_prefix_sha256"]) == 64
    serialized = json.dumps(attributes, ensure_ascii=False)
    assert "private prompt" not in serialized
    assert "private schema" not in serialized
    assert "第一轮" not in serialized
    assert "第二轮" not in serialized


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


def test_phase_f_repeated_l3_grows_verified_sources_without_duplicates() -> None:
    policy = ContextBudgetPolicy(
        model_context_tokens=8_000,
        soft_limit_tokens=1,
        l3_keep_recent_frozen_segments=1,
    )
    first_messages = [
        *_turn("第一轮", "一" * 2_000, call_id="call-1"),
        *_turn("第二轮", "二" * 2_000, call_id="call-2"),
        UserMsg(name="buyer", content="第三轮"),
    ]
    first_state, _ = advance_context_state(
        context_messages=first_messages,
        namespace={"session_context": L4Context(revision=2).to_dict()},
        fixed_messages=[],
        tool_schemas=[],
        policy=policy,
        has_pending=False,
        has_interrupt=False,
    )
    first_summary = StageSummary.from_dict(first_state["stage_summary"])
    assert first_summary.source_segment_ids == ("frozen-0-2",)

    second_messages = [
        *_turn("第一轮", "一" * 2_000, call_id="call-1"),
        *_turn("第二轮", "二" * 2_000, call_id="call-2"),
        *_turn("第三轮", "三" * 2_000, call_id="call-3"),
        *_turn("第四轮", "四" * 2_000, call_id="call-4"),
        UserMsg(name="buyer", content="第五轮"),
    ]
    second_state, second_view = advance_context_state(
        context_messages=second_messages,
        namespace=first_state,
        fixed_messages=[],
        tool_schemas=[],
        policy=policy,
        has_pending=False,
        has_interrupt=False,
    )
    second_summary = StageSummary.from_dict(second_state["stage_summary"])

    assert second_summary.source_segment_ids == (
        "frozen-0-2",
        "frozen-2-4",
        "frozen-4-6",
    )
    assert len(second_summary.source_segment_ids) == len(
        set(second_summary.source_segment_ids)
    )
    assert second_summary.summary_hash != first_summary.summary_hash
    visible_summaries = [
        message.get_text_content() or ""
        for message in second_view
        if "<stage-summary>" in (message.get_text_content() or "")
    ]
    assert len(visible_summaries) == 1
    assert second_summary.summary_hash in visible_summaries[0]
    assert first_summary.summary_hash not in visible_summaries[0]
    assert '"source_segments"' not in visible_summaries[0]
    assert second_state["context_compression"]["action"] == "l3_stage_summary"
    assert (
        second_state["context_compression"]["after_tokens"]
        < second_state["context_compression"]["before_tokens"]
    )

    restored = AgentState.model_validate_json(
        AgentState(
            session_id="session-repeat",
            context=second_messages,
            middle_context={CONTEXT_NAMESPACE: second_state},
        ).model_dump_json(),
    )
    restored_summary = StageSummary.from_dict(
        restored.middle_context[CONTEXT_NAMESPACE]["stage_summary"],
    )
    assert restored_summary == second_summary


def _summary_segment(
    index: int,
    *,
    user: str | None = None,
    tools: list[dict] | None = None,
    assistant: str | None = None,
) -> FrozenSegment:
    return FrozenSegment(
        segment_id=f"summary-source-{index}",
        content=json.dumps(
            {
                "user": user or f"第 {index} 轮需求",
                "tools": tools or [],
                "assistant": assistant or f"第 {index} 轮结果",
            },
            ensure_ascii=False,
        ),
        source_message_start=index * 2,
        source_message_end=index * 2 + 2,
        context_revision=index,
    )


@pytest.mark.parametrize("generations", [1, 2, 3, 5])
def test_l3_recompression_keeps_one_bounded_summary_and_complete_manifest(
    generations: int,
) -> None:
    summary = None
    all_segments: list[FrozenSegment] = []
    for generation in range(generations):
        new_segments = [
            _summary_segment(
                generation * 20 + index,
                user=(f"阶段 {generation} 事实 {index}: " + ("需求证据" * 30)),
                assistant=(f"阶段 {generation} 结果 {index}: " + ("结果证据" * 30)),
            )
            for index in range(20)
        ]
        all_segments.extend(new_segments)
        summary = summarize_frozen_segments(
            new_segments,
            existing=summary,
            target_tokens=8_000,
            hard_limit_tokens=12_000,
        )

    assert summary is not None
    assert summary.generation == generations
    assert summary.estimated_tokens <= 12_000
    assert summary.source_segment_ids == tuple(item.segment_id for item in all_segments)
    assert [item["content_hash"] for item in summary.source_segments] == [
        item.content_hash for item in all_segments
    ]
    prompt = summary.to_prompt_dict()
    assert prompt["source_segment_count"] == len(all_segments)
    assert "source_segments" not in prompt
    assert prompt["summary_budget"]["refinement"].startswith("semantic_journey_")
    assert len(summary.shopping_journeys) == 1
    assert len(summary.shopping_journeys[0]["focus"]) == generations * 20 - 1


def test_l3_recompression_applies_newest_state_and_never_invents_facts() -> None:
    placed = _summary_segment(
        0,
        user="预算=1000；改成商品 P-OLD",
        tools=[
            {
                "tool": "order_status_tool",
                "tool_call_id": "placed",
                "args": {"order_id": "ORDER-7"},
                "status": "placed",
                "refs": ["ORDER-7"],
            }
        ],
    )
    canceled = _summary_segment(
        1,
        user="预算=600；换成商品 P-NEW，并排除红色",
        tools=[
            {
                "tool": "order_status_tool",
                "tool_call_id": "canceled",
                "args": {"order_id": "ORDER-7"},
                "status": "canceled",
                "refs": ["ORDER-7"],
            }
        ],
    )

    old = summarize_frozen_segments([placed])
    current = summarize_frozen_segments([canceled], existing=old)

    assert current.user_requests == ()
    assert current.shopping_journeys[0]["request"] == (
        "预算=600；换成商品 P-NEW，并排除红色"
    )
    assert current.shopping_journeys[0]["budget_max_major"] == 600
    assert len(current.tool_facts) == 1
    assert current.tool_facts[0]["status"] == "canceled"
    assert current.tool_facts[0]["tool_call_ids"] == ["placed", "canceled"]
    serialized = json.dumps(current.to_prompt_dict(), ensure_ascii=False)
    assert "P-OLD" not in serialized
    assert "ORDER-FAKE" not in serialized
    assert "session-context is authoritative" in serialized


def test_l3_excludes_structured_output_prose_and_keeps_selection_in_journey() -> None:
    repeated_prose = "这段最终回答不应作为工具事实重复进入摘要。" * 100
    segment = _summary_segment(
        0,
        user="切换到露营折叠椅，只看 CrossShop，预算500元，找2款候选。",
        tools=[
            {
                "tool": "product_search_tool",
                "tool_call_id": "search-chair",
                "args": {
                    "normalized_query": "露营折叠椅",
                    "category": "露营折叠椅",
                    "platform": "crossshop_reference",
                    "price_max_major": 500,
                    "target_currency": "CNY",
                },
                "status": "success",
                "refs": ["P-CHAIR-1", "P-CHAIR-2"],
            },
            {
                "tool": "GenerateStructuredOutput",
                "tool_call_id": "structured-chair",
                "args": {
                    "final_text": repeated_prose,
                    "selected_products": [
                        {"platform": "crossshop_reference", "product_id": "P-CHAIR-1"}
                    ],
                },
                "status": "success",
            },
        ],
        assistant="1. P-CHAIR-1 | 舒适露营折叠椅\n价格：399 CNY。",
    )

    summary = summarize_frozen_segments([segment])
    serialized = json.dumps(summary.to_prompt_dict(), ensure_ascii=False)

    assert "GenerateStructuredOutput" not in serialized
    assert repeated_prose not in serialized
    assert summary.omitted_counts["duplicate_structured_outputs"] == 1
    assert summary.shopping_journeys[0]["product_refs"] == [
        "P-CHAIR-1",
        "P-CHAIR-2",
    ]
    assert summary.shopping_journeys[0]["anchor_product"] == {
        "product_id": "P-CHAIR-1",
        "title": "舒适露营折叠椅",
        "price": "399",
        "currency": "CNY",
    }


def test_l3_recompression_migrates_legacy_summary_without_raw_history_replay() -> None:
    old_segment = _summary_segment(0, user="早期事实")
    legacy = StageSummary(
        source_segments=(
            {
                "segment_id": old_segment.segment_id,
                "content_hash": old_segment.content_hash,
            },
        ),
        user_requests=("早期事实",),
        tool_facts=(),
        assistant_outcomes=("早期结果",),
        source_message_start=0,
        source_message_end=2,
        schema_version="stage-summary-v1",
    )

    migrated = summarize_frozen_segments(
        [_summary_segment(1, user="近期事实")],
        existing=legacy,
    )

    assert migrated.schema_version == "stage-summary-v4"
    assert migrated.generation == 2
    assert migrated.user_requests == ()
    assert migrated.shopping_journeys[0]["request"] == "早期事实"
    assert migrated.shopping_journeys[0]["focus"] == ["近期事实"]
    assert migrated.source_segment_ids == (
        "summary-source-0",
        "summary-source-1",
    )


def test_legacy_schema_upgrade_rebuilds_exact_authored_turns_from_l2() -> None:
    segments: list[FrozenSegment] = []
    actual_index = 0
    for authored_turn in range(1, 92):
        if authored_turn == 62:
            segments.append(
                _summary_segment(
                    actual_index,
                    user="准确告诉我本次会话第 61 轮讨论了什么。",
                )
            )
            actual_index += 1
        user = "继续比较"
        tools = None
        if authored_turn == 1:
            user = "先看通勤背包，预算300元。"
        elif authored_turn == 86:
            user = "最后看轻量冲锋衣，只看京东，预算800元。"
        elif authored_turn == 91:
            user = "切换到露营折叠椅，只看 CrossShop，预算500元。"
            tools = [
                {
                    "tool": "product_search_tool",
                    "args": {
                        "normalized_query": "露营折叠椅",
                        "platform": "crossshop_reference",
                        "price_max_major": 500,
                        "target_currency": "CNY",
                    },
                    "status": "success",
                    "refs": ["P-CHAIR-1"],
                }
            ]
        segments.append(_summary_segment(actual_index, user=user, tools=tools))
        actual_index += 1

    legacy = StageSummary(
        source_segments=tuple(
            {
                "segment_id": item.segment_id,
                "content_hash": item.content_hash,
            }
            for item in segments
        ),
        user_requests=(
            "最后看轻量冲锋衣，只看京东，预算800元。",
            "切换到露营折叠椅，只看 CrossShop，预算500元。",
        ),
        # This deliberately unrelated legacy fact must not be positionally
        # attached to either sampled request during migration.
        tool_facts=(
            {
                "tool": "product_search_tool",
                "args": {
                    "platform": "amazon",
                    "price_max_major": 150,
                    "target_currency": "USD",
                },
                "status": "success",
            },
        ),
        assistant_outcomes=(),
        shopping_journeys=(
            {
                "journey_id": "journey-corrupted-v3",
                "request": "切换到露营折叠椅，只看 CrossShop，预算500元。",
                "authored_turn_start": 2,
                "platform": "amazon",
                "budget_max_major": 150,
            },
        ),
        schema_version="stage-summary-v3",
    )

    rebuild_sources, rebuild_existing = _summary_rebuild_inputs(
        frozen=segments,
        eligible=(),
        existing=legacy,
    )
    rebuilt = summarize_frozen_segments(
        rebuild_sources,
        existing=rebuild_existing,
    )

    jacket = next(
        item for item in rebuilt.shopping_journeys if "冲锋衣" in item["request"]
    )
    chair = next(
        item for item in rebuilt.shopping_journeys if "折叠椅" in item["request"]
    )
    assert jacket["authored_turn_start"] == 86
    assert jacket["platform"] == "jd"
    assert jacket["budget_max_major"] == 800
    assert chair["authored_turn_start"] == 91
    assert chair["platform"] == "crossshop_reference"
    assert chair["budget_max_major"] == 500
    assert chair["currency"] == "CNY"
    assert chair["product_refs"] == ["P-CHAIR-1"]
    serialized = json.dumps(rebuilt.to_prompt_dict(), ensure_ascii=False)
    assert "准确告诉我本次会话" not in serialized
    assert '"platform": "amazon"' not in serialized
    assert "authored_turn_start/end" in serialized


def test_l3_semantic_summary_keeps_every_journey_core_without_sampling() -> None:
    segments = [
        _summary_segment(
            index,
            user=(f"切换到品类-{index}，只看 CrossShop，预算{100 + index}元，找2款候选。"),
            tools=[
                {
                    "tool": "product_search_tool",
                    "tool_call_id": f"search-{index}",
                    "args": {
                        "normalized_query": f"品类-{index}",
                        "category": f"品类-{index}",
                        "platform": "crossshop_reference",
                        "price_max_major": 100 + index,
                        "target_currency": "CNY",
                    },
                    "status": "success",
                    "refs": [f"P-{index}"],
                }
            ],
            assistant=(f"1. P-{index} | 商品-{index}\n价格：{100 + index} CNY。"),
        )
        for index in range(30)
    ]

    summary = summarize_frozen_segments(
        segments,
        target_tokens=8_000,
        hard_limit_tokens=12_000,
    )

    assert len(summary.shopping_journeys) == 30
    assert [item["category"] for item in summary.shopping_journeys] == [
        f"品类-{index}" for index in range(30)
    ]
    assert [item["budget_max_major"] for item in summary.shopping_journeys] == [
        100 + index for index in range(30)
    ]
    assert summary.omitted_counts == {"duplicate_structured_outputs": 0}
    assert summary.estimated_tokens <= 8_000


def test_l3_hard_limit_failure_is_explicit() -> None:
    segments = [
        _summary_segment(
            index,
            user=f"切换到不可丢弃品类-{index}，找2款候选。",
            tools=[
                {
                    "tool": "product_search_tool",
                    "tool_call_id": f"hard-{index}",
                    "args": {"normalized_query": f"不可丢弃品类-{index}"},
                    "status": "success",
                    "refs": [f"P-{index}"],
                }
            ],
        )
        for index in range(20)
    ]

    with pytest.raises(RuntimeError, match="above hard limit"):
        summarize_frozen_segments(
            segments,
            target_tokens=100,
            hard_limit_tokens=150,
        )


@pytest.mark.parametrize("failure", ["generation", "schema", "hard_limit"])
def test_l3_failure_keeps_previous_valid_summary_and_model_view(failure: str) -> None:
    first = _summary_segment(0)
    previous = summarize_frozen_segments([first])
    messages = [
        *_turn("第一轮", "一" * 2_000, call_id="call-1"),
        *_turn("第二轮", "二" * 2_000, call_id="call-2"),
        UserMsg(name="buyer", content="第三轮"),
    ]
    namespace = {
        "frozen_segments": [first.to_dict()],
        "freeze_cursor": 2,
        "stage_summary": previous.to_dict(),
        "session_context": L4Context(revision=2).to_dict(),
    }

    def broken_builder(*_args, **_kwargs):
        if failure == "generation":
            raise RuntimeError("offline generator failed")
        if failure == "schema":
            return {"schema_version": "invalid"}
        return summarize_frozen_segments(
            _args[0],
            existing=_kwargs.get("existing"),
            target_tokens=50,
            hard_limit_tokens=60,
        )

    state, model_view = advance_context_state(
        context_messages=messages,
        namespace=namespace,
        fixed_messages=[],
        tool_schemas=[],
        policy=ContextBudgetPolicy(
            model_context_tokens=8_000,
            soft_limit_tokens=1,
            l3_keep_recent_frozen_segments=0,
            summary_target_tokens=100,
            summary_hard_limit_tokens=150,
        ),
        has_pending=False,
        has_interrupt=False,
        summary_builder=broken_builder,
    )

    assert state["context_compression"]["action"] == "l3_failed"
    assert state["stage_summary"]["summary_hash"] == previous.summary_hash
    visible_summaries = [
        message.get_text_content() or ""
        for message in model_view
        if "<stage-summary>" in (message.get_text_content() or "")
    ]
    assert len(visible_summaries) == 1
    assert previous.summary_hash in visible_summaries[0]


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
