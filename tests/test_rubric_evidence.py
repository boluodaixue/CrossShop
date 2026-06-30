"""Tests for privacy-safe local EventBus and span normalization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry.trace import Status, StatusCode

from app.infrastructure.eventbus import TradeEvent, TradeEventBus
from scripts.eval.rubric_evidence import (
    EvidenceNormalizationError,
    build_evaluation_evidence,
    build_structured_state_evidence,
    build_turn_evidence,
    normalize_trace_spans,
)
from scripts.eval.rubric_contract import PreferenceStateEvidence


def _event(event_type: str, payload: dict) -> dict:
    return {"type": event_type, "payload": payload, "occurred_at": "now"}


def _preference(count: int = 0, marker: str = "0") -> PreferenceStateEvidence:
    return PreferenceStateEvidence(count=count, content_hash=marker * 64)


def _state():
    return build_structured_state_evidence(
        {},
        preference_before=_preference(),
        preference_after=_preference(),
    )


def _span(
    name: str,
    *,
    trace_id: int = 1,
    span_id: int = 2,
    parent_id: int | None = None,
    attributes: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        context=SimpleNamespace(trace_id=trace_id, span_id=span_id),
        parent=(SimpleNamespace(span_id=parent_id) if parent_id is not None else None),
        status=Status(StatusCode.OK),
        start_time=1_000_000,
        end_time=3_500_000,
        attributes=attributes or {},
    )


def test_direct_product_search_keeps_safe_args_and_product_id_summaries() -> None:
    turn = build_turn_evidence(
        turn_index=1,
        user_input="电话 13800138000，只看淘宝露营灯",
        final_text="联系 13800138000，推荐第一件商品",
        displayed_products=[
            {
                "rank": 1,
                "platform": "taobao",
                "site_locale": None,
                "card": {"product_id": "taobao:1", "title": "露营灯"},
            },
        ],
        events=[
            _event(
                "tool.invoke",
                {
                    "tool": "product_search_tool",
                    "args": {
                        "normalized_query": "露营灯",
                        "platform": "taobao",
                        "shipping_address": "不应保留",
                    },
                },
            ),
            _event(
                "tool.result",
                {
                    "tool": "product_search_tool",
                    "platform": "taobao",
                    "hit_count": 1,
                    "filtered_count": 1,
                    "recall_strategy": "hybrid_rrf",
                    "hits": [
                        {
                            "product_id": "taobao:1",
                            "title": "露营灯",
                            "price_major": 89,
                            "currency": "CNY",
                            "landed_price": {
                                "subtotal_major": 89,
                                "freight_major": 25,
                                "tariff_major": 0,
                                "landed_total_major": 114,
                                "currency": "CNY",
                            },
                            "description": "不进入评测证据",
                        },
                    ],
                    "filtered_out": [
                        {
                            "product_id": "taobao:2",
                            "price_major": 399,
                            "currency": "CNY",
                            "reason": "over_price_cap",
                        },
                    ],
                },
            ),
        ],
        structured_state=_state(),
    )

    assert turn.route == "main.direct"
    assert turn.user_input == "电话 [redacted]，只看淘宝露营灯"
    assert "13800138000" not in turn.final_text
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.agent == "main"
    assert call.status == "success"
    assert call.arguments == {"normalized_query": "露营灯", "platform": "taobao"}
    assert call.result_summary["hit_product_ids"] == ["taobao:1"]
    assert call.result_summary["filtered_product_ids"] == ["taobao:2"]
    assert call.result_summary["hit_facts"] == [
        {
            "product_id": "taobao:1",
            "title": "露营灯",
            "price_major": 89,
            "currency": "CNY",
            "landed_price": {
                "subtotal_major": 89,
                "freight_major": 25,
                "tariff_major": 0,
                "landed_total_major": 114,
                "currency": "CNY",
            },
        },
    ]
    assert call.result_summary["filtered_facts"] == [
        {
            "product_id": "taobao:2",
            "price_major": 399,
            "currency": "CNY",
            "reason": "over_price_cap",
        },
    ]
    assert "description" not in call.result_summary["hit_facts"][0]


def test_real_event_bus_event_object_is_normalized() -> None:
    bus = TradeEventBus()
    queue = bus.subscribe("session-1")
    bus.deliver_local(
        TradeEvent(
            shopping_session_id="session-1",
            type="tool.invoke",
            payload={
                "tool": "product_search_tool",
                "args": {"normalized_query": "露营灯", "platform": "taobao"},
            },
            occurred_at="2026-08-29T00:00:00+00:00",
        ),
    )

    turn = build_turn_evidence(
        turn_index=1,
        user_input="找露营灯",
        final_text="正在搜索。",
        displayed_products=[],
        events=[queue.get_nowait()],
        structured_state=_state(),
    )

    assert turn.tool_calls[0].tool == "product_search_tool"
    assert turn.tool_calls[0].arguments == {
        "normalized_query": "露营灯",
        "platform": "taobao",
    }


def test_shipping_name_address_phone_and_postal_code_are_redacted() -> None:
    turn = build_turn_evidence(
        turn_index=1,
        user_input=(
            "帮我买灯，寄到：张三，中国 浙江 杭州市 西湖区某路1号，"
            "邮编310000，电话13800000000。"
        ),
        final_text=(
            "确认卡\n收件人：张三\n收货地址：中国 浙江 杭州市 西湖区某路1号\n"
            "邮编：310000\n电话：13800000000"
        ),
        displayed_products=[],
        events=[],
        structured_state=_state(),
    )

    serialized = turn.model_dump_json()
    assert "张三" not in serialized
    assert "西湖区某路1号" not in serialized
    assert "310000" not in serialized
    assert "13800000000" not in serialized
    assert "shipping details redacted" in turn.user_input
    assert "收件人：[redacted]" in turn.final_text


def test_markdown_personal_detail_table_row_is_fully_redacted() -> None:
    turn = build_turn_evidence(
        turn_index=1,
        user_input="取消订单",
        final_text=(
            "| 项目 | 内容 |\n"
            "|---|---|\n"
            "| 收货地址 | 测试收件人，中国 测试省 测试市 测试路1号，邮编 310000 |\n"
            "| 电话 | +86 571 8888 9999 |"
        ),
        displayed_products=[],
        events=[],
        structured_state=_state(),
    )

    assert "测试收件人" not in turn.final_text
    assert "测试省" not in turn.final_text
    assert "测试市" not in turn.final_text
    assert "310000" not in turn.final_text
    assert "8888 9999" not in turn.final_text
    assert "| 收货地址 | [redacted] |" in turn.final_text
    assert "| 电话 | [redacted] |" in turn.final_text


def test_parallel_search_dispatches_link_calls_by_correlation_id() -> None:
    events = [
        _event(
            "agent.dispatch",
            {
                "agent": "search_agent",
                "platform": "taobao",
                "dispatch_correlation_id": "corr-taobao",
            },
        ),
        _event(
            "agent.dispatch",
            {
                "agent": "search_agent",
                "platform": "amazon",
                "site_locale": "jp",
                "dispatch_correlation_id": "corr-amazon",
            },
        ),
        _event(
            "tool.invoke",
            {
                "tool": "product_search_tool",
                "dispatch_correlation_id": "corr-amazon",
                "args": {"platform": "amazon", "site_locale": "jp"},
            },
        ),
        _event(
            "tool.invoke",
            {
                "tool": "product_search_tool",
                "dispatch_correlation_id": "corr-taobao",
                "args": {"platform": "taobao"},
            },
        ),
        _event(
            "tool.result",
            {
                "tool": "product_search_tool",
                "dispatch_correlation_id": "corr-taobao",
                "hits": [],
                "filtered_out": [],
            },
        ),
        _event(
            "tool.result",
            {
                "tool": "product_search_tool",
                "dispatch_correlation_id": "corr-amazon",
                "hits": [],
                "filtered_out": [],
            },
        ),
        _event(
            "tool.result",
            {
                "tool": "task_dispatch",
                "agent": "search_agent",
                "platform": "taobao",
                "dispatch_correlation_id": "corr-taobao",
                "elapsed_ms": 20,
            },
        ),
        _event(
            "tool.result",
            {
                "tool": "task_dispatch",
                "agent": "search_agent",
                "platform": "amazon",
                "dispatch_correlation_id": "corr-amazon",
                "elapsed_ms": 25,
            },
        ),
    ]

    turn = build_turn_evidence(
        turn_index=1,
        user_input="比较淘宝和日本亚马逊",
        final_text="暂未找到合适商品。",
        displayed_products=[],
        events=events,
        structured_state=_state(),
    )

    assert turn.route == "main.dispatch.search_agent"
    assert [dispatch.platform for dispatch in turn.dispatches] == ["taobao", "amazon"]
    assert [call.correlation_id for call in turn.tool_calls] == [None, None]
    assert [call.call_id for call in turn.tool_calls] == ["call-1", "call-2"]
    assert all(call.agent == "search_agent" for call in turn.tool_calls)
    assert all(call.status == "success" for call in turn.tool_calls)


def test_circuit_result_without_business_invoke_stays_visible() -> None:
    turn = build_turn_evidence(
        turn_index=1,
        user_input="查询商品",
        final_text="系统暂时不可用。",
        displayed_products=[],
        events=[
            _event(
                "tool.result",
                {
                    "tool": "product_search_tool",
                    "circuit": "open",
                    "error": "private upstream response sk-never-export",
                },
            ),
        ],
        structured_state=_state(),
    )

    assert turn.tool_calls[0].status == "error"
    assert turn.tool_calls[0].error_type == "open"
    assert turn.runtime_signals[0].signal == "orphan_tool_result"
    assert turn.runtime_signals[1].signal == "circuit"
    assert "sk-never-export" not in turn.model_dump_json()


def test_span_normalization_keeps_topology_and_operational_attributes_only() -> None:
    root = _span(
        "globex.shopping_intent",
        span_id=1,
        attributes={"globex.session.hash": "private-session"},
    )
    child = _span(
        "globex.product.opensearch.hybrid",
        span_id=2,
        parent_id=1,
        attributes={
            "globex.opensearch.hit_count": 5,
            "globex.dependency.attempts": 2,
            "globex.dependency.retry_reason": {"secret": "sk-never-export"},
            "gen_ai.input.messages": "private query",
        },
    )

    trace_id, spans = normalize_trace_spans([child, root])

    assert trace_id is None
    assert [span.span_id for span in spans] == [
        "0000000000000001",
        "0000000000000002",
    ]
    assert spans[1].parent_span_id == "0000000000000001"
    assert spans[1].duration_ms == 2.5
    assert spans[1].attributes == {
        "globex.opensearch.hit_count": 5,
        "globex.dependency.attempts": 2,
        "globex.dependency.retry_reason": "{'secret': '[redacted]'}",
    }


def test_multiple_trace_ids_in_one_turn_are_rejected() -> None:
    with pytest.raises(EvidenceNormalizationError, match="multiple traces"):
        normalize_trace_spans(
            [_span("one", trace_id=1), _span("two", trace_id=2, span_id=3)],
        )


def test_case_hashes_session_and_keeps_per_turn_trace_id() -> None:
    turn = build_turn_evidence(
        turn_index=1,
        user_input="你好",
        final_text="你好，我是购物助手。",
        displayed_products=[],
        events=[],
        spans=[_span("globex.shopping_intent")],
        structured_state=_state(),
    )

    evidence = build_evaluation_evidence(
        case_id="chat",
        session_id="private-session",
        turns=[turn],
    )

    assert evidence.session_id_hash != "private-session"
    assert len(evidence.session_id_hash) == 16
    assert evidence.turns[0].trace_id is None


def test_structured_state_keeps_refs_and_preference_fingerprints_only() -> None:
    state = build_structured_state_evidence(
        {
            "schema_version": "l4-v1",
            "revision": 3,
            "session": {"buyer_id": "private-buyer"},
            "last_search": {
                "args": {"normalized_query": "露营灯", "ship_to": "US"},
                "product_refs": ["P1008"],
                "filtered_product_refs": ["P1005"],
            },
            "last_recommendation": {
                "displayed_products": [
                    {
                        "platform": "globex_reference",
                        "card": {"product_id": "P1008", "title": "露营灯"},
                    },
                ],
            },
        },
        preference_before=_preference(0, "0"),
        preference_after=_preference(1, "a"),
    )

    serialized = state.model_dump_json()
    assert state.last_search_product_refs == ["P1008"]
    assert state.current_recommendations[0].product_id == "P1008"
    assert state.preference_after.count == 1
    assert "private-buyer" not in serialized
    assert "title" not in serialized


def test_redaction_covers_international_phone_url_secret_and_plain_address() -> None:
    turn = build_turn_evidence(
        turn_index=1,
        user_input=(
            "寄往中国浙江省杭州市西湖区文三路90号，电话 +86 571 8888 9999，"
            "回调 https://example.test/path?api_key=secret-value"
        ),
        final_text="Phone +1 (415) 555-2671; 12 Main Street, San Francisco, CA",
        displayed_products=[],
        events=[],
        structured_state=_state(),
    )

    serialized = turn.model_dump_json()
    assert "文三路90号" not in serialized
    assert "8888 9999" not in serialized
    assert "secret-value" not in serialized
    assert "555-2671" not in serialized
    assert "12 Main Street" not in serialized


def test_redaction_removes_health_detail_but_keeps_actionable_preference() -> None:
    turn = build_turn_evidence(
        turn_index=1,
        user_input="记住：不要塑料材质，我对塑料过敏。",
        final_text="已记住不要塑料材质（您对塑料过敏）。",
        displayed_products=[],
        events=[],
        structured_state=_state(),
    )

    serialized = turn.model_dump_json()
    assert "不要塑料材质" in serialized
    assert "过敏" not in serialized
    assert "sensitive health detail redacted" in serialized
