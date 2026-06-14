from __future__ import annotations

from globex_agent.application.context import (
    FrozenSegment,
    L4Context,
    TokenCounter,
    measure_budget,
    reduce_l4,
)


def _intent(raw_query: str = "预算 500 元") -> dict:
    return {
        "shopping_session_id": "session-1",
        "buyer_id": "buyer-1",
        "locale": "zh-CN",
        "currency": "CNY",
        "raw_query": raw_query,
    }


def _search_events() -> list[dict]:
    return [
        {
            "type": "tool.invoke",
            "occurred_at": "2026-08-24T00:00:01Z",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-a",
                "args": {"normalized_query": "backpack", "ship_to": "US", "top_k": 5},
            },
        },
        {
            "type": "tool.invoke",
            "occurred_at": "2026-08-24T00:00:02Z",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-b",
                "args": {"normalized_query": "laptop bag", "ship_to": "CA"},
            },
        },
        # B finishes before A: completion order, not invoke order, decides the
        # last successful search.
        {
            "type": "tool.result",
            "occurred_at": "2026-08-24T00:00:03Z",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-b",
                "hit_count": 1,
                "model_output": {"hits": [{"item_id": "item-b"}]},
            },
        },
        {
            "type": "tool.result",
            "occurred_at": "2026-08-24T00:00:04Z",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-a",
                "hit_count": 1,
                "model_output": {"hits": [{"item_id": "item-a"}]},
            },
        },
        {
            "type": "final.result",
            "payload": {
                "verification_status": "supported",
                "recommended_cards": [{"item_id": "item-a", "title": "A", "private": "drop"}],
            },
        },
    ]


def test_reducer_traces_sources_and_keeps_origin_only_once() -> None:
    first = reduce_l4(_intent(), _search_events(), now="t1")
    second = reduce_l4(_intent("第二个商品怎么样"), [], first, now="t2")

    assert first.session == {
        "shopping_session_id": "session-1",
        "buyer_id": "buyer-1",
        "locale": "zh-CN",
        "currency": "CNY",
    }
    assert first.request == {
        "session_origin_raw_query": "预算 500 元",
        "current_raw_query": "预算 500 元",
    }
    assert first.last_search["tool_call_id"] == "search-a"
    assert first.last_search["args"] == {
        "normalized_query": "backpack",
        "ship_to": "US",
        "top_k": 5,
    }
    assert first.last_recommendation == {
        "verification_status": "supported",
        "items": [{"item_id": "item-a", "title": "A"}],
    }
    assert second.request == {
        "session_origin_raw_query": "预算 500 元",
        "current_raw_query": "第二个商品怎么样",
    }
    assert second.revision == 2


def test_reducer_ignores_unpaired_or_failed_search_and_does_not_invent_order_facts() -> None:
    events = [
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "missing-invoke",
                "model_output": {"hits": [{"item_id": "not-authoritative"}]},
            },
        },
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "failed",
                "args": {"normalized_query": "x"},
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "failed",
                "error": "timeout",
            },
        },
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "prepare_order_tool",
                "args": {"items": [{"item_id": "i1"}], "shipping_address": {"country": "US"}},
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "prepare_order_tool",
                "error": "item unavailable",
                "model_output": {"error": "item unavailable"},
            },
        },
    ]
    state = reduce_l4(_intent(), events, now="t")

    assert state.last_search is None
    assert state.order is None


def test_reducer_is_idempotent_on_event_replay() -> None:
    state = reduce_l4(_intent(), _search_events(), now="t1")
    replay = reduce_l4(_intent(), _search_events(), state, now="different")
    assert replay == state
    assert replay.revision == 1
    assert replay.updated_at == "2026-08-24T00:00:04Z"


def test_order_mapping_uses_only_actual_fields() -> None:
    events = [
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "prepare_order_tool",
                "args": {
                    "items": [{"item_id": "i1", "variant_id": "v1"}],
                    "shipping_address": {"country": "US"},
                },
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "prepare_order_tool",
                "confirmation_required": True,
                "confirmation_id": "confirm-1",
                "model_output": {
                    "confirmation_required": True,
                    "confirmation_id": "confirm-1",
                    "items": [{"item_id": "i1", "variant_id": "v1", "quantity": 2}],
                },
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "query_order_tool",
                "model_output": {"order_id": "order-1", "status": "CONFIRMED"},
            },
        },
    ]
    state = reduce_l4(_intent(), events, now="t")
    assert state.order == {
        "items": [{"item_id": "i1", "variant_id": "v1", "quantity": 2}],
        "confirmation_id": "confirm-1",
        "confirmation_status": "required",
        "shipping_country": "US",
        "order_id": "order-1",
        "status": "CONFIRMED",
    }
    assert state.order["items"][0]["quantity"] == 2


def test_frozen_segment_hash_is_stable() -> None:
    segment = FrozenSegment(segment_id="seg-1", content="用户：背包")
    assert segment.content_hash
    assert segment.content_hash == FrozenSegment(
        segment_id="seg-1", content="用户：背包"
    ).content_hash
    assert L4Context.from_dict(segment.to_dict()).revision == 0


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text)))


class _FakeModel:
    def get_num_tokens(self, text: str) -> int:
        return len(text) + 10

    def get_num_tokens_from_messages(self, messages: list[dict]) -> int:
        return len(messages) + 100


def test_budget_report_counts_each_layer_and_each_tool_result() -> None:
    report = measure_budget(
        fixed_prompt="system",
        tool_schemas={"search": {"type": "function"}},
        l4_candidate={"request": {"current_raw_query": "bag"}},
        history=[{"role": "user", "content": "old"}],
        active_current=[{"role": "tool", "content": "active"}],
        tool_results={"product_search_tool": {"hits": [{"item_id": "i1"}]}},
        model_context_tokens=30,
        reply_reserved_tokens=5,
        safety_margin_tokens=2,
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
    )
    assert set(report.layers) == {
        "fixed_prompt",
        "tool_schemas",
        "l4_candidate",
        "history",
        "active_current",
    }
    assert report.tool_results["product_search_tool"].tokens > 0
    assert report.total_input_tokens == sum(x.tokens for x in report.layers.values()) + sum(
        x.tokens for x in report.tool_results.values()
    )
    assert report.estimated is False
    assert report.counter_method == "mixed"
    assert report.overflow_tokens == max(0, report.total_input_tokens - 23)


def test_budget_prefers_provider_message_counter_for_message_layers() -> None:
    report = measure_budget(
        history=[{"role": "user", "content": "old"}],
        active_current=[{"role": "tool", "content": "active"}],
        model_context_tokens=1000,
        model=_FakeModel(),
    )
    assert report.layers["history"].tokens == 101
    assert report.layers["history"].method == "model.get_num_tokens_from_messages"


def test_budget_fallback_is_explicit_estimate_not_message_count() -> None:
    count = TokenCounter().count("中文文本", name="history")
    assert count.estimated is True
    assert count.method == "estimate.utf8_bytes_div_4"
    assert count.tokens > 1
