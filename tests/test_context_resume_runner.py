import json

import pytest
from agentscope.message import AssistantMsg, UserMsg
from agentscope.state import AgentState

from scripts.eval.context_compression_runner import DEFAULT_SCENARIO, load_scenario
from scripts.eval.context_resume_runner import (
    classify_resume_status,
    clone_session_state_json,
    completed_turns_from_state_json,
    evaluate_warm_cache_window,
    remaining_natural_queries,
)


def test_clone_session_state_retargets_copy_and_keeps_source_immutable() -> None:
    source = AgentState(
        session_id="source-agent-session",
        context=[UserMsg(name="buyer-1", content="私密原始问题")],
        middle_context={
            "globex_context_v1": {
                "current_intent": {
                    "shopping_session_id": "source-shopping-session",
                    "buyer_id": "buyer-1",
                },
                "current_turn_events": [{"type": "old"}],
                "session_context": {
                    "session": {
                        "shopping_session_id": "source-shopping-session",
                        "buyer_id": "buyer-1",
                    },
                },
            },
        },
    )
    source_json = source.model_dump_json()

    cloned_json, buyer_id = clone_session_state_json(
        source_json,
        target_session_id="target-session",
    )

    assert source.model_dump_json() == source_json
    cloned = AgentState.model_validate_json(cloned_json)
    namespace = cloned.middle_context["globex_context_v1"]
    assert buyer_id == "buyer-1"
    assert cloned.session_id == "target-session"
    assert namespace["current_intent"]["shopping_session_id"] == "target-session"
    assert namespace["session_context"]["session"]["shopping_session_id"] == (
        "target-session"
    )
    assert namespace["current_turn_events"] == []
    assert "私密原始问题" in cloned_json


def test_clone_session_state_falls_back_to_buyer_message_name() -> None:
    source = AgentState(
        session_id="source",
        context=[UserMsg(name="buyer-fallback", content="query")],
    )

    cloned_json, buyer_id = clone_session_state_json(
        source.model_dump_json(),
        target_session_id="target",
    )

    assert buyer_id == "buyer-fallback"
    cloned = json.loads(cloned_json)
    assert (
        cloned["middle_context"]["globex_context_v1"]["current_intent"]["buyer_id"]
        == "buyer-fallback"
    )


def test_remaining_queries_continue_after_source_turn_count() -> None:
    scenario = load_scenario(DEFAULT_SCENARIO)
    all_queries = [
        *scenario.anchor_turns,
        *(query for journey in scenario.long_journeys for query in journey),
    ]

    assert (
        remaining_natural_queries(
            scenario,
            completed_turns=53,
            limit=2,
        )
        == all_queries[53:55]
    )
    assert (
        remaining_natural_queries(
            scenario,
            completed_turns=60,
            limit=3,
        )
        == all_queries[60:63]
    )


def test_completed_turns_are_derived_from_persisted_state() -> None:
    state = AgentState(
        session_id="source",
        context=[
            UserMsg(name="buyer", content="第一轮"),
            AssistantMsg(name="Main", content="回答一"),
            UserMsg(name="buyer", content="第二轮"),
            AssistantMsg(name="Main", content="回答二"),
            UserMsg(name="buyer", content="尚未完成"),
        ],
    )

    assert completed_turns_from_state_json(state.model_dump_json()) == 2


def test_warm_cache_window_requires_cache_l2_and_business_stability() -> None:
    title_one = "商品完整名称一"
    title_two = "商品完整名称二"
    source_turns = [
        {
            "displayed_products": [
                {"card": {"product_id": "P-1", "title": title_one}},
                {"card": {"product_id": "P-2", "title": title_two}},
            ],
        },
    ]
    turns = [
        {
            "turn_usage": {
                "model_calls": 1,
                "input_tokens": 1_000,
                "cached_input_tokens": 850,
                "token_weighted_cache_hit_ratio": 0.85,
            },
            "events": [],
            "context": {"freeze_cursor": 108 + index * 2},
            "displayed_products": [
                {"card": {"product_id": "P-1", "title": title_one}},
                {"card": {"product_id": "P-2", "title": title_two}},
            ],
            "final_text": f"{title_one}；{title_two}",
            "compression_event": None,
        }
        for index in range(1, 6)
    ]

    result = evaluate_warm_cache_window(
        turns=turns,
        source_turns=source_turns,
        before_freeze_cursor=108,
        expected_turns=5,
        minimum_turn_ratio=0.75,
        minimum_weighted_ratio=0.80,
        source_unchanged=True,
    )

    assert result["passed"] is True
    assert result["weighted_cache_hit_ratio"] == 0.85
    assert all(result["checks"].values())


def test_warm_cache_window_rejects_search_and_low_ratio() -> None:
    result = evaluate_warm_cache_window(
        turns=[
            {
                "turn_usage": {
                    "model_calls": 1,
                    "input_tokens": 1_000,
                    "cached_input_tokens": 700,
                    "token_weighted_cache_hit_ratio": 0.70,
                },
                "events": [
                    {"type": "tool.invoke", "tool": "product_search_tool"},
                ],
                "context": {"freeze_cursor": 110},
                "displayed_products": [],
                "final_text": "",
                "compression_event": None,
            },
        ],
        source_turns=[
            {
                "displayed_products": [
                    {"card": {"product_id": "P-1", "title": "商品一"}},
                ],
            },
        ],
        before_freeze_cursor=108,
        expected_turns=1,
        minimum_turn_ratio=0.75,
        minimum_weighted_ratio=0.80,
        source_unchanged=True,
    )

    assert result["passed"] is False
    assert result["checks"]["no_product_search"] is False
    assert result["checks"]["minimum_turn_ratio"] is False
    assert result["checks"]["weighted_ratio"] is False


@pytest.mark.parametrize(
    ("source_unchanged", "l2_passed", "compressed", "recovered", "expected"),
    [
        (False, True, True, True, "SOURCE_MUTATED"),
        (True, False, False, False, "L2_NOT_ADVANCED"),
        (True, True, False, False, "L2_PASS_L3_NOT_REACHED"),
        (True, True, True, False, "L3_RECOVERY_FAIL"),
        (True, True, True, True, "L3_RECOVERY_PASS"),
    ],
)
def test_resume_status_keeps_l2_and_l3_outcomes_distinct(
    source_unchanged: bool,
    l2_passed: bool,
    compressed: bool,
    recovered: bool,
    expected: str,
) -> None:
    assert (
        classify_resume_status(
            source_unchanged=source_unchanged,
            l2_repair_passed=l2_passed,
            compression_observed=compressed,
            recovery_passed=recovered,
        )
        == expected
    )
