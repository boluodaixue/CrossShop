from app.infrastructure.model_usage import ModelUsageSample
from scripts.eval.context_compression_runner import (
    RecoveryProbe,
    detect_freeze_stall,
    evaluate_recovery_probe,
    load_scenario,
    rolling_cache_ratios,
    summarize_run,
)
from scripts.eval.context_compression_runner import DEFAULT_SCENARIO


def _usage(input_tokens: int, cached_tokens: int) -> ModelUsageSample:
    return ModelUsageSample(
        requested_model="deepseek-v4-flash",
        input_tokens=input_tokens,
        output_tokens=100,
        total_tokens=input_tokens + 100,
        cached_input_tokens=cached_tokens,
        cache_creation_input_tokens=0,
        elapsed_seconds=1.0,
    )


def test_context_scenario_has_ten_anchor_turns_and_default_threshold_contract() -> None:
    scenario = load_scenario(DEFAULT_SCENARIO)

    assert len(scenario.anchor_turns) == 10
    assert scenario.targets.cumulative_turns == 10
    assert scenario.targets.max_single_call_total_tokens == 30_000
    assert scenario.targets.freeze_stall_check_turns == 3
    assert scenario.targets.min_rolling_cache_hit_ratio == 0.8
    assert scenario.targets.warm_cache_turns == 5
    assert scenario.targets.min_warm_turn_cache_hit_ratio == 0.75
    assert scenario.targets.max_turns_until_compression == 90
    assert scenario.targets.target_compression_events == 1
    assert len(scenario.warm_cache_queries) == 5
    assert sum(len(journey) for journey in scenario.long_journeys) == 110


def test_rolling_cache_ratio_excludes_only_declared_warmup() -> None:
    samples = [_usage(1_000, 0), *[_usage(1_000, 800) for _ in range(6)]]

    ratios = rolling_cache_ratios(samples, warmup_calls=1, window_calls=5)

    assert ratios == [0.8, 0.8]


def test_freeze_stall_fails_fast_after_declared_turn_count() -> None:
    turns = [
        {
            "turn_index": index,
            "phase": "natural-long-session",
            "context": {
                "raw_context_message_count": index * 2,
                "freeze_cursor": 0,
                "frozen_segment_count": 0,
                "stage_summary_present": False,
            },
        }
        for index in range(1, 4)
    ]

    assert detect_freeze_stall(turns[:2], check_after_turns=3) is None
    assert detect_freeze_stall(turns, check_after_turns=3) == {
        "detected": True,
        "turn_index": 3,
        "raw_context_message_count": 6,
        "freeze_cursor": 0,
        "frozen_segment_count": 0,
        "reason": "compression_precondition_stalled",
    }


def test_freeze_stall_allows_progressed_cursor() -> None:
    turns = [
        {
            "turn_index": 3,
            "phase": "natural-long-session",
            "context": {
                "raw_context_message_count": 6,
                "freeze_cursor": 4,
                "frozen_segment_count": 2,
                "stage_summary_present": False,
            },
        }
    ] * 3

    assert detect_freeze_stall(turns, check_after_turns=3) is None


def test_recovery_probe_requires_verified_stage_exact_facts_and_no_search() -> None:
    probe = RecoveryProbe(
        id="recover-first",
        query="恢复最早商品",
        require_anchor_title_price_currency=True,
        forbid_product_search=True,
    )
    turn = {
        "final_text": "LumenGo 便携露营灯，价格 199.00 CNY。",
        "events": [],
        "context": {"stage_summary_verified": True},
    }

    result = evaluate_recovery_probe(
        turn,
        probe=probe,
        anchor={
            "product_id": "P1008",
            "title": "LumenGo 便携露营灯",
            "price_major": 199,
            "currency": "CNY",
        },
    )

    assert result["passed"] is True


def test_recovery_probe_rejects_unnecessary_search() -> None:
    probe = RecoveryProbe(id="recover-first", query="恢复最早商品")
    result = evaluate_recovery_probe(
        {
            "final_text": "LumenGo 便携露营灯，价格 199 CNY。",
            "events": [{"type": "tool.invoke", "tool": "product_search_tool"}],
            "context": {"stage_summary_verified": True},
        },
        probe=probe,
        anchor={"title": "LumenGo 便携露营灯", "price_major": 199, "currency": "CNY"},
    )

    assert result["passed"] is False
    assert result["checks"]["no_product_search"] is False


def test_summary_separates_cumulative_cost_from_per_call_cap() -> None:
    scenario = load_scenario(DEFAULT_SCENARIO)
    turns = [
        {
            "turn_index": index,
            "phase": "natural-long-session",
            "model_usage": [_usage(2_000, 1_600).to_dict()],
            "turn_usage": {
                "input_tokens": 2_000,
                "output_tokens": 100,
                "total_tokens": 2_100,
            },
        }
        for index in range(1, 11)
    ]
    samples = [_usage(1_000, 0), *[_usage(1_000, 800) for _ in range(10)]]

    summary = summarize_run(
        scenario=scenario,
        turns=turns,
        samples=samples,
        compression_turn=10,
        recovery_results=[{"passed": True}],
    )

    assert summary["ten_turn"] == {
        "turns_observed": 10,
        "cumulative_input_tokens": 20_000,
        "cumulative_output_tokens": 1_000,
        "cumulative_total_tokens": 21_000,
        "max_single_call_input_tokens": 2_000,
        "max_single_call_total_tokens": 2_100,
        "target_max_single_call_total_tokens": 30_000,
        "passed": True,
    }
    assert summary["long_session_token_cap"] == {
        "model_calls_observed": 10,
        "max_single_call_total_tokens": 2_100,
        "target_max_single_call_total_tokens": 30_000,
        "first_breach_turn": None,
        "passed": True,
    }
    assert summary["prompt_cache"]["minimum_rolling_hit_ratio"] == 0.8
    assert summary["prompt_cache"]["passed"] is True


def test_summary_requires_authored_repeated_compression_count() -> None:
    scenario = load_scenario(DEFAULT_SCENARIO).model_copy(
        update={
            "targets": load_scenario(DEFAULT_SCENARIO).targets.model_copy(
                update={"target_compression_events": 2},
            ),
        },
    )
    turns = [
        {
            "turn_index": index,
            "phase": "natural-long-session",
            "model_usage": [_usage(2_000, 1_600).to_dict()],
            "turn_usage": {
                "input_tokens": 2_000,
                "output_tokens": 100,
                "total_tokens": 2_100,
            },
        }
        for index in range(1, 13)
    ]

    summary = summarize_run(
        scenario=scenario,
        turns=turns,
        samples=[_usage(1_000, 800) for _ in range(12)],
        compression_turn=10,
        compression_turns=[10, 12],
        recovery_results=[{"passed": True}],
    )

    assert summary["compression"] == {
        "observed": True,
        "observed_count": 2,
        "target_count": 2,
        "trigger_turns": [10, 12],
        "first_trigger_turn": 10,
        "max_turns_until_compression": 90,
        "passed": True,
    }
