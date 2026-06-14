"""Settings loading defaults for the migration architecture."""

from __future__ import annotations

import pytest

from globex_agent.infrastructure.resilience import DEFAULT_TIMEOUTS
from globex_agent.infrastructure.settings import load_settings


def test_defaults_use_json_file_storage_and_no_queue(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("QUEUE_ENABLED", raising=False)
    monkeypatch.delenv("SEMANTIC_CACHE_ENABLED", raising=False)
    settings = load_settings()
    assert settings.database_url == "file"
    assert settings.queue_enabled is False
    assert settings.semantic_cache_enabled is True
    assert settings.context_budget_mode == "observe_only"
    assert settings.context_soft_limit_tokens == 0
    assert settings.context_safety_margin_tokens == 0
    assert settings.l3_keep_recent_frozen_segments == 1


def test_context_budget_mode_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_BUDGET_MODE", "guess")
    try:
        load_settings()
    except ValueError as error:
        assert "observe_only or enforce" in str(error)
    else:
        raise AssertionError("unknown CONTEXT_BUDGET_MODE was accepted")


def test_l3_recent_frozen_window_rejects_negative_value(monkeypatch) -> None:
    monkeypatch.setenv("L3_KEEP_RECENT_FROZEN_SEGMENTS", "-1")
    with pytest.raises(ValueError, match="cannot be negative"):
        load_settings()


def test_product_search_timeout_covers_persistent_reranker_budget() -> None:
    assert DEFAULT_TIMEOUTS["product_search_tool"] == 75.0


def test_online_judge_does_not_inherit_main_llm_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "main-model")
    monkeypatch.delenv("ONLINE_EVIDENCE_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("LLM_JUDGE", raising=False)
    settings = load_settings()
    assert settings.online_evidence_judge_model == "deepseek-v4-flash"


def test_online_and_offline_judge_models_have_independent_environment_names(monkeypatch) -> None:
    monkeypatch.setenv("ONLINE_EVIDENCE_JUDGE_MODEL", "online-small")
    monkeypatch.setenv("EVAL_FINAL_JUDGE_MODEL", "offline-final")
    settings = load_settings()
    assert settings.online_evidence_judge_model == "online-small"
    assert settings.eval_final_judge_model == "offline-final"


def test_rubric_generator_falls_back_to_final_judge_and_threshold_defaults(monkeypatch) -> None:
    monkeypatch.delenv("EVAL_RUBRIC_GENERATOR_MODEL", raising=False)
    monkeypatch.setenv("EVAL_FINAL_JUDGE_MODEL", "offline-final")
    monkeypatch.delenv("EVAL_PASS_THRESHOLD", raising=False)
    settings = load_settings()
    assert settings.eval_rubric_generator_model == "offline-final"
    assert settings.eval_pass_threshold == 0.85


def test_eval_pass_threshold_rejects_values_outside_unit_interval(monkeypatch) -> None:
    monkeypatch.setenv("EVAL_PASS_THRESHOLD", "1.01")
    try:
        load_settings()
    except ValueError as error:
        assert "between 0 and 1" in str(error)
    else:
        raise AssertionError("out-of-range EVAL_PASS_THRESHOLD was accepted")


def test_online_verification_budget_defaults_are_bounded_for_rest_client() -> None:
    settings = load_settings()
    assert settings.rewrite_timeout_seconds == 25.0
    assert settings.evidence_judge_timeout_seconds == 30.0
    assert settings.evidence_judge_total_timeout_seconds == 70.0
    assert settings.turn_timeout_seconds < 90.0
