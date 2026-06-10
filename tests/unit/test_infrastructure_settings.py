"""Settings loading defaults for the migration architecture."""

from __future__ import annotations

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


def test_product_search_timeout_covers_persistent_reranker_budget() -> None:
    assert DEFAULT_TIMEOUTS["product_search_tool"] == 75.0


def test_online_judge_does_not_inherit_main_llm_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "main-model")
    monkeypatch.delenv("ONLINE_EVIDENCE_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("LLM_JUDGE", raising=False)
    settings = load_settings()
    assert settings.online_evidence_judge_model == "qwen-plus"


def test_online_and_offline_judge_models_have_independent_environment_names(monkeypatch) -> None:
    monkeypatch.setenv("ONLINE_EVIDENCE_JUDGE_MODEL", "online-small")
    monkeypatch.setenv("EVAL_FINAL_JUDGE_MODEL", "offline-final")
    settings = load_settings()
    assert settings.online_evidence_judge_model == "online-small"
    assert settings.eval_final_judge_model == "offline-final"


def test_online_verification_budget_defaults_are_bounded_for_rest_client() -> None:
    settings = load_settings()
    assert settings.rewrite_timeout_seconds == 25.0
    assert settings.evidence_judge_timeout_seconds == 30.0
    assert settings.evidence_judge_total_timeout_seconds == 70.0
    assert settings.turn_timeout_seconds < 90.0
