"""Settings loading defaults for the migration architecture."""

from __future__ import annotations

from globex_agent.infrastructure.settings import load_settings


def test_defaults_use_json_file_storage_and_no_queue(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("QUEUE_ENABLED", raising=False)
    settings = load_settings()
    assert settings.database_url == "file"
    assert settings.queue_enabled is False
    assert settings.semantic_cache_enabled is True
