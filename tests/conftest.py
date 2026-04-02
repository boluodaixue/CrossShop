from pathlib import Path

import pytest

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import SearchRequest, UserProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "demo"


@pytest.fixture(scope="session")
def demo_catalog() -> LocalCatalog:
    return LocalCatalog.from_jsonl(DATA_DIR / "products.jsonl", strict=True).catalog


@pytest.fixture(scope="session")
def demo_requests() -> list[SearchRequest]:
    return [
        SearchRequest.model_validate_json(line)
        for line in (DATA_DIR / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="session")
def demo_profiles() -> dict[str, UserProfile]:
    profiles = [
        UserProfile.model_validate_json(line)
        for line in (DATA_DIR / "users.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {profile.user_id: profile for profile in profiles}
