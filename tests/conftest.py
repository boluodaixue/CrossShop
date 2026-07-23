"""Public-suite portability guards for optional catalog/index fixtures.

The published repository intentionally excludes private or license-unclear catalog
snapshots. Tests that validate those exact snapshots remain available, but are
skipped unless a maintainer explicitly supplies the local fixture release.
"""

from pathlib import Path

import pytest


OPTIONAL_FIXTURE_MODULES = {
    "test_catalogs_v2.py",
    "test_category_candidates_v1.py",
    "test_category_insight_release_v1.py",
    "test_category_recall_eval.py",
    "test_h3_catalog_search_integration.py",
    "test_h4_agent_acceptance.py",
    "test_h4_runtime_wiring.py",
    "test_jsonl_product_repository_h3.py",
    "test_rubric_ground_truth.py",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    fixture_root = Path(__file__).parents[1] / "data" / "processed" / "catalogs-v2"
    if fixture_root.exists() or config.getoption("--run-optional-fixtures", default=False):
        return
    reason = "optional catalog fixtures are not redistributed; set --run-optional-fixtures after providing them"
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        if Path(str(item.fspath)).name in OPTIONAL_FIXTURE_MODULES:
            item.add_marker(marker)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-optional-fixtures",
        action="store_true",
        help="run tests requiring the separately supplied catalog fixture release",
    )
