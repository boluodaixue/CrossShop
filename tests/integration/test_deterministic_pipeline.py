from decimal import Decimal

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import ResultStatus, SearchRequest, UserProfile
from globex_agent.pipeline import run_deterministic_pipeline


def test_full_pipeline_returns_expected_commuting_headphones(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    result = run_deterministic_pipeline(
        demo_catalog,
        demo_requests[0],
        demo_profiles["demo-user-001"],
    )

    assert result.search.status is ResultStatus.OK
    assert len(result.price_comparison.comparisons) == 3
    assert len(result.shipping.quotes) == 9
    assert [pick.canonical_product_id for pick in result.selection.picks] == [
        "hp-budget-wave",
        "hp-sonic-commute",
        "hp-aurora-quietpro",
    ]
    assert all(pick.landed_price <= Decimal("1500.00") for pick in result.selection.picks)
    assert result.summary.status is ResultStatus.OK


def test_pipeline_is_deterministic_across_repeated_runs(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    first = run_deterministic_pipeline(demo_catalog, demo_requests[3])
    second = run_deterministic_pipeline(demo_catalog, demo_requests[3])

    assert first == second


def test_pipeline_returns_structured_empty_result(demo_catalog: LocalCatalog) -> None:
    request = SearchRequest(query="zzzzzzzz", budget=Decimal("100.00"), top_k=3)

    result = run_deterministic_pipeline(demo_catalog, request)

    assert result.search.status is ResultStatus.NO_RESULTS
    assert result.selection.status is ResultStatus.NO_RESULTS
    assert result.summary.status is ResultStatus.NO_RESULTS
    assert result.summary.recommendation.items == []


def test_pipeline_propagates_invalid_destination(demo_catalog: LocalCatalog) -> None:
    request = SearchRequest(query="头戴式降噪耳机", top_k=3)

    result = run_deterministic_pipeline(demo_catalog, request, destination="US")

    assert result.shipping.status is ResultStatus.INVALID_INPUT
    assert result.summary.status is ResultStatus.INVALID_INPUT
    assert "仅支持目的地 CN" in result.summary.final_text
