from decimal import Decimal

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import ResultStatus, SearchRequest, UserProfile
from globex_agent.pipeline import run_deterministic_pipeline


def test_full_pipeline_preserves_course_stage_boundaries(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    result = run_deterministic_pipeline(
        demo_catalog,
        demo_requests[0],
        demo_profiles["demo-user-001"],
    )

    assert len(result.search) == 3
    assert all(output.status is ResultStatus.OK for output in result.search)
    assert len(result.price_comparison.ranked) == 9
    assert len(result.shipping.items) == 9
    assert [pick.same_group_id for pick in result.selection.picks] == [
        "hp-nimbus-lite",
        "hp-sonic-commute",
        "hp-aurora-quietpro",
    ]
    assert all(pick.landed_cny <= Decimal("1500.00") for pick in result.selection.picks)
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

    assert all(output.status is ResultStatus.NO_RESULTS for output in result.search)
    assert result.selection.status is ResultStatus.NO_RESULTS
    assert result.summary.status is ResultStatus.NO_RESULTS
    assert result.summary.picks == []


def test_pipeline_propagates_invalid_destination(demo_catalog: LocalCatalog) -> None:
    request = SearchRequest(query="头戴式降噪耳机", top_k=3)

    result = run_deterministic_pipeline(demo_catalog, request, destination="US")

    assert result.shipping.status is ResultStatus.INVALID_INPUT
    assert result.summary.status is ResultStatus.INVALID_INPUT
    assert "仅支持目的地 CN" in result.summary.final_text
