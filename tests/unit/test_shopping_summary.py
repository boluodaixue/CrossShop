from globex_agent.catalog import LocalCatalog
from globex_agent.domain import ResultStatus, SearchRequest, UserProfile
from globex_agent.pipeline import run_deterministic_pipeline


def test_summary_is_stable_and_discloses_demo_assumptions(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    request = demo_requests[0]

    first = run_deterministic_pipeline(
        demo_catalog, request, demo_profiles["demo-user-001"]
    ).summary
    second = run_deterministic_pipeline(
        demo_catalog, request, demo_profiles["demo-user-001"]
    ).summary

    assert first == second
    assert first.status is ResultStatus.OK
    assert "找到 3 个合规推荐" in first.final_text
    assert "离线演示估算" in first.final_text
    assert len(first.recommendation.items) == 3


def test_summary_explains_when_all_candidates_are_rejected(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    result = run_deterministic_pipeline(demo_catalog, demo_requests[2]).summary

    assert result.status is ResultStatus.NO_RESULTS
    assert "未找到满足" in result.final_text
    assert "超过预算" in result.final_text
