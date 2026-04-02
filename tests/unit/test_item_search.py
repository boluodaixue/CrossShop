from globex_agent.catalog import LocalCatalog
from globex_agent.domain import Platform, ResultStatus, SearchRequest, UserProfile
from globex_agent.tools.item_search import search_items


def test_search_returns_ranked_headphone_candidates(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    request = demo_requests[0]
    result = search_items(demo_catalog, request, demo_profiles["demo-user-001"])

    assert result.status is ResultStatus.OK
    assert len(result.candidates) == 3
    assert result.matched_catalog_items > len(result.candidates)
    assert result.truncated is True
    assert all("耳机" in candidate.item.category_path for candidate in result.candidates)
    assert result.candidates == sorted(
        result.candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.item.title.casefold(),
            candidate.item.canonical_product_id,
        ),
    )


def test_search_filters_platforms_and_out_of_scope_offers(
    demo_catalog: LocalCatalog,
) -> None:
    request = SearchRequest(
        query="头戴式降噪耳机",
        platforms={Platform.AMAZON},
        top_k=2,
    )

    result = search_items(demo_catalog, request)

    assert len(result.candidates) == 2
    assert all(
        {offer.platform for offer in candidate.item.offers} == {Platform.AMAZON}
        for candidate in result.candidates
    )


def test_search_returns_structured_no_results(demo_catalog: LocalCatalog) -> None:
    result = search_items(demo_catalog, SearchRequest(query="zzzzzzzz", top_k=3))

    assert result.status is ResultStatus.NO_RESULTS
    assert result.candidates == []
    assert result.issues[0].code == "no_search_match"
