from decimal import Decimal

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import Currency, SearchRequest
from globex_agent.tools.item_search import search_items
from globex_agent.tools.price_compare import FX_RATE_VERSION, compare_prices, convert_money


def _merged_candidates(
    catalog: LocalCatalog,
    request: SearchRequest,
):
    return [
        candidate
        for platform in sorted(request.platforms, key=lambda value: value.value)
        for candidate in search_items(
            catalog,
            request.query,
            platform,
            request.top_k,
            request.hard_constraints,
        ).candidates
    ]


def test_price_compare_ranks_merged_candidates_and_keeps_group_relation(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    candidates = _merged_candidates(demo_catalog, demo_requests[0])

    result = compare_prices(candidates)
    aurora = [
        point for point in result.ranked if point.same_group_id == "hp-aurora-quietpro"
    ]

    assert result.fx_rate_version == FX_RATE_VERSION
    assert len(result.ranked) == 9
    assert len(aurora) == 3
    assert result.cheapest_per_group["hp-aurora-quietpro"] == "aliexpress:aqp-3001"
    assert [point.price_cny for point in aurora] == [
        Decimal("1219.00"),
        Decimal("1269.00"),
        Decimal("1299.00"),
    ]


def test_price_compare_applies_course_top_n_pruning(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    result = compare_prices(_merged_candidates(demo_catalog, demo_requests[0]), top_n=4)

    assert len(result.ranked) == 4
    assert result.ranked == sorted(result.ranked, key=lambda point: point.price_cny)


def test_fixed_demo_exchange_rate_is_reproducible() -> None:
    assert convert_money(Decimal("10.00"), Currency.USD, Currency.CNY) == Decimal("71.80")
    assert convert_money(Decimal("71.80"), Currency.CNY, Currency.USD) == Decimal("10.00")
