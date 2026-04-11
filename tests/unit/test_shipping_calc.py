from decimal import Decimal

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import ResultStatus, SearchRequest
from globex_agent.tools.item_search import search_items
from globex_agent.tools.price_compare import compare_prices
from globex_agent.tools.shipping_calc import SHIPPING_RULE_VERSION, calculate_shipping


def _price_points(
    catalog: LocalCatalog,
    request: SearchRequest,
):
    candidates = [
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
    return compare_prices(candidates).ranked


def test_shipping_calc_matches_chapter_twelve_tables(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    points = _price_points(demo_catalog, demo_requests[0])

    result = calculate_shipping(points)
    item = next(item for item in result.items if item.item_id == "shopee:aqp-2001")

    assert result.rule_version == SHIPPING_RULE_VERSION
    assert item.price_cny == Decimal("1269.00")
    assert item.shipping_cny == Decimal("60.00")
    assert item.duty_cny == Decimal("76.14")
    assert item.landed_cny == Decimal("1405.14")
    assert item.duty_tier == "免征"


def test_landed_cost_can_change_sticker_price_winner(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    points = _price_points(demo_catalog, demo_requests[0])
    aurora_points = [
        point for point in points if point.same_group_id == "hp-aurora-quietpro"
    ]
    result = calculate_shipping(points)
    aurora_items = [
        item for item in result.items if item.same_group_id == "hp-aurora-quietpro"
    ]

    assert min(aurora_points, key=lambda point: point.price_cny).item_id == (
        "aliexpress:aqp-3001"
    )
    assert min(aurora_items, key=lambda item: item.landed_cny).item_id == "shopee:aqp-2001"


def test_unsupported_destination_returns_structured_failure(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    result = calculate_shipping(
        _price_points(demo_catalog, demo_requests[0]),
        destination="US",
    )

    assert result.status is ResultStatus.INVALID_INPUT
    assert result.items == []
    assert result.issues[0].code == "unsupported_destination"
