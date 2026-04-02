from decimal import Decimal

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import ResultStatus, SearchRequest, UserProfile
from globex_agent.tools.item_search import search_items
from globex_agent.tools.price_compare import compare_prices
from globex_agent.tools.shipping_calc import SHIPPING_RULE_VERSION, calculate_shipping


def test_shipping_calc_matches_hand_calculated_landed_cost(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    search = search_items(demo_catalog, demo_requests[0], demo_profiles["demo-user-001"])
    prices = compare_prices(search)

    result = calculate_shipping(prices)
    quote = next(quote for quote in result.quotes if quote.offer_id == "shopee:aqp-2001")

    assert result.rule_version == SHIPPING_RULE_VERSION
    assert quote.item_price == Decimal("1269.00")
    assert quote.shipping_fee == Decimal("18.00")
    assert quote.tax_fee == Decimal("76.14")
    assert quote.landed_price == Decimal("1363.14")


def test_landed_cost_can_change_sticker_price_winner(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    search = search_items(demo_catalog, demo_requests[0])
    prices = compare_prices(search)
    aurora = next(
        comparison
        for comparison in prices.comparisons
        if comparison.canonical_product_id == "hp-aurora-quietpro"
    )

    result = calculate_shipping(prices)
    aurora_quotes = [
        quote for quote in result.quotes if quote.canonical_product_id == "hp-aurora-quietpro"
    ]

    assert aurora.best_offer_id == "aliexpress:aqp-3001"
    assert min(aurora_quotes, key=lambda quote: quote.landed_price).offer_id == "shopee:aqp-2001"


def test_unsupported_destination_returns_structured_failure(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    prices = compare_prices(search_items(demo_catalog, demo_requests[0]))

    result = calculate_shipping(prices, destination="US")

    assert result.status is ResultStatus.INVALID_INPUT
    assert result.quotes == []
    assert result.issues[0].code == "unsupported_destination"
