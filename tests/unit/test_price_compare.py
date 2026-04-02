from decimal import Decimal

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import Currency, SearchRequest, UserProfile
from globex_agent.tools.item_search import search_items
from globex_agent.tools.price_compare import FX_RATE_VERSION, compare_prices, convert_money


def test_price_compare_groups_same_product_and_finds_sticker_price_winner(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    search = search_items(demo_catalog, demo_requests[0], demo_profiles["demo-user-001"])

    result = compare_prices(search)
    aurora = next(
        comparison
        for comparison in result.comparisons
        if comparison.canonical_product_id == "hp-aurora-quietpro"
    )

    assert result.fx_rate_version == FX_RATE_VERSION
    assert len(aurora.offers) == 3
    assert aurora.best_offer_id == "aliexpress:aqp-3001"
    assert [point.normalized_price for point in aurora.offers] == [
        Decimal("1219.00"),
        Decimal("1269.00"),
        Decimal("1299.00"),
    ]


def test_fixed_demo_exchange_rate_is_reproducible() -> None:
    assert convert_money(Decimal("10.00"), Currency.USD, Currency.CNY) == Decimal("72.00")
    assert convert_money(Decimal("72.00"), Currency.CNY, Currency.USD) == Decimal("10.00")
