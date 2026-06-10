from __future__ import annotations

from pathlib import Path

from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.application.usecases.confirmation_usecases import OrderConfirmationService
from globex_agent.application.usecases.order_usecases import OrderItemInput, PlaceOrderUseCase
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable
from globex_agent.domain.catalog.models import (
    AvailabilityStatus,
    PriceSource,
    StandardItemVariant,
    VariantOption,
)
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec
from globex_agent.domain.order.address import Address
from globex_agent.domain.shipping.tariff_schedule import TariffSchedule
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
    InMemoryOrderRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _catalog():
    return LocalCatalog.from_jsonl(
        PROJECT_ROOT / "data" / "demo" / "products.jsonl", strict=True
    ).catalog


def _variant_item():
    base = next(product for product in _catalog().items if product.price_cny is not None)
    return base.model_copy(
        update={
            "price_cny": None,
            "price_source": PriceSource.UNAVAILABLE,
            "ships_to": ["CN"],
            "variants": [
                StandardItemVariant(
                    variant_id="sku-完整-001",
                    options=[VariantOption(name="颜色", value="黑色")],
                    price_cny=10,
                    price_source=PriceSource.OBSERVED,
                    availability=AvailabilityStatus.AVAILABLE,
                ),
                StandardItemVariant(
                    variant_id="sku-完整-002",
                    options=[VariantOption(name="颜色", value="白色")],
                    price_cny=20,
                    price_source=PriceSource.OBSERVED,
                    availability=AvailabilityStatus.AVAILABLE,
                ),
            ],
        }
    )


async def test_multi_variant_search_quotes_each_available_sku() -> None:
    item = _variant_item()
    tariff = TariffSchedule(rates=ExchangeRateTable())
    usecase = CatalogSearchUseCase(InMemoryItemRepository([item]), tariff_schedule=tariff)
    spec = ProductSearchSpec(
        normalized_query=item.title,
        ship_to="CN",
        top_k=5,
        target_currency="CNY",
    )
    result = await usecase.execute(spec)
    card = next(card for card in result["hits"] if card["item_id"] == item.item_id)
    assert "landed_price" not in card
    available_priced = [
        variant
        for variant in card["variants"]
        if variant["availability"] == "available" and variant["price_major"] is not None
    ]
    assert available_priced
    for variant in available_priced:
        quote = variant["landed_price"]
        assert quote["quantity"] == 1
        assert quote["subtotal_major"] + quote["freight_major"] + quote["tariff_major"] == quote[
            "landed_total_major"
        ]
        assert quote["subtotal_major"] >= 0


async def test_search_without_ship_to_does_not_quote() -> None:
    catalog = _catalog()
    item = next(
        product
        for product in catalog.items
        if not product.variants and product.price_cny is not None
    )
    usecase = CatalogSearchUseCase(InMemoryItemRepository(list(catalog.items)))
    result = await usecase.execute(
        ProductSearchSpec(normalized_query=item.title, top_k=5, target_currency="CNY")
    )
    card = next(card for card in result["hits"] if card["item_id"] == item.item_id)
    assert "landed_price" not in card


async def test_unsupported_destination_keeps_explicit_unavailable_quote_metadata() -> None:
    catalog = _catalog()
    base = next(
        product
        for product in catalog.items
        if not product.variants and product.price_cny is not None
    )
    item = base.model_copy(update={"ships_to": ["XZ"]})
    usecase = CatalogSearchUseCase(InMemoryItemRepository([item]))
    result = await usecase.execute(
        ProductSearchSpec(
            normalized_query=item.title,
            ship_to="XZ",
            top_k=5,
            target_currency="CNY",
        )
    )
    card = next(card for card in result["hits"] if card["item_id"] == item.item_id)
    quote = card["landed_price"]
    assert quote["quantity"] == 1
    assert quote["ship_to"] == "XZ"
    assert quote["currency"] == "CNY"
    assert "unavailable_reason" in quote


async def test_prepare_and_confirm_recalculate_same_line_quotes() -> None:
    catalog = _catalog()
    item = next(
        product
        for product in catalog.items
        if not product.variants and product.price_cny is not None
    )
    item_repo = InMemoryItemRepository(list(catalog.items))
    order_repo = InMemoryOrderRepository()
    tariff = TariffSchedule(rates=ExchangeRateTable())
    place = PlaceOrderUseCase(item_repo, order_repo, tariff_schedule=tariff)
    service = OrderConfirmationService(
        item_repo,
        order_repo,
        place,
        cancel_order=None,
        tariff_schedule=tariff,
    )
    address = Address(
        recipient_name="Test Buyer",
        country="CN",
        state="ZJ",
        city="Hangzhou",
        address_line="redacted line",
        postal_code="310000",
        phone="13800000000",
    )
    preview = await service.prepare_order(
        session_id="quote-session",
        user_id="buyer",
        items=[OrderItemInput(item.item_id, None, 2)],
        shipping_address=address,
        include_token=True,
    )
    assert preview["merchandise_subtotal_major"] > 0
    assert preview["landed_total_major"] == preview["total_amount_major"]
    assert preview["pricing"]["line_quotes"][0]["quantity"] == 2
    assert preview["freight_major"] > 0
    confirmed = await service.confirm_order(
        confirmation_id=preview["confirmation_id"],
        token=preview["confirmation_token"],
        session_id="quote-session",
    )
    assert confirmed["pricing"] == preview["pricing"]
    assert confirmed["landed_total_major"] == preview["landed_total_major"]
