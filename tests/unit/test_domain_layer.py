"""Domain-layer tests for money, shipping rules, and the order aggregate."""

import pytest

from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable
from globex_agent.domain.catalog.money import Money
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec
from globex_agent.domain.order.address import Address
from globex_agent.domain.order.order import Order, OrderStatus
from globex_agent.domain.order.order_line import OrderLine
from globex_agent.domain.shipping.tariff_schedule import TariffSchedule


def _address() -> Address:
    return Address(
        recipient_name="张三",
        country="CN",
        state="浙江",
        city="杭州",
        address_line="西湖区某路 1 号",
        postal_code="310000",
        phone="13800000000",
    )


def _line(currency: str = "CNY", major: float = 189.0, quantity: int = 2) -> OrderLine:
    return OrderLine(
        item_id="taobao:item-1",
        variant_id="taobao:item-1:green",
        title="Nomadica 旅行三件套（军绿色）",
        unit_price=Money.from_major_units(major, currency),
        quantity=quantity,
    )


class TestMoney:
    def test_minor_units_storage(self) -> None:
        money = Money.from_major_units(189.0, "CNY")
        assert money.amount_in_minor_units == 18900
        assert money.to_major_units() == 189.0

    def test_add_and_multiply(self) -> None:
        total = Money.from_major_units(10.5, "USD").add(
            Money.from_major_units(0.5, "USD")
        )
        assert total.amount_in_minor_units == 1100
        assert total.multiply(3).amount_in_minor_units == 3300

    def test_reject_invalid_amount_and_currency(self) -> None:
        with pytest.raises(ValueError):
            Money.of(-1, "CNY")
        with pytest.raises(ValueError):
            Money.of(100, "XXX")
        with pytest.raises(ValueError, match="币种不一致"):
            Money.of(100, "CNY").add(Money.of(100, "USD"))


class TestOrderStateMachine:
    def test_place_enters_confirmed(self) -> None:
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line()])
        assert order.status is OrderStatus.CONFIRMED
        assert order.confirmed_at is not None

    def test_total_amount_and_snapshot(self) -> None:
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line(quantity=2)])
        assert order.total_amount().to_major_units() == 378.0
        snapshot = order.snapshot()
        assert snapshot["lines"][0]["item_id"] == "taobao:item-1"
        assert snapshot["lines"][0]["variant_id"] == "taobao:item-1:green"

    def test_cancel_confirmed_order_requires_reason(self) -> None:
        order = Order.place("GBX-000001", "buyer-1", _address(), [_line()])
        with pytest.raises(ValueError, match="reason"):
            order.cancel("  ")
        order.cancel("买家改主意了")
        assert order.status is OrderStatus.CANCELLED
        with pytest.raises(ValueError, match="仅 CONFIRMED"):
            order.cancel("  ")

    def test_reject_mixed_currency_and_empty_lines(self) -> None:
        with pytest.raises(ValueError, match="币种不一致"):
            Order.place("GBX-000001", "buyer-1", _address(), [_line("CNY"), _line("USD")])
        with pytest.raises(ValueError, match="订单行"):
            Order.place("GBX-000001", "buyer-1", _address(), [])


class TestTariffSchedule:
    def test_de_minimis_and_multi_quantity_freight(self) -> None:
        rates = ExchangeRateTable()
        schedule = TariffSchedule(rates=rates)
        quote = schedule.quote(
            subtotal=Money.from_major_units(189, "CNY"),
            category="旅行装备",
            ship_to="CN",
            quantity=1,
            target_currency="CNY",
        )
        assert quote.de_minimis_applied is True
        assert quote.tariff.to_major_units() == 0.0
        assert quote.freight.to_major_units() == 25.0
        triple = schedule.quote(
            subtotal=Money.from_major_units(300, "CNY"),
            category="旅行装备",
            ship_to="CN",
            quantity=3,
            target_currency="CNY",
        )
        assert triple.freight.to_major_units() == pytest.approx(55.0)

    def test_currency_conversion_and_unsupported_destination(self) -> None:
        schedule = TariffSchedule(rates=ExchangeRateTable())
        quote = schedule.quote(
            subtotal=Money.from_major_units(710, "CNY"),
            category="旅行装备",
            ship_to="US",
            quantity=1,
            target_currency="USD",
        )
        assert quote.subtotal.currency == "USD"
        assert quote.subtotal.to_major_units() == pytest.approx(100.0, abs=0.05)
        with pytest.raises(ValueError, match="暂不支持的目的国"):
            schedule.quote(
                subtotal=Money.from_major_units(100, "CNY"),
                category="旅行装备",
                ship_to="BR",
                quantity=1,
                target_currency="CNY",
            )


class TestProductSearchSpec:
    def test_rejects_empty_query_and_non_positive_top_k(self) -> None:
        with pytest.raises(ValueError):
            ProductSearchSpec(normalized_query="  ")
        with pytest.raises(ValueError):
            ProductSearchSpec(normalized_query="旅行背包", top_k=0)
