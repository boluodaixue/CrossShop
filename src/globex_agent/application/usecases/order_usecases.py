"""Order use cases: place, query, and cancel using item_id + variant_id."""

from __future__ import annotations

from dataclasses import dataclass

from globex_agent.domain.catalog.models import AvailabilityStatus, StandardItem
from globex_agent.domain.catalog.money import Money
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.domain.order.address import Address
from globex_agent.domain.order.order import Order
from globex_agent.domain.order.order_line import OrderLine
from globex_agent.domain.order.ports.order_repository import OrderRepository


@dataclass(frozen=True)
class OrderItemInput:
    item_id: str
    variant_id: str | None
    quantity: int


class PlaceOrderUseCase:
    def __init__(self, item_repo: ItemRepository, order_repo: OrderRepository) -> None:
        self._item_repo = item_repo
        self._order_repo = order_repo

    async def execute(
        self,
        buyer_id: str,
        items: list[OrderItemInput],
        shipping_address: Address,
    ) -> dict:
        raise ValueError(
            "直接下单已禁用：必须先通过 PrepareOrder，再由可信 ConfirmOrder 执行"
        )

    async def _execute_confirmed(
        self,
        buyer_id: str,
        items: list[OrderItemInput],
        shipping_address: Address,
    ) -> dict:
        if not items:
            raise ValueError("PlaceOrder.items 不能为空")
        lines, _ = await _build_order_lines(self._item_repo, items)
        order = Order.place(
            order_id=await self._order_repo.next_order_id(),
            buyer_id=buyer_id,
            shipping_address=shipping_address,
            lines=lines,
        )
        await self._order_repo.save(order)
        return order.snapshot()


async def _build_order_lines(
    item_repo: ItemRepository,
    items: list[OrderItemInput],
) -> tuple[list[OrderLine], list[dict]]:
    """Resolve current catalog facts for prepare and confirm comparisons."""

    if not items:
        raise ValueError("PlaceOrder.items 不能为空")
    lines: list[OrderLine] = []
    facts: list[dict] = []
    for item in items:
        if item.quantity <= 0:
            raise ValueError("OrderLine.quantity 必须为正整数")
        catalog_item = await item_repo.find_by_id(item.item_id)
        if catalog_item is None:
            raise ValueError(f"商品不存在：{item.item_id}")
        if catalog_item.availability is not AvailabilityStatus.AVAILABLE:
            raise ValueError(f"商品不可售：{item.item_id}")
        variant = _resolve_variant(catalog_item, item.variant_id)
        unit_price = _item_price(catalog_item, variant)
        title = _line_title(catalog_item, variant)
        variant_display_name = (
            " / ".join(f"{option.name}: {option.value}" for option in variant.options)
            if variant is not None
            else ""
        )
        lines.append(
            OrderLine(
                item_id=catalog_item.item_id,
                variant_id=variant.variant_id if variant is not None else None,
                title=title,
                unit_price=unit_price,
                quantity=item.quantity,
                variant_display_name=variant_display_name,
            )
        )
        facts.append(
            {
                "item_id": catalog_item.item_id,
                "variant_id": variant.variant_id if variant is not None else None,
                "quantity": item.quantity,
                "title": title,
                "variant_display_name": variant_display_name,
                "unit_price_minor": unit_price.amount_in_minor_units,
                "currency": unit_price.currency,
                "availability": catalog_item.availability.value,
                "variant_availability": (
                    variant.availability.value if variant is not None else None
                ),
            }
        )
    return lines, facts


class QueryOrderUseCase:
    def __init__(self, order_repo: OrderRepository) -> None:
        self._order_repo = order_repo

    async def execute(self, order_id: str) -> dict:
        order = await self._order_repo.find_by_id(order_id)
        if order is None:
            raise ValueError(f"订单不存在：{order_id}")
        return order.snapshot()


class CancelOrderUseCase:
    def __init__(self, order_repo: OrderRepository) -> None:
        self._order_repo = order_repo

    async def execute(self, order_id: str, reason: str) -> dict:
        raise ValueError(
            "直接取消已禁用：必须先通过 PrepareCancel，再由可信 ConfirmCancel 执行"
        )

    async def _execute_confirmed(self, order_id: str, reason: str) -> dict:
        order = await self._order_repo.find_by_id(order_id)
        if order is None:
            raise ValueError(f"订单不存在：{order_id}")
        order.cancel(reason)
        await self._order_repo.save(order)
        return order.snapshot()


def _item_price(item: StandardItem, variant=None) -> Money:
    price = variant.price_cny if variant is not None else item.price_cny
    if price is None:
        raise ValueError(f"商品缺少价格：{item.item_id}")
    return Money.from_major_units(float(price), "CNY")


def _resolve_variant(item: StandardItem, requested: str | None):
    if not item.variants:
        if requested is not None:
            raise ValueError(f"无规格商品的 variant_id 必须为 null：{item.item_id}")
        return None
    for variant in item.variants:
        if variant.variant_id == requested:
            if variant.availability is not AvailabilityStatus.AVAILABLE:
                raise ValueError(f"规格不可售：{item.item_id}/{requested}")
            return variant
    raise ValueError(f"变体不存在：{item.item_id}/{requested}")


def _line_title(item: StandardItem, variant) -> str:
    if not item.variants:
        return item.title
    display = " / ".join(f"{option.name}: {option.value}" for option in variant.options)
    return f"{item.title}（{display}）" if display else item.title
