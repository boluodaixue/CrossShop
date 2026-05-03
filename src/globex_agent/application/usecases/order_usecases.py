"""Order use cases: place, query, and cancel using item_id + variant_id."""

from __future__ import annotations

from dataclasses import dataclass

from globex_agent.domain.catalog.models import StandardItem
from globex_agent.domain.catalog.money import Money
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.domain.order.address import Address
from globex_agent.domain.order.order import Order
from globex_agent.domain.order.order_line import OrderLine
from globex_agent.domain.order.ports.order_repository import OrderRepository


@dataclass(frozen=True)
class OrderItemInput:
    item_id: str
    variant_id: str
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
        if not items:
            raise ValueError("PlaceOrder.items 不能为空")
        lines: list[OrderLine] = []
        for item in items:
            if item.quantity <= 0:
                raise ValueError("OrderLine.quantity 必须为正整数")
            catalog_item = await self._item_repo.find_by_id(item.item_id)
            if catalog_item is None:
                raise ValueError(f"商品不存在：{item.item_id}")
            if not catalog_item.is_available:
                raise ValueError(f"商品不可售：{item.item_id}")
            unit_price = _item_price(catalog_item)
            title = _line_title(catalog_item, item.variant_id)
            lines.append(
                OrderLine(
                    item_id=catalog_item.item_id,
                    variant_id=_resolved_variant_id(catalog_item, item.variant_id),
                    title=title,
                    unit_price=unit_price,
                    quantity=item.quantity,
                )
            )
        order = Order.place(
            order_id=await self._order_repo.next_order_id(),
            buyer_id=buyer_id,
            shipping_address=shipping_address,
            lines=lines,
        )
        await self._order_repo.save(order)
        return order.snapshot()


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
        order = await self._order_repo.find_by_id(order_id)
        if order is None:
            raise ValueError(f"订单不存在：{order_id}")
        order.cancel(reason)
        await self._order_repo.save(order)
        return order.snapshot()


def _item_price(item: StandardItem) -> Money:
    if item.price_cny is None:
        raise ValueError(f"商品缺少价格：{item.item_id}")
    return Money.from_major_units(float(item.price_cny), "CNY")


def _resolved_variant_id(item: StandardItem, requested: str) -> str:
    if not item.variants:
        return requested or f"{item.item_id}:default"
    for variant in item.variants:
        variant_id = str(variant.get("variant_id") or variant.get("id") or "")
        if variant_id == requested:
            return variant_id
    raise ValueError(f"变体不存在：{item.item_id}/{requested}")


def _line_title(item: StandardItem, requested: str) -> str:
    if not item.variants:
        return item.title
    for variant in item.variants:
        variant_id = str(variant.get("variant_id") or variant.get("id") or "")
        if variant_id == requested:
            spec = str(variant.get("spec") or variant.get("name") or "")
            return f"{item.title}（{spec}）" if spec else item.title
    return item.title
