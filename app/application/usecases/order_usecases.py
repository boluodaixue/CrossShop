# -*- coding: utf-8 -*-
"""订单三个 UseCase：PlaceOrder / QueryOrder / CancelOrder

TradeAgent 的工具层只做参数搬运，业务规则（库存扣减、状态机、金额计算）全部收敛在这里与 Order 聚合内。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.order.address import Address
from app.domain.order.order import Order
from app.domain.order.order_line import OrderLine
from app.domain.order.ports.order_repository import OrderRepository

_ORDER_NOT_ACCESSIBLE = "订单不存在或无权访问"


@dataclass(frozen=True)
class OrderItemInput:
    product_id: str
    sku_id: str
    quantity: int


class PlaceOrderUseCase:
    def __init__(
        self, product_repo: ProductRepository, order_repo: OrderRepository
    ) -> None:
        self._product_repo = product_repo
        self._order_repo = order_repo

    async def execute(
        self, buyer_id: str, items: list[OrderItemInput], shipping_address: Address
    ) -> dict:
        if not items:
            raise ValueError("PlaceOrder.items 不能为空")

        lines: list[OrderLine] = []
        deducted: list[tuple] = []  # 已扣减的 (sku, quantity)，失败时回滚
        try:
            for item in items:
                product = await self._product_repo.find_by_id(item.product_id)
                if product is None:
                    raise ValueError(f"商品不存在：{item.product_id}")
                sku = product.find_sku(item.sku_id)
                if sku is None:
                    raise ValueError(f"Sku 不存在：{item.product_id}/{item.sku_id}")
                sku.deduct_stock(item.quantity)
                deducted.append((sku, item.quantity))
                lines.append(
                    OrderLine(
                        product_id=product.product_id,
                        sku_id=sku.sku_id,
                        title=f"{product.title}（{sku.spec}）",
                        unit_price=sku.price,
                        quantity=item.quantity,
                    ),
                )
            order = Order.place(
                order_id=await self._order_repo.next_order_id(),
                buyer_id=buyer_id,
                shipping_address=shipping_address,
                lines=lines,
            )
            await self._order_repo.save(order)
        except Exception:
            for sku, quantity in deducted:
                sku.restore_stock(quantity)
            raise
        return order.snapshot()


class QueryOrderUseCase:
    def __init__(self, order_repo: OrderRepository) -> None:
        self._order_repo = order_repo

    async def execute(self, buyer_id: str, order_id: str) -> dict:
        order = await self._order_repo.find_by_id(order_id)
        if order is None or order.buyer_id != buyer_id:
            raise ValueError(_ORDER_NOT_ACCESSIBLE)
        return order.snapshot()


class CancelOrderUseCase:
    def __init__(
        self, product_repo: ProductRepository, order_repo: OrderRepository
    ) -> None:
        self._product_repo = product_repo
        self._order_repo = order_repo

    async def execute(self, buyer_id: str, order_id: str, reason: str) -> dict:
        order = await self._order_repo.find_by_id(order_id)
        if order is None or order.buyer_id != buyer_id:
            raise ValueError(_ORDER_NOT_ACCESSIBLE)
        previous_state = (order.status, order.cancelled_at, order.cancel_reason)
        restored: list[tuple] = []
        try:
            order.cancel(reason)
            # 取消后回补库存；若订单保存失败，下面会补偿性恢复库存和聚合状态。
            for line in order.lines:
                product = await self._product_repo.find_by_id(line.product_id)
                if product is not None:
                    sku = product.find_sku(line.sku_id)
                    if sku is not None:
                        sku.restore_stock(line.quantity)
                        restored.append((sku, line.quantity))
            await self._order_repo.save(order)
        except Exception:
            for sku, quantity in reversed(restored):
                sku.deduct_stock(quantity)
            order.status, order.cancelled_at, order.cancel_reason = previous_state
            raise
        return order.snapshot()
