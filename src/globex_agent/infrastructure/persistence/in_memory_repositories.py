"""In-memory repositories for development and tests."""

from __future__ import annotations

import itertools

from globex_agent.domain.catalog.models import StandardItem
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.domain.order.order import Order
from globex_agent.domain.order.ports.order_repository import OrderRepository


class InMemoryItemRepository(ItemRepository):
    def __init__(self, items: list[StandardItem] | None = None) -> None:
        self._items: dict[str, StandardItem] = {
            item.item_id: item for item in (items or [])
        }

    async def find_by_id(self, item_id: str) -> StandardItem | None:
        return self._items.get(item_id)

    async def find_by_ids(self, item_ids: list[str]) -> list[StandardItem]:
        return [self._items[item_id] for item_id in item_ids if item_id in self._items]

    async def list_all(self) -> list[StandardItem]:
        return list(self._items.values())


class InMemoryOrderRepository(OrderRepository):
    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._counter = itertools.count(1)

    async def save(self, order: Order) -> None:
        self._orders[order.order_id] = order

    async def find_by_id(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    async def next_order_id(self) -> str:
        return f"GBX-{next(self._counter):06d}"
