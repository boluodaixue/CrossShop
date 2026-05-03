"""Item repository port owned by the catalog domain."""

from __future__ import annotations

from abc import ABC, abstractmethod

from globex_agent.domain.catalog.models import StandardItem


class ItemRepository(ABC):
    @abstractmethod
    async def find_by_id(self, item_id: str) -> StandardItem | None:
        ...

    @abstractmethod
    async def find_by_ids(self, item_ids: list[str]) -> list[StandardItem]:
        ...

    @abstractmethod
    async def list_all(self) -> list[StandardItem]:
        ...
