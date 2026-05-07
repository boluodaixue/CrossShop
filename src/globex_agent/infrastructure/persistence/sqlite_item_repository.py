"""SQLite-backed ItemRepository for the two platform catalog databases."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from globex_agent.domain.catalog.models import StandardItem
from globex_agent.domain.catalog.ports.item_repository import ItemRepository


class SqliteItemRepository(ItemRepository):
    """Read StandardItem records from one or more platform catalog SQLite files."""

    def __init__(self, database_paths: list[Path]) -> None:
        self._paths = [Path(path) for path in database_paths]
        missing = [str(path) for path in self._paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"catalog database not found: {missing}")

    async def find_by_id(self, item_id: str) -> StandardItem | None:
        found = await self.find_by_ids([item_id])
        return found[0] if found else None

    async def find_by_ids(self, item_ids: list[str]) -> list[StandardItem]:
        if not item_ids:
            return []
        return await asyncio.to_thread(self._find_by_ids_sync, item_ids)

    async def list_all(self) -> list[StandardItem]:
        return await asyncio.to_thread(self._list_all_sync)

    def _find_by_ids_sync(self, item_ids: list[str]) -> list[StandardItem]:
        items: list[StandardItem] = []
        wanted = list(dict.fromkeys(item_ids))
        placeholders = ",".join("?" for _ in wanted)
        for path in self._paths:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                rows = connection.execute(
                    f"SELECT record_json FROM items WHERE item_id IN ({placeholders})",
                    wanted,
                ).fetchall()
            items.extend(
                StandardItem.model_validate_json(row[0])
                for row in rows
            )
        by_id = {item.item_id: item for item in items}
        return [by_id[item_id] for item_id in item_ids if item_id in by_id]

    def _list_all_sync(self) -> list[StandardItem]:
        items: list[StandardItem] = []
        for path in self._paths:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                rows = connection.execute(
                    "SELECT record_json FROM items"
                ).fetchall()
            items.extend(
                StandardItem.model_validate_json(row[0])
                for row in rows
            )
        return items
