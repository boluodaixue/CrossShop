"""Tests for SQLite-backed runtime catalog and partitioned Faiss search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from globex_agent.infrastructure.persistence.sqlite_item_repository import (
    SqliteItemRepository,
)
from globex_agent.infrastructure.recall.index import FaissHNSWIndex
from globex_agent.infrastructure.vector.faiss_product_index import (
    PartitionedFaissProductIndex,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMAZON_DB = (
    PROJECT_ROOT / "data" / "processed" / "databases" / "amazon" / "catalog.sqlite3"
)


class TestSqliteItemRepository:
    async def test_find_by_id_returns_standard_item(self) -> None:
        repo = SqliteItemRepository([AMAZON_DB])
        item = await repo.find_by_id("amazon:us:B00BJ17WKK")
        assert item is not None
        assert item.platform.value == "amazon"
        assert item.locale is not None and item.locale.value == "us"

    async def test_find_by_ids_preserves_request_order(self) -> None:
        repo = SqliteItemRepository([AMAZON_DB])
        items = await repo.find_by_ids(
            ["amazon:us:B00BJ17WKK", "missing:item", "amazon:us:B00TZIW41G"]
        )
        assert [item.item_id for item in items] == [
            "amazon:us:B00BJ17WKK",
            "amazon:us:B00TZIW41G",
        ]


class TestPartitionedFaissProductIndex:
    async def test_merges_hits_from_multiple_partitions(self) -> None:
        left = FaissHNSWIndex(
            ["a1", "a2"],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        right = FaissHNSWIndex(
            ["b1", "b2"],
            np.asarray([[0.8, 0.6], [0.0, 1.0]], dtype=np.float32),
        )
        index = PartitionedFaissProductIndex()
        index.add_partition("left", left)
        index.add_partition("right", right)
        hits = await index.search([1.0, 0.0], top_n=2)
        assert [hit.item_id for hit in hits] == ["a1", "b1"]
        await index.close()
