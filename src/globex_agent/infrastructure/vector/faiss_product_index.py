"""Async adapter around the existing Faiss HNSW product index."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from globex_agent.domain.catalog.models import StandardItem
from globex_agent.domain.catalog.ports.retrieval_ports import (
    ItemVectorIndex,
    VectorHit,
)
from globex_agent.infrastructure.recall.index import FaissHNSWIndex


class FaissProductIndex(ItemVectorIndex):
    def __init__(
        self,
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        self._index: FaissHNSWIndex | None = None
        self._items: dict[str, StandardItem] = {}
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

    async def ensure_ready(self, vector_dim: int) -> None:
        if self._index is not None:
            return
        empty = np.empty((0, vector_dim), dtype=np.float32)
        self._index = FaissHNSWIndex(
            (),
            empty,
            m=self._m,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
        )

    async def upsert_items(
        self,
        items: list[StandardItem],
        embeddings: list[list[float]],
    ) -> None:
        await asyncio.to_thread(self._upsert_sync, items, embeddings)

    def _upsert_sync(
        self,
        items: list[StandardItem],
        embeddings: list[list[float]],
    ) -> None:
        if len(items) != len(embeddings):
            raise ValueError("items and embeddings must be aligned")
        item_ids = [item.item_id for item in items]
        matrix = np.asarray(embeddings, dtype=np.float32)
        self._index = FaissHNSWIndex(
            item_ids,
            matrix,
            m=self._m,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
        )
        self._items = {item.item_id: item for item in items}

    async def search(
        self,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        if self._index is None:
            raise RuntimeError("Faiss index is not ready")
        return await asyncio.to_thread(self._search_sync, embedding, top_n)

    def _search_sync(
        self,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        assert self._index is not None
        hits = self._index.search(
            np.asarray(embedding, dtype=np.float32),
            top_k=top_n,
        )
        return [VectorHit(item_id=hit.document_id, score=hit.score) for hit in hits]

    def save(self, path: Path) -> None:
        if self._index is None:
            raise RuntimeError("Faiss index is not ready")
        self._index.save(path)

    @classmethod
    def load(cls, path: Path, items: list[StandardItem]) -> FaissProductIndex:
        instance = cls()
        instance._index = FaissHNSWIndex.load(
            path,
            [item.item_id for item in items],
        )
        instance._items = {item.item_id: item for item in items}
        return instance

    async def close(self) -> None:
        self._index = None
        self._items = {}


class PartitionedFaissProductIndex(ItemVectorIndex):
    """Search multiple saved platform/locale Faiss indexes and merge results."""

    def __init__(self) -> None:
        self._partitions: list[tuple[str, FaissHNSWIndex]] = []

    def add_partition(self, partition_id: str, index: FaissHNSWIndex) -> None:
        self._partitions.append((partition_id, index))

    async def ensure_ready(self, vector_dim: int) -> None:
        return None

    async def upsert_items(
        self,
        items: list[StandardItem],
        embeddings: list[list[float]],
    ) -> None:
        raise RuntimeError("PartitionedFaissProductIndex is read-only")

    async def search(
        self,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        return await asyncio.to_thread(self._search_sync, embedding, top_n)

    def _search_sync(
        self,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        rows: list[tuple[float, str]] = []
        vector = np.asarray(embedding, dtype=np.float32)
        for _partition_id, index in self._partitions:
            hits = index.search(vector, top_k=max(1, top_n))
            rows.extend((hit.score, hit.document_id) for hit in hits)
        rows.sort(key=lambda row: (-row[0], row[1]))
        seen: set[str] = set()
        ranked: list[VectorHit] = []
        for score, item_id in rows:
            if item_id in seen:
                continue
            seen.add(item_id)
            ranked.append(VectorHit(item_id=item_id, score=score))
            if len(ranked) >= top_n:
                break
        return ranked

    async def close(self) -> None:
        self._partitions.clear()
