"""Async adapter tests for BGE-M3 embedding, Faiss, and reranking."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np

from globex_agent.domain import Currency, DataProvenance, Platform, ProvenanceKind, StandardItem
from globex_agent.infrastructure.embedding.bge_m3_embedding import BgeM3EmbeddingClient
from globex_agent.infrastructure.rerank.bge_reranker import BgeReranker
from globex_agent.infrastructure.vector.faiss_product_index import FaissProductIndex


def _item(item_id: str) -> StandardItem:
    return StandardItem(
        item_id=item_id,
        same_group_id=item_id,
        platform=Platform.AMAZON,
        title=f"Item {item_id}",
        category_path=["demo"],
        price_cny=Decimal("10.00"),
        currency_raw=Currency.CNY,
        ingested_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        provenance=DataProvenance(
            kind=ProvenanceKind.SYNTHETIC,
            source="adapter test",
            generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        ),
    )


class FakeEncoder:
    encoder_id = "fake-bge-m3"

    def encode_queries(self, texts):
        return np.asarray([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)


class FakeReranker:
    reranker_id = "fake-reranker"

    def score_texts(self, query: str, texts):
        return tuple(float(len(text)) for text in texts)


class TestAsyncAdapters:
    async def test_bge_embedding_adapter_uses_fake_encoder(self) -> None:
        client = BgeM3EmbeddingClient(encoder=FakeEncoder())
        assert await client.embed("query") == [1.0, 0.0, 0.0]
        assert await client.embed_batch([]) == []

    async def test_faiss_product_index_search(self) -> None:
        items = [_item("amazon:a"), _item("amazon:b"), _item("amazon:c")]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        index = FaissProductIndex()
        await index.ensure_ready(3)
        await index.upsert_items(items, embeddings)
        hits = await index.search([1.0, 0.0, 0.0], top_n=2)
        assert hits[0].item_id == "amazon:a"
        assert len(hits) == 2
        await index.close()

    async def test_bge_reranker_adapter(self) -> None:
        reranker = BgeReranker(reranker=FakeReranker())
        assert await reranker.rerank("q", ["abc", "a"]) == [3.0, 1.0]
        assert await reranker.rerank("q", []) == []
