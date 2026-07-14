from __future__ import annotations

import inspect
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.catalog.opensearch_product_h1 import (
    INDEX_NAMES,
    RRF_PIPELINE_NAME,
    VECTOR_DIMENSION,
    hybrid_query,
)
from app.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ProductVectorIndex,
    VectorHit,
)
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryProductRepository,
)
from app.infrastructure.vector.opensearch_product_index import OpenSearchProductIndex
from app.infrastructure.vector.qdrant_product_index import QdrantProductIndex


def _vector() -> list[float]:
    value = 1.0 / math.sqrt(VECTOR_DIMENSION)
    return [value] * VECTOR_DIMENSION


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "http://127.0.0.1:9200/_search"),
    )


def _hits(*product_ids: str) -> dict:
    return {
        "hits": {
            "hits": [
                {
                    "_id": product_id,
                    "_score": 1.0 / (index + 1),
                    "_source": {"product_id": product_id},
                }
                for index, product_id in enumerate(product_ids)
            ]
        }
    }


def test_port_requires_keyword_only_query_embedding_and_top_n() -> None:
    parameters = inspect.signature(ProductVectorIndex.search).parameters
    assert list(parameters) == ["self", "query", "embedding", "top_n"]
    assert all(
        parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("query", "embedding", "top_n")
    )
    assert parameters["query"].default is inspect.Parameter.empty


@pytest.mark.parametrize("index_name", tuple(INDEX_NAMES.values()))
async def test_adapter_sends_exactly_one_frozen_hybrid_request(index_name: str) -> None:
    client = AsyncMock()
    client.post.return_value = _response(_hits("p1", "p2"))
    with patch(
        "app.infrastructure.vector.opensearch_product_index.httpx.AsyncClient",
        return_value=client,
    ):
        index = OpenSearchProductIndex("http://127.0.0.1:9200", index_name)
        vector = _vector()
        hits = await index.search(query="轻便旅行背包", embedding=vector, top_n=2)

    assert hits == [VectorHit("p1", 1.0), VectorHit("p2", 0.5)]
    client.post.assert_awaited_once_with(
        f"/{index_name}/_search",
        params={"search_pipeline": RRF_PIPELINE_NAME},
        json=hybrid_query("轻便旅行背包", vector, size=2),
    )
    client.get.assert_not_awaited()


async def test_amazon_site_adapter_filters_both_branches_in_the_same_request() -> None:
    client = AsyncMock()
    client.post.return_value = _response(_hits("amazon:jp:p1"))
    with patch(
        "app.infrastructure.vector.opensearch_product_index.httpx.AsyncClient",
        return_value=client,
    ):
        index = OpenSearchProductIndex(
            "http://127.0.0.1:9200",
            INDEX_NAMES["amazon"],
            site_locale="jp",
        )
        vector = _vector()
        await index.search(query="travel bag", embedding=vector, top_n=1)

    client.post.assert_awaited_once_with(
        f"/{INDEX_NAMES['amazon']}/_search",
        params={"search_pipeline": RRF_PIPELINE_NAME},
        json=hybrid_query("travel bag", vector, size=1, site_locale="jp"),
    )


def test_site_locale_is_restricted_to_the_amazon_index() -> None:
    with pytest.raises(ValueError, match="only valid for the Amazon"):
        OpenSearchProductIndex(
            "http://127.0.0.1:9200",
            INDEX_NAMES["reference_seed"],
            site_locale="jp",
        )
    with pytest.raises(ValueError, match="unsupported Amazon site locale"):
        OpenSearchProductIndex(
            "http://127.0.0.1:9200",
            INDEX_NAMES["amazon"],
            site_locale="cn",
        )


async def test_adapter_rejects_identity_mismatch() -> None:
    client = AsyncMock()
    client.post.return_value = _response(
        {
            "hits": {
                "hits": [{"_id": "p1", "_score": 1.0, "_source": {"product_id": "p2"}}]
            }
        }
    )
    with patch(
        "app.infrastructure.vector.opensearch_product_index.httpx.AsyncClient",
        return_value=client,
    ):
        index = OpenSearchProductIndex(
            "http://127.0.0.1:9200", INDEX_NAMES["crossshop_reference"]
        )
        with pytest.raises(RuntimeError, match="identity mismatch"):
            await index.search(query="耳机", embedding=_vector(), top_n=1)


async def test_adapter_rejects_duplicate_product_id_hits() -> None:
    client = AsyncMock()
    client.post.return_value = _response(_hits("p1", "p1"))
    with patch(
        "app.infrastructure.vector.opensearch_product_index.httpx.AsyncClient",
        return_value=client,
    ):
        index = OpenSearchProductIndex(
            "http://127.0.0.1:9200", INDEX_NAMES["crossshop_reference"]
        )
        with pytest.raises(RuntimeError, match="duplicate Product hit"):
            await index.search(query="耳机", embedding=_vector(), top_n=2)


async def test_adapter_retries_503_once_with_identical_hybrid_request() -> None:
    failure = httpx.Response(
        503,
        content=b"busy",
        request=httpx.Request("POST", "http://127.0.0.1:9200/_search"),
    )
    client = AsyncMock()
    client.post.side_effect = [failure, _response(_hits("p1"))]
    with (
        patch(
            "app.infrastructure.vector.opensearch_product_index.httpx.AsyncClient",
            return_value=client,
        ),
        patch("app.infrastructure.transient._RETRY_DELAY_SECONDS", 0),
    ):
        index = OpenSearchProductIndex(
            "http://127.0.0.1:9200", INDEX_NAMES["crossshop_reference"]
        )
        hits = await index.search(query="耳机", embedding=_vector(), top_n=1)

    assert hits == [VectorHit("p1", 1.0)]
    assert client.post.await_count == 2
    assert client.post.await_args_list[0] == client.post.await_args_list[1]


async def test_adapter_does_not_retry_http_500() -> None:
    failure = httpx.Response(
        500,
        content=b"boom",
        request=httpx.Request("POST", "http://127.0.0.1:9200/_search"),
    )
    client = AsyncMock()
    client.post.return_value = failure
    with patch(
        "app.infrastructure.vector.opensearch_product_index.httpx.AsyncClient",
        return_value=client,
    ):
        index = OpenSearchProductIndex(
            "http://127.0.0.1:9200", INDEX_NAMES["crossshop_reference"]
        )
        with pytest.raises(httpx.HTTPStatusError):
            await index.search(query="耳机", embedding=_vector(), top_n=1)

    assert client.post.await_count == 1


async def test_adapter_verifies_mapping_and_forbids_online_upsert() -> None:
    index_name = INDEX_NAMES["crossshop_reference"]
    client = AsyncMock()
    client.get.return_value = _response(
        {
            index_name: {
                "mappings": {
                    "properties": {
                        "content_vector": {"type": "knn_vector", "dimension": 1024}
                    }
                }
            }
        }
    )
    with patch(
        "app.infrastructure.vector.opensearch_product_index.httpx.AsyncClient",
        return_value=client,
    ):
        index = OpenSearchProductIndex("http://127.0.0.1:9200", index_name)
        await index.ensure_ready(VECTOR_DIMENSION)
        with pytest.raises(RuntimeError, match="offline-built"):
            await index.upsert_products([], [])
    client.get.assert_awaited_once_with(f"/{index_name}/_mapping")
    client.post.assert_not_awaited()


def test_adapter_accepts_only_the_three_frozen_indexes() -> None:
    with pytest.raises(ValueError, match="unsupported frozen product index"):
        OpenSearchProductIndex("http://127.0.0.1:9200", "crossshop-products-all-v1")


async def test_qdrant_keeps_vector_only_behavior_and_ignores_query() -> None:
    index = QdrantProductIndex.__new__(QdrantProductIndex)
    index._collection = "products"
    index._client = AsyncMock()
    index._client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(payload={"product_id": "p1"}, score=0.75)]
    )

    hits = await index.search(query="ignored BM25 text", embedding=[1.0], top_n=3)

    assert hits == [VectorHit(product_id="p1", score=0.75)]
    index._client.query_points.assert_awaited_once_with(
        collection_name="products",
        query=[1.0],
        limit=3,
        with_payload=True,
    )


class _FixedEmbedding(EmbeddingClient):
    async def embed(self, text: str) -> list[float]:
        del text
        return _vector()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_vector() for _ in texts]


class _RecordingIndex(ProductVectorIndex):
    def __init__(self) -> None:
        self.search_args: dict | None = None

    async def ensure_ready(self, vector_dim: int) -> None:
        del vector_dim

    async def upsert_products(
        self,
        products: list[Product],
        embeddings: list[list[float]],
    ) -> None:
        del products, embeddings

    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        self.search_args = {
            "query": query,
            "embedding": embedding,
            "top_n": top_n,
        }
        return [VectorHit(product_id="P1008", score=0.9)]


async def test_catalog_usecase_passes_normalized_query_to_existing_port() -> None:
    index = _RecordingIndex()
    usecase = CatalogSearchUseCase(
        InMemoryProductRepository(),
        embedder=_FixedEmbedding(),
        vector_index=index,
    )
    await usecase.execute(ProductSearchSpec(normalized_query="降噪 旅行耳机"))
    assert index.search_args == {
        "query": "降噪 旅行耳机",
        "embedding": _vector(),
        "top_n": 8,
    }
