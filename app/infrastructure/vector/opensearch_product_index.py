"""Online read-only Product Hybrid retrieval backed by the frozen H1 indexes."""

from __future__ import annotations

import math

import httpx

from app.catalog.opensearch_product_h1 import (
    INDEX_NAMES,
    RRF_PIPELINE_NAME,
    VECTOR_DIMENSION,
    hybrid_query,
)
from app.domain.catalog.ports.retrieval_ports import ProductVectorIndex, VectorHit
from app.domain.catalog.product import Product
from app.infrastructure.tracing import set_span_attributes, text_digest, trace_span
from app.infrastructure.transient import retry_dependency_once


class OpenSearchProductIndex(ProductVectorIndex):
    """Bind one instance to exactly one frozen platform index.

    H1 owns all index writes. This online adapter performs one Hybrid request per
    ``search`` call and never creates or mutates product documents.
    """

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        *,
        timeout_seconds: float = 30.0,
        site_locale: str | None = None,
    ) -> None:
        if index_name not in frozenset(INDEX_NAMES.values()):
            raise ValueError(f"unsupported frozen product index: {index_name}")
        if not endpoint.strip():
            raise ValueError("OpenSearch endpoint required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if site_locale not in {None, "us", "es", "jp"}:
            raise ValueError(f"unsupported Amazon site locale: {site_locale}")
        if site_locale is not None and index_name != INDEX_NAMES["amazon"]:
            raise ValueError("site_locale is only valid for the Amazon Product index")
        self._index_name = index_name
        self._site_locale = site_locale
        self._client = httpx.AsyncClient(
            base_url=endpoint.rstrip("/"),
            timeout=timeout_seconds,
        )

    @property
    def index_name(self) -> str:
        return self._index_name

    @property
    def site_locale(self) -> str | None:
        return self._site_locale

    async def ensure_ready(self, vector_dim: int) -> None:
        if vector_dim != VECTOR_DIMENSION:
            raise ValueError(
                f"OpenSearch Product vector dimension must be {VECTOR_DIMENSION}, got {vector_dim}"
            )

        async def _request_mapping() -> httpx.Response:
            response = await self._client.get(f"/{self._index_name}/_mapping")
            response.raise_for_status()
            return response

        response = await retry_dependency_once(
            _request_mapping,
            dependency="opensearch",
        )
        mapping = response.json()[self._index_name]["mappings"]["properties"]
        actual_dimension = mapping["content_vector"]["dimension"]
        if actual_dimension != VECTOR_DIMENSION:
            raise RuntimeError(
                f"{self._index_name} content_vector dimension is {actual_dimension}"
            )

    async def upsert_products(
        self,
        products: list[Product],
        embeddings: list[list[float]],
    ) -> None:
        del products, embeddings
        raise RuntimeError(
            "OpenSearch Product indexes are offline-built and read-only online"
        )

    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        if not query.strip():
            raise ValueError("query required")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if len(embedding) != VECTOR_DIMENSION:
            raise ValueError(f"query embedding must have {VECTOR_DIMENSION} dimensions")
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError("query embedding contains a non-finite value")

        with trace_span(
            "globex.product.opensearch.hybrid",
            {
                "globex.opensearch.index": self._index_name,
                "globex.opensearch.site_locale": self._site_locale or "all",
                "globex.opensearch.top_n": top_n,
                "globex.opensearch.query_digest": text_digest(query),
                "globex.opensearch.vector_dimension": len(embedding),
                "globex.opensearch.pipeline": RRF_PIPELINE_NAME,
            },
        ) as span:
            request_body = hybrid_query(
                query,
                embedding,
                size=top_n,
                site_locale=self._site_locale,
            )

            async def _request_search() -> httpx.Response:
                response = await self._client.post(
                    f"/{self._index_name}/_search",
                    params={"search_pipeline": RRF_PIPELINE_NAME},
                    json=request_body,
                )
                response.raise_for_status()
                return response

            response = await retry_dependency_once(
                _request_search,
                dependency="opensearch",
                span=span,
            )

            hits: list[VectorHit] = []
            seen_product_ids: set[str] = set()
            for hit in response.json()["hits"]["hits"]:
                product_id = hit.get("_source", {}).get("product_id")
                if not isinstance(product_id, str) or not product_id:
                    raise RuntimeError("OpenSearch Product hit is missing product_id")
                if hit.get("_id") != product_id:
                    raise RuntimeError(
                        "OpenSearch identity mismatch: "
                        f"_id={hit.get('_id')!r}, product_id={product_id!r}"
                    )
                if product_id in seen_product_ids:
                    raise RuntimeError(f"duplicate Product hit: {product_id}")
                score = float(hit["_score"])
                if not math.isfinite(score):
                    raise RuntimeError(f"non-finite score for Product {product_id}")
                seen_product_ids.add(product_id)
                hits.append(VectorHit(product_id=product_id, score=score))
            set_span_attributes(span, {"globex.opensearch.hit_count": len(hits)})
            return hits

    async def close(self) -> None:
        await self._client.aclose()
