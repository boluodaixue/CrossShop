"""Verify H3 by manually composing the full catalog, H2 adapter, and UseCase.

The script performs exactly one Product Hybrid search for each frozen platform
index.  It reuses the real 1024-dimensional BGE-M3 query vectors saved by H2,
so the run is offline-reproducible and does not need to load the model again.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.catalog.opensearch_product_h1 import (
    INDEX_NAMES,
    RRF_PIPELINE_NAME,
    VECTOR_DIMENSION,
    hybrid_query,
)
from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.persistence.jsonl_product_repository import (
    JsonlProductRepository,
)
from app.infrastructure.vector.opensearch_product_index import (
    OpenSearchProductIndex,
)

PLATFORM_SPECS = {
    "crossshop_reference": ProductSearchSpec(
        normalized_query="降噪耳机",
        ship_to="CN",
        locale="zh-CN",
        top_k=5,
        target_currency="USD",
        price_max_major=30.0,
    ),
    "reference_seed": ProductSearchSpec(
        normalized_query="轻便旅行背包",
        ship_to="CN",
        locale="zh-CN",
        top_k=5,
        target_currency="USD",
        price_max_major=12.0,
    ),
    "amazon": ProductSearchSpec(
        normalized_query="noise cancelling headphones",
        ship_to="CN",
        locale="en-US",
        top_k=5,
        target_currency="USD",
        price_max_major=35.0,
    ),
}


class RecordedQueryEmbedder:
    """Return one recorded H2 vector only for its exact original query."""

    def __init__(self, query: str, vector: list[float]) -> None:
        self._query = query
        self._vector = vector
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if text != self._query:
            raise ValueError(
                "recorded query vector mismatch: "
                f"expected={self._query!r}, got={text!r}"
            )
        return list(self._vector)


class DeterministicReranker:
    """Keep the eligible RRF order while proving only eligible docs are reranked."""

    def __init__(self) -> None:
        self.calls = 0
        self.document_counts: list[int] = []

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not query.strip():
            raise ValueError("rerank query required")
        self.calls += 1
        self.document_counts.append(len(documents))
        return [float(len(documents) - index) for index in range(len(documents))]


class CountingOpenSearchProductIndex(OpenSearchProductIndex):
    """Test-only counter around the production H2 adapter."""

    def __init__(self, endpoint: str, index_name: str) -> None:
        super().__init__(endpoint, index_name)
        self.search_calls = 0
        self.last_search: dict[str, Any] | None = None

    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ):
        self.search_calls += 1
        self.last_search = {
            "query": query,
            "embedding_dimension": len(embedding),
            "top_n": top_n,
        }
        return await super().search(query=query, embedding=embedding, top_n=top_n)


def _load_recorded_vector(path: Path, expected_query: str) -> list[float]:
    request = json.loads(path.read_text(encoding="utf-8"))
    queries = request["body"]["query"]["hybrid"]["queries"]
    vector = queries[0]["knn"]["content_vector"]["vector"]
    recorded_query = queries[1]["bool"]["must"][0]["multi_match"]["query"]
    if recorded_query != expected_query:
        raise ValueError(
            f"H2 request query mismatch in {path}: "
            f"expected={expected_query!r}, actual={recorded_query!r}"
        )
    if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSION:
        actual = len(vector) if isinstance(vector, list) else type(vector).__name__
        raise ValueError(
            f"H2 request vector dimension must be {VECTOR_DIMENSION}, actual={actual}"
        )
    return [float(value) for value in vector]


def _spec_dict(spec: ProductSearchSpec) -> dict[str, Any]:
    return {
        "normalized_query": spec.normalized_query,
        "category": spec.category,
        "ship_to": spec.ship_to,
        "locale": spec.locale,
        "top_k": spec.top_k,
        "target_currency": spec.target_currency,
        "price_max_major": spec.price_max_major,
    }


async def _assert_result_semantics(
    repository: JsonlProductRepository,
    spec: ProductSearchSpec,
    result: dict[str, Any],
) -> dict[str, Any]:
    hits = result["hits"]
    if len(hits) > spec.top_k:
        raise RuntimeError(f"hit count exceeds top_k: {len(hits)} > {spec.top_k}")
    product_ids = [hit["product_id"] for hit in hits]
    if len(product_ids) != len(set(product_ids)):
        raise RuntimeError(f"duplicate Product cards: {product_ids}")

    products = await repository.find_by_ids(product_ids)
    restored_ids = [product.product_id for product in products]
    if restored_ids != product_ids:
        raise RuntimeError(
            f"Product cards were not exactly restored: "
            f"expected={product_ids}, actual={restored_ids}"
        )

    rates = ExchangeRateTable()
    for card, product in zip(hits, products, strict=True):
        if spec.ship_to and spec.ship_to not in product.ships_to:
            raise RuntimeError(f"non-deliverable Product card: {product.product_id}")

        expected = []
        for sku in product.skus:
            if sku.stock <= 0:
                continue
            converted = rates.convert(sku.price, spec.target_currency)
            if (
                spec.price_max_major is not None
                and converted.to_major_units() > spec.price_max_major
            ):
                continue
            expected.append((sku, converted))
        expected.sort(key=lambda pair: (pair[1].amount_in_minor_units, pair[0].sku_id))
        if not expected:
            raise RuntimeError(
                f"Product card has no eligible SKU: {product.product_id}"
            )

        expected_ids = [sku.sku_id for sku, _ in expected]
        actual_ids = [sku["sku_id"] for sku in card["skus"]]
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"eligible SKU projection mismatch for {product.product_id}: "
                f"expected={expected_ids}, actual={actual_ids}"
            )
        for actual, (sku, converted) in zip(card["skus"], expected, strict=True):
            if actual["stock"] != sku.stock or actual["stock"] <= 0:
                raise RuntimeError(f"invalid projected stock for {sku.sku_id}")
            if actual["currency"] != spec.target_currency:
                raise RuntimeError(f"invalid projected currency for {sku.sku_id}")
            if actual["price_major"] != converted.to_major_units():
                raise RuntimeError(f"invalid projected price for {sku.sku_id}")
            if (
                spec.price_max_major is not None
                and actual["price_major"] > spec.price_max_major
            ):
                raise RuntimeError(f"over-budget SKU leaked into card: {sku.sku_id}")

        if card["price_major"] != expected[0][1].to_major_units():
            raise RuntimeError(f"starting price mismatch for {product.product_id}")
        if card["currency"] != spec.target_currency:
            raise RuntimeError(f"starting currency mismatch for {product.product_id}")

    filtered_out = result.get("filtered_out", [])
    allowed_reasons = {"ship_to_unavailable", "over_price_cap"}
    if any(item["reason"] not in allowed_reasons for item in filtered_out):
        raise RuntimeError(f"unexpected filtered_out reason: {filtered_out}")
    if any(item["currency"] != spec.target_currency for item in filtered_out):
        raise RuntimeError("filtered_out currency does not match target_currency")

    return {
        "hit_count": len(hits),
        "product_ids": product_ids,
        "projected_sku_count": sum(len(hit["skus"]) for hit in hits),
        "filtered_out_count": len(filtered_out),
        "filtered_out_reasons": [item["reason"] for item in filtered_out],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        base_url=args.endpoint.rstrip("/"), timeout=args.timeout_seconds
    ) as client:
        response = await client.get("/")
        response.raise_for_status()
        opensearch_version = response.json()["version"]["number"]
    if opensearch_version != "2.19.1":
        raise RuntimeError(f"OpenSearch 2.19.1 required, got {opensearch_version}")

    load_started = time.perf_counter()
    repository = JsonlProductRepository(args.catalog_root)
    repository_load_seconds = time.perf_counter() - load_started
    if repository.product_count != 45_286 or repository.sku_count != 261_369:
        raise RuntimeError(
            "unexpected full catalog counts: "
            f"products={repository.product_count}, skus={repository.sku_count}"
        )

    report: dict[str, Any] = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "opensearch_version": opensearch_version,
        "pipeline": RRF_PIPELINE_NAME,
        "repository": {
            "catalog_root": str(args.catalog_root.resolve()),
            "load_count": 1,
            "load_seconds": repository_load_seconds,
            "product_count": repository.product_count,
            "sku_count": repository.sku_count,
        },
        "platforms": {},
    }

    for platform, spec in PLATFORM_SPECS.items():
        vector_path = (
            args.h2_artifact_root / f"{platform}-hybrid-request.json"
        ).resolve()
        vector = _load_recorded_vector(vector_path, spec.normalized_query)
        vector_sha256 = hashlib.sha256(vector_path.read_bytes()).hexdigest()
        embedder = RecordedQueryEmbedder(spec.normalized_query, vector)
        reranker = DeterministicReranker()
        index = CountingOpenSearchProductIndex(
            args.endpoint,
            INDEX_NAMES[platform],
        )
        try:
            result = await CatalogSearchUseCase(
                repository,
                embedder=embedder,
                vector_index=index,
                reranker=reranker,
            ).execute(spec)
        finally:
            await index.close()

        if index.search_calls != 1 or index.last_search is None:
            raise RuntimeError(
                f"{platform} must execute exactly one Hybrid search, "
                f"actual={index.search_calls}"
            )
        if embedder.calls != 1:
            raise RuntimeError(f"{platform} embedder calls={embedder.calls}")
        if result["total_candidates"] <= 0:
            raise RuntimeError(f"{platform} produced no eligible candidates")
        if reranker.calls != 1 or reranker.document_counts != [
            result["total_candidates"]
        ]:
            raise RuntimeError(
                f"{platform} reranker did not receive exactly the eligible set"
            )
        if result["recall_strategy"] != "embedding_rerank":
            raise RuntimeError(
                f"{platform} unexpected strategy: {result['recall_strategy']}"
            )
        if result["rerank_applied"] is not True:
            raise RuntimeError(f"{platform} reranker was not applied")

        semantics = await _assert_result_semantics(repository, spec, result)
        raw_request = {
            "method": "POST",
            "path": (
                f"/{INDEX_NAMES[platform]}/_search?search_pipeline={RRF_PIPELINE_NAME}"
            ),
            "body": hybrid_query(
                spec.normalized_query,
                vector,
                size=index.last_search["top_n"],
            ),
        }
        request_path = output_root / f"{platform}-h3-hybrid-request.json"
        result_path = output_root / f"{platform}-h3-usecase-result.json"
        request_path.write_text(
            json.dumps(raw_request, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["platforms"][platform] = {
            "index": INDEX_NAMES[platform],
            "spec": _spec_dict(spec),
            "hybrid_request_count": index.search_calls,
            "embedder_call_count": embedder.calls,
            "query_vector_dimension": len(vector),
            "query_vector_source": str(vector_path.relative_to(PROJECT_ROOT)),
            "query_vector_source_sha256": vector_sha256,
            "reranker_call_count": reranker.calls,
            "reranked_document_count": reranker.document_counts[0],
            "total_candidates": result["total_candidates"],
            "request_artifact": str(request_path.relative_to(PROJECT_ROOT)),
            "result_artifact": str(result_path.relative_to(PROJECT_ROOT)),
            **semantics,
        }

    report_path = output_root / "integration-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("GLOBEX_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"),
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "catalogs-v2",
    )
    parser.add_argument(
        "--h2-artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "h2-opensearch",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "h3-catalog-search",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    report = asyncio.run(run(build_parser().parse_args()))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
