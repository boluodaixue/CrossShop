"""Frozen offline Product-level OpenSearch contract for migration stage H1."""

from __future__ import annotations

import copy
import math
from typing import Any

VECTOR_DIMENSION = 1024
EMBEDDING_VERSION = "bge-m3-cls-v1"
RRF_PIPELINE_NAME = "globex-products-rrf-v1"

INDEX_NAMES = {
    "globex_reference": "globex-products-reference-v1",
    "taobao": "globex-products-taobao-v1",
    "amazon": "globex-products-amazon-v1",
}

PRODUCT_FIELDS = frozenset(
    {
        "product_id",
        "title",
        "brand",
        "category",
        "origin_country",
        "description",
        "highlights",
        "ships_to",
        "skus",
    }
)
HIGHLIGHT_FIELDS = frozenset({"label", "detail"})
SKU_FIELDS = frozenset({"sku_id", "spec", "price", "stock"})
MONEY_FIELDS = frozenset({"amount_in_minor_units", "currency"})
INDEX_FIELDS = PRODUCT_FIELDS | {"locale", "content_vector", "embedding_version"}

# SKU variants are purchase choices, not recall units.
TEXT_FIELDS = (
    "title^4",
    "brand^3",
    "category^3",
    "origin_country",
    "description",
    "highlights.label^2",
    "highlights.detail^2",
)


def product_index_body() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "product_id": {"type": "keyword"},
        "title": {"type": "text"},
        "brand": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
        "category": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
        "origin_country": {"type": "keyword"},
        "description": {"type": "text"},
        "highlights": {
            "properties": {
                "label": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "detail": {"type": "text"},
            },
        },
        "ships_to": {"type": "keyword"},
        "skus": {
            "type": "nested",
            "properties": {
                "sku_id": {"type": "keyword"},
                "spec": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "price": {
                    "properties": {
                        "amount_in_minor_units": {"type": "long"},
                        "currency": {"type": "keyword"},
                    },
                },
                "stock": {"type": "integer"},
            },
        },
        "locale": {"type": "keyword"},
        "embedding_version": {"type": "keyword"},
        "content_vector": {
            "type": "knn_vector",
            "dimension": VECTOR_DIMENSION,
            "method": {
                "name": "hnsw",
                "engine": "faiss",
                "space_type": "innerproduct",
                "parameters": {"ef_construction": 100, "m": 16},
            },
        },
    }
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "-1",
            }
        },
        "mappings": {"dynamic": "strict", "properties": properties},
    }


def rrf_pipeline_body() -> dict[str, Any]:
    return {
        "description": "Globex Product ANN and BM25 reciprocal rank fusion",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {"technique": "rrf", "rank_constant": 60}
                }
            }
        ],
    }


def _require_exact_fields(
    value: dict[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} field mismatch: missing={missing}, extra={extra}")


def validate_product(product: dict[str, Any]) -> None:
    _require_exact_fields(product, PRODUCT_FIELDS, "Product")
    if not product["product_id"]:
        raise ValueError("Product.product_id required")
    if not product["skus"]:
        raise ValueError("Product.skus must not be empty")
    for highlight in product["highlights"]:
        _require_exact_fields(highlight, HIGHLIGHT_FIELDS, "ProductHighlight")
    seen_skus: set[str] = set()
    for sku in product["skus"]:
        _require_exact_fields(sku, SKU_FIELDS, "Sku")
        _require_exact_fields(sku["price"], MONEY_FIELDS, "Money")
        if not sku["sku_id"] or sku["sku_id"] in seen_skus:
            raise ValueError(f"invalid or duplicate sku_id in {product['product_id']}")
        if sku["stock"] < 0 or sku["price"]["amount_in_minor_units"] < 0:
            raise ValueError(f"negative price/stock in {sku['sku_id']}")
        seen_skus.add(sku["sku_id"])


def product_searchable_text(product: dict[str, Any]) -> str:
    """Exactly mirror ``Product.searchable_text`` from the V2 domain model."""
    validate_product(product)
    highlight_text = " ".join(
        f"{highlight['label']} {highlight['detail']}"
        for highlight in product["highlights"]
    )
    return " ".join(
        [
            product["title"],
            product["brand"],
            product["category"],
            product["origin_country"],
            product["description"],
            highlight_text,
        ]
    )


def index_document(
    product: dict[str, Any], locale: str | None, vector: list[float]
) -> dict[str, Any]:
    validate_product(product)
    if locale == "":
        raise ValueError("locale must be a non-empty string or None")
    if len(vector) != VECTOR_DIMENSION:
        raise ValueError(f"content_vector must have {VECTOR_DIMENSION} dimensions")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("content_vector contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"content_vector must be unit-normalized; norm={norm:.6f}")
    document = copy.deepcopy(product)
    document["locale"] = locale
    document["content_vector"] = vector
    document["embedding_version"] = EMBEDDING_VERSION
    return document


def stock_filter() -> dict[str, Any]:
    return {
        "nested": {
            "path": "skus",
            "query": {"range": {"skus.stock": {"gt": 0}}},
            "score_mode": "none",
        }
    }


def ann_query(vector: list[float], *, size: int = 10) -> dict[str, Any]:
    return {
        "size": size,
        "_source": {"excludes": ["content_vector"]},
        "query": {
            "knn": {
                "content_vector": {
                    "vector": vector,
                    "k": size,
                    "filter": stock_filter(),
                }
            }
        },
    }


def bm25_query(query: str, *, size: int = 10) -> dict[str, Any]:
    return {
        "size": size,
        "_source": {"excludes": ["content_vector"]},
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": query, "fields": list(TEXT_FIELDS)}}
                ],
                "filter": [stock_filter()],
            }
        },
    }


def hybrid_query(query: str, vector: list[float], *, size: int = 10) -> dict[str, Any]:
    return {
        "size": size,
        "_source": {"excludes": ["content_vector"]},
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "knn": {
                            "content_vector": {
                                "vector": vector,
                                "k": size,
                                "filter": stock_filter(),
                            }
                        }
                    },
                    {
                        "bool": {
                            "must": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": list(TEXT_FIELDS),
                                    }
                                }
                            ],
                            "filter": [stock_filter()],
                        }
                    },
                ]
            }
        },
    }
