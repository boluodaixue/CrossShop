"""Frozen OpenSearch product-index contract for migration stage H1.

This module is deliberately offline-only.  It does not implement a runtime
repository or connect OpenSearch to any AgentScope agent.
"""

from __future__ import annotations

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

SEARCH_UNIT_FIELDS = frozenset(
    {
        "attributes_text",
        "brand",
        "category",
        "currency",
        "description",
        "highlights_text",
        "index_schema_version",
        "ingested_at",
        "language",
        "locale",
        "origin_country",
        "platform",
        "price_minor",
        "product_id",
        "rating",
        "review_count",
        "same_group_id",
        "search_unit_id",
        "searchable_text",
        "ships_to",
        "sku_id",
        "sku_spec",
        "source_updated_at",
        "stock",
        "title",
    }
)

TEXT_FIELDS = (
    "title^4",
    "brand^3",
    "category^3",
    "sku_spec^2",
    "description",
    "highlights_text^2",
    "attributes_text^2",
    "searchable_text",
)


def product_index_body() -> dict[str, Any]:
    keyword = {
        name: {"type": "keyword"}
        for name in (
            "product_id",
            "sku_id",
            "search_unit_id",
            "same_group_id",
            "platform",
            "locale",
            "language",
            "origin_country",
            "currency",
            "ships_to",
            "index_schema_version",
            "embedding_version",
        )
    }
    text = {
        name: {"type": "text"}
        for name in (
            "title",
            "description",
            "highlights_text",
            "attributes_text",
            "searchable_text",
        )
    }
    text_with_raw = {
        name: {"type": "text", "fields": {"raw": {"type": "keyword"}}}
        for name in ("brand", "category", "sku_spec")
    }
    properties: dict[str, Any] = {
        **keyword,
        **text,
        **text_with_raw,
        "price_minor": {"type": "long"},
        "stock": {"type": "integer"},
        "review_count": {"type": "integer"},
        "rating": {"type": "float"},
        "ingested_at": {"type": "date"},
        "source_updated_at": {"type": "date"},
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
        "description": "Globex product ANN and BM25 reciprocal rank fusion",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": 60,
                    }
                }
            }
        ],
    }


def index_document(search_unit: dict[str, Any], vector: list[float]) -> dict[str, Any]:
    actual = frozenset(search_unit)
    if actual != SEARCH_UNIT_FIELDS:
        missing = sorted(SEARCH_UNIT_FIELDS - actual)
        extra = sorted(actual - SEARCH_UNIT_FIELDS)
        raise ValueError(f"SearchUnit field mismatch: missing={missing}, extra={extra}")
    if len(vector) != VECTOR_DIMENSION:
        raise ValueError(f"content_vector must have {VECTOR_DIMENSION} dimensions")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("content_vector contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"content_vector must be unit-normalized; norm={norm:.6f}")

    document = dict(search_unit)
    for date_field in ("ingested_at", "source_updated_at"):
        if document[date_field] == "":
            document[date_field] = None
    document["content_vector"] = vector
    document["embedding_version"] = EMBEDDING_VERSION
    return document


def stock_filter() -> dict[str, Any]:
    return {"range": {"stock": {"gt": 0}}}


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
                "must": [{"multi_match": {"query": query, "fields": list(TEXT_FIELDS)}}],
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
                ],
            }
        },
    }
