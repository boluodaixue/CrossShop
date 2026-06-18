from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.catalog.opensearch_product_h1 import (
    EMBEDDING_VERSION,
    INDEX_NAMES,
    RRF_PIPELINE_NAME,
    SEARCH_UNIT_FIELDS,
    VECTOR_DIMENSION,
    ann_query,
    bm25_query,
    hybrid_query,
    index_document,
    product_index_body,
    rrf_pipeline_body,
)
from scripts.index.build_product_opensearch import (
    PARTITIONS,
    checkpoint_contract,
    load_checkpoint,
    write_checkpoint,
)


def _unit() -> dict:
    return {
        "attributes_text": "材质 棉",
        "brand": "Globex",
        "category": "服饰",
        "currency": "CNY",
        "description": "轻便",
        "highlights_text": "透气",
        "index_schema_version": "product-search-unit-v2",
        "ingested_at": "",
        "language": "zh",
        "locale": "cn",
        "origin_country": "CN",
        "platform": "taobao",
        "price_minor": 9900,
        "product_id": "p1",
        "rating": 4.8,
        "review_count": 10,
        "same_group_id": "p1",
        "search_unit_id": "p1::s1",
        "searchable_text": "Globex 轻便",
        "ships_to": ["CN"],
        "sku_id": "s1",
        "sku_spec": "白色",
        "source_updated_at": None,
        "stock": 3,
        "title": "轻便上衣",
    }


def _vector() -> list[float]:
    value = 1.0 / math.sqrt(VECTOR_DIMENSION)
    return [value] * VECTOR_DIMENSION


def test_mapping_is_exact_and_uses_faiss_hnsw() -> None:
    body = product_index_body()
    properties = body["mappings"]["properties"]
    assert body["mappings"]["dynamic"] == "strict"
    assert set(properties) == SEARCH_UNIT_FIELDS | {"content_vector", "embedding_version"}
    vector = properties["content_vector"]
    assert vector["dimension"] == 1024
    assert vector["method"] == {
        "name": "hnsw",
        "engine": "faiss",
        "space_type": "innerproduct",
        "parameters": {"ef_construction": 100, "m": 16},
    }
    assert properties["price_minor"]["type"] == "long"
    assert properties["stock"]["type"] == "integer"
    assert properties["ships_to"]["type"] == "keyword"
    assert properties["ingested_at"]["type"] == "date"
    forbidden = {
        "availability",
        "materials",
        "listing_status",
        "compliance_status",
        "item_id",
        "variant_id",
        "price_cny",
    }
    assert not forbidden & set(properties)


def test_document_conversion_only_adds_frozen_derived_fields() -> None:
    source = _unit()
    document = index_document(source, _vector())
    assert source["ingested_at"] == ""
    assert document["ingested_at"] is None
    assert set(document) == SEARCH_UNIT_FIELDS | {"content_vector", "embedding_version"}
    assert document["embedding_version"] == EMBEDDING_VERSION


@pytest.mark.parametrize("mutation", ["extra", "missing", "dimension", "not_normalized"])
def test_document_conversion_rejects_contract_drift(mutation: str) -> None:
    source = _unit()
    vector = _vector()
    if mutation == "extra":
        source["availability"] = True
    elif mutation == "missing":
        del source["brand"]
    elif mutation == "dimension":
        vector.pop()
    else:
        vector = [1.0] * VECTOR_DIMENSION
    with pytest.raises(ValueError):
        index_document(source, vector)


def test_three_frozen_indexes_and_rrf_pipeline() -> None:
    assert INDEX_NAMES == {
        "globex_reference": "globex-products-reference-v1",
        "taobao": "globex-products-taobao-v1",
        "amazon": "globex-products-amazon-v1",
    }
    assert RRF_PIPELINE_NAME == "globex-products-rrf-v1"
    combination = rrf_pipeline_body()["phase_results_processors"][0][
        "score-ranker-processor"
    ]["combination"]
    assert combination == {"technique": "rrf", "rank_constant": 60}


def test_queries_apply_only_stock_before_recall() -> None:
    vector = _vector()
    ann = ann_query(vector)
    bm25 = bm25_query("轻便上衣")
    hybrid = hybrid_query("轻便上衣", vector)
    assert ann["query"]["knn"]["content_vector"]["filter"] == {
        "range": {"stock": {"gt": 0}}
    }
    assert bm25["query"]["bool"]["filter"] == [{"range": {"stock": {"gt": 0}}}]
    hybrid_queries = hybrid["query"]["hybrid"]["queries"]
    assert hybrid_queries[0]["knn"]["content_vector"]["filter"] == {
        "range": {"stock": {"gt": 0}}
    }
    assert hybrid_queries[1]["bool"]["filter"] == [
        {"range": {"stock": {"gt": 0}}}
    ]
    serialized = repr((ann, bm25, hybrid))
    for forbidden in ("ship_to", "price_max_major", "platform", "locale", "material"):
        assert forbidden not in serialized
    assert len(hybrid_queries) == 2


def test_partition_plan_uses_three_indexes_and_one_amazon_index() -> None:
    assert [partition.partition_id for partition in PARTITIONS] == [
        "globex_reference",
        "taobao",
        "amazon_us",
        "amazon_es",
        "amazon_jp",
    ]
    amazon_indexes = {
        partition.index_name
        for partition in PARTITIONS
        if partition.partition_id.startswith("amazon_")
    }
    assert amazon_indexes == {INDEX_NAMES["amazon"]}


def test_checkpoint_is_atomic_resumable_and_rejects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "search_units.jsonl"
    source.write_text('{"search_unit_id":"p::s"}\n', encoding="utf-8")
    partition = PARTITIONS[0]
    contract = checkpoint_contract(partition, source, "D:/models/bge-m3", 512)
    checkpoint = tmp_path / "checkpoint.json"
    initial = load_checkpoint(checkpoint, contract)
    assert initial["next_line"] == 0
    assert initial["completed"] is False

    initial.update(next_line=64, docs_written=64, last_search_unit_id="p::s")
    write_checkpoint(checkpoint, initial)
    assert not checkpoint.with_suffix(".tmp").exists()
    assert load_checkpoint(checkpoint, contract)["docs_written"] == 64

    source.write_text('{"search_unit_id":"changed"}\n', encoding="utf-8")
    changed_contract = checkpoint_contract(partition, source, "D:/models/bge-m3", 512)
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        load_checkpoint(checkpoint, changed_contract)
