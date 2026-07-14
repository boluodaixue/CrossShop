from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.catalog.opensearch_product_h1 import (
    EMBEDDING_VERSION,
    INDEX_FIELDS,
    INDEX_NAMES,
    PRODUCT_FIELDS,
    RRF_PIPELINE_NAME,
    TEXT_FIELDS,
    VECTOR_DIMENSION,
    ann_query,
    bm25_query,
    hybrid_query,
    index_document,
    product_index_body,
    product_searchable_text,
    rrf_pipeline_body,
    stock_filter,
)
from app.domain.catalog.money import Money
from app.domain.catalog.product import Product, ProductHighlight
from app.domain.catalog.sku import Sku
from scripts.index.build_product_opensearch import (
    PARTITIONS,
    checkpoint_contract,
    load_checkpoint,
    write_checkpoint,
)


def _product() -> dict:
    return {
        "brand": "CrossShop",
        "category": "服饰",
        "description": "轻便",
        "highlights": [{"detail": "透气", "label": "面料"}],
        "origin_country": "CN",
        "product_id": "p1",
        "ships_to": ["CN"],
        "skus": [
            {
                "price": {"amount_in_minor_units": 9900, "currency": "CNY"},
                "sku_id": "s1",
                "spec": "白色",
                "stock": 3,
            },
            {
                "price": {"amount_in_minor_units": 10900, "currency": "CNY"},
                "sku_id": "s2",
                "spec": "黑色",
                "stock": 0,
            },
        ],
        "title": "轻便上衣",
    }


def _vector() -> list[float]:
    value = 1.0 / math.sqrt(VECTOR_DIMENSION)
    return [value] * VECTOR_DIMENSION


def test_mapping_is_exact_product_contract_and_uses_faiss_hnsw() -> None:
    body = product_index_body()
    properties = body["mappings"]["properties"]
    assert body["mappings"]["dynamic"] == "strict"
    assert set(properties) == INDEX_FIELDS
    assert set(properties) == PRODUCT_FIELDS | {
        "locale",
        "content_vector",
        "embedding_version",
    }
    vector = properties["content_vector"]
    assert vector["dimension"] == 1024
    assert vector["method"] == {
        "name": "hnsw",
        "engine": "faiss",
        "space_type": "innerproduct",
        "parameters": {"ef_construction": 100, "m": 16},
    }
    assert properties["skus"]["type"] == "nested"
    sku_properties = properties["skus"]["properties"]
    assert sku_properties["stock"]["type"] == "integer"
    assert (
        sku_properties["price"]["properties"]["amount_in_minor_units"]["type"] == "long"
    )
    assert sku_properties["price"]["properties"]["currency"]["type"] == "keyword"
    forbidden = {
        "availability",
        "materials",
        "listing_status",
        "compliance_status",
        "item_id",
        "variant_id",
        "price_cny",
        "search_unit_id",
        "platform",
    }
    assert not forbidden & set(properties)


def test_document_conversion_only_adds_frozen_index_fields() -> None:
    source = _product()
    document = index_document(source, "cn", _vector())
    assert source == _product()
    assert set(document) == INDEX_FIELDS
    assert document["product_id"] == "p1"
    assert document["skus"] == source["skus"]
    assert document["locale"] == "cn"
    assert document["embedding_version"] == EMBEDDING_VERSION


def test_reference_locale_is_null_because_source_has_no_locale_fact() -> None:
    document = index_document(_product(), None, _vector())
    assert document["locale"] is None
    reference = next(
        partition
        for partition in PARTITIONS
        if partition.partition_id == "crossshop_reference"
    )
    assert reference.locale is None


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "nested_extra",
        "duplicate_sku",
        "dimension",
        "not_normalized",
    ],
)
def test_document_conversion_rejects_contract_drift(mutation: str) -> None:
    source = _product()
    vector = _vector()
    if mutation == "extra":
        source["availability"] = True
    elif mutation == "missing":
        del source["brand"]
    elif mutation == "nested_extra":
        source["skus"][0]["price"]["major"] = 99
    elif mutation == "duplicate_sku":
        source["skus"][1]["sku_id"] = "s1"
    elif mutation == "dimension":
        vector.pop()
    else:
        vector = [1.0] * VECTOR_DIMENSION
    with pytest.raises(ValueError):
        index_document(source, "cn", vector)


def test_item_text_exactly_matches_v2_product_searchable_text() -> None:
    source = _product()
    domain_product = Product(
        product_id=source["product_id"],
        title=source["title"],
        brand=source["brand"],
        category=source["category"],
        origin_country=source["origin_country"],
        description=source["description"],
        highlights=[ProductHighlight(**value) for value in source["highlights"]],
        ships_to=source["ships_to"],
        skus=[
            Sku(
                sku_id=value["sku_id"],
                spec=value["spec"],
                price=Money(**value["price"]),
                stock=value["stock"],
            )
            for value in source["skus"]
        ],
    )
    assert product_searchable_text(source) == domain_product.searchable_text()
    assert "白色" not in product_searchable_text(source)
    assert all("skus.spec" not in field for field in TEXT_FIELDS)


def test_three_frozen_indexes_and_rrf_pipeline() -> None:
    assert INDEX_NAMES == {
        "crossshop_reference": "crossshop-products-reference-v1",
        "reference_seed": "crossshop-products-reference_seed-v1",
        "amazon": "crossshop-products-amazon-v1",
    }
    assert RRF_PIPELINE_NAME == "crossshop-products-rrf-v1"
    combination = rrf_pipeline_body()["phase_results_processors"][0][
        "score-ranker-processor"
    ]["combination"]
    assert combination == {"technique": "rrf", "rank_constant": 60}


def test_queries_apply_only_any_nested_sku_in_stock_before_recall() -> None:
    expected = {
        "nested": {
            "path": "skus",
            "query": {"range": {"skus.stock": {"gt": 0}}},
            "score_mode": "none",
        }
    }
    vector = _vector()
    ann = ann_query(vector)
    bm25 = bm25_query("轻便上衣")
    hybrid = hybrid_query("轻便上衣", vector)
    assert ann["query"]["knn"]["content_vector"]["filter"] == expected
    assert bm25["query"]["bool"]["filter"] == [expected]
    hybrid_queries = hybrid["query"]["hybrid"]["queries"]
    assert hybrid_queries[0]["knn"]["content_vector"]["filter"] == expected
    assert hybrid_queries[1]["bool"]["filter"] == [expected]
    serialized = repr((ann, bm25, hybrid))
    for forbidden in ("ship_to", "price_max_major", "platform", "locale", "material"):
        assert forbidden not in serialized
    assert len(hybrid_queries) == 2


def test_explicit_amazon_site_filter_is_applied_to_both_hybrid_branches() -> None:
    hybrid = hybrid_query("travel bag", _vector(), site_locale="jp")
    queries = hybrid["query"]["hybrid"]["queries"]
    expected_locale_filter = {"term": {"locale": "jp"}}
    ann_filter = queries[0]["knn"]["content_vector"]["filter"]
    bm25_filters = queries[1]["bool"]["filter"]

    assert ann_filter["bool"]["filter"][1] == expected_locale_filter
    assert bm25_filters[1] == expected_locale_filter
    assert ann_filter["bool"]["filter"][0] == stock_filter()
    assert bm25_filters[0] == stock_filter()


def test_hybrid_query_rejects_unknown_amazon_site_locale() -> None:
    with pytest.raises(ValueError, match="unsupported Amazon site locale"):
        hybrid_query("travel bag", _vector(), site_locale="cn")


def test_partition_plan_uses_product_sources_and_one_amazon_index() -> None:
    assert [partition.partition_id for partition in PARTITIONS] == [
        "crossshop_reference",
        "reference_seed",
        "amazon_us",
        "amazon_es",
        "amazon_jp",
    ]
    assert all(
        partition.relative_path.endswith("products.jsonl") for partition in PARTITIONS
    )
    assert {
        partition.locale
        for partition in PARTITIONS
        if partition.partition_id.startswith("amazon_")
    } == {"us", "es", "jp"}
    assert {
        partition.index_name
        for partition in PARTITIONS
        if partition.partition_id.startswith("amazon_")
    } == {INDEX_NAMES["amazon"]}


def test_checkpoint_is_atomic_resumable_and_rejects_source_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "products.jsonl"
    source.write_text('{"product_id":"p1"}\n', encoding="utf-8")
    partition = PARTITIONS[0]
    contract = checkpoint_contract(partition, source, "D:/models/bge-m3", 512)
    checkpoint = tmp_path / "checkpoint.json"
    initial = load_checkpoint(checkpoint, contract)
    assert initial["next_line"] == 0
    assert initial["completed"] is False

    initial.update(next_line=64, docs_written=64, last_product_id="p1")
    write_checkpoint(checkpoint, initial)
    assert not checkpoint.with_suffix(".tmp").exists()
    assert load_checkpoint(checkpoint, contract)["docs_written"] == 64

    source.write_text('{"product_id":"changed"}\n', encoding="utf-8")
    changed_contract = checkpoint_contract(partition, source, "D:/models/bge-m3", 512)
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        load_checkpoint(checkpoint, changed_contract)
