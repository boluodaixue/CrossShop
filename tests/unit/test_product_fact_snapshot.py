from __future__ import annotations

import json
from pathlib import Path

from globex_agent.application.evidence import (
    build_product_fact_snapshot,
    product_card_from_snapshot,
    sanitize_snapshot_payload,
)
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.catalog.models import ProductAttribute


def _items():
    path = Path(__file__).parents[2] / "data" / "demo" / "products.jsonl"
    return list(LocalCatalog.from_jsonl(path, strict=True).catalog.items)


def test_snapshot_hash_and_audit_identity_are_stable() -> None:
    item = _items()[0]
    first = build_product_fact_snapshot(item)
    second = build_product_fact_snapshot(item)
    assert first.content_hash == second.content_hash
    assert first.evidence_id == second.evidence_id
    assert first.schema_version == "product-fact-snapshot-v1"


def test_snapshot_and_card_keep_all_variants_and_highlights() -> None:
    item = _items()[0].model_copy(
        update={
            "attributes": [
                ProductAttribute(code=f"attr-{index}", name=f"属性{index}", value=f"值{index}")
                for index in range(10)
            ]
        }
    )
    snapshot = build_product_fact_snapshot(item)
    card = product_card_from_snapshot(
        snapshot,
        score=0.5,
        landed_price=None,
        variant_landed_prices=None,
        price_min_major=None,
        price_max_major=None,
        requires_variant_selection=False,
        matching_variant_ids=[],
        warnings=[],
    )
    persisted = sanitize_snapshot_payload(snapshot)
    assert len(snapshot.highlights) == 10
    assert card.highlights == snapshot.highlights
    assert persisted["highlights"] == snapshot.highlights
    assert persisted["exposed_facts"]["variants"] == snapshot.exposed_facts["variants"]
    assert "evidence_catalog" not in json.dumps(persisted, ensure_ascii=False)


def test_snapshot_and_card_hide_internal_trace_attributes_but_keep_source_item() -> None:
    item = _items()[0].model_copy(
        update={
            "attributes": [
                ProductAttribute(code="shop_name", name="店铺", value="好店"),
                ProductAttribute(
                    code="source_attributes", name="来源属性", value="内部原始属性"
                ),
                ProductAttribute(code="material", name="材质", value="铝合金"),
                ProductAttribute(code="source_tag", name="来源标签", value="train"),
            ]
        }
    )
    snapshot = build_product_fact_snapshot(item)
    card = product_card_from_snapshot(
        snapshot,
        score=0.5,
        landed_price=None,
        variant_landed_prices=None,
        price_min_major=None,
        price_max_major=None,
        requires_variant_selection=False,
        matching_variant_ids=[],
        warnings=[],
    )

    assert [attribute.code for attribute in item.attributes] == [
        "shop_name",
        "source_attributes",
        "material",
        "source_tag",
    ]
    assert snapshot.highlights == ["店铺: 好店", "材质: 铝合金"]
    assert card.highlights == snapshot.highlights
    assert card.to_dict()["highlights"] == snapshot.highlights
    persisted = sanitize_snapshot_payload(snapshot)
    assert persisted["highlights"] == snapshot.highlights
    payload_text = json.dumps(card.to_dict(), ensure_ascii=False)
    assert "source_attributes" not in payload_text
    assert "source_tag" not in payload_text


def test_product_card_has_no_field_level_evidence_catalog() -> None:
    snapshot = build_product_fact_snapshot(_items()[0])
    card = product_card_from_snapshot(
        snapshot,
        score=0.5,
        landed_price=None,
        price_min_major=None,
        price_max_major=None,
        requires_variant_selection=False,
        matching_variant_ids=[],
        warnings=[],
    )
    payload = card.to_dict()
    assert "evidence_refs" not in payload
    assert "variant_evidence_refs" not in payload


def test_snapshot_pii_is_redacted_without_truncating_sku_fields() -> None:
    payload = sanitize_snapshot_payload(
        {
            "evidence_id": "e1",
            "schema_version": "product-fact-snapshot-v1",
            "content_hash": "h",
            "item_id": "taobao:1",
            "variants": [
                {
                    "variant_id": "sku-完整-001",
                    "options": [{"name": "颜色", "value": "黑色"}],
                }
            ],
            "highlights": ["联系人 13800000000"],
            "provenance": {"source": "demo"},
        }
    )
    assert payload["variants"][0]["variant_id"] == "sku-完整-001"
    assert payload["highlights"] == ["联系人 [redacted]"]
