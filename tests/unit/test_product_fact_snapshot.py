from __future__ import annotations

import json
from pathlib import Path

from globex_agent.application.evidence import (
    build_product_fact_snapshot,
    sanitize_snapshot_payload,
)
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.catalog.models import (
    AvailabilityStatus,
    PriceSource,
    StandardItemVariant,
    VariantOption,
)
from globex_agent.eval.fact_guard import validate_final_response


def _items():
    path = Path(__file__).parents[2] / "data" / "demo" / "products.jsonl"
    return list(LocalCatalog.from_jsonl(path, strict=True).catalog.items)


def test_snapshot_hash_and_evidence_id_are_stable() -> None:
    item = _items()[0]
    first = build_product_fact_snapshot(item)
    second = build_product_fact_snapshot(item)

    assert first.content_hash == second.content_hash
    assert first.evidence_id == second.evidence_id
    assert first.schema_version == "product-fact-snapshot-v1"


def test_snapshot_contains_exposed_variant_facts_and_bounded_audit_payload() -> None:
    item = _items()[0].model_copy(
        update={
            "price_cny": None,
            "price_source": PriceSource.UNAVAILABLE,
            "variants": [
                StandardItemVariant(
                    variant_id="amazon:variant:black",
                    options=[VariantOption(name="颜色", value="黑色")],
                    price_cny=329,
                    price_source=PriceSource.OBSERVED,
                    availability=AvailabilityStatus.AVAILABLE,
                )
            ],
        }
    )
    snapshot = build_product_fact_snapshot(item)
    persisted = sanitize_snapshot_payload(snapshot)
    encoded = json.dumps(persisted, ensure_ascii=False)

    assert snapshot.variants
    assert "variants[].price_major" in snapshot.exposed_fields
    assert snapshot.exposed_facts["variants"]
    assert "[omitted" not in encoded
    assert "raw_evidence" not in encoded


def test_variant_price_requires_matching_exposed_option() -> None:
    facts = [
        {
            "item_id": "taobao:1",
            "price_major": None,
            "price_min_major": 329,
            "price_max_major": 399,
            "variants": [
                {
                    "options": [{"name": "颜色", "value": "黑色"}],
                    "display_name": "颜色: 黑色",
                    "price_major": 329,
                },
                {
                    "options": [{"name": "颜色", "value": "白色"}],
                    "display_name": "颜色: 白色",
                    "price_major": 399,
                },
            ],
            "availability": "available",
        }
    ]
    assert validate_final_response("黑色款价格329元。", product_facts=facts) == ()
    assert validate_final_response("白色款价格329元。", product_facts=facts)
    assert validate_final_response("价格是329元。", product_facts=facts)


def test_snapshot_store_and_inventory_are_explicitly_guardable() -> None:
    facts = [
        {
            "item_id": "taobao:1",
            "store": "示例店",
            "availability": "available",
            "price_major": 199,
        }
    ]
    assert validate_final_response("示例店现货，价格是199元。", product_facts=facts) == ()
    assert validate_final_response("另一家店现货，价格是199元。", product_facts=facts)


def test_snapshot_pii_is_redacted_without_depth_truncation() -> None:
    payload = sanitize_snapshot_payload(
        {
            "evidence_id": "e1",
            "schema_version": "product-fact-snapshot-v1",
            "content_hash": "h",
            "item_id": "taobao:1",
            "variants": [{"options": [{"name": "颜色", "value": "黑色"}]}],
            "highlights": ["联系人 13800000000"],
            "provenance": {"source": "demo"},
        }
    )
    assert payload["highlights"] == ["联系人 [redacted]"]
    assert payload["variants"][0]["options"][0]["value"] == "黑色"


def test_snapshot_object_branch_also_redacts_pii_and_bounds_payload() -> None:
    snapshot = build_product_fact_snapshot(_items()[0]).model_copy(
        update={"provenance": {"phone": "13800000000", "raw": ["x" * 5000] * 1000}}
    )
    payload = sanitize_snapshot_payload(snapshot)
    encoded = json.dumps(payload, ensure_ascii=False)

    assert "13800000000" not in encoded
    assert "[redacted]" in encoded
    assert len(encoded.encode("utf-8")) <= 120_000
    assert "[omitted: snapshot depth limit]" not in json.dumps(
        payload.get("variants", []), ensure_ascii=False
    )
