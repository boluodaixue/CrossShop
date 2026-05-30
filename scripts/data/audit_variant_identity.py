"""Read-only audit of ShopSimulator variant identity and offer semantics.

This script never writes catalog data or SQLite.  It reads the immutable raw
gzip, the already-published schema-v2 staging JSONL, and the SQLite manifest,
then writes one deterministic JSON audit artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data" / "raw" / "taobao_shopsimulator" / "fine_items_eval_train_all.json.gz"
DEFAULT_STAGING = (
    ROOT / "data" / "processed" / "staging" / "catalog-schema-v2" / "taobao" / "cn" / "items.jsonl"
)
DEFAULT_SQLITE = ROOT / "data" / "processed" / "databases" / "taobao" / "catalog.sqlite3"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "staging" / "catalog-schema-v2" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "output" / "audit" / "shopsimulator_variant_identity.json"
OPTION_KEYS = ("option_name", "name", "spec")
VALUE_KEYS = ("value", "option_value")
PRICE_KEYS = ("price_cny", "price", "sale_price", "sku_price")
AVAILABILITY_KEYS = ("availability", "is_available", "available")
TIME_KEYS = ("created_at", "updated_at", "snapshot_at", "observed_at", "timestamp")
SELLER_KEYS = ("seller_id", "shop_id", "store_id", "merchant_id")
LISTING_KEYS = ("listing_id", "item_id", "product_id", "asin")
SKU_KEYS = ("sku", "sku_id", "variant_id", "id", "asin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit(args.raw, args.staging, args.sqlite, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["summary"], ensure_ascii=False, sort_keys=True))


def build_audit(
    raw_path: Path, staging_path: Path, sqlite_path: Path, manifest_path: Path
) -> dict[str, Any]:
    raw_bytes = raw_path.read_bytes()
    with gzip.open(raw_path, "rt", encoding="utf-8") as source:
        raw_rows = json.load(source)
    if not isinstance(raw_rows, list):
        raise ValueError("raw ShopSimulator root must be a JSON list")

    source_fields = _field_inventory(raw_rows)
    source_variant_rows = _source_variants(raw_rows)
    staging_rows = _load_jsonl(staging_path)
    staging_variant_rows = [
        (row["item_id"], variant) for row in staging_rows for variant in row.get("variants", [])
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sqlite_summary = _sqlite_summary(sqlite_path)
    duplicate_audit = _duplicate_audit(source_variant_rows)
    key_audit = _key_audit(source_variant_rows)

    return {
        "audit_version": "shopsimulator-variant-identity-audit-v1",
        "source": {
            "path": _relative(raw_path),
            "sha256": _sha256_bytes(raw_bytes),
            "record_count": len(raw_rows),
            "top_level_fields": sorted({key for row in raw_rows for key in row}),
            "field_inventory": source_fields,
            "seller_fields_present": sorted(
                _present_fields(source_fields, SELLER_KEYS, root_only=True)
            ),
            "listing_fields_present": sorted(
                _present_fields(source_fields, LISTING_KEYS, root_only=True)
            ),
            "sku_fields_present": sorted(
                _present_fields(source_fields, SKU_KEYS, variant_only=True)
            ),
            "offer_fields_present": sorted(_present_fields(source_fields, ("offer", "offers"))),
            "time_fields_present": sorted(_present_fields(source_fields, TIME_KEYS)),
        },
        "canonical_lineage": {
            "migration_script": "scripts/data/migrate_catalog_schema_v2.py",
            "platform": "taobao",
            "item_id": "taobao:cn:{asin}",
            "variant_id": (
                "internal canonical ID is sha256(platform|item_id|canonical options) with a "
                "24-hex digest; source_variant_id preserves raw asin/variant_id/id for trace"
            ),
            "variant_id_algorithm": (
                "NFKC + whitespace collapse + casefold for option name/value; sort options "
                "by normalized name/value; JSON serialization excludes price, stock and input order"
            ),
            "options": (
                "raw customization_options.<option_name>[].option_name/name plus value; "
                "current raw uses the mapping key as option name and entry.value as value"
            ),
            "price_cny": (
                "raw variants[].price_cny, then price; item price_cny is nulled "
                "whenever variants are present"
            ),
            "availability": (
                "raw variants[].availability, then is_available; missing remains unknown"
            ),
            "seller_listing_offer": (
                "no source field observed; no seller, listing, offer, or time dimension "
                "is inferred"
            ),
        },
        "normalized_catalog": {
            "path": _relative(staging_path),
            "sha256": _sha256_file(staging_path),
            "item_count": len(staging_rows),
            "variant_count": len(staging_variant_rows),
            "duplicate_suffix_variant_ids": sum(
                "~" in str(variant.get("variant_id", "")) for _, variant in staging_variant_rows
            ),
            "source_variant_id_present": sum(
                bool(variant.get("source_variant_id")) for _, variant in staging_variant_rows
            ),
            "manifest_partition": manifest.get("partitions", {}).get("taobao:cn", {}),
            "sqlite": sqlite_summary,
        },
        "duplicate_scope_audit": duplicate_audit,
        "candidate_key_audit": key_audit,
        "summary": {
            "source_items": len(raw_rows),
            "source_variants": len(source_variant_rows),
            "staging_items": len(staging_rows),
            "staging_variants": len(staging_variant_rows),
            "same_item_duplicate_records": duplicate_audit["same_item_source_id"][
                "duplicate_records"
            ],
            "same_item_duplicate_groups": duplicate_audit["same_item_source_id"][
                "duplicate_groups"
            ],
            "cross_item_source_id_groups": duplicate_audit["cross_item_source_id"][
                "duplicate_groups"
            ],
            "seller_scoped_duplicate_groups": None,
            "listing_scoped_duplicate_groups": None,
            "seller_fields_present": bool(
                _present_fields(source_fields, SELLER_KEYS, root_only=True)
            ),
            "listing_fields_present": bool(
                _present_fields(source_fields, LISTING_KEYS, root_only=True)
            ),
            "source_variant_id_missing_records": sum(
                not row["source_sku_id"] for row in source_variant_rows
            ),
            "classification_total_groups": sum(duplicate_audit["classification_counts"].values()),
            "classification_unclassified": duplicate_audit["classification_counts"].get(
                "G_information_insufficient", 0
            ),
        },
        "model_conclusion": {
            "reliable_seller_listing_sku": False,
            "recommended_shape": (
                "ambiguous_offer_conflict; do not auto-upgrade to "
                "ProductVariant + SkuOffer"
            ),
            "definitions": {
                "same_group_id": (
                    "cross-platform logical association only; not a seller or offer identity"
                ),
                "source_item": (
                    "raw asin/product listing record; canonical item_id=taobao:cn:{asin}"
                ),
                "seller": "not observable in this source snapshot",
                "source_sku": "raw variants[].variant_id or id when present",
                "variant": (
                    "an item-scoped option/price/availability record, "
                    "not proven seller-scoped SKU"
                ),
                "offer": "not observable because seller/listing/offer/time fields are absent",
            },
            "current_suffix_safety": (
                "safe only as a lossless item-local collision-preserving staging identifier; "
                "it must not be interpreted as seller/offer identity or used to claim "
                "distinct purchasable offers"
            ),
        },
    }


def _source_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in rows:
        item_id = f"taobao:cn:{item.get('asin', '')}"
        options = item.get("customization_options")
        if not isinstance(options, dict):
            continue
        for option_name, values in options.items():
            if not isinstance(values, list):
                continue
            for index, raw in enumerate(values, 1):
                if not isinstance(raw, dict):
                    continue
                source_id = _first(raw, ("variant_id", "id", "sku", "sku_id", "asin"))
                result.append(
                    {
                        "item_id": item_id,
                        "source_sku_id": str(source_id).strip()
                        if source_id not in (None, "")
                        else "",
                        "option_fingerprint": _option_fingerprint(
                            [{"name": str(option_name), "value": raw.get("value", "")}]
                        ),
                        "options": [{"name": str(option_name), "value": str(raw.get("value", ""))}],
                        "price": _first(raw, PRICE_KEYS),
                        "availability": _availability(raw),
                        "time": {key: raw.get(key) for key in TIME_KEYS if key in raw},
                        "source_index": index,
                    }
                )
    return result


def _duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = _groups(
        rows, lambda row: row["source_sku_id"] if row["source_sku_id"] else "<missing>"
    )
    by_item_source = _groups(rows, lambda row: (row["item_id"], row["source_sku_id"]))
    cross_item = {
        key: group
        for key, group in by_source.items()
        if key != "<missing>" and len({row["item_id"] for row in group}) > 1
    }
    same_item = {key: group for key, group in by_item_source.items() if key[1] and len(group) > 1}
    classifications = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in same_item.values():
        category = _classify(group)
        classifications[category] += 1
        if len(samples[category]) < 3:
            samples[category].append(_sample(group))
    return {
        "source_id_scope": (
            "raw variant ID is evaluated as (item_id, source_sku_id); "
            "seller/listing scopes are unavailable"
        ),
        "same_item_source_id": {
            "duplicate_records": sum(len(group) - 1 for group in same_item.values()),
            "duplicate_groups": len(same_item),
            "affected_items": len({key[0] for key in same_item}),
        },
        "cross_item_source_id": {
            "duplicate_records": sum(len(group) - 1 for group in cross_item.values()),
            "duplicate_groups": len(cross_item),
            "affected_items": len(
                {row["item_id"] for group in cross_item.values() for row in group}
            ),
        },
        "same_seller_source_id": {
            "available": False,
            "duplicate_groups": None,
            "affected_sellers": None,
        },
        "same_listing_source_id": {
            "available": False,
            "duplicate_groups": None,
            "affected_listings": None,
        },
        "classification_counts": dict(sorted(classifications.items())),
        "classification_samples": dict(sorted(samples.items())),
    }


def _key_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = {
        "platform_source_sku_id": lambda row: ("taobao", row["source_sku_id"]),
        "platform_item_source_sku_id": lambda row: ("taobao", row["item_id"], row["source_sku_id"]),
        "platform_seller_source_sku_id": None,
        "platform_seller_source_item_source_sku_id": None,
        "item_options_fingerprint": lambda row: (row["item_id"], row["option_fingerprint"]),
    }
    result = {}
    for name, key_fn in candidates.items():
        if key_fn is None:
            result[name] = {
                "available": False,
                "reason": "seller_id/shop_id/store_id/merchant_id absent",
            }
            continue
        groups = _groups(rows, key_fn)
        duplicate = {str(key): group for key, group in groups.items() if key[-1] and len(group) > 1}
        result[name] = {
            "available": True,
            "duplicate_groups": len(duplicate),
            "duplicate_records": sum(len(group) - 1 for group in duplicate.values()),
            "affected_items": len(
                {row["item_id"] for group in duplicate.values() for row in group}
            ),
        }
    return result


def _classify(group: list[dict[str, Any]]) -> str:
    if any(not row["source_sku_id"] or not row["options"] for row in group):
        return "G_information_insufficient"
    option_set = {row["option_fingerprint"] for row in group}
    prices = {_stable_json(row["price"]) for row in group}
    availability = {_stable_json(row["availability"]) for row in group}
    times = {_stable_json(row["time"]) for row in group}
    if len(option_set) > 1:
        if len(prices) == 1 and len(availability) == 1 and len(times) > 1:
            return "F_time_snapshot_only"
        return "B_options_different"
    if len(prices) == 1 and len(availability) == 1:
        return "F_time_snapshot_only" if len(times) > 1 else "A_completely_same"
    if len(prices) > 1 and len(availability) > 1:
        return "E_price_and_availability_conflict"
    if len(prices) > 1:
        return "C_same_options_different_price"
    if len(availability) > 1:
        return "D_same_options_price_different_availability"
    return "G_information_insufficient"


def _sample(group: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "item_id": group[0]["item_id"],
        "source_sku_id": group[0]["source_sku_id"],
        "records": [
            {
                "options": row["options"],
                "price": row["price"],
                "availability": row["availability"],
                "time": row["time"],
            }
            for row in group[:4]
        ],
    }


def _field_inventory(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    types: dict[str, Counter[str]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    for row in rows:
        _walk_fields(row, "", types, counts)
    return {
        path: {"count": counts[path], "types": dict(sorted(types[path].items()))}
        for path in sorted(types)
    }


def _walk_fields(
    value: Any, path: str, types: dict[str, Counter[str]], counts: Counter[str]
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}".strip(".")
            counts[child_path] += 1
            types[child_path][_type_name(child)] += 1
            _walk_fields(child, child_path, types, counts)
    elif isinstance(value, list):
        item_path = f"{path}[]"
        for child in value:
            types[item_path][_type_name(child)] += 1
            _walk_fields(child, item_path, types, counts)


def _sqlite_summary(path: Path) -> dict[str, Any]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        item_count = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
    return {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "item_count": item_count,
        "metadata": metadata,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _groups(rows: list[dict[str, Any]], key_fn) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def _option_fingerprint(options: list[dict[str, str]]) -> str:
    normalized = sorted(
        (
            " ".join(str(option.get("name", "")).split()).casefold(),
            " ".join(str(option.get("value", "")).split()).casefold(),
        )
        for option in options
    )
    return _stable_json(normalized)


def _availability(row: dict[str, Any]) -> Any:
    for key in AVAILABILITY_KEYS:
        if key in row:
            return row[key]
    return None


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _present_fields(
    inventory: dict[str, Any],
    names: tuple[str, ...],
    *,
    root_only: bool = False,
    variant_only: bool = False,
) -> set[str]:
    paths = set(inventory)
    if root_only:
        return {path for path in paths if path in names}
    if variant_only:
        return {
            "customization_options.<option_name>[]." + name
            for name in names
            if any(
                "customization_options." in path and path.endswith(f"[].{name}")
                for path in paths
            )
        }
    return {path for path in paths if path.rsplit(".", 1)[-1] in names}


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


if __name__ == "__main__":
    main()
