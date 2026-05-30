"""Migrate legacy JSONL catalog rows into typed catalog-schema-v2 staging files.

The legacy files are read-only inputs.  Each output is written to a separate
staging tree and is validated by Pydantic before it is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from globex_agent.domain import (
    AvailabilityStatus,
    MaterialComponent,
    PriceSource,
    ProductAttribute,
    StandardItem,
    StandardItemVariant,
    VariantOption,
    canonical_options_serialization,
    canonical_variant_id,
)

ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = (("amazon", "us"), ("amazon", "es"), ("amazon", "jp"), ("taobao", "cn"))
ALIASES = {
    "真皮": ("genuine_leather", "真皮"),
    "合成革": ("synthetic_leather", "合成革"),
    "蛋白皮": ("protein_leather", "蛋白皮"),
    "乳胶": ("latex", "乳胶"),
    "棉": ("cotton", "棉"),
    "纯棉": ("cotton", "棉"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path, default=ROOT / "data" / "processed" / "catalogs"
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=ROOT / "data" / "processed" / "staging" / "catalog-schema-v2",
    )
    parser.add_argument("--demo-source", type=Path, default=None)
    parser.add_argument("--demo-output", type=Path, default=None)
    args = parser.parse_args()
    if args.demo_source and args.demo_output:
        migrate_partition(args.demo_source, args.demo_output)
        return
    manifest: dict = {
        "schema_version": "catalog-schema-v2",
        "material_taxonomy_version": "material-taxonomy-v1",
        "partitions": {},
    }
    for platform, locale in PARTITIONS:
        source = args.source_root / platform / locale / "items.jsonl"
        output = args.staging_root / platform / locale / "items.jsonl"
        stats = migrate_partition(source, output)
        manifest["partitions"][f"{platform}:{locale}"] = stats
        print(
            f"{platform}:{locale} items={stats['items']} "
            f"variants={stats['variants']} failures={stats['failures']} "
            f"duplicate_variant_ids={stats['duplicate_variant_ids']}"
        )
    # The manifest is reproducible: source/output hashes, not wall-clock time,
    # identify this migration run.
    manifest["manifest_sha256"] = _write_manifest(args.staging_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def migrate_partition(source: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    errors: list[dict] = []
    duplicate_ids = 0
    duplicate_source_id_groups = 0
    duplicate_groups = 0
    conflict_groups = 0
    merged_records = 0
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, 1):
            if not line.strip():
                continue
            counts["source_records"] += 1
            try:
                row = json.loads(line)
                item = convert(row)
                writer.write(item.model_dump_json() + "\n")
                counts["items"] += 1
                counts["variants"] += len(item.variants)
                counts["source_variants"] += len(row.get("variants") or [])
                counts["unknown_availability"] += int(
                    item.availability is AvailabilityStatus.UNKNOWN
                )
                counts["unknown_variant_availability"] += sum(
                    v.availability is AvailabilityStatus.UNKNOWN for v in item.variants
                )
                counts["unknown_variant_prices"] += sum(v.price_cny is None for v in item.variants)
                counts["mapped_materials"] += sum(m.code is not None for m in item.materials)
                counts["unmapped_materials"] += sum(m.code is None for m in item.materials)
                item_stats = getattr(item, "_migration_stats", {})
                duplicate_ids += int(item_stats.get("duplicate_source_records", 0))
                duplicate_source_id_groups += int(
                    item_stats.get("duplicate_source_id_groups", 0)
                )
                duplicate_groups += int(item_stats.get("duplicate_option_groups", 0))
                conflict_groups += int(item_stats.get("conflict_groups", 0))
                merged_records += int(item_stats.get("merged_records", 0))
            except Exception as exc:  # preserve a bounded audit trail
                counts["failures"] += 1
                errors.append({"line": line_number, "error": str(exc)[:300]})
    if errors:
        error_path = output.with_name("migration-errors.jsonl")
        error_path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors), encoding="utf-8"
        )
    counts["duplicate_variant_ids"] = 0
    counts["duplicate_source_variant_records"] = duplicate_ids
    counts["duplicate_source_id_groups"] = duplicate_source_id_groups
    counts["duplicate_option_groups"] = duplicate_groups
    counts["merged_duplicate_records"] = merged_records
    counts["migration_conflict_groups"] = conflict_groups
    counts["failures"] = len(errors)
    counts["source_sha256"] = _sha256(source)
    counts["output_sha256"] = _sha256(output)
    return dict(counts)


def convert(row: dict) -> StandardItem:
    raw_ids = [
        str(value.get("variant_id") or value.get("id") or "")
        for value in (row.get("variants") or [])
        if isinstance(value, dict)
    ]
    variants, migration_stats = _variants(
        row.get("variants") or [], str(row["platform"]), str(row["item_id"])
    )
    migration_stats["duplicate_source_records"] = len(raw_ids) - len(set(raw_ids))
    migration_stats["duplicate_source_id_groups"] = sum(
        count > 1 for count in Counter(raw_ids).values()
    )
    attrs: list[ProductAttribute] = []
    materials: list[MaterialComponent] = []
    ships_to = None
    for key, value in (row.get("attributes") or {}).items():
        normalized = str(key).strip().casefold()
        if normalized in {"ships_to", "ship_to"}:
            if isinstance(value, (list, tuple, set)):
                ships_to = [str(v).strip().upper() for v in value if str(v).strip()]
            continue
        if normalized in {"material", "materials", "材质", "fabric"}:
            materials.extend(_materials(value))
            continue
        if normalized == "source_attributes" and isinstance(value, list):
            material_values = [entry for entry in value if _looks_like_material(entry)]
            materials.extend(_materials(material_values))
            value = [entry for entry in value if not _looks_like_material(entry)]
            if not value:
                continue
        if normalized in {
            "price",
            "price_cny",
            "price_range_cny",
            "price_requires_variant_selection",
            "availability",
            "is_available",
        }:
            continue
        if not str(key).strip():
            continue
        safe_value = (
            value
            if isinstance(value, (str, int, float, bool)) or value is None
            else json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        attrs.append(
            ProductAttribute(code=str(key).strip(), name=str(key).strip(), value=safe_value or "")
        )
    item_price = _money(row.get("price_cny")) if not variants else None
    source = PriceSource.OBSERVED if item_price is not None else PriceSource.UNAVAILABLE
    if variants:
        source = PriceSource.UNAVAILABLE
    platform = str(row["platform"])
    item = StandardItem(
        item_id=str(row["item_id"]),
        same_group_id=str(row.get("same_group_id") or row["item_id"]),
        platform=platform,
        locale=row.get("locale"),
        language=row.get("language"),
        title=str(row["title"]),
        description=str(row.get("description") or ""),
        brand=row.get("brand"),
        category_path=row.get("category_path") or ["未知类目"],
        price_cny=item_price,
        original_price_cny=_money(row.get("original_price_cny")) if not variants else None,
        currency_raw=row.get("currency_raw"),
        price_source=source,
        rating=row.get("rating"),
        review_count=row.get("review_count", 0),
        attributes=attrs,
        materials=materials,
        variants=variants,
        availability=_availability(row.get("availability", row.get("is_available")), variants),
        ships_to=ships_to,
        source_updated_at=row.get("source_updated_at"),
        ingested_at=row["ingested_at"],
        url=row.get("url"),
        provenance=row["provenance"],
    )
    # Keep migration counters out of the canonical model and expose them to the
    # writer without introducing runtime identity/offer fields.
    object.__setattr__(item, "_migration_stats", migration_stats)
    return item


def _variants(
    raw_values: list[dict], platform: str, item_id: str
) -> tuple[list[StandardItemVariant], dict]:
    records = [_variant_record(raw, index) for index, raw in enumerate(raw_values, 1)]
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(canonical_options_serialization(record["options"]), []).append(record)
    result: list[StandardItemVariant] = []
    duplicate_source_records = 0
    duplicate_option_groups = 0
    merged_records = 0
    conflict_groups = 0
    for fingerprint in sorted(groups):
        group = sorted(
            groups[fingerprint],
            key=lambda record: (
                record["source_variant_id"],
                str(record["price_cny"]),
                record["availability"].value,
            ),
        )
        source_ids = [record["source_variant_id"] for record in group]
        duplicate_source_records += len(source_ids) - len(set(source_ids))
        if len(group) > 1:
            duplicate_option_groups += 1
            merged_records += len(group) - 1
        price_states = {(record["price_cny"], record["original_price_cny"]) for record in group}
        availability_states = {record["availability"] for record in group}
        conflict = len(price_states) > 1 or len(availability_states) > 1
        if conflict:
            conflict_groups += 1
        chosen = group[0]
        result.append(
            StandardItemVariant(
                variant_id=canonical_variant_id(platform, item_id, chosen["options"]),
                source_variant_id=source_ids[0],
                options=chosen["options"],
                price_cny=None if conflict else chosen["price_cny"],
                original_price_cny=None if conflict else chosen["original_price_cny"],
                price_source=(
                    PriceSource.UNAVAILABLE
                    if conflict or chosen["price_cny"] is None
                    else PriceSource.OBSERVED
                ),
                availability=AvailabilityStatus.UNKNOWN if conflict else chosen["availability"],
                materials=chosen["materials"],
            )
        )
    return result, {
        "duplicate_source_records": duplicate_source_records,
        "duplicate_option_groups": duplicate_option_groups,
        "merged_records": merged_records,
        "conflict_groups": conflict_groups,
    }


def _variant_record(raw: dict, index: int) -> dict:
    price = _money(raw.get("price_cny", raw.get("price")))
    options = raw.get("options")
    if not isinstance(options, list) or not options:
        option_name = str(raw.get("option_name") or raw.get("name") or "规格")
        value = str(raw.get("value") or raw.get("spec") or "未指定")
        options = [{"name": option_name, "value": value}]
    typed_options = [
        VariantOption(
            code=entry.get("code"),
            name=str(entry.get("name") or entry.get("option_name") or "规格"),
            value=str(entry.get("value") or entry.get("spec") or "未指定"),
        )
        for entry in options
        if isinstance(entry, dict)
    ]
    source_id = str(
        raw.get("source_variant_id")
        or raw.get("variant_id")
        or raw.get("id")
        or f"source-variant-{index}"
    )
    original = _money(raw.get("original_price_cny", raw.get("original_price")))
    return {
        "source_variant_id": source_id,
        "options": typed_options,
        "price_cny": price,
        "original_price_cny": original if price is not None and original is not None else None,
        "availability": _availability(raw.get("availability", raw.get("is_available")), []),
        "materials": _materials(raw.get("material", raw.get("材质"))) or None,
    }


def _money(value):
    try:
        result = Decimal(str(value))
        return result.quantize(Decimal("0.01")) if result.is_finite() and result > 0 else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def _availability(value, variants) -> AvailabilityStatus:
    if isinstance(value, bool):
        return AvailabilityStatus.AVAILABLE if value else AvailabilityStatus.UNAVAILABLE
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"available", "true", "1", "yes", "在售"}:
            return AvailabilityStatus.AVAILABLE
        if normalized in {"unavailable", "false", "0", "no", "下架", "缺货"}:
            return AvailabilityStatus.UNAVAILABLE
    statuses = {v.availability for v in variants}
    if AvailabilityStatus.AVAILABLE in statuses:
        return AvailabilityStatus.AVAILABLE
    if statuses == {AvailabilityStatus.UNAVAILABLE}:
        return AvailabilityStatus.UNAVAILABLE
    return AvailabilityStatus.UNKNOWN


def _materials(value) -> list[MaterialComponent]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for entry in values:
        name = str(entry).strip()
        code, canonical = ALIASES.get(name, (None, name))
        if code is None:
            for alias, (alias_code, alias_name) in ALIASES.items():
                if alias in name:
                    code, canonical = alias_code, alias_name
                    break
        if name:
            result.append(MaterialComponent(code=code, name=canonical))
    return result


def _looks_like_material(value: object) -> bool:
    text = str(value).strip()
    return any(alias in text for alias in ALIASES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(path: Path, value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return _sha256(path)


if __name__ == "__main__":
    main()
