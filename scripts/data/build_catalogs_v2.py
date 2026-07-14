"""Build strict V2 Product/Sku JSONL catalogs.

The source staging tree is read-only.  All provider-specific filling happens
once in this offline conversion and is persisted in the published JSONL.  A
temporary build directory and manifest checks make publication failure-safe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import unicodedata
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.catalog.product_sku_contract_v2 import Product, ProductHighlight, Sku
from app.domain.catalog.money import Money

ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = ROOT
PARTITIONS = (("amazon", "us"), ("amazon", "es"), ("amazon", "jp"), ("reference_seed", "cn"))
CONTRACT_VERSION = "product-sku-contract-v2"
CONVERTER_VERSION = "catalog-product-sku-builder-v3-product-only"
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
SKU_FIELDS = frozenset({"sku_id", "spec", "price", "stock"})
SOURCE_ITEM_FIELDS = frozenset(
    {
        "attributes",
        "availability",
        "brand",
        "category_path",
        "currency_raw",
        "description",
        "ingested_at",
        "item_id",
        "language",
        "locale",
        "materials",
        "original_price_cny",
        "platform",
        "price_cny",
        "price_source",
        "provenance",
        "rating",
        "review_count",
        "same_group_id",
        "ships_to",
        "source_updated_at",
        "title",
        "url",
        "variants",
    }
)


@dataclass(frozen=True)
class ProductSource:
    product: Product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "data" / "processed" / "staging" / "catalog-schema-v2",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "data" / "processed" / "catalogs-v2"
    )
    args = parser.parse_args()
    manifest = run(args.source_root, args.output_root)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


def run(source_root: Path, output_root: Path) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    sources = _partition_paths(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = (
        output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex[:12]}"
    )
    temporary_root.mkdir(parents=True, exist_ok=False)
    try:
        manifest = _build_into(sources, temporary_root)
        _publish(temporary_root, output_root)
        temporary_root = None
        return manifest
    finally:
        if temporary_root is not None and temporary_root.exists():
            shutil.rmtree(temporary_root)


def _partition_paths(source_root: Path) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    for platform, locale in PARTITIONS:
        path = source_root / platform / locale / "items.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"required staging partition is missing: {path}")
        paths[(platform, locale)] = path
    return paths


def _build_into(sources: dict[tuple[str, str], Path], root: Path) -> dict:
    partitions: dict[str, dict] = {}
    all_product_ids: set[str] = set()
    all_sku_ids: set[str] = set()
    rejections: list[dict[str, str]] = []
    counts: dict[str, int] = {}

    reference_sources = _reference_products()
    _write_partition(
        root / "crossshop_reference",
        "crossshop_reference",
        reference_sources,
        all_product_ids,
        all_sku_ids,
        partitions,
    )
    counts.update(_partition_counts("crossshop_reference", reference_sources))

    for platform, locale in (
        ("reference_seed", "cn"),
        ("amazon", "us"),
        ("amazon", "es"),
        ("amazon", "jp"),
    ):
        rows = list(
            _validated_partition_rows(sources[(platform, locale)], platform, locale)
        )
        converted = (
            [_convert_reference_seed(row) for row in rows]
            if platform == "reference_seed"
            else [_convert_amazon(row) for row in rows]
        )
        converted_sources = [item for item in converted if item is not None]
        rejections.extend(
            {
                "platform": platform,
                "locale": locale,
                "product_id": str(row.get("item_id") or ""),
                "reason": "no_priced_sku",
            }
            for row, item in zip(rows, converted, strict=True)
            if item is None
        )
        key = platform if platform == "reference_seed" else f"{platform}/{locale}"
        _write_partition(
            root / key, key, converted_sources, all_product_ids, all_sku_ids, partitions
        )
        counts.update(_partition_counts(key, converted_sources))
        if len(converted_sources) != len(rows):
            counts[f"{key}_rejected"] = len(rows) - len(converted_sources)

    counts["partitions"] = len(partitions)
    rejection_path = root / "rejections.jsonl"
    _write_jsonl(
        rejection_path,
        sorted(
            rejections,
            key=lambda item: (item["platform"], item["locale"], item["product_id"]),
        ),
    )
    counts["rejected_records"] = len(rejections)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "counts": dict(sorted(counts.items())),
        "partitions": partitions,
        "rejections": _file_metadata(rejection_path, len(rejections)),
        "rules": {
            "reference_seed": {
                "brand": "attributes.shop_name",
                "category": "category_path[-1]",
                "origin_country": "CN",
                "ships_to": "all CN; sha256(product_id) bucket 0/5 adds US,JP,SG",
                "stock": "unavailable -> 0; otherwise sha256(sku_id + ':stock') % 100 + 1",
            },
            "amazon": {
                "sku": "one per item: product_id + ':default', spec='默认规格'",
                "brand_missing": "未知",
                "category": "未知",
                "price_cny": "persisted stable 50.00..2000.00 CNY",
                "stock": "persisted stable 0..100",
                "locale_facts": {
                    "us": {"origin_country": "US", "ships_to": ["US", "CN"]},
                    "es": {"origin_country": "ES", "ships_to": ["ES", "CN"]},
                    "jp": {"origin_country": "JP", "ships_to": ["JP", "CN"]},
                },
            },
            "reference": "V2 seed Product/Sku values preserved",
        },
        "source_partitions": {
            f"{platform}:{locale}": _file_metadata(
                path, relative_path=f"{platform}/{locale}/items.jsonl"
            )
            for (platform, locale), path in sorted(sources.items())
        },
        "reference_source": _file_metadata(
            V2_ROOT / "app" / "infrastructure" / "persistence" / "seed_products.py",
            relative_path="D:/PycharmProjects/crossshop-agent-mainV2/app/infrastructure/persistence/seed_products.py",
        ),
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _write_partition(
    partition_root: Path,
    partition_name: str,
    sources: list[ProductSource],
    product_ids: set[str],
    sku_ids: set[str],
    partitions: dict[str, dict],
) -> None:
    partition_root.mkdir(parents=True, exist_ok=True)
    products = [
        source.product.to_dict()
        for source in sorted(sources, key=lambda s: s.product.product_id)
    ]
    for product in products:
        _validate_product_row(product)
        if product["product_id"] in product_ids:
            raise ValueError(f"duplicate global product_id: {product['product_id']}")
        product_ids.add(product["product_id"])
        for sku in product["skus"]:
            if sku["sku_id"] in sku_ids:
                raise ValueError(f"duplicate global sku_id: {sku['sku_id']}")
            sku_ids.add(sku["sku_id"])
    product_path = partition_root / "products.jsonl"
    _write_jsonl(product_path, products)
    coverage_path = partition_root / "coverage.json"
    _write_json(coverage_path, {"products": _field_coverage(products)})
    partitions[partition_name] = {
        "products": _file_metadata(
            product_path, len(products), f"{partition_name}/products.jsonl"
        ),
        "coverage": _file_metadata(
            coverage_path, relative_path=f"{partition_name}/coverage.json"
        ),
    }


def _partition_counts(key: str, sources: list[ProductSource]) -> dict[str, int]:
    return {
        f"{key}_products": len(sources),
        f"{key}_skus": sum(len(s.product.skus) for s in sources),
    }


def _reference_products() -> list[ProductSource]:
    if not V2_ROOT.is_dir():
        raise FileNotFoundError(f"V2 reference repository missing: {V2_ROOT}")
    sys.path.insert(0, str(V2_ROOT))
    try:
        module = importlib.import_module("app.infrastructure.persistence.seed_products")
        products = module.build_seed_products()
    finally:
        if sys.path and sys.path[0] == str(V2_ROOT):
            sys.path.pop(0)
    result = []
    for product in products:
        result.append(
            ProductSource(
                Product(
                    product.product_id,
                    product.title,
                    product.brand,
                    product.category,
                    product.origin_country,
                    product.description,
                    [ProductHighlight(h.label, h.detail) for h in product.highlights],
                    list(product.ships_to),
                    [
                        Sku(
                            s.sku_id,
                            s.spec,
                            Money.of(s.price.amount_in_minor_units, s.price.currency),
                            s.stock,
                        )
                        for s in product.skus
                    ],
                )
            )
        )
    return result


def _convert_reference_seed(row: dict) -> ProductSource | None:
    product_id = str(row["item_id"])
    shop_name = _shop_name(row) or "未知"
    category = _category(row.get("category_path")) or "未知"
    skus = []
    for variant in sorted(
        row.get("variants") or [], key=lambda value: str(value.get("variant_id") or "")
    ):
        sku_id = str(variant.get("variant_id") or "")
        price = _price(variant.get("price_cny"))
        if (
            not sku_id
            or price is None
            or str(variant.get("price_source") or "") != "observed"
        ):
            continue
        availability = str(variant.get("availability") or "unknown")
        stock = (
            0
            if availability == "unavailable"
            else 1 + _hash_int(sku_id + ":stock", 100)
        )
        skus.append(Sku(sku_id, _spec(variant.get("options")), price, stock))
    if not skus:
        return None
    product = Product(
        product_id,
        str(row.get("title") or "未知"),
        shop_name,
        category,
        "CN",
        str(row.get("description") or ""),
        [],
        _reference_seed_ships_to(product_id),
        skus,
    )
    return ProductSource(product)


def _convert_amazon(row: dict) -> ProductSource:
    product_id = str(row["item_id"])
    locale = str(row["locale"])
    origin, ships_to = {
        "us": ("US", ["US", "CN"]),
        "es": ("ES", ["ES", "CN"]),
        "jp": ("JP", ["JP", "CN"]),
    }[locale]
    sku_id = f"{product_id}:default"
    price = Money.of(5000 + _hash_int(product_id + ":price", 195001), "CNY")
    stock = _hash_int(sku_id + ":stock", 101)
    product = Product(
        product_id,
        str(row.get("title") or "未知"),
        _optional_text(row.get("brand")) or "未知",
        "未知",
        origin,
        str(row.get("description") or ""),
        [],
        ships_to,
        [Sku(sku_id, "默认规格", price, stock)],
    )
    return ProductSource(product)


def _validated_partition_rows(path: Path, platform: str, locale: str) -> Iterable[dict]:
    expected_prefix = f"{platform}:{locale}:"
    for row in _read_jsonl(path):
        if set(row) != SOURCE_ITEM_FIELDS:
            raise ValueError(
                f"source item field mismatch in {path}: "
                f"{sorted(set(row) ^ SOURCE_ITEM_FIELDS)}"
            )
        if row.get("platform") != platform or row.get("locale") != locale:
            raise ValueError(f"partition identity mismatch in {path}")
        if not str(row.get("item_id") or "").startswith(expected_prefix):
            raise ValueError(f"partition item_id prefix mismatch in {path}")
        yield row


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSON object required at {path}:{line_number}")
                yield value


def _validate_product_row(row: dict) -> None:
    if set(row) != PRODUCT_FIELDS or not row["skus"]:
        raise ValueError(f"Product field mismatch: {sorted(set(row) ^ PRODUCT_FIELDS)}")
    if any(set(sku) != SKU_FIELDS or sku["stock"] < 0 for sku in row["skus"]):
        raise ValueError(f"Sku field or stock mismatch: {row['product_id']}")


def _shop_name(row: dict) -> str | None:
    for attr in row.get("attributes") or []:
        if (
            isinstance(attr, dict)
            and str(attr.get("code") or attr.get("name") or "").casefold()
            == "shop_name"
        ):
            return _optional_text(attr.get("value"))
    return None


def _reference_seed_ships_to(product_id: str) -> list[str]:
    return ["CN", "US", "JP", "SG"] if _hash_int(product_id, 5) == 0 else ["CN"]


def _hash_int(value: str, modulo: int) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest(), 16) % modulo


def _price(value: object) -> Money | None:
    try:
        amount = Decimal(str(value))
        minor = amount * 100
        if (
            value is None
            or not amount.is_finite()
            or amount <= 0
            or minor != minor.to_integral_value()
        ):
            return None
        return Money.of(int(minor), "CNY")
    except (InvalidOperation, TypeError, ValueError):
        return None


def _category(value: object) -> str | None:
    if not isinstance(value, list):
        return None
    values = [str(v).strip() for v in value if str(v).strip()]
    return values[-1] if values else None


def _spec(options: object) -> str:
    values = []
    for option in options if isinstance(options, list) else []:
        if not isinstance(option, dict):
            continue
        name, value = (
            _optional_text(option.get("name")),
            _optional_text(option.get("value")),
        )
        if name or value:
            display = "：".join(part for part in (name, value) if part)
            values.append(
                (
                    _normalized_option_text(name or ""),
                    _normalized_option_text(value or ""),
                    display,
                )
            )
    return " / ".join(item[2] for item in sorted(values))


def _normalized_option_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value not in (None, "") else ""
    return text or None


def _field_coverage(rows: list[dict]) -> dict:
    return {
        field: {
            "non_null": sum(row.get(field) is not None for row in rows),
            "null": sum(row.get(field) is None for row in rows),
        }
        for field in sorted({field for row in rows for field in row})
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(
    path: Path, records: int | None = None, relative_path: str | None = None
) -> dict:
    result = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if records is not None:
        result["records"] = records
    if relative_path is not None:
        result["path"] = relative_path
    return result


def _publish(temporary_root: Path, output_root: Path) -> None:
    backup = output_root.with_name(f".{output_root.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    moved = False
    try:
        if output_root.exists():
            os.replace(output_root, backup)
            moved = True
        os.replace(temporary_root, output_root)
    except Exception:
        if moved and backup.exists() and not output_root.exists():
            os.replace(backup, output_root)
        raise
    if backup.exists():
        shutil.rmtree(backup)


if __name__ == "__main__":
    main()
