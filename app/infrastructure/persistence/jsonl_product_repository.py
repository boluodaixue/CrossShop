"""Strict, manifest-verified Product repository backed by the H0 JSONL catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.domain.catalog.money import Money
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.product import Product, ProductHighlight
from app.domain.catalog.sku import Sku

CONTRACT_VERSION = "product-sku-contract-v2"
CONVERTER_VERSION = "catalog-product-sku-builder-v3-product-only"
PARTITION_PRODUCT_PATHS = {
    "amazon/es": "amazon/es/products.jsonl",
    "amazon/jp": "amazon/jp/products.jsonl",
    "amazon/us": "amazon/us/products.jsonl",
    "globex_reference": "globex_reference/products.jsonl",
    "taobao": "taobao/products.jsonl",
}

_PRODUCT_FIELDS = {
    "brand",
    "category",
    "description",
    "highlights",
    "origin_country",
    "product_id",
    "ships_to",
    "skus",
    "title",
}
_HIGHLIGHT_FIELDS = {"detail", "label"}
_SKU_FIELDS = {"price", "sku_id", "spec", "stock"}
_MONEY_FIELDS = {"amount_in_minor_units", "currency"}


class JsonlProductRepository(ProductRepository):
    """Load all five immutable H0 partitions into memory after strict validation."""

    def __init__(self, catalog_root: Path | str) -> None:
        root = Path(catalog_root).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"catalog_root must be a directory: {root}")

        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"catalog manifest is missing: {manifest_path}")
        manifest = _load_json_object(manifest_path)
        _require_exact_value(
            manifest, "contract_version", CONTRACT_VERSION, manifest_path
        )
        _require_exact_value(
            manifest, "converter_version", CONVERTER_VERSION, manifest_path
        )

        partitions = manifest.get("partitions")
        if not isinstance(partitions, dict) or set(partitions) != set(
            PARTITION_PRODUCT_PATHS
        ):
            actual = (
                sorted(partitions)
                if isinstance(partitions, dict)
                else type(partitions).__name__
            )
            raise ValueError(
                "manifest partitions must be exactly "
                f"{sorted(PARTITION_PRODUCT_PATHS)}, actual={actual}"
            )

        products_by_id: dict[str, Product] = {}
        sku_ids: set[str] = set()
        for partition, expected_relative_path in PARTITION_PRODUCT_PATHS.items():
            product_meta = _product_metadata(partitions[partition], partition)
            relative_path = product_meta.get("path")
            if relative_path != expected_relative_path:
                raise ValueError(
                    "unsafe or unexpected product path for "
                    f"{partition}: {relative_path!r}"
                )
            product_path = (root / expected_relative_path).resolve(strict=True)
            _require_within_root(product_path, root)
            loaded = _load_partition(product_path, product_meta)
            for product in loaded:
                if product.product_id in products_by_id:
                    raise ValueError(
                        f"duplicate global product_id: {product.product_id}"
                    )
                for sku in product.skus:
                    if sku.sku_id in sku_ids:
                        raise ValueError(f"duplicate global sku_id: {sku.sku_id}")
                    sku_ids.add(sku.sku_id)
                products_by_id[product.product_id] = product

        # Publish state only after every partition and global identifier has passed.
        self._catalog_root = root
        self._products = products_by_id
        self._sku_count = len(sku_ids)

    @property
    def product_count(self) -> int:
        return len(self._products)

    @property
    def sku_count(self) -> int:
        return self._sku_count

    async def find_by_id(self, product_id: str) -> Product | None:
        return self._products.get(product_id)

    async def find_by_ids(self, product_ids: list[str]) -> list[Product]:
        return [
            self._products[product_id]
            for product_id in product_ids
            if product_id in self._products
        ]

    async def list_all(self) -> list[Product]:
        return list(self._products.values())


def _load_partition(path: Path, metadata: dict[str, Any]) -> list[Product]:
    expected_bytes = _strict_non_negative_int(
        metadata.get("bytes"), f"{path}: manifest bytes"
    )
    expected_records = _strict_non_negative_int(
        metadata.get("records"), f"{path}: manifest records"
    )
    expected_hash = metadata.get("sha256")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(char not in "0123456789abcdef" for char in expected_hash)
    ):
        raise ValueError(f"{path}: manifest sha256 must be 64 lowercase hex characters")

    digest = hashlib.sha256()
    byte_count = 0
    products: list[Product] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            byte_count += len(raw_line)
            if not raw_line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"invalid UTF-8 at {path}:{line_number}") from exc
            row = _strict_json_loads(text, f"{path}:{line_number}")
            if not isinstance(row, dict):
                raise TypeError(f"JSON object required at {path}:{line_number}")
            products.append(_restore_product(row, f"{path}:{line_number}"))

    if byte_count != expected_bytes:
        raise ValueError(
            f"byte count mismatch for {path}: "
            f"expected={expected_bytes}, actual={byte_count}"
        )
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"sha256 mismatch for {path}: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    if len(products) != expected_records:
        raise ValueError(
            f"record count mismatch for {path}: "
            f"expected={expected_records}, actual={len(products)}"
        )
    return products


def _restore_product(row: dict[str, Any], location: str) -> Product:
    _require_fields(row, _PRODUCT_FIELDS, "Product", location)
    product_id = _required_text(row["product_id"], f"{location}.product_id")
    title = _required_text(row["title"], f"{location}.title")
    brand = _required_text(row["brand"], f"{location}.brand")
    category = _required_text(row["category"], f"{location}.category")
    origin_country = _required_text(row["origin_country"], f"{location}.origin_country")
    description = _text(row["description"], f"{location}.description")

    highlights_value = row["highlights"]
    if not isinstance(highlights_value, list):
        raise TypeError(f"{location}.highlights must be a list")
    highlights = [
        _restore_highlight(value, f"{location}.highlights[{index}]")
        for index, value in enumerate(highlights_value)
    ]

    ships_to_value = row["ships_to"]
    if not isinstance(ships_to_value, list):
        raise TypeError(f"{location}.ships_to must be a list")
    ships_to = [
        _required_text(value, f"{location}.ships_to[{index}]")
        for index, value in enumerate(ships_to_value)
    ]

    skus_value = row["skus"]
    if not isinstance(skus_value, list) or not skus_value:
        raise ValueError(f"{location}.skus must be a non-empty list")
    skus = [
        _restore_sku(value, f"{location}.skus[{index}]")
        for index, value in enumerate(skus_value)
    ]
    return Product(
        product_id=product_id,
        title=title,
        brand=brand,
        category=category,
        origin_country=origin_country,
        description=description,
        highlights=highlights,
        ships_to=ships_to,
        skus=skus,
    )


def _restore_highlight(value: Any, location: str) -> ProductHighlight:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be an object")
    _require_fields(value, _HIGHLIGHT_FIELDS, "ProductHighlight", location)
    return ProductHighlight(
        label=_required_text(value["label"], f"{location}.label"),
        detail=_text(value["detail"], f"{location}.detail"),
    )


def _restore_sku(value: Any, location: str) -> Sku:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be an object")
    _require_fields(value, _SKU_FIELDS, "Sku", location)
    price_value = value["price"]
    if not isinstance(price_value, dict):
        raise TypeError(f"{location}.price must be an object")
    _require_fields(price_value, _MONEY_FIELDS, "Money", f"{location}.price")
    amount = _strict_non_negative_int(
        price_value["amount_in_minor_units"], f"{location}.price.amount_in_minor_units"
    )
    currency = _required_text(price_value["currency"], f"{location}.price.currency")
    stock = _strict_non_negative_int(value["stock"], f"{location}.stock")
    return Sku(
        sku_id=_required_text(value["sku_id"], f"{location}.sku_id"),
        spec=_text(value["spec"], f"{location}.spec"),
        price=Money.of(amount, currency),
        stock=stock,
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 in {path}") from exc
    value = _strict_json_loads(text, str(path))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required in {path}")
    return value


def _strict_json_loads(text: str, location: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} at {location}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number {value!r} at {location}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {location}: {exc.msg}") from exc


def _product_metadata(value: Any, partition: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"manifest partition {partition!r} must be an object")
    metadata = value.get("products")
    if not isinstance(metadata, dict):
        raise TypeError(f"manifest partition {partition!r}.products must be an object")
    required = {"bytes", "path", "records", "sha256"}
    if set(metadata) != required:
        raise ValueError(
            f"manifest partition {partition!r}.products fields must be exactly "
            f"{sorted(required)}"
        )
    return metadata


def _require_exact_value(
    value: dict[str, Any], key: str, expected: str, path: Path
) -> None:
    if value.get(key) != expected:
        raise ValueError(
            f"{path}: {key} must be {expected!r}, actual={value.get(key)!r}"
        )


def _require_within_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"catalog path escapes catalog_root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"catalog product file is missing: {path}")


def _require_fields(
    value: dict[str, Any], expected: set[str], name: str, location: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{location}: {name} fields must be exactly {sorted(expected)}, "
            f"difference={sorted(set(value) ^ expected)}"
        )


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{location} must be a string")
    return value


def _required_text(value: Any, location: str) -> str:
    text = _text(value, location)
    if not text.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return text


def _strict_non_negative_int(value: Any, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value
