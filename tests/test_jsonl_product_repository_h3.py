from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from app.infrastructure.persistence.jsonl_product_repository import (
    CONTRACT_VERSION,
    CONVERTER_VERSION,
    PARTITION_PRODUCT_PATHS,
    JsonlProductRepository,
)


def _product(product_id: str, sku_id: str) -> dict:
    return {
        "brand": "Globex",
        "category": "耳机",
        "description": "",
        "highlights": [{"detail": "长续航", "label": "续航"}],
        "origin_country": "CN",
        "product_id": product_id,
        "ships_to": ["CN", "US"],
        "skus": [
            {
                "price": {"amount_in_minor_units": 12900, "currency": "CNY"},
                "sku_id": sku_id,
                "spec": "黑色",
                "stock": 8,
            }
        ],
        "title": "降噪耳机",
    }


def _write_catalog(root: Path, rows: dict[str, list[dict]] | None = None) -> Path:
    rows = rows or {
        partition: [_product(f"product-{index}", f"sku-{index}")]
        for index, partition in enumerate(PARTITION_PRODUCT_PATHS, 1)
    }
    partitions = {}
    for partition, relative_path in PARTITION_PRODUCT_PATHS.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(
            (
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
                + "\n"
            ).encode()
            for row in rows[partition]
        )
        path.write_bytes(payload)
        partitions[partition] = {
            "coverage": {},
            "products": {
                "bytes": len(payload),
                "path": relative_path,
                "records": len(rows[partition]),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "converter_version": CONVERTER_VERSION,
                "partitions": partitions,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def test_loads_exact_contract_and_preserves_lookup_order(tmp_path: Path) -> None:
    repository = JsonlProductRepository(_write_catalog(tmp_path / "catalog"))

    assert repository.product_count == 5
    assert repository.sku_count == 5
    requested = ["product-3", "missing", "product-1", "product-3"]
    found = asyncio.run(repository.find_by_ids(requested))
    assert [product.product_id for product in found] == [
        "product-3",
        "product-1",
        "product-3",
    ]
    first = asyncio.run(repository.find_by_id("product-1"))
    assert first is not None
    assert first.highlights[0].label == "续航"
    assert first.skus[0].price.amount_in_minor_units == 12900
    assert asyncio.run(repository.find_by_id("missing")) is None
    assert len(asyncio.run(repository.list_all())) == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", "old-contract"),
        ("converter_version", "old-converter"),
    ],
)
def test_rejects_manifest_version_mismatch(
    tmp_path: Path, field: str, value: str
) -> None:
    root = _write_catalog(tmp_path / "catalog")
    manifest = _manifest(root)
    manifest[field] = value
    _write_manifest(root, manifest)

    with pytest.raises(ValueError, match=field):
        JsonlProductRepository(root)


def test_requires_exactly_the_five_fixed_partitions(tmp_path: Path) -> None:
    root = _write_catalog(tmp_path / "catalog")
    manifest = _manifest(root)
    manifest["partitions"].pop("amazon/es")
    _write_manifest(root, manifest)

    with pytest.raises(ValueError, match="partitions must be exactly"):
        JsonlProductRepository(root)


def test_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    root = _write_catalog(tmp_path / "catalog")
    manifest = _manifest(root)
    manifest["partitions"]["taobao"]["products"]["path"] = "../outside.jsonl"
    _write_manifest(root, manifest)

    with pytest.raises(ValueError, match="unsafe or unexpected"):
        JsonlProductRepository(root)


@pytest.mark.parametrize("metadata_field", ["bytes", "records", "sha256"])
def test_rejects_manifest_file_evidence_mismatch(
    tmp_path: Path, metadata_field: str
) -> None:
    root = _write_catalog(tmp_path / "catalog")
    manifest = _manifest(root)
    metadata = manifest["partitions"]["taobao"]["products"]
    metadata[metadata_field] = (
        "0" * 64 if metadata_field == "sha256" else metadata[metadata_field] + 1
    )
    _write_manifest(root, manifest)

    with pytest.raises(
        ValueError, match=f"{metadata_field}|sha256|byte count|record count"
    ):
        JsonlProductRepository(root)


def test_rejects_unknown_product_field_without_compatibility(tmp_path: Path) -> None:
    rows = {
        partition: [_product(f"product-{index}", f"sku-{index}")]
        for index, partition in enumerate(PARTITION_PRODUCT_PATHS, 1)
    }
    rows["taobao"][0]["item_id"] = rows["taobao"][0].pop("product_id")
    root = _write_catalog(tmp_path / "catalog", rows)

    with pytest.raises(ValueError, match="Product fields must be exactly"):
        JsonlProductRepository(root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("ships_to", "CN"),
        lambda row: row.__setitem__("title", 123),
        lambda row: row["highlights"][0].__setitem__("label", ""),
        lambda row: row["skus"][0].__setitem__("stock", True),
        lambda row: row["skus"][0]["price"].__setitem__(
            "amount_in_minor_units", "12900"
        ),
        lambda row: row["skus"][0]["price"].__setitem__("currency", "RMB"),
    ],
)
def test_rejects_wrong_nested_types_and_values(tmp_path: Path, mutate) -> None:
    rows = {
        partition: [_product(f"product-{index}", f"sku-{index}")]
        for index, partition in enumerate(PARTITION_PRODUCT_PATHS, 1)
    }
    mutate(rows["taobao"][0])
    root = _write_catalog(tmp_path / "catalog", rows)

    with pytest.raises((TypeError, ValueError)):
        JsonlProductRepository(root)


@pytest.mark.parametrize("duplicate_kind", ["product", "sku"])
def test_rejects_global_duplicate_ids_atomically(
    tmp_path: Path, duplicate_kind: str
) -> None:
    rows = {
        partition: [_product(f"product-{index}", f"sku-{index}")]
        for index, partition in enumerate(PARTITION_PRODUCT_PATHS, 1)
    }
    first = rows["amazon/es"][0]
    second = rows["amazon/jp"][0]
    if duplicate_kind == "product":
        second["product_id"] = first["product_id"]
    else:
        second["skus"][0]["sku_id"] = first["skus"][0]["sku_id"]
    root = _write_catalog(tmp_path / "catalog", rows)
    repository = JsonlProductRepository.__new__(JsonlProductRepository)

    with pytest.raises(ValueError, match=f"duplicate global {duplicate_kind}_id"):
        repository.__init__(root)
    assert not hasattr(repository, "_products")


def test_restores_every_saved_h2_hit_from_the_real_catalog() -> None:
    root = Path(__file__).resolve().parents[1]
    repository = JsonlProductRepository(root / "data" / "processed" / "catalogs-v2")
    report = json.loads(
        (root / "artifacts" / "h2-opensearch" / "adapter-smoke-report.json").read_text(
            encoding="utf-8"
        )
    )
    product_ids = [
        hit["product_id"]
        for platform in report["platforms"].values()
        for hit in platform["hits"]
    ]

    restored = asyncio.run(repository.find_by_ids(product_ids))

    assert [product.product_id for product in restored] == product_ids
    assert all(product.skus for product in restored)
