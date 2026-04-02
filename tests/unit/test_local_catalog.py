from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from globex_agent.catalog import CatalogValidationError, LocalCatalog
from globex_agent.domain import (
    CatalogRecord,
    Currency,
    DataProvenance,
    Offer,
    Platform,
    ProvenanceKind,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CATALOG_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


@pytest.fixture
def catalog_path(request: pytest.FixtureRequest):
    """Use an existing directory to avoid Windows temp ACL issues in this workspace."""
    path = Path(__file__).with_name(f".{request.node.name}.jsonl")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def make_provenance(record_id: str) -> DataProvenance:
    return DataProvenance(
        kind=ProvenanceKind.SYNTHETIC,
        source="unit test",
        source_record_id=record_id,
        generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )


def make_record(
    product_id: str,
    offer_id: str,
    price: str,
    *,
    title: str = "演示商品",
) -> CatalogRecord:
    platform = Platform(offer_id.split(":", maxsplit=1)[0])
    return CatalogRecord(
        canonical_product_id=product_id,
        title=title,
        category_path=["演示类目"],
        attributes={"material": "织物"},
        offer=Offer(
            offer_id=offer_id,
            platform=platform,
            listing_id=offer_id.split(":", maxsplit=1)[1],
            price=Decimal(price),
            currency=Currency.CNY,
            url=f"https://example.com/{offer_id.replace(':', '/')}",
            source_updated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
            provenance=make_provenance(offer_id),
        ),
        provenance=make_provenance(product_id),
    )


def write_records(path: Path, records: list[CatalogRecord]) -> None:
    content = "\n".join(record.model_dump_json() for record in records) + "\n"
    path.write_text(content, encoding="utf-8")


def test_demo_catalog_loads_and_groups_cross_platform_offers() -> None:
    result = LocalCatalog.from_jsonl(DEMO_CATALOG_PATH, strict=True)

    assert result.total_records == 36
    assert result.accepted_records == 36
    assert result.rejected_records == 0
    assert len(result.catalog) == 12
    assert len(result.catalog.offers) == 36
    assert {offer.platform for offer in result.catalog.offers_for("hp-aurora-quietpro")} == {
        Platform.AMAZON,
        Platform.SHOPEE,
        Platform.ALIEXPRESS,
    }


def test_invalid_currency_negative_price_and_empty_id_are_reported(
    catalog_path: Path,
) -> None:
    valid = make_record("product-1", "amazon:offer-1", "100.00")
    invalid_currency = valid.model_copy(deep=True).model_dump(mode="json")
    invalid_currency["offer"]["currency"] = "BTC"
    negative_price = valid.model_copy(deep=True).model_dump(mode="json")
    negative_price["offer"]["price"] = "-1.00"
    empty_id = valid.model_copy(deep=True).model_dump(mode="json")
    empty_id["canonical_product_id"] = ""
    rows = [invalid_currency, negative_price, empty_id]
    catalog_path.write_text("\n".join(_json_dump(row) for row in rows), encoding="utf-8")

    result = LocalCatalog.from_jsonl(catalog_path)

    assert result.accepted_records == 0
    assert result.rejected_records == 3
    assert len(result.catalog) == 0
    with pytest.raises(CatalogValidationError):
        LocalCatalog.from_jsonl(catalog_path, strict=True)


def test_duplicate_offer_id_is_rejected(catalog_path: Path) -> None:
    write_records(
        catalog_path,
        [
            make_record("product-1", "amazon:offer-1", "100.00"),
            make_record("product-2", "amazon:offer-1", "110.00"),
        ],
    )

    result = LocalCatalog.from_jsonl(catalog_path)

    assert result.accepted_records == 1
    assert result.rejected_records == 1
    assert "duplicate offer_id" in result.errors[0].reason


def test_conflicting_master_data_is_rejected(catalog_path: Path) -> None:
    write_records(
        catalog_path,
        [
            make_record("product-1", "amazon:offer-1", "100.00", title="原始标题"),
            make_record("product-1", "shopee:offer-2", "105.00", title="冲突标题"),
        ],
    )

    result = LocalCatalog.from_jsonl(catalog_path)

    assert result.accepted_records == 1
    assert result.rejected_records == 1
    assert "conflicting product data" in result.errors[0].reason


def test_price_more_than_fifty_times_category_median_is_rejected(
    catalog_path: Path,
) -> None:
    write_records(
        catalog_path,
        [
            make_record("product-1", "amazon:offer-1", "100.00"),
            make_record("product-2", "shopee:offer-2", "110.00"),
            make_record("product-3", "aliexpress:offer-3", "100000.00"),
        ],
    )

    result = LocalCatalog.from_jsonl(catalog_path)

    assert result.accepted_records == 2
    assert result.rejected_records == 1
    assert "category median" in result.errors[0].reason


def _json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
