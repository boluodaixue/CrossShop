from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from globex_agent.catalog import CatalogValidationError, LocalCatalog
from globex_agent.domain import (
    Currency,
    DataProvenance,
    Platform,
    ProvenanceKind,
    StandardItem,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CATALOG_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


@pytest.fixture
def catalog_path(request: pytest.FixtureRequest):
    """Use the workspace because the Windows temp directory can have restrictive ACLs."""

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


def make_item(
    same_group_id: str,
    item_id: str,
    price: str,
    *,
    title: str = "演示商品",
    rating: float | None = None,
) -> StandardItem:
    platform = Platform(item_id.split(":", maxsplit=1)[0])
    timestamp = datetime(2026, 8, 12, tzinfo=timezone.utc)
    return StandardItem(
        item_id=item_id,
        same_group_id=same_group_id,
        platform=platform,
        title=title,
        category_path=["演示类目"],
        price_cny=Decimal(price),
        currency_raw=Currency.CNY,
        rating=rating,
        attributes={"material": "织物"},
        source_updated_at=timestamp,
        ingested_at=timestamp,
        url=f"https://example.com/{item_id.replace(':', '/')}",
        provenance=make_provenance(item_id),
    )


def write_items(path: Path, items: list[StandardItem]) -> None:
    content = "\n".join(item.model_dump_json() for item in items) + "\n"
    path.write_text(content, encoding="utf-8")


def test_demo_catalog_keeps_cross_platform_items_separate() -> None:
    result = LocalCatalog.from_jsonl(DEMO_CATALOG_PATH, strict=True)

    assert result.total_records == 36
    assert result.accepted_records == 36
    assert result.rejected_records == 0
    assert len(result.catalog) == 36
    assert result.catalog.group_count == 12
    assert {item.platform for item in result.catalog.items_for_group("hp-aurora-quietpro")} == {
        Platform.AMAZON,
        Platform.SHOPEE,
        Platform.ALIEXPRESS,
    }


def test_invalid_currency_negative_price_and_empty_id_are_reported(
    catalog_path: Path,
) -> None:
    valid = make_item("product-1", "amazon:item-1", "100.00")
    invalid_currency = valid.model_copy(deep=True).model_dump(mode="json")
    invalid_currency["currency_raw"] = "BTC"
    negative_price = valid.model_copy(deep=True).model_dump(mode="json")
    negative_price["price_cny"] = "-1.00"
    empty_id = valid.model_copy(deep=True).model_dump(mode="json")
    empty_id["item_id"] = ""
    rows = [invalid_currency, negative_price, empty_id]
    catalog_path.write_text("\n".join(_json_dump(row) for row in rows), encoding="utf-8")

    result = LocalCatalog.from_jsonl(catalog_path)

    assert result.accepted_records == 0
    assert result.rejected_records == 3
    assert len(result.catalog) == 0
    with pytest.raises(CatalogValidationError):
        LocalCatalog.from_jsonl(catalog_path, strict=True)


def test_duplicate_item_id_is_rejected(catalog_path: Path) -> None:
    write_items(
        catalog_path,
        [
            make_item("product-1", "amazon:item-1", "100.00"),
            make_item("product-2", "amazon:item-1", "110.00"),
        ],
    )

    result = LocalCatalog.from_jsonl(catalog_path)

    assert result.accepted_records == 1
    assert result.rejected_records == 1
    assert "duplicate item_id" in result.errors[0].reason


def test_same_group_id_associates_but_does_not_merge_rows(catalog_path: Path) -> None:
    write_items(
        catalog_path,
        [
            make_item("product-1", "amazon:item-1", "100.00", title="Amazon 标题"),
            make_item("product-1", "shopee:item-2", "105.00", title="Shopee 标题"),
        ],
    )

    result = LocalCatalog.from_jsonl(catalog_path, strict=True)

    assert result.accepted_records == 2
    assert len(result.catalog) == 2
    assert len(result.catalog.items_for_group("product-1")) == 2


def test_same_platform_same_group_keeps_higher_rated_item(catalog_path: Path) -> None:
    write_items(
        catalog_path,
        [
            make_item("product-1", "amazon:item-1", "100.00", rating=4.2),
            make_item("product-1", "amazon:item-2", "105.00", rating=4.8),
        ],
    )

    result = LocalCatalog.from_jsonl(catalog_path, strict=True)

    assert result.accepted_records == 1
    assert result.deduplicated_records == 1
    assert result.catalog.items[0].item_id == "amazon:item-2"


def test_same_platform_parallel_languages_remain_separate(catalog_path: Path) -> None:
    english = make_item("product-1", "amazon:item-en", "100.00")
    chinese = make_item("product-1", "amazon:item-zh", "100.00")
    english = english.model_copy(update={"language": "en", "title": "Travel bag"})
    chinese = chinese.model_copy(update={"language": "zh", "title": "旅行包"})
    write_items(catalog_path, [english, chinese])

    result = LocalCatalog.from_jsonl(catalog_path, strict=True)

    assert result.accepted_records == 2
    assert result.deduplicated_records == 0
    assert {item.language for item in result.catalog.items} == {"en", "zh"}


def test_price_more_than_fifty_times_category_median_is_rejected(
    catalog_path: Path,
) -> None:
    write_items(
        catalog_path,
        [
            make_item("product-1", "amazon:item-1", "100.00"),
            make_item("product-2", "shopee:item-2", "110.00"),
            make_item("product-3", "aliexpress:item-3", "100000.00"),
        ],
    )

    result = LocalCatalog.from_jsonl(catalog_path)

    assert result.accepted_records == 2
    assert result.rejected_records == 1
    assert "category median" in result.errors[0].reason


def _json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)
