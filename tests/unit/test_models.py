from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from globex_agent.domain import (
    Currency,
    DataProvenance,
    LandedCost,
    Platform,
    ProvenanceKind,
    SearchRequest,
    StandardItem,
    UserProfile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_USERS_PATH = PROJECT_ROOT / "data" / "demo" / "users.jsonl"
DEMO_QUERIES_PATH = PROJECT_ROOT / "data" / "demo" / "queries.jsonl"


def make_provenance(record_id: str = "demo-record") -> DataProvenance:
    return DataProvenance(
        kind=ProvenanceKind.SYNTHETIC,
        source="unit test",
        source_record_id=record_id,
        generated_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
    )


def make_standard_item(**overrides: object) -> StandardItem:
    values: dict[str, object] = {
        "item_id": "amazon:demo-001",
        "same_group_id": "demo-product",
        "platform": Platform.AMAZON,
        "title": "演示商品",
        "category_path": ["演示类目"],
        "price_cny": Decimal("199.00"),
        "currency_raw": Currency.CNY,
        "source_updated_at": datetime(2026, 8, 12, tzinfo=timezone.utc),
        "ingested_at": datetime(2026, 8, 12, tzinfo=timezone.utc),
        "url": "https://example.com/amazon/demo-001",
        "provenance": make_provenance("amazon:demo-001"),
    }
    values.update(overrides)
    return StandardItem.model_validate(values)


def test_standard_item_accepts_a_valid_platform_item() -> None:
    item = make_standard_item()

    assert item.item_id == "amazon:demo-001"
    assert item.same_group_id == "demo-product"
    assert item.price_cny == Decimal("199.00")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price_cny", Decimal("-1.00")),
        ("currency_raw", "BTC"),
        ("platform", "unknown"),
        ("item_id", ""),
    ],
)
def test_standard_item_rejects_invalid_core_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_standard_item(**{field: value})


def test_item_id_must_use_platform_prefix() -> None:
    with pytest.raises(ValidationError, match="item_id must start"):
        make_standard_item(item_id="shopee:demo-001", platform=Platform.AMAZON)


def test_original_price_cannot_be_lower_than_current_price() -> None:
    with pytest.raises(ValidationError, match="original_price_cny cannot be lower"):
        make_standard_item(
            price_cny=Decimal("200.00"),
            original_price_cny=Decimal("199.00"),
        )


def test_same_group_id_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        make_standard_item(same_group_id=" ")


def test_landed_cost_requires_exact_component_sum() -> None:
    with pytest.raises(ValidationError, match="landed_cny must equal"):
        LandedCost(
            item_id="amazon:demo-001",
            platform=Platform.AMAZON,
            price_cny=Decimal("100.00"),
            shipping_cny=Decimal("10.00"),
            duty_cny=Decimal("5.00"),
            landed_cny=Decimal("114.00"),
            eta_days=12,
            duty_tier="标准",
            same_group_id="demo-product",
            title="演示商品",
            category_path=["演示类目"],
            duty_rate=Decimal("0.13"),
            rule_version="unit-test-v1",
        )


def test_provenance_timestamp_must_include_timezone() -> None:
    with pytest.raises(ValidationError, match="must include a timezone"):
        DataProvenance(
            kind=ProvenanceKind.SYNTHETIC,
            source="unit test",
            generated_at=datetime(2026, 8, 12),
        )


def test_demo_users_follow_the_profile_contract() -> None:
    profiles = [
        UserProfile.model_validate_json(line)
        for line in DEMO_USERS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert [profile.user_id for profile in profiles] == [
        "demo-user-001",
        "demo-user-002",
        "demo-user-003",
    ]
    assert all(":" in profile.positive_item_ids[0] for profile in profiles)
    assert all(profile.provenance.kind is ProvenanceKind.SYNTHETIC for profile in profiles)


def test_demo_queries_follow_the_search_request_contract() -> None:
    requests = [
        SearchRequest.model_validate_json(line)
        for line in DEMO_QUERIES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(requests) == 6
    assert all(request.currency is Currency.CNY for request in requests)
    assert all(request.top_k == 3 for request in requests)
