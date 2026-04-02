from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from globex_agent.domain import (
    Currency,
    DataProvenance,
    Offer,
    Platform,
    ProvenanceKind,
    SearchRequest,
    ShippingQuote,
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


def make_offer(**overrides: object) -> Offer:
    values: dict[str, object] = {
        "offer_id": "amazon:demo-001",
        "platform": Platform.AMAZON,
        "listing_id": "demo-001",
        "price": Decimal("199.00"),
        "currency": Currency.CNY,
        "shipping_fee": Decimal("0.00"),
        "url": "https://example.com/amazon/demo-001",
        "source_updated_at": datetime(2026, 8, 12, tzinfo=timezone.utc),
        "provenance": make_provenance("amazon:demo-001"),
    }
    values.update(overrides)
    return Offer.model_validate(values)


def test_standard_item_accepts_a_valid_offer() -> None:
    item = StandardItem(
        canonical_product_id="demo-product",
        title="演示商品",
        category_path=["演示类目"],
        offers=[make_offer()],
        provenance=make_provenance("demo-product"),
    )

    assert item.offers[0].price == Decimal("199.00")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price", Decimal("-1.00")),
        ("currency", "BTC"),
        ("platform", "unknown"),
        ("offer_id", ""),
    ],
)
def test_offer_rejects_invalid_core_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_offer(**{field: value})


def test_original_price_cannot_be_lower_than_current_price() -> None:
    with pytest.raises(ValidationError, match="original_price cannot be lower"):
        make_offer(price=Decimal("200.00"), original_price=Decimal("199.00"))


def test_product_id_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        StandardItem(
            canonical_product_id=" ",
            title="演示商品",
            category_path=["演示类目"],
            offers=[make_offer()],
            provenance=make_provenance(),
        )


def test_shipping_quote_requires_exact_component_sum() -> None:
    with pytest.raises(ValidationError, match="landed_price must equal"):
        ShippingQuote(
            canonical_product_id="demo-product",
            offer_id="amazon:demo-001",
            platform=Platform.AMAZON,
            item_price=Decimal("100.00"),
            shipping_fee=Decimal("10.00"),
            tax_fee=Decimal("5.00"),
            landed_price=Decimal("114.00"),
            currency=Currency.CNY,
            eta_days=12,
            tax_rate=Decimal("0.13"),
            tax_tier="标准",
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
