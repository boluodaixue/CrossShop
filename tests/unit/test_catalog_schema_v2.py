from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from globex_agent.domain.catalog.constraints import ConstraintEvaluator
from globex_agent.domain.catalog.models import (
    AvailabilityStatus,
    DataProvenance,
    MaterialComponent,
    PriceSource,
    ProvenanceKind,
    StandardItem,
    StandardItemVariant,
    VariantOption,
)
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec


def _provenance() -> DataProvenance:
    return DataProvenance(
        kind=ProvenanceKind.SYNTHETIC,
        source="test",
        generated_at=datetime.now(timezone.utc),
    )


def _item(**overrides) -> StandardItem:
    values = {
        "item_id": "taobao:cn:test",
        "same_group_id": "test",
        "platform": "taobao",
        "title": "测试商品",
        "category_path": ["测试"],
        "availability": "available",
        "price_cny": Decimal("10"),
        "price_source": "observed",
        "ingested_at": datetime.now(timezone.utc),
        "provenance": _provenance(),
    }
    values.update(overrides)
    return StandardItem(**values)


def test_variant_prices_are_only_source_for_spec_items() -> None:
    variant = StandardItemVariant(
        variant_id="v1",
        options=[VariantOption(name="颜色", value="黑")],
        price_cny=Decimal("9"),
        price_source=PriceSource.OBSERVED,
        availability=AvailabilityStatus.AVAILABLE,
    )
    item = _item(
        variants=[variant],
        price_cny=None,
        price_source=PriceSource.UNAVAILABLE,
    )
    assert item.price_cny is None
    assert item.variants[0].price_cny == Decimal("9.00")

    with pytest.raises(ValidationError):
        _item(variants=[variant], price_cny=Decimal("9"))


def test_unknown_availability_is_not_orderable_or_budget_eligible() -> None:
    item = _item(availability=AvailabilityStatus.UNKNOWN)
    decision = ConstraintEvaluator().evaluate(item, ProductSearchSpec("测试"))
    assert not decision.eligible
    assert decision.reason == "item_availability_unknown"


def test_variant_budget_returns_matching_ids_only() -> None:
    variants = [
        StandardItemVariant(
            variant_id="cheap",
            options=[VariantOption(name="套餐", value="基础")],
            price_cny=Decimal("9"),
            price_source=PriceSource.OBSERVED,
            availability=AvailabilityStatus.AVAILABLE,
        ),
        StandardItemVariant(
            variant_id="expensive",
            options=[VariantOption(name="套餐", value="高级")],
            price_cny=Decimal("19"),
            price_source=PriceSource.OBSERVED,
            availability=AvailabilityStatus.AVAILABLE,
        ),
    ]
    item = _item(variants=variants, price_cny=None, price_source=PriceSource.UNAVAILABLE)
    decision = ConstraintEvaluator().evaluate(item, ProductSearchSpec("测试", price_max_major=10))
    assert decision.eligible
    assert decision.matching_variant_ids == ["cheap"]


def test_material_constraint_uses_normalized_code() -> None:
    item = _item(materials=[MaterialComponent(code="cotton", name="棉")])
    assert ConstraintEvaluator().evaluate(
        item, ProductSearchSpec("测试", material_include=("cotton",))
    ).eligible
    assert not ConstraintEvaluator().evaluate(
        item, ProductSearchSpec("测试", material_exclude=("cotton",))
    ).eligible
