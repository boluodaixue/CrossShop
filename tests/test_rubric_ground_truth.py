"""Offline and real-catalog checks for bounded Rubric Product facts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.domain.catalog.money import Money
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.product import Product, ProductHighlight
from app.domain.catalog.sku import Sku
from app.infrastructure.persistence.jsonl_product_repository import (
    JsonlProductRepository,
)
from scripts.eval.rubric_contract import (
    DisplayedProductEvidence,
    EvaluationEvidence,
    PreferenceStateEvidence,
    StructuredStateEvidence,
    ToolCallEvidence,
    TurnEvidence,
)


def _state() -> StructuredStateEvidence:
    empty = PreferenceStateEvidence(count=0, content_hash="0" * 64)
    return StructuredStateEvidence(
        preference_before=empty,
        preference_after=empty,
    )


from scripts.eval.rubric_ground_truth import (
    ProductFactContractError,
    ProductFactWindowTooLarge,
    build_product_fact_window,
    collect_case_product_ids,
)


def _product(product_id: str, amount: int = 8900) -> Product:
    return Product(
        product_id=product_id,
        title=f"商品 {product_id}",
        brand="Globex",
        category="户外运动",
        origin_country="CN",
        description="防水耐用",
        highlights=[ProductHighlight(label="防护", detail="IPX5")],
        ships_to=["CN", "US"],
        skus=[
            Sku(
                sku_id=f"{product_id}-S1",
                spec="军绿",
                price=Money.of(amount, "CNY"),
                stock=12,
            ),
        ],
    )


def _evidence() -> EvaluationEvidence:
    return EvaluationEvidence(
        case_id="bounded-facts",
        session_id_hash="session-hash",
        turns=[
            TurnEvidence(
                turn_index=1,
                user_input="推荐露营灯",
                route="main.direct",
                tool_calls=[
                    ToolCallEvidence(
                        order=1,
                        call_id="call-1",
                        agent="main",
                        tool="product_search_tool",
                        result_summary={
                            "hit_product_ids": ["P1008", "P1004"],
                            "filtered_product_ids": ["P1005"],
                        },
                    ),
                ],
                displayed_products=[
                    DisplayedProductEvidence(
                        rank=1,
                        product_id="P1004",
                        platform="globex_reference",
                    ),
                ],
                structured_state=_state(),
                final_text="推荐一款露营灯。",
            ),
        ],
    )


def test_collects_expected_retrieved_filtered_and_displayed_ids_in_order() -> None:
    assert collect_case_product_ids(["P1001", "P1008"], _evidence()) == [
        "P1001",
        "P1008",
        "P1004",
        "P1005",
    ]


@pytest.mark.asyncio
async def test_builds_ordered_window_and_reports_missing_ids() -> None:
    repository = AsyncMock(spec=ProductRepository)
    repository.find_by_ids.return_value = [_product("P1008"), _product("P1004", 21900)]

    window = await build_product_fact_window(
        repository,
        ["P1008", "missing", "P1004", "P1008"],
    )

    repository.find_by_ids.assert_awaited_once_with(["P1008", "missing", "P1004"])
    assert window.requested_product_ids == ["P1008", "missing", "P1004"]
    assert [product.product_id for product in window.products] == ["P1008", "P1004"]
    assert window.missing_product_ids == ["missing"]
    assert window.products[0].skus[0].amount_in_minor_units == 8900
    assert len(window.facts_sha256) == 64


@pytest.mark.asyncio
async def test_fact_window_limit_fails_instead_of_silent_truncation() -> None:
    repository = AsyncMock(spec=ProductRepository)

    with pytest.raises(ProductFactWindowTooLarge, match="max_products=2"):
        await build_product_fact_window(
            repository,
            ["P1", "P2", "P3"],
            max_products=2,
        )

    repository.find_by_ids.assert_not_awaited()


def test_rejects_malformed_product_ids_in_tool_summary() -> None:
    evidence = _evidence()
    evidence.turns[0].tool_calls[0].result_summary["hit_product_ids"] = "P1008"

    with pytest.raises(ProductFactContractError, match="must be a list"):
        collect_case_product_ids([], evidence)


@pytest.mark.asyncio
async def test_real_catalog_facts_come_from_current_jsonl_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    repository = JsonlProductRepository(root / "data" / "processed" / "catalogs-v2")

    window = await build_product_fact_window(
        repository,
        ["P1008", "P1004", "not-in-current-catalog"],
    )

    assert [product.product_id for product in window.products] == ["P1008", "P1004"]
    assert window.missing_product_ids == ["not-in-current-catalog"]
    lumen = window.products[0]
    assert lumen.title == "LumenGo 便携露营灯 可充电"
    assert lumen.skus[0].amount_in_minor_units == 8900
    assert lumen.skus[0].currency == "CNY"
    assert lumen.skus[0].stock == 150
    assert "US" in lumen.ships_to
