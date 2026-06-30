"""Build bounded Rubric facts from the current Product repository.

The retired regression script builds one prompt-wide table from seed products.
Current evaluation instead resolves only product ids that a case explicitly
expects, retrieves, filters, or displays.  The repository remains the sole
source of Product/SKU facts, and missing ids stay visible so callers can
classify a data-consistency failure instead of silently treating it as no
result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.product import Product
from scripts.eval.rubric_contract import EvaluationEvidence

DEFAULT_MAX_FACT_PRODUCTS = 50
_SUMMARY_PRODUCT_ID_FIELDS = ("hit_product_ids", "filtered_product_ids")


class ProductFactWindowTooLarge(ValueError):
    """The case asks the judge to inspect more products than the bounded window."""


class ProductFactContractError(ValueError):
    """Structured evaluation evidence contains malformed product identifiers."""


class SkuFact(BaseModel):
    """Exact sellable-unit facts restored by ProductRepository."""

    sku_id: str = Field(min_length=1)
    spec: str
    amount_in_minor_units: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    stock: int = Field(ge=0)


class HighlightFact(BaseModel):
    label: str = Field(min_length=1)
    detail: str


class ProductFact(BaseModel):
    """Product-level facts allowed into one Rubric judge prompt."""

    product_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    category: str = Field(min_length=1)
    origin_country: str = Field(min_length=1)
    description: str
    ships_to: list[str]
    highlights: list[HighlightFact]
    skus: list[SkuFact] = Field(min_length=1)


class ProductFactWindow(BaseModel):
    """Auditable, bounded Product facts for exactly one evaluation case."""

    schema_version: Literal["rubric-product-facts-v1"] = "rubric-product-facts-v1"
    source: Literal["ProductRepository"] = "ProductRepository"
    requested_product_ids: list[str]
    products: list[ProductFact]
    missing_product_ids: list[str]
    facts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _deduplicate_product_ids(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ProductFactContractError(
                "product ids in evaluation evidence must be non-empty strings",
            )
        product_id = value.strip()
        if product_id not in seen:
            seen.add(product_id)
            result.append(product_id)
    return result


def _summary_product_ids(summary: dict, field: str) -> list[str]:
    value = summary.get(field, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProductFactContractError(f"tool result_summary.{field} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ProductFactContractError(
            f"tool result_summary.{field} must contain non-empty strings",
        )
    return value


def collect_case_product_ids(
    explicit_product_ids: Iterable[str],
    evidence: EvaluationEvidence,
) -> list[str]:
    """Union expected, retrieved, filtered, and displayed ids in stable order."""

    collected = list(explicit_product_ids)
    for turn in evidence.turns:
        for call in sorted(turn.tool_calls, key=lambda item: item.order):
            for field in _SUMMARY_PRODUCT_ID_FIELDS:
                collected.extend(_summary_product_ids(call.result_summary, field))
        collected.extend(item.product_id for item in turn.displayed_products)
    return _deduplicate_product_ids(collected)


def _product_fact(product: Product) -> ProductFact:
    return ProductFact(
        product_id=product.product_id,
        title=product.title,
        brand=product.brand,
        category=product.category,
        origin_country=product.origin_country,
        description=product.description,
        ships_to=list(product.ships_to),
        highlights=[
            HighlightFact(label=item.label, detail=item.detail)
            for item in product.highlights
        ],
        skus=[
            SkuFact(
                sku_id=sku.sku_id,
                spec=sku.spec,
                amount_in_minor_units=sku.price.amount_in_minor_units,
                currency=sku.price.currency,
                stock=sku.stock,
            )
            for sku in product.skus
        ],
    )


def _facts_digest(products: list[ProductFact]) -> str:
    payload = json.dumps(
        [product.model_dump(mode="json") for product in products],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def build_product_fact_window(
    repository: ProductRepository,
    product_ids: Iterable[str],
    *,
    max_products: int = DEFAULT_MAX_FACT_PRODUCTS,
) -> ProductFactWindow:
    """Restore an ordered fact window without listing or copying the full catalog."""

    if max_products < 1:
        raise ValueError("max_products must be positive")
    requested = _deduplicate_product_ids(product_ids)
    if len(requested) > max_products:
        raise ProductFactWindowTooLarge(
            f"case requested {len(requested)} products; max_products={max_products}",
        )

    restored = await repository.find_by_ids(requested)
    by_id = {product.product_id: product for product in restored}
    ordered = [
        _product_fact(by_id[product_id])
        for product_id in requested
        if product_id in by_id
    ]
    missing = [product_id for product_id in requested if product_id not in by_id]
    return ProductFactWindow(
        requested_product_ids=requested,
        products=ordered,
        missing_product_ids=missing,
        facts_sha256=_facts_digest(ordered),
    )
