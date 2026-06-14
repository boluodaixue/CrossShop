"""Versioned product evidence shared by search output and audit/evaluation.

``ProductCard`` is intentionally a small model-facing projection.  This module
owns the richer, internal evidence contract and the one-way projection from a
``StandardItem``.  Keeping the two projections together prevents prices,
variant labels, and availability from being assembled by separate callers.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from globex_agent.domain.catalog.models import (
    AvailabilityStatus,
    Currency,
    ProductAttribute,
    StandardItem,
    StandardItemVariant,
)

PRODUCT_FACT_SCHEMA_VERSION = "product-fact-snapshot-v1"
_SNAPSHOT_MAX_DEPTH = 8
_SNAPSHOT_MAX_LIST_ITEMS = 20
_SNAPSHOT_MAX_STRING_CHARS = 2_000
_FULL_EVIDENCE_LIST_KEYS = {
    "variants",
    "options",
    "highlights",
    "category_path",
    "ships_to",
    "exposed_fields",
}
_INTERNAL_TRACE_ATTRIBUTE_CODES = frozenset({"source_attributes", "source_tag"})


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InventoryFact(EvidenceModel):
    status: AvailabilityStatus
    quantity: int | None = Field(default=None, ge=0)
    source: str


class VariantFact(EvidenceModel):
    variant_id: str
    source_variant_id: str | None = None
    options: list[dict[str, str | None]]
    display_name: str
    price_major: Decimal | None = None
    currency: Currency = Currency.CNY
    price_source: str
    inventory: InventoryFact


class ProductFactSnapshot(EvidenceModel):
    """Internal, versioned evidence DTO for one normalized product.

    ``exposed_fields`` and ``exposed_facts`` describe the exact complete
    projection that was made available to the model and is persisted for
    audit.  They are descriptive data, not a field-level evidence catalog.
    """

    evidence_id: str
    schema_version: str = PRODUCT_FACT_SCHEMA_VERSION
    content_hash: str
    item_id: str
    same_group_id: str
    platform: str
    locale: str | None
    title: str
    brand: str | None
    category_path: list[str]
    store: str | None
    origin_country: str
    origin_country_source: str
    ships_to: list[str]
    ships_to_source: str
    product_price_major: Decimal | None
    product_price_source: str
    currency: Currency | None
    inventory: InventoryFact
    variants: list[VariantFact]
    highlights: list[str]
    provenance: dict[str, Any]
    exposed_fields: list[str]
    exposed_facts: dict[str, Any]

    def to_persisted_dict(self) -> dict[str, Any]:
        """Return the schema-aware audit/eval whitelist, without raw source."""

        return sanitize_snapshot_payload(self.model_dump(mode="python"))

    def evidence_ref(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "content_hash": self.content_hash,
            "schema_version": self.schema_version,
        }


def _is_internal_trace_attribute(attribute: ProductAttribute) -> bool:
    """Return whether an attribute is source tracing metadata, not a user fact."""

    return attribute.code.casefold() in _INTERNAL_TRACE_ATTRIBUTE_CODES


def build_product_fact_snapshot(item: StandardItem) -> ProductFactSnapshot:
    """Build one deterministic snapshot from one ``StandardItem``."""

    store = _attribute_text(item, {"shop_name", "shop", "store", "store_name", "店铺"})
    origin_country, origin_country_source = _derived_origin(item)
    ships_to, ships_to_source = _derived_ships_to(item)
    product_quantity = _attribute_int(
        item,
        {"stock", "inventory", "stock_quantity", "inventory_quantity", "库存"},
    )
    product_inventory = InventoryFact(
        status=item.availability,
        quantity=product_quantity,
        source="standard_item.availability"
        if product_quantity is None
        else "standard_item.attributes.inventory",
    )
    variants = [_variant_fact(variant) for variant in item.variants]
    highlights = [
        f"{attribute.name}: {attribute.value}{(' ' + attribute.unit) if attribute.unit else ''}"
        for attribute in item.attributes
        if not _is_internal_trace_attribute(attribute)
    ]
    exposed_fields = [
        "item_id",
        "title",
        "brand",
        "category",
        "category_path",
        "store",
        "origin_country",
        "origin_country_source",
        "ships_to",
        "ships_to_source",
        "product_price_major",
        "product_price_source",
        "price_min_major",
        "price_max_major",
        "currency",
        "inventory.status",
        "highlights",
        "variants[].variant_id",
        "variants[].options",
        "variants[].display_name",
        "variants[].price_major",
        "variants[].currency",
        "variants[].inventory.status",
    ]
    exposed_facts = {
        "item_id": item.item_id,
        "title": item.title,
        "brand": item.brand,
        "category": item.category_path[-1] if item.category_path else "",
        "category_path": list(item.category_path),
        "store": store,
        "origin_country": origin_country,
        "ships_to": ships_to,
        "price_major": _decimal_json(item.price_cny),
        "price_min_major": _variant_price_range(item),
        "price_max_major": _variant_price_range(item, maximum=True),
        "price_source": item.price_source.value,
        "currency": item.currency_raw.value if item.currency_raw else None,
        "availability": item.availability.value,
        "highlights": list(highlights),
        "variants": [
            {
                "variant_id": variant.variant_id,
                "options": list(variant.options),
                "display_name": variant.display_name,
                "price_major": _decimal_float(variant.price_major),
                "currency": variant.currency.value,
                "availability": variant.inventory.status.value,
            }
            for variant in variants
        ],
    }
    payload = {
        "schema_version": PRODUCT_FACT_SCHEMA_VERSION,
        "item_id": item.item_id,
        "same_group_id": item.same_group_id,
        "platform": item.platform.value,
        "locale": item.locale.value if item.locale else None,
        "title": item.title,
        "brand": item.brand,
        "category_path": list(item.category_path),
        "store": store,
        "origin_country": origin_country,
        "origin_country_source": origin_country_source,
        "ships_to": list(ships_to),
        "ships_to_source": ships_to_source,
        "product_price_major": item.price_cny,
        "product_price_source": item.price_source.value,
        "currency": item.currency_raw,
        "inventory": product_inventory,
        "variants": variants,
        "highlights": highlights,
        "provenance": item.provenance.model_dump(mode="json"),
        "exposed_fields": exposed_fields,
        "exposed_facts": exposed_facts,
    }
    canonical = _canonical_json(payload)
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ProductFactSnapshot(
        evidence_id=f"product-evidence:{content_hash[:24]}",
        content_hash=content_hash,
        **payload,
    )


def product_card_from_snapshot(
    snapshot: ProductFactSnapshot,
    *,
    score: float,
    landed_price: dict[str, Any] | None,
    variant_landed_prices: dict[str, dict[str, Any]] | None = None,
    price_min_major: float | None,
    price_max_major: float | None,
    requires_variant_selection: bool,
    matching_variant_ids: list[str],
    warnings: list[str],
):
    """Project the exact same snapshot into the compact model-facing card."""

    # Local import avoids a module cycle: the use case owns the public card
    # type, while this module owns the evidence DTO.
    from globex_agent.application.usecases.catalog_search import ProductCard

    return ProductCard(
        item_id=snapshot.item_id,
        title=snapshot.title,
        brand=snapshot.brand or "",
        category=snapshot.category_path[-1] if snapshot.category_path else "",
        category_path=list(snapshot.category_path),
        store=snapshot.store,
        origin_country=snapshot.origin_country,
        origin_country_source=snapshot.origin_country_source,
        ships_to=list(snapshot.ships_to),
        ships_to_source=snapshot.ships_to_source,
        price_major=_decimal_float(snapshot.product_price_major),
        price_source=snapshot.product_price_source,
        currency=snapshot.currency.value if snapshot.currency else "CNY",
        availability=snapshot.inventory.status.value,
        highlights=list(snapshot.highlights),
        variants=[
            _variant_card(variant, (variant_landed_prices or {}).get(variant.variant_id))
            for variant in snapshot.variants
        ],
        score=score,
        landed_price=landed_price,
        price_min_major=price_min_major,
        price_max_major=price_max_major,
        requires_variant_selection=requires_variant_selection,
        matching_variant_ids=matching_variant_ids,
        warnings=warnings,
    )


def sanitize_snapshot_payload(value: Any) -> Any:
    """Persist the complete model-visible ProductFactSnapshot whitelist.

    The product search projection and this payload are both built from the
    same snapshot.  PII is still redacted and provenance remains bounded, but
    model-visible product fields are never reduced by an audit-size fallback.
    """

    if isinstance(value, ProductFactSnapshot):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        allowed = {
            "evidence_id",
            "schema_version",
            "content_hash",
            "item_id",
            "same_group_id",
            "platform",
            "locale",
            "title",
            "brand",
            "category_path",
            "store",
            "origin_country",
            "origin_country_source",
            "ships_to",
            "ships_to_source",
            "product_price_major",
            "product_price_source",
            "currency",
            "inventory",
            "variants",
            "highlights",
            "provenance",
            "exposed_fields",
            "exposed_facts",
        }
        return _sanitize_snapshot_pii(
            {key: _jsonable(value[key]) for key in allowed if key in value},
            max_string_chars=None,
        )
    return value


def _variant_card(
    variant: VariantFact,
    landed_price: dict[str, Any] | None,
) -> dict[str, Any]:
    card = {
        "variant_id": variant.variant_id,
        "options": list(variant.options),
        "display_name": variant.display_name,
        "price_major": _decimal_float(variant.price_major),
        "currency": variant.currency.value,
        "availability": variant.inventory.status.value,
    }
    if landed_price is not None:
        card["landed_price"] = landed_price
    return card


def _variant_fact(variant: StandardItemVariant) -> VariantFact:
    return VariantFact(
        variant_id=variant.variant_id,
        source_variant_id=variant.source_variant_id,
        options=[
            {"code": option.code, "name": option.name, "value": option.value}
            for option in variant.options
        ],
        display_name=" / ".join(f"{option.name}: {option.value}" for option in variant.options),
        price_major=variant.price_cny,
        price_source=variant.price_source.value,
        inventory=InventoryFact(
            status=variant.availability,
            quantity=None,
            source="standard_item.variant.availability",
        ),
    )


def _derived_origin(item: StandardItem) -> tuple[str, str]:
    explicit = next(
        (
            attribute.value.strip()
            for attribute in item.attributes
            if attribute.code == "origin_country"
            and isinstance(attribute.value, str)
            and attribute.value.strip()
        ),
        None,
    )
    if explicit:
        return explicit, "explicit"
    locale_country = {"cn": "CN", "jp": "JP"}.get(item.locale.value if item.locale else "")
    if locale_country:
        return locale_country, "derived"
    return (
        {
            "amazon": "US",
            "taobao": "CN",
            "shopee": "SG",
            "aliexpress": "CN",
            "ebay": "US",
        }.get(item.platform.value, ""),
        "derived",
    )


def _derived_ships_to(item: StandardItem) -> tuple[list[str], str]:
    if item.ships_to is not None:
        return list(item.ships_to), "explicit"
    locale_ships = {"cn": ["CN"], "jp": ["JP"]}.get(item.locale.value if item.locale else "")
    if locale_ships:
        return locale_ships, "derived"
    return {
        "amazon": ["US"],
        "taobao": ["CN"],
        "shopee": ["SG"],
        "aliexpress": ["CN", "US", "EU"],
        "ebay": ["US"],
    }.get(item.platform.value, []), "derived"


def _variant_price_range(
    item: StandardItem,
    *,
    maximum: bool = False,
) -> float | None:
    prices = [
        float(variant.price_cny)
        for variant in item.variants
        if variant.price_cny is not None and variant.availability is AvailabilityStatus.AVAILABLE
    ]
    if not prices:
        return None
    return max(prices) if maximum else min(prices)


def _attribute_text(item: StandardItem, names: set[str]) -> str | None:
    for attribute in item.attributes:
        if (
            (attribute.code.casefold() in names or attribute.name.casefold() in names)
            and isinstance(attribute.value, str)
            and attribute.value.strip()
        ):
            return attribute.value.strip()
    return None


def _attribute_int(item: StandardItem, names: set[str]) -> int | None:
    for attribute in item.attributes:
        if attribute.code.casefold() in names or attribute.name.casefold() in names:
            try:
                value = int(attribute.value)
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None
    return None


def _decimal_float(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def _decimal_json(value: Decimal | None) -> float | None:
    return _decimal_float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return value.value
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_PII_KEYS = {"phone", "address", "postal_code", "recipient", "recipient_name"}
_PHONE_RE = re.compile(r"(?<!\d)1\d{10}(?!\d)")


def _sanitize_snapshot_pii(
    value: Any,
    *,
    key: str = "",
    depth: int = 0,
    max_list_items: int = _SNAPSHOT_MAX_LIST_ITEMS,
    max_string_chars: int | None = _SNAPSHOT_MAX_STRING_CHARS,
) -> Any:
    if key.casefold() in _PII_KEYS:
        return "[redacted]"
    if depth >= _SNAPSHOT_MAX_DEPTH:
        return "[omitted: snapshot depth limit]"
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_snapshot_pii(
                child,
                key=str(child_key),
                depth=depth + 1,
                max_list_items=max_list_items,
                max_string_chars=(
                    _SNAPSHOT_MAX_STRING_CHARS
                    if str(child_key) == "provenance"
                    else max_string_chars
                ),
            )
            for child_key, child in value.items()
        }
    if isinstance(value, list):
        limit = None if key in _FULL_EVIDENCE_LIST_KEYS else max_list_items
        children = value if limit is None else value[:limit]
        return [
            _sanitize_snapshot_pii(
                child,
                key=key,
                depth=depth + 1,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for child in children
        ]
    if isinstance(value, str):
        sanitized = _PHONE_RE.sub("[redacted]", value)
        return sanitized if max_string_chars is None else sanitized[:max_string_chars]
    return value
