"""One canonical, non-transactional document representation for Item search."""

from __future__ import annotations

import json
from typing import Any

from globex_agent.domain.catalog.models import StandardItem
from globex_agent.infrastructure.recall.base import SearchDocument
from globex_agent.infrastructure.recall.embedding import reranker_document_text

# Bump whenever the document field set or serialization semantics changes.
# Persisted indexes with an older version must be rebuilt before use.
ITEM_TEXT_FORMAT_VERSION = "item-text-v5-catalog-schema-v2"

# These exact keys are prohibited at every attributes nesting level.  The list
# intentionally does not contain a generic ``id``: some product attributes use
# it as a meaningful semantic label.  Transactional identifiers are named
# explicitly instead.
_EXCLUDED_ATTRIBUTE_KEYS = frozenset(
    {
        "availability",
        "available",
        "currency",
        "currency_raw",
        "delivery",
        "delivery_info",
        "generated_at",
        "image_url",
        "ingested_at",
        "inventory",
        "is_available",
        "item_id",
        "list_price",
        "max_price",
        "min_price",
        "original_price",
        "original_price_cny",
        "price",
        "price_cny",
        "price_major",
        "product_id",
        "provenance",
        "sale_price",
        "shipping",
        "shipping_cost",
        "shipping_info",
        "ship_to",
        "ships_to",
        "sku",
        "sku_id",
        "source_updated_at",
        "source_url",
        "stock",
        "timestamp",
        "updated_at",
        "url",
        "variant_id",
    }
)


def standard_item_to_search_document(item: StandardItem) -> SearchDocument:
    """Build the exact Item-tower document used at index and rerank time.

    Search text intentionally includes descriptive fields only.  Transactional
    facts stay structured so they can never influence semantic relevance.
    """

    body = " ".join(
        part
        for part in (
            item.brand or "",
            " ".join(item.category_path),
            _json_text(_searchable_attributes(item.attributes, item.materials)),
            _json_text(_searchable_variant_options(item.variants)),
            item.description,
        )
        if part
    )
    return SearchDocument(document_id=item.item_id, title=item.title, body=body)


def standard_item_search_text(item: StandardItem) -> str:
    """Return canonical text for the runtime cross-encoder."""

    document = standard_item_to_search_document(item)
    return reranker_document_text(document.title, document.body)


def _searchable_attributes(attributes: list, materials: list) -> list[dict[str, Any]]:
    """Serialize only typed semantic fields; never price/availability metadata."""
    result = [
        {"code": attr.code, "name": attr.name, "value": attr.value, "unit": attr.unit}
        for attr in attributes
        if attr.code.casefold() not in _EXCLUDED_ATTRIBUTE_KEYS
    ]
    result.extend(
        {"code": material.code, "name": material.name, "part": material.part}
        for material in materials
    )
    return sorted(
        result,
        key=lambda value: (
            str(value.get("code") or "").casefold(),
            str(value.get("name") or "").casefold(),
            json.dumps(value, ensure_ascii=False, sort_keys=True),
        ),
    )


def _clean_searchable_value(value: Any) -> Any:
    """Recursively remove transactional metadata from free-form attributes."""

    if isinstance(value, dict):
        return {
            key: _clean_searchable_value(nested)
            for key, nested in sorted(value.items())
            if isinstance(key, str) and key.casefold() not in _EXCLUDED_ATTRIBUTE_KEYS
        }
    if isinstance(value, list):
        return [_clean_searchable_value(nested) for nested in value]
    if isinstance(value, tuple):
        return [_clean_searchable_value(nested) for nested in value]
    return value


def _searchable_variant_options(variants: list) -> list[dict[str, str]]:
    """Extract canonical option labels from typed variants."""

    options: list[dict[str, str]] = []
    for variant in variants:
        for entry in variant.options:
            option = {"option_name": entry.name, "value": entry.value}
            options.append(option)
    return sorted(
        options,
        key=lambda value: (
            value["option_name"].casefold(),
            value["value"].casefold(),
        ),
    )


def _semantic_text(value: Any) -> str | None:
    """Accept direct option labels/values, never serialize nested metadata."""

    if isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        return text or None
    return None


def _json_text(value: Any) -> str:
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
