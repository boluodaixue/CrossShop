"""ItemSearch adapter over a replaceable retrieval backend."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import (
    Candidate,
    Currency,
    ItemSearchOutput,
    MarketLocale,
    Platform,
    ResultStatus,
    StandardItem,
    ToolIssue,
)
from globex_agent.infrastructure.recall import (
    CanonicalDedupSearchBackend,
    CanonicalListing,
    KeywordSearchBackend,
    ListingLanguage,
    PairReranker,
    PreferredLanguageSearchBackend,
    RerankedSearchBackend,
    SearchBackend,
    SearchDocument,
    standard_item_to_search_document,
)
from globex_agent.tools.constraints import SUPPORTED_HARD_CONSTRAINTS


def search_items(
    catalog: LocalCatalog,
    query: str,
    platform: Platform,
    top_k: int = 20,
    hard_constraints: dict[str, Any] | None = None,
    *,
    locale: MarketLocale | None = None,
    backend: SearchBackend | None = None,
    reranker: PairReranker | None = None,
    listing_languages: Mapping[str, ListingLanguage] | None = None,
) -> ItemSearchOutput:
    """Search one platform without using user-profile recall signals.

    ``catalog`` and the optional ANN backend/reranker are dependency-injection
    values; none appears in the model-facing schema. An injected backend is
    canonicalized before an optional reranker, then the preferred display
    language is selected without changing relevance scores or canonical order.
    Without injection the first-stage teaching example remains BM25-only.
    """

    top_k = max(1, min(top_k, 50))
    constraints = hard_constraints or {}
    issues = [
        ToolIssue(
            code="unsupported_constraint",
            message=f"ItemSearch 无法识别硬约束 {key}，交由 ItemPicker 拒绝处理",
            subject_id=key,
        )
        for key in sorted(constraints)
        if key not in SUPPORTED_HARD_CONSTRAINTS
    ]

    platform_items = tuple(
        item
        for item in catalog.items
        if (
            item.platform is platform
            and item.is_available
            and (locale is None or item.locale is locale)
        )
    )
    documents = [item_to_search_document(item) for item in platform_items]
    if backend is None:
        search_backend: SearchBackend = KeywordSearchBackend(documents)
    else:
        listings = [
            CanonicalListing(
                document_id=item.item_id,
                canonical_id=item.same_group_id,
                language=_listing_language(item, listing_languages),
            )
            for item in platform_items
        ]
        relevance_backend: SearchBackend = CanonicalDedupSearchBackend(
            backend,
            listings,
        )
        if reranker is not None:
            relevance_backend = RerankedSearchBackend(
                relevance_backend,
                documents,
                reranker,
                candidate_k=100,
            )
        search_backend = PreferredLanguageSearchBackend(
            relevance_backend,
            listings,
            preferred_language=_text_language(query),
        )
    recalled = search_backend.search(query, top_k=top_k)
    item_by_id = {item.item_id: item for item in platform_items}
    candidates = [
        _to_candidate(item_by_id[hit.document_id])
        for hit in recalled.hits
        if hit.document_id in item_by_id
    ]

    if not candidates:
        status = ResultStatus.NO_RESULTS
        issues.append(
            ToolIssue(
                code="no_search_match",
                message=f"{platform.value} 本地商品目录没有匹配该查询的候选",
                subject_id=platform.value,
            )
        )
    elif issues:
        status = ResultStatus.PARTIAL
    else:
        status = ResultStatus.OK

    return ItemSearchOutput(
        platform=platform,
        locale=locale,
        candidates=candidates,
        total_recall=recalled.total_recall,
        truncated=recalled.truncated,
        status=status,
        issues=issues,
    )


def item_to_search_document(item: StandardItem) -> SearchDocument:
    """Map the stable product contract to the Item-tower/search text contract."""

    return standard_item_to_search_document(item)


def _listing_language(
    item: StandardItem,
    overrides: Mapping[str, ListingLanguage] | None,
) -> ListingLanguage:
    if overrides is not None and item.item_id in overrides:
        return overrides[item.item_id]
    if item.language is not None:
        return item.language
    return _text_language(f"{item.title} {item.description}")


def _text_language(text: str) -> ListingLanguage:
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    if re.search(r"[áéíóúüñ¿¡]", text.casefold()):
        return "es"
    return "en"


def _flatten_attribute_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _to_candidate(item: StandardItem) -> Candidate:
    return Candidate(
        item_id=item.item_id,
        platform=item.platform,
        locale=item.locale,
        title=item.title,
        price=item.price_cny,
        currency=Currency.CNY if item.price_cny is not None else item.currency_raw,
        price_source=item.price_source,
        rating=item.rating,
        sales=None,
        image_url=None,
        attributes={
            key: _flatten_attribute_value(value)
            for key, value in item.attributes.items()
        },
        same_group_id=item.same_group_id,
        brand=item.brand,
        category_path=item.category_path,
    )
