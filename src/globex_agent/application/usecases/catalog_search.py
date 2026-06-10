"""Catalog search use case with embedding recall, rerank, and hard filters."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from globex_agent.application.evidence import (
    ProductFactSnapshot,
    build_product_fact_snapshot,
    product_card_from_snapshot,
)
from globex_agent.domain.catalog.constraints import ConstraintEvaluator
from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable
from globex_agent.domain.catalog.models import (
    AvailabilityStatus,
    MarketLocale,
    Platform,
    StandardItem,
)
from globex_agent.domain.catalog.money import Money
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ItemVectorIndex,
    Reranker,
)
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec
from globex_agent.domain.shipping.tariff_schedule import TariffSchedule
from globex_agent.infrastructure.recall.search_document import standard_item_search_text

logger = logging.getLogger(__name__)

# The post-filtering refill sequence is deliberately bounded.  It preserves the
# Faiss ANN mainline while giving hard constraints a chance to recover enough
# candidates without silently turning a request into a full-catalog scan.
_ANN_RECALL_DEPTHS = (100, 200, 400, 500)
_FILTERED_OUT_LIMIT = 3

_PLATFORM_SHIP_TO: dict[Platform, tuple[str, ...]] = {
    Platform.AMAZON: ("US",),
    Platform.TAOBAO: ("CN",),
    Platform.SHOPEE: ("SG",),
    Platform.ALIEXPRESS: ("CN", "US", "EU"),
    Platform.EBAY: ("US",),
}
_PLATFORM_ORIGIN: dict[Platform, str] = {
    Platform.AMAZON: "US",
    Platform.TAOBAO: "CN",
    Platform.SHOPEE: "SG",
    Platform.ALIEXPRESS: "CN",
    Platform.EBAY: "US",
}


@dataclass(frozen=True)
class ProductCard:
    item_id: str
    title: str
    brand: str
    category: str
    category_path: list[str]
    store: str | None
    origin_country: str
    origin_country_source: str
    ships_to: list[str]
    ships_to_source: str
    price_major: float | None
    price_source: str
    currency: str
    availability: str
    highlights: list[str]
    variants: list[dict]
    score: float
    landed_price: dict | None
    price_min_major: float | None = None
    price_max_major: float | None = None
    requires_variant_selection: bool = False
    matching_variant_ids: list[str] | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict:
        card: dict[str, Any] = {
            "item_id": self.item_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "category_path": self.category_path,
            "store": self.store,
            "origin_country": self.origin_country,
            "origin_country_source": self.origin_country_source,
            "ships_to": self.ships_to,
            "ships_to_source": self.ships_to_source,
            "price_major": self.price_major,
            "price_source": self.price_source,
            "currency": self.currency,
            "availability": self.availability,
            "highlights": self.highlights,
            "variants": self.variants,
            "price_min_major": self.price_min_major,
            "price_max_major": self.price_max_major,
            "requires_variant_selection": self.requires_variant_selection,
            "matching_variant_ids": self.matching_variant_ids or [],
            "warnings": self.warnings or [],
            "score": round(self.score, 4),
        }
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        return card


@dataclass(frozen=True)
class _RecallPool:
    scored: list[tuple[float, StandardItem]]
    recall_depth: int
    ann_candidate_count: int
    filtered_count_by_reason: dict[str, int]
    filtered_out: list[dict]
    exhaustion_reason: str | None


def tokenize(text: str) -> set[str]:
    """Minimal whitespace plus CJK bigram tokenization for keyword fallback."""

    terms: set[str] = set()
    for chunk in text.casefold().split():
        terms.add(chunk)
        if any("\u4e00" <= ch <= "\u9fff" for ch in chunk) and len(chunk) >= 2:
            terms.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return terms


class CatalogSearchUseCase:
    def __init__(
        self,
        item_repo: ItemRepository,
        embedder: EmbeddingClient | None = None,
        vector_index: ItemVectorIndex | None = None,
        reranker: Reranker | None = None,
        tariff_schedule: TariffSchedule | None = None,
    ) -> None:
        self._item_repo = item_repo
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._tariff = tariff_schedule or TariffSchedule(rates=ExchangeRateTable())
        self._constraints = ConstraintEvaluator(self._tariff.rates)

    async def execute(self, spec: ProductSearchSpec) -> dict:
        scored: list[tuple[float, StandardItem]] = []
        recall_strategy = "bm25_fallback_no_ann_config"
        rerank_applied = False
        ann_recall_depth = 0
        ann_candidate_count = 0
        filtered_count_by_reason: dict[str, int] = {}
        filtered_out: list[dict] = []
        exhaustion_reason: str | None = None

        if self._embedder is not None and self._vector_index is not None:
            try:
                pool = await self._progressive_vector_recall(spec)
                scored = pool.scored
                recall_strategy = "embedding_only"
                ann_recall_depth = pool.recall_depth
                ann_candidate_count = pool.ann_candidate_count
                filtered_count_by_reason = pool.filtered_count_by_reason
                filtered_out = pool.filtered_out
                exhaustion_reason = pool.exhaustion_reason
            except Exception as err:  # noqa: BLE001 - recall must degrade safely
                logger.warning("向量召回不可用，降级关键词召回：%s", err)
                recall_strategy = "bm25_fallback_after_ann_failure"

        if recall_strategy == "embedding_only":
            try:
                scored = await self._rerank(spec, scored)
                recall_strategy = "embedding_rerank"
                rerank_applied = True
            except Exception as err:  # noqa: BLE001 - rerank may degrade
                logger.warning("rerank 不可用，按向量分排序：%s", err)
        else:
            fallback_scored = await self._keyword_recall(spec)
            scored, filtered_count_by_reason, filtered_out = self._filter_scored(
                fallback_scored,
                spec,
            )
            exhaustion_reason = "ann_unavailable_bm25_fallback"
            # BM25 is only an honest fault/configuration fallback.  It still
            # benefits from the configured cross-encoder after hard filtering.
            if self._reranker is not None:
                try:
                    scored = await self._rerank(spec, scored)
                    rerank_applied = True
                    recall_strategy = f"{recall_strategy}_rerank"
                except Exception as err:  # noqa: BLE001
                    logger.warning("BM25 降级后的 rerank 不可用：%s", err)

        snapshots: list[ProductFactSnapshot] = []
        hits: list[ProductCard] = []
        for score, item in scored[: spec.top_k]:
            snapshot = build_product_fact_snapshot(item)
            snapshots.append(snapshot)
            hits.append(self._to_card(score, item, spec, snapshot=snapshot))
        returned_count = len(hits)
        eligible_count = len(scored)
        result: dict[str, Any] = {
            "hits": [card.to_dict() for card in hits],
            # Existing field retained for callers; it now has the explicit
            # meaning of eligible candidates before requested Top-K truncation.
            "total_candidates": eligible_count,
            "recall_strategy": recall_strategy,
            "rerank_applied": rerank_applied,
            "requested_top_k": spec.top_k,
            "ann_recall_depth": ann_recall_depth,
            "ann_candidate_count": ann_candidate_count,
            "eligible_count": eligible_count,
            "filtered_count": sum(filtered_count_by_reason.values()),
            "returned_count": returned_count,
            "is_partial": returned_count < spec.top_k,
            "filtered_count_by_reason": filtered_count_by_reason,
            "evidence_snapshots": [snapshot.to_persisted_dict() for snapshot in snapshots],
        }
        if exhaustion_reason is not None:
            result["exhaustion_reason"] = exhaustion_reason
        if filtered_out:
            result["filtered_out"] = filtered_out
        return result

    async def _progressive_vector_recall(self, spec: ProductSearchSpec) -> _RecallPool:
        assert self._embedder is not None and self._vector_index is not None
        embedding = await self._embedder.embed(spec.normalized_query)
        seen_item_ids: set[str] = set()
        eligible: list[tuple[float, StandardItem]] = []
        filtered_count_by_reason: dict[str, int] = {}
        filtered_out: list[dict] = []
        recall_depth = 0
        ann_candidate_count = 0
        exhaustion_reason: str | None = None

        for depth in _ANN_RECALL_DEPTHS:
            vector_hits = await self._vector_index.search(embedding, top_n=depth)
            recall_depth = depth
            new_hits = []
            for hit in vector_hits:
                if hit.item_id in seen_item_ids:
                    continue
                seen_item_ids.add(hit.item_id)
                new_hits.append(hit)
            ann_candidate_count = len(seen_item_ids)
            if new_hits:
                items = await self._item_repo.find_by_ids([hit.item_id for hit in new_hits])
                by_id = {item.item_id: item for item in items}
                for hit in new_hits:
                    item = by_id.get(hit.item_id)
                    if item is None:
                        filtered_count_by_reason["item_not_found"] = (
                            filtered_count_by_reason.get("item_not_found", 0) + 1
                        )
                        continue
                    decision = self._constraints.evaluate(item, spec)
                    if decision.eligible:
                        eligible.append((hit.score, item))
                    else:
                        assert decision.reason is not None
                        filtered_count_by_reason[decision.reason] = (
                            filtered_count_by_reason.get(decision.reason, 0) + 1
                        )
                        if len(filtered_out) < _FILTERED_OUT_LIMIT:
                            filtered_out.append(self._to_rejected(item, spec, decision.reason))
            if len(eligible) >= spec.top_k:
                exhaustion_reason = "requested_top_k_reached"
                break
            if len(vector_hits) < depth:
                exhaustion_reason = "index_exhausted"
                break
            if depth == _ANN_RECALL_DEPTHS[-1]:
                exhaustion_reason = "max_recall_depth_reached"

        return _RecallPool(
            scored=eligible,
            recall_depth=recall_depth,
            ann_candidate_count=ann_candidate_count,
            filtered_count_by_reason=filtered_count_by_reason,
            filtered_out=filtered_out,
            exhaustion_reason=exhaustion_reason,
        )

    def _filter_scored(
        self,
        scored: list[tuple[float, StandardItem]],
        spec: ProductSearchSpec,
    ) -> tuple[list[tuple[float, StandardItem]], dict[str, int], list[dict]]:
        eligible: list[tuple[float, StandardItem]] = []
        counts: dict[str, int] = {}
        filtered_out: list[dict] = []
        for score, item in scored:
            decision = self._constraints.evaluate(item, spec)
            if decision.eligible:
                eligible.append((score, item))
                continue
            assert decision.reason is not None
            counts[decision.reason] = counts.get(decision.reason, 0) + 1
            if len(filtered_out) < _FILTERED_OUT_LIMIT:
                filtered_out.append(self._to_rejected(item, spec, decision.reason))
        return eligible, counts, filtered_out

    def _to_rejected(self, item: StandardItem, spec: ProductSearchSpec, reason: str) -> dict:
        price = _primary_price_money(item)
        converted = (
            self._tariff.rates.convert(price, spec.target_currency).to_major_units()
            if price is not None
            else None
        )
        return {
            "item_id": item.item_id,
            "title": item.title,
            "category": _category_label(item),
            "price_major": round(converted, 2) if converted is not None else None,
            "currency": spec.target_currency,
            "reason": reason,
        }

    async def _rerank(
        self,
        spec: ProductSearchSpec,
        scored: list[tuple[float, StandardItem]],
    ) -> list[tuple[float, StandardItem]]:
        if self._reranker is None:
            raise RuntimeError("Reranker 未配置")
        documents = [standard_item_search_text(item) for _, item in scored]
        rerank_scores = await self._reranker.rerank(spec.normalized_query, documents)
        if len(rerank_scores) != len(scored):
            raise ValueError("reranker score count must match candidate count")
        reranked = [(rerank_scores[index], item) for index, (_, item) in enumerate(scored)]
        reranked.sort(key=lambda pair: (-pair[0], pair[1].item_id))
        return reranked

    async def _keyword_recall(self, spec: ProductSearchSpec) -> list[tuple[float, StandardItem]]:
        query_terms = tokenize(spec.normalized_query)
        items = await self._item_repo.list_all()
        documents = [tokenize(standard_item_search_text(item)) for item in items]
        document_frequency = {
            term: sum(term in document for document in documents) for term in query_terms
        }
        avg_length = sum(len(document) for document in documents) / max(len(documents), 1)
        candidates: list[tuple[float, StandardItem]] = []
        for item, terms in zip(items, documents, strict=True):
            score = _bm25_score(query_terms, terms, document_frequency, len(items), avg_length)
            if score > 0:
                candidates.append((score, item))
        candidates.sort(key=lambda pair: (-pair[0], pair[1].item_id))
        return candidates

    def _to_card(
        self,
        score: float,
        item: StandardItem,
        spec: ProductSearchSpec,
        *,
        snapshot: ProductFactSnapshot | None = None,
    ) -> ProductCard:
        price = _primary_price_money(item)
        landed_price: dict | None = None
        variant_landed_prices: dict[str, dict] = {}
        if spec.ship_to and not item.variants and price is not None:
            try:
                quote = self._tariff.quote(
                    subtotal=price,
                    category=_category_label(item),
                    ship_to=spec.ship_to,
                    quantity=1,
                    target_currency=spec.target_currency,
                )
                landed_price = quote.to_dict()
            except ValueError as err:
                landed_price = {
                    "unavailable_reason": str(err),
                    "currency": spec.target_currency,
                    "ship_to": spec.ship_to.strip().upper(),
                    "quantity": 1,
                }
        elif spec.ship_to and item.variants:
            for variant in item.variants:
                if (
                    variant.price_cny is None
                    or variant.availability is not AvailabilityStatus.AVAILABLE
                ):
                    continue
                try:
                    quote = self._tariff.quote(
                        subtotal=Money.from_major_units(float(variant.price_cny), "CNY"),
                        category=_category_label(item),
                        ship_to=spec.ship_to,
                        quantity=1,
                        target_currency=spec.target_currency,
                    )
                    variant_landed_prices[variant.variant_id] = quote.to_dict()
                except ValueError as err:
                    # Keep the reason beside the exact SKU.  A top-level
                    # unavailable quote would incorrectly apply to every SKU.
                    variant_landed_prices[variant.variant_id] = {
                        "unavailable_reason": str(err),
                        "currency": spec.target_currency,
                        "ship_to": spec.ship_to.strip().upper(),
                        "quantity": 1,
                    }
        snapshot = snapshot or build_product_fact_snapshot(item)
        card = product_card_from_snapshot(
            snapshot,
            score=score,
            landed_price=landed_price,
            variant_landed_prices=variant_landed_prices,
            price_min_major=_variant_price_range(item)[0],
            price_max_major=_variant_price_range(item)[1],
            requires_variant_selection=bool(item.variants),
            matching_variant_ids=self._constraints.evaluate(item, spec).matching_variant_ids,
            warnings=self._constraints.evaluate(item, spec).warnings,
        )
        return card


def _primary_price_money(item: StandardItem) -> Money | None:
    if item.price_cny is None:
        return None
    return Money.from_major_units(float(item.price_cny), "CNY")


def _category_label(item: StandardItem) -> str:
    return item.category_path[-1] if item.category_path else ""


def searchable_text(item: StandardItem) -> str:
    """Backward-compatible alias for the canonical runtime search document."""

    return standard_item_search_text(item)


def _bm25_score(
    query_terms: set[str],
    document_terms: set[str],
    document_frequency: dict[str, int],
    document_count: int,
    average_length: float,
) -> float:
    # A compact BM25 implementation for the explicit ANN-failure fallback.
    # The current tokenizer is set-based, so term frequency is 1 by design.
    k1 = 1.2
    b = 0.75
    normalization = 1.0 - b + b * len(document_terms) / max(average_length, 1.0)
    score = 0.0
    for term in query_terms & document_terms:
        frequency = document_frequency.get(term, 0)
        idf = max(
            0.0,
            math.log((document_count - frequency + 0.5) / (frequency + 0.5) + 1.0),
        )
        score += idf * (k1 + 1.0) / (1.0 + k1 * normalization)
    return score


def derived_origin_country(item: StandardItem) -> tuple[str, str]:
    explicit = next(
        (attribute.value for attribute in item.attributes if attribute.code == "origin_country"),
        None,
    )
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip(), "explicit"
    locale_country = {
        MarketLocale.CN: "CN",
        MarketLocale.JP: "JP",
    }.get(item.locale)
    if locale_country:
        return locale_country, "derived"
    return _PLATFORM_ORIGIN.get(item.platform, ""), "derived"


def derived_ships_to(item: StandardItem) -> tuple[tuple[str, ...], str]:
    if item.ships_to is not None:
        return tuple(item.ships_to), "explicit"
    locale_ships = {
        MarketLocale.CN: ("CN",),
        MarketLocale.JP: ("JP",),
    }.get(item.locale)
    if locale_ships:
        return locale_ships, "derived"
    return _PLATFORM_SHIP_TO.get(item.platform, ()), "derived"


def _variant_cards(item: StandardItem) -> list[dict]:
    cards: list[dict] = []
    for variant in item.variants or []:
        price_major = float(variant.price_cny) if variant.price_cny is not None else None
        cards.append(
            {
                "variant_id": variant.variant_id,
                "options": [
                    {"code": option.code, "name": option.name, "value": option.value}
                    for option in variant.options
                ],
                "display_name": " / ".join(
                    f"{option.name}: {option.value}" for option in variant.options
                ),
                "price_major": price_major,
                "currency": "CNY",
                "availability": variant.availability.value,
            }
        )
    return cards


def _variant_price_range(item: StandardItem) -> tuple[float | None, float | None]:
    prices = [
        float(variant.price_cny)
        for variant in item.variants
        if variant.price_cny is not None and variant.availability is AvailabilityStatus.AVAILABLE
    ]
    if not prices:
        return None, None
    return min(prices), max(prices)


def _attribute_highlights(item: StandardItem) -> list[str]:
    return [
        f"{attribute.name}: {attribute.value}{(' ' + attribute.unit) if attribute.unit else ''}"
        for attribute in item.attributes
    ]
