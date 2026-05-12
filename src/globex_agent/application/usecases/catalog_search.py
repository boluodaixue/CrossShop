"""Catalog search use case with embedding recall, rerank, and hard filters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable
from globex_agent.domain.catalog.models import MarketLocale, Platform, StandardItem
from globex_agent.domain.catalog.money import Money
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ItemVectorIndex,
    Reranker,
)
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec
from globex_agent.domain.shipping.tariff_schedule import TariffSchedule

logger = logging.getLogger(__name__)

_RECALL_TOP_N = 8
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
    origin_country: str
    origin_country_source: str
    ships_to: list[str]
    ships_to_source: str
    price_major: float | None
    price_source: str
    currency: str
    highlights: list[str]
    variants: list[dict]
    score: float
    landed_price: dict | None

    def to_dict(self) -> dict:
        card: dict[str, Any] = {
            "item_id": self.item_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "category_path": self.category_path,
            "origin_country": self.origin_country,
            "origin_country_source": self.origin_country_source,
            "ships_to": self.ships_to,
            "ships_to_source": self.ships_to_source,
            "price_major": self.price_major,
            "price_source": self.price_source,
            "currency": self.currency,
            "highlights": self.highlights,
            "variants": self.variants,
            "score": round(self.score, 4),
        }
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        return card


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

    async def execute(self, spec: ProductSearchSpec) -> dict:
        scored: list[tuple[float, StandardItem]] = []
        recall_strategy = "keyword_bm25"
        rerank_applied = False

        if self._embedder is not None and self._vector_index is not None:
            try:
                scored = await self._vector_recall(spec)
                recall_strategy = "embedding_only"
            except Exception as err:  # noqa: BLE001 - recall must degrade safely
                logger.warning("向量召回不可用，降级关键词召回：%s", err)
                scored = []

        if recall_strategy == "embedding_only" and scored:
            try:
                scored = await self._rerank(spec, scored)
                recall_strategy = "embedding_rerank"
                rerank_applied = True
            except Exception as err:  # noqa: BLE001 - rerank may degrade
                logger.warning("rerank 不可用，按向量分排序：%s", err)
        elif not scored:
            scored = await self._keyword_recall(spec)

        filtered: list[tuple[float, StandardItem]] = []
        filtered_out: list[dict] = []
        for score, item in scored:
            reason = self._reject_reason(item, spec)
            if reason is None:
                filtered.append((score, item))
            elif len(filtered_out) < _FILTERED_OUT_LIMIT:
                filtered_out.append(self._to_rejected(item, spec, reason))

        hits = [self._to_card(score, item, spec) for score, item in filtered[: spec.top_k]]
        result: dict[str, Any] = {
            "hits": [card.to_dict() for card in hits],
            "total_candidates": len(filtered),
            "recall_strategy": recall_strategy,
            "rerank_applied": rerank_applied,
        }
        if filtered_out:
            result["filtered_out"] = filtered_out
        return result

    def _reject_reason(self, item: StandardItem, spec: ProductSearchSpec) -> str | None:
        if not item.is_available:
            return "item_unavailable"
        ships_to, _ = derived_ships_to(item)
        if spec.ship_to and spec.ship_to not in ships_to:
            return "ship_to_unavailable"
        if not self._within_price_cap(item, spec):
            return "over_price_cap"
        return None

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

    def _within_price_cap(self, item: StandardItem, spec: ProductSearchSpec) -> bool:
        if spec.price_max_major is None:
            return True
        price = _primary_price_money(item)
        if price is None:
            return False
        converted = self._tariff.rates.convert(price, spec.target_currency)
        return converted.to_major_units() <= spec.price_max_major

    async def _vector_recall(self, spec: ProductSearchSpec) -> list[tuple[float, StandardItem]]:
        assert self._embedder is not None and self._vector_index is not None
        embedding = await self._embedder.embed(spec.normalized_query)
        vector_hits = await self._vector_index.search(embedding, top_n=_RECALL_TOP_N)
        items = await self._item_repo.find_by_ids([hit.item_id for hit in vector_hits])
        by_id = {item.item_id: item for item in items}
        return [
            (hit.score, by_id[hit.item_id])
            for hit in vector_hits
            if hit.item_id in by_id
        ]

    async def _rerank(
        self,
        spec: ProductSearchSpec,
        scored: list[tuple[float, StandardItem]],
    ) -> list[tuple[float, StandardItem]]:
        if self._reranker is None:
            raise RuntimeError("Reranker 未配置")
        documents = [searchable_text(item) for _, item in scored]
        rerank_scores = await self._reranker.rerank(spec.normalized_query, documents)
        reranked = [
            (rerank_scores[index], item)
            for index, (_, item) in enumerate(scored)
        ]
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        return reranked

    async def _keyword_recall(self, spec: ProductSearchSpec) -> list[tuple[float, StandardItem]]:
        query_terms = tokenize(spec.normalized_query)
        candidates: list[tuple[float, StandardItem]] = []
        for item in await self._item_repo.list_all():
            score = _keyword_score(query_terms, item, spec)
            if score > 0:
                candidates.append((score, item))
        candidates.sort(key=lambda pair: (-pair[0], pair[1].item_id))
        return candidates

    def _to_card(
        self,
        score: float,
        item: StandardItem,
        spec: ProductSearchSpec,
    ) -> ProductCard:
        price = _primary_price_money(item)
        landed_price: dict | None = None
        if spec.ship_to and price is not None:
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
                landed_price = {"unavailable_reason": str(err)}
        origin_country, origin_source = derived_origin_country(item)
        ships_to, ships_source = derived_ships_to(item)
        return ProductCard(
            item_id=item.item_id,
            title=item.title,
            brand=item.brand or "",
            category=_category_label(item),
            category_path=list(item.category_path),
            origin_country=origin_country,
            origin_country_source=origin_source,
            ships_to=list(ships_to),
            ships_to_source=ships_source,
            price_major=price.to_major_units() if price is not None else None,
            price_source=item.price_source.value,
            currency=item.currency_raw.value if item.currency_raw else "CNY",
            highlights=[
                f"{key}: {value}"
                for key, value in item.attributes.items()
            ][:8],
            variants=_variant_cards(item),
            score=score,
            landed_price=landed_price,
        )


def _primary_price_money(item: StandardItem) -> Money | None:
    if item.price_cny is None:
        return None
    return Money.from_major_units(float(item.price_cny), "CNY")


def _category_label(item: StandardItem) -> str:
    return item.category_path[-1] if item.category_path else ""


def searchable_text(item: StandardItem) -> str:
    attribute_text = " ".join(
        f"{key} {value}" for key, value in item.attributes.items()
    )
    return " ".join(
        part
        for part in (
            item.title,
            item.brand or "",
            " ".join(item.category_path),
            item.description,
            attribute_text,
        )
        if part
    )


def _keyword_score(
    query_terms: set[str],
    item: StandardItem,
    spec: ProductSearchSpec,
) -> float:
    doc_terms = tokenize(searchable_text(item))
    matched = query_terms & doc_terms
    if not matched:
        return 0.0
    score = float(len(matched))
    if spec.category and spec.category in _category_label(item):
        score += 3.0
    return score


def derived_origin_country(item: StandardItem) -> tuple[str, str]:
    explicit = item.attributes.get("origin_country")
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
    explicit = item.attributes.get("ships_to")
    if isinstance(explicit, list):
        values = tuple(str(value) for value in explicit if str(value).strip())
        if values:
            return values, "explicit"
    locale_ships = {
        MarketLocale.CN: ("CN",),
        MarketLocale.JP: ("JP",),
    }.get(item.locale)
    if locale_ships:
        return locale_ships, "derived"
    return _PLATFORM_SHIP_TO.get(item.platform, ()), "derived"


def _variant_cards(item: StandardItem) -> list[dict]:
    cards: list[dict] = []
    primary = _primary_price_money(item)
    for variant in item.variants or []:
        if not isinstance(variant, dict):
            continue
        variant_id = variant.get("variant_id") or variant.get("id")
        if not variant_id:
            continue
        price_raw = variant.get("price_cny")
        price_major = (
            float(price_raw)
            if isinstance(price_raw, (int, float, str))
            else (primary.to_major_units() if primary is not None else None)
        )
        cards.append(
            {
                "variant_id": str(variant_id),
                "spec": str(variant.get("spec") or variant.get("name") or ""),
                "price_major": price_major,
                "currency": str(variant.get("currency_raw") or "CNY"),
                "is_available": bool(variant.get("is_available", True)),
            }
        )
    return cards
