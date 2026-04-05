"""Recall, rerank, and deterministically distill CategoryInsight output."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from globex_agent.category_insight.models import (
    AttributeDist,
    Bestseller,
    CategoryCard,
    CategoryInsightDepth,
    CategoryInsightOutput,
    PriceTier,
)
from globex_agent.category_insight.reranking import (
    RerankerDocumentMode,
    rerank_category_hits,
)
from globex_agent.recall.category_kb import (
    DEFAULT_COARSE_K,
    CategoryCardHit,
    CategoryQueryType,
    CategorySearchResult,
)
from globex_agent.recall.embedding import TextEncoder
from globex_agent.recall.reranker import RawTextPairReranker

FINE_K_QUICK = 8
FINE_K_DEEP = 15
_PRICE_PATTERN = re.compile(
    r"便宜款\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*/\s*"
    r"中档\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*/\s*"
    r"高端\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)"
)


class CategoryKnowledgeBase(Protocol):
    def search(
        self,
        query: str,
        query_embedding: list[float],
        *,
        top_k: int,
        query_type: CategoryQueryType | str | None = None,
    ) -> CategorySearchResult: ...

    def search_bm25(
        self, query: str, *, top_k: int
    ) -> tuple[CategoryCardHit, ...]: ...


@dataclass(frozen=True, slots=True)
class CategoryInsightDiagnostics:
    recall_mode: str
    query_type: str | None
    coarse_count: int
    final_count: int
    reranked: bool
    reranker_document_mode: str
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CategoryInsightRun:
    output: CategoryInsightOutput
    diagnostics: CategoryInsightDiagnostics


class CategoryTaxonomy:
    def __init__(self, payload: dict[str, object]) -> None:
        categories = payload.get("categories")
        if not isinstance(categories, list):
            raise ValueError("taxonomy must contain a categories list")
        self._aliases: dict[str, str] = {}
        self._kinds: dict[str, str] = {}
        for raw in categories:
            if not isinstance(raw, dict):
                raise ValueError("taxonomy category entries must be objects")
            category = str(raw["category"])
            self._kinds[category.casefold()] = str(raw["category_kind"])
            aliases = raw.get("aliases", [])
            if not isinstance(aliases, list):
                raise ValueError("taxonomy aliases must be a list")
            for alias in [category, *aliases]:
                self._aliases[str(alias).strip().casefold()] = category

    @classmethod
    def from_json(cls, path: Path) -> CategoryTaxonomy:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def normalize(self, raw: str) -> str:
        clean = " ".join(raw.split())
        return self._aliases.get(clean.casefold(), clean)

    def category_kind(self, category: str) -> str:
        return self._kinds.get(category.casefold(), "ordinary")

    def is_known(self, category: str) -> bool:
        return category.casefold() in self._kinds


class CategoryInsightService:
    def __init__(
        self,
        taxonomy: CategoryTaxonomy,
        encoder: TextEncoder,
        knowledge_base: CategoryKnowledgeBase,
        reranker: RawTextPairReranker | None = None,
        *,
        reranker_document_mode: RerankerDocumentMode = "contextual",
    ) -> None:
        if reranker_document_mode not in ("summary_only", "contextual"):
            raise ValueError(
                "reranker_document_mode must be summary_only or contextual"
            )
        self._taxonomy = taxonomy
        self._encoder = encoder
        self._knowledge_base = knowledge_base
        self._reranker = reranker
        self._reranker_document_mode = reranker_document_mode

    def insight(
        self,
        category: str,
        *,
        depth: CategoryInsightDepth = "quick",
        query_type: CategoryQueryType | str | None = None,
    ) -> CategoryInsightRun:
        if depth not in ("quick", "deep"):
            raise ValueError("depth must be quick or deep")
        normalized = self._taxonomy.normalize(category)
        final_k = FINE_K_QUICK if depth == "quick" else FINE_K_DEEP
        recall_mode = "hybrid"
        fallback_reason: str | None = None
        resolved_query_type: str | None = None
        try:
            embedding = self._encoder.encode_queries([normalized])
            if embedding.shape[0] != 1:
                raise ValueError("query encoder must return one vector")
            search = self._knowledge_base.search(
                normalized,
                embedding[0].astype(float).tolist(),
                top_k=DEFAULT_COARSE_K,
                query_type=query_type,
            )
            coarse_hits = search.hits
            resolved_query_type = search.query_type.value
            recall_mode = "hybrid" if search.used_bm25 else "knn"
        except Exception as vector_exc:  # noqa: BLE001 - course tower fallback
            try:
                coarse_hits = self._knowledge_base.search_bm25(
                    normalized, top_k=DEFAULT_COARSE_K
                )
                recall_mode = "bm25_fallback"
                fallback_reason = f"vector recall failed: {type(vector_exc).__name__}"
            except Exception as search_exc:  # noqa: BLE001 - OS-down fallback
                output = _empty_output(normalized)
                return CategoryInsightRun(
                    output,
                    CategoryInsightDiagnostics(
                        recall_mode="empty_fallback",
                        query_type=None,
                        coarse_count=0,
                        final_count=0,
                        reranked=False,
                        reranker_document_mode=self._reranker_document_mode,
                        fallback_reason=(
                            f"knowledge base failed: {type(search_exc).__name__}"
                        ),
                    ),
                )

        if self._reranker is None:
            final_hits = coarse_hits[:final_k]
            reranked_flag = False
        else:
            reranked = rerank_category_hits(
                normalized,
                coarse_hits,
                self._reranker,
                final_k=final_k,
                document_mode=self._reranker_document_mode,
            )
            if reranked.bypass_reason:
                fallback_reason = "; ".join(
                    reason
                    for reason in (fallback_reason, reranked.bypass_reason)
                    if reason
                )
            final_hits = reranked.hits
            reranked_flag = reranked.reranked
        cards = [hit.card for hit in final_hits]
        resolved_category = _resolve_output_category(normalized, cards, self._taxonomy)
        relevant_cards = [card for card in cards if card.category == resolved_category]
        confidence = (
            round(sum(card.confidence for card in relevant_cards) / len(relevant_cards), 2)
            if relevant_cards
            else 0.0
        )
        output = CategoryInsightOutput(
            category=resolved_category,
            components=(
                _extract_components(relevant_cards)
                if self._taxonomy.category_kind(resolved_category) == "bundle"
                else []
            ),
            bestsellers=_extract_bestsellers(relevant_cards),
            attributes=(
                _extract_attributes(relevant_cards) if depth == "deep" else []
            ),
            price_tiers=_extract_price_tiers(relevant_cards),
            confidence=confidence,
        )
        return CategoryInsightRun(
            output,
            CategoryInsightDiagnostics(
                recall_mode=recall_mode,
                query_type=resolved_query_type,
                coarse_count=len(coarse_hits),
                final_count=len(final_hits),
                reranked=reranked_flag,
                reranker_document_mode=self._reranker_document_mode,
                fallback_reason=fallback_reason,
            ),
        )


def _resolve_output_category(
    normalized: str,
    cards: list[CategoryCard],
    taxonomy: CategoryTaxonomy,
) -> str:
    if taxonomy.is_known(normalized):
        return normalized
    if cards:
        counts = Counter(card.category for card in cards)
        return sorted(counts, key=lambda category: (-counts[category], category))[0]
    return normalized


def _extract_components(cards: list[CategoryCard]) -> list[str]:
    components: set[str] = set()
    for card in cards:
        if card.card_type != "bestseller" or "：" not in card.summary:
            continue
        components.update(
            token.strip()
            for token in card.summary.split("：", 1)[1].split("/")
            if token.strip()
        )
    return sorted(components)


def _extract_bestsellers(cards: list[CategoryCard]) -> list[Bestseller]:
    output: list[Bestseller] = []
    seen: set[str] = set()
    for card in cards:
        if card.card_type != "bestseller":
            continue
        for evidence in card.raw_evidence:
            parts = [part.strip() for part in evidence.split("|")]
            if len(parts) != 3 or parts[0] in seen:
                continue
            try:
                price = float(parts[1])
            except ValueError:
                continue
            output.append(
                Bestseller(
                    name=parts[0],
                    typical_price_cny=price,
                    why_popular=parts[2],
                )
            )
            seen.add(parts[0])
    return output[:5]


def _extract_attributes(cards: list[CategoryCard]) -> list[AttributeDist]:
    output: list[AttributeDist] = []
    for card in cards:
        if card.card_type != "attribute" or "：" not in card.summary:
            continue
        name, raw_distribution = card.summary.split("：", 1)
        distribution: dict[str, float] = {}
        for token in raw_distribution.split("/"):
            parts = token.strip().rsplit(" ", 1)
            if len(parts) != 2 or not parts[1].endswith("%"):
                continue
            try:
                distribution[parts[0]] = round(
                    float(parts[1].rstrip("%")) / 100, 4
                )
            except ValueError:
                continue
        if distribution:
            output.append(AttributeDist(name=name, distribution=distribution))
    return output


def _extract_price_tiers(cards: list[CategoryCard]) -> list[PriceTier]:
    for card in cards:
        if card.card_type != "price_range":
            continue
        match = _PRICE_PATTERN.fullmatch(card.summary)
        if match is None:
            continue
        values = [float(value) for value in match.groups()]
        return [
            PriceTier(
                tier="budget", range_cny=(values[0], values[1]), notes=card.summary
            ),
            PriceTier(tier="mid", range_cny=(values[2], values[3]), notes=card.summary),
            PriceTier(
                tier="premium", range_cny=(values[4], values[5]), notes=card.summary
            ),
        ]
    return []


def _empty_output(category: str) -> CategoryInsightOutput:
    return CategoryInsightOutput(
        category=category,
        components=[],
        bestsellers=[],
        attributes=[],
        price_tiers=[],
        confidence=0.0,
    )
