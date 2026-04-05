"""Optional BM25/vector Hybrid ablation for product retrieval experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from globex_agent.recall.base import RecallHit, RecallResult, SearchBackend


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """Weights from the course's hybrid retrieval example."""

    semantic: float = 0.7
    lexical: float = 0.3

    def __post_init__(self) -> None:
        if self.semantic < 0 or self.lexical < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.semantic + self.lexical == 0:
            raise ValueError("at least one fusion weight must be positive")


class WeightedFusionSearchBackend:
    """Min-max normalize, weighted-sum, deduplicate, and rank both recall lists."""

    def __init__(
        self,
        lexical_backend: SearchBackend,
        semantic_backend: SearchBackend,
        *,
        candidate_k: int = 100,
        weights: FusionWeights | None = None,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self._lexical = lexical_backend
        self._semantic = semantic_backend
        self._candidate_k = candidate_k
        self._weights = weights or FusionWeights()

    @property
    def backend_id(self) -> str:
        return (
            "fusion:minmax-weighted-v1"
            f":semantic={self._weights.semantic:g}"
            f":lexical={self._weights.lexical:g}"
        )

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        return self._search_queries(
            semantic_query=query,
            lexical_query=query,
            top_k=top_k,
        )

    def _search_queries(
        self,
        *,
        semantic_query: str,
        lexical_query: str,
        top_k: int,
    ) -> RecallResult:
        top_k = max(1, top_k)
        recall_k = max(top_k, self._candidate_k)

        lexical_result: RecallResult | None = None
        semantic_result: RecallResult | None = None
        lexical_error: Exception | None = None
        semantic_error: Exception | None = None
        try:
            lexical_result = self._lexical.search(lexical_query, top_k=recall_k)
        except Exception as exc:  # noqa: BLE001 - fallback is part of this boundary
            lexical_error = exc
        try:
            semantic_result = self._semantic.search(semantic_query, top_k=recall_k)
        except Exception as exc:  # noqa: BLE001 - fallback is part of this boundary
            semantic_error = exc

        if lexical_result is None and semantic_result is None:
            raise RuntimeError("both lexical and semantic recall failed") from (
                semantic_error or lexical_error
            )
        if semantic_result is None:
            assert lexical_result is not None
            return _fallback_result(
                lexical_result,
                backend_id=self.backend_id,
                top_k=top_k,
                reason=f"semantic recall failed: {type(semantic_error).__name__}",
            )
        if lexical_result is None:
            return _fallback_result(
                semantic_result,
                backend_id=self.backend_id,
                top_k=top_k,
                reason=f"lexical recall failed: {type(lexical_error).__name__}",
            )

        lexical_scores = _min_max_scores(lexical_result.hits)
        semantic_scores = _min_max_scores(semantic_result.hits)
        document_ids = set(lexical_scores) | set(semantic_scores)
        ranked = sorted(
            (
                (
                    self._weights.semantic * semantic_scores.get(document_id, 0.0)
                    + self._weights.lexical * lexical_scores.get(document_id, 0.0),
                    document_id,
                )
                for document_id in document_ids
            ),
            key=lambda entry: (-entry[0], entry[1]),
        )
        hits = tuple(
            RecallHit(document_id=document_id, score=round(score, 8), rank=rank)
            for rank, (score, document_id) in enumerate(ranked[:top_k], start=1)
        )
        fallback_reasons = tuple(
            reason
            for reason in (lexical_result.fallback_reason, semantic_result.fallback_reason)
            if reason
        )
        return RecallResult(
            backend_id=self.backend_id,
            hits=hits,
            total_recall=len(ranked),
            truncated=len(ranked) > top_k,
            fallback_reason="; ".join(fallback_reasons) or None,
        )


class LocalizedFusionSearchBackend(WeightedFusionSearchBackend):
    """Keep the original semantic query; localize only the lexical branch."""

    def __init__(
        self,
        lexical_backend: SearchBackend,
        semantic_backend: SearchBackend,
        *,
        lexical_query_localizer: Callable[[str], str],
        candidate_k: int = 100,
        weights: FusionWeights | None = None,
    ) -> None:
        super().__init__(
            lexical_backend,
            semantic_backend,
            candidate_k=candidate_k,
            weights=weights,
        )
        self._lexical_query_localizer = lexical_query_localizer

    @property
    def backend_id(self) -> str:
        return f"{super().backend_id}:localized-lexical-only"

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        localized_query = self._lexical_query_localizer(query).strip() or query
        return self._search_queries(
            semantic_query=query,
            lexical_query=localized_query,
            top_k=top_k,
        )


def _min_max_scores(hits: tuple[RecallHit, ...]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [hit.score for hit in hits]
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return {hit.document_id: 1.0 for hit in hits}
    return {
        hit.document_id: (hit.score - minimum) / (maximum - minimum)
        for hit in hits
    }


def _fallback_result(
    result: RecallResult,
    *,
    backend_id: str,
    top_k: int,
    reason: str,
) -> RecallResult:
    hits = tuple(
        RecallHit(document_id=hit.document_id, score=hit.score, rank=rank)
        for rank, hit in enumerate(result.hits[:top_k], start=1)
    )
    inherited = f"; {result.fallback_reason}" if result.fallback_reason else ""
    return RecallResult(
        backend_id=backend_id,
        hits=hits,
        total_recall=result.total_recall,
        truncated=result.total_recall > top_k,
        fallback_reason=f"{reason}{inherited}",
    )
