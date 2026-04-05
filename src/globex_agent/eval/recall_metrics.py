"""Deterministic Recall@K, MRR@K, NDCG@K, and empty-recall metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    empty_recall: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def recall_at_k(
    ranked_ids: list[str] | tuple[str, ...],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of all known relevant documents found in top K."""

    _validate_k(k)
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(
    ranked_ids: list[str] | tuple[str, ...],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant document in top K."""

    _validate_k(k)
    for rank, document_id in enumerate(ranked_ids[:k], start=1):
        if document_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: list[str] | tuple[str, ...],
    relevance: dict[str, float],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain using graded relevance."""

    _validate_k(k)
    gains = [max(0.0, relevance.get(document_id, 0.0)) for document_id in ranked_ids[:k]]
    ideal_gains = sorted((max(0.0, gain) for gain in relevance.values()), reverse=True)[:k]
    ideal = _discounted_cumulative_gain(ideal_gains)
    if ideal == 0:
        return 0.0
    return _discounted_cumulative_gain(gains) / ideal


def evaluate_ranking(
    ranked_ids: list[str] | tuple[str, ...],
    relevance: dict[str, float],
    k: int,
    *,
    ndcg_relevance: dict[str, float] | None = None,
) -> RankingMetrics:
    """Evaluate binary Recall/MRR and optionally separate graded NDCG gains."""

    relevant_ids = {document_id for document_id, gain in relevance.items() if gain > 0}
    return RankingMetrics(
        recall_at_k=recall_at_k(ranked_ids, relevant_ids, k),
        mrr_at_k=reciprocal_rank_at_k(ranked_ids, relevant_ids, k),
        ndcg_at_k=ndcg_at_k(
            ranked_ids,
            relevance if ndcg_relevance is None else ndcg_relevance,
            k,
        ),
        empty_recall=float(not ranked_ids),
    )


def aggregate_metrics(metrics: list[RankingMetrics]) -> RankingMetrics:
    if not metrics:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0)
    return RankingMetrics(
        recall_at_k=fmean(metric.recall_at_k for metric in metrics),
        mrr_at_k=fmean(metric.mrr_at_k for metric in metrics),
        ndcg_at_k=fmean(metric.ndcg_at_k for metric in metrics),
        empty_recall=fmean(metric.empty_recall for metric in metrics),
    )


def _discounted_cumulative_gain(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")
