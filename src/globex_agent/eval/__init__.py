"""Offline evaluation helpers for retrieval baselines."""

from globex_agent.eval.recall_metrics import (
    RankingMetrics,
    aggregate_metrics,
    evaluate_ranking,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from globex_agent.eval.shopsimulator_retrieval import (
    DATASET_VERSION as SHOPSIMULATOR_RETRIEVAL_DATASET_VERSION,
)
from globex_agent.eval.shopsimulator_retrieval import (
    ItemRelevanceAnnotation,
    QueryRelevanceAnnotation,
    RetrievalPoolCandidate,
    RetrievalPoolCase,
    SelectedRetrievalTask,
)

__all__ = [
    "RankingMetrics",
    "aggregate_metrics",
    "evaluate_ranking",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "SHOPSIMULATOR_RETRIEVAL_DATASET_VERSION",
    "ItemRelevanceAnnotation",
    "QueryRelevanceAnnotation",
    "RetrievalPoolCandidate",
    "RetrievalPoolCase",
    "SelectedRetrievalTask",
]
