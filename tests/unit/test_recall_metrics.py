import pytest

from globex_agent.eval import (
    aggregate_metrics,
    evaluate_ranking,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_and_mrr_at_k() -> None:
    ranking = ["not-relevant", "relevant-a", "relevant-b"]
    relevant = {"relevant-a", "relevant-b", "missing"}

    assert recall_at_k(ranking, relevant, 2) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k(ranking, relevant, 2) == 0.5


def test_ndcg_uses_graded_relevance_and_ideal_order() -> None:
    relevance = {"exact": 1.0, "complement": 0.1, "substitute": 0.01}

    assert ndcg_at_k(["exact", "complement", "substitute"], relevance, 3) == 1.0
    assert ndcg_at_k(["substitute", "complement", "exact"], relevance, 3) < 1.0


def test_evaluation_and_aggregation_include_empty_recall_rate() -> None:
    hit = evaluate_ranking(["exact"], {"exact": 1.0}, k=3)
    miss = evaluate_ranking([], {"exact": 1.0}, k=3)

    aggregate = aggregate_metrics([hit, miss])

    assert aggregate.recall_at_k == 0.5
    assert aggregate.mrr_at_k == 0.5
    assert aggregate.ndcg_at_k == 0.5
    assert aggregate.empty_recall == 0.5


def test_evaluation_separates_exact_recall_from_graded_ndcg() -> None:
    metrics = evaluate_ranking(
        ["substitute", "exact"],
        {"substitute": 0.0, "exact": 1.0},
        k=2,
        ndcg_relevance={"substitute": 0.01, "exact": 1.0},
    )

    assert metrics.recall_at_k == 1.0
    assert metrics.mrr_at_k == 0.5
    assert 0.0 < metrics.ndcg_at_k < 1.0


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be at least 1"):
        recall_at_k([], set(), 0)
