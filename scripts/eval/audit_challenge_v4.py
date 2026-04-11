"""Audit v4 challenge construction and explain the BM25 baseline by slice."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from globex_agent.eval import RankingMetrics, aggregate_metrics, evaluate_ranking
from globex_agent.recall import KeywordSearchBackend
from globex_agent.recall.keyword import tokenize
from globex_agent.recall.persistence import load_search_documents_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "product_v4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "eval" / "challenge_v4_bias_audit.json",
    )
    args = parser.parse_args()

    items_path = args.data_dir / "recall_items.jsonl"
    cases_path = args.data_dir / "recall_cases.jsonl"
    documents = load_search_documents_jsonl(items_path)
    item_rows = _read_jsonl(items_path)
    cases = _read_jsonl(cases_path)
    document_by_id = {document.document_id: document for document in documents}
    item_by_id = {str(row["document_id"]): row for row in item_rows}

    metrics_by_kind: dict[str, list[RankingMetrics]] = defaultdict(list)
    metrics_by_language: dict[str, list[RankingMetrics]] = defaultdict(list)
    metrics_overall: list[RankingMetrics] = []
    overlap_by_label: dict[str, list[float]] = defaultdict(list)
    same_category_negative_counts: list[int] = []
    source_counts: Counter[str] = Counter()

    for case in cases:
        pool = [document_by_id[document_id] for document_id in case["candidate_ids"]]
        ranked_ids = [
            hit.document_id
            for hit in KeywordSearchBackend(pool).search(case["query"], top_k=10).hits
        ]
        metric = evaluate_ranking(
            ranked_ids,
            case["relevance"],
            10,
            ndcg_relevance=case["graded_relevance"],
        )
        metrics_overall.append(metric)
        metrics_by_kind[str(case["challenge_kind"])].append(metric)
        metrics_by_language[str(case["language"])].append(metric)

        query_tokens = set(tokenize(str(case["query"])))
        for document_id, label in case["labels"].items():
            document = document_by_id[document_id]
            document_tokens = set(tokenize(f"{document.title} {document.body}"))
            coverage = (
                len(query_tokens & document_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            overlap_by_label[label].append(coverage)
        same_category_negative_counts.append(
            sum(
                item_by_id[document_id]["category"] == case["category"]
                and label != "Exact"
                for document_id, label in case["labels"].items()
            )
        )
        if case["language"] == "en":
            source_counts.update(
                item_by_id[document_id]["source_kind"]
                for document_id, label in case["labels"].items()
                if label == "Exact"
            )

    report = {
        "dataset_version": "challenge-v4-bias-audit",
        "scope": (
            "post-hoc read-only audit; does not alter the frozen test data "
            "or ranking configuration"
        ),
        "bm25_at_10": {
            "overall": _metrics(aggregate_metrics(metrics_overall)),
            "by_challenge_kind": {
                key: _metrics(aggregate_metrics(value))
                for key, value in sorted(metrics_by_kind.items())
            },
            "by_language": {
                key: _metrics(aggregate_metrics(value))
                for key, value in sorted(metrics_by_language.items())
            },
        },
        "lexical_query_token_coverage": {
            label: {
                "mean": statistics.fmean(values),
                "p50": statistics.median(values),
            }
            for label, values in sorted(overlap_by_label.items())
        },
        "same_category_negative_count": {
            "minimum": min(same_category_negative_counts),
            "mean": statistics.fmean(same_category_negative_counts),
            "maximum": max(same_category_negative_counts),
        },
        "exact_positive_source_counts_by_intent": dict(sorted(source_counts.items())),
        "limitations": [
            "all v4 queries are model-authored offline fixtures",
            "Chinese queries and product text are machine-generated and unreviewed",
            (
                "attribute labels are still derived from the same structured category "
                "specifications as the items"
            ),
            "BM25 title terms receive weight three in the local baseline",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["bm25_at_10"], ensure_ascii=False, indent=2))
    print(f"report: {args.output}")


def _metrics(metric: RankingMetrics) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in asdict(metric).items()
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
