"""Run the deterministic BM25 baseline against the checked-in ESCI subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from globex_agent.eval import RankingMetrics, aggregate_metrics, evaluate_ranking
from globex_agent.infrastructure.recall import KeywordSearchBackend, SearchDocument

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "eval" / "recall_report.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run data preparation first; existing checked-in files are reused.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="With --prepare, redownload the selected public rows.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.refresh and not args.prepare:
        parser.error("--refresh requires --prepare")

    if args.prepare:
        _run_preparation(args.data_dir, refresh=args.refresh)

    report = run_evaluation(args.data_dir, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_summary(report, args.output)


def run_evaluation(data_dir: Path, *, top_k: int) -> dict[str, Any]:
    items_path = data_dir / "recall_items.jsonl"
    cases_path = data_dir / "recall_cases.jsonl"
    manifest_path = data_dir / "recall_manifest.json"
    documents = [
        SearchDocument(
            document_id=record["document_id"],
            title=record["title"],
            body=record["body"],
        )
        for record in _read_jsonl(items_path)
    ]
    cases = _read_jsonl(cases_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    document_by_id = {document.document_id: document for document in documents}

    by_split: dict[str, list[RankingMetrics]] = defaultdict(list)
    all_metrics: list[RankingMetrics] = []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_documents = [
            document_by_id[document_id] for document_id in case["candidate_ids"]
        ]
        backend = KeywordSearchBackend(case_documents)
        result = backend.search(case["query"], top_k=top_k)
        ranked_ids = [hit.document_id for hit in result.hits]
        metrics = evaluate_ranking(
            ranked_ids,
            case["relevance"],
            top_k,
            ndcg_relevance=case["graded_relevance"],
        )
        judged_coverage = _judged_coverage(ranked_ids, case["labels"])
        if judged_coverage != 1.0:
            raise RuntimeError(f"unjudged result for query_id={case['query_id']}")
        by_split[case["split"]].append(metrics)
        all_metrics.append(metrics)
        case_results.append(
            {
                "query_id": case["query_id"],
                "split": case["split"],
                "returned": len(ranked_ids),
                "candidate_pool_size": len(case["candidate_ids"]),
                "exact_positive_count": sum(
                    label == "Exact" for label in case["labels"].values()
                ),
                "judged_coverage": judged_coverage,
                "metrics": _rounded(metrics),
            }
        )

    return {
        "backend": {
            "backend_id": KeywordSearchBackend.backend_id,
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "title_weight": 3,
        },
        "evaluation": {
            "top_k": top_k,
            "judged_coverage": 1.0,
            "metric_policy": {
                "recall_mrr_positive_labels": ["Exact"],
                "ndcg_gain_mapping": manifest["gain_mapping"],
                "empty_recall": "no lexical hit returned for the query",
                "candidate_pool": "per-query closed and fully judged",
                "unjudged_results": "forbidden",
            },
        },
        "dataset": {
            **manifest,
            "items_sha256": _sha256(items_path),
            "cases_sha256": _sha256(cases_path),
        },
        "metrics": {
            "overall": _rounded(aggregate_metrics(all_metrics)),
            "by_split": {
                split: _rounded(aggregate_metrics(by_split.get(split, [])))
                for split in ("train", "dev", "test")
            },
        },
        "cases": case_results,
    }


def _run_preparation(data_dir: Path, *, refresh: bool) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "data" / "prepare_esci_subset.py"),
        "--output-dir",
        str(data_dir),
    ]
    if refresh:
        command.append("--refresh")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _rounded(metrics: RankingMetrics) -> dict[str, float]:
    return {key: round(value, 8) for key, value in metrics.as_dict().items()}


def _judged_coverage(ranked_ids: list[str], labels: dict[str, str]) -> float:
    if not ranked_ids:
        return 1.0
    return sum(document_id in labels for document_id in ranked_ids) / len(ranked_ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_summary(report: dict[str, Any], output_path: Path) -> None:
    dataset = report["dataset"]
    metrics = report["metrics"]["overall"]
    print(
        f"dataset: {dataset['dataset_version']} "
        f"({dataset['query_count']} queries / {dataset['product_count']} products)"
    )
    print(f"backend: {report['backend']['backend_id']}")
    print(f"Recall@{report['evaluation']['top_k']}: {metrics['recall_at_k']:.6f}")
    print(f"MRR@{report['evaluation']['top_k']}: {metrics['mrr_at_k']:.6f}")
    print(f"NDCG@{report['evaluation']['top_k']}: {metrics['ndcg_at_k']:.6f}")
    print(f"empty recall rate: {metrics['empty_recall']:.6f}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
