"""Evaluate course-aligned CategoryCard BM25, KNN, Hybrid, and reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from globex_agent.category_insight import rerank_category_hits
from globex_agent.category_insight.reranking import RERANK_BYPASS_TOP_SCORE
from globex_agent.eval import RankingMetrics, aggregate_metrics, evaluate_ranking
from globex_agent.recall.category_kb import (
    DEFAULT_CATEGORY_INDEX,
    CategoryQueryType,
    OpenSearchCategoryKnowledgeBase,
    OpenSearchHttpClient,
    classify_category_query,
)
from globex_agent.recall.embedding import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerTextEncoder,
)
from globex_agent.recall.reranker import (
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    SubprocessCrossEncoderReranker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COARSE_K = 30
DEFAULT_TOP_K = 10
QUICK_TOP_K = 8
CARD_TYPES = ("bestseller", "attribute", "price_range")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "category_insight",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:9200")
    parser.add_argument("--index-name", default=DEFAULT_CATEGORY_INDEX)
    parser.add_argument(
        "--cards-filename", default="category_cards.jsonl"
    )
    parser.add_argument(
        "--cases-filename", default="category_recall_cases_v2.jsonl"
    )
    parser.add_argument(
        "--manifest-filename", default="category_recall_manifest_v2.json"
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    )
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-python", type=Path)
    parser.add_argument("--reranker-device", default="cuda:0")
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument(
        "--reranker-max-length", type=int, default=DEFAULT_RERANKER_MAX_LENGTH
    )
    parser.add_argument("--reranker-fp32", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--coarse-k", type=int, default=DEFAULT_COARSE_K)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "dev", "test", "final_test"),
        default=["train", "dev", "test"],
    )
    parser.add_argument(
        "--reranker-modes",
        nargs="+",
        choices=("summary_only", "contextual"),
        default=["contextual"],
    )
    parser.add_argument(
        "--rerank-all",
        action="store_true",
        help="rerank every coarse hit instead of using the top-score bypass",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "eval" / "category_recall_v2.json",
    )
    args = parser.parse_args()
    if args.coarse_k < args.top_k:
        parser.error("--coarse-k must be at least --top-k")
    if args.top_k != 10:
        parser.error("the frozen course gate uses --top-k 10")

    report = run_evaluation(
        args.data_dir,
        endpoint=args.endpoint,
        index_name=args.index_name,
        cards_filename=args.cards_filename,
        cases_filename=args.cases_filename,
        manifest_filename=args.manifest_filename,
        embedding_model=args.embedding_model,
        device=args.device,
        max_seq_length=args.max_seq_length,
        reranker_model=args.reranker_model,
        reranker_python=args.reranker_python,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_fp16=not args.reranker_fp32,
        local_files_only=args.local_files_only,
        coarse_k=args.coarse_k,
        top_k=args.top_k,
        splits=tuple(dict.fromkeys(args.splits)),
        reranker_modes=tuple(dict.fromkeys(args.reranker_modes)),
        rerank_all=args.rerank_all,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_summary(report, args.output)


def run_evaluation(
    data_dir: Path,
    *,
    endpoint: str,
    index_name: str,
    cards_filename: str,
    cases_filename: str,
    manifest_filename: str,
    embedding_model: str,
    device: str,
    max_seq_length: int,
    reranker_model: str,
    reranker_python: Path | None,
    reranker_device: str,
    reranker_batch_size: int,
    reranker_max_length: int,
    reranker_fp16: bool,
    local_files_only: bool,
    coarse_k: int,
    top_k: int,
    splits: tuple[str, ...],
    reranker_modes: tuple[str, ...],
    rerank_all: bool = False,
) -> dict[str, Any]:
    cards_path = data_dir / cards_filename
    cases_path = data_dir / cases_filename
    manifest_path = data_dir / manifest_filename
    cards = _read_jsonl(cards_path)
    cases = _read_jsonl(cases_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_cases(cards, cases, manifest)
    all_case_count = len(cases)
    cases = [case for case in cases if case["split"] in splits]
    if not cases:
        raise ValueError("the selected splits contain no evaluation cases")

    client = OpenSearchHttpClient(endpoint, timeout=30)
    server_version = client.request("GET", "")["version"]["number"]
    index_analyzers = _index_analyzers(client, index_name)
    indexed_count = client.request("GET", f"{index_name}/_count")["count"]
    if indexed_count != len(cards):
        raise RuntimeError(
            f"OpenSearch has {indexed_count} cards, expected {len(cards)}; rebuild index"
        )
    kb = OpenSearchCategoryKnowledgeBase(client, index_name=index_name)
    encoder = SentenceTransformerTextEncoder(
        embedding_model,
        device=device,
        max_seq_length=max_seq_length,
        local_files_only=local_files_only,
    )
    if reranker_python is None:
        reranker = CrossEncoderReranker(
            reranker_model,
            device=device,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            local_files_only=local_files_only,
        )
        reranker_execution = "in-process"
    else:
        reranker = SubprocessCrossEncoderReranker(
            reranker_python,
            reranker_model,
            device=reranker_device,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            use_fp16=reranker_fp16,
            local_files_only=local_files_only,
        )
        reranker_execution = "persistent subprocess"

    warmup_query = "category retrieval warmup"
    warmup_vector = encoder.encode_queries([warmup_query])[0]
    kb.search_bm25(warmup_query, top_k=top_k)
    kb.search(
        warmup_query,
        warmup_vector,
        top_k=coarse_k,
        query_type=CategoryQueryType.ATTRIBUTE_CONSTRAINT,
    )
    reranker.score_texts(
        warmup_query,
        [
            "Category: warmup. Knowledge type: product attribute distribution. "
            "Summary: category card summary warmup"
        ],
    )

    scheme_metrics: dict[
        str, list[tuple[dict[str, Any], RankingMetrics, list[str]]]
    ] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    per_query: list[dict[str, Any]] = []
    rerank_counts: dict[str, Counter[str]] = {
        mode: Counter() for mode in reranker_modes
    }
    classifier_counts: Counter[str] = Counter()
    try:
        for index, case in enumerate(cases, start=1):
            query = case["query"]
            expected_type = CategoryQueryType(case["query_type"])
            predicted_type = classify_category_query(query)
            classifier_counts["total"] += 1
            classifier_counts["correct"] += int(predicted_type == expected_type)

            started = time.perf_counter()
            bm25_hits = kb.search_bm25(query, top_k=top_k)
            bm25_ms = (time.perf_counter() - started) * 1000

            encode_started = time.perf_counter()
            query_vector = encoder.encode_queries([query])[0]
            encode_ms = (time.perf_counter() - encode_started) * 1000

            started = time.perf_counter()
            knn = kb.search(
                query,
                query_vector,
                top_k=coarse_k,
                query_type=CategoryQueryType.COLLOQUIAL,
            )
            knn_ms = encode_ms + (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            hybrid = kb.search(
                query,
                query_vector,
                top_k=coarse_k,
                query_type=expected_type,
            )
            hybrid_ms = encode_ms + (time.perf_counter() - started) * 1000

            rankings = {
                "bm25": [hit.card.card_id for hit in bm25_hits],
                "knn_bge_m3": [hit.card.card_id for hit in knn.hits[:top_k]],
                "hybrid_dynamic": [
                    hit.card.card_id for hit in hybrid.hits[:top_k]
                ],
            }
            timings = {
                "bm25": bm25_ms,
                "knn_bge_m3": knn_ms,
                "hybrid_dynamic": hybrid_ms,
            }
            rerank_diagnostics: dict[str, dict[str, Any]] = {}
            for mode in reranker_modes:
                started = time.perf_counter()
                reranked = rerank_category_hits(
                    query,
                    hybrid.hits,
                    reranker,
                    final_k=top_k,
                    document_mode=mode,
                    bypass_top_score=(
                        None if rerank_all else RERANK_BYPASS_TOP_SCORE
                    ),
                )
                rerank_ms = hybrid_ms + (time.perf_counter() - started) * 1000
                scheme_name = f"hybrid_bge_reranker_{mode}"
                rankings[scheme_name] = [
                    hit.card.card_id for hit in reranked.hits
                ]
                timings[scheme_name] = rerank_ms
                rerank_counts[mode][
                    "executed" if reranked.reranked else "bypassed"
                ] += 1
                if reranked.bypass_reason:
                    rerank_counts[mode][reranked.bypass_reason] += 1
                rerank_diagnostics[mode] = {
                    "reranked": reranked.reranked,
                    "bypass_reason": reranked.bypass_reason,
                }
            for scheme, ranked_ids in rankings.items():
                metric = evaluate_ranking(
                    ranked_ids,
                    case["relevance"],
                    top_k,
                    ndcg_relevance=case["graded_relevance"],
                )
                scheme_metrics[scheme].append((case, metric, ranked_ids))
                latencies[scheme].append(timings[scheme])

            for scheme, ranked_ids in rankings.items():
                quick_name = f"{scheme}_quick_top8"
                quick_ids = ranked_ids[:QUICK_TOP_K]
                quick_metric = evaluate_ranking(
                    quick_ids,
                    case["relevance"],
                    QUICK_TOP_K,
                    ndcg_relevance=case["graded_relevance"],
                )
                scheme_metrics[quick_name].append((case, quick_metric, quick_ids))
                latencies[quick_name].append(timings[scheme])
            per_query.append(
                {
                    "query_id": case["query_id"],
                    **(
                        {"intent_id": case["intent_id"]}
                        if "intent_id" in case
                        else {}
                    ),
                    **(
                        {"language": case["language"]}
                        if "language" in case
                        else {}
                    ),
                    **(
                        {"query_source_kind": case["query_source_kind"]}
                        if "query_source_kind" in case
                        else {}
                    ),
                    "split": case["split"],
                    "query_type": expected_type.value,
                    "predicted_query_type": predicted_type.value,
                    "query_type_match": predicted_type == expected_type,
                    "hybrid_weights": {
                        "semantic": hybrid.weights.semantic,
                        "lexical": hybrid.weights.lexical,
                    },
                    "hybrid_used_bm25": hybrid.used_bm25,
                    "coarse_top_score": hybrid.hits[0].score if hybrid.hits else None,
                    "reranker": rerank_diagnostics,
                    "rankings": rankings,
                    "latency_ms": timings,
                }
            )
            print(f"evaluated category query {index}/{len(cases)}")
    finally:
        close = getattr(reranker, "close", None)
        if callable(close):
            close()

    card_type_by_id = {card["card_id"]: card["card_type"] for card in cards}
    schemes = {
        scheme: _summarize_scheme(
            rows,
            latencies[scheme],
            top_k=QUICK_TOP_K if scheme.endswith("top8") else top_k,
            card_type_by_id=card_type_by_id,
        )
        for scheme, rows in scheme_metrics.items()
    }
    classifier_by_split = {
        split: _classifier_accuracy(
            [row for row in per_query if split == "overall" or row["split"] == split]
        )
        for split in ("overall", *splits)
    }
    return {
        "dataset": {
            "version": manifest["dataset_version"],
            "query_count": len(cases),
            "full_query_count": all_case_count,
            "card_count": len(cards),
            "split_counts": dict(sorted(Counter(case["split"] for case in cases).items())),
            "unjudged_policy": manifest["unjudged_policy"],
            "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "cards_sha256": hashlib.sha256(cards_path.read_bytes()).hexdigest(),
        },
        "configuration": {
            "opensearch_endpoint": endpoint,
            "opensearch_version": server_version,
            "index_name": index_name,
            "index_analyzer": index_analyzers["analyzer"],
            "search_analyzer": index_analyzers["search_analyzer"],
            "embedding_model": embedding_model,
            "embedding_max_seq_length": max_seq_length,
            "coarse_k": coarse_k,
            "gate_top_k": top_k,
            "quick_top_k": QUICK_TOP_K,
            "evaluated_splits": list(splits),
            "hybrid_weights_source": "course table selected by annotated Planner query type",
            "reranker_model": reranker_model,
            "reranker_execution": reranker_execution,
            "reranker_device": reranker_device if reranker_python else device,
            "reranker_precision": "fp16" if reranker_fp16 else "fp32",
            "reranker_max_length": reranker_max_length,
            "reranker_document_modes": list(reranker_modes),
            "contextual_document_policy": (
                "the versioned retrieval_text stored in OpenSearch; language policy "
                "comes from the index sidecar"
            ),
            "rerank_all": rerank_all,
            "rerank_bypass_top_score": (
                None if rerank_all else RERANK_BYPASS_TOP_SCORE
            ),
        },
        "schemes": schemes,
        "query_type_classifier": {
            "role": "fallback diagnostic; retrieval uses annotated Planner query type",
            "correct": classifier_counts["correct"],
            "total": classifier_counts["total"],
            "accuracy_by_split": classifier_by_split,
        },
        "reranker_usage": {
            mode: dict(sorted(counts.items()))
            for mode, counts in rerank_counts.items()
        },
        "per_query": per_query,
    }


def _summarize_scheme(
    rows: list[tuple[dict[str, Any], RankingMetrics, list[str]]],
    latency_values: list[float],
    *,
    top_k: int,
    card_type_by_id: dict[str, str],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, float]] = {}
    present_splits = sorted({str(case["split"]) for case, _, _ in rows})
    for split in ("overall", *present_splits):
        selected_rows = [
            row for row in rows if split == "overall" or row[0]["split"] == split
        ]
        metrics = aggregate_metrics([metric for _, metric, _ in selected_rows]).as_dict()
        metrics.update(
            _card_type_coverage(selected_rows, top_k, card_type_by_id)
        )
        by_split[split] = metrics
    by_type = {
        query_type: aggregate_metrics(
            [
                metric
                for case, metric, _ in rows
                if case["query_type"] == query_type
            ]
        ).as_dict()
        for query_type in sorted({case["query_type"] for case, _, _ in rows})
    }
    return {
        "metric_k": top_k,
        "metrics_by_split": by_split,
        "metrics_by_query_type": by_type,
        "metrics_by_language": _metrics_by_case_field(rows, "language"),
        "metrics_by_query_source_kind": _metrics_by_case_field(
            rows, "query_source_kind"
        ),
        "paired_language_gap": _paired_language_gap(rows),
        "latency_ms": {
            "p50": statistics.median(latency_values),
            "p99": _percentile(latency_values, 0.99),
        },
    }


def _card_type_coverage(
    rows: list[tuple[dict[str, Any], RankingMetrics, list[str]]],
    top_k: int,
    card_type_by_id: dict[str, str],
) -> dict[str, float]:
    if not rows:
        return {
            "bestseller_coverage_at_k": 0.0,
            "attribute_coverage_at_k": 0.0,
            "price_range_coverage_at_k": 0.0,
            "all_card_types_coverage_at_k": 0.0,
        }
    covered = Counter()
    for case, _, ranked_ids in rows:
        relevant_ids = set(case["relevant_card_ids"])
        retrieved_types = {
            card_type_by_id[card_id]
            for card_id in ranked_ids[:top_k]
            if card_id in relevant_ids
        }
        for card_type in CARD_TYPES:
            covered[card_type] += int(card_type in retrieved_types)
        covered["all"] += int(set(CARD_TYPES) <= retrieved_types)
    count = len(rows)
    return {
        "bestseller_coverage_at_k": covered["bestseller"] / count,
        "attribute_coverage_at_k": covered["attribute"] / count,
        "price_range_coverage_at_k": covered["price_range"] / count,
        "all_card_types_coverage_at_k": covered["all"] / count,
    }


def _classifier_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(row["query_type_match"] for row in rows) / len(rows)


def _metrics_by_case_field(
    rows: list[tuple[dict[str, Any], RankingMetrics, list[str]]], field: str
) -> dict[str, dict[str, float]]:
    values = sorted({str(case[field]) for case, _, _ in rows if field in case})
    return {
        value: aggregate_metrics(
            [metric for case, metric, _ in rows if str(case.get(field)) == value]
        ).as_dict()
        for value in values
    }


def _paired_language_gap(
    rows: list[tuple[dict[str, Any], RankingMetrics, list[str]]],
) -> dict[str, float | int | str]:
    pairs: dict[str, dict[str, RankingMetrics]] = defaultdict(dict)
    for case, metric, _ in rows:
        if "intent_id" in case and "language" in case:
            pairs[str(case["intent_id"])][str(case["language"])] = metric
    complete = [pair for pair in pairs.values() if set(pair) == {"en", "zh"}]
    if not complete:
        return {"pair_count": 0, "direction": "zh_minus_en"}
    output: dict[str, float | int | str] = {
        "pair_count": len(complete),
        "direction": "zh_minus_en",
    }
    for field in ("recall_at_k", "mrr_at_k", "ndcg_at_k"):
        gaps = [getattr(pair["zh"], field) - getattr(pair["en"], field) for pair in complete]
        output[field] = statistics.fmean(gaps)
        output[f"mean_absolute_{field}"] = statistics.fmean(abs(gap) for gap in gaps)
    return output


def _index_analyzers(
    client: OpenSearchHttpClient, index_name: str
) -> dict[str, str]:
    response = client.request("GET", f"{index_name}/_mapping")
    properties = response[index_name]["mappings"]["properties"]
    retrieval_mapping = properties["retrieval_text"]
    return {
        "analyzer": str(retrieval_mapping.get("analyzer", "standard")),
        "search_analyzer": str(
            retrieval_mapping.get(
                "search_analyzer", retrieval_mapping.get("analyzer", "standard")
            )
        ),
    }


def _validate_cases(
    cards: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    card_ids = {card["card_id"] for card in cards}
    if len(cases) != manifest["query_count"] or len(cards) != manifest["card_count"]:
        raise ValueError("manifest counts do not match checked-in data")
    positive_policy = manifest["positive_count_per_query"]
    if isinstance(positive_policy, dict):
        min_positives = int(positive_policy["min"])
        max_positives = int(positive_policy["max"])
    else:
        min_positives = max_positives = int(positive_policy)
    for case in cases:
        candidate_ids = set(case["candidate_ids"])
        if not (
            candidate_ids
            == card_ids
            == set(case["relevance"])
            == set(case["graded_relevance"])
            == set(case["judgments"])
        ):
            raise ValueError(f"query {case['query_id']} contains unjudged cards")
        if not min_positives <= len(case["relevant_card_ids"]) <= max_positives:
            raise ValueError(
                f"query {case['query_id']} must have between {min_positives} "
                f"and {max_positives} positives"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _print_summary(report: dict[str, Any], output: Path) -> None:
    print("\nCategory card retrieval")
    for split in report["configuration"]["evaluated_splits"]:
        print(f"[{split}]")
        for name, scheme in report["schemes"].items():
            metrics = scheme["metrics_by_split"][split]
            print(
                f"{name}: Recall@{scheme['metric_k']}={metrics['recall_at_k']:.4f} "
                f"MRR={metrics['mrr_at_k']:.4f} NDCG={metrics['ndcg_at_k']:.4f} "
                f"AllTypes={metrics['all_card_types_coverage_at_k']:.4f} "
                f"P50={scheme['latency_ms']['p50']:.2f}ms"
            )
    print(f"report: {output}")


if __name__ == "__main__":
    main()
