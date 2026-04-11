"""Evaluate the product-recall mainline on closed, fully judged ESCI pools.

The course mainline is Query/Item vector recall Top-100 followed by a
cross-encoder Top-10.  BM25 is reported as a baseline.  Hybrid retrieval is an
explicit optional ablation and never supplies candidates to the main reranker.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from globex_agent.eval import RankingMetrics, aggregate_metrics, evaluate_ranking
from globex_agent.recall import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    ITEM_TEXT_FORMAT_VERSION,
    CrossEncoderReranker,
    EmbeddingSearchBackend,
    FusionWeights,
    KeywordSearchBackend,
    RerankedSearchBackend,
    SearchBackend,
    SearchDocument,
    SentenceTransformerTextEncoder,
    SubprocessCrossEncoderReranker,
    WeightedFusionSearchBackend,
)
from globex_agent.recall.persistence import (
    load_search_documents_jsonl,
    sha256,
    write_index_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_K = 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reranker-python",
        type=Path,
        help="Python executable for the persistent GPU reranker worker",
    )
    parser.add_argument("--reranker-device", default="cuda:0")
    parser.add_argument("--reranker-batch-size", type=int, default=1)
    parser.add_argument("--reranker-fp32", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=DEFAULT_RERANKER_MAX_LENGTH,
    )
    parser.add_argument(
        "--include-hybrid-extension",
        action="store_true",
        help="also report a separately labelled BM25/vector Hybrid ablation",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        help="vector weight for the optional Hybrid ablation",
    )
    parser.add_argument(
        "--lexical-weight",
        type=float,
        help="BM25 weight for the optional Hybrid ablation",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--index-backend",
        choices=("faiss-hnsw", "exact"),
        default="faiss-hnsw",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "output" / "index" / "esci-bge-m3-hnsw-ip.faiss",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "eval" / "retrieval_comparison.json",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.candidate_k < args.top_k:
        parser.error("--candidate-k must be at least --top-k")
    has_both_hybrid_weights = (
        args.semantic_weight is not None and args.lexical_weight is not None
    )
    if args.include_hybrid_extension and not has_both_hybrid_weights:
        parser.error(
            "--include-hybrid-extension requires both --semantic-weight "
            "and --lexical-weight"
        )
    if not args.include_hybrid_extension and (
        args.semantic_weight is not None or args.lexical_weight is not None
    ):
        parser.error("Hybrid weights require --include-hybrid-extension")
    if args.index_backend == "faiss-hnsw" and args.index.suffix != ".faiss":
        parser.error("--index must end in .faiss for --index-backend faiss-hnsw")
    if args.index_backend == "exact" and args.index.suffix != ".npz":
        parser.error("--index must end in .npz for --index-backend exact")

    report = run_comparison(
        args.data_dir,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        index_path=args.index,
        device=args.device,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_seq_length=args.max_seq_length,
        reranker_max_length=args.reranker_max_length,
        reranker_python=args.reranker_python,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_fp16=not args.reranker_fp32,
        index_backend=args.index_backend,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        include_hybrid_extension=args.include_hybrid_extension,
        semantic_weight=args.semantic_weight,
        lexical_weight=args.lexical_weight,
        local_files_only=args.local_files_only,
        rebuild_index=args.rebuild_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_summary(report, args.output)


def run_comparison(
    data_dir: Path,
    *,
    embedding_model: str,
    reranker_model: str,
    index_path: Path,
    device: str,
    top_k: int,
    candidate_k: int,
    max_seq_length: int,
    reranker_max_length: int,
    reranker_python: Path | None,
    reranker_device: str,
    reranker_batch_size: int,
    reranker_fp16: bool,
    index_backend: str,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
    include_hybrid_extension: bool,
    semantic_weight: float | None,
    lexical_weight: float | None,
    local_files_only: bool,
    rebuild_index: bool,
) -> dict[str, Any]:
    items_path = data_dir / "recall_items.jsonl"
    cases_path = data_dir / "recall_cases.jsonl"
    manifest_path = data_dir / "recall_manifest.json"
    documents = load_search_documents_jsonl(items_path)
    cases = _read_jsonl(cases_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_closed_cases(cases, documents, manifest)

    encoder = SentenceTransformerTextEncoder(
        embedding_model,
        device=device,
        max_seq_length=max_seq_length,
        local_files_only=local_files_only,
    )
    global_embedding = _load_or_build_embedding(
        documents,
        encoder,
        index_path=index_path,
        items_path=items_path,
        rebuild=rebuild_index,
        index_backend=index_backend,
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )
    case_backends, case_documents = _build_base_case_backends(
        cases,
        documents,
        global_embedding,
    )

    hybrid_config: dict[str, Any] | None = None
    if include_hybrid_extension:
        if semantic_weight is None or lexical_weight is None:
            raise ValueError("Hybrid extension requires explicit weights")
        weights = FusionWeights(semantic=semantic_weight, lexical=lexical_weight)
        hybrid_config = {
            "role": "optional extension ablation; not part of the course mainline",
            "weight_source": "explicit CLI values; no automatic RAG-style routing",
            "semantic": weights.semantic,
            "lexical": weights.lexical,
        }

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
    for case in cases:
        query_id = case["query_id"]
        case_backends[query_id]["ann_reranker"] = RerankedSearchBackend(
            case_backends[query_id]["ann_recall"],
            case_documents[query_id],
            reranker,
            candidate_k=candidate_k,
        )
        if include_hybrid_extension:
            case_backends[query_id]["hybrid_extension"] = (
                WeightedFusionSearchBackend(
                    case_backends[query_id]["bm25_baseline"],
                    case_backends[query_id]["ann_recall"],
                    candidate_k=candidate_k,
                    weights=weights,
                )
            )

    evaluations: dict[str, dict[str, Any]] = {}
    per_case: dict[str, dict[str, dict[str, Any]]] = {}
    backend_names = ["bm25_baseline", "ann_recall", "ann_reranker"]
    if include_hybrid_extension:
        backend_names.append("hybrid_extension")
    for name in backend_names:
        print(f"evaluating: {name}", flush=True)
        evaluation, case_metrics = _evaluate_backend(
            case_backends,
            name,
            cases,
            top_k=top_k,
        )
        evaluations[name] = evaluation
        per_case[name] = case_metrics

    coarse_recall = {
        "bm25_baseline": _evaluate_recall(
            case_backends, "bm25_baseline", cases, candidate_k
        ),
        "ann_recall": _evaluate_recall(
            case_backends, "ann_recall", cases, candidate_k
        ),
    }
    coarse_recall["ann_reranker"] = coarse_recall["ann_recall"]
    if include_hybrid_extension:
        coarse_recall["hybrid_extension"] = _evaluate_recall(
            case_backends, "hybrid_extension", cases, candidate_k
        )
    for name in evaluations:
        evaluations[name][f"candidate_recall_at_{candidate_k}"] = round(
            coarse_recall[name], 8
        )

    pool_sizes = [len(case["candidate_ids"]) for case in cases]
    positive_counts = [
        sum(label == "Exact" for label in case["labels"].values())
        for case in cases
    ]
    return {
        "architecture": {
            "towers": ["query", "item"],
            "user_tower": False,
            "course_mainline": (
                f"Query/Item vector recall top-{candidate_k} -> "
                f"cross-encoder top-{top_k}"
            ),
            "local_vector_index": global_embedding.index_id,
            "index_parameters": global_embedding.index_parameters,
            "bm25_role": "baseline only",
            "hybrid_role": "optional extension ablation only",
        },
        "models": {
            "embedding": embedding_model,
            "embedding_max_seq_length": max_seq_length,
            "item_text_format_version": ITEM_TEXT_FORMAT_VERSION,
            "reranker": reranker_model,
            "reranker_max_length": reranker_max_length,
            "embedding_device": device,
            "reranker_device": reranker_device if reranker_python else device,
            "reranker_precision": "fp16" if reranker_fp16 else "fp32",
            "reranker_execution": reranker_execution,
        },
        "dataset": {
            "dataset_version": manifest["dataset_version"],
            "query_count": len(cases),
            "document_count": len(documents),
            "candidate_pool_size": _range_summary(pool_sizes),
            "exact_positive_count": _range_summary(positive_counts),
            "closed_pool": True,
            "unjudged_policy": "forbidden",
            "language_counts": dict(
                sorted(
                    {
                        language: sum(
                            case.get("language") == language for case in cases
                        )
                        for language in {
                            str(case["language"])
                            for case in cases
                            if "language" in case
                        }
                    }.items()
                )
            ),
            "items_sha256": sha256(items_path),
            "cases_sha256": sha256(cases_path),
        },
        "evaluation": {
            "top_k": top_k,
            "candidate_k": candidate_k,
            "primary_holdout_split": "test",
            "recall_mrr_positive_labels": ["Exact"],
            "ndcg_gain_mapping": manifest["gain_mapping"],
            "warmup_excluded_from_latency": True,
            "note": (
                "Recall at candidate_k is bounded by each closed judged pool; "
                "candidate_k may exceed the pool size."
            ),
        },
        "hybrid_extension": hybrid_config,
        "results": evaluations,
        "bad_cases_vs_bm25": _compare_cases(
            per_case,
            baseline="bm25_baseline",
        ),
    }


def _load_or_build_embedding(
    documents: list[SearchDocument],
    encoder: SentenceTransformerTextEncoder,
    *,
    index_path: Path,
    items_path: Path,
    rebuild: bool,
    index_backend: str,
    hnsw_m: int,
    ef_construction: int,
    ef_search: int,
) -> EmbeddingSearchBackend:
    manifest_path = index_path.with_suffix(".manifest.json")
    can_reuse = index_path.exists() and manifest_path.exists() and not rebuild
    if can_reuse:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "encoder_model": encoder.encoder_id,
            "items_sha256": sha256(items_path),
            "document_count": len(documents),
            "max_seq_length": encoder.max_seq_length,
            "text_format_version": ITEM_TEXT_FORMAT_VERSION,
            "index_type": (
                "faiss-hnsw-inner-product-v1"
                if index_backend == "faiss-hnsw"
                else "exact-cosine-inner-product-v1"
            ),
        }
        can_reuse = all(manifest.get(key) == value for key, value in expected.items())
        if can_reuse and index_backend == "faiss-hnsw":
            expected_parameters = {
                "metric": "inner_product",
                "normalized": True,
                "m": hnsw_m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
            }
            can_reuse = manifest.get("index_parameters") == expected_parameters
    if can_reuse:
        print(f"loading index: {index_path}", flush=True)
        if index_backend == "faiss-hnsw":
            return EmbeddingSearchBackend.from_faiss_hnsw(
                documents,
                encoder,
                index_path,
                m=hnsw_m,
                ef_construction=ef_construction,
                ef_search=ef_search,
            )
        return EmbeddingSearchBackend.from_index(documents, encoder, index_path)

    print(f"building index: {index_path}", flush=True)
    if index_backend == "faiss-hnsw":
        backend = EmbeddingSearchBackend.with_faiss_hnsw(
            documents,
            encoder,
            m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
    else:
        backend = EmbeddingSearchBackend(documents, encoder)
    backend.save_index(index_path)
    write_index_manifest(
        index_path,
        items_path=items_path,
        model_name=encoder.encoder_id,
        document_count=len(documents),
        dimension=backend.dimension,
        max_seq_length=encoder.max_seq_length,
        text_format_version=ITEM_TEXT_FORMAT_VERSION,
        index_type=backend.index_id,
        index_parameters=backend.index_parameters,
    )
    return backend


def _build_base_case_backends(
    cases: list[dict[str, Any]],
    documents: list[SearchDocument],
    global_embedding: EmbeddingSearchBackend,
) -> tuple[
    dict[str, dict[str, SearchBackend]],
    dict[str, list[SearchDocument]],
]:
    document_by_id = {document.document_id: document for document in documents}
    backends: dict[str, dict[str, SearchBackend]] = {}
    case_documents: dict[str, list[SearchDocument]] = {}
    for case in cases:
        query_id = case["query_id"]
        selected = [
            document_by_id[document_id] for document_id in case["candidate_ids"]
        ]
        case_documents[query_id] = selected
        backends[query_id] = {
            "bm25_baseline": KeywordSearchBackend(selected),
            "ann_recall": global_embedding.subset(selected),
        }
    return backends, case_documents


def _evaluate_backend(
    case_backends: dict[str, dict[str, SearchBackend]],
    backend_name: str,
    cases: list[dict[str, Any]],
    *,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    first_case = cases[0]
    first_backend = case_backends[first_case["query_id"]][backend_name]
    first_backend.search(first_case["query"], top_k=top_k)
    all_metrics: list[RankingMetrics] = []
    by_split: dict[str, list[RankingMetrics]] = defaultdict(list)
    by_language: dict[str, list[RankingMetrics]] = defaultdict(list)
    by_source_kind: dict[str, list[RankingMetrics]] = defaultdict(list)
    latencies_ms: list[float] = []
    coverages: list[float] = []
    fallbacks = 0
    case_metrics: dict[str, dict[str, Any]] = {}
    for case in cases:
        backend = case_backends[case["query_id"]][backend_name]
        started = time.perf_counter_ns()
        result = backend.search(case["query"], top_k=top_k)
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        ranked_ids = [hit.document_id for hit in result.hits]
        coverage = _judged_coverage(ranked_ids, case["labels"])
        if coverage != 1.0:
            raise RuntimeError(f"unjudged result for query_id={case['query_id']}")
        metrics = evaluate_ranking(
            ranked_ids,
            case["relevance"],
            top_k,
            ndcg_relevance=case["graded_relevance"],
        )
        all_metrics.append(metrics)
        by_split[case["split"]].append(metrics)
        if "language" in case:
            by_language[str(case["language"])].append(metrics)
        if "query_source_kind" in case:
            by_source_kind[str(case["query_source_kind"])].append(metrics)
        coverages.append(coverage)
        fallbacks += int(result.fallback_reason is not None)
        case_metrics[case["query_id"]] = {
            "query": case["query"],
            **(
                {"language": case["language"]}
                if "language" in case
                else {}
            ),
            "split": case["split"],
            "candidate_pool_size": len(case["candidate_ids"]),
            "exact_positive_count": sum(
                label == "Exact" for label in case["labels"].values()
            ),
            "judged_coverage": coverage,
            "ranked_ids": ranked_ids,
            "metrics": _rounded(metrics),
            "fallback_reason": result.fallback_reason,
        }

    return (
        {
            "backend_id": first_backend.backend_id,
            "metrics": {
                "overall": _rounded(aggregate_metrics(all_metrics)),
                "by_split": {
                    split: _rounded(aggregate_metrics(by_split.get(split, [])))
                    for split in ("train", "dev", "test")
                },
                "by_language": {
                    language: _rounded(aggregate_metrics(metrics))
                    for language, metrics in sorted(by_language.items())
                },
                "by_query_source_kind": {
                    source_kind: _rounded(aggregate_metrics(metrics))
                    for source_kind, metrics in sorted(by_source_kind.items())
                },
            },
            "judged_coverage": round(statistics.fmean(coverages), 8),
            "latency_ms": _latency_summary(latencies_ms),
            "fallback_queries": fallbacks,
        },
        case_metrics,
    )


def _evaluate_recall(
    case_backends: dict[str, dict[str, SearchBackend]],
    backend_name: str,
    cases: list[dict[str, Any]],
    candidate_k: int,
) -> float:
    recalls = []
    for case in cases:
        backend = case_backends[case["query_id"]][backend_name]
        result = backend.search(case["query"], top_k=candidate_k)
        ranked_ids = [hit.document_id for hit in result.hits]
        if _judged_coverage(ranked_ids, case["labels"]) != 1.0:
            raise RuntimeError(f"unjudged result for query_id={case['query_id']}")
        metrics = evaluate_ranking(ranked_ids, case["relevance"], candidate_k)
        recalls.append(metrics.recall_at_k)
    return statistics.fmean(recalls) if recalls else 0.0


def _validate_closed_cases(
    cases: list[dict[str, Any]],
    documents: list[SearchDocument],
    manifest: dict[str, Any],
) -> None:
    if not manifest.get("closed_pool") or manifest.get("unjudged_policy") != "forbidden":
        raise RuntimeError("evaluation requires a closed, fully judged dataset manifest")
    document_ids = {document.document_id for document in documents}
    for case in cases:
        candidate_ids = set(case["candidate_ids"])
        if not (
            candidate_ids
            == set(case["labels"])
            == set(case["relevance"])
            == set(case["graded_relevance"])
        ):
            raise RuntimeError(f"incomplete judgments for query_id={case['query_id']}")
        if not candidate_ids.issubset(document_ids):
            raise RuntimeError(f"missing candidate document for query_id={case['query_id']}")


def _judged_coverage(ranked_ids: list[str], labels: dict[str, str]) -> float:
    if not ranked_ids:
        return 1.0
    return sum(document_id in labels for document_id in ranked_ids) / len(ranked_ids)


def _compare_cases(
    per_case: dict[str, dict[str, dict[str, Any]]],
    *,
    baseline: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    comparison: dict[str, dict[str, list[dict[str, Any]]]] = {}
    baseline_cases = per_case[baseline]
    for name, cases in per_case.items():
        if name == baseline:
            continue
        deltas = []
        for query_id, case in cases.items():
            baseline_case = baseline_cases[query_id]
            delta = (
                case["metrics"]["ndcg_at_k"]
                - baseline_case["metrics"]["ndcg_at_k"]
            )
            deltas.append(
                {
                    "query_id": query_id,
                    "query": case["query"],
                    "ndcg_delta": round(delta, 8),
                    "bm25_top_ids": baseline_case["ranked_ids"][:3],
                    "candidate_top_ids": case["ranked_ids"][:3],
                }
            )
        comparison[name] = {
            "largest_gains": sorted(
                deltas,
                key=lambda row: (-row["ndcg_delta"], row["query_id"]),
            )[:3],
            "largest_regressions": sorted(
                deltas,
                key=lambda row: (row["ndcg_delta"], row["query_id"]),
            )[:3],
        }
    return comparison


def _range_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "maximum": 0, "mean": 0.0}
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": round(statistics.fmean(values), 4),
    }


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p99": 0.0}
    return {
        "mean": round(statistics.fmean(ordered), 4),
        "p50": round(_percentile(ordered, 0.50), 4),
        "p99": round(_percentile(ordered, 0.99), 4),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _rounded(metrics: RankingMetrics) -> dict[str, float]:
    return {key: round(value, 8) for key, value in metrics.as_dict().items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _print_summary(report: dict[str, Any], output_path: Path) -> None:
    top_k = report["evaluation"]["top_k"]
    candidate_k = report["evaluation"]["candidate_k"]
    print("\nproduct retrieval (test split; closed, fully judged pools)")
    print(f"course mainline: ANN Top-{candidate_k} -> Reranker Top-{top_k}")
    print("backend              Recall  MRR     NDCG    Recall(candidates)  P50ms   P99ms")
    for name, result in report["results"].items():
        metrics = result["metrics"]["by_split"]["test"]
        latency = result["latency_ms"]
        print(
            f"{name:<20}"
            f"{metrics['recall_at_k']:<8.4f}"
            f"{metrics['mrr_at_k']:<8.4f}"
            f"{metrics['ndcg_at_k']:<8.4f}"
            f"{result[f'candidate_recall_at_{candidate_k}']:<20.4f}"
            f"{latency['p50']:<8.2f}"
            f"{latency['p99']:<8.2f}"
        )
    print(f"final metrics: Recall@{top_k} / MRR@{top_k} / NDCG@{top_k}")
    print(f"candidate diagnostic: Recall@{candidate_k}")
    if report["hybrid_extension"] is not None:
        extension = report["hybrid_extension"]
        print(
            "optional Hybrid ablation: "
            f"vector={extension['semantic']:.1f}, BM25={extension['lexical']:.1f}"
        )
    print("judged coverage: 1.000000")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
