"""Evaluate one platform/locale catalog with its real query and qrel files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from globex_agent.domain import MarketLocale, Platform, RelevanceJudgment, RetrievalQuery
from globex_agent.eval import RankingMetrics, aggregate_metrics, evaluate_ranking
from globex_agent.recall import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingSearchBackend,
    LocalizedFusionSearchBackend,
    SentenceTransformerTextEncoder,
    SQLiteFtsSearchBackend,
)
from globex_agent.recall.persistence import load_standard_item_documents_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("amazon", "taobao"))
    parser.add_argument("--locale", required=True, choices=("us", "es", "jp", "cn"))
    parser.add_argument("--backend", choices=("fts", "semantic", "hybrid"), default="fts")
    parser.add_argument(
        "--dataset",
        choices=("default", "shopsimulator_retrieval_v1"),
        default="default",
        help="Use the original source task labels or the separate pooled retrieval set.",
    )
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--query-overrides",
        type=Path,
        help=(
            "Optional JSONL with query_id/query/language for a separately labeled "
            "cross-language evaluation set."
        ),
    )
    parser.add_argument(
        "--lexical-query-overrides",
        type=Path,
        help=(
            "Optional JSONL with query_id/localized_query. It affects only the FTS "
            "branch of hybrid retrieval."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    platform = Platform(args.platform)
    locale = MarketLocale(args.locale)
    _validate_partition(platform, locale, parser)
    output_stem = f"{platform.value}-{locale.value}-{args.backend}-{args.split}"
    if args.dataset != "default":
        output_stem = f"{args.dataset}-{output_stem}"
    output = args.output or (
        PROJECT_ROOT
        / "output"
        / "eval"
        / f"{output_stem}.json"
    )
    report = run_evaluation(
        platform=platform,
        locale=locale,
        backend_name=args.backend,
        split=args.split,
        top_k=args.top_k,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
        query_overrides_path=args.query_overrides,
        lexical_overrides_path=args.lexical_query_overrides,
        dataset_name=args.dataset,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    print(f"partition: {platform.value}:{locale.value}")
    print(f"backend: {report['backend_id']}")
    print(f"queries: {report['query_count']:,}")
    labels = report["metric_labels"]
    print(f"{labels['recall']}@{args.top_k}: {metrics['recall_at_k']:.6f}")
    print(f"{labels['mrr']}@{args.top_k}: {metrics['mrr_at_k']:.6f}")
    print(f"{labels['ndcg']}@{args.top_k}: {metrics['ndcg_at_k']:.6f}")
    print(f"empty recall rate: {metrics['empty_recall']:.6f}")
    print(f"mean judged coverage: {report['mean_judged_coverage']:.6f}")
    print(f"report: {output}")


def run_evaluation(
    *,
    platform: Platform,
    locale: MarketLocale,
    backend_name: str,
    split: str,
    top_k: int,
    model_name: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
    query_overrides_path: Path | None,
    lexical_overrides_path: Path | None,
    dataset_name: str = "default",
) -> dict[str, Any]:
    processed = PROJECT_ROOT / "data" / "processed"
    if dataset_name == "shopsimulator_retrieval_v1":
        if platform is not Platform.TAOBAO or locale is not MarketLocale.CN:
            raise ValueError("shopsimulator_retrieval_v1 is only valid for taobao:cn")
        dataset = dataset_name
    else:
        dataset = "amazon_esci" if platform is Platform.AMAZON else "shopsimulator"
    query_path = processed / "eval" / dataset / locale.value / "queries.jsonl"
    qrel_path = processed / "eval" / dataset / locale.value / "qrels.jsonl"
    if platform is Platform.TAOBAO:
        query_path = processed / "eval" / dataset / "queries.jsonl"
        qrel_path = processed / "eval" / dataset / "qrels.jsonl"
    item_path = processed / "catalogs" / platform.value / locale.value / "items.jsonl"
    database_path = processed / "databases" / platform.value / "catalog.sqlite3"
    index_path = (
        PROJECT_ROOT
        / "output"
        / "index"
        / "catalog"
        / platform.value
        / locale.value
        / "bge-m3-hnsw-ip.faiss"
    )

    queries = [
        query
        for query in _load_models(query_path, RetrievalQuery)
        if query.split == split
    ]
    if not queries:
        raise RuntimeError(f"no {split} queries in {query_path}")
    judgments = _load_models(qrel_path, RelevanceJudgment)
    qrels_by_query: dict[str, list[RelevanceJudgment]] = defaultdict(list)
    for judgment in judgments:
        qrels_by_query[judgment.query_id].append(judgment)
    query_overrides = _load_overrides(query_overrides_path, "query")
    lexical_overrides = _load_overrides(
        lexical_overrides_path,
        "localized_query",
    )
    unknown_overrides = (set(query_overrides) | set(lexical_overrides)) - {
        query.query_id for query in queries
    }
    if unknown_overrides:
        raise RuntimeError(
            f"overrides reference unknown query IDs: {sorted(unknown_overrides)[:3]}"
        )

    lexical_backend = SQLiteFtsSearchBackend(
        database_path,
        platform=platform,
        locale=locale,
    )
    semantic_backend = None
    if backend_name in {"semantic", "hybrid"}:
        if not index_path.exists():
            raise FileNotFoundError(f"semantic index not found: {index_path}")
        documents = load_standard_item_documents_jsonl(item_path)
        encoder = SentenceTransformerTextEncoder(
            model_name,
            device=device,
            batch_size=batch_size,
            local_files_only=local_files_only,
        )
        semantic_backend = EmbeddingSearchBackend.from_faiss_hnsw(
            documents,
            encoder,
            index_path,
        )

    metrics: list[RankingMetrics] = []
    judged_coverages: list[float] = []
    cases: list[dict[str, Any]] = []
    for query in queries:
        semantic_query = query_overrides.get(query.query_id, query.query)
        if backend_name == "fts":
            result = lexical_backend.search(semantic_query, top_k=top_k)
        elif backend_name == "semantic":
            assert semantic_backend is not None
            result = semantic_backend.search(semantic_query, top_k=top_k)
        else:
            assert semantic_backend is not None
            localized_query = lexical_overrides.get(query.query_id, semantic_query)
            hybrid = LocalizedFusionSearchBackend(
                lexical_backend,
                semantic_backend,
                lexical_query_localizer=lambda _query, value=localized_query: value,
            )
            result = hybrid.search(semantic_query, top_k=top_k)

        query_qrels = qrels_by_query.get(query.query_id, [])
        if dataset == "shopsimulator_retrieval_v1":
            binary = {
                judgment.item_id: float(judgment.gain >= 2)
                for judgment in query_qrels
            }
        else:
            binary = {
                judgment.item_id: float(judgment.label == "Exact")
                for judgment in query_qrels
            }
        graded = {judgment.item_id: judgment.gain for judgment in query_qrels}
        ranked_ids = [hit.document_id for hit in result.hits]
        case_metrics = evaluate_ranking(
            ranked_ids,
            binary,
            top_k,
            ndcg_relevance=graded,
        )
        judged_ids = set(graded)
        coverage = (
            sum(item_id in judged_ids for item_id in ranked_ids) / len(ranked_ids)
            if ranked_ids
            else 1.0
        )
        metrics.append(case_metrics)
        judged_coverages.append(coverage)
        cases.append(
            {
                "query_id": query.query_id,
                "query": semantic_query,
                "source_query": query.query,
                "split": query.split,
                "returned": len(ranked_ids),
                "judged_coverage": round(coverage, 8),
                "metrics": _rounded(case_metrics),
            }
        )

    aggregate = aggregate_metrics(metrics)
    backend_id = (
        lexical_backend.backend_id
        if backend_name == "fts"
        else semantic_backend.backend_id
        if backend_name == "semantic" and semantic_backend is not None
        else "fusion:minmax-weighted-v1:localized-lexical-only"
    )
    single_target = platform is Platform.TAOBAO and dataset == "shopsimulator"
    metric_labels = (
        {
            "recall": "GoldTargetHit",
            "mrr": "GoldTargetMRR",
            "ndcg": "GoldTargetNDCG",
        }
        if single_target
        else {"recall": "Recall", "mrr": "MRR", "ndcg": "NDCG"}
    )
    return {
        "dataset": dataset,
        "platform": platform.value,
        "locale": locale.value,
        "split": split,
        "backend_id": backend_id,
        "top_k": top_k,
        "query_count": len(queries),
        "query_override_count": len(query_overrides),
        "lexical_override_count": len(lexical_overrides),
        "semantic_query_policy": "original evaluation query; never lexical translation",
        "lexical_query_policy": (
            "localized override when supplied; otherwise the evaluation query"
        ),
        "positive_policy": (
            "Exact/Substitute (gain >= 2) for Recall/MRR; 0-3 gain for NDCG"
            if dataset == "shopsimulator_retrieval_v1"
            else "Exact only for Recall/MRR; source gain for NDCG"
        ),
        "judgment_scope": (
            "ESCI selected query pool"
            if platform is Platform.AMAZON
            else "pooled multi-product judgments"
            if dataset == "shopsimulator_retrieval_v1"
            else "single source target; other products are unjudged"
        ),
        "unjudged_policy": "count as nonrelevant and report judged coverage",
        "metrics": _rounded(aggregate),
        "mean_judged_coverage": round(fmean(judged_coverages), 8),
        "metric_labels": metric_labels,
        "cases": cases,
    }


def _validate_partition(
    platform: Platform,
    locale: MarketLocale,
    parser: argparse.ArgumentParser,
) -> None:
    amazon_locales = {MarketLocale.US, MarketLocale.ES, MarketLocale.JP}
    valid = (platform is Platform.AMAZON and locale in amazon_locales) or (
        platform is Platform.TAOBAO and locale is MarketLocale.CN
    )
    if not valid:
        parser.error(f"invalid platform/locale partition: {platform.value}:{locale.value}")


def _load_models(path: Path, model: Any) -> list[Any]:
    with path.open(encoding="utf-8") as source:
        return [model.model_validate_json(line) for line in source if line.strip()]


def _load_overrides(path: Path | None, value_field: str) -> dict[str, str]:
    if path is None:
        return {}
    overrides: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", "")).strip()
            value = " ".join(str(row.get(value_field, "")).split())
            if not query_id or not value:
                raise RuntimeError(f"invalid override at {path}:{line_number}")
            if query_id in overrides:
                raise RuntimeError(f"duplicate override query_id: {query_id}")
            overrides[query_id] = value
    return overrides


def _rounded(metrics: RankingMetrics) -> dict[str, float]:
    return {key: round(value, 8) for key, value in metrics.as_dict().items()}


if __name__ == "__main__":
    main()
