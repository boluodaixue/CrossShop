"""Build, machine-label, and finalize a pooled Chinese product retrieval set."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from globex_agent.domain import (
    MarketLocale,
    Platform,
    RelevanceJudgment,
    RetrievalQuery,
    ShoppingTask,
    StandardItem,
)
from globex_agent.eval.shopsimulator_retrieval import (
    DATASET_VERSION,
    ItemRelevanceAnnotation,
    QueryRelevanceAnnotation,
    RetrievalPoolCandidate,
    RetrievalPoolCase,
    SelectedRetrievalTask,
    annotation_prompt,
    pool_fingerprint,
    query_needs_review,
    select_retrieval_tasks,
    stable_audit_query_ids,
    validate_annotation_for_case,
)
from globex_agent.recall import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerTextEncoder,
    SQLiteFtsSearchBackend,
)
from globex_agent.recall.persistence import load_standard_item_documents_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROCESSED / "eval" / "shopsimulator_retrieval_v1"
ITEMS_PATH = PROCESSED / "catalogs" / "taobao" / "cn" / "items.jsonl"
TASKS_PATH = PROCESSED / "tasks" / "shopsimulator" / "tasks.jsonl"
DATABASE_PATH = PROCESSED / "databases" / "taobao" / "catalog.sqlite3"
INDEX_PATH = (
    PROJECT_ROOT
    / "output"
    / "index"
    / "catalog"
    / "taobao"
    / "cn"
    / "bge-m3-hnsw-ip.faiss"
)


class _ModelBatch(BaseModel):
    grades: list[int] = Field(min_length=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("pool", "label", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-count", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--final-dev-count", type=int, default=15)
    parser.add_argument("--final-test-count", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=60)
    parser.add_argument("--model-env", default="LLM_MAIN")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=240)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.pool_size < 20:
        parser.error("--pool-size must be at least 20")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    if args.stage in {"pool", "all"}:
        build_pool(
            args.output_root,
            dev_count=args.dev_count,
            test_count=args.test_count,
            pool_size=args.pool_size,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
    if args.stage in {"label", "all"}:
        asyncio.run(
            label_pool(
                args.output_root,
                model_env=args.model_env,
                concurrency=args.concurrency,
                request_timeout=args.request_timeout,
                limit=args.limit,
            )
        )
    if args.stage in {"finalize", "all"}:
        finalize(
            args.output_root,
            dev_count=args.final_dev_count,
            test_count=args.final_test_count,
        )


def build_pool(
    output_root: Path,
    *,
    dev_count: int,
    test_count: int,
    pool_size: int,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> None:
    items = _load_models(ITEMS_PATH, StandardItem)
    tasks = _load_models(TASKS_PATH, ShoppingTask)
    items_by_id = {item.item_id: item for item in items}
    selected = select_retrieval_tasks(
        tasks,
        items_by_id,
        dev_count=dev_count,
        test_count=test_count,
    )

    # EmbeddingSearchBackend sorts documents by document_id before constructing or
    # loading an index. Direct batched Faiss search must use the identical position
    # mapping; JSONL source order is not the persisted index order.
    documents = _load_index_ordered_documents(ITEMS_PATH)
    encoder = SentenceTransformerTextEncoder(
        DEFAULT_EMBEDDING_MODEL,
        device=device,
        batch_size=batch_size,
        local_files_only=local_files_only,
    )
    embeddings = encoder.encode_queries([row.query for row in selected])
    index = faiss.read_index(str(INDEX_PATH))
    if int(index.ntotal) != len(documents):
        raise RuntimeError("Faiss index count does not match sorted catalog documents")
    semantic_scores, semantic_indexes = index.search(embeddings, 100)
    if semantic_indexes.shape != (len(selected), 100):
        raise RuntimeError("unexpected semantic search result shape")

    lexical = SQLiteFtsSearchBackend(
        DATABASE_PATH,
        platform=Platform.TAOBAO,
        locale=MarketLocale.CN,
    )
    items_by_category: dict[tuple[str, ...], list[StandardItem]] = defaultdict(list)
    for item in items:
        items_by_category[tuple(item.category_path)].append(item)

    cases = []
    for row_index, selected_task in enumerate(selected):
        lexical_hits = lexical.search(selected_task.query, top_k=100).hits
        semantic_hits = [
            (documents[int(index_id)].document_id, float(score))
            for index_id, score in zip(
                semantic_indexes[row_index], semantic_scores[row_index], strict=True
            )
            if int(index_id) >= 0
        ]
        source_rankings = {
            "fts": [(hit.document_id, hit.score) for hit in lexical_hits],
            "semantic": semantic_hits,
        }
        source_rankings["hybrid"] = _hybrid_ranking(source_rankings, top_k=10)
        category_items = items_by_category[tuple(selected_task.category_path)]
        source_rankings["category_attributes"] = _category_attribute_ranking(
            selected_task,
            category_items,
            items_by_id,
            top_k=15,
        )
        source_rankings["category_sample"] = [
            (item.item_id, 0.0)
            for item in sorted(
                category_items,
                key=lambda item: _stable_key(
                    f"{selected_task.query_id}:sample:{item.item_id}"
                ),
            )[:3]
        ]
        case = _build_pool_case(
            selected_task,
            source_rankings,
            items_by_id,
            pool_size=pool_size,
        )
        cases.append(case)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "selected_tasks.jsonl", selected)
    _write_jsonl(output_root / "candidate_pools.jsonl", cases)
    print(f"selected queries: {len(selected):,}")
    print(f"dev/test: {dev_count:,}/{test_count:,}")
    print(f"mean pool size: {np.mean([len(case.candidates) for case in cases]):.2f}")
    print(f"candidate pools: {output_root / 'candidate_pools.jsonl'}")


async def label_pool(
    output_root: Path,
    *,
    model_env: str,
    concurrency: int,
    request_timeout: float,
    limit: int | None,
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    required = ["OPENAI_API_KEY", "OPENAI_BASE_URL", model_env]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"missing model configuration: {', '.join(missing)}")
    model_name = os.environ[model_env]
    model = init_chat_model(
        model_name,
        model_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=0,
    )
    cases = _load_models(output_root / "candidate_pools.jsonl", RetrievalPoolCase)
    cases_by_id = {case.query_id: case for case in cases}
    annotation_path = output_root / "annotations.jsonl"
    completed = {}
    for row in _load_models_if_exists(annotation_path, QueryRelevanceAnnotation):
        case = cases_by_id.get(row.query_id)
        if case is None:
            raise RuntimeError(f"annotation references unknown query: {row.query_id}")
        validate_annotation_for_case(row, case)
        review_status = "needs_review" if query_needs_review(row, case) else "machine_initial"
        completed[row.query_id] = row.model_copy(update={"review_status": review_status})
    pending = [case for case in cases if case.query_id not in completed]
    if limit is not None:
        pending = pending[:limit]
    semaphore = asyncio.Semaphore(concurrency)

    async def annotate(case: RetrievalPoolCase) -> QueryRelevanceAnnotation:
        async with semaphore:
            last_error: Exception | None = None
            for _attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        model.ainvoke(annotation_prompt(case)),
                        timeout=request_timeout,
                    )
                    payload = _parse_json_response(_response_text(response.content))
                    parsed = _ModelBatch.model_validate(payload)
                    if len(parsed.grades) != len(case.candidates):
                        raise ValueError(
                            "model grade count does not match candidate count"
                        )
                    grade_to_label = {
                        3: "Exact",
                        2: "Substitute",
                        1: "Partial",
                        0: "Irrelevant",
                    }
                    if any(grade not in grade_to_label for grade in parsed.grades):
                        raise ValueError("model grades must be between zero and three")
                    annotation = QueryRelevanceAnnotation(
                        query_id=case.query_id,
                        pool_fingerprint=pool_fingerprint(case),
                        annotator_model=model_name,
                        annotation_policy="llm_silver_v1",
                        review_status="machine_initial",
                        items=[
                            ItemRelevanceAnnotation(
                                item_id=candidate.item_id,
                                label=grade_to_label[grade],
                                gain=float(grade),
                            )
                            for candidate, grade in zip(
                                case.candidates,
                                parsed.grades,
                                strict=True,
                            )
                        ],
                    )
                    validate_annotation_for_case(annotation, case)
                    if query_needs_review(annotation, case):
                        annotation = annotation.model_copy(
                            update={"review_status": "needs_review"}
                        )
                    return annotation
                except Exception as exc:  # noqa: BLE001 - retry external model boundary
                    last_error = exc
            raise RuntimeError(f"failed to label {case.query_id}: {last_error}")

    for completed_count, future in enumerate(
        asyncio.as_completed([annotate(case) for case in pending]),
        start=1,
    ):
        annotation = await future
        completed[annotation.query_id] = annotation
        _write_jsonl(
            annotation_path,
            sorted(completed.values(), key=lambda row: _stable_key(row.query_id)),
        )
        if completed_count % 10 == 0 or completed_count == len(pending):
            print(f"labeled this run: {completed_count:,}/{len(pending):,}", flush=True)
    print(f"annotation model: {model_name}")
    print(f"annotations total: {len(completed):,}")


def finalize(output_root: Path, *, dev_count: int, test_count: int) -> None:
    selected = _load_models(output_root / "selected_tasks.jsonl", SelectedRetrievalTask)
    cases = _load_models(output_root / "candidate_pools.jsonl", RetrievalPoolCase)
    annotations = _load_models(
        output_root / "annotations.jsonl", QueryRelevanceAnnotation
    )
    cases_by_id = {case.query_id: case for case in cases}
    annotations_by_id = {row.query_id: row for row in annotations}
    missing = {row.query_id for row in selected} - set(annotations_by_id)
    if missing:
        raise RuntimeError(f"missing annotations for {len(missing)} queries")

    valid_rows: dict[str, list[SelectedRetrievalTask]] = {"dev": [], "test": []}
    invalid_query_ids = []
    for row in selected:
        case = cases_by_id[row.query_id]
        annotation = annotations_by_id[row.query_id]
        validate_annotation_for_case(annotation, case)
        needs_review = query_needs_review(annotation, case)
        if needs_review:
            invalid_query_ids.append(row.query_id)
            continue
        valid_rows[row.split].append(row)

    all_valid_query_ids = {
        row.query_id for split_rows in valid_rows.values() for row in split_rows
    }
    targets = {"dev": dev_count, "test": test_count}
    for split, count in targets.items():
        valid_rows[split].sort(key=lambda row: _stable_key(row.query_id))
        if len(valid_rows[split]) < count:
            raise RuntimeError(
                f"only {len(valid_rows[split])} valid {split} queries; need {count}"
            )
        valid_rows[split] = valid_rows[split][:count]
    chosen = valid_rows["dev"] + valid_rows["test"]
    chosen_ids = {row.query_id for row in chosen}
    reserve_query_ids = all_valid_query_ids - chosen_ids

    queries = []
    qrels = []
    for row in chosen:
        annotation = annotations_by_id[row.query_id]
        queries.append(
            RetrievalQuery(
                query_id=row.query_id,
                query=row.query,
                language="zh",
                platform=Platform.TAOBAO,
                locale=MarketLocale.CN,
                split=row.split,
                source=DATASET_VERSION,
                source_split="shopsimulator-eval-reannotated",
            )
        )
        qrels.extend(
            RelevanceJudgment(
                query_id=row.query_id,
                item_id=item.item_id,
                label=item.label,
                gain=item.gain,
            )
            for item in annotation.items
        )

    query_ids = {query.query_id for query in queries}
    audit_ids = stable_audit_query_ids(query_ids, ratio=0.1)
    _write_jsonl(output_root / "queries.jsonl", queries)
    _write_jsonl(output_root / "qrels.jsonl", qrels)
    _write_jsonl(
        output_root / "review_queue.jsonl",
        (
            {
                "query_id": query_id,
                "reason": "fewer than two Exact/Substitute positives",
            }
            for query_id in sorted(invalid_query_ids, key=_stable_key)
        ),
    )
    _write_jsonl(
        output_root / "valid_reserve.jsonl",
        (
            {"query_id": query_id, "reason": "valid reserve beyond frozen split size"}
            for query_id in sorted(reserve_query_ids, key=_stable_key)
        ),
    )
    _write_jsonl(
        output_root / "audit_sample.jsonl",
        (
            {
                "query_id": query_id,
                "review_status": "pending_human_review",
            }
            for query_id in sorted(audit_ids, key=_stable_key)
        ),
    )
    positives = defaultdict(int)
    for qrel in qrels:
        if qrel.gain >= 2:
            positives[qrel.query_id] += 1
    manifest = {
        "dataset_version": DATASET_VERSION,
        "source": "ShopSimulator eval queries re-annotated over the 23,421-item catalog",
        "query_policy": (
            "prefer ShopSimulator simple_query and fall back to cleaned full query; "
            "do not expose source target to annotator"
        ),
        "judgment_policy": "pooled LLM silver labels; Exact/Substitute are binary positives",
        "pool_policy": [
            "SQLite FTS5 top10 (retrieved to depth 100 for pooling)",
            "BGE-M3 top10 (retrieved to depth 100 for pooling)",
            "Hybrid 0.7/0.3 top10 computed from both depth-100 branches",
            "same-category attribute top15",
            "same-category stable sample 3",
            "source target for provenance coverage",
        ],
        "unjudged_policy": "unjudged catalog products are unknown, not explicit negatives",
        "human_review_required_for_resume_claims": True,
        "query_count": len(queries),
        "dev_count": sum(query.split == "dev" for query in queries),
        "test_count": sum(query.split == "test" for query in queries),
        "qrel_count": len(qrels),
        "selected_candidate_query_count": len(selected),
        "review_queue_count": len(invalid_query_ids),
        "valid_reserve_count": len(reserve_query_ids),
        "audit_sample_count": len(audit_ids),
        "positive_per_query": {
            "min": min(positives.values(), default=0),
            "mean": round(float(np.mean(list(positives.values()))), 4) if positives else 0,
            "max": max(positives.values(), default=0),
        },
        "files": {
            name: {
                "sha256": _sha256(output_root / name),
                "path": str((output_root / name).relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
            for name in (
                "selected_tasks.jsonl",
                "candidate_pools.jsonl",
                "annotations.jsonl",
                "queries.jsonl",
                "qrels.jsonl",
                "review_queue.jsonl",
                "valid_reserve.jsonl",
                "audit_sample.jsonl",
            )
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    print(f"finalized queries: {len(queries):,}")
    print(f"qrels: {len(qrels):,}")
    print(f"review queue: {len(invalid_query_ids):,}")
    print(f"valid reserve: {len(reserve_query_ids):,}")
    print(f"audit sample: {len(audit_ids):,}")


def _build_pool_case(
    selected: SelectedRetrievalTask,
    rankings: dict[str, list[tuple[str, float]]],
    items_by_id: dict[str, StandardItem],
    *,
    pool_size: int,
) -> RetrievalPoolCase:
    ranks_by_item: dict[str, dict[str, int]] = defaultdict(dict)
    for source, ranking in rankings.items():
        for rank, (item_id, _score) in enumerate(ranking, start=1):
            ranks_by_item[item_id][source] = rank
    ranks_by_item[selected.source_target_item_id]["source_target"] = 1

    item_ids = [selected.source_target_item_id]
    source_limits = {
        "category_attributes": 15,
        "category_sample": 3,
        "fts": 10,
        "semantic": 10,
        "hybrid": 10,
    }
    for source, limit in source_limits.items():
        for item_id, _score in rankings[source][:limit]:
            if item_id not in item_ids:
                item_ids.append(item_id)
    item_ids = item_ids[:pool_size]
    candidates = []
    for item_id in item_ids:
        item = items_by_id[item_id]
        source_ranks = ranks_by_item[item_id]
        candidates.append(
            RetrievalPoolCandidate(
                item_id=item.item_id,
                title=item.title,
                category_path=item.category_path,
                description=_truncate(item.description, 180),
                source_attributes=[
                    _truncate(str(value), 40)
                    for value in item.attributes.get("source_attributes", [])[:12]
                ],
                variant_values=[
                    _variant_summary(variant)
                    for variant in item.variants[:12]
                    if str(variant.get("value", "")).strip()
                ],
                price_summary=_price_summary(item),
                pool_sources=sorted(source_ranks),
                source_ranks=dict(sorted(source_ranks.items())),
            )
        )
    return RetrievalPoolCase(
        query_id=selected.query_id,
        source_task_id=selected.source_task_id,
        source_target_item_id=selected.source_target_item_id,
        query=selected.query,
        split=selected.split,
        category_path=selected.category_path,
        candidates=candidates,
    )


def _load_index_ordered_documents(path: Path) -> list[Any]:
    documents = sorted(
        load_standard_item_documents_jsonl(path),
        key=lambda document: document.document_id,
    )
    if len({document.document_id for document in documents}) != len(documents):
        raise RuntimeError("catalog documents must have unique IDs")
    return documents


def _category_attribute_ranking(
    selected: SelectedRetrievalTask,
    category_items: Sequence[StandardItem],
    items_by_id: dict[str, StandardItem],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    target = items_by_id[selected.source_target_item_id]
    target_attributes = _normalized_attributes(target)
    query_terms = _character_bigrams(selected.query)
    ranked = []
    for item in category_items:
        if item.item_id == target.item_id:
            continue
        overlap = len(target_attributes & _normalized_attributes(item))
        lexical = _jaccard(query_terms, _character_bigrams(item.title))
        ranked.append((item.item_id, overlap * 10 + lexical))
    return sorted(ranked, key=lambda row: (-row[1], row[0]))[:top_k]


def _hybrid_ranking(
    rankings: dict[str, list[tuple[str, float]]],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    lexical = _min_max(dict(rankings["fts"]))
    semantic = _min_max(dict(rankings["semantic"]))
    item_ids = set(lexical) | set(semantic)
    scored = [
        (item_id, 0.7 * semantic.get(item_id, 0) + 0.3 * lexical.get(item_id, 0))
        for item_id in item_ids
    ]
    return sorted(scored, key=lambda row: (-row[1], row[0]))[:top_k]


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return {item_id: 1.0 for item_id in scores}
    return {item_id: (score - low) / (high - low) for item_id, score in scores.items()}


def _price_summary(item: StandardItem) -> str | None:
    if item.price_cny is not None:
        return f"CNY {item.price_cny}"
    values = [
        Decimal(str(variant["price_cny"]))
        for variant in item.variants
        if variant.get("price_cny") is not None
    ]
    if values:
        return f"CNY {min(values)}-{max(values)}（按规格）"
    return None


def _variant_summary(variant: dict[str, Any]) -> str:
    value = _truncate(str(variant.get("value", "")), 48)
    price = variant.get("price_cny")
    return f"{value} @ CNY {price}" if price is not None else value


def _parse_json_response(text: str) -> dict[str, Any]:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response does not contain a JSON object")
    payload = json.loads(clean[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response root must be an object")
    return payload


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _normalized_attributes(item: StandardItem) -> set[str]:
    return {
        " ".join(str(value).casefold().split())
        for value in item.attributes.get("source_attributes", [])
        if str(value).strip()
    }


def _character_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\W+", "", text.casefold())
    return {compact[index : index + 2] for index in range(max(0, len(compact) - 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _truncate(value: str, length: int) -> str:
    clean = " ".join(value.split())
    return clean if len(clean) <= length else clean[: length - 1] + "…"


def _load_models(path: Path, model: Any) -> list[Any]:
    with path.open(encoding="utf-8") as source:
        return [model.model_validate_json(line) for line in source if line.strip()]


def _load_models_if_exists(path: Path, model: Any) -> list[Any]:
    return _load_models(path, model) if path.exists() else []


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            payload = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
            target.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(value: str) -> str:
    return hashlib.sha256(f"{DATASET_VERSION}:{value}".encode()).hexdigest()


if __name__ == "__main__":
    main()
