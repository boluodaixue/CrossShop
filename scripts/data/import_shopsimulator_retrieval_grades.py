"""Validate compact offline grade batches and write annotation shards."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from globex_agent.eval.shopsimulator_retrieval import (
    ItemRelevanceAnnotation,
    QueryRelevanceAnnotation,
    RetrievalPoolCase,
    pool_fingerprint,
    query_needs_review,
    validate_annotation_for_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOLS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eval"
    / "shopsimulator_retrieval_v1"
    / "candidate_pools.jsonl"
)
GRADE_TO_LABEL = {
    3: "Exact",
    2: "Substitute",
    1: "Partial",
    0: "Irrelevant",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pools", type=Path, default=DEFAULT_POOLS)
    parser.add_argument("--annotator-model", default="gpt-5.6-luna")
    args = parser.parse_args()

    cases = {
        case.query_id: case for case in _load_models(args.pools, RetrievalPoolCase)
    }
    existing = {
        row.query_id: row
        for row in _load_models_if_exists(args.output, QueryRelevanceAnnotation)
    }
    grade_rows = _read_grade_rows(args.grades)
    for row in grade_rows:
        query_id = str(row.get("query_id", "")).strip()
        grades = row.get("grades")
        if query_id not in cases:
            raise ValueError(f"unknown query_id in grade batch: {query_id}")
        if query_id in existing:
            raise ValueError(f"query_id is already present in output shard: {query_id}")
        case = cases[query_id]
        if not isinstance(grades, list) or len(grades) != len(case.candidates):
            raise ValueError(
                f"{query_id} needs exactly {len(case.candidates)} integer grades"
            )
        if any(type(grade) is not int or grade not in GRADE_TO_LABEL for grade in grades):
            raise ValueError(f"{query_id} grades must be integers from zero to three")
        annotation = QueryRelevanceAnnotation(
            query_id=query_id,
            pool_fingerprint=pool_fingerprint(case),
            annotator_model=args.annotator_model,
            annotation_policy="llm_silver_v1",
            review_status="machine_initial",
            items=[
                ItemRelevanceAnnotation(
                    item_id=candidate.item_id,
                    label=GRADE_TO_LABEL[grade],
                    gain=float(grade),
                )
                for candidate, grade in zip(case.candidates, grades, strict=True)
            ],
        )
        validate_annotation_for_case(annotation, case)
        if query_needs_review(annotation, case):
            annotation = annotation.model_copy(update={"review_status": "needs_review"})
        existing[query_id] = annotation

    _write_jsonl(args.output, sorted(existing.values(), key=lambda row: row.query_id))
    print(f"imported grade rows: {len(grade_rows):,}")
    print(f"annotations in shard: {len(existing):,}")
    print(f"output: {args.output}")


def _read_grade_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise ValueError("JSON grade batch must be a list of objects")
        return payload
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL grade batch rows must be objects")
    return rows


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
            target.write(
                json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)


if __name__ == "__main__":
    main()
