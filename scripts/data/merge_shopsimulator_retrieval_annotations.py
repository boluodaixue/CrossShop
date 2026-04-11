"""Merge independently written annotation shards after strict pool validation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from globex_agent.eval.shopsimulator_retrieval import (
    QueryRelevanceAnnotation,
    RetrievalPoolCase,
    validate_annotation_for_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eval"
    / "shopsimulator_retrieval_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=DATASET_ROOT / "annotations.jsonl",
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=DATASET_ROOT / "luna_shards",
    )
    parser.add_argument(
        "--pools",
        type=Path,
        default=DATASET_ROOT / "candidate_pools.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "annotations.jsonl",
    )
    args = parser.parse_args()

    cases = {
        case.query_id: case for case in _load_models(args.pools, RetrievalPoolCase)
    }
    sources = [args.base]
    if args.shard_dir.exists():
        sources.extend(sorted(args.shard_dir.glob("*.jsonl")))
    annotations: dict[str, QueryRelevanceAnnotation] = {}
    provenance: dict[str, Path] = {}
    for path in sources:
        if not path.exists():
            continue
        for annotation in _load_models(path, QueryRelevanceAnnotation):
            case = cases.get(annotation.query_id)
            if case is None:
                raise RuntimeError(
                    f"{path} references unknown query: {annotation.query_id}"
                )
            validate_annotation_for_case(annotation, case)
            existing = annotations.get(annotation.query_id)
            if existing is not None:
                if existing.model_dump() != annotation.model_dump():
                    raise RuntimeError(
                        f"conflicting annotation for {annotation.query_id}: "
                        f"{provenance[annotation.query_id]} versus {path}"
                    )
                continue
            annotations[annotation.query_id] = annotation
            provenance[annotation.query_id] = path

    _write_jsonl(
        args.output,
        sorted(annotations.values(), key=lambda row: row.query_id),
    )
    print(f"sources read: {len([path for path in sources if path.exists()]):,}")
    print(f"merged annotations: {len(annotations):,}/{len(cases):,}")
    print(f"output: {args.output}")


def _load_models(path: Path, model: Any) -> list[Any]:
    with path.open(encoding="utf-8") as source:
        return [model.model_validate_json(line) for line in source if line.strip()]


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
