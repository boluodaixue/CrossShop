"""Normalize all 23,421 ShopSimulator items, tasks, queries, and qrels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from globex_agent.catalog import (
    normalize_shopsimulator_product,
    normalize_shopsimulator_task,
)
from globex_agent.domain import PriceSource

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "taobao_shopsimulator"
    / "fine_items_eval_train_all.json.gz"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
EXPECTED_ITEM_COUNT = 23_421
SOURCE_URL = "https://github.com/YYHDBL/shopping-grpo-longhorizon"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--allow-unexpected-count",
        action="store_true",
        help="Permit a source snapshot whose item count is not 23,421.",
    )
    args = parser.parse_args()

    source_rows = _read_source(args.input)
    if len(source_rows) != EXPECTED_ITEM_COUNT and not args.allow_unexpected_count:
        raise RuntimeError(
            f"expected {EXPECTED_ITEM_COUNT:,} source items, found {len(source_rows):,}"
        )

    ingested_at = datetime.now(timezone.utc)
    items = []
    tasks = []
    queries = []
    qrels = []
    for row in source_rows:
        items.append(
            normalize_shopsimulator_product(row, ingested_at=ingested_at)
        )
        task, query, judgment = normalize_shopsimulator_task(row)
        tasks.append(task)
        queries.append(query)
        qrels.append(judgment)

    _validate(items, tasks, queries, qrels)
    paths = _output_paths(args.output_root)
    _write_jsonl(paths["items"], (item.model_dump(mode="json") for item in items))
    _write_jsonl(paths["tasks"], (task.model_dump(mode="json") for task in tasks))
    _write_jsonl(paths["queries"], (query.model_dump(mode="json") for query in queries))
    _write_jsonl(paths["qrels"], (qrel.model_dump(mode="json") for qrel in qrels))

    split_counts = Counter(query.split for query in queries)
    source_split_counts = Counter(query.source_split for query in queries)
    price_counts = Counter(item.price_source.value for item in items)
    manifest = {
        "dataset_version": "shopsimulator-23421-normalized-v1",
        "source_url": SOURCE_URL,
        "source_file": args.input.name,
        "source_sha256": _sha256(args.input),
        "license": "not-declared-in-upstream-repository-review-before-redistribution",
        "ingested_at": ingested_at.isoformat(),
        "platform": "taobao",
        "locale": "cn",
        "language": "zh",
        "item_count": len(items),
        "task_count": len(tasks),
        "query_count": len(queries),
        "qrel_count": len(qrels),
        "split_counts": dict(sorted(split_counts.items())),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "price_counts": dict(sorted(price_counts.items())),
        "price_policy": (
            "Expose one distinct observed price; preserve multi-price variants and "
            "mark product price unavailable until a concrete variant is selected."
        ),
        "retrieval_judgment_scope": (
            "single target product per query; unjudged products are not explicit "
            "irrelevant judgments"
        ),
        "agent_task_policy": (
            "Keep full instruction, simple instruction, target options, and target "
            "attributes separate from the product catalog."
        ),
        "outputs": {
            key: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    _write_json(paths["manifest"], manifest)

    print(f"source items: {len(source_rows):,}")
    print(f"normalized items: {len(items):,}")
    print(f"tasks/queries/qrels: {len(tasks):,}/{len(queries):,}/{len(qrels):,}")
    print(f"splits: {dict(sorted(split_counts.items()))}")
    print(f"price availability: {dict(sorted(price_counts.items()))}")
    print(f"catalog: {paths['items']}")
    print(f"manifest: {paths['manifest']}")


def _read_source(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ShopSimulator source not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise RuntimeError("ShopSimulator source must be a JSON list of objects")
    return payload


def _validate(items: list[Any], tasks: list[Any], queries: list[Any], qrels: list[Any]) -> None:
    counts = {len(items), len(tasks), len(queries), len(qrels)}
    if len(counts) != 1:
        raise RuntimeError("item/task/query/qrel counts do not match")
    item_ids = {item.item_id for item in items}
    query_ids = {query.query_id for query in queries}
    if len(item_ids) != len(items):
        raise RuntimeError("duplicate normalized ShopSimulator item_id")
    if len(query_ids) != len(queries):
        raise RuntimeError("duplicate normalized ShopSimulator query_id")
    if any(task.target_item_id not in item_ids for task in tasks):
        raise RuntimeError("one or more tasks reference a missing item")
    if any(qrel.item_id not in item_ids or qrel.query_id not in query_ids for qrel in qrels):
        raise RuntimeError("one or more qrels reference a missing item/query")
    if any(item.price_source is PriceSource.SIMULATED for item in items):
        raise RuntimeError("source normalization must not invent simulated prices")


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "items": root / "catalogs" / "taobao" / "cn" / "items.jsonl",
        "tasks": root / "tasks" / "shopsimulator" / "tasks.jsonl",
        "queries": root / "eval" / "shopsimulator" / "queries.jsonl",
        "qrels": root / "eval" / "shopsimulator" / "qrels.jsonl",
        "manifest": root / "manifests" / "shopsimulator.json",
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
