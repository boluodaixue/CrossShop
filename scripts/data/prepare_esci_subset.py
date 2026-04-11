"""Build a small query-grouped ESCI subset through the Hugging Face viewer API.

The upstream source is Amazon's Apache-2.0 Shopping Queries Dataset. The
``tasksource/esci`` mirror joins the official example and product tables, which
lets this script download only the selected rows instead of the multi-GB corpus.
No training or evaluation code depends on Hugging Face at runtime after the
generated JSONL files have been checked in.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET = "tasksource/esci"
CONFIG = "default"
LOCALE = "us"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "amazon-esci-task1-en-us-closed-recall-v2"
TRAIN_CASES = 24
DEV_CASES = 8
TEST_CASES = 8
DISCOVERY_OFFSETS = (0, 1000, 10000, 100000, 500000, 1000000, 1500000)
GAIN_BY_LABEL = {
    "Exact": 1.0,
    "Complement": 0.1,
    "Substitute": 0.01,
    "Irrelevant": 0.0,
}
MIN_EXACT_POSITIVES = 2
MAX_EXACT_POSITIVES = 5
ALLOWED_LABELS = frozenset(GAIN_BY_LABEL)
OFFICIAL_SOURCE_URL = "https://github.com/amazon-science/esci-data"
MIRROR_URL = "https://huggingface.co/datasets/tasksource/esci"
API_ROOT = "https://datasets-server.huggingface.co"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval",
        help="Directory for recall_cases.jsonl, recall_items.jsonl, and manifest.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download again even when all generated files already exist.",
    )
    parser.add_argument(
        "--migrate-existing",
        action="store_true",
        help=(
            "Rewrite the existing v1 files into the closed, fully judged v2 format "
            "without downloading data."
        ),
    )
    args = parser.parse_args()

    paths = _output_paths(args.output_dir)
    if args.migrate_existing:
        _migrate_existing(paths)
        print(f"migrated ESCI subset to {DATASET_VERSION}: {args.output_dir}")
        return
    if not args.refresh and all(path.exists() for path in paths.values()):
        _validate_existing(paths)
        print(f"ESCI subset already prepared: {args.output_dir}")
        return

    train_query_ids = _discover_query_ids("train", TRAIN_CASES + DEV_CASES)
    test_query_ids = _discover_query_ids("test", TEST_CASES)
    rows = [
        *_fetch_query_groups("train", train_query_ids),
        *_fetch_query_groups("test", test_query_ids),
    ]
    cases, documents = _build_records(rows, train_query_ids, test_query_ids)
    _validate_records(cases, documents)
    source_query_ids = {
        "train": [str(value) for value in train_query_ids[:TRAIN_CASES]],
        "dev": [str(value) for value in train_query_ids[TRAIN_CASES:]],
        "test": [str(value) for value in test_query_ids],
    }
    _write_outputs(paths, cases, documents, source_query_ids)
    print(
        f"prepared {len(cases)} queries / {len(documents)} unique products "
        f"in {args.output_dir}"
    )


def _discover_query_ids(split: str, target_count: int) -> list[int]:
    query_ids: list[int] = []
    seen: set[int] = set()
    for offset in DISCOVERY_OFFSETS:
        payload = _request_json(
            "/rows",
            dataset=DATASET,
            config=CONFIG,
            split=split,
            offset=offset,
            length=100,
        )
        for wrapped in payload["rows"]:
            row = wrapped["row"]
            query_id = int(row["query_id"])
            if (
                int(row["small_version"]) != 1
                or row["product_locale"] != LOCALE
                or query_id in seen
            ):
                continue
            seen.add(query_id)
            query_ids.append(query_id)
            if len(query_ids) == target_count:
                return query_ids
    raise RuntimeError(
        f"could only discover {len(query_ids)} qualifying {split} queries; "
        f"need {target_count}"
    )


def _fetch_query_groups(split: str, query_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(query_ids, start=1):
        offset = 0
        while True:
            payload = _request_json(
                "/filter",
                dataset=DATASET,
                config=CONFIG,
                split=split,
                where=f'"query_id"={query_id}',
                offset=offset,
                length=100,
            )
            page = [wrapped["row"] for wrapped in payload["rows"]]
            rows.extend(page)
            total = int(payload["num_rows_total"])
            offset += len(page)
            if offset >= total:
                break
            if not page:
                raise RuntimeError(
                    f"empty {split} page before reaching declared total {total}"
                )
        print(f"downloaded {split} query group {index}/{len(query_ids)}")
    return rows


def _request_json(path: str, **params: object) -> dict[str, Any]:
    url = f"{API_ROOT}{path}?{urlencode(params)}"
    last_error = "unknown error"
    for attempt in range(1, 13):
        try:
            request = Request(url, headers={"User-Agent": "globex-agent-learning/0.1"})
            with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS host
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = str(exc)
        else:
            if "error" not in payload:
                return payload
            last_error = str(payload["error"])
        if attempt < 12:
            print(f"dataset API not ready ({last_error}); retry {attempt}/12")
            time.sleep(5)
    raise RuntimeError(f"dataset API failed: {last_error}")


def _build_records(
    rows: list[dict[str, Any]],
    train_query_ids: list[int],
    test_query_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    train_set = set(train_query_ids[:TRAIN_CASES])
    dev_set = set(train_query_ids[TRAIN_CASES:])
    test_set = set(test_query_ids)
    rows_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_query[int(row["query_id"])].append(row)

    source_documents: dict[str, dict[str, str]] = {}
    documents: dict[str, dict[str, str]] = {}
    cases: list[dict[str, Any]] = []
    ordered_ids = [*train_query_ids, *test_query_ids]
    for query_id in ordered_ids:
        query_rows = rows_by_query.get(query_id, [])
        if not query_rows:
            raise RuntimeError(f"downloaded no rows for query_id={query_id}")
        split = "train" if query_id in train_set else "dev" if query_id in dev_set else "test"
        if query_id not in train_set | dev_set | test_set:
            raise RuntimeError(f"query_id={query_id} was not assigned to a split")

        labels: dict[str, str] = {}
        for row in sorted(query_rows, key=lambda value: str(value["product_id"])):
            product_id = str(row["product_id"])
            label = str(row["esci_label"])
            labels[product_id] = label
            title = _clean_text(row.get("product_title"))
            source_documents.setdefault(
                product_id,
                {
                    "document_id": product_id,
                    "title": title,
                    "body": _clean_product_body(
                        title,
                        _clean_text(row.get("product_text")),
                    ),
                },
            )
        case = _build_closed_case(
            query_id=str(query_id),
            query=_clean_text(query_rows[0]["query"]),
            split=split,
            labels=labels,
        )
        if case is None:
            continue
        cases.append(case)
        for product_id in case["candidate_ids"]:
            documents[product_id] = source_documents[product_id]
    return cases, documents


def _build_closed_case(
    *,
    query_id: str,
    query: str,
    split: str,
    labels: dict[str, str],
) -> dict[str, Any] | None:
    exact_ids = sorted(
        product_id for product_id, label in labels.items() if label == "Exact"
    )
    if len(exact_ids) < MIN_EXACT_POSITIVES:
        return None

    retained_exact_ids = set(exact_ids[:MAX_EXACT_POSITIVES])
    retained_labels = {
        product_id: label
        for product_id, label in sorted(labels.items())
        if label != "Exact" or product_id in retained_exact_ids
    }
    candidate_ids = sorted(retained_labels)
    return {
        "dataset_version": DATASET_VERSION,
        "query_id": query_id,
        "query": query,
        "split": split,
        "candidate_ids": candidate_ids,
        "relevance": {
            product_id: float(retained_labels[product_id] == "Exact")
            for product_id in candidate_ids
        },
        "graded_relevance": {
            product_id: GAIN_BY_LABEL[retained_labels[product_id]]
            for product_id in candidate_ids
        },
        "labels": retained_labels,
    }


def _validate_records(
    cases: list[dict[str, Any]],
    documents: dict[str, dict[str, str]],
) -> None:
    actual_counts = {
        split: sum(case["split"] == split for case in cases)
        for split in ("train", "dev", "test")
    }
    if any(count == 0 for count in actual_counts.values()):
        raise RuntimeError(f"one or more splits became empty: {actual_counts}")
    if len({case["query_id"] for case in cases}) != len(cases):
        raise RuntimeError("query_id leaked across splits")
    referenced_ids: set[str] = set()
    for case in cases:
        if case.get("dataset_version") != DATASET_VERSION:
            raise RuntimeError(f"wrong dataset version for query_id={case['query_id']}")
        if not case["query"]:
            raise RuntimeError(f"empty query for query_id={case['query_id']}")
        candidate_ids = case["candidate_ids"]
        candidate_set = set(candidate_ids)
        if len(candidate_set) != len(candidate_ids):
            raise RuntimeError(f"duplicate candidate for query_id={case['query_id']}")
        if not (
            candidate_set
            == set(case["labels"])
            == set(case["relevance"])
            == set(case["graded_relevance"])
        ):
            raise RuntimeError(f"unjudged candidate for query_id={case['query_id']}")
        if not set(case["labels"].values()).issubset(ALLOWED_LABELS):
            raise RuntimeError(f"unknown ESCI label for query_id={case['query_id']}")
        exact_ids = {
            product_id
            for product_id, label in case["labels"].items()
            if label == "Exact"
        }
        binary_positive_ids = {
            product_id
            for product_id, gain in case["relevance"].items()
            if gain > 0
        }
        if exact_ids != binary_positive_ids:
            raise RuntimeError(f"Recall positives are not Exact-only: {case['query_id']}")
        if not MIN_EXACT_POSITIVES <= len(exact_ids) <= MAX_EXACT_POSITIVES:
            raise RuntimeError(f"invalid Exact count for query_id={case['query_id']}")
        if not candidate_set.issubset(documents):
            raise RuntimeError(f"missing product text for query_id={case['query_id']}")
        referenced_ids.update(candidate_set)
    if referenced_ids != set(documents):
        raise RuntimeError("recall_items contains documents outside the judged pools")
    for document in documents.values():
        title = document["title"]
        body = document["body"]
        if title and body.startswith(title):
            raise RuntimeError(f"body repeats title for document_id={document['document_id']}")
        if "None" in body.split():
            raise RuntimeError(
                f"body contains literal None for document_id={document['document_id']}"
            )


def _write_outputs(
    paths: dict[str, Path],
    cases: list[dict[str, Any]],
    documents: dict[str, dict[str, str]],
    source_query_ids: dict[str, list[str]],
) -> None:
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(paths["cases"], cases)
    _write_jsonl(paths["items"], [documents[key] for key in sorted(documents)])
    manifest = {
        "dataset_version": DATASET_VERSION,
        "official_source_url": OFFICIAL_SOURCE_URL,
        "download_mirror_url": MIRROR_URL,
        "license": "Apache-2.0",
        "locale": LOCALE,
        "task_version": "small_version=1",
        "random_seed": None,
        "selection_method": (
            "first qualifying query IDs at fixed viewer offsets; no random sampling"
        ),
        "source_split_policy": (
            "24 upstream train -> train; 8 upstream train -> dev; "
            "8 upstream test -> test; exclude query groups with fewer than 2 Exact"
        ),
        "source_query_ids": source_query_ids,
        "query_ids": {
            split: [case["query_id"] for case in cases if case["split"] == split]
            for split in ("train", "dev", "test")
        },
        "excluded_query_ids": {
            split: sorted(
                set(source_query_ids.get(split, []))
                - {case["query_id"] for case in cases if case["split"] == split}
            )
            for split in ("train", "dev", "test")
        },
        "gain_mapping": GAIN_BY_LABEL,
        "positive_labels_for_recall_mrr": ["Exact"],
        "exact_positive_count_per_query": {
            "minimum": MIN_EXACT_POSITIVES,
            "maximum": MAX_EXACT_POSITIVES,
            "mean": statistics.fmean(
                sum(label == "Exact" for label in case["labels"].values())
                for case in cases
            ),
        },
        "candidate_pool_policy": (
            "per-query closed pool; every candidate has an explicit ESCI judgment"
        ),
        "unjudged_policy": "forbidden",
        "closed_pool": True,
        "surplus_exact_policy": (
            "retain the first 5 sorted Exact product IDs; remove surplus Exact rows "
            "from that query pool without relabeling"
        ),
        "item_text_policy": (
            "title stored once; remove leading duplicate title and literal None tokens"
        ),
        "query_count": len(cases),
        "product_count": len(documents),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_existing(paths: dict[str, Path]) -> None:
    with paths["cases"].open(encoding="utf-8") as source:
        cases = [json.loads(line) for line in source if line.strip()]
    with paths["items"].open(encoding="utf-8") as source:
        document_rows = [json.loads(line) for line in source if line.strip()]
    documents = {row["document_id"]: row for row in document_rows}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    _validate_records(cases, documents)
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise RuntimeError("existing ESCI subset uses an unexpected dataset version")
    if manifest.get("query_count") != len(cases):
        raise RuntimeError("existing ESCI manifest query_count does not match cases")
    if manifest.get("product_count") != len(documents):
        raise RuntimeError("existing ESCI manifest product_count does not match items")


def _migrate_existing(paths: dict[str, Path]) -> None:
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"cannot migrate missing files: {missing}")

    with paths["cases"].open(encoding="utf-8") as source:
        source_cases = [json.loads(line) for line in source if line.strip()]
    with paths["items"].open(encoding="utf-8") as source:
        source_document_rows = [json.loads(line) for line in source if line.strip()]
    source_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    source_documents = {
        row["document_id"]: {
            "document_id": row["document_id"],
            "title": _clean_text(row.get("title")),
            "body": _clean_product_body(
                _clean_text(row.get("title")),
                _clean_text(row.get("body")),
            ),
        }
        for row in source_document_rows
    }
    cases: list[dict[str, Any]] = []
    documents: dict[str, dict[str, str]] = {}
    for source_case in source_cases:
        case = _build_closed_case(
            query_id=str(source_case["query_id"]),
            query=_clean_text(source_case["query"]),
            split=str(source_case["split"]),
            labels={
                str(product_id): str(label)
                for product_id, label in source_case["labels"].items()
            },
        )
        if case is None:
            continue
        cases.append(case)
        for product_id in case["candidate_ids"]:
            try:
                documents[product_id] = source_documents[product_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"missing source product text for document_id={product_id}"
                ) from exc

    source_query_ids = source_manifest.get("source_query_ids") or source_manifest.get(
        "query_ids"
    )
    if not isinstance(source_query_ids, dict):
        source_query_ids = {
            split: [
                str(case["query_id"])
                for case in source_cases
                if case["split"] == split
            ]
            for split in ("train", "dev", "test")
        }
    normalized_source_ids = {
        split: [str(query_id) for query_id in source_query_ids.get(split, [])]
        for split in ("train", "dev", "test")
    }
    _validate_records(cases, documents)
    _write_outputs(paths, cases, documents, normalized_source_ids)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    path.write_text(content, encoding="utf-8")


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "cases": output_dir / "recall_cases.jsonl",
        "items": output_dir / "recall_items.jsonl",
        "manifest": output_dir / "recall_manifest.json",
    }


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _clean_product_body(title: str, body: str) -> str:
    clean_title = _clean_text(title)
    clean_body = _clean_text(body)
    if clean_title and clean_body.startswith(clean_title):
        clean_body = clean_body[len(clean_title) :].lstrip(" :-|\t\n")
    return " ".join(token for token in clean_body.split() if token != "None")


if __name__ == "__main__":
    main()
