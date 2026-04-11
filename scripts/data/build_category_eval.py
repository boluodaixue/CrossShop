"""Expand human-authored card-query annotations into a fully judged eval set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "category-card-recall-en-v1"
EXPECTED_SPLITS = {"train": 30, "dev": 10, "test": 10}
EXPECTED_QUERY_TYPES = {
    "noun": 12,
    "attribute_constraint": 12,
    "style": 12,
    "colloquial": 14,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "category_insight" / "category_cards.jsonl"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "eval_query_annotations.jsonl"
        ),
    )
    parser.add_argument(
        "--source-esci-cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "recall_cases.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "category_insight",
    )
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument(
        "--cases-filename", default="category_recall_cases.jsonl"
    )
    parser.add_argument(
        "--manifest-filename", default="category_recall_manifest.json"
    )
    parser.add_argument(
        "--forbid-test-reuse-from",
        type=Path,
        help="Optional historical cases file whose test queries must not reappear.",
    )
    args = parser.parse_args()

    cards = _read_jsonl(args.cards)
    annotations = _read_jsonl(args.annotations)
    source_esci_cases = _read_jsonl(args.source_esci_cases)
    forbidden_test_queries: set[str] = set()
    if args.forbid_test_reuse_from:
        historical_cases = _read_jsonl(args.forbid_test_reuse_from)
        forbidden_test_queries = {
            " ".join(str(case["query"]).split()).casefold()
            for case in historical_cases
            if case["split"] == "test"
        }
    cases = _build_cases(
        cards,
        annotations,
        source_esci_cases,
        dataset_version=args.dataset_version,
        forbidden_test_queries=forbidden_test_queries,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / args.cases_filename
    manifest_path = args.output_dir / args.manifest_filename
    _write_jsonl(cases_path, cases)
    manifest = _build_manifest(
        cards=cards,
        cases=cases,
        input_paths={
            "cards": args.cards,
            "annotations": args.annotations,
            "source_esci_cases": args.source_esci_cases,
            **(
                {"historical_cases": args.forbid_test_reuse_from}
                if args.forbid_test_reuse_from
                else {}
            ),
        },
        cases_path=cases_path,
        dataset_version=args.dataset_version,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"built {len(cases)} fully judged queries over {len(cards)} cards; "
        f"splits={dict(Counter(case['split'] for case in cases))}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_cases(
    cards: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    source_esci_cases: list[dict[str, Any]],
    *,
    dataset_version: str = DATASET_VERSION,
    forbidden_test_queries: set[str] | None = None,
) -> list[dict[str, Any]]:
    if len(annotations) != 50:
        raise ValueError(f"expected 50 query annotations, got {len(annotations)}")
    card_by_id = {card["card_id"]: card for card in cards}
    if len(card_by_id) != len(cards):
        raise ValueError("card_id must be unique")
    candidate_ids = sorted(card_by_id)
    categories = {card["category"] for card in cards}
    source_query_texts = {case["query"].casefold() for case in source_esci_cases}

    cases: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    seen_queries: set[str] = set()
    forbidden_test_queries = forbidden_test_queries or set()
    for annotation in annotations:
        query_id = str(annotation["query_id"])
        query = " ".join(str(annotation["query"]).split())
        category = str(annotation["category"])
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        if query.casefold() in seen_queries:
            raise ValueError(f"duplicate query text: {query}")
        if query.casefold() in source_query_texts:
            raise ValueError(f"evaluation query copies an ESCI source query: {query}")
        if (
            annotation["split"] == "test"
            and query.casefold() in forbidden_test_queries
        ):
            raise ValueError(f"test query reuses a historical test query: {query}")
        if category not in categories:
            raise ValueError(f"unknown category in annotation: {category}")
        seen_query_ids.add(query_id)
        seen_queries.add(query.casefold())

        relevant_ids = [
            _role_to_card_id(category, role) for role in annotation["relevant_roles"]
        ]
        if len(relevant_ids) != 5 or len(set(relevant_ids)) != 5:
            raise ValueError(f"query {query_id} must have five distinct relevant cards")
        missing = set(relevant_ids) - card_by_id.keys()
        if missing:
            raise ValueError(f"query {query_id} references missing cards: {sorted(missing)}")
        if any(card_by_id[card_id]["category"] != category for card_id in relevant_ids):
            raise ValueError(f"query {query_id} has a cross-category relevant card")
        price_relevance = annotation.get("price_relevance")
        if price_relevance is not None:
            if price_relevance not in {"strong", "medium", "coverage_weak"}:
                raise ValueError(f"query {query_id} has invalid price_relevance")
            price_id = _role_to_card_id(category, "price-range-01")
            if price_id not in relevant_ids:
                raise ValueError(f"query {query_id} must include its price card")
            price_gain = 5 - relevant_ids.index(price_id)
            allowed_gains = {
                "strong": {4, 5},
                "medium": {2, 3},
                "coverage_weak": {1},
            }
            if price_gain not in allowed_gains[price_relevance]:
                raise ValueError(
                    f"query {query_id} price gain {price_gain} conflicts with "
                    f"{price_relevance}"
                )

        gain_by_id = {
            card_id: float(5 - rank) for rank, card_id in enumerate(relevant_ids)
        }
        cases.append(
            {
                "dataset_version": dataset_version,
                "query_id": query_id,
                "query": query,
                "query_type": annotation["query_type"],
                "category": category,
                "split": annotation["split"],
                **(
                    {"price_relevance": price_relevance}
                    if price_relevance is not None
                    else {}
                ),
                "candidate_ids": candidate_ids,
                "relevant_card_ids": relevant_ids,
                "relevance": {
                    card_id: float(card_id in gain_by_id) for card_id in candidate_ids
                },
                "graded_relevance": {
                    card_id: gain_by_id.get(card_id, 0.0) for card_id in candidate_ids
                },
                "judgments": {
                    card_id: (
                        f"relevant_rank_{6 - int(gain_by_id[card_id])}"
                        if card_id in gain_by_id
                        else "not_relevant"
                    )
                    for card_id in candidate_ids
                },
            }
        )

    split_counts = Counter(case["split"] for case in cases)
    type_counts = Counter(case["query_type"] for case in cases)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise ValueError(f"unexpected split counts: {dict(split_counts)}")
    if dict(type_counts) != EXPECTED_QUERY_TYPES:
        raise ValueError(f"unexpected query type counts: {dict(type_counts)}")
    for split in EXPECTED_SPLITS:
        present_types = {case["query_type"] for case in cases if case["split"] == split}
        if present_types != set(EXPECTED_QUERY_TYPES):
            raise ValueError(f"split {split} does not cover all query types")
    return cases


def _role_to_card_id(category: str, role: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", category.casefold()).strip("-")
    return f"cc-{slug}-{role}"


def _build_manifest(
    *,
    cards: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    input_paths: dict[str, Path],
    cases_path: Path,
    dataset_version: str = DATASET_VERSION,
) -> dict[str, Any]:
    return {
        "dataset_version": dataset_version,
        "language": "en",
        "selection_method": "human-authored category queries and relevant card roles",
        "query_count": len(cases),
        "card_count": len(cards),
        "positive_count_per_query": 5,
        "split_counts": dict(sorted(Counter(case["split"] for case in cases).items())),
        "query_type_counts": dict(
            sorted(Counter(case["query_type"] for case in cases).items())
        ),
        "candidate_pool_policy": "global card corpus; every card is explicitly judged",
        "unjudged_policy": "forbidden",
        "closed_pool": True,
        "relevance_policy": "five binary positives per query",
        "graded_gain_policy": "ordered relevant_card_ids receive gains 5,4,3,2,1",
        "price_card_policy": (
            "every query includes its category price card; explicit price intent gains "
            "4-5, broad category gains 2-3, other intent gains 1"
        ),
        "price_relevance_counts": dict(
            sorted(Counter(case.get("price_relevance", "unspecified") for case in cases).items())
        ),
        "source_query_reuse_policy": "verbatim ESCI source queries forbidden",
        "tuning_policy": "tune on train/dev; report test once after configuration freeze",
        "input_sha256": {key: _sha256(path) for key, path in input_paths.items()},
        "cases_sha256": _sha256(cases_path),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
