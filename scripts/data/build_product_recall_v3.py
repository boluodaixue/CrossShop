"""Build a separate fully judged bilingual product-recall v3 diagnostic.

This synthetic benchmark never replaces the real ESCI report. It reuses the
versioned CategoryCard source facts, selects five deterministic Exact products
per attribute-constrained intent, and constructs a 120-item closed pool with
same-category hard negatives plus cross-category negatives. English/Chinese
query pairs share the same labels and split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "product-recall-synthetic-bilingual-v3"
POOL_SIZE = 120
EXACT_COUNT = 5
FACET_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facts",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "category_item_facts_v3.jsonl"
        ),
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "category_generation_specs_v3.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "product_v3",
    )
    args = parser.parse_args()

    facts = _read_jsonl(args.facts)
    specs = json.loads(args.specs.read_text(encoding="utf-8"))["categories"]
    documents = [_document(fact) for fact in facts]
    cases = _build_cases(facts, specs)
    _validate(cases, documents)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items_path = args.output_dir / "recall_items.jsonl"
    cases_path = args.output_dir / "recall_cases.jsonl"
    manifest_path = args.output_dir / "recall_manifest.json"
    _write_jsonl(items_path, documents)
    _write_jsonl(cases_path, cases)
    manifest = _manifest(
        facts=facts,
        cases=cases,
        documents=documents,
        facts_path=args.facts,
        specs_path=args.specs,
        items_path=items_path,
        cases_path=cases_path,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"built product v3 documents={len(documents)} queries={len(cases)} "
        f"intents={manifest['intent_count']} pool={POOL_SIZE} exact={EXACT_COUNT}"
    )


def _document(fact: dict[str, Any]) -> dict[str, Any]:
    attributes_en = "; ".join(
        f"{row['name']}: {row['value']}" for row in fact["text_supported_attributes"]
    )
    attributes_zh = "；".join(
        f"{row['name_zh']}：{row['value_zh']}"
        for row in fact["text_supported_attributes"]
    )
    title = f"{fact['title_en']} / {fact['title_zh']}"
    body = (
        f"Category: {fact['category']}. Attributes: {attributes_en}. "
        f"品类：{fact['category_zh']}。属性：{attributes_zh}。"
    )
    return {
        "dataset_version": DATASET_VERSION,
        "document_id": str(fact["fact_id"]),
        "title": title,
        "body": body,
        "category": fact["category"],
        "category_zh": fact["category_zh"],
        "source_kind": fact["source_kind"],
        "source_product_id": fact["product_id"],
    }


def _build_cases(
    facts: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    facts_by_category = {
        category: sorted(
            (fact for fact in facts if fact["category"] == category),
            key=lambda row: str(row["fact_id"]),
        )
        for category in {str(fact["category"]) for fact in facts}
    }
    spec_by_category = {str(spec["category"]): spec for spec in specs}
    all_facts = sorted(facts, key=lambda row: str(row["fact_id"]))
    cases: list[dict[str, Any]] = []
    for category in sorted(spec_by_category):
        spec = spec_by_category[category]
        category_facts = facts_by_category[category]
        synthetic = [
            fact
            for fact in category_facts
            if fact["source_kind"] == "synthetic_llm_template"
        ]
        for variant, facet_indexes in enumerate(FACET_PAIRS, start=1):
            seed = synthetic[variant - 1]
            target = _target_attributes(seed, facet_indexes)
            matching = [
                fact
                for fact in synthetic
                if _matches(fact, target) == len(target)
            ]
            if len(matching) < EXACT_COUNT:
                raise ValueError(
                    f"{category} variant {variant} has only {len(matching)} exacts"
                )
            exact = matching[:EXACT_COUNT]
            excluded_exact_ids = {str(fact["fact_id"]) for fact in matching}
            hard_negatives = [
                fact
                for fact in category_facts
                if str(fact["fact_id"]) not in excluded_exact_ids
            ]
            cross_negatives = [
                fact for fact in all_facts if fact["category"] != category
            ]
            intent_id = f"prv3-{_slug(category)}-attribute-{variant:02d}"
            selected_negatives = _select_negatives(
                intent_id,
                hard_negatives,
                cross_negatives,
                POOL_SIZE - EXACT_COUNT,
            )
            pool = [*exact, *selected_negatives]
            query_en, query_zh = _query_pair(spec, seed, target)
            cases.extend(
                _paired_cases(
                    intent_id=intent_id,
                    category=category,
                    query_en=query_en,
                    query_zh=query_zh,
                    exact=exact,
                    pool=pool,
                    target=target,
                )
            )
    return sorted(cases, key=lambda row: (row["intent_id"], row["language"]))


def _target_attributes(
    fact: dict[str, Any], facet_indexes: tuple[int, int]
) -> dict[str, str]:
    attributes = fact["text_supported_attributes"]
    return {
        str(attributes[index]["name"]): str(attributes[index]["value"])
        for index in facet_indexes
    }


def _matches(fact: dict[str, Any], target: dict[str, str]) -> int:
    attributes = {
        str(row["name"]): str(row["value"])
        for row in fact["text_supported_attributes"]
    }
    return sum(attributes.get(name) == value for name, value in target.items())


def _select_negatives(
    intent_id: str,
    hard_negatives: list[dict[str, Any]],
    cross_negatives: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    ordered_hard = sorted(
        hard_negatives,
        key=lambda row: _stable_order(intent_id, str(row["fact_id"]), "hard"),
    )
    hard_count = min(55, count, len(ordered_hard))
    selected = ordered_hard[:hard_count]
    selected_ids = {str(fact["fact_id"]) for fact in selected}
    ordered_cross = sorted(
        (
            fact
            for fact in cross_negatives
            if str(fact["fact_id"]) not in selected_ids
        ),
        key=lambda row: _stable_order(intent_id, str(row["fact_id"]), "cross"),
    )
    selected.extend(ordered_cross[: count - len(selected)])
    if len(selected) != count:
        raise ValueError(f"{intent_id} could only construct {len(selected)} negatives")
    return selected


def _query_pair(
    spec: dict[str, Any], seed: dict[str, Any], target: dict[str, str]
) -> tuple[str, str]:
    attribute_by_name = {
        str(row["name"]): row for row in seed["text_supported_attributes"]
    }
    values_en = [target[name] for name in target]
    values_zh = [str(attribute_by_name[name]["value_zh"]) for name in target]
    form_index = int(str(seed["product_id"]).rsplit("-", 1)[-1]) - 1
    form_en, form_zh = spec["forms"][form_index % len(spec["forms"])]
    return (
        f"{form_en} with {values_en[0]} and {values_en[1]}",
        f"想买{form_zh}，要求{values_zh[0]}并且{values_zh[1]}",
    )


def _paired_cases(
    *,
    intent_id: str,
    category: str,
    query_en: str,
    query_zh: str,
    exact: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    target: dict[str, str],
) -> list[dict[str, Any]]:
    exact_ids = {str(fact["fact_id"]) for fact in exact}
    candidate_ids = sorted(str(fact["fact_id"]) for fact in pool)
    fact_by_id = {str(fact["fact_id"]): fact for fact in pool}
    labels = {
        document_id: (
            "Exact"
            if document_id in exact_ids
            else (
                "Substitute"
                if fact_by_id[document_id]["category"] == category
                and _matches(fact_by_id[document_id], target) == 1
                else "Irrelevant"
            )
        )
        for document_id in candidate_ids
    }
    relevance = {
        document_id: float(label == "Exact")
        for document_id, label in labels.items()
    }
    graded = {
        document_id: 1.0 if label == "Exact" else 0.01 if label == "Substitute" else 0.0
        for document_id, label in labels.items()
    }
    output = []
    for language, query in (("en", query_en), ("zh", query_zh)):
        output.append(
            {
                "dataset_version": DATASET_VERSION,
                "intent_id": intent_id,
                "query_id": f"{intent_id}-{language}",
                "query": query,
                "language": language,
                "query_source_kind": (
                    "model_authored_synthetic"
                    if language == "en"
                    else "synthetic_translation_pair"
                ),
                "category": category,
                "split": "test",
                "candidate_ids": candidate_ids,
                "relevance": relevance,
                "graded_relevance": graded,
                "labels": labels,
            }
        )
    return output


def _validate(
    cases: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> None:
    document_ids = {str(document["document_id"]) for document in documents}
    if len(documents) != 1356 or len(document_ids) != len(documents):
        raise ValueError("product v3 requires 1356 unique fact documents")
    if len(cases) != 160 or len({case["intent_id"] for case in cases}) != 80:
        raise ValueError("product v3 requires 80 paired intents / 160 query rows")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        pairs.setdefault(case["intent_id"], []).append(case)
        candidates = set(case["candidate_ids"])
        if len(candidates) != POOL_SIZE or not candidates.issubset(document_ids):
            raise ValueError(f"{case['query_id']} has an invalid candidate pool")
        if not (
            candidates
            == set(case["labels"])
            == set(case["relevance"])
            == set(case["graded_relevance"])
        ):
            raise ValueError(f"{case['query_id']} contains unjudged candidates")
        if sum(label == "Exact" for label in case["labels"].values()) != EXACT_COUNT:
            raise ValueError(f"{case['query_id']} must have five Exact products")
    for intent_id, pair in pairs.items():
        if {row["language"] for row in pair} != {"en", "zh"}:
            raise ValueError(f"{intent_id} is not bilingual")
        for field in ("candidate_ids", "labels", "relevance", "graded_relevance"):
            if pair[0][field] != pair[1][field]:
                raise ValueError(f"{intent_id} has mismatched {field}")


def _manifest(
    *,
    facts: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    facts_path: Path,
    specs_path: Path,
    items_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "role": "separate synthetic bilingual diagnostic; real ESCI remains primary",
        "closed_pool": True,
        "unjudged_policy": "forbidden",
        "unjudged_policy_detail": "every item in each 120-item query pool is labeled",
        "query_count": len(cases),
        "intent_count": len({case["intent_id"] for case in cases}),
        "document_count": len(documents),
        "category_count": len({document["category"] for document in documents}),
        "candidate_count_per_query": POOL_SIZE,
        "exact_positive_count_per_query": EXACT_COUNT,
        "language_counts": dict(sorted(Counter(case["language"] for case in cases).items())),
        "document_source_counts": dict(
            sorted(Counter(fact["source_kind"] for fact in facts).items())
        ),
        "gain_mapping": {"Exact": 1.0, "Substitute": 0.01, "Irrelevant": 0.0},
        "binary_positive_policy": "Exact only",
        "pair_policy": "English and Chinese rows share candidates and all judgments",
        "pool_policy": (
            "five Exact synthetic products, up to 55 same-category nonmatching hard "
            "negatives, then deterministic cross-category negatives; additional exact "
            "matches are excluded before the closed pool is frozen"
        ),
        "truth_boundary": (
            "queries and most products are synthetic offline fixtures; Chinese text is "
            "machine generated and unreviewed"
        ),
        "input_sha256": {
            "facts": _sha256(facts_path),
            "generation_specs": _sha256(specs_path),
        },
        "output_sha256": {
            "items": _sha256(items_path),
            "cases": _sha256(cases_path),
        },
    }


def _stable_order(intent_id: str, document_id: str, salt: str) -> str:
    return hashlib.sha256(f"{intent_id}:{salt}:{document_id}".encode()).hexdigest()


def _slug(value: str) -> str:
    return "-".join(value.casefold().split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
