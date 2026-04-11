"""Build a fully judged, intent-paired English/Chinese CategoryCard v3 eval.

The queries are model-authored offline fixtures. English and Chinese rows sharing
an ``intent_id`` express the same information need and therefore share exactly
the same candidate pool, five positive cards, gains, and split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "category-card-recall-bilingual-v3"
FINAL_HOLDOUT_VERSION = "category-card-recall-bilingual-final-holdout-v3"
QUERY_TYPES = ("noun", "attribute_constraint", "style", "colloquial")
SPLITS = ("train", "dev", "test")
LANGUAGES = ("en", "zh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "category_cards_v3.jsonl"
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
        default=PROJECT_ROOT / "data" / "category_insight",
    )
    args = parser.parse_args()

    cards = _read_jsonl(args.cards)
    specs_document = json.loads(args.specs.read_text(encoding="utf-8"))
    specs = specs_document["categories"]
    cases = _build_cases(cards, specs)
    final_holdout = _build_final_holdout(cards, specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = args.output_dir / "category_recall_cases_v3.jsonl"
    manifest_path = args.output_dir / "category_recall_manifest_v3.json"
    final_cases_path = args.output_dir / "category_recall_final_test_v3.jsonl"
    final_manifest_path = (
        args.output_dir / "category_recall_final_test_manifest_v3.json"
    )
    _write_jsonl(cases_path, cases)
    _write_jsonl(final_cases_path, final_holdout)
    manifest = _build_manifest(
        cards=cards,
        cases=cases,
        cards_path=args.cards,
        specs_path=args.specs,
        cases_path=cases_path,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_manifest = _build_manifest(
        cards=cards,
        cases=final_holdout,
        cards_path=args.cards,
        specs_path=args.specs,
        cases_path=final_cases_path,
        dataset_version=FINAL_HOLDOUT_VERSION,
        selection_method=(
            "sealed model-authored bilingual holdout created after retrieval freeze"
        ),
    )
    final_manifest_path.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"built v3 intents={manifest['intent_count']} query_rows={len(cases)} "
        f"final_holdout_intents={final_manifest['intent_count']} "
        f"final_holdout_rows={len(final_holdout)} cards={len(cards)} "
        f"fully_judged={manifest['closed_pool']}"
    )


def _build_cases(
    cards: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    card_by_id = {str(card["card_id"]): card for card in cards}
    if len(card_by_id) != 200 or len(card_by_id) != len(cards):
        raise ValueError("v3 evaluation requires 200 unique cards")
    candidate_ids = sorted(card_by_id)
    spec_by_category = {str(spec["category"]): spec for spec in specs}
    card_categories = {str(card["category"]) for card in cards}
    if card_categories != set(spec_by_category):
        raise ValueError("card categories and generation specifications differ")

    cases: list[dict[str, Any]] = []
    for category in sorted(spec_by_category):
        spec = spec_by_category[category]
        for split in SPLITS:
            for query_type in QUERY_TYPES:
                cases.extend(
                    _paired_cases(
                        spec=spec,
                        query_type=query_type,
                        split=split,
                        candidate_ids=candidate_ids,
                        card_by_id=card_by_id,
                        dataset_version=DATASET_VERSION,
                    )
                )
    _validate_cases(cases, candidate_ids, expected_intents=240)
    return sorted(cases, key=lambda row: (row["intent_id"], row["language"]))


def _build_final_holdout(
    cards: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    card_by_id = {str(card["card_id"]): card for card in cards}
    candidate_ids = sorted(card_by_id)
    cases: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda row: str(row["category"])):
        for query_type in QUERY_TYPES:
            for variant in (1, 2):
                cases.extend(
                    _paired_cases(
                        spec=spec,
                        query_type=query_type,
                        split="final_test",
                        candidate_ids=candidate_ids,
                        card_by_id=card_by_id,
                        dataset_version=FINAL_HOLDOUT_VERSION,
                        variant=variant,
                    )
                )
    _validate_cases(cases, candidate_ids, expected_intents=160)
    return sorted(cases, key=lambda row: (row["intent_id"], row["language"]))


def _paired_cases(
    *,
    spec: dict[str, Any],
    query_type: str,
    split: str,
    candidate_ids: list[str],
    card_by_id: dict[str, dict[str, Any]],
    dataset_version: str,
    variant: int | None = None,
) -> list[dict[str, Any]]:
    category = str(spec["category"])
    variant_suffix = f"-{variant}" if variant is not None else ""
    intent_id = f"kcv3-{_slug(category)}-{query_type}-{split}{variant_suffix}"
    if split == "final_test":
        if variant is None:
            raise ValueError("final_test requires a query variant")
        query_en, query_zh = _final_holdout_query_pair(spec, query_type, variant)
        relevant_roles, price_relevance = _final_holdout_roles(query_type, variant)
    else:
        query_en, query_zh = _query_pair(spec, query_type, split)
        relevant_roles, price_relevance = _relevant_roles(query_type, split)
    relevant_ids = [_role_to_card_id(category, role) for role in relevant_roles]
    missing = set(relevant_ids) - card_by_id.keys()
    if missing:
        raise ValueError(f"{intent_id} references missing cards: {sorted(missing)}")
    if any(card_by_id[card_id]["category"] != category for card_id in relevant_ids):
        raise ValueError(f"{intent_id} contains cross-category positives")
    gain_by_id = {
        card_id: float(5 - rank) for rank, card_id in enumerate(relevant_ids)
    }

    output = []
    for language, query in (("en", query_en), ("zh", query_zh)):
        output.append(
            {
                "dataset_version": dataset_version,
                "intent_id": intent_id,
                "query_id": f"{intent_id}-{language}",
                "query": query,
                "language": language,
                "query_source_kind": (
                    "model_authored_synthetic"
                    if language == "en"
                    else "synthetic_translation_pair"
                ),
                "translation_review_status": "machine_generated_unreviewed",
                "query_type": query_type,
                "category": category,
                "category_zh": spec["category_zh"],
                "split": split,
                "price_relevance": price_relevance,
                "candidate_ids": candidate_ids,
                "relevant_card_ids": relevant_ids,
                "relevance": {
                    card_id: float(card_id in gain_by_id) for card_id in candidate_ids
                },
                "graded_relevance": {
                    card_id: gain_by_id.get(card_id, 0.0)
                    for card_id in candidate_ids
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
    return output


def _query_pair(
    spec: dict[str, Any], query_type: str, split: str
) -> tuple[str, str]:
    category = str(spec["category"])
    category_zh = str(spec["category_zh"])
    split_index = SPLITS.index(split)
    facets = spec["facets"]
    form_en, form_zh = spec["forms"][split_index % len(spec["forms"])]

    if query_type == "noun":
        if split == "train":
            return (
                f"compare price tiers and popular options for {category}",
                f"比较{category_zh}的价格档位和热门选择",
            )
        if split == "dev":
            return (
                f"{category} buying guide and common choices",
                f"{category_zh}选购指南和常见选择",
            )
        return (
            f"popular types of {category} worth knowing",
            f"{category_zh}有哪些值得了解的热门类型",
        )

    if query_type == "attribute_constraint":
        first = facets[(split_index * 2) % len(facets)]
        second = facets[(split_index * 2 + 1) % len(facets)]
        first_en, first_zh = first["values"][split_index % 3]
        second_en, second_zh = second["values"][(split_index + 1) % 3]
        return (
            f"{form_en} with {first_en} and {second_en}",
            f"想要{first_zh}并且{second_zh}的{form_zh}",
        )

    if query_type == "style":
        use_facet = facets[-1]
        use_en, use_zh = use_facet["values"][split_index % 3]
        tone_en = ("practical", "clean-looking", "premium-feeling")[split_index]
        tone_zh = ("实用", "外观简洁", "有质感")[split_index]
        return (
            f"a {tone_en} {form_en} suited to {use_en}",
            f"适合{use_zh}、{tone_zh}的{form_zh}",
        )

    if query_type != "colloquial":
        raise ValueError(f"unsupported query type: {query_type}")
    if split == "train":
        return (
            f"I just want a dependable {form_en} that is easy to live with",
            f"就想买个靠谱省心的{form_zh}",
        )
    if split == "dev":
        return (
            f"help me pick a no-fuss {form_en} for everyday use",
            f"帮我挑个日常用着不折腾的{form_zh}",
        )
    budget = int(float(spec["price_bounds_cny"][0]) * 1.8)
    return (
        f"which {form_en} is good without going over CNY {budget}",
        f"预算不超过{budget}元，哪个{form_zh}比较好",
    )


def _relevant_roles(query_type: str, split: str) -> tuple[list[str], str]:
    if query_type == "noun":
        if split == "train":
            return (
                [
                    "price-range-01",
                    "bestseller-01",
                    "bestseller-02",
                    "attribute-01",
                    "attribute-02",
                ],
                "strong",
            )
        return (
            [
                "bestseller-01",
                "bestseller-02",
                "price-range-01",
                "attribute-01",
                "attribute-02",
            ],
            "medium",
        )
    if query_type == "attribute_constraint":
        offset = SPLITS.index(split) * 2
        return (
            [
                f"attribute-{offset + 1:02d}",
                f"attribute-{offset + 2:02d}",
                "bestseller-01",
                "bestseller-02",
                "price-range-01",
            ],
            "coverage_weak",
        )
    if query_type == "style":
        return (
            [
                "attribute-07",
                "attribute-01",
                "bestseller-01",
                "bestseller-02",
                "price-range-01",
            ],
            "coverage_weak",
        )
    if split == "test":
        return (
            [
                "price-range-01",
                "bestseller-01",
                "bestseller-02",
                "attribute-01",
                "attribute-02",
            ],
            "strong",
        )
    attribute_role = "attribute-06" if split == "train" else "attribute-07"
    return (
        [
            attribute_role,
            "bestseller-01",
            "bestseller-02",
            "attribute-01",
            "price-range-01",
        ],
        "coverage_weak",
    )


def _final_holdout_query_pair(
    spec: dict[str, Any], query_type: str, variant: int
) -> tuple[str, str]:
    category = str(spec["category"])
    category_zh = str(spec["category_zh"])
    form_en, form_zh = spec["forms"][(variant + 1) % len(spec["forms"])]
    if query_type == "noun":
        if variant == 1:
            return (
                f"what are the main {category} options people choose",
                f"{category_zh}主要有哪些常见选择",
            )
        return (
            f"how much should I budget for {category} and what sells well",
            f"买{category_zh}大概要多少预算，哪些比较热门",
        )
    if query_type == "attribute_constraint":
        facet_indexes = (1, 4) if variant == 1 else (2, 5)
        first = spec["facets"][facet_indexes[0]]
        second = spec["facets"][facet_indexes[1]]
        first_en, first_zh = first["values"][(variant + 1) % 3]
        second_en, second_zh = second["values"][variant % 3]
        return (
            f"find a {form_en}: {first_en}, preferably {second_en}",
            f"找一款{form_zh}，要求{first_zh}，最好还有{second_zh}",
        )
    if query_type == "style":
        use_en, use_zh = spec["facets"][-1]["values"][variant % 3]
        tone_en = "understated" if variant == 1 else "rugged-looking"
        tone_zh = "低调耐看" if variant == 1 else "看起来结实耐用"
        return (
            f"an {tone_en} {form_en} that fits {use_en}",
            f"适合{use_zh}、{tone_zh}的{form_zh}",
        )
    if query_type != "colloquial":
        raise ValueError(f"unsupported query type: {query_type}")
    if variant == 1:
        return (
            f"not sure what to buy, show me a sensible {form_en}",
            f"不太懂，给我推荐个省心的{form_zh}",
        )
    budget = int(float(spec["price_bounds_cny"][0]) * 2.4)
    return (
        f"I'd like a decent {form_en} for around CNY {budget}",
        f"手头大约{budget}元预算，想买个不错的{form_zh}",
    )


def _final_holdout_roles(query_type: str, variant: int) -> tuple[list[str], str]:
    if query_type == "noun" and variant == 1:
        return (
            [
                "bestseller-01",
                "bestseller-02",
                "price-range-01",
                "attribute-01",
                "attribute-02",
            ],
            "medium",
        )
    if query_type == "noun" or (query_type == "colloquial" and variant == 2):
        return (
            [
                "price-range-01",
                "bestseller-01",
                "bestseller-02",
                "attribute-01",
                "attribute-02",
            ],
            "strong",
        )
    if query_type == "attribute_constraint":
        roles = ("attribute-02", "attribute-05") if variant == 1 else (
            "attribute-03",
            "attribute-06",
        )
        return (
            [
                *roles,
                "bestseller-01",
                "bestseller-02",
                "price-range-01",
            ],
            "coverage_weak",
        )
    if query_type == "style":
        return (
            [
                "attribute-07",
                "attribute-02",
                "bestseller-01",
                "bestseller-02",
                "price-range-01",
            ],
            "coverage_weak",
        )
    return (
        [
            "attribute-07",
            "bestseller-01",
            "bestseller-02",
            "attribute-01",
            "price-range-01",
        ],
        "coverage_weak",
    )


def _validate_cases(
    cases: list[dict[str, Any]],
    candidate_ids: list[str],
    *,
    expected_intents: int,
) -> None:
    if len(cases) != expected_intents * len(LANGUAGES):
        raise ValueError(f"expected {expected_intents * 2} rows, got {len(cases)}")
    if len({case["query_id"] for case in cases}) != len(cases):
        raise ValueError("query_id must be unique")
    if len({case["query"].casefold() for case in cases}) != len(cases):
        raise ValueError("query text must be unique")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        pairs.setdefault(case["intent_id"], []).append(case)
        if case["candidate_ids"] != candidate_ids:
            raise ValueError(f"{case['query_id']} has an incomplete candidate pool")
        if set(case["judgments"]) != set(candidate_ids):
            raise ValueError(f"{case['query_id']} contains unjudged candidates")
        if len(case["relevant_card_ids"]) != 5:
            raise ValueError(f"{case['query_id']} must have five positives")
    if len(pairs) != expected_intents:
        raise ValueError(f"expected {expected_intents} intent pairs")
    for intent_id, pair in pairs.items():
        if {row["language"] for row in pair} != set(LANGUAGES):
            raise ValueError(f"{intent_id} is not a complete language pair")
        comparable_fields = (
            "category",
            "query_type",
            "split",
            "candidate_ids",
            "relevant_card_ids",
            "relevance",
            "graded_relevance",
            "judgments",
        )
        if any(pair[0][field] != pair[1][field] for field in comparable_fields):
            raise ValueError(f"{intent_id} has mismatched paired judgments")


def _build_manifest(
    *,
    cards: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    cards_path: Path,
    specs_path: Path,
    cases_path: Path,
    dataset_version: str = DATASET_VERSION,
    selection_method: str = (
        "model-authored deterministic bilingual offline fixtures"
    ),
) -> dict[str, Any]:
    return {
        "dataset_version": dataset_version,
        "language_policy": "English and Chinese rows are paired by intent_id",
        "selection_method": selection_method,
        "query_source_truth_boundary": {
            "model_authored_synthetic": "English test fixture, not an ESCI query",
            "synthetic_translation_pair": (
                "Chinese paired fixture, machine generated and unreviewed"
            ),
        },
        "query_count": len(cases),
        "intent_count": len({case["intent_id"] for case in cases}),
        "card_count": len(cards),
        "positive_count_per_query": 5,
        "category_count": len({case["category"] for case in cases}),
        "language_counts": dict(sorted(Counter(case["language"] for case in cases).items())),
        "split_counts": dict(sorted(Counter(case["split"] for case in cases).items())),
        "query_type_counts": dict(
            sorted(Counter(case["query_type"] for case in cases).items())
        ),
        "source_kind_counts": dict(
            sorted(Counter(case["query_source_kind"] for case in cases).items())
        ),
        "candidate_pool_policy": "global 200-card corpus; every card explicitly judged",
        "unjudged_policy": "forbidden",
        "closed_pool": True,
        "relevance_policy": "exactly five binary positives per query",
        "graded_gain_policy": "ordered relevant_card_ids receive gains 5,4,3,2,1",
        "pair_policy": (
            "paired languages share split, type, candidates, positives, gains, and judgments"
        ),
        "split_policy": (
            "each category has one intent per query type in each split; paired rows never split"
        ),
        "tuning_policy": "tune on train/dev; report test once after configuration freeze",
        "input_sha256": {
            "cards": _sha256(cards_path),
            "generation_specs": _sha256(specs_path),
        },
        "cases_sha256": _sha256(cases_path),
    }


def _role_to_card_id(category: str, role: str) -> str:
    return f"cc-{_slug(category)}-{role}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
