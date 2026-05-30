"""Build a fully judged Chinese Taobao CategoryCard recall evaluation.

The corpus has eight ordinary categories and 48 cards. The 50 Chinese queries
cover noun, attribute-constraint, style, and colloquial forms. Every query
explicitly judges the full 48-card pool and receives exactly five ordered
positives with gains 5, 4, 3, 2, and 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "category-card-recall-taobao-zh-v3"
EXPECTED_SPLITS = {"train": 30, "dev": 10, "test": 10}
EXPECTED_QUERY_TYPES = {
    "noun": 14,
    "attribute_constraint": 12,
    "style": 12,
    "colloquial": 12,
}
BASE_SPLITS = ["train", "train", "train", "train", "train", "train", "dev", "test"]
EXTRA_TYPES = {
    "latex-pillow": ["noun", "attribute_constraint", "style", "colloquial"],
    "children-study-chair": ["noun", "attribute_constraint", "style"],
    "phone-live-fill-light": ["noun", "attribute_constraint", "colloquial"],
    "car-ambient-light": ["noun", "style", "colloquial"],
    "tablet-stand": ["noun", "attribute_constraint", "colloquial"],
    "neck-massager": ["noun", "style"],
}
EXTRA_SPLITS = {
    "noun": ["train", "train", "dev", "dev", "test", "test"],
    "attribute_constraint": ["train", "train", "dev", "test"],
    "style": ["train", "dev", "dev", "test"],
    "colloquial": ["train", "dev", "test", "test"],
}

# Preserved v2 natural colloquial scenarios; the v1 baseline does not use them.
COLLOQUIAL_SCENARIOS = {
    "latex-pillow": [
        {
            "query": "想买个软点的枕头",
            "relevant_roles": ["bestseller-02", "attribute-02", "attribute-01"],
            "price_relevance": "not_relevant",
        },
        {
            "query": "想买个护颈的枕头",
            "relevant_roles": ["bestseller-02", "attribute-02", "attribute-03"],
            "price_relevance": "not_relevant",
        },
    ],
    "children-study-chair": [
        {
            "query": "孩子用的可调节的椅子",
            "relevant_roles": ["bestseller-01", "attribute-01", "attribute-03"],
            "price_relevance": "not_relevant",
        }
    ],
    "phone-live-fill-light": [
        {
            "query": "直播的时候脸太暗，想补光",
            "relevant_roles": [
                "bestseller-01",
                "attribute-01",
                "attribute-02",
                "attribute-03",
            ],
            "price_relevance": "not_relevant",
        },
        {
            "query": "出门拍视频想补光",
            "relevant_roles": [
                "bestseller-01",
                "bestseller-02",
                "attribute-01",
                "attribute-03",
            ],
            "price_relevance": "not_relevant",
        },
    ],
    "car-ambient-light": [
        {
            "query": "车里晚上太暗，想加个灯",
            "relevant_roles": [
                "bestseller-01",
                "bestseller-02",
                "attribute-01",
                "attribute-02",
            ],
            "price_relevance": "not_relevant",
        },
        {
            "query": "想装个开门会亮的灯",
            "relevant_roles": ["bestseller-01", "attribute-01", "attribute-02"],
            "price_relevance": "not_relevant",
        },
    ],
    "tablet-stand": [
        {
            "query": "追剧用的架子",
            "relevant_roles": [
                "bestseller-01",
                "bestseller-02",
                "attribute-01",
                "attribute-02",
            ],
            "price_relevance": "not_relevant",
        },
        {
            "query": "躺着看视频用的架子",
            "relevant_roles": ["bestseller-02", "attribute-01", "attribute-02"],
            "price_relevance": "not_relevant",
        },
    ],
    "neck-massager": [
        {
            "query": "脖子酸，想按按",
            "relevant_roles": [
                "bestseller-01",
                "bestseller-02",
                "attribute-01",
                "attribute-02",
            ],
            "price_relevance": "not_relevant",
        }
    ],
    "badminton-bag": [
        {
            "query": "能装球拍的包",
            "relevant_roles": [
                "bestseller-01",
                "attribute-01",
                "attribute-02",
                "bestseller-02",
            ],
            "price_relevance": "not_relevant",
        }
    ],
    "portable-power-station": [
        {
            "query": "露营充电用的",
            "relevant_roles": [
                "bestseller-01",
                "attribute-01",
                "attribute-02",
                "attribute-03",
            ],
            "price_relevance": "not_relevant",
        }
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "taobao_zh"
            / "category_cards_taobao_zh.jsonl"
        ),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "taobao_zh"
            / "category_taxonomy_taobao_zh.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "category_insight" / "taobao_zh",
    )
    args = parser.parse_args()

    cards = _read_jsonl(args.cards)
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    annotations = _build_annotations(taxonomy["categories"])
    cases = _build_cases(cards, taxonomy["categories"], annotations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.output_dir / "eval_query_annotations_taobao_zh.jsonl"
    cases_path = args.output_dir / "category_recall_cases_taobao_zh.jsonl"
    manifest_path = args.output_dir / "category_recall_manifest_taobao_zh.json"
    _write_jsonl(annotations_path, annotations)
    _write_jsonl(cases_path, cases)
    manifest = _build_manifest(
        cards=cards,
        cases=cases,
        cards_path=args.cards,
        taxonomy_path=args.taxonomy,
        cases_path=cases_path,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"built {len(cases)} fully judged queries over {len(cards)} cards; "
        f"splits={dict(Counter(case['split'] for case in cases))}"
    )


def _build_annotations(
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    extra_counters: dict[str, int] = defaultdict(int)
    for category_index, spec in enumerate(specs):
        slug = str(spec["slug"])
        for query_type in ("noun", "attribute_constraint", "style", "colloquial"):
            split = BASE_SPLITS[category_index % len(BASE_SPLITS)]
            annotations.append(
                _annotation(
                    spec,
                    query_type=query_type,
                    split=split,
                    variant=0,
                )
            )
        for query_type in EXTRA_TYPES.get(slug, []):
            split = EXTRA_SPLITS[query_type][extra_counters[query_type]]
            extra_counters[query_type] += 1
            annotations.append(
                _annotation(
                    spec,
                    query_type=query_type,
                    split=split,
                    variant=1,
                )
            )
    return sorted(annotations, key=lambda row: row["query_id"])


def _annotation(
    spec: dict[str, Any],
    *,
    query_type: str,
    split: str,
    variant: int,
) -> dict[str, Any]:
    slug = str(spec["slug"])
    query_id = f"cizh-{slug}-{query_type}-{split}-{variant + 1:02d}"
    relevant_roles = _relevant_roles(query_type, variant)
    price_relevance = "strong" if query_type in {"noun", "colloquial"} else "coverage_weak"
    return {
        "query_id": query_id,
        "query": _query_text(spec, query_type, variant),
        "query_type": query_type,
        "category": str(spec["category"]),
        "category_slug": slug,
        "split": split,
        "price_relevance": price_relevance,
        "relevant_roles": relevant_roles,
        "source_kind": "deterministic_course_fixture_taobao_zh",
        "variant": variant,
    }


def _query_text(spec: dict[str, Any], query_type: str, variant: int) -> str:
    category = str(spec["category"])
    if query_type == "noun":
        return (
            f"{category}选购指南和常见热门类型"
            if variant == 0
            else f"{category}主要有哪些值得了解的选择"
        )

    attributes = spec["attribute_rules"]
    if query_type == "attribute_constraint":
        first = attributes[(0 + variant) % 3]
        second = attributes[(1 + variant) % 3]
        first_value = next(iter(first["values"]))
        second_values = list(second["values"])
        second_value = second_values[min(variant, len(second_values) - 1)]
        return f"想要{first_value}并且{second_value}的{category}"

    if query_type == "style":
        style_word = "外观简洁" if variant == 0 else "有质感"
        return (
            f"{style_word}的{category}"
            if variant == 0
            else f"适合日常使用、{style_word}的{category}"
        )

    if query_type == "colloquial":
        return (
            f"就想买个靠谱省心的{category}"
            if variant == 0
            else f"预算有限，帮我挑个实用的{category}"
        )
    raise ValueError(f"unsupported query type: {query_type}")


def _relevant_roles(query_type: str, variant: int) -> list[str]:
    del variant
    if query_type == "noun":
        return [
            "price-range-01",
            "bestseller-01",
            "bestseller-02",
            "attribute-01",
            "attribute-02",
        ]
    if query_type == "attribute_constraint":
        return [
            "attribute-01",
            "attribute-02",
            "bestseller-01",
            "bestseller-02",
            "price-range-01",
        ]
    if query_type == "style":
        return [
            "attribute-03",
            "attribute-01",
            "bestseller-01",
            "bestseller-02",
            "price-range-01",
        ]
    if query_type == "colloquial":
        return [
            "price-range-01",
            "bestseller-01",
            "bestseller-02",
            "attribute-01",
            "attribute-02",
        ]
    raise ValueError(f"unsupported query type: {query_type}")


def _build_cases(
    cards: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(cards) != 48:
        raise ValueError(f"expected 48 Taobao Chinese cards, got {len(cards)}")
    if len(annotations) != 50:
        raise ValueError(f"expected 50 query annotations, got {len(annotations)}")
    card_by_id = {card["card_id"]: card for card in cards}
    if len(card_by_id) != len(cards):
        raise ValueError("card_id must be unique")
    slug_by_category = {str(spec["category"]): str(spec["slug"]) for spec in specs}
    categories = {card["category"] for card in cards}
    candidate_ids = sorted(card_by_id)
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for annotation in annotations:
        query_id = str(annotation["query_id"])
        query = " ".join(str(annotation["query"]).split())
        category = str(annotation["category"])
        if query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        if query.casefold() in seen_queries:
            raise ValueError(f"duplicate query text: {query}")
        if category not in categories:
            raise ValueError(f"unknown category in annotation: {category}")
        slug = slug_by_category[category]
        relevant_ids = [f"cc-{slug}-{role}" for role in annotation["relevant_roles"]]
        if len(relevant_ids) != 5 or len(set(relevant_ids)) != 5:
            raise ValueError(f"query {query_id} must have five distinct relevant cards")
        missing = set(relevant_ids) - card_by_id.keys()
        if missing:
            raise ValueError(f"query {query_id} references missing cards: {sorted(missing)}")
        if any(card_by_id[card_id]["category"] != category for card_id in relevant_ids):
            raise ValueError(f"query {query_id} has a cross-category relevant card")
        seen_ids.add(query_id)
        seen_queries.add(query.casefold())
        gain_by_id = {
            card_id: float(5 - rank) for rank, card_id in enumerate(relevant_ids)
        }
        cases.append(
            {
                "dataset_version": DATASET_VERSION,
                "query_id": query_id,
                "query": query,
                "language": "zh",
                "query_type": annotation["query_type"],
                "category": category,
                "category_slug": slug,
                "split": annotation["split"],
                "price_relevance": annotation["price_relevance"],
                "query_source_kind": annotation["source_kind"],
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
    return sorted(cases, key=lambda row: row["query_id"])


def _build_manifest(
    *,
    cards: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    cards_path: Path,
    taxonomy_path: Path,
    cases_path: Path,
) -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "language": "zh",
        "selection_method": "deterministic course-aligned Taobao Chinese fixtures",
        "query_count": len(cases),
        "card_count": len(cards),
        "positive_count_per_query": 5,
        "split_counts": dict(sorted(Counter(case["split"] for case in cases).items())),
        "query_type_counts": dict(
            sorted(Counter(case["query_type"] for case in cases).items())
        ),
        "candidate_pool_policy": "global 48-card corpus; every card is explicitly judged",
        "unjudged_policy": "forbidden",
        "closed_pool": True,
        "relevance_policy": "exactly five binary positives per query",
        "graded_gain_policy": "ordered relevant_card_ids receive gains 5,4,3,2,1",
        "price_card_policy": (
            "every query includes its category price card; noun/colloquial queries "
            "treat the robust directory listing/reference range as strong evidence, "
            "never as a concrete SKU or real-time price"
        ),
        "source_query_reuse_policy": "historical ESCI query texts are not reused",
        "tuning_policy": "tune on train/dev; report test once after configuration freeze",
        "input_sha256": {
            "cards": _sha256(cards_path),
            "taxonomy": _sha256(taxonomy_path),
        },
        "cases_sha256": _sha256(cases_path),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
