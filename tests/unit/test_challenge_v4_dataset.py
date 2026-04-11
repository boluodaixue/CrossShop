from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from globex_agent.category_insight import admit_card

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATEGORY_DIR = PROJECT_ROOT / "data" / "category_insight"
PRODUCT_DIR = PROJECT_ROOT / "data" / "eval" / "product_v4"


def test_v4_card_corpus_is_larger_distinguishable_and_admissible() -> None:
    facts = _read_jsonl(CATEGORY_DIR / "category_item_facts_v4.jsonl")
    cards = _read_jsonl(CATEGORY_DIR / "category_cards_v4.jsonl")
    retrieval = _read_jsonl(CATEGORY_DIR / "category_retrieval_texts_v4.jsonl")
    specs = json.loads(
        (CATEGORY_DIR / "category_generation_specs_v3.json").read_text(
            encoding="utf-8"
        )
    )["categories"]

    facts_by_category = Counter(row["category"] for row in facts)
    cards_by_category = Counter(row["category"] for row in cards)
    assert len(facts) == 2356
    assert len(facts_by_category) == 20
    assert min(facts_by_category.values()) >= 100
    assert len(cards) == 300
    assert set(cards_by_category.values()) == {15}
    assert Counter(row["card_type"] for row in cards) == {
        "attribute": 240,
        "bestseller": 40,
        "price_range": 20,
    }
    assert all(admit_card(card).accepted for card in cards)

    retrieval_by_id = {row["card_id"]: row for row in retrieval}
    assert set(retrieval_by_id) == {card["card_id"] for card in cards}
    spec_by_category = {row["category"]: row for row in specs}
    for card in cards:
        text = retrieval_by_id[card["card_id"]]["retrieval_text_en"].casefold()
        forms = spec_by_category[card["category"]]["forms"]
        included_forms = sum(form_en.casefold() in text for form_en, _ in forms)
        assert included_forms < len(forms), card["card_id"]


def test_v4_knowledge_challenge_is_new_paired_and_fully_judged() -> None:
    cards = _read_jsonl(CATEGORY_DIR / "category_cards_v4.jsonl")
    cases = _read_jsonl(CATEGORY_DIR / "category_recall_challenge_v4.jsonl")
    v3_cases = _read_jsonl(CATEGORY_DIR / "category_recall_final_test_v3.jsonl")
    manifest = json.loads(
        (CATEGORY_DIR / "category_recall_challenge_manifest_v4.json").read_text(
            encoding="utf-8"
        )
    )
    card_ids = sorted(card["card_id"] for card in cards)

    assert manifest["closed_pool"] is True
    assert manifest["unjudged_policy"] == "forbidden"
    assert len(cases) == 320
    assert len({row["intent_id"] for row in cases}) == 160
    assert len({row["query"].casefold() for row in cases}) == 320
    assert not (
        {row["query"].casefold() for row in cases}
        & {row["query"].casefold() for row in v3_cases}
    )
    assert Counter(row["query_type"] for row in cases) == {
        "attribute_constraint": 80,
        "colloquial": 80,
        "noun": 80,
        "style": 80,
    }

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        pairs[case["intent_id"]].append(case)
        assert case["candidate_ids"] == card_ids
        assert set(case["judgments"]) == set(card_ids)
        assert len(case["relevant_card_ids"]) == 5
    for pair in pairs.values():
        assert {row["language"] for row in pair} == {"en", "zh"}
        for field in (
            "candidate_ids",
            "relevant_card_ids",
            "relevance",
            "graded_relevance",
            "judgments",
        ):
            assert pair[0][field] == pair[1][field]


def test_v4_product_challenge_has_large_hard_fully_judged_pools() -> None:
    items = _read_jsonl(PRODUCT_DIR / "recall_items.jsonl")
    cases = _read_jsonl(PRODUCT_DIR / "recall_cases.jsonl")
    v3_cases = _read_jsonl(
        PROJECT_ROOT / "data" / "eval" / "product_v3" / "recall_cases.jsonl"
    )
    manifest = json.loads(
        (PRODUCT_DIR / "recall_manifest.json").read_text(encoding="utf-8")
    )
    item_by_id = {row["document_id"]: row for row in items}

    assert manifest["closed_pool"] is True
    assert manifest["unjudged_policy"] == "forbidden"
    assert len(items) == 2356
    assert len(cases) == 160
    assert len({row["intent_id"] for row in cases}) == 80
    assert not (
        {row["query"].casefold() for row in cases}
        & {row["query"].casefold() for row in v3_cases}
    )
    assert Counter(row["challenge_kind"] for row in cases) == {
        "colloquial_scenario": 40,
        "lexical_constraints": 40,
        "negative_constraint": 40,
        "scenario_paraphrase": 40,
    }

    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        pairs[case["intent_id"]].append(case)
        candidates = set(case["candidate_ids"])
        assert len(candidates) == 500
        assert candidates == set(case["labels"])
        assert candidates == set(case["relevance"])
        assert candidates == set(case["graded_relevance"])
        assert sum(label == "Exact" for label in case["labels"].values()) == 5
        same_category_negatives = sum(
            item_by_id[document_id]["category"] == case["category"]
            and case["labels"][document_id] != "Exact"
            for document_id in candidates
        )
        assert same_category_negatives >= 60
    for pair in pairs.values():
        assert {row["language"] for row in pair} == {"en", "zh"}
        for field in (
            "candidate_ids",
            "labels",
            "relevance",
            "graded_relevance",
            "target_contract",
        ):
            assert pair[0][field] == pair[1][field]


def test_v4_manifests_and_freeze_hashes_match_checked_in_data() -> None:
    card_manifest = json.loads(
        (CATEGORY_DIR / "category_card_manifest_v4.json").read_text(encoding="utf-8")
    )
    knowledge_manifest = json.loads(
        (CATEGORY_DIR / "category_recall_challenge_manifest_v4.json").read_text(
            encoding="utf-8"
        )
    )
    product_manifest = json.loads(
        (PRODUCT_DIR / "recall_manifest.json").read_text(encoding="utf-8")
    )
    freeze = json.loads(
        (CATEGORY_DIR / "challenge_freeze_v4.json").read_text(encoding="utf-8")
    )
    expected = {
        "facts": CATEGORY_DIR / "category_item_facts_v4.jsonl",
        "cards": CATEGORY_DIR / "category_cards_v4.jsonl",
        "provenance": CATEGORY_DIR / "category_card_provenance_v4.jsonl",
        "retrieval": CATEGORY_DIR / "category_retrieval_texts_v4.jsonl",
        "knowledge_cases": CATEGORY_DIR / "category_recall_challenge_v4.jsonl",
        "product_items": PRODUCT_DIR / "recall_items.jsonl",
        "product_cases": PRODUCT_DIR / "recall_cases.jsonl",
    }

    assert freeze["data_sha256"] == {
        name: _sha256(path) for name, path in expected.items()
    }
    assert card_manifest["output_sha256"] == {
        name: _sha256(expected[name])
        for name in ("facts", "cards", "provenance", "retrieval")
    }
    assert knowledge_manifest["cases_sha256"] == _sha256(
        expected["knowledge_cases"]
    )
    assert product_manifest["output_sha256"] == {
        "items": _sha256(expected["product_items"]),
        "cases": _sha256(expected["product_cases"]),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
