import hashlib
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "category_insight"
TAOBAO_DIR = DATA_DIR / "taobao_zh"
ESCI_DIR = PROJECT_ROOT / "data" / "eval"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_category_recall_set_has_expected_split_and_query_type_coverage() -> None:
    cases = _read_jsonl(DATA_DIR / "category_recall_cases.jsonl")
    assert len(cases) == 50
    assert len({case["query_id"] for case in cases}) == 50
    assert len({case["query"] for case in cases}) == 50
    assert Counter(case["split"] for case in cases) == {
        "train": 30,
        "dev": 10,
        "test": 10,
    }
    assert Counter(case["query_type"] for case in cases) == {
        "noun": 12,
        "attribute_constraint": 12,
        "style": 12,
        "colloquial": 14,
    }
    expected_types = {"noun", "attribute_constraint", "style", "colloquial"}
    for split in ("train", "dev", "test"):
        assert {case["query_type"] for case in cases if case["split"] == split} == (
            expected_types
        )


def test_every_query_explicitly_judges_the_entire_card_pool() -> None:
    cards = _read_jsonl(DATA_DIR / "category_cards.jsonl")
    cases = _read_jsonl(DATA_DIR / "category_recall_cases.jsonl")
    card_by_id = {card["card_id"]: card for card in cards}
    card_ids = set(card_by_id)
    assert len(card_ids) == 72

    for case in cases:
        candidate_ids = set(case["candidate_ids"])
        assert len(case["candidate_ids"]) == 72
        assert candidate_ids == card_ids
        assert candidate_ids == set(case["relevance"])
        assert candidate_ids == set(case["graded_relevance"])
        assert candidate_ids == set(case["judgments"])
        assert set(case["judgments"].values()) <= {
            "not_relevant",
            "relevant_rank_1",
            "relevant_rank_2",
            "relevant_rank_3",
            "relevant_rank_4",
            "relevant_rank_5",
        }

        relevant_ids = case["relevant_card_ids"]
        assert len(relevant_ids) == len(set(relevant_ids)) == 5
        assert relevant_ids == [
            card_id
            for card_id, gain in sorted(
                case["graded_relevance"].items(),
                key=lambda item: (-item[1], item[0]),
            )
            if gain > 0
        ]
        assert [case["graded_relevance"][card_id] for card_id in relevant_ids] == [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
        ]
        assert {card_id for card_id, gain in case["relevance"].items() if gain > 0} == (
            set(relevant_ids)
        )
        assert all(card_by_id[card_id]["category"] == case["category"] for card_id in relevant_ids)


def test_identical_reranker_inputs_never_receive_conflicting_labels() -> None:
    cards = _read_jsonl(DATA_DIR / "category_cards.jsonl")
    cases = _read_jsonl(DATA_DIR / "category_recall_cases.jsonl")
    ids_by_summary: dict[tuple[str, str], list[str]] = {}
    for card in cards:
        ids_by_summary.setdefault((card["category"], card["summary"]), []).append(
            card["card_id"]
        )

    for case in cases:
        for card_ids in ids_by_summary.values():
            labels = {case["relevance"][card_id] for card_id in card_ids}
            assert len(labels) == 1


def test_test_split_has_no_unjudged_cards_or_source_query_copies() -> None:
    cases = _read_jsonl(DATA_DIR / "category_recall_cases.jsonl")
    source_cases = _read_jsonl(ESCI_DIR / "recall_cases.jsonl")
    source_queries = {case["query"].casefold() for case in source_cases}
    test_cases = [case for case in cases if case["split"] == "test"]

    assert len(test_cases) == 10
    for case in test_cases:
        assert set(case["candidate_ids"]) == set(case["judgments"])
        assert case["query"].casefold() not in source_queries


def test_category_recall_manifest_matches_frozen_files() -> None:
    manifest = json.loads(
        (DATA_DIR / "category_recall_manifest.json").read_text(encoding="utf-8")
    )
    cases_path = DATA_DIR / "category_recall_cases.jsonl"
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    assert manifest["query_count"] == 50
    assert manifest["card_count"] == 72
    assert manifest["positive_count_per_query"] == 5
    assert manifest["closed_pool"] is True
    assert manifest["unjudged_policy"] == "forbidden"
    assert manifest["cases_sha256"] == digest


def test_v2_keeps_train_dev_and_replaces_the_historical_test_split() -> None:
    v1_cases = _read_jsonl(DATA_DIR / "category_recall_cases.jsonl")
    v2_cases = _read_jsonl(DATA_DIR / "category_recall_cases_v2.jsonl")
    v1_train_dev = {
        case["query_id"]: case["split"]
        for case in v1_cases
        if case["split"] in {"train", "dev"}
    }
    v2_train_dev = {
        case["query_id"]: case["split"]
        for case in v2_cases
        if case["split"] in {"train", "dev"}
    }
    historical_test_queries = {
        case["query"].casefold() for case in v1_cases if case["split"] == "test"
    }
    v2_test = [case for case in v2_cases if case["split"] == "test"]

    assert v2_train_dev == v1_train_dev
    assert len(v2_test) == 10
    assert all(case["query_id"].startswith("ciq-v2-") for case in v2_test)
    assert not historical_test_queries & {
        case["query"].casefold() for case in v2_test
    }


def test_v2_price_gain_depends_on_query_intent_and_pool_is_fully_judged() -> None:
    cards = _read_jsonl(DATA_DIR / "category_cards.jsonl")
    cases = _read_jsonl(DATA_DIR / "category_recall_cases_v2.jsonl")
    card_ids = {card["card_id"] for card in cards}
    allowed_gains = {
        "strong": {4.0, 5.0},
        "medium": {2.0, 3.0},
        "coverage_weak": {1.0},
    }

    assert {case["price_relevance"] for case in cases} == set(allowed_gains)
    for case in cases:
        price_id = next(
            card_id
            for card_id in case["relevant_card_ids"]
            if card_id.endswith("-price-range-01")
        )
        assert case["graded_relevance"][price_id] in allowed_gains[
            case["price_relevance"]
        ]
        assert set(case["candidate_ids"]) == card_ids
        assert set(case["judgments"]) == card_ids
        assert len(case["relevant_card_ids"]) == 5


def test_v2_manifest_matches_versioned_cases() -> None:
    manifest = json.loads(
        (DATA_DIR / "category_recall_manifest_v2.json").read_text(encoding="utf-8")
    )
    cases_path = DATA_DIR / "category_recall_cases_v2.jsonl"

    assert manifest["dataset_version"] == "category-card-recall-en-v2"
    assert manifest["query_count"] == 50
    assert manifest["card_count"] == 72
    assert manifest["unjudged_policy"] == "forbidden"
    assert manifest["price_relevance_counts"] == {
        "coverage_weak": 37,
        "medium": 8,
        "strong": 5,
    }
    assert manifest["cases_sha256"] == hashlib.sha256(
        cases_path.read_bytes()
    ).hexdigest()


def test_taobao_zh_recall_set_has_expected_split_and_query_type_coverage() -> None:
    cases = _read_jsonl(TAOBAO_DIR / "category_recall_cases_taobao_zh.jsonl")

    assert len(cases) == 50
    assert len({case["query_id"] for case in cases}) == 50
    assert len({case["query"] for case in cases}) == 50
    assert Counter(case["split"] for case in cases) == {
        "train": 30,
        "dev": 10,
        "test": 10,
    }
    assert Counter(case["query_type"] for case in cases) == {
        "noun": 14,
        "attribute_constraint": 12,
        "style": 12,
        "colloquial": 12,
    }
    for split in ("train", "dev", "test"):
        assert {case["query_type"] for case in cases if case["split"] == split} == {
            "noun",
            "attribute_constraint",
            "style",
            "colloquial",
        }


def test_taobao_zh_every_query_explicitly_judges_the_entire_48_card_pool() -> None:
    cards = _read_jsonl(TAOBAO_DIR / "category_cards_taobao_zh.jsonl")
    cases = _read_jsonl(TAOBAO_DIR / "category_recall_cases_taobao_zh.jsonl")
    card_by_id = {card["card_id"]: card for card in cards}
    card_ids = set(card_by_id)
    assert len(card_ids) == 48

    for case in cases:
        assert case["language"] == "zh"
        assert set(case["candidate_ids"]) == card_ids
        assert set(case["relevance"]) == card_ids
        assert set(case["graded_relevance"]) == card_ids
        assert set(case["judgments"]) == card_ids
        assert len(case["relevant_card_ids"]) == len(set(case["relevant_card_ids"])) == 5
        assert [
            case["graded_relevance"][card_id]
            for card_id in case["relevant_card_ids"]
        ] == [5.0, 4.0, 3.0, 2.0, 1.0]
        assert all(
            card_by_id[card_id]["category"] == case["category"]
            for card_id in case["relevant_card_ids"]
        )


def test_taobao_zh_manifest_matches_frozen_cases() -> None:
    manifest = json.loads(
        (TAOBAO_DIR / "category_recall_manifest_taobao_zh.json").read_text(
            encoding="utf-8"
        )
    )
    cases_path = TAOBAO_DIR / "category_recall_cases_taobao_zh.jsonl"
    digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    assert manifest["dataset_version"] == "category-card-recall-taobao-zh-v3"
    assert manifest["language"] == "zh"
    assert manifest["query_count"] == 50
    assert manifest["card_count"] == 48
    assert manifest["positive_count_per_query"] == 5
    assert manifest["closed_pool"] is True
    assert manifest["unjudged_policy"] == "forbidden"
    assert manifest["cases_sha256"] == digest
