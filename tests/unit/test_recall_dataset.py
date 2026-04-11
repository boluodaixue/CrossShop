import hashlib
import json
from collections import Counter
from pathlib import Path


def test_checked_in_esci_subset_has_disjoint_query_groups() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "eval"
    cases = [
        json.loads(line)
        for line in (data_dir / "recall_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items = [
        json.loads(line)
        for line in (data_dir / "recall_items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((data_dir / "recall_manifest.json").read_text(encoding="utf-8"))

    assert len(cases) == 35
    assert len(items) == 720
    assert len({case["query_id"] for case in cases}) == 35
    assert len({case["query"] for case in cases}) == 35
    assert Counter(case["split"] for case in cases) == {
        "train": 21,
        "dev": 6,
        "test": 8,
    }
    item_ids = {item["document_id"] for item in items}
    for case in cases:
        candidate_ids = set(case["candidate_ids"])
        assert candidate_ids == set(case["labels"])
        assert candidate_ids == set(case["relevance"])
        assert candidate_ids == set(case["graded_relevance"])
        assert candidate_ids <= item_ids
        exact_ids = {
            document_id
            for document_id, label in case["labels"].items()
            if label == "Exact"
        }
        assert 2 <= len(exact_ids) <= 5
        assert exact_ids == {
            document_id
            for document_id, gain in case["relevance"].items()
            if gain > 0
        }
    assert all(
        not item["body"].startswith(item["title"])
        for item in items
        if item["title"]
    )
    assert all("None" not in item["body"].split() for item in items)
    assert manifest["license"] == "Apache-2.0"
    assert manifest["random_seed"] is None
    assert manifest["closed_pool"] is True
    assert manifest["unjudged_policy"] == "forbidden"
    assert manifest["positive_labels_for_recall_mrr"] == ["Exact"]


def test_synthetic_bilingual_product_v3_is_separate_and_fully_judged() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "eval" / "product_v3"
    cases_path = data_dir / "recall_cases.jsonl"
    items_path = data_dir / "recall_items.jsonl"
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items = [
        json.loads(line)
        for line in items_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(
        (data_dir / "recall_manifest.json").read_text(encoding="utf-8")
    )

    assert len(items) == 1356
    assert len({item["document_id"] for item in items}) == 1356
    assert Counter(item["source_kind"] for item in items) == {
        "real_esci": 370,
        "synthetic_llm_template": 986,
    }
    assert len(cases) == 160
    assert len({case["intent_id"] for case in cases}) == 80
    assert Counter(case["language"] for case in cases) == {"en": 80, "zh": 80}
    item_ids = {item["document_id"] for item in items}
    by_intent: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        by_intent.setdefault(case["intent_id"], []).append(case)
        candidates = set(case["candidate_ids"])
        assert len(candidates) == 120
        assert candidates <= item_ids
        assert candidates == set(case["labels"])
        assert candidates == set(case["relevance"])
        assert candidates == set(case["graded_relevance"])
        assert sum(label == "Exact" for label in case["labels"].values()) == 5
        assert {
            document_id
            for document_id, gain in case["relevance"].items()
            if gain > 0
        } == {
            document_id
            for document_id, label in case["labels"].items()
            if label == "Exact"
        }
    for pair in by_intent.values():
        assert {case["language"] for case in pair} == {"en", "zh"}
        assert pair[0]["candidate_ids"] == pair[1]["candidate_ids"]
        assert pair[0]["labels"] == pair[1]["labels"]

    assert manifest["role"].startswith("separate synthetic bilingual")
    assert manifest["closed_pool"] is True
    assert manifest["unjudged_policy"].startswith("forbidden")
    assert manifest["output_sha256"] == {
        "cases": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "items": hashlib.sha256(items_path.read_bytes()).hexdigest(),
    }
