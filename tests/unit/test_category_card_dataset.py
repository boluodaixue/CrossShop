import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from globex_agent.category_insight import CategoryCard, admit_card

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_category_taxonomy_accounts_for_every_source_query_once() -> None:
    taxonomy = json.loads(
        (DATA_DIR / "category_taxonomy.json").read_text(encoding="utf-8")
    )
    cases = _read_jsonl(ESCI_DIR / "recall_cases.jsonl")
    source_ids = {str(case["query_id"]) for case in cases}
    assigned_ids = [
        str(query_id)
        for category in taxonomy["categories"]
        for query_id in category["source_query_ids"]
    ]
    excluded_ids = set(taxonomy["excluded_query_groups"])

    assert len(assigned_ids) == len(set(assigned_ids))
    assert set(assigned_ids).isdisjoint(excluded_ids)
    assert set(assigned_ids) | excluded_ids == source_ids


def test_checked_in_category_cards_pass_course_admission_contract() -> None:
    cards = _read_jsonl(DATA_DIR / "category_cards.jsonl")
    assert len(cards) == 72
    assert len({card["card_id"] for card in cards}) == len(cards)
    assert Counter(card["card_type"] for card in cards) == {
        "bestseller": 24,
        "attribute": 36,
        "price_range": 12,
    }
    assert set(Counter(card["category"] for card in cards).values()) == {6}

    for raw in cards:
        assert set(raw) == {
            "card_id",
            "category",
            "card_type",
            "summary",
            "raw_evidence",
            "last_updated",
            "confidence",
        }
        card = CategoryCard.model_validate(raw)
        result = admit_card(raw)
        assert result.accepted, result.reason
        assert len(card.summary) <= 200
        assert 1 <= len(card.raw_evidence) <= 3
        assert all(len(evidence) <= 80 for evidence in card.raw_evidence)
        if card.card_type == "attribute":
            percentages = [
                float(token.strip().rsplit(" ", 1)[1].rstrip("%"))
                for token in card.summary.split("：", 1)[1].split("/")
            ]
            assert 99.8 <= sum(percentages) <= 100.2


def test_card_provenance_keeps_generated_values_out_of_observed_truth() -> None:
    facts = _read_jsonl(DATA_DIR / "category_item_facts.jsonl")
    cards = _read_jsonl(DATA_DIR / "category_cards.jsonl")
    provenance = _read_jsonl(DATA_DIR / "category_card_provenance.jsonl")
    source_items = {
        row["document_id"] for row in _read_jsonl(ESCI_DIR / "recall_items.jsonl")
    }

    assert len(facts) == 370
    assert {fact["product_id"] for fact in facts} <= source_items
    assert {row["card_id"] for row in provenance} == {
        card["card_id"] for card in cards
    }
    for fact in facts:
        generated = fact["generated_fields"]
        assert generated["price_source"] == "generated_offline_not_observed"
        assert generated["popularity_source"] == "generated_offline_not_observed"
        assert generated["seed"] == 20260815

    bestseller_provenance = [
        row for row in provenance if "not_real_world_bestseller" in row
    ]
    price_provenance = [
        row for row in provenance if "not_observed_transaction_price" in row
    ]
    assert bestseller_provenance
    assert all(row["not_real_world_bestseller"] for row in bestseller_provenance)
    assert price_provenance
    assert all(row["not_observed_transaction_price"] for row in price_provenance)


def test_title_exclusion_gate_and_attribute_precedence_block_audited_noise() -> None:
    facts = _read_jsonl(DATA_DIR / "category_item_facts.jsonl")
    rejected = _read_jsonl(DATA_DIR / "rejected_memberships.jsonl")
    fact_ids = {fact["product_id"] for fact in facts}
    rejected_by_id = {row["product_id"]: row for row in rejected}

    noisy_ids = {
        "B000COC67E",  # generic garage seat
        "B01511D1VM",  # compatible label tape
        "B075WSCG6T",  # compatible label tape
        "B07HSRC4Z9",  # ATV tire/rim accessory
        "B0899QWSHD",  # gasoline go-kart steering part
        "B07JKJTDWC",  # bicycle-only tool
        "B07Y38GJMB",  # bicycle-only tool
    }
    assert noisy_ids.isdisjoint(fact_ids)
    assert noisy_ids <= rejected_by_id.keys()
    assert all(
        rejected_by_id[product_id]["reason"]
        == "title matches category excluded_terms gate"
        for product_id in noisy_ids
    )

    no_hole_item = next(fact for fact in facts if fact["product_id"] == "B0058PTK6M")
    drainage = [
        row for row in no_hole_item["text_supported_attributes"]
        if row["name"] == "Drainage"
    ]
    assert drainage == [
        {
            "matched_term": "no drain holes",
            "name": "Drainage",
            "value": "without holes",
        }
    ]


def test_manifest_hashes_and_audit_ratio_match_checked_in_outputs() -> None:
    manifest = json.loads(
        (DATA_DIR / "category_card_manifest.json").read_text(encoding="utf-8")
    )
    cards = _read_jsonl(DATA_DIR / "category_cards.jsonl")
    audit_queue = _read_jsonl(DATA_DIR / "audit_queue.jsonl")
    output_names = {
        "facts": "category_item_facts.jsonl",
        "cards": "category_cards.jsonl",
        "provenance": "category_card_provenance.jsonl",
        "rejected_memberships": "rejected_memberships.jsonl",
        "rejected_cards": "rejected_cards.jsonl",
        "audit_queue": "audit_queue.jsonl",
    }

    assert manifest["truth_boundary"]["generated"] == [
        "typical_price_cny",
        "order_count_90d",
    ]
    assert len(audit_queue) == math.ceil(len(cards) * 0.10)
    assert {row["card_id"] for row in audit_queue} <= {
        card["card_id"] for card in cards
    }
    for key, filename in output_names.items():
        assert manifest["output_sha256"][key] == _sha256(DATA_DIR / filename)


def test_manual_audit_results_cover_the_deterministic_sample() -> None:
    manifest = json.loads(
        (DATA_DIR / "category_card_manifest.json").read_text(encoding="utf-8")
    )
    queue = _read_jsonl(DATA_DIR / "audit_queue.jsonl")
    results = _read_jsonl(DATA_DIR / "audit_results.jsonl")

    assert {row["card_id"] for row in results} == {row["card_id"] for row in queue}
    assert all(row["status"] in {"passed", "passed_after_correction"} for row in results)
    assert all(
        row["reviewed_card_corpus_sha256"] == manifest["output_sha256"]["cards"]
        for row in results
    )


def test_admission_rejects_undeclared_or_unparseable_cards() -> None:
    valid = {
        "card_id": "cc-test",
        "category": "test category",
        "card_type": "bestseller",
        "summary": "test category：form one / form two",
        "raw_evidence": ["item | 99 | generated orders=10"],
        "last_updated": "2026-08-15T00:00:00+08:00",
        "confidence": 0.6,
    }
    assert admit_card(valid).accepted
    assert not admit_card({**valid, "unexpected": True}).accepted
    assert not admit_card({**valid, "confidence": 0.49}).accepted
    assert not admit_card({**valid, "raw_evidence": ["not parseable"]}).accepted


def test_v3_bilingual_corpus_preserves_contract_and_source_boundaries() -> None:
    facts = _read_jsonl(DATA_DIR / "category_item_facts_v3.jsonl")
    cards = _read_jsonl(DATA_DIR / "category_cards_v3.jsonl")
    provenance = _read_jsonl(DATA_DIR / "category_card_provenance_v3.jsonl")
    retrieval = _read_jsonl(DATA_DIR / "category_retrieval_texts_v3.jsonl")
    manifest = json.loads(
        (DATA_DIR / "category_card_manifest_v3.json").read_text(encoding="utf-8")
    )

    assert len(facts) == 1356
    assert Counter(fact["source_kind"] for fact in facts) == {
        "real_esci": 370,
        "synthetic_llm_template": 986,
    }
    assert min(Counter(fact["category"] for fact in facts).values()) >= 60
    assert all(fact["category_zh"] for fact in facts)
    for fact in facts:
        if fact["source_kind"] == "real_esci":
            assert (
                fact["bilingual_translation"]["review_status"]
                == "machine_generated_unreviewed"
            )
        else:
            assert fact["source"]["review_status"] == "machine_generated_unreviewed"

    assert len(cards) == 200
    assert Counter(card["card_type"] for card in cards) == {
        "attribute": 140,
        "bestseller": 40,
        "price_range": 20,
    }
    assert set(Counter(card["category"] for card in cards).values()) == {10}
    for raw in cards:
        assert set(raw) == {
            "card_id",
            "category",
            "card_type",
            "summary",
            "raw_evidence",
            "last_updated",
            "confidence",
        }
        assert admit_card(raw).accepted
        if raw["card_type"] == "bestseller":
            assert all(len(str(evidence).split("|")) == 3 for evidence in raw["raw_evidence"])

    card_ids = {card["card_id"] for card in cards}
    assert {row["card_id"] for row in provenance} == card_ids
    assert {row["card_id"] for row in retrieval} == card_ids
    assert all(row["retrieval_text_en"] for row in retrieval)
    assert all(row["retrieval_text_zh"] for row in retrieval)
    assert manifest["truth_boundary"]["price_and_popularity"].startswith("generated")


def test_v3_eval_is_fully_judged_and_language_pairs_share_labels() -> None:
    cards = _read_jsonl(DATA_DIR / "category_cards_v3.jsonl")
    cases = _read_jsonl(DATA_DIR / "category_recall_cases_v3.jsonl")
    manifest = json.loads(
        (DATA_DIR / "category_recall_manifest_v3.json").read_text(encoding="utf-8")
    )
    card_ids = {card["card_id"] for card in cards}

    assert len(cases) == 480
    assert manifest["intent_count"] == 240
    assert manifest["language_counts"] == {"en": 240, "zh": 240}
    assert manifest["split_counts"] == {"dev": 160, "test": 160, "train": 160}
    assert manifest["query_type_counts"] == {
        "attribute_constraint": 120,
        "colloquial": 120,
        "noun": 120,
        "style": 120,
    }
    assert manifest["closed_pool"] is True
    by_intent: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        by_intent.setdefault(str(case["intent_id"]), []).append(case)
        assert set(case["candidate_ids"]) == card_ids
        assert set(case["judgments"]) == card_ids
        assert len(case["relevant_card_ids"]) == 5
        assert sum(value > 0 for value in case["graded_relevance"].values()) == 5

    for pair in by_intent.values():
        assert {case["language"] for case in pair} == {"en", "zh"}
        assert pair[0]["split"] == pair[1]["split"]
        assert pair[0]["query_type"] == pair[1]["query_type"]
        assert pair[0]["candidate_ids"] == pair[1]["candidate_ids"]
        assert pair[0]["relevant_card_ids"] == pair[1]["relevant_card_ids"]
        assert pair[0]["graded_relevance"] == pair[1]["graded_relevance"]


def test_v3_final_holdout_is_new_balanced_and_fully_judged() -> None:
    cards = _read_jsonl(DATA_DIR / "category_cards_v3.jsonl")
    development_cases = _read_jsonl(DATA_DIR / "category_recall_cases_v3.jsonl")
    cases = _read_jsonl(DATA_DIR / "category_recall_final_test_v3.jsonl")
    manifest = json.loads(
        (DATA_DIR / "category_recall_final_test_manifest_v3.json").read_text(
            encoding="utf-8"
        )
    )
    card_ids = {card["card_id"] for card in cards}

    assert len(cases) == 320
    assert manifest["intent_count"] == 160
    assert manifest["language_counts"] == {"en": 160, "zh": 160}
    assert manifest["split_counts"] == {"final_test": 320}
    assert manifest["query_type_counts"] == {
        "attribute_constraint": 80,
        "colloquial": 80,
        "noun": 80,
        "style": 80,
    }
    assert set(Counter(case["category"] for case in cases).values()) == {16}
    assert {case["query"].casefold() for case in cases}.isdisjoint(
        {case["query"].casefold() for case in development_cases}
    )

    by_intent: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        by_intent.setdefault(str(case["intent_id"]), []).append(case)
        assert set(case["candidate_ids"]) == card_ids
        assert set(case["judgments"]) == card_ids
        assert len(case["relevant_card_ids"]) == 5
    for pair in by_intent.values():
        assert {case["language"] for case in pair} == {"en", "zh"}
        assert pair[0]["candidate_ids"] == pair[1]["candidate_ids"]
        assert pair[0]["graded_relevance"] == pair[1]["graded_relevance"]


def test_taobao_zh_taxonomy_defines_eight_ordinary_categories() -> None:
    taxonomy = json.loads(
        (TAOBAO_DIR / "category_taxonomy_taobao_zh.json").read_text(encoding="utf-8")
    )
    categories = taxonomy["categories"]

    assert len(categories) == 8
    assert len({category["category"] for category in categories}) == 8
    assert len({category["slug"] for category in categories}) == 8
    assert all(category.get("category_kind", taxonomy["category_kind"]) == "ordinary"
               for category in categories)
    assert all(len(category["popular_forms"]) == 6 for category in categories)
    assert all(len(category["attribute_rules"]) == 3 for category in categories)


def test_taobao_zh_cards_pass_course_admission_and_use_real_observed_prices() -> None:
    cards = _read_jsonl(TAOBAO_DIR / "category_cards_taobao_zh.jsonl")
    provenance = _read_jsonl(TAOBAO_DIR / "category_card_provenance_taobao_zh.jsonl")

    assert len(cards) == 48
    assert len({card["card_id"] for card in cards}) == len(cards)
    assert Counter(card["card_type"] for card in cards) == {
        "bestseller": 16,
        "attribute": 24,
        "price_range": 8,
    }
    assert set(Counter(card["category"] for card in cards).values()) == {6}
    assert {row["card_id"] for row in provenance} == {card["card_id"] for card in cards}

    for raw in cards:
        assert set(raw) == {
            "card_id",
            "category",
            "card_type",
            "summary",
            "raw_evidence",
            "last_updated",
            "confidence",
        }
        card = CategoryCard.model_validate(raw)
        assert admit_card(raw).accepted
        assert len(card.summary) <= 200
        assert 1 <= len(card.raw_evidence) <= 3
        assert all(len(evidence) <= 80 for evidence in card.raw_evidence)
        if card.card_type == "attribute":
            percentages = [
                float(token.strip().rsplit(" ", 1)[1].rstrip("%"))
                for token in card.summary.split("：", 1)[1].split("/")
            ]
            assert 99.8 <= sum(percentages) <= 100.2

    price_provenance = [
        row for row in provenance if row.get("price_measure") == "observed_listed_prices_cny"
    ]
    assert len(price_provenance) == 8
    assert all(row["listed_price_not_transaction"] for row in price_provenance)
    assert all(row["sample_count"] > 0 for row in price_provenance)


def test_taobao_zh_bestseller_proxy_and_audit_queue_are_explicit() -> None:
    facts = _read_jsonl(TAOBAO_DIR / "category_item_facts_taobao_zh.jsonl")
    provenance = _read_jsonl(TAOBAO_DIR / "category_card_provenance_taobao_zh.jsonl")
    audit_queue = _read_jsonl(TAOBAO_DIR / "audit_queue_taobao_zh.jsonl")
    manifest = json.loads(
        (TAOBAO_DIR / "category_card_manifest_taobao_zh.json").read_text(encoding="utf-8")
    )

    assert len(facts) == 779
    assert all(
        fact["proxy_fields"]["bestseller_proxy_source"] == "generated_offline_proxy"
        for fact in facts
    )
    assert all(fact["price_source"] == "observed" for fact in facts)
    assert all(
        fact["price_quality"]["policy_version"]
        == "log-iqr-1.5-positive-noncomparable-v2"
        for fact in facts
    )
    assert all(
        all(float(value) > 0 for value in fact["price_observations_cny"])
        for fact in facts
        if fact["price_observations_cny"]
    )
    assert any(fact["price_exclusions"] for fact in facts)
    car_facts = [fact for fact in facts if fact["category"] == "汽车氛围灯"]
    assert car_facts
    pollution_terms = ("警示", "爆闪", "日行灯", "日间行车", "刹车灯", "尾灯", "转向灯")
    assert all(not any(term in fact["title"] for term in pollution_terms) for fact in car_facts)
    bestseller_provenance = [
        row
        for row in provenance
        if row.get("evidence_source", "").startswith("generated_offline_proxy")
    ]
    assert len(bestseller_provenance) == 16
    assert all("generated_offline_proxy" in row["evidence_source"] for row in bestseller_provenance)
    assert all(
        row["proxy_scope"] == "目录高频款型代理/高频形态，不是真实销量榜"
        for row in bestseller_provenance
    )
    price_provenance = [
        row
        for row in provenance
        if row.get("price_measure") == "observed_listed_prices_cny"
    ]
    assert len(price_provenance) == 8
    assert all(
        row["price_scope"]
        == "目录挂牌/规格价样本的品类参考区间，不是成交价、具体 SKU 价格或实时价格"
        for row in price_provenance
    )
    assert all(
        row["quality_policy_version"]
        == "log-iqr-1.5-positive-noncomparable-v2"
        for row in price_provenance
    )
    assert all(row["excluded_observation_count"] >= 0 for row in price_provenance)
    assert len(audit_queue) == math.ceil(48 * 0.10)
    assert manifest["truth_boundary"]["generated"] == ["bestseller_proxy_rank"]
    assert manifest["dataset_version"] == "category-cards-taobao-zh-v3"
    assert (
        manifest["generation_rules"]["price_policy"]["version"]
        == "log-iqr-1.5-positive-noncomparable-v2"
    )
