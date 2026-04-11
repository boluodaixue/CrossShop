"""Build the frozen v4 challenge sets for cards and product recall.

v3 remains an intentionally easy smoke set.  This script creates new files
instead of replacing it.  The v4 sets use a larger corpus, more same-category
competition, fully judged closed pools, paired bilingual queries, and exactly
five binary positives per intent.
"""

# ruff: noqa: E501, E731 -- bilingual fixtures and inline predicates stay atomic.

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from globex_agent.category_insight import admit_card

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATEGORY_DIR = PROJECT_ROOT / "data" / "category_insight"
PRODUCT_DIR = PROJECT_ROOT / "data" / "eval" / "product_v4"
SNAPSHOT_TIME = "2026-08-15T00:00:00+08:00"
SEED = 20260815
FACT_VERSION = "category-item-facts-challenge-v4"
CARD_VERSION = "category-cards-challenge-v4"
KNOWLEDGE_EVAL_VERSION = "category-recall-challenge-v4"
PRODUCT_EVAL_VERSION = "product-recall-challenge-v4"
NEW_SOURCE_KIND = "synthetic_challenge_v4"
FACTS_PER_CATEGORY = 100
MIN_NEW_FACTS_PER_CATEGORY = 50
CARDS_PER_CATEGORY = 15
KNOWLEDGE_POSITIVES = 5
PRODUCT_POOL_SIZE = 500
PRODUCT_POSITIVES = 5
QUERY_TYPES = ("noun", "attribute_constraint", "style", "colloquial")
CONDITIONAL_FACETS = (0, 2, 4, 5, 6)


SCENARIOS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "home ventilation fans": (("keep a humid bathroom fresh", "让潮湿浴室更清爽"), ("clear cooking smells without a noisy room", "减少厨房异味又别太吵")),
    "utility tires and wheels": (("replace a cart wheel that keeps going flat", "替换总漏气的园艺车轮"), ("move heavy loads over a rough yard", "在不平的院子里搬重物")),
    "lawn mower equipment": (("tidy a small lawn with little storage space", "打理小草坪而且收纳空间有限"), ("finish a larger yard with less effort", "省力完成较大草坪的修剪")),
    "outdoor fencing and gates": (("screen a backyard from passing eyes", "遮挡后院视线"), ("mark a garden boundary without digging", "不用挖地划分花园边界")),
    "seedling and plant trays": (("start seedlings on an indoor shelf", "在室内架子上育苗"), ("keep propagation moisture under control", "控制扦插育苗的湿度")),
    "label makers and printers": (("organize cables and storage boxes", "整理线缆和收纳箱"), ("print shipping labels from a desk", "在桌面打印快递面单")),
    "extension cords and power strips": (("power several devices while travelling", "旅行时给多个设备供电"), ("reach outdoor equipment safely", "安全连接较远的户外设备")),
    "garden warning signs": (("ask visitors to protect a flower bed", "提醒访客保护花坛"), ("make a yard warning visible at night", "让庭院提示在夜间也醒目")),
    "graphic slogan t-shirts": (("pick a relaxed conversation starter", "挑一件轻松有话题感的衣服"), ("find a simple gift with some humor", "找一件简洁又有幽默感的礼物")),
    "mens zipper wallets": (("carry many cards without accidental opening", "多卡收纳并避免意外散开"), ("keep travel cards harder to scan", "旅行时降低卡片被盗刷风险")),
    "pedal go-karts": (("let a child ride around the yard", "让孩子在院子里骑行"), ("adjust the ride as a child grows", "随着孩子长大调节乘坐尺寸")),
    "dirt bike tools and accessories": (("handle a trackside chain repair", "在赛道边处理链条维修"), ("carry basic motorcycle service tools", "随车携带基础摩托保养工具")),
    "wireless earbuds": (("hear calls clearly on a noisy commute", "嘈杂通勤中听清电话"), ("exercise without worrying about sweat", "运动时减少汗水带来的担心")),
    "portable bluetooth speakers": (("play music beside a pool", "在泳池边播放音乐"), ("carry fuller sound to a small gathering", "为小型聚会带来更饱满的声音")),
    "mechanical keyboards": (("type quietly in a shared office", "在共享办公室安静打字"), ("get faster response for desk gaming", "桌面游戏需要更快响应")),
    "wireless computer mice": (("work all day without wrist strain", "全天办公减少手腕负担"), ("switch between a laptop and a tablet", "在笔记本和平板之间切换")),
    "USB-C hubs and docking stations": (("connect one laptop to a desk setup", "把一台笔记本接入桌面设备"), ("add displays and fast card access", "扩展显示器和高速读卡")),
    "power banks": (("keep a phone alive during a trip", "旅行途中给手机续航"), ("charge a laptop away from an outlet", "离开插座时给笔记本充电")),
    "webcams": (("look clearer in a dim video meeting", "在昏暗视频会议中更清晰"), ("stream with smoother motion and framing", "直播时获得更流畅画面和构图")),
    "Wi-Fi routers": (("cover several rooms without dead zones", "覆盖多个房间并减少死角"), ("keep gaming stable while others stream", "家人看视频时仍保持游戏稳定")),
}

NEIGHBOR_GROUPS = (
    ("wireless earbuds", "portable bluetooth speakers", "mechanical keyboards", "wireless computer mice", "USB-C hubs and docking stations", "power banks", "webcams", "Wi-Fi routers", "label makers and printers", "extension cords and power strips"),
    ("home ventilation fans", "lawn mower equipment", "outdoor fencing and gates", "seedling and plant trays", "garden warning signs"),
    ("utility tires and wheels", "pedal go-karts", "dirt bike tools and accessories", "lawn mower equipment"),
    ("graphic slogan t-shirts", "mens zipper wallets"),
)


def main() -> None:
    specs_path = CATEGORY_DIR / "category_generation_specs_v3.json"
    v3_facts_path = CATEGORY_DIR / "category_item_facts_v3.jsonl"
    specs = json.loads(specs_path.read_text(encoding="utf-8"))["categories"]
    v3_facts = _read_jsonl(v3_facts_path)
    facts = _build_facts(v3_facts, specs)
    cards, provenance, retrieval = _build_cards(facts, specs)
    knowledge_cases = _build_knowledge_cases(cards, specs)
    product_documents, product_cases = _build_product_eval(facts, specs)

    CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    PRODUCT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "facts": CATEGORY_DIR / "category_item_facts_v4.jsonl",
        "cards": CATEGORY_DIR / "category_cards_v4.jsonl",
        "provenance": CATEGORY_DIR / "category_card_provenance_v4.jsonl",
        "retrieval": CATEGORY_DIR / "category_retrieval_texts_v4.jsonl",
        "knowledge_cases": CATEGORY_DIR / "category_recall_challenge_v4.jsonl",
        "product_items": PRODUCT_DIR / "recall_items.jsonl",
        "product_cases": PRODUCT_DIR / "recall_cases.jsonl",
    }
    _write_jsonl(paths["facts"], facts)
    _write_jsonl(paths["cards"], cards)
    _write_jsonl(paths["provenance"], provenance)
    _write_jsonl(paths["retrieval"], retrieval)
    _write_jsonl(paths["knowledge_cases"], knowledge_cases)
    _write_jsonl(paths["product_items"], product_documents)
    _write_jsonl(paths["product_cases"], product_cases)

    card_manifest = _card_manifest(facts, cards, paths, v3_facts_path, specs_path)
    knowledge_manifest = _knowledge_manifest(cards, knowledge_cases, paths, specs_path)
    product_manifest = _product_manifest(facts, product_documents, product_cases, paths)
    card_manifest_path = CATEGORY_DIR / "category_card_manifest_v4.json"
    knowledge_manifest_path = CATEGORY_DIR / "category_recall_challenge_manifest_v4.json"
    product_manifest_path = PRODUCT_DIR / "recall_manifest.json"
    _write_json(card_manifest_path, card_manifest)
    _write_json(knowledge_manifest_path, knowledge_manifest)
    _write_json(product_manifest_path, product_manifest)

    freeze = {
        "dataset_version": "challenge-freeze-v4",
        "frozen_at": SNAPSHOT_TIME,
        "purpose": "one-shot challenge evaluation; v3 remains the smoke/easy set",
        "knowledge_retrieval": {
            "index_name": "globex_category_kb_v4",
            "opensearch_analyzer": "ik_max_word",
            "opensearch_search_analyzer": "ik_smart",
            "embedding_model": "BAAI/bge-m3",
            "coarse_k": 30,
            "top_k": 10,
            "dynamic_weights": {
                "noun": {"semantic": 0.5, "bm25": 0.5},
                "attribute_constraint": {"semantic": 0.7, "bm25": 0.3},
                "style": {"semantic": 0.9, "bm25": 0.1},
                "colloquial": {"semantic": 1.0, "bm25": 0.0},
            },
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_mode": "contextual",
            "reranker_device": "cuda:0",
            "reranker_precision": "fp16",
        },
        "product_retrieval": {
            "mainline": "Query/Item BGE-M3 ANN Top-100 -> BGE reranker Top-10",
            "index": "Faiss HNSW + inner product",
            "candidate_k": 100,
            "top_k": 10,
            "bm25_role": "baseline only",
            "hybrid_role": "optional ablation only",
        },
        "data_sha256": {
            name: _sha256(path)
            for name, path in paths.items()
        },
        "manifest_sha256": {
            "cards": _sha256(card_manifest_path),
            "knowledge": _sha256(knowledge_manifest_path),
            "products": _sha256(product_manifest_path),
        },
    }
    _write_json(CATEGORY_DIR / "challenge_freeze_v4.json", freeze)
    print(
        f"built v4 facts={len(facts)} cards={len(cards)} "
        f"knowledge_queries={len(knowledge_cases)} product_queries={len(product_cases)}"
    )


def _build_facts(v3_facts: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in v3_facts:
        by_category[str(fact["category"])].append(fact)
    output: list[dict[str, Any]] = []
    for spec in specs:
        category = str(spec["category"])
        existing = sorted(by_category[category], key=lambda row: str(row["product_id"]))
        for fact in existing:
            normalized = dict(fact)
            normalized["dataset_version"] = FACT_VERSION
            form_en, form_zh = _infer_form(normalized, spec)
            normalized["retrieval_form_en"] = form_en
            normalized["retrieval_form_zh"] = form_zh
            normalized["v4_lineage"] = "retained_from_v3"
            output.append(normalized)
        new_count = max(
            MIN_NEW_FACTS_PER_CATEGORY,
            FACTS_PER_CATEGORY - len(existing),
        )
        for index in range(new_count):
            output.append(_challenge_fact(spec, index))
    output.sort(key=lambda row: (str(row["category"]), str(row["product_id"])))
    counts = Counter(row["category"] for row in output)
    if len(counts) != 20 or min(counts.values()) < FACTS_PER_CATEGORY:
        raise ValueError("v4 requires twenty categories and at least 100 facts per category")
    return output


def _challenge_fact(spec: dict[str, Any], index: int) -> dict[str, Any]:
    category = str(spec["category"])
    product_id = f"CH4-{_slug(category).upper()}-{index + 1:03d}"
    form_en, form_zh = spec["forms"][index % len(spec["forms"])]
    attributes = []
    for facet_index, facet in enumerate(spec["facets"]):
        value_en, value_zh = _hashed_pick(
            facet["values"], category, product_id, f"facet-{facet_index}"
        )
        attributes.append(
            {
                "name": facet["name"],
                "name_zh": facet["name_zh"],
                "value": value_en,
                "value_zh": value_zh,
                "evidence_kind": NEW_SOURCE_KIND,
            }
        )
    visible = sorted(
        range(len(attributes)),
        key=lambda facet_index: _digest(category, product_id, f"title-{facet_index}"),
    )[:2]
    title_en = f"Globex Offline {form_en}, " + ", ".join(attributes[i]["value"] for i in visible)
    title_zh = f"Globex离线样例 {form_zh}，" + "、".join(attributes[i]["value_zh"] for i in visible)
    body_en = "; ".join(f"{row['name']}: {row['value']}" for row in attributes)
    body_zh = "；".join(f"{row['name_zh']}：{row['value_zh']}" for row in attributes)
    lower, upper = (float(value) for value in spec["price_bounds_cny"])
    return {
        "dataset_version": FACT_VERSION,
        "fact_id": f"{_slug(category)}:{product_id}",
        "category": category,
        "category_zh": spec["category_zh"],
        "category_kind": "ordinary",
        "product_id": product_id,
        "title": title_en,
        "title_en": title_en,
        "title_zh": title_zh,
        "body_en": body_en,
        "body_zh": body_zh,
        "retrieval_form_en": form_en,
        "retrieval_form_zh": form_zh,
        "text_supported_attributes": attributes,
        "generated_fields": {
            "typical_price_cny": _number(category, product_id, lower, upper, "price"),
            "order_count_90d": int(_number(category, product_id, 40, 1000, "orders")),
            "price_source": "generated_offline_not_observed",
            "popularity_source": "generated_offline_not_observed",
            "seed": SEED,
            "rule_version": "challenge-v4-independent-hash-facets",
        },
        "source_kind": NEW_SOURCE_KIND,
        "source": {
            "kind": NEW_SOURCE_KIND,
            "generation_seed": SEED,
            "review_status": "machine_generated_unreviewed",
        },
        "v4_lineage": "new_challenge_expansion",
    }


def _infer_form(fact: dict[str, Any], spec: dict[str, Any]) -> tuple[str, str]:
    title = str(fact.get("title_en", "")).casefold()
    for form_en, form_zh in spec["forms"]:
        if str(form_en).casefold() in title:
            return str(form_en), str(form_zh)
    return _hashed_pick(spec["forms"], str(spec["category"]), str(fact["product_id"]), "inferred-form")


def _build_cards(
    facts: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        by_category[str(fact["category"])].append(fact)
    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for spec in specs:
        category_facts = by_category[str(spec["category"])]
        generated = _bestseller_cards(spec, category_facts)
        generated.extend(_global_attribute_cards(spec, category_facts))
        generated.extend(_conditional_attribute_cards(spec, category_facts))
        generated.append(_price_card(spec, category_facts))
        if len(generated) != CARDS_PER_CATEGORY:
            raise ValueError(f"{spec['category']} did not produce fifteen cards")
        for raw_card, card_provenance, summary_zh in generated:
            admission = admit_card(raw_card)
            if not admission.accepted or admission.card is None:
                raise ValueError(f"v4 card rejected {raw_card['card_id']}: {admission.reason}")
            card = admission.card.model_dump(mode="json")
            cards.append(card)
            provenance.append(card_provenance)
            retrieval_rows.append(_retrieval_row(card, spec, summary_zh))
    cards.sort(key=lambda row: str(row["card_id"]))
    provenance.sort(key=lambda row: str(row["card_id"]))
    retrieval_rows.sort(key=lambda row: str(row["card_id"]))
    if len(cards) != 300 or len({row["card_id"] for row in cards}) != 300:
        raise ValueError("v4 requires 300 unique cards")
    return cards, provenance, retrieval_rows


def _bestseller_cards(spec: dict[str, Any], facts: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    ranked = sorted(facts, key=lambda row: (-int(row["generated_fields"]["order_count_90d"]), str(row["product_id"])))[:6]
    output = []
    for index in range(2):
        batch = ranked[index * 3 : index * 3 + 3]
        forms = [spec["forms"][(index * 2 + offset) % len(spec["forms"])] for offset in range(3)]
        card_id = f"cc-{_slug(spec['category'])}-bestseller-{index + 1:02d}"
        card = {
            "card_id": card_id,
            "category": spec["category"],
            "card_type": "bestseller",
            "summary": f"{spec['category']}：{' / '.join(row[0] for row in forms)}",
            "raw_evidence": [_bestseller_evidence(row) for row in batch],
            "last_updated": SNAPSHOT_TIME,
            "confidence": _confidence(facts),
        }
        output.append((card, _provenance(card_id, facts, batch, "generated popularity ranking"), f"{spec['category_zh']}：{' / '.join(row[1] for row in forms)}"))
    return output


def _global_attribute_cards(spec: dict[str, Any], facts: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    return [
        _attribute_card(spec, facts, facts, facet_index, card_index=facet_index + 1, condition=None)
        for facet_index in range(7)
    ]


def _conditional_attribute_cards(spec: dict[str, Any], facts: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    output = []
    for offset, facet_index in enumerate(CONDITIONAL_FACETS):
        form_en, form_zh = spec["forms"][(offset + 1) % len(spec["forms"])]
        cohort = [row for row in facts if row["retrieval_form_en"] == form_en]
        if len(cohort) < 10:
            raise ValueError(f"{spec['category']} form cohort {form_en} is too small")
        output.append(
            _attribute_card(
                spec,
                facts,
                cohort,
                facet_index,
                card_index=8 + offset,
                condition=(str(form_en), str(form_zh)),
            )
        )
    return output


def _attribute_card(
    spec: dict[str, Any],
    category_facts: list[dict[str, Any]],
    cohort: list[dict[str, Any]],
    facet_index: int,
    *,
    card_index: int,
    condition: tuple[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    facet = spec["facets"][facet_index]
    value_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    value_zh = dict(facet["values"])
    for fact in cohort:
        attributes = _attributes(fact)
        if facet["name"] in attributes:
            value_rows[attributes[facet["name"]]].append(fact)
    count = sum(len(rows) for rows in value_rows.values())
    if count < 10:
        raise ValueError(f"{spec['category']} facet {facet['name']} has only {count} rows")
    ranked = sorted(
        value_rows.items(), key=lambda item: (-len(item[1]), item[0])
    )[:3]
    tokens_en = [f"{value} {len(rows) / count * 100:.1f}%" for value, rows in ranked]
    tokens_zh = [
        f"{value_zh.get(value, value)} {len(rows) / count * 100:.1f}%"
        for value, rows in ranked
    ]
    name_en = str(facet["name"])
    name_zh = str(facet["name_zh"])
    if condition is not None:
        name_en = f"{name_en} among {condition[0]}"
        name_zh = f"{condition[1]}中的{name_zh}"
    card_id = f"cc-{_slug(spec['category'])}-attribute-{card_index:02d}"
    source_rows = [row for _, rows in ranked for row in rows]
    card = {
        "card_id": card_id,
        "category": spec["category"],
        "card_type": "attribute",
        "summary": f"{name_en}：{' / '.join(tokens_en)}",
        "raw_evidence": [
            _truncate(f"{value}: {rows[0]['product_id']} {rows[0]['title_en']}")
            for value, rows in ranked[:3]
        ],
        "last_updated": SNAPSHOT_TIME,
        "confidence": round(min(0.9, 0.58 + count / len(category_facts) * 0.3), 2),
    }
    provenance = _provenance(card_id, category_facts, source_rows, "deterministic attribute aggregation")
    provenance.update({"attribute_name": facet["name"], "valid_sample_count": count, "condition": condition[0] if condition else None, "distribution_counts": {value: len(rows) for value, rows in ranked}})
    return card, provenance, f"{name_zh}：{' / '.join(tokens_zh)}"


def _price_card(spec: dict[str, Any], facts: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str]:
    prices = sorted(float(row["generated_fields"]["typical_price_cny"]) for row in facts)
    minimum = math.floor(prices[0] / 10) * 10
    first = max(minimum + 10, math.ceil(_percentile(prices, 1 / 3) / 10) * 10)
    second = max(first + 10, math.ceil(_percentile(prices, 2 / 3) / 10) * 10)
    maximum = max(second + 10, math.ceil(prices[-1] / 10) * 10)
    card_id = f"cc-{_slug(spec['category'])}-price-range-01"
    summary = f"便宜款 {minimum:.0f}-{first:.0f} / 中档 {first:.0f}-{second:.0f} / 高端 {second:.0f}-{maximum:.0f}"
    card = {
        "card_id": card_id,
        "category": spec["category"],
        "card_type": "price_range",
        "summary": summary,
        "raw_evidence": [f"generated CNY prices: n={len(prices)}, seed={SEED}", "offline tiers; not observed transactions"],
        "last_updated": SNAPSHOT_TIME,
        "confidence": 0.55,
    }
    return card, _provenance(card_id, facts, facts, "generated offline price quantiles"), f"入门档 {minimum:.0f}-{first:.0f}元 / 中端档 {first:.0f}-{second:.0f}元 / 高端档 {second:.0f}-{maximum:.0f}元"


def _retrieval_row(card: dict[str, Any], spec: dict[str, Any], summary_zh: str) -> dict[str, Any]:
    type_en = {"bestseller": "popular choices", "attribute": "attribute evidence", "price_range": "budget tiers"}[card["card_type"]]
    type_zh = {"bestseller": "热门选择", "attribute": "属性证据", "price_range": "预算档位"}[card["card_type"]]
    text_en = f"Category: {card['category']}. Knowledge type: {type_en}. Summary: {card['summary']}"
    text_zh = f"品类：{spec['category_zh']}。知识类型：{type_zh}。摘要：{summary_zh}"
    return {"card_id": card["card_id"], "retrieval_text_en": text_en, "retrieval_text_zh": text_zh, "retrieval_text_bilingual": f"{text_en} {text_zh}"}


def _build_knowledge_cases(cards: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    card_by_id = {str(row["card_id"]): row for row in cards}
    candidate_ids = sorted(card_by_id)
    cases = []
    for spec in sorted(specs, key=lambda row: str(row["category"])):
        for query_type in QUERY_TYPES:
            for variant in (1, 2):
                query_en, query_zh = _knowledge_query(spec, query_type, variant)
                roles, price_relevance = _knowledge_roles(query_type, variant)
                relevant = [_card_id(str(spec["category"]), role) for role in roles]
                gain = {card_id: float(5 - index) for index, card_id in enumerate(relevant)}
                intent_id = f"kcv4-{_slug(spec['category'])}-{query_type}-{variant:02d}"
                for language, query in (("en", query_en), ("zh", query_zh)):
                    cases.append({
                        "dataset_version": KNOWLEDGE_EVAL_VERSION,
                        "intent_id": intent_id,
                        "query_id": f"{intent_id}-{language}",
                        "query": query,
                        "language": language,
                        "query_source_kind": "model_authored_independent_challenge" if language == "en" else "synthetic_translation_pair",
                        "translation_review_status": "machine_generated_unreviewed",
                        "query_type": query_type,
                        "category": spec["category"],
                        "category_zh": spec["category_zh"],
                        "split": "final_test",
                        "price_relevance": price_relevance,
                        "candidate_ids": candidate_ids,
                        "relevant_card_ids": relevant,
                        "relevance": {card_id: float(card_id in gain) for card_id in candidate_ids},
                        "graded_relevance": {card_id: gain.get(card_id, 0.0) for card_id in candidate_ids},
                        "judgments": {card_id: (f"relevant_gain_{int(gain[card_id])}" if card_id in gain else "not_relevant") for card_id in candidate_ids},
                    })
    _validate_knowledge(cases, cards)
    return sorted(cases, key=lambda row: (str(row["intent_id"]), str(row["language"])))


def _knowledge_query(spec: dict[str, Any], query_type: str, variant: int) -> tuple[str, str]:
    category, category_zh = str(spec["category"]), str(spec["category_zh"])
    form_en, form_zh = spec["forms"][(variant + 1) % len(spec["forms"])]
    scenario_en, scenario_zh = SCENARIOS[category][variant - 1]
    if query_type == "noun":
        if variant == 1:
            return f"a practical overview of choices in {category}", f"想系统了解{category_zh}有哪些常见选择"
        budget = int(float(spec["price_bounds_cny"][0]) * 2.3)
        return f"what sells in {category} and is CNY {budget} a realistic budget", f"{category_zh}哪些卖得多，{budget}元预算是否合适"
    if query_type == "attribute_constraint":
        facet_index = 0 if variant == 1 else 2
        facet = spec["facets"][facet_index]
        value_en, value_zh = facet["values"][(variant + 1) % len(facet["values"])]
        return f"for a {form_en}, how common is {value_en} and what else matters", f"选{form_zh}时，{value_zh}常见吗，还要关注什么"
    if query_type == "style":
        return f"something understated and dependable to {scenario_en}", f"想要低调耐用的方案，用来{scenario_zh}"
    if query_type != "colloquial":
        raise ValueError(f"unsupported query type {query_type}")
    if variant == 1:
        return f"I'm lost; what would you pick to {scenario_en}", f"不太懂，想{scenario_zh}该怎么选"
    budget = int(float(spec["price_bounds_cny"][0]) * 2.0)
    return f"help me avoid overbuying; I need to {scenario_en} around CNY {budget}", f"别让我买过头，约{budget}元解决“{scenario_zh}”"


def _knowledge_roles(query_type: str, variant: int) -> tuple[list[str], str]:
    if query_type == "noun" and variant == 1:
        return ["bestseller-01", "bestseller-02", "attribute-01", "attribute-08", "price-range-01"], "medium"
    if query_type == "noun":
        return ["price-range-01", "bestseller-01", "bestseller-02", "attribute-01", "attribute-08"], "strong"
    if query_type == "attribute_constraint" and variant == 1:
        return ["attribute-01", "attribute-08", "bestseller-01", "bestseller-02", "price-range-01"], "coverage_weak"
    if query_type == "attribute_constraint":
        return ["attribute-03", "attribute-09", "bestseller-01", "bestseller-02", "price-range-01"], "coverage_weak"
    if query_type == "style":
        return ["attribute-07", "attribute-12", "bestseller-01", "bestseller-02", "attribute-06"], "not_relevant"
    if variant == 1:
        return ["attribute-07", "attribute-12", "bestseller-01", "bestseller-02", "attribute-01"], "not_relevant"
    return ["price-range-01", "attribute-07", "attribute-12", "bestseller-01", "bestseller-02"], "strong"


def _build_product_eval(facts: list[dict[str, Any]], specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [_product_document(fact) for fact in facts]
    fact_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        fact_by_category[str(fact["category"])].append(fact)
    all_facts = sorted(facts, key=lambda row: str(row["fact_id"]))
    cases = []
    for spec in sorted(specs, key=lambda row: str(row["category"])):
        category = str(spec["category"])
        category_facts = fact_by_category[category]
        for variant in range(1, 5):
            predicate, partial, query_en, query_zh, challenge_kind, target = _product_intent(spec, category_facts, variant)
            all_exact = [fact for fact in category_facts if predicate(fact)]
            if len(all_exact) < PRODUCT_POSITIVES:
                raise ValueError(f"{category} variant {variant} has only {len(all_exact)} exacts")
            exact = sorted(all_exact, key=lambda row: _stable_order(f"{category}-{variant}", str(row["fact_id"]), "positive"))[:PRODUCT_POSITIVES]
            excluded_exact = {str(row["fact_id"]) for row in all_exact}
            same_category = [row for row in category_facts if str(row["fact_id"]) not in excluded_exact]
            neighbors = [row for row in all_facts if row["category"] in _neighbors(category)]
            cross = [row for row in all_facts if row["category"] != category and row["category"] not in _neighbors(category)]
            intent_id = f"prv4-{_slug(category)}-{challenge_kind}-{variant:02d}"
            negatives = _negative_pool(intent_id, same_category, neighbors, cross, PRODUCT_POOL_SIZE - PRODUCT_POSITIVES)
            pool = [*exact, *negatives]
            exact_ids = {str(row["fact_id"]) for row in exact}
            candidate_ids = sorted(str(row["fact_id"]) for row in pool)
            pool_by_id = {str(row["fact_id"]): row for row in pool}
            labels = {
                document_id: ("Exact" if document_id in exact_ids else ("Substitute" if partial(pool_by_id[document_id]) else "Irrelevant"))
                for document_id in candidate_ids
            }
            for language, query in (("en", query_en), ("zh", query_zh)):
                cases.append({
                    "dataset_version": PRODUCT_EVAL_VERSION,
                    "intent_id": intent_id,
                    "query_id": f"{intent_id}-{language}",
                    "query": query,
                    "language": language,
                    "query_source_kind": "model_authored_independent_challenge" if language == "en" else "synthetic_translation_pair",
                    "category": category,
                    "split": "test",
                    "challenge_kind": challenge_kind,
                    "target_contract": target,
                    "candidate_ids": candidate_ids,
                    "relevance": {document_id: float(label == "Exact") for document_id, label in labels.items()},
                    "graded_relevance": {document_id: 1.0 if label == "Exact" else 0.01 if label == "Substitute" else 0.0 for document_id, label in labels.items()},
                    "labels": labels,
                })
    _validate_products(cases, documents)
    return sorted(documents, key=lambda row: str(row["document_id"])), sorted(cases, key=lambda row: (str(row["intent_id"]), str(row["language"])))


def _product_document(fact: dict[str, Any]) -> dict[str, Any]:
    attributes_en = "; ".join(f"{row['name']}: {row['value']}" for row in fact["text_supported_attributes"])
    attributes_zh = "；".join(f"{row['name_zh']}：{row['value_zh']}" for row in fact["text_supported_attributes"])
    price = float(fact["generated_fields"]["typical_price_cny"])
    return {
        "dataset_version": PRODUCT_EVAL_VERSION,
        "document_id": str(fact["fact_id"]),
        "title": f"{fact['title_en']} / {fact['title_zh']}",
        "body": f"Category: {fact['category']}. Form: {fact['retrieval_form_en']}. Attributes: {attributes_en}. Offline price: CNY {price:.0f}. 品类：{fact['category_zh']}。形态：{fact['retrieval_form_zh']}。属性：{attributes_zh}。离线价格：{price:.0f}元。",
        "category": fact["category"],
        "category_zh": fact["category_zh"],
        "source_kind": fact["source_kind"],
        "source_product_id": fact["product_id"],
    }


def _product_intent(
    spec: dict[str, Any], facts: list[dict[str, Any]], variant: int
) -> tuple[Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], bool], str, str, str, dict[str, Any]]:
    category = str(spec["category"])
    if variant in (1, 2):
        indexes = (0, 3) if variant == 1 else (1, 5)
        values = _most_common_pair(facts, spec, indexes)
        facets = [str(spec["facets"][index]["name"]) for index in indexes]
        zh_maps = [dict(spec["facets"][index]["values"]) for index in indexes]
        target = dict(zip(facets, values, strict=True))
        predicate = lambda fact: all(_attributes(fact).get(name) == value for name, value in target.items())
        partial = lambda fact: fact["category"] == category and sum(_attributes(fact).get(name) == value for name, value in target.items()) == 1
        if variant == 1:
            query_en = f"{category} with {values[0]}, also {values[1]}"
            query_zh = f"找{spec['category_zh']}，要{zh_maps[0][values[0]]}，同时要{zh_maps[1][values[1]]}"
            kind = "lexical_constraints"
        else:
            scenario_en, scenario_zh = SCENARIOS[category][0]
            query_en = f"for {scenario_en}, prioritize {values[0]} and lean toward {values[1]}"
            query_zh = f"为了{scenario_zh}，优先{zh_maps[0][values[0]]}，并倾向{zh_maps[1][values[1]]}"
            kind = "scenario_paraphrase"
        return predicate, partial, query_en, query_zh, kind, {"required_attributes": target}
    if variant == 3:
        facet = spec["facets"][2]
        counts = Counter(_attributes(fact).get(facet["name"]) for fact in facts)
        ranked = [value for value, _ in counts.most_common() if value is not None]
        desired, excluded = ranked[:2]
        zh = dict(facet["values"])
        predicate = lambda fact: fact["category"] == category and _attributes(fact).get(facet["name"]) == desired
        partial = lambda fact: fact["category"] == category and _attributes(fact).get(facet["name"]) == excluded
        query_en = f"need {category} with {desired}; exclude {excluded}"
        query_zh = f"需要{spec['category_zh']}，要{zh[desired]}，不要{zh[excluded]}"
        return predicate, partial, query_en, query_zh, "negative_constraint", {"required": {facet["name"]: desired}, "excluded": {facet["name"]: excluded}}
    form, use_value = _most_common_form_use(facts, spec)
    form_zh = dict(spec["forms"])[form]
    use_facet = spec["facets"][6]
    use_zh = dict(use_facet["values"])[use_value]
    predicate = lambda fact: fact["category"] == category and fact["retrieval_form_en"] == form and _attributes(fact).get(use_facet["name"]) == use_value
    partial = lambda fact: fact["category"] == category and (fact["retrieval_form_en"] == form or _attributes(fact).get(use_facet["name"]) == use_value)
    scenario_en, scenario_zh = SCENARIOS[category][1]
    return predicate, partial, f"what should I get to {scenario_en}? preferably a {form}", f"想{scenario_zh}，最好是{form_zh}，适合{use_zh}", "colloquial_scenario", {"form": form, "use": use_value}


def _most_common_pair(facts: list[dict[str, Any]], spec: dict[str, Any], indexes: tuple[int, int]) -> tuple[str, str]:
    names = [str(spec["facets"][index]["name"]) for index in indexes]
    pairs = Counter(
        tuple(_attributes(fact).get(name) for name in names)
        for fact in facts
        if all(_attributes(fact).get(name) is not None for name in names)
    )
    return tuple(pairs.most_common(1)[0][0])  # type: ignore[return-value]


def _most_common_form_use(facts: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[str, str]:
    use_name = str(spec["facets"][6]["name"])
    pairs = Counter(
        (str(fact["retrieval_form_en"]), _attributes(fact).get(use_name))
        for fact in facts
        if _attributes(fact).get(use_name) is not None
    )
    return pairs.most_common(1)[0][0]  # type: ignore[return-value]


def _negative_pool(intent_id: str, same: list[dict[str, Any]], neighbors: list[dict[str, Any]], cross: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for salt, rows, limit in (("same", same, 95), ("neighbor", neighbors, 220), ("cross", cross, count)):
        ordered = sorted(rows, key=lambda row: _stable_order(intent_id, str(row["fact_id"]), salt))
        for row in ordered:
            document_id = str(row["fact_id"])
            if document_id in used or len(selected) >= count:
                continue
            if salt != "cross" and sum(item["category"] == row["category"] for item in selected) >= limit:
                continue
            selected.append(row)
            used.add(document_id)
            if salt == "same" and len(selected) >= min(95, count):
                break
            if salt == "neighbor" and len(selected) >= min(95 + 220, count):
                break
    if len(selected) != count:
        raise ValueError(f"{intent_id} built only {len(selected)} negatives")
    return selected


def _neighbors(category: str) -> set[str]:
    output: set[str] = set()
    for group in NEIGHBOR_GROUPS:
        if category in group:
            output.update(group)
    output.discard(category)
    return output


def _validate_knowledge(cases: list[dict[str, Any]], cards: list[dict[str, Any]]) -> None:
    card_ids = sorted(str(card["card_id"]) for card in cards)
    if len(cases) != 320 or len({row["intent_id"] for row in cases}) != 160:
        raise ValueError("knowledge v4 requires 160 paired intents / 320 query rows")
    for case in cases:
        if case["candidate_ids"] != card_ids or set(case["judgments"]) != set(card_ids):
            raise ValueError(f"{case['query_id']} has unjudged cards")
        if len(case["relevant_card_ids"]) != KNOWLEDGE_POSITIVES:
            raise ValueError(f"{case['query_id']} must have five positives")
    _validate_pairs(cases, ("candidate_ids", "relevant_card_ids", "relevance", "graded_relevance", "judgments"))


def _validate_products(cases: list[dict[str, Any]], documents: list[dict[str, Any]]) -> None:
    document_ids = {str(row["document_id"]) for row in documents}
    if len(cases) != 160 or len({row["intent_id"] for row in cases}) != 80:
        raise ValueError("product v4 requires 80 paired intents / 160 query rows")
    for case in cases:
        candidates = set(case["candidate_ids"])
        if len(candidates) != PRODUCT_POOL_SIZE or not candidates.issubset(document_ids):
            raise ValueError(f"{case['query_id']} has an invalid pool")
        if candidates != set(case["labels"]) or candidates != set(case["relevance"]) or candidates != set(case["graded_relevance"]):
            raise ValueError(f"{case['query_id']} contains unjudged products")
        if sum(label == "Exact" for label in case["labels"].values()) != PRODUCT_POSITIVES:
            raise ValueError(f"{case['query_id']} must have five Exact products")
    _validate_pairs(cases, ("candidate_ids", "labels", "relevance", "graded_relevance", "target_contract"))


def _validate_pairs(cases: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        pairs[str(case["intent_id"])].append(case)
    for intent_id, pair in pairs.items():
        if {row["language"] for row in pair} != {"en", "zh"} or len(pair) != 2:
            raise ValueError(f"{intent_id} is not a bilingual pair")
        if any(pair[0][field] != pair[1][field] for field in fields):
            raise ValueError(f"{intent_id} has mismatched paired judgments")


def _card_manifest(facts: list[dict[str, Any]], cards: list[dict[str, Any]], paths: dict[str, Path], v3_facts_path: Path, specs_path: Path) -> dict[str, Any]:
    return {
        "dataset_version": CARD_VERSION,
        "role": "challenge corpus; v3 is retained as easy/smoke",
        "generation_seed": SEED,
        "truth_boundary": "real ESCI facts are retained; translations, prices, popularity and synthetic expansions are offline generated and unreviewed",
        "generation_policy": {"minimum_facts_per_category": FACTS_PER_CATEGORY, "minimum_new_challenge_facts_per_category": MIN_NEW_FACTS_PER_CATEGORY, "cards_per_category": {"bestseller": 2, "attribute_global": 7, "attribute_form_conditioned": 5, "price_range": 1}},
        "retrieval_text_policy": "category + type + own summary only; no all-forms expansion on every card",
        "counts": {"categories": len({row['category'] for row in facts}), "facts": len(facts), "facts_by_category": dict(sorted(Counter(row['category'] for row in facts).items())), "facts_by_source": dict(sorted(Counter(row['source_kind'] for row in facts).items())), "cards": len(cards), "cards_by_type": dict(sorted(Counter(row['card_type'] for row in cards).items()))},
        "input_sha256": {"v3_facts": _sha256(v3_facts_path), "generation_specs": _sha256(specs_path)},
        "output_sha256": {name: _sha256(paths[name]) for name in ("facts", "cards", "provenance", "retrieval")},
    }


def _knowledge_manifest(cards: list[dict[str, Any]], cases: list[dict[str, Any]], paths: dict[str, Path], specs_path: Path) -> dict[str, Any]:
    return {
        "dataset_version": KNOWLEDGE_EVAL_VERSION,
        "query_count": len(cases),
        "intent_count": len({row['intent_id'] for row in cases}),
        "card_count": len(cards),
        "category_count": 20,
        "positive_count_per_query": KNOWLEDGE_POSITIVES,
        "closed_pool": True,
        "unjudged_policy": "forbidden",
        "candidate_pool_policy": "global 300-card corpus; every card explicitly judged",
        "challenge_policy": "15 cards per category, Top-10 is smaller than the category card count, and style/colloquial queries may omit the canonical category name",
        "graded_gain_policy": "ordered relevant_card_ids receive gains 5,4,3,2,1",
        "language_counts": dict(sorted(Counter(row['language'] for row in cases).items())),
        "split_counts": {"final_test": len(cases)},
        "query_type_counts": dict(sorted(Counter(row['query_type'] for row in cases).items())),
        "input_sha256": {"cards": _sha256(paths['cards']), "generation_specs": _sha256(specs_path)},
        "cases_sha256": _sha256(paths["knowledge_cases"]),
    }


def _product_manifest(facts: list[dict[str, Any]], documents: list[dict[str, Any]], cases: list[dict[str, Any]], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "dataset_version": PRODUCT_EVAL_VERSION,
        "role": "synthetic bilingual challenge diagnostic; not a commercial benchmark",
        "closed_pool": True,
        "unjudged_policy": "forbidden",
        "unjudged_policy_detail": "every product in each 500-item pool is labeled",
        "query_count": len(cases),
        "intent_count": len({row['intent_id'] for row in cases}),
        "document_count": len(documents),
        "category_count": 20,
        "candidate_count_per_query": PRODUCT_POOL_SIZE,
        "exact_positive_count_per_query": PRODUCT_POSITIVES,
        "language_counts": dict(sorted(Counter(row['language'] for row in cases).items())),
        "challenge_kind_counts": dict(sorted(Counter(row['challenge_kind'] for row in cases).items())),
        "document_source_counts": dict(sorted(Counter(row['source_kind'] for row in facts).items())),
        "gain_mapping": {"Exact": 1.0, "Substitute": 0.01, "Irrelevant": 0.0},
        "binary_positive_policy": "Exact only",
        "pool_policy": "five Exact, up to 95 same-category hard negatives, then neighbor-category and cross-category negatives; additional exact matches are excluded",
        "truth_boundary": "queries, Chinese text, prices, and synthetic products are offline generated and unreviewed",
        "input_sha256": {"facts": _sha256(paths['facts'])},
        "output_sha256": {"items": _sha256(paths['product_items']), "cases": _sha256(paths['product_cases'])},
    }


def _attributes(fact: dict[str, Any]) -> dict[str, str]:
    return {str(row["name"]): str(row["value"]) for row in fact["text_supported_attributes"]}


def _provenance(card_id: str, category_facts: list[dict[str, Any]], source_rows: list[dict[str, Any]], summary_source: str) -> dict[str, Any]:
    return {"card_id": card_id, "category_item_count": len(category_facts), "source_item_ids": sorted({str(row['product_id']) for row in source_rows}), "source_kind_counts": dict(sorted(Counter(row['source_kind'] for row in source_rows).items())), "summary_source": summary_source, "contains_generated_evidence": True}


def _confidence(facts: list[dict[str, Any]]) -> float:
    real_ratio = sum(row["source_kind"] == "real_esci" for row in facts) / len(facts)
    return round(min(0.85, 0.58 + real_ratio * 0.2), 2)


def _bestseller_evidence(row: dict[str, Any]) -> str:
    price = float(row["generated_fields"]["typical_price_cny"])
    orders = int(row["generated_fields"]["order_count_90d"])
    suffix = f" | {price:.0f} | offline orders={orders}"
    title = " ".join(str(row["title_en"]).replace("|", "/").split())
    title = title[: max(1, 80 - len(suffix))]
    return f"{title}{suffix}"[:80]


def _truncate(value: str, limit: int = 80) -> str:
    clean = " ".join(value.replace("|", "/").split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _card_id(category: str, role: str) -> str:
    return f"cc-{_slug(category)}-{role}"


def _hashed_pick(values: list[Any], category: str, product_id: str, salt: str) -> Any:
    return values[int(_digest(category, product_id, salt)[:12], 16) % len(values)]


def _number(category: str, product_id: str, minimum: float, maximum: float, salt: str) -> float:
    ratio = int(_digest(category, product_id, salt)[:12], 16) / float(16**12 - 1)
    return round(minimum + (maximum - minimum) * ratio, 2)


def _digest(category: str, product_id: str, salt: str) -> str:
    return hashlib.sha256(f"{SEED}:{salt}:{category}:{product_id}".encode()).hexdigest()


def _stable_order(intent_id: str, document_id: str, salt: str) -> str:
    return hashlib.sha256(f"{intent_id}:{salt}:{document_id}".encode()).hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
