"""Build course-aligned Chinese CategoryCards from the Taobao CN catalog.

The source is the real ShopSimulator Taobao catalog. It provides Chinese
category paths, titles, source attributes, and observed listing prices. It
does not provide sales. Bestseller cards therefore use a reproducible offline
proxy based on shared attributes and attribute coverage; no order count or
transaction price is synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from globex_agent.category_insight import admit_card

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_TIME = "2026-08-18T00:00:00+08:00"
GENERATOR_VERSION = "category-card-taobao-zh-v3"
FACT_DATASET_VERSION = "category-item-facts-taobao-zh-v3"
CARD_DATASET_VERSION = "category-cards-taobao-zh-v3"
AUDIT_RATIO = 0.10
MIN_SHARED_ATTRIBUTES = 2
ATTRIBUTE_COVERAGE_DENOMINATOR = 10.0
PRICE_POLICY_VERSION = "log-iqr-1.5-positive-noncomparable-v2"
MIN_VALID_PRICE_CNY = 5.0
MAX_PRICE_TO_MEDIAN_RATIO = 30.0
NON_COMPARABLE_PRICE_TERMS = (
    "配件",
    "线材",
    "充电线",
    "车充线",
    "电源线",
    "租赁",
    "出租",
    "定制",
    "充电宝",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "catalogs" / "taobao" / "cn" / "items.jsonl",
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

    items = _read_jsonl(args.items)
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    _validate_taxonomy(taxonomy)

    facts, rejected_memberships = _build_item_facts(items, taxonomy)
    _apply_price_quality_policy(facts)
    raw_cards, provenance = _build_cards(facts, taxonomy)
    cards, rejected_cards = _admit_cards(raw_cards)
    audit_queue = _build_audit_queue(cards)
    retrieval_rows = _build_retrieval_rows(cards)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "facts": args.output_dir / "category_item_facts_taobao_zh.jsonl",
        "cards": args.output_dir / "category_cards_taobao_zh.jsonl",
        "provenance": args.output_dir / "category_card_provenance_taobao_zh.jsonl",
        "rejected_memberships": args.output_dir / "rejected_memberships_taobao_zh.jsonl",
        "rejected_cards": args.output_dir / "rejected_cards_taobao_zh.jsonl",
        "audit_queue": args.output_dir / "audit_queue_taobao_zh.jsonl",
        "retrieval": args.output_dir / "category_retrieval_texts_taobao_zh.jsonl",
        "manifest": args.output_dir / "category_card_manifest_taobao_zh.json",
    }
    accepted_ids = {card["card_id"] for card in cards}
    _write_jsonl(paths["facts"], facts)
    _write_jsonl(paths["cards"], cards)
    _write_jsonl(
        paths["provenance"],
        [row for row in provenance if row["card_id"] in accepted_ids],
    )
    _write_jsonl(paths["rejected_memberships"], rejected_memberships)
    _write_jsonl(paths["rejected_cards"], rejected_cards)
    _write_jsonl(paths["audit_queue"], audit_queue)
    _write_jsonl(paths["retrieval"], retrieval_rows)

    manifest = _build_manifest(
        items=items,
        taxonomy=taxonomy,
        facts=facts,
        cards=cards,
        rejected_memberships=rejected_memberships,
        rejected_cards=rejected_cards,
        audit_queue=audit_queue,
        input_paths={"items": args.items, "taxonomy": args.taxonomy},
        output_paths={key: value for key, value in paths.items() if key != "manifest"},
    )
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"built {len(cards)} cards from {len(facts)} category-item facts; "
        f"rejected cards={len(rejected_cards)}, audit queue={len(audit_queue)}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_taxonomy(taxonomy: dict[str, Any]) -> None:
    if taxonomy.get("source_dataset_version") != "shopsimulator-cn-items-v1":
        raise ValueError("taxonomy source_dataset_version does not match Taobao CN catalog")
    categories = taxonomy.get("categories")
    if not isinstance(categories, list) or len(categories) != 8:
        raise ValueError("taxonomy must define exactly eight ordinary categories")
    slugs = {str(category["slug"]) for category in categories}
    category_names = {str(category["category"]) for category in categories}
    leaf_names = {
        leaf
        for category in categories
        for leaf in category.get("source_leaf_names", [])
    }
    if len(slugs) != 8 or len(category_names) != 8 or len(leaf_names) != 8:
        raise ValueError("taxonomy categories, slugs, and source leaf names must be unique")
    default_kind = str(taxonomy.get("category_kind", "ordinary"))
    for category in categories:
        if str(category.get("category_kind", default_kind)) != "ordinary":
            raise ValueError("the Taobao Chinese v1 corpus only supports ordinary categories")
        if len(category.get("popular_forms", [])) != 6:
            raise ValueError(f"{category['category']} must define six popular forms")
        if len(category.get("attribute_rules", [])) != 3:
            raise ValueError(f"{category['category']} must define three attribute rules")


def _build_item_facts(
    items: list[dict[str, Any]],
    taxonomy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for category_config in taxonomy["categories"]:
        category = str(category_config["category"])
        leaf_names = set(category_config["source_leaf_names"])
        candidates = [
            item
            for item in items
            if item.get("category_path")
            and str(item["category_path"][-1]) in leaf_names
        ]
        accepted_candidates: list[tuple[dict[str, Any], str]] = []
        for item in sorted(candidates, key=lambda row: str(row["item_id"])):
            title = _clean_display_text(str(item.get("title") or ""))
            excluded = _first_matching_term(
                title,
                [str(term) for term in category_config.get("excluded_terms", [])],
            )
            if excluded is not None:
                rejected.append(
                    {
                        "category": category,
                        "item_id": str(item["item_id"]),
                        "reason": "title matches category excluded_terms gate",
                        "excluded_term": excluded,
                        "category_path": list(item.get("category_path") or []),
                    }
                )
                continue
            matched = _first_matching_term(title, category_config["required_terms"])
            if matched is None:
                rejected.append(
                    {
                        "category": category,
                        "item_id": str(item["item_id"]),
                        "reason": "title does not pass category required_terms gate",
                        "category_path": list(item.get("category_path") or []),
                    }
                )
                continue
            accepted_candidates.append((item, matched))

        shared_counts = _shared_attribute_counts(accepted_candidates)
        for item, matched in accepted_candidates:
            item_id = str(item["item_id"])
            source_attributes = [
                str(value)
                for value in (item.get("attributes", {}).get("source_attributes") or [])
            ]
            prices = _observed_prices(item)
            facts.append(
                {
                    "dataset_version": FACT_DATASET_VERSION,
                    "category": category,
                    "category_kind": "ordinary",
                    "item_id": item_id,
                    "source_record_id": _source_record_id(item),
                    "title": _clean_display_text(str(item.get("title") or "")),
                    "category_path": list(item.get("category_path") or []),
                    "source_attributes": source_attributes,
                    "source_tag": item.get("attributes", {}).get("source_tag"),
                    "price_observations_cny": prices,
                    "raw_price_observations_cny": list(prices),
                    "price_exclusions": [],
                    "price_measure": _price_measure(item),
                    "price_source": "observed",
                    "representative_price_cny": (
                        round(statistics.median(prices), 2) if prices else None
                    ),
                    "text_supported_attributes": _extract_attributes(
                        source_attributes, category_config["attribute_rules"]
                    ),
                    "category_assignment": {
                        "method": "taobao_leaf_category_path_plus_chinese_title_gate",
                        "matched_required_term": matched,
                        "source_leaf_name": str(item["category_path"][-1]),
                    },
                    "proxy_fields": {
                        "homogeneous_item_count": shared_counts[item_id],
                        "attribute_coverage": round(
                            min(1.0, len(source_attributes) / ATTRIBUTE_COVERAGE_DENOMINATOR),
                            2,
                        ),
                        "bestseller_proxy_source": "generated_offline_proxy",
                    },
                    "source": {
                        "kind": "external_public",
                        "dataset": "ShopSimulator Taobao CN",
                        "dataset_version": taxonomy["source_dataset_version"],
                        "catalog_path": taxonomy["source_catalog_path"],
                    },
                }
            )
    return facts, rejected


def _apply_price_quality_policy(facts: list[dict[str, Any]]) -> None:
    """Remove invalid/extreme listing observations with auditable reasons."""

    facts_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        facts_by_category[str(fact["category"])].append(fact)

    for category_facts in facts_by_category.values():
        for fact in category_facts:
            raw = list(fact.get("raw_price_observations_cny") or [])
            matched_non_comparable_term = _first_matching_term(
                str(fact.get("title", "")), NON_COMPARABLE_PRICE_TERMS
            )
            valid = [
                price
                for price in raw
                if (
                    matched_non_comparable_term is None
                    and math.isfinite(price)
                    and price >= MIN_VALID_PRICE_CNY
                )
            ]
            fact["price_observations_cny"] = valid
            fact["price_exclusions"] = []
            if matched_non_comparable_term is not None:
                fact["price_exclusions"].extend(
                    {
                        "price_cny": price,
                        "reason": "non_comparable_listing_type",
                        "matched_term": matched_non_comparable_term,
                    }
                    for price in raw
                )
            else:
                fact["price_exclusions"].extend(
                    {
                        "price_cny": price,
                        "reason": "below_minimum_or_non_finite",
                    }
                    for price in raw
                    if not math.isfinite(price) or price < MIN_VALID_PRICE_CNY
                )

        all_prices = sorted(
            price for fact in category_facts for price in fact["price_observations_cny"]
        )
        if not all_prices:
            continue
        median = statistics.median(all_prices)
        if len(all_prices) < 4:
            lower = MIN_VALID_PRICE_CNY
            upper = median * MAX_PRICE_TO_MEDIAN_RATIO
        else:
            log_prices = sorted(math.log(price) for price in all_prices)
            q1 = _percentile(log_prices, 0.25)
            q3 = _percentile(log_prices, 0.75)
            iqr = q3 - q1
            lower = max(MIN_VALID_PRICE_CNY, math.exp(q1 - 1.5 * iqr))
            upper = min(math.exp(q3 + 1.5 * iqr), median * MAX_PRICE_TO_MEDIAN_RATIO)
        for fact in category_facts:
            kept: list[float] = []
            for price in fact["price_observations_cny"]:
                if price < lower or price > upper:
                    fact["price_exclusions"].append(
                        {
                            "price_cny": price,
                            "reason": "outside_category_robust_fence",
                            "lower_fence_cny": round(lower, 2),
                            "upper_fence_cny": round(upper, 2),
                        }
                    )
                else:
                    kept.append(price)
            fact["price_observations_cny"] = kept
            fact["representative_price_cny"] = (
                round(statistics.median(kept), 2) if kept else None
            )
            fact["price_quality"] = {
                "policy_version": PRICE_POLICY_VERSION,
                "minimum_valid_price_cny": MIN_VALID_PRICE_CNY,
                "max_price_to_median_ratio": MAX_PRICE_TO_MEDIAN_RATIO,
                "category_median_cny": round(median, 2),
                "lower_fence_cny": round(lower, 2),
                "upper_fence_cny": round(upper, 2),
                "raw_count": len(fact.get("raw_price_observations_cny") or []),
                "kept_count": len(kept),
                "excluded_count": len(fact["price_exclusions"]),
            }


def _shared_attribute_counts(
    candidates: list[tuple[dict[str, Any], str]],
) -> dict[str, int]:
    by_id: dict[str, set[str]] = {
        str(item["item_id"]): set(
            str(value)
            for value in (item.get("attributes", {}).get("source_attributes") or [])
        )
        for item, _ in candidates
    }
    output: dict[str, int] = {}
    for item, _ in candidates:
        item_id = str(item["item_id"])
        source_attributes = by_id[item_id]
        output[item_id] = sum(
            1
            for other_id, other_attributes in by_id.items()
            if other_id != item_id
            and len(source_attributes & other_attributes) >= MIN_SHARED_ATTRIBUTES
        )
    return output


def _observed_prices(item: dict[str, Any]) -> list[float]:
    if item.get("price_cny") is not None:
        price = _to_float(item["price_cny"])
        return [price] if price is not None else []
    prices: list[float] = []
    for variant in item.get("variants") or []:
        price = _to_float(variant.get("price_cny"))
        if price is not None:
            prices.append(price)
    return sorted(prices)


def _price_measure(item: dict[str, Any]) -> str:
    if item.get("price_cny") is not None:
        return "single_listed_price_cny"
    if item.get("variants"):
        return "variant_listed_prices_cny"
    return "unavailable"


def _source_record_id(item: dict[str, Any]) -> str:
    provenance = item.get("provenance") or {}
    value = provenance.get("source_record_id")
    return str(value) if value is not None else str(item.get("item_id"))


def _to_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _extract_attributes(
    source_attributes: list[str],
    rules: list[dict[str, Any]],
) -> list[dict[str, str]]:
    joined = _normalized_text(" ".join(source_attributes))
    output: list[dict[str, str]] = []
    for rule in rules:
        for value, terms in rule["values"].items():
            matched = _first_matching_term(joined, terms)
            if matched is not None:
                output.append({"name": rule["name"], "value": value, "matched_term": matched})
                break
    return output


def _build_cards(
    facts: list[dict[str, Any]],
    taxonomy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        facts_by_category[fact["category"]].append(fact)

    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for category_config in taxonomy["categories"]:
        category = str(category_config["category"])
        category_facts = facts_by_category[category]
        if len(category_facts) < 3:
            raise ValueError(f"{category} has fewer than three accepted item facts")

        generated_cards, generated_provenance = _build_bestseller_cards(
            category_config, category_facts
        )
        cards.extend(generated_cards)
        provenance.extend(generated_provenance)

        generated_cards, generated_provenance = _build_attribute_cards(
            category_config, category_facts
        )
        cards.extend(generated_cards)
        provenance.extend(generated_provenance)

        price_card, price_provenance = _build_price_card(category_config, category_facts)
        cards.append(price_card)
        provenance.append(price_provenance)

    expected = len(taxonomy["categories"]) * 6
    if len(cards) != expected:
        raise ValueError(f"expected {expected} raw cards, generated {len(cards)}")
    return cards, provenance


def _build_bestseller_cards(
    category_config: dict[str, Any],
    facts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category = str(category_config["category"])
    slug = str(category_config["slug"])
    ranked = sorted(
        [fact for fact in facts if fact.get("representative_price_cny") is not None],
        key=lambda row: (
            -row["proxy_fields"]["homogeneous_item_count"],
            -row["proxy_fields"]["attribute_coverage"],
            str(row["item_id"]),
        ),
    )[:6]
    if len(ranked) < 6:
        raise ValueError(f"{category} has fewer than six bestseller evidence facts")
    proxy_coverage = statistics.fmean(
        row["proxy_fields"]["attribute_coverage"] for row in ranked
    )
    confidence = round(min(0.85, 0.55 + len(facts) / 1000 + proxy_coverage * 0.15), 2)
    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, 6, 3), start=1):
        batch = ranked[start : start + 3]
        forms = category_config["popular_forms"][start : start + 3]
        card_id = f"cc-{slug}-bestseller-{index:02d}"
        summary = f"{category}：{' / '.join(forms)}"
        cards.append(
            {
                "card_id": card_id,
                "category": category,
                "card_type": "bestseller",
                "summary": summary,
                "raw_evidence": [_bestseller_evidence(row) for row in batch],
                "last_updated": SNAPSHOT_TIME,
                "confidence": confidence,
            }
        )
        provenance.append(
            {
                "card_id": card_id,
                "category_kind": "ordinary",
                "source_item_ids": [str(row["item_id"]) for row in batch],
                "category_item_count": len(facts),
                "summary_source": "human_maintained_taobao_popular_forms",
                "evidence_source": (
                    "generated_offline_proxy_homogeneous_items_and_attribute_coverage"
                ),
                "proxy_rank_fields": ["homogeneous_item_count", "attribute_coverage"],
                "proxy_note": "淘宝目录无真实销量，未生成销量",
                "proxy_scope": "目录高频款型代理/高频形态，不是真实销量榜",
                "price_measure": "median_observed_item_price_cny",
                "observed_source_fields": [
                    "item_id",
                    "title",
                    "category_path",
                    "source_attributes",
                    "price_cny",
                    "variants[].price_cny",
                ],
                "generated_fields": ["bestseller_proxy_rank"],
                "generator_version": GENERATOR_VERSION,
            }
        )
    return cards, provenance


def _build_attribute_cards(
    category_config: dict[str, Any],
    facts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_attribute: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fact in facts:
        for attribute in fact["text_supported_attributes"]:
            by_attribute[attribute["name"]][attribute["value"]].append(fact)

    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    slug = str(category_config["slug"])
    for index, rule in enumerate(category_config["attribute_rules"], start=1):
        attribute_name = str(rule["name"])
        values = by_attribute.get(attribute_name, {})
        valid_count = sum(len(rows) for rows in values.values())
        if valid_count < 2:
            raise ValueError(
                f"{category_config['category']} attribute {attribute_name} "
                f"has only {valid_count} valid facts"
            )
        ranked = sorted(values.items(), key=lambda item: (-len(item[1]), item[0]))
        selected = ranked[:2]
        other_rows = [row for _, rows in ranked[2:] for row in rows]
        if other_rows:
            selected = [*selected, ("其他", other_rows)]
        tokens = [
            f"{value} {len(rows) / valid_count * 100:.1f}%"
            for value, rows in selected
        ]
        card_id = f"cc-{slug}-attribute-{index:02d}"
        evidence = [
            _truncate_evidence(f"{value}: {rows[0]['title']}")
            for value, rows in selected
            if rows
        ]
        coverage = valid_count / len(facts)
        cards.append(
            {
                "card_id": card_id,
                "category": str(category_config["category"]),
                "card_type": "attribute",
                "summary": f"{attribute_name}：{' / '.join(tokens)}",
                "raw_evidence": evidence,
                "last_updated": SNAPSHOT_TIME,
                "confidence": round(min(0.9, 0.6 + coverage * 0.3), 2),
            }
        )
        provenance.append(
            {
                "card_id": card_id,
                "category_kind": "ordinary",
                "source_item_ids": sorted(
                    {
                        str(row["item_id"])
                        for _, rows in selected
                        for row in rows
                    }
                ),
                "category_item_count": len(facts),
                "attribute_name": attribute_name,
                "valid_sample_count": valid_count,
                "distribution_counts": {
                    value: len(rows) for value, rows in selected
                },
                "summary_source": "deterministic_taobao_source_attributes_aggregation",
                "claim_scope": (
                    "当前目录样本中的标题/属性出现率；不是市场规律；"
                    "医疗或功效词仅为商品标题声明，未验证功效"
                ),
                "generated_fields": [],
                "generator_version": GENERATOR_VERSION,
            }
        )
    return cards, provenance


def _build_price_card(
    category_config: dict[str, Any],
    facts: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    prices = sorted(
        price
        for fact in facts
        for price in fact["price_observations_cny"]
    )
    if len(prices) < 2:
        raise ValueError(f"{category_config['category']} has fewer than two price observations")
    minimum, first, second, maximum = _price_edges(prices)
    category = str(category_config["category"])
    slug = str(category_config["slug"])
    card_id = f"cc-{slug}-price-range-01"
    single_count = sum(fact["price_measure"] == "single_listed_price_cny" for fact in facts)
    multi_count = sum(fact["price_measure"] == "variant_listed_prices_cny" for fact in facts)
    card = {
        "card_id": card_id,
        "category": category,
        "card_type": "price_range",
        "summary": (
            f"便宜款 {minimum:.0f}-{first:.0f} / "
            f"中档 {first:.0f}-{second:.0f} / "
            f"高端 {second:.0f}-{maximum:.0f}"
        ),
        "raw_evidence": [
            f"观察价格样本 n={len(prices)}",
            f"单规格 {single_count} 件，多规格 {multi_count} 件",
            "目录挂牌/规格价参考区间，不表示成交价、具体 SKU 价格或实时价格",
        ],
        "last_updated": SNAPSHOT_TIME,
        "confidence": round(min(0.9, 0.6 + len(prices) / 2000 * 0.3), 2),
    }
    provenance = {
        "card_id": card_id,
        "category_kind": "ordinary",
        "source_item_ids": [str(fact["item_id"]) for fact in facts],
        "category_item_count": len(facts),
        "sample_count": len(prices),
        "single_price_item_count": single_count,
        "multi_price_item_count": multi_count,
        "price_measure": "observed_listed_prices_cny",
        "currency": "CNY",
        "percentile_method": "linear_interpolation",
        "quality_policy_version": PRICE_POLICY_VERSION,
        "quality_policy": (
            "先排除低于 5 CNY 或非有限价格，再按品类价格的 log-space IQR 1.5 "
            "fence 与 30 倍品类中位数上限排除明显极端值；不把异常值用于价格档位"
        ),
        "excluded_observation_count": sum(
            len(fact.get("price_exclusions") or []) for fact in facts
        ),
        "listed_price_not_transaction": True,
        "price_scope": "目录挂牌/规格价样本的品类参考区间，不是成交价、具体 SKU 价格或实时价格",
        "observed_source_fields": ["price_cny", "variants[].price_cny"],
        "generated_fields": [],
        "generator_version": GENERATOR_VERSION,
    }
    return card, provenance


def _price_edges(prices: list[float]) -> tuple[int, int, int, int]:
    unit = 10
    minimum = max(
        int(MIN_VALID_PRICE_CNY),
        math.floor(prices[0] / unit) * unit,
    )
    first = math.ceil(_percentile(prices, 1 / 3) / unit) * unit
    second = math.ceil(_percentile(prices, 2 / 3) / unit) * unit
    maximum = math.ceil(prices[-1] / unit) * unit
    while first <= minimum:
        first += unit
    while second <= first:
        second += unit
    while maximum <= second:
        maximum += unit
    return minimum, first, second, maximum


def _bestseller_evidence(fact: dict[str, Any]) -> str:
    price = float(fact["representative_price_cny"])
    proxy = fact["proxy_fields"]
    suffix = (
        f" | {price:.0f} | 同质商品数={proxy['homogeneous_item_count']}，"
        f"属性覆盖={proxy['attribute_coverage']:.2f}"
    )
    title = _clean_display_text(str(fact["title"])).replace("|", "/")
    return _fit_title_with_suffix(title, suffix)


def _fit_title_with_suffix(title: str, suffix: str) -> str:
    title_limit = max(1, 80 - len(suffix))
    if len(title) > title_limit:
        title = title[: max(1, title_limit - 1)].rstrip() + "…"
    return title + suffix


def _truncate_evidence(value: str) -> str:
    clean = " ".join(str(value).replace("|", "/").split())
    return clean if len(clean) <= 80 else clean[:79].rstrip() + "…"


def _admit_cards(
    raw_cards: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in raw_cards:
        result = admit_card(raw)
        if result.accepted and result.card is not None:
            accepted.append(result.card.model_dump(mode="json"))
        else:
            rejected.append({"card": raw, "reason": result.reason})
    return accepted, rejected


def _build_audit_queue(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_size = math.ceil(len(cards) * AUDIT_RATIO)
    ranked = sorted(
        cards,
        key=lambda card: _sha256_text(f"{GENERATOR_VERSION}|audit|{card['card_id']}"),
    )[:sample_size]
    return [
        {
            "card_id": card["card_id"],
            "status": "pending_human_review",
            "sampling_method": "deterministic_hash_sample",
            "audit_ratio": AUDIT_RATIO,
        }
        for card in ranked
    ]


def _build_retrieval_rows(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    type_zh = {
        "bestseller": "目录高频款型代理",
        "attribute": "目录样本属性出现率",
        "price_range": "品类参考价格档位",
    }
    rows = []
    for card in cards:
        rows.append(
            {
                "card_id": card["card_id"],
                "retrieval_text_zh": (
                    f"品类：{card['category']}。"
                    f"知识类型：{type_zh[card['card_type']]}。"
                    f"摘要：{card['summary']}"
                ),
            }
        )
    return rows


def _build_manifest(
    *,
    items: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    facts: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    rejected_memberships: list[dict[str, Any]],
    rejected_cards: list[dict[str, Any]],
    audit_queue: list[dict[str, Any]],
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    card_counts = Counter(card["card_type"] for card in cards)
    category_counts = Counter(card["category"] for card in cards)
    fact_counts = Counter(fact["category"] for fact in facts)
    return {
        "dataset_version": CARD_DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": SNAPSHOT_TIME,
        "intended_use": "offline course pipeline and Chinese knowledge-card evaluation only",
        "source_dataset": {
            "name": "ShopSimulator Taobao CN",
            "version": taxonomy["source_dataset_version"],
            "catalog_path": taxonomy["source_catalog_path"],
            "item_count": len(items),
        },
        "truth_boundary": {
            "observed": [
                "item_id",
                "title",
                "category_path",
                "source_attributes",
                "price_cny",
                "variants[].price_cny",
            ],
            "generated": ["bestseller_proxy_rank"],
            "bestseller_semantics": (
                "catalog frequency/form proxy based on homogeneous items and "
                "attribute coverage; not a real sales ranking"
            ),
            "attribute_semantics": (
                "current catalog title/attribute occurrence rates; not market-wide "
                "distributions; medical/efficacy terms are unverified title claims"
            ),
            "price_semantics": (
                "robust-range reference from observed listing/SKU prices; not historical "
                "transaction prices, a concrete SKU price, or real-time price"
            ),
        },
        "category_assignment": {
            "method": "taobao_leaf_category_path_plus_chinese_title_gate",
            "assigned_category_count": len(taxonomy["categories"]),
        },
        "generation_rules": {
            "price": (
                "real price_cny or variants[].price_cny; positive finite observations "
                "then log-space IQR 1.5 outlier fence before linear percentile edges"
            ),
            "price_quality_policy": PRICE_POLICY_VERSION,
            "price_policy": {
                "version": PRICE_POLICY_VERSION,
                "minimum_valid_price_cny": MIN_VALID_PRICE_CNY,
                "max_price_to_median_ratio": MAX_PRICE_TO_MEDIAN_RATIO,
                "outlier_rule": "category-local log-space IQR 1.5 fence",
                "excluded_observation_reasons": [
                    "below_minimum_or_non_finite",
                    "non_comparable_listing_type",
                    "outside_category_robust_fence",
                ],
                "non_comparable_title_terms": list(NON_COMPARABLE_PRICE_TERMS),
            },
            "bestseller_proxy": (
                "shared source attributes >= 2 plus normalized attribute coverage; "
                "displayed as catalog frequency/form proxy"
            ),
            "attributes": (
                "deterministic source_attributes aggregation from taxonomy rules; "
                "percentages are current catalog occurrence rates"
            ),
            "summary": "deterministic formatting; no generative LLM",
        },
        "admission": {
            "minimum_confidence": 0.5,
            "maximum_summary_chars": 200,
            "raw_evidence_count": [1, 3],
            "maximum_raw_evidence_chars": 80,
            "audit_ratio": AUDIT_RATIO,
        },
        "counts": {
            "category_item_facts": len(facts),
            "unique_fact_items": len({fact["item_id"] for fact in facts}),
            "rejected_memberships": len(rejected_memberships),
            "accepted_cards": len(cards),
            "rejected_cards": len(rejected_cards),
            "audit_queue": len(audit_queue),
            "cards_by_type": dict(sorted(card_counts.items())),
            "cards_by_category": dict(sorted(category_counts.items())),
            "facts_by_category": dict(sorted(fact_counts.items())),
        },
        "input_sha256": {key: _sha256_file(path) for key, path in input_paths.items()},
        "output_sha256": {
            key: _sha256_file(path) for key, path in output_paths.items()
        },
    }


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _first_matching_term(text: str, terms: list[str]) -> str | None:
    normalized = _normalized_text(text)
    for term in terms:
        if _normalized_text(term) in normalized:
            return term
    return None


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _clean_display_text(value: str) -> str:
    return " ".join(value.replace("\ufffd", "").split())


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
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
