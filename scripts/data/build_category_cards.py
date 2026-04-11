"""Build traceable CategoryInsight cards from the checked-in ESCI item subset.

The ESCI source provides real product text and relevance judgments, but no
prices or sales. This offline learning dataset therefore keeps source-derived
fields separate from deterministic generated CNY prices and 90-day order
counts. Generated values are suitable for exercising the course pipeline only;
they are not Amazon observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from globex_agent.category_insight import admit_card

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "esci-category-card-builder-v1"
FACT_DATASET_VERSION = "esci-category-item-facts-v1"
CARD_DATASET_VERSION = "esci-category-cards-v1"
SNAPSHOT_TIME = "2026-08-15T00:00:00+08:00"
GENERATION_SEED = 20260815
AUDIT_RATIO = 0.10
LABEL_WEIGHT = {"Exact": 1.0, "Substitute": 0.7, "Complement": 0.4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "recall_items.jsonl",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "recall_cases.jsonl",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "category_taxonomy.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "category_insight",
    )
    args = parser.parse_args()

    items = {row["document_id"]: row for row in _read_jsonl(args.items)}
    cases = {row["query_id"]: row for row in _read_jsonl(args.cases)}
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    _validate_inputs(items, cases, taxonomy)

    facts, rejected_memberships = _build_item_facts(items, cases, taxonomy)
    raw_cards, provenance = _build_cards(facts, taxonomy)
    cards, rejected_cards = _admit_cards(raw_cards)
    audit_queue = _build_audit_queue(cards)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "facts": args.output_dir / "category_item_facts.jsonl",
        "cards": args.output_dir / "category_cards.jsonl",
        "provenance": args.output_dir / "category_card_provenance.jsonl",
        "rejected_memberships": args.output_dir / "rejected_memberships.jsonl",
        "rejected_cards": args.output_dir / "rejected_cards.jsonl",
        "audit_queue": args.output_dir / "audit_queue.jsonl",
        "manifest": args.output_dir / "category_card_manifest.json",
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

    manifest = _build_manifest(
        items=items,
        cases=cases,
        taxonomy=taxonomy,
        facts=facts,
        cards=cards,
        rejected_memberships=rejected_memberships,
        rejected_cards=rejected_cards,
        audit_queue=audit_queue,
        input_paths={"items": args.items, "cases": args.cases, "taxonomy": args.taxonomy},
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


def _validate_inputs(
    items: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    taxonomy: dict[str, Any],
) -> None:
    if taxonomy.get("source_dataset_version") != "amazon-esci-task1-en-us-closed-recall-v2":
        raise ValueError("taxonomy source_dataset_version does not match the ESCI subset")
    seen_query_ids: set[str] = set()
    for category in taxonomy["categories"]:
        query_ids = set(category["source_query_ids"])
        overlap = query_ids & seen_query_ids
        if overlap:
            raise ValueError(f"query IDs assigned to multiple categories: {sorted(overlap)}")
        missing = query_ids - cases.keys()
        if missing:
            raise ValueError(f"taxonomy references missing query IDs: {sorted(missing)}")
        seen_query_ids.update(query_ids)
    excluded = set(taxonomy.get("excluded_query_groups", {}))
    if seen_query_ids & excluded:
        raise ValueError("a query ID cannot be both assigned and excluded")
    if seen_query_ids | excluded != cases.keys():
        missing = cases.keys() - seen_query_ids - excluded
        raise ValueError(f"taxonomy leaves query IDs undecided: {sorted(missing)}")
    referenced = {
        product_id
        for case in cases.values()
        for product_id in case["candidate_ids"]
    }
    if not referenced <= items.keys():
        raise ValueError("one or more case candidates are absent from recall_items")


def _build_item_facts(
    items: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    taxonomy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed_labels = set(taxonomy["membership_labels"])
    facts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for category_config in taxonomy["categories"]:
        judgments_by_item: dict[str, list[dict[str, str]]] = defaultdict(list)
        for query_id in category_config["source_query_ids"]:
            case = cases[query_id]
            for product_id, label in case["labels"].items():
                if label not in allowed_labels:
                    continue
                judgments_by_item[product_id].append(
                    {"query_id": query_id, "query": case["query"], "label": label}
                )

        for product_id, judgments in sorted(judgments_by_item.items()):
            item = items[product_id]
            title_text = _normalized_text(item["title"])
            text = _normalized_text(f"{item['title']} {item['body']}")
            matched_gate = _first_matching_term(
                title_text, category_config["required_terms"]
            )
            if matched_gate is None:
                rejected.append(
                    {
                        "category": category_config["category"],
                        "product_id": product_id,
                        "reason": "title does not pass category required_terms gate",
                        "source_judgments": judgments,
                    }
                )
                continue
            matched_exclusion = _first_matching_term(
                title_text, category_config.get("excluded_terms", [])
            )
            if matched_exclusion is not None:
                rejected.append(
                    {
                        "category": category_config["category"],
                        "product_id": product_id,
                        "reason": "title matches category excluded_terms gate",
                        "matched_excluded_term": matched_exclusion,
                        "source_judgments": judgments,
                    }
                )
                continue
            strongest_label = max(judgments, key=lambda row: LABEL_WEIGHT[row["label"]])[
                "label"
            ]
            attributes = _extract_attributes(text, category_config["attribute_rules"])
            price_cny = _generated_price(
                category_config["category"],
                product_id,
                category_config["generated_price_cny_bounds"],
            )
            order_count = _generated_order_count(
                category_config["category"], product_id, strongest_label
            )
            facts.append(
                {
                    "dataset_version": FACT_DATASET_VERSION,
                    "category": category_config["category"],
                    "category_kind": category_config["category_kind"],
                    "product_id": product_id,
                    "title": _clean_display_text(item["title"]),
                    "source_body_sha256": _sha256_text(item["body"]),
                    "category_assignment": {
                        "method": (
                            "human_query_map_plus_product_title_include_exclude_gate"
                        ),
                        "matched_required_term": matched_gate,
                        "strongest_esci_label": strongest_label,
                        "source_judgments": judgments,
                    },
                    "text_supported_attributes": attributes,
                    "generated_fields": {
                        "typical_price_cny": price_cny,
                        "order_count_90d": order_count,
                        "price_source": "generated_offline_not_observed",
                        "popularity_source": "generated_offline_not_observed",
                        "seed": GENERATION_SEED,
                        "rule_version": GENERATOR_VERSION,
                    },
                    "source": {
                        "kind": "external_public",
                        "dataset": "Amazon Shopping Queries ESCI",
                        "dataset_version": taxonomy["source_dataset_version"],
                        "license": "Apache-2.0",
                    },
                }
            )
    return facts, rejected


def _extract_attributes(
    text: str, attribute_rules: list[dict[str, Any]]
) -> list[dict[str, str]]:
    attributes: list[dict[str, str]] = []
    for rule in attribute_rules:
        for value, terms in rule["values"].items():
            matched = _first_matching_term(text, terms)
            if matched is not None:
                attributes.append(
                    {"name": rule["name"], "value": value, "matched_term": matched}
                )
                break
    return attributes


def _build_cards(
    facts: list[dict[str, Any]], taxonomy: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        facts_by_category[fact["category"]].append(fact)

    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for category_config in taxonomy["categories"]:
        category = category_config["category"]
        category_facts = facts_by_category[category]
        if len(category_facts) < 3:
            continue
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
    return cards, provenance


def _build_bestseller_cards(
    category_config: dict[str, Any], facts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category = category_config["category"]
    slug = _slug(category)
    ranked = sorted(
        facts,
        key=lambda row: (-row["generated_fields"]["order_count_90d"], row["product_id"]),
    )[:6]
    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(ranked), 3), start=1):
        batch = ranked[start : start + 3]
        if not batch:
            continue
        card_id = f"cc-{slug}-bestseller-{index:02d}"
        popular_forms = category_config["popular_forms"]
        summary_forms = popular_forms[:2] if index == 1 else popular_forms[1:]
        summary = f"{category}：{' / '.join(summary_forms)}"
        evidence = [_bestseller_evidence(row) for row in batch]
        cards.append(
            {
                "card_id": card_id,
                "category": category,
                "card_type": "bestseller",
                "summary": summary,
                "raw_evidence": evidence,
                "last_updated": SNAPSHOT_TIME,
                "confidence": round(0.55 + min(len(facts), 50) / 500, 2),
            }
        )
        provenance.append(
            {
                "card_id": card_id,
                "category_kind": category_config["category_kind"],
                "source_item_ids": [row["product_id"] for row in batch],
                "source_query_ids": category_config["source_query_ids"],
                "summary_source": "human_maintained_taxonomy.popular_forms",
                "evidence_source": "generated_offline_order_count_and_price",
                "observed_source_fields": ["product_id", "title", "ESCI label"],
                "generated_fields": ["typical_price_cny", "order_count_90d"],
                "not_real_world_bestseller": True,
                "generator_version": GENERATOR_VERSION,
            }
        )
    return cards, provenance


def _build_attribute_cards(
    category_config: dict[str, Any], facts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_attribute: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fact in facts:
        for attribute in fact["text_supported_attributes"]:
            by_attribute[attribute["name"]][attribute["value"]].append(fact)

    cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    slug = _slug(category_config["category"])
    for index, rule in enumerate(category_config["attribute_rules"], start=1):
        attribute_name = rule["name"]
        values = by_attribute.get(attribute_name, {})
        valid_count = sum(len(rows) for rows in values.values())
        if valid_count < 2:
            continue
        all_ranked_values = sorted(
            values.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(all_ranked_values) > 3:
            other_rows = [row for _, rows in all_ranked_values[2:] for row in rows]
            ranked_values = [*all_ranked_values[:2], ("other", other_rows)]
        else:
            ranked_values = all_ranked_values
        tokens = [
            f"{value} {len(rows) / valid_count * 100:.1f}%"
            for value, rows in ranked_values
        ]
        card_id = f"cc-{slug}-attribute-{index:02d}"
        evidence = [
            _truncate_evidence(
                f"{value}: {rows[0]['product_id']} {_clean_display_text(rows[0]['title'])}"
            )
            for value, rows in ranked_values
        ]
        coverage = valid_count / len(facts)
        cards.append(
            {
                "card_id": card_id,
                "category": category_config["category"],
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
                "category_kind": category_config["category_kind"],
                "source_item_ids": sorted(
                    {row["product_id"] for _, rows in ranked_values for row in rows}
                ),
                "source_query_ids": category_config["source_query_ids"],
                "attribute_name": attribute_name,
                "valid_sample_count": valid_count,
                "category_item_count": len(facts),
                "distribution_counts": {
                    value: len(rows) for value, rows in ranked_values
                },
                "summary_source": "deterministic_product_text_aggregation",
                "generated_fields": [],
                "generator_version": GENERATOR_VERSION,
            }
        )
    return cards, provenance


def _build_price_card(
    category_config: dict[str, Any], facts: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    prices = sorted(float(row["generated_fields"]["typical_price_cny"]) for row in facts)
    minimum = _round_down(prices[0], 10)
    first = _round_up(_percentile(prices, 1 / 3), 10)
    second = _round_up(_percentile(prices, 2 / 3), 10)
    maximum = _round_up(prices[-1], 10)
    first = max(first, minimum + 10)
    second = max(second, first + 10)
    maximum = max(maximum, second + 10)
    category = category_config["category"]
    card_id = f"cc-{_slug(category)}-price-range-01"
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
            f"generated CNY prices: n={len(prices)}, seed={GENERATION_SEED}",
            "observed ESCI fields contain no price; "
            f"bounds={category_config['generated_price_cny_bounds']}",
        ],
        "last_updated": SNAPSHOT_TIME,
        "confidence": 0.55,
    }
    provenance = {
        "card_id": card_id,
        "category_kind": category_config["category_kind"],
        "source_item_ids": [row["product_id"] for row in facts],
        "source_query_ids": category_config["source_query_ids"],
        "sample_count": len(prices),
        "price_measure": "generated_typical_price_cny",
        "currency": "CNY",
        "percentile_method": "linear_interpolation",
        "generated_bounds": category_config["generated_price_cny_bounds"],
        "not_observed_transaction_price": True,
        "generator_version": GENERATOR_VERSION,
    }
    return card, provenance


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
        key=lambda card: _sha256_text(
            f"{GENERATION_SEED}|audit|{card['card_id']}"
        ),
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


def _build_manifest(
    *,
    items: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
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
        "generation_seed": GENERATION_SEED,
        "intended_use": "offline course pipeline and retrieval evaluation only",
        "source_dataset": {
            "name": "Amazon Shopping Queries ESCI",
            "version": taxonomy["source_dataset_version"],
            "license": "Apache-2.0",
            "product_count": len(items),
            "query_count": len(cases),
        },
        "truth_boundary": {
            "observed": ["product_id", "title", "body", "ESCI relevance label"],
            "generated": ["typical_price_cny", "order_count_90d"],
            "bestseller_semantics": "generated offline popularity, not real sales ranking",
            "price_semantics": "generated typical CNY price, not listing or transaction price",
        },
        "category_assignment": {
            "method": (
                "human query-to-category map plus product title include/exclude gate"
            ),
            "assigned_category_count": len(taxonomy["categories"]),
            "excluded_query_groups": taxonomy.get("excluded_query_groups", {}),
            "included_esci_labels": taxonomy["membership_labels"],
        },
        "generation_rules": {
            "price": "category bounds with deterministic sha256 quantile, low-price skew",
            "popularity": "deterministic sha256 score adjusted by strongest ESCI label",
            "attributes": "first matching configured value per attribute from title plus body",
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
            "unique_fact_products": len({fact["product_id"] for fact in facts}),
            "rejected_memberships": len(rejected_memberships),
            "accepted_cards": len(cards),
            "rejected_cards": len(rejected_cards),
            "audit_queue": len(audit_queue),
            "cards_by_type": dict(sorted(card_counts.items())),
            "cards_by_category": dict(sorted(category_counts.items())),
            "facts_by_category": dict(sorted(fact_counts.items())),
        },
        "input_sha256": {
            key: _sha256_file(path) for key, path in input_paths.items()
        },
        "output_sha256": {
            key: _sha256_file(path) for key, path in output_paths.items()
        },
    }


def _generated_price(category: str, product_id: str, bounds: list[float]) -> float:
    low, high = map(float, bounds)
    quantile = _stable_unit(f"price|{category}|{product_id}") ** 1.7
    return float(max(10, round((low + (high - low) * quantile) / 10) * 10))


def _generated_order_count(category: str, product_id: str, label: str) -> int:
    unit = _stable_unit(f"orders|{category}|{product_id}")
    score = 0.55 * unit + 0.45 * LABEL_WEIGHT[label]
    return int(round(10 + 990 * score**2))


def _stable_unit(value: str) -> float:
    digest = hashlib.sha256(f"{GENERATION_SEED}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def _bestseller_evidence(fact: dict[str, Any]) -> str:
    price = fact["generated_fields"]["typical_price_cny"]
    orders = fact["generated_fields"]["order_count_90d"]
    suffix = f" | {price:.0f} | generated orders={orders}"
    title = fact["title"].replace("|", "/")
    max_title_length = max(8, 80 - len(suffix))
    if len(title) > max_title_length:
        title = title[: max_title_length - 1].rstrip() + "…"
    return title + suffix


def _truncate_evidence(value: str) -> str:
    clean = " ".join(value.replace("|", "/").split())
    return clean if len(clean) <= 80 else clean[:79].rstrip() + "…"


def _first_matching_term(text: str, terms: list[str]) -> str | None:
    for term in terms:
        if _normalized_text(term) in text:
            return term
    return None


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _clean_display_text(value: str) -> str:
    return " ".join(value.replace("\ufffd", "").split())


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


def _round_down(value: float, unit: int) -> float:
    return math.floor(value / unit) * unit


def _round_up(value: float, unit: int) -> float:
    return math.ceil(value / unit) * unit


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
