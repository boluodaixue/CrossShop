"""Build the source-aligned CategoryInsight audit dataset.

The builder preserves the approved ReferenceSeed CategoryCard seed bytes, validates
all source contracts, and publishes the approved promotion inputs beside the
legacy audit artifacts. Runtime Markdown is built separately by
``promote_category_insight_v1.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "data" / "category_insight" / "sources"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "category-insight-v1"
DEFAULT_H0_PRODUCTS = (
    ROOT / "data" / "processed" / "catalogs-v2" / "reference_seed" / "products.jsonl"
)

DATASET_VERSION = "category-insight-approved-input-v1"
BUILDER_VERSION = "build-category-candidates-v1"
LEGACY_CARD_FIELDS = frozenset(
    {
        "card_id",
        "category",
        "card_type",
        "summary",
        "raw_evidence",
        "last_updated",
        "confidence",
    }
)
KNOWLEDGE_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "category",
        "candidate_type",
        "summary",
        "raw_evidence",
        "source_ref",
        "claim_scope",
        "last_updated",
        "confidence",
        "review_status",
    }
)
LEGACY_CARD_TYPES = frozenset({"attribute", "bestseller", "price_range"})
CANDIDATE_TYPES = frozenset({"selection_guide", "pitfall"})
EXPECTED_LEGACY_COUNTS = {
    "attribute": 24,
    "bestseller": 16,
    "price_range": 8,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSON object required at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_confidence(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    confidence = float(value)
    if not 0 <= confidence <= 1:
        raise ValueError(f"{field} must be within 0..1")
    return confidence


def _validate_evidence(value: object, field: str) -> None:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError(f"{field} must contain 1..3 evidence strings")
    for evidence in value:
        text = _require_text(evidence, field)
        if len(text) > 80:
            raise ValueError(f"{field} evidence exceeds 80 characters")


def _validate_legacy_cards(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 48:
        raise ValueError(f"expected 48 legacy cards, got {len(rows)}")
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    for row in rows:
        if set(row) != LEGACY_CARD_FIELDS:
            raise ValueError(
                f"legacy card field mismatch: {sorted(set(row) ^ LEGACY_CARD_FIELDS)}"
            )
        card_id = _require_text(row["card_id"], "card_id")
        if card_id in ids:
            raise ValueError(f"duplicate card_id: {card_id}")
        ids.add(card_id)
        category = _require_text(row["category"], "category")
        card_type = _require_text(row["card_type"], "card_type")
        if card_type not in LEGACY_CARD_TYPES:
            raise ValueError(f"unsupported legacy card_type: {card_type}")
        summary = _require_text(row["summary"], "summary")
        if len(summary) > 200:
            raise ValueError(f"summary exceeds 200 characters: {card_id}")
        _validate_evidence(row["raw_evidence"], "raw_evidence")
        _require_text(row["last_updated"], "last_updated")
        if _require_confidence(row["confidence"], "confidence") < 0.5:
            raise ValueError(f"legacy confidence below admission threshold: {card_id}")
        counts[card_type] += 1
        category_counts[category] += 1
    if dict(counts) != EXPECTED_LEGACY_COUNTS:
        raise ValueError(f"unexpected legacy card distribution: {dict(counts)}")
    if len(category_counts) != 8 or set(category_counts.values()) != {6}:
        raise ValueError(f"unexpected legacy category distribution: {category_counts}")


def _validate_candidates(
    rows: list[dict[str, Any]],
    legacy_cards: list[dict[str, Any]],
) -> None:
    if len(rows) != 16:
        raise ValueError(f"expected 16 knowledge candidates, got {len(rows)}")
    ids: set[str] = set()
    legacy_by_id = {str(row["card_id"]): row for row in legacy_cards}
    for row in rows:
        if set(row) != KNOWLEDGE_CANDIDATE_FIELDS:
            raise ValueError(
                "knowledge candidate field mismatch: "
                f"{sorted(set(row) ^ KNOWLEDGE_CANDIDATE_FIELDS)}"
            )
        candidate_id = _require_text(row["candidate_id"], "candidate_id")
        if candidate_id in ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        ids.add(candidate_id)
        _require_text(row["category"], "category")
        candidate_type = _require_text(row["candidate_type"], "candidate_type")
        if candidate_type not in CANDIDATE_TYPES:
            raise ValueError(f"unsupported candidate_type: {candidate_type}")
        summary = _require_text(row["summary"], "summary")
        if len(summary) > 200:
            raise ValueError(f"summary exceeds 200 characters: {candidate_id}")
        _validate_evidence(row["raw_evidence"], "raw_evidence")
        source_ref = _require_text(row["source_ref"], "source_ref")
        _require_text(row["claim_scope"], "claim_scope")
        last_updated = _require_text(row["last_updated"], "last_updated")
        parsed_timestamp = datetime.fromisoformat(last_updated)
        if parsed_timestamp.tzinfo is None:
            raise ValueError(f"last_updated must include a timezone: {candidate_id}")
        _require_confidence(row["confidence"], "confidence")
        if row["review_status"] != "approved":
            raise ValueError(f"review_status must be approved: {candidate_id}")

        if source_ref.startswith("legacy_cards:"):
            source_ids = source_ref.removeprefix("legacy_cards:").split("|")
            if len(source_ids) != 3 or any(
                value not in legacy_by_id for value in source_ids
            ):
                raise ValueError(f"invalid legacy card source_ref: {candidate_id}")
            source_summaries = {
                str(legacy_by_id[value]["summary"]) for value in source_ids
            }
            if set(row["raw_evidence"]) != source_summaries:
                raise ValueError(f"legacy evidence is not source-exact: {candidate_id}")
            expected_confidence = min(
                float(legacy_by_id[value]["confidence"]) for value in source_ids
            )
            if float(row["confidence"]) != expected_confidence:
                raise ValueError(
                    f"legacy-derived confidence must equal its source minimum: {candidate_id}"
                )
        elif source_ref.startswith("knowledge/"):
            relative_path, separator, heading_text = source_ref.partition("#")
            source_path = ROOT / relative_path
            if not separator or not source_path.is_file():
                raise ValueError(f"invalid Markdown source_ref: {candidate_id}")
            source_text = source_path.read_text(encoding="utf-8")
            headings = [value.strip() for value in heading_text.split(",")]
            if any(f"## {heading}" not in source_text for heading in headings):
                raise ValueError(f"missing Markdown heading source: {candidate_id}")
            if any(
                str(evidence) not in source_text for evidence in row["raw_evidence"]
            ):
                raise ValueError(
                    f"Markdown evidence is not source-exact: {candidate_id}"
                )
            if float(row["confidence"]) != 0.0:
                raise ValueError(
                    "officially reviewed Markdown confidence must remain 0.0 "
                    f"without a calibrated scoring rule: {candidate_id}"
                )
        else:
            raise ValueError(f"unsupported source_ref: {candidate_id}")


def _validate_provenance(
    rows: list[dict[str, Any]],
    card_ids: set[str],
) -> set[str]:
    if len(rows) != 48:
        raise ValueError(f"expected 48 provenance rows, got {len(rows)}")
    provenance_ids = {str(row.get("card_id") or "") for row in rows}
    if provenance_ids != card_ids:
        raise ValueError("provenance card_ids do not exactly cover legacy cards")
    source_ids: set[str] = set()
    for row in rows:
        values = row.get("source_item_ids")
        if not isinstance(values, list) or not values:
            raise ValueError(f"source_item_ids required: {row.get('card_id')}")
        for value in values:
            source_ids.add(_require_text(value, "source_item_ids"))
    if len(source_ids) != 779:
        raise ValueError(f"expected 779 unique source item ids, got {len(source_ids)}")
    return source_ids


def _validate_taxonomy(path: Path, categories: set[str]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("categories"), list):
        raise TypeError("taxonomy.categories must be a list")
    taxonomy_categories = {
        _require_text(row.get("category"), "taxonomy.category")
        for row in value["categories"]
        if isinstance(row, dict)
    }
    if taxonomy_categories != categories:
        raise ValueError("taxonomy categories do not match legacy card categories")


def _h0_products_by_id(path: Path) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        product_id = _require_text(row.get("product_id"), "product_id")
        if product_id in products:
            raise ValueError(f"duplicate H0 product_id: {product_id}")
        products[product_id] = row
    return products


def _price_minor(value: object) -> int:
    decimal_value = Decimal(str(value)) * 100
    integral = decimal_value.to_integral_value()
    if decimal_value != integral or integral <= 0:
        raise ValueError(f"invalid CNY price observation: {value}")
    return int(integral)


def _validate_facts(
    rows: list[dict[str, Any]],
    source_item_ids: set[str],
    h0_products: dict[str, dict[str, Any]],
) -> None:
    if len(rows) != 779:
        raise ValueError(f"expected 779 legacy facts, got {len(rows)}")
    fact_ids = {_require_text(row.get("item_id"), "fact.item_id") for row in rows}
    if len(fact_ids) != len(rows) or fact_ids != source_item_ids:
        raise ValueError("legacy facts do not exactly cover provenance source ids")

    for row in rows:
        item_id = str(row["item_id"])
        product = h0_products.get(item_id)
        if product is None:
            raise ValueError(f"legacy fact is absent from H0: {item_id}")
        if row.get("title") != product.get("title"):
            raise ValueError(f"legacy fact title differs from H0: {item_id}")
        assignment = row.get("category_assignment")
        if not isinstance(assignment, dict):
            raise TypeError(f"category_assignment required: {item_id}")
        if assignment.get("source_leaf_name") != product.get("category"):
            raise ValueError(f"legacy fact leaf category differs from H0: {item_id}")

        current_prices: Counter[int] = Counter()
        skus = product.get("skus")
        if not isinstance(skus, list) or not skus:
            raise ValueError(f"H0 skus required: {item_id}")
        for sku in skus:
            if not isinstance(sku, dict) or not isinstance(sku.get("price"), dict):
                raise TypeError(f"H0 sku price required: {item_id}")
            price = sku["price"]
            if price.get("currency") != "CNY":
                raise ValueError(f"unexpected H0 currency: {item_id}")
            current_prices[int(price["amount_in_minor_units"])] += 1
        for field in ("raw_price_observations_cny", "price_observations_cny"):
            observations = row.get(field)
            if not isinstance(observations, list):
                raise TypeError(f"{field} must be a list: {item_id}")
            observed_prices = Counter(_price_minor(value) for value in observations)
            if any(
                current_prices[value] < count
                for value, count in observed_prices.items()
            ):
                raise ValueError(
                    f"legacy fact {field} is not an H0 SKU subset: {item_id}"
                )


def _file_entry(path: Path, records: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        entry["records"] = records
    return entry


def build(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    h0_products: Path = DEFAULT_H0_PRODUCTS,
) -> dict[str, Any]:
    reference_seed_dir = source_dir / "reference_seed_zh"
    legacy_cards_path = reference_seed_dir / "category_cards_reference_seed_zh.jsonl"
    legacy_provenance_path = reference_seed_dir / "category_card_provenance_reference_seed_zh.jsonl"
    taxonomy_path = reference_seed_dir / "category_taxonomy_reference_seed_zh.json"
    legacy_manifest_path = reference_seed_dir / "category_card_manifest_reference_seed_zh.json"
    legacy_facts_path = reference_seed_dir / "category_item_facts_reference_seed_zh.jsonl"
    candidates_path = source_dir / "knowledge_candidates.jsonl"

    legacy_cards = _read_jsonl(legacy_cards_path)
    provenance = _read_jsonl(legacy_provenance_path)
    facts = _read_jsonl(legacy_facts_path)
    candidates = _read_jsonl(candidates_path)
    _validate_legacy_cards(legacy_cards)
    _validate_candidates(candidates, legacy_cards)
    card_ids = {str(row["card_id"]) for row in legacy_cards}
    categories = {str(row["category"]) for row in legacy_cards}
    source_item_ids = _validate_provenance(provenance, card_ids)
    _validate_taxonomy(taxonomy_path, categories)

    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    expected_hashes = legacy_manifest.get("output_sha256", {})
    if expected_hashes.get("cards") != _sha256(legacy_cards_path):
        raise ValueError("legacy card hash does not match its legacy manifest")
    if expected_hashes.get("provenance") != _sha256(legacy_provenance_path):
        raise ValueError("legacy provenance hash does not match its legacy manifest")

    h0_by_id = _h0_products_by_id(h0_products)
    missing = source_item_ids - set(h0_by_id)
    if missing:
        raise ValueError(f"legacy provenance has {len(missing)} ids absent from H0")
    _validate_facts(facts, source_item_ids, h0_by_id)

    if expected_hashes.get("facts") != _sha256(legacy_facts_path):
        raise ValueError("legacy facts hash does not match its legacy manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_cards = output_dir / "legacy_category_cards.jsonl"
    output_provenance = output_dir / "legacy_category_card_provenance.jsonl"
    output_candidates = output_dir / "knowledge_candidates.jsonl"
    shutil.copyfile(legacy_cards_path, output_cards)
    shutil.copyfile(legacy_provenance_path, output_provenance)
    shutil.copyfile(candidates_path, output_candidates)

    card_type_counts = Counter(str(row["card_type"]) for row in legacy_cards)
    candidate_type_counts = Counter(str(row["candidate_type"]) for row in candidates)
    manifest: dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "counts": {
            "h0_aligned_source_item_ids": len(source_item_ids),
            "knowledge_candidates": len(candidates),
            "knowledge_candidates_by_type": dict(sorted(candidate_type_counts.items())),
            "legacy_cards": len(legacy_cards),
            "legacy_cards_by_type": dict(sorted(card_type_counts.items())),
            "legacy_categories": len(categories),
            "legacy_facts": len(facts),
            "legacy_provenance": len(provenance),
            "legacy_source_item_ids": len(source_item_ids),
        },
        "dataset_version": DATASET_VERSION,
        "h0_alignment": {
            "matched_source_item_ids": len(source_item_ids),
            "products": _file_entry(h0_products, len(h0_by_id)),
        },
        "inputs": {
            "knowledge_candidates": _file_entry(candidates_path, len(candidates)),
            "legacy_cards": _file_entry(legacy_cards_path, len(legacy_cards)),
            "legacy_facts": _file_entry(legacy_facts_path, len(facts)),
            "legacy_manifest": _file_entry(legacy_manifest_path),
            "legacy_provenance": _file_entry(legacy_provenance_path, len(provenance)),
            "legacy_taxonomy": _file_entry(taxonomy_path),
        },
        "intended_use": (
            "approved promotion inputs plus legacy audit artifacts; runtime "
            "Markdown is built by promote_category_insight_v1.py"
        ),
        "migration_source": {
            "artifact_commit": "e6852d0886bbdf42d16fe7adedb35fca8fad97d8",
            "repository": "CrossShop reference seed",
        },
        "outputs": {
            "knowledge_candidates": _file_entry(output_candidates, len(candidates)),
            "legacy_cards": _file_entry(output_cards, len(legacy_cards)),
            "legacy_provenance": _file_entry(output_provenance, len(provenance)),
        },
        "truth_boundary": {
            "bestseller": "catalog frequency/form proxy; not a real sales ranking",
            "legacy_facts": "offline provenance sidecar only; never indexed as RAG content",
            "knowledge_candidates": "user-approved promotion inputs; provenance remains sidecar-only",
            "price_range": "observed listing/specification price reference; not transaction or real-time price",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--h0-products", type=Path, default=DEFAULT_H0_PRODUCTS)
    args = parser.parse_args()
    manifest = build(args.source_dir, args.output_dir, args.h0_products)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
