import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE_DIR = ROOT / "data" / "category_insight" / "sources"
REFERENCE_SEED_SOURCE_DIR = SOURCE_DIR / "reference_seed_zh"
PUBLISHED_DIR = ROOT / "data" / "processed" / "category-insight-v1"
H0_PRODUCTS = ROOT / "data" / "processed" / "catalogs-v2" / "reference_seed" / "products.jsonl"

LEGACY_FIELDS = {
    "card_id",
    "category",
    "card_type",
    "summary",
    "raw_evidence",
    "last_updated",
    "confidence",
}
CANDIDATE_FIELDS = {
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
EXPECTED_SEED_HASHES = {
    "category_card_manifest_reference_seed_zh.json": (
        "f8e804a3a89c076a1bbdec04ce87c1c3d5e28f145832375eb913dac8dfdcb5cf"
    ),
    "category_card_provenance_reference_seed_zh.jsonl": (
        "4b2b8c30e4d5ff884a07384f239df919bcf947a906c08dc48f5564f002231a11"
    ),
    "category_cards_reference_seed_zh.jsonl": (
        "e2e21da46087de496c1c8e02b417a894f14722e0fba335d9e9cbc97322ff3879"
    ),
    "category_taxonomy_reference_seed_zh.json": (
        "1a46d7a963a3aef9f6018b3bb2a282fb49fe24f1d66b0187563ad243003f5722"
    ),
    "category_item_facts_reference_seed_zh.jsonl": (
        "c42a8a7be3e0f0dcb5064f97ad9f3c85cd058e837e155172ac5740010532e58c"
    ),
}


def _load_builder():
    path = ROOT / "scripts" / "data" / "build_category_candidates_v1.py"
    spec = importlib.util.spec_from_file_location("category_candidates_v1", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_seed_files_are_byte_exact_and_cards_are_strict() -> None:
    for filename, expected_hash in EXPECTED_SEED_HASHES.items():
        assert _sha256(REFERENCE_SEED_SOURCE_DIR / filename) == expected_hash

    cards = _read_jsonl(REFERENCE_SEED_SOURCE_DIR / "category_cards_reference_seed_zh.jsonl")
    assert len(cards) == 48
    assert len({card["card_id"] for card in cards}) == 48
    assert Counter(card["card_type"] for card in cards) == {
        "attribute": 24,
        "bestseller": 16,
        "price_range": 8,
    }
    assert len({card["category"] for card in cards}) == 8
    assert set(Counter(card["category"] for card in cards).values()) == {6}
    for card in cards:
        assert set(card) == LEGACY_FIELDS
        assert 1 <= len(card["raw_evidence"]) <= 3
        assert all(1 <= len(item) <= 80 for item in card["raw_evidence"])
        assert 1 <= len(card["summary"]) <= 200
        assert 0.5 <= card["confidence"] <= 1


def test_knowledge_candidates_use_the_approved_promotion_contract() -> None:
    candidates = _read_jsonl(SOURCE_DIR / "knowledge_candidates.jsonl")
    legacy_cards = {
        row["card_id"]: row
        for row in _read_jsonl(REFERENCE_SEED_SOURCE_DIR / "category_cards_reference_seed_zh.jsonl")
    }
    assert len(candidates) == 16
    assert len({row["candidate_id"] for row in candidates}) == 16
    assert {row["candidate_type"] for row in candidates} <= {
        "selection_guide",
        "pitfall",
    }
    assert {row["review_status"] for row in candidates} == {"approved"}
    for candidate in candidates:
        assert set(candidate) == CANDIDATE_FIELDS
        assert 1 <= len(candidate["raw_evidence"]) <= 3
        assert all(1 <= len(item) <= 80 for item in candidate["raw_evidence"])
        assert 1 <= len(candidate["summary"]) <= 200
        assert 0 <= candidate["confidence"] <= 1
        if candidate["source_ref"].startswith("legacy_cards:"):
            source_ids = (
                candidate["source_ref"].removeprefix("legacy_cards:").split("|")
            )
            assert candidate["confidence"] == min(
                legacy_cards[card_id]["confidence"] for card_id in source_ids
            )
        else:
            assert candidate["source_ref"].startswith("knowledge/")
            assert candidate["confidence"] == 0.0


def test_review_checklist_exactly_mirrors_candidate_contract() -> None:
    candidates = _read_jsonl(SOURCE_DIR / "knowledge_candidates.jsonl")
    checklist = (
        ROOT / "data" / "category_insight" / "CANDIDATE_REVIEW_CHECKLIST.md"
    ).read_text(encoding="utf-8")
    assert checklist.count("- 选择：[x] 通过　[ ] 修改　[ ] 删除") == 16
    for index, candidate in enumerate(candidates, 1):
        marker = f"## {index}. `{candidate['candidate_id']}`"
        start = checklist.index(marker)
        end = checklist.find("\n## ", start + len(marker))
        section = checklist[start:] if end == -1 else checklist[start:end]
        assert f"- 品类：{candidate['category']}" in section
        assert f"- 类型：`{candidate['candidate_type']}`" in section
        assert f"- 摘要：{candidate['summary']}" in section
        assert f"- 来源：`{candidate['source_ref']}`" in section
        assert f"- 限制：{candidate['claim_scope']}" in section
        assert f"- 置信度：`{candidate['confidence']}`" in section


def test_provenance_exactly_covers_cards_and_all_source_ids_exist_in_h0() -> None:
    cards = _read_jsonl(REFERENCE_SEED_SOURCE_DIR / "category_cards_reference_seed_zh.jsonl")
    provenance = _read_jsonl(
        REFERENCE_SEED_SOURCE_DIR / "category_card_provenance_reference_seed_zh.jsonl"
    )
    assert len(provenance) == 48
    assert {row["card_id"] for row in provenance} == {row["card_id"] for row in cards}
    source_ids = {item_id for row in provenance for item_id in row["source_item_ids"]}
    assert len(source_ids) == 779

    h0_ids = {row["product_id"] for row in _read_jsonl(H0_PRODUCTS)}
    assert len(h0_ids) == 23409
    assert source_ids <= h0_ids


def test_published_dataset_hashes_and_rebuild_are_stable(tmp_path: Path) -> None:
    filenames = (
        "legacy_category_cards.jsonl",
        "legacy_category_card_provenance.jsonl",
        "knowledge_candidates.jsonl",
        "manifest.json",
    )
    published_hashes = {
        filename: _sha256(PUBLISHED_DIR / filename) for filename in filenames
    }
    fresh_output = tmp_path / "fresh"
    first_manifest = BUILDER.build(SOURCE_DIR, fresh_output, H0_PRODUCTS)
    first_hashes = {
        filename: _sha256(fresh_output / filename) for filename in filenames
    }
    second_manifest = BUILDER.build(SOURCE_DIR, fresh_output, H0_PRODUCTS)
    second_hashes = {
        filename: _sha256(fresh_output / filename) for filename in filenames
    }
    assert first_manifest == second_manifest
    assert published_hashes == first_hashes == second_hashes
    assert first_manifest["counts"] == {
        "h0_aligned_source_item_ids": 779,
        "knowledge_candidates": 16,
        "knowledge_candidates_by_type": dict(
            sorted(
                Counter(
                    row["candidate_type"]
                    for row in _read_jsonl(SOURCE_DIR / "knowledge_candidates.jsonl")
                ).items()
            )
        ),
        "legacy_cards": 48,
        "legacy_cards_by_type": {
            "attribute": 24,
            "bestseller": 16,
            "price_range": 8,
        },
        "legacy_categories": 8,
        "legacy_facts": 779,
        "legacy_provenance": 48,
        "legacy_source_item_ids": 779,
    }
    assert (PUBLISHED_DIR / "legacy_category_cards.jsonl").read_bytes() == (
        REFERENCE_SEED_SOURCE_DIR / "category_cards_reference_seed_zh.jsonl"
    ).read_bytes()
    assert (PUBLISHED_DIR / "legacy_category_card_provenance.jsonl").read_bytes() == (
        REFERENCE_SEED_SOURCE_DIR / "category_card_provenance_reference_seed_zh.jsonl"
    ).read_bytes()


def test_runtime_uses_versioned_markdown_not_source_json() -> None:
    runtime_source = (
        ROOT / "app" / "infrastructure" / "rag" / "category_knowledge.py"
    ).read_text(encoding="utf-8")
    assert 'PROJECT_ROOT / "knowledge" / "category-insight-v1"' in runtime_source
    assert "knowledge_candidates.jsonl" not in runtime_source
    assert 'directory.glob("*.md")' in runtime_source
    assert not (ROOT / "knowledge" / "knowledge_candidates.jsonl").exists()


def test_candidate_builder_rejects_missing_candidate_source() -> None:
    missing_source = SOURCE_DIR / "__missing__"
    with pytest.raises(FileNotFoundError):
        BUILDER.build(missing_source, PUBLISHED_DIR, H0_PRODUCTS)


def _copy_sources(tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(SOURCE_DIR, destination)
    return destination


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_candidate_builder_rejects_invalid_source_ref_and_evidence(
    tmp_path: Path,
) -> None:
    invalid_ref_source = _copy_sources(tmp_path, "invalid-ref")
    path = invalid_ref_source / "knowledge_candidates.jsonl"
    rows = _read_jsonl(path)
    rows[0]["source_ref"] = "legacy_cards:missing|missing-2|missing-3"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="invalid legacy card source_ref"):
        BUILDER.build(invalid_ref_source, tmp_path / "invalid-ref-output", H0_PRODUCTS)

    invalid_evidence_source = _copy_sources(tmp_path, "invalid-evidence")
    path = invalid_evidence_source / "knowledge_candidates.jsonl"
    rows = _read_jsonl(path)
    rows[0]["raw_evidence"][0] = "并不存在于来源中的证据"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="legacy evidence is not source-exact"):
        BUILDER.build(
            invalid_evidence_source,
            tmp_path / "invalid-evidence-output",
            H0_PRODUCTS,
        )


def test_candidate_builder_rejects_naive_timestamp_and_missing_h0_ids(
    tmp_path: Path,
) -> None:
    naive_time_source = _copy_sources(tmp_path, "naive-time")
    path = naive_time_source / "knowledge_candidates.jsonl"
    rows = _read_jsonl(path)
    rows[0]["last_updated"] = "2026-08-28T00:00:00"
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="must include a timezone"):
        BUILDER.build(naive_time_source, tmp_path / "naive-time-output", H0_PRODUCTS)

    incomplete_h0 = tmp_path / "incomplete-products.jsonl"
    with H0_PRODUCTS.open(encoding="utf-8") as handle:
        first_product = handle.readline()
    incomplete_h0.write_text(
        first_product,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ids absent from H0"):
        BUILDER.build(SOURCE_DIR, tmp_path / "incomplete-h0-output", incomplete_h0)


def test_candidate_builder_enforces_evidence_based_confidence(tmp_path: Path) -> None:
    legacy_source = _copy_sources(tmp_path, "legacy-confidence")
    path = legacy_source / "knowledge_candidates.jsonl"
    rows = _read_jsonl(path)
    rows[0]["confidence"] = 0.5
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="source minimum"):
        BUILDER.build(legacy_source, tmp_path / "legacy-confidence-output", H0_PRODUCTS)

    markdown_source = _copy_sources(tmp_path, "markdown-confidence")
    path = markdown_source / "knowledge_candidates.jsonl"
    rows = _read_jsonl(path)
    markdown_row = next(
        row for row in rows if row["source_ref"].startswith("knowledge/")
    )
    markdown_row["confidence"] = 0.5
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="must remain 0.0"):
        BUILDER.build(
            markdown_source,
            tmp_path / "markdown-confidence-output",
            H0_PRODUCTS,
        )


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "title",
            lambda row: row.__setitem__("title", "篡改标题"),
            "title differs from H0",
        ),
        (
            "leaf",
            lambda row: row["category_assignment"].__setitem__(
                "source_leaf_name", "篡改类目"
            ),
            "leaf category differs from H0",
        ),
        (
            "raw-price",
            lambda row: row.__setitem__("raw_price_observations_cny", [99999999.0]),
            "raw_price_observations_cny is not an H0 SKU subset",
        ),
        (
            "filtered-price",
            lambda row: row.__setitem__("price_observations_cny", [99999999.0]),
            "price_observations_cny is not an H0 SKU subset",
        ),
    ],
)
def test_legacy_fact_validation_is_fail_closed(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict], None],
    message: str,
) -> None:
    source = _copy_sources(tmp_path, f"fact-{name}")
    path = source / "reference_seed_zh" / "category_item_facts_reference_seed_zh.jsonl"
    rows = _read_jsonl(path)
    mutate(rows[0])
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match=message):
        BUILDER.build(source, tmp_path / f"fact-{name}-output", H0_PRODUCTS)
