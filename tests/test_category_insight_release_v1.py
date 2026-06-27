import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from agentscope.credential import OpenAICredential
from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse
from agentscope.rag import KnowledgeBase, QdrantStore

from app.infrastructure.rag.category_knowledge import (
    KNOWLEDGE_DIR,
    bootstrap_category_knowledge,
)
from app.infrastructure.settings import PROJECT_ROOT, load_settings

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data" / "category_insight" / "sources" / "knowledge_candidates.jsonl"
LEGACY_CARDS = (
    ROOT
    / "data"
    / "category_insight"
    / "sources"
    / "taobao_zh"
    / "category_cards_taobao_zh.jsonl"
)
LEGACY_PROVENANCE = (
    ROOT
    / "data"
    / "category_insight"
    / "sources"
    / "taobao_zh"
    / "category_card_provenance_taobao_zh.jsonl"
)
RELEASE_DIR = ROOT / "data" / "category_insight" / "releases" / "category-insight-v1"
RUNTIME_DIR = ROOT / "knowledge" / "category-insight-v1"

APPROVED_FIELDS = {
    "card_id",
    "category",
    "card_type",
    "summary",
    "usage_boundary",
    "review_status",
}
FORBIDDEN_RUNTIME_TERMS = {
    "参考仓库",
    "内部知识",
    "候选",
    "审核",
    "置信度",
    "尚未外部验证",
    "当前淘宝目录",
    "已删除原文",
}


def _load_builder():
    path = ROOT / "scripts" / "data" / "promote_category_insight_v1.py"
    spec = importlib.util.spec_from_file_location("promote_category_insight_v1", path)
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


def test_approved_source_contract_and_type_counts() -> None:
    rows = _read_jsonl(SOURCE)
    assert len(rows) == 16
    assert len({row["candidate_id"] for row in rows}) == 16
    assert Counter(row["candidate_type"] for row in rows) == {
        "selection_guide": 12,
        "pitfall": 4,
    }
    assert {row["review_status"] for row in rows} == {"approved"}
    legacy_cards = {row["card_id"]: row for row in _read_jsonl(LEGACY_CARDS)}
    for row in rows:
        if row["source_ref"].startswith("legacy_cards:"):
            source_ids = row["source_ref"].removeprefix("legacy_cards:").split("|")
            assert row["confidence"] == min(
                legacy_cards[card_id]["confidence"] for card_id in source_ids
            )
        else:
            assert row["confidence"] == 0.0
    assert all(row["summary"].startswith("可比较维度包括") for row in rows[:8])
    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["kc-car-ambient-light-selection-guide-01"]["category"] == "汽车氛围灯"
    assert (
        "特定车型与年款适配必须回到具体商品说明复核"
        in by_id["kc-car-ambient-light-selection-guide-01"]["summary"]
    )
    assert by_id["kc-neck-massager-selection-guide-01"]["category"] == "颈椎按摩器"


def test_release_schema_provenance_and_official_sources() -> None:
    cards = _read_jsonl(RELEASE_DIR / "approved_cards.jsonl")
    provenance = _read_jsonl(RELEASE_DIR / "approved_card_provenance.jsonl")
    assert len(cards) == len(provenance) == 64
    assert all(set(card) == APPROVED_FIELDS for card in cards)
    assert {card["review_status"] for card in cards} == {"approved"}
    assert Counter(card["card_type"] for card in cards) == {
        "attribute": 24,
        "bestseller": 16,
        "price_range": 8,
        "selection_guide": 12,
        "pitfall": 4,
    }
    assert {row["card_id"] for row in provenance} == {card["card_id"] for card in cards}
    externally_sourced = [row for row in provenance if row.get("official_sources")]
    assert len(externally_sourced) == 8
    by_id = {row["card_id"]: row for row in provenance}
    digital_urls = {
        source["url"]
        for source in by_id["kc-digital-accessories-selection-guide-01"][
            "official_sources"
        ]
    }
    assert digital_urls == {
        "https://www.usb.org/usb-charger-pd",
        "https://support.apple.com/en-ie/111805",
        "https://support.google.com/pixelphone/answer/16479972?hl=en",
    }
    digital_provenance = by_id["kc-digital-accessories-selection-guide-01"]
    apple = next(
        source
        for source in digital_provenance["official_sources"]
        if source["url"] == "https://support.apple.com/en-ie/111805"
    )
    assert apple == {
        "authority": "Apple Support",
        "jurisdiction": (
            "35W Dual USB-C Port Power Adapter; power distribution when two "
            "devices are connected"
        ),
        "reviewed_on": "2026-08-28",
        "url": "https://support.apple.com/en-ie/111805",
        "version": (
            "Apple Support article 111805: How to use the 35W Dual USB-C Port "
            "Power Adapter; reviewed 2026-08-28"
        ),
    }
    assert digital_provenance["verified_evidence"] == [
        "USB-IF 说明 USB Power Delivery 由供电端与所接设备协商供电能力。",
        "Apple 说明 35W 双 USB-C 端口电源适配器同时连接两台设备时，会根据设备功率需求自动分配电力。",
        "Google Pixelsnap 帮助页按具体 Pixel 型号、Qi2 充电支持和兼容磁吸保护壳说明适用条件。",
    ]
    anc_card = next(
        card for card in cards if card["card_id"] == "kc-digital-accessories-pitfall-01"
    )
    assert anc_card["summary"] == (
        "ANC 使用电子方式抵消噪声；仅写“降噪”不足以证明支持 ANC；"
        "普通 ANC 不能自动视为听力防护。"
    )
    anc_provenance = by_id["kc-digital-accessories-pitfall-01"]
    assert anc_provenance["verified_evidence"] == [
        "NIOSH 说明主动降噪耳机或耳塞使用电子方式抵消声音；除非标有 NRR，否则不应视为听力保护器。",
    ]
    assert "被动" not in "".join(anc_provenance["verified_evidence"])
    home_pitfall = by_id["kc-home-living-pitfall-01"]
    assert home_pitfall["verified_evidence"] == [
        "FedEx UK 的易碎品指引要求使用坚固硬箱、逐件缓冲、填满空隙并紧密放置。",
        "DHL Express UK 2026 指南要求易碎物品远离外箱四角，并受具体运输条款约束。",
    ]
    assert "四角" not in home_pitfall["verified_evidence"][0]
    assert "四角" in home_pitfall["verified_evidence"][1]
    for row in provenance:
        if row["card_id"].startswith("cc-"):
            assert row["raw_evidence"]
            assert row["raw_evidence_status"] == "catalog_evidence_not_runtime"
            assert row["official_sources"] == []
            assert row["verification_status"] == "catalog_evidence_reviewed"
            assert "verified_evidence" not in row
            continue
        assert row["source_ref"]
        assert row["raw_evidence"]
        assert 0 <= row["confidence"] <= 1
        assert row["reviewer"] == "user_explicit_approval"
        assert row["reviewed_at"] == "2026-08-28T00:00:00+08:00"
        assert row["source_reviewed_on"] == "2026-08-28"
        assert row["applicability_scope"]
        if row["official_sources"]:
            assert row["confidence"] == 0.0
            assert row["raw_evidence_status"] == "legacy_unverified_not_runtime"
            assert row["verification_status"] == "official_source_reviewed"
            assert row["verified_evidence"]
            assert all(item.strip() for item in row["verified_evidence"])
        else:
            assert row["confidence"] > 0.0
            assert row["raw_evidence_status"] == "catalog_evidence_not_runtime"
            assert row["verification_status"] == "catalog_evidence_reviewed"
            assert "verified_evidence" not in row
        for source in row["official_sources"]:
            assert source["url"].startswith("https://")
            assert source["authority"]
            assert source["jurisdiction"]
            assert source["version"]
            assert source["reviewed_on"] == "2026-08-28"


def test_legacy_cards_and_provenance_are_preserved_exactly() -> None:
    source_cards = {row["card_id"]: row for row in _read_jsonl(LEGACY_CARDS)}
    source_provenance = {row["card_id"]: row for row in _read_jsonl(LEGACY_PROVENANCE)}
    release_cards = {
        row["card_id"]: row for row in _read_jsonl(RELEASE_DIR / "approved_cards.jsonl")
    }
    release_provenance = {
        row["card_id"]: row
        for row in _read_jsonl(RELEASE_DIR / "approved_card_provenance.jsonl")
    }
    assert len(source_cards) == len(source_provenance) == 48
    for card_id, source_card in source_cards.items():
        released = release_cards[card_id]
        for field in ("card_id", "category", "card_type", "summary"):
            assert released[field] == source_card[field]
        boundary = released["usage_boundary"]
        if source_card["card_type"] == "bestseller":
            assert "不代表实时热销、销量排行或市场份额" in boundary
        elif source_card["card_type"] == "attribute":
            assert "历史样本的标题或属性出现分布" in boundary
            assert "不代表市场占比" in boundary
            assert "医疗或功效词仅为商品标题声明，未验证功效" in boundary
        else:
            assert source_card["card_type"] == "price_range"
            assert "人民币历史目录" in boundary
            assert "以实时商品搜索为准" in boundary
        sidecar = release_provenance[card_id]
        assert sidecar["raw_evidence"] == source_card["raw_evidence"]
        assert sidecar["last_updated"] == source_card["last_updated"]
        assert sidecar["confidence"] == source_card["confidence"]
        for field, value in source_provenance[card_id].items():
            assert sidecar[field] == value


def test_each_card_has_one_clean_runtime_markdown() -> None:
    cards = _read_jsonl(RELEASE_DIR / "approved_cards.jsonl")
    provenance = {
        row["card_id"]: row
        for row in _read_jsonl(RELEASE_DIR / "approved_card_provenance.jsonl")
    }
    paths = sorted(RUNTIME_DIR.glob("*.md"))
    assert len(paths) == len(cards) == 64
    assert {path.stem for path in paths} == {card["card_id"] for card in cards}
    by_id = {card["card_id"]: card for card in cards}
    for path in paths:
        content = path.read_text(encoding="utf-8")
        card = by_id[path.stem]
        assert content == BUILDER._runtime_markdown(card)
        lines = content.splitlines()
        assert lines[0] == f"# 品类：{card['category']}"
        assert lines[2] == f"类型：{card['card_type']}"
        assert lines[4] == f"正文：{card['summary']}"
        assert lines[6] == f"使用边界：{card['usage_boundary']}"
        if path.stem.startswith("kc-"):
            assert not re.search(r"\d+(?:\.\d+)?%", content)
        assert not any(term in content for term in FORBIDDEN_RUNTIME_TERMS)
        assert "source_ref" not in content
        assert "raw_evidence" not in content
        assert "confidence" not in content
        assert "verification_status" not in content
        assert all(
            item not in content for item in provenance[path.stem]["raw_evidence"]
        )
        assert "http://" not in content and "https://" not in content
        assert "reviewed_at" not in content and "reviewer" not in content


def _output_hashes(release_dir: Path, runtime_dir: Path) -> dict[str, str]:
    paths = [
        release_dir / "approved_cards.jsonl",
        release_dir / "approved_card_provenance.jsonl",
        release_dir / "retrieval_eval_cases.jsonl",
        release_dir / "manifest.json",
        *sorted(runtime_dir.glob("*.md")),
    ]
    return {path.name: _sha256(path) for path in paths}


def test_promotion_rebuild_is_stable_and_matches_checked_in_release(
    tmp_path: Path,
) -> None:
    fresh_release = tmp_path / "release"
    fresh_runtime = tmp_path / "knowledge"
    first = BUILDER.build(SOURCE, fresh_release, fresh_runtime)
    first_hashes = _output_hashes(fresh_release, fresh_runtime)
    second = BUILDER.build(SOURCE, fresh_release, fresh_runtime)
    second_hashes = _output_hashes(fresh_release, fresh_runtime)
    checked_in_hashes = _output_hashes(RELEASE_DIR, RUNTIME_DIR)
    assert first == second
    assert first_hashes == second_hashes == checked_in_hashes
    assert first["counts"] == {
        "approved_cards": 64,
        "approved_cards_by_type": {
            "attribute": 24,
            "bestseller": 16,
            "pitfall": 4,
            "price_range": 8,
            "selection_guide": 12,
        },
        "legacy_cards": 48,
        "officially_sourced_cards": 8,
        "retrieval_eval_cases": 64,
        "runtime_markdown_documents": 64,
    }


def test_manifest_hashes_cover_release_and_runtime_documents() -> None:
    manifest = json.loads((RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    acceptance = RELEASE_DIR / "REAL_BGE_M3_ACCEPTANCE.md"
    assert acceptance.is_file()
    assert acceptance.name not in json.dumps(manifest, ensure_ascii=False)
    assert manifest["release_version"] == "category-insight-v1"
    assert manifest["runtime"]["collection_default"] == "globex_category_kb_v1"
    assert manifest["runtime"]["knowledge_directory"] == "knowledge/category-insight-v1"
    for name, entry in manifest["outputs"].items():
        if name == "runtime_documents":
            for filename, runtime_entry in entry.items():
                assert runtime_entry["sha256"] == _sha256(RUNTIME_DIR / filename)
            continue
        filename = {
            "approved_cards": "approved_cards.jsonl",
            "approved_card_provenance": "approved_card_provenance.jsonl",
            "retrieval_eval_cases": "retrieval_eval_cases.jsonl",
        }[name]
        assert entry["sha256"] == _sha256(RELEASE_DIR / filename)


def test_eval_dataset_has_one_non_verbatim_query_per_card() -> None:
    cards = {
        row["card_id"]: row for row in _read_jsonl(RELEASE_DIR / "approved_cards.jsonl")
    }
    cases = _read_jsonl(RELEASE_DIR / "retrieval_eval_cases.jsonl")
    assert len(cases) == 64
    assert {case["expected_card_id"] for case in cases} == set(cards)
    assert {case["kind"] for case in cases} == {
        "attribute",
        "bestseller",
        "price_range",
        "selection_guide",
        "pitfall",
    }
    for case in cases:
        card = cards[case["expected_card_id"]]
        assert case["query"] != card["summary"]
        assert case["query"] not in card["summary"]
        assert case["relevant"] == [f"{card['card_id']}.md"]


def test_default_runtime_directory_and_collection_are_versioned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert KNOWLEDGE_DIR == PROJECT_ROOT / "knowledge" / "category-insight-v1"
    assert KNOWLEDGE_DIR == RUNTIME_DIR
    assert not any(
        path.parent == ROOT / "knowledge" for path in KNOWLEDGE_DIR.glob("*.md")
    )
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CATEGORY_KB_COLLECTION", raising=False)
    settings = load_settings()
    assert settings.category_kb_collection == "globex_category_kb_v1"


_TERM_GROUPS = (
    ("护颈和助眠", "乳胶枕"),
    ("写字椅", "儿童学习椅"),
    ("折起来收纳", "平板电脑支架"),
    ("便携储能", "户外电源"),
    ("补光设备", "手机直播补光灯"),
    ("特定年款汽车", "汽车氛围灯"),
    ("球包", "羽毛球包"),
    ("按摩设备", "颈椎按摩器"),
    ("舱位和行李额度", "计划使用的舱位"),
    ("二十寸登机箱", "20 寸"),
    ("标准化性能项", "光束距离"),
    ("超亮", "测试信息"),
    ("合规资料", "食品接触用途"),
    ("邮寄陶瓷茶具", "坚固外箱"),
    ("多口充电器", "端口分配"),
    ("工作场所听力防护", "听力防护"),
)


class ReleaseTermEmbedding(EmbeddingModelBase):
    def __init__(self) -> None:
        super().__init__(
            credential=OpenAICredential(api_key="sk-test"),
            model="release-term-axis",
            dimensions=len(_TERM_GROUPS),
            parameters=None,
            context_size=8192,
            batch_size=16,
            max_retries=0,
            retry_delay=0.0,
        )

    async def __call__(self, inputs, **kwargs) -> EmbeddingResponse:
        del kwargs
        texts = inputs if isinstance(inputs, list) else [inputs]
        embeddings = [
            [
                1.0 if any(term in str(text) for term in group) else 0.0
                for group in _TERM_GROUPS
            ]
            for text in texts
        ]
        return EmbeddingResponse(embeddings=embeddings)


async def test_bootstrap_count_idempotence_and_retrieval_source_restoration(
    tmp_path: Path,
) -> None:
    knowledge_base = KnowledgeBase(
        name="category_insight_release_v1_test",
        description="release v1 deterministic retrieval test",
        embedding_model=ReleaseTermEmbedding(),
        vector_store=QdrantStore(path=str(tmp_path / "qdrant")),
        collection="category_insight_release_v1_test",
    )
    assert await bootstrap_category_knowledge(knowledge_base) == 64
    assert await bootstrap_category_knowledge(knowledge_base) == 0
    documents = await knowledge_base.list_documents()
    assert len(documents) == 64
    expected_ids = {path.stem for path in RUNTIME_DIR.glob("*.md")}
    expected_sources = {path.name for path in RUNTIME_DIR.glob("*.md")}
    assert {document.document_id for document in documents} == expected_ids
    assert {document.source for document in documents} == expected_sources
    cards = _read_jsonl(RELEASE_DIR / "approved_cards.jsonl")
    assert {card["card_type"] for card in cards} == {
        "attribute",
        "bestseller",
        "price_range",
        "selection_guide",
        "pitfall",
    }
