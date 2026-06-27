"""Promote the 16 approved CategoryInsight records into a runtime release.

The release keeps review, provenance, confidence, and official-source metadata
outside the Markdown corpus consumed by AgentScope ``KnowledgeBase``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT / "data" / "category_insight" / "sources" / "knowledge_candidates.jsonl"
)
DEFAULT_LEGACY_CARDS = (
    ROOT
    / "data"
    / "category_insight"
    / "sources"
    / "taobao_zh"
    / "category_cards_taobao_zh.jsonl"
)
DEFAULT_LEGACY_PROVENANCE = (
    ROOT
    / "data"
    / "category_insight"
    / "sources"
    / "taobao_zh"
    / "category_card_provenance_taobao_zh.jsonl"
)
DEFAULT_RELEASE_DIR = (
    ROOT / "data" / "category_insight" / "releases" / "category-insight-v1"
)
DEFAULT_RUNTIME_DIR = ROOT / "knowledge" / "category-insight-v1"

RELEASE_VERSION = "category-insight-v1"
BUILDER_VERSION = "promote-category-insight-v1"
REVIEWED_AT = "2026-08-28T00:00:00+08:00"
SOURCE_REVIEWED_ON = "2026-08-28"

CANDIDATE_FIELDS = frozenset(
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
APPROVED_CARD_FIELDS = frozenset(
    {
        "card_id",
        "category",
        "card_type",
        "summary",
        "usage_boundary",
        "review_status",
    }
)
EXPECTED_TYPES = {"selection_guide": 12, "pitfall": 4}
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
EXPECTED_LEGACY_TYPES = {"attribute": 24, "bestseller": 16, "price_range": 8}
EXPECTED_RELEASE_TYPES = {
    **EXPECTED_LEGACY_TYPES,
    "selection_guide": 12,
    "pitfall": 4,
}
FORBIDDEN_RUNTIME_TERMS = (
    "参考仓库",
    "内部知识",
    "候选",
    "审核",
    "置信度",
    "尚未外部验证",
    "当前淘宝目录",
    "已删除原文",
)


def _source(
    authority: str,
    url: str,
    jurisdiction: str,
    version: str,
) -> dict[str, str]:
    return {
        "authority": authority,
        "jurisdiction": jurisdiction,
        "reviewed_on": SOURCE_REVIEWED_ON,
        "url": url,
        "version": version,
    }


IATA_SOURCE = _source(
    "International Air Transport Association (IATA)",
    "https://www.iata.org/bags/index",
    "international; each airline's conditions of carriage remain controlling",
    "Passenger Baggage Rules, reviewed 2026-08-28",
)
ANSI_FL1_SOURCE = _source(
    "American National Standards Institute / PLATO",
    "https://webstore.ansi.org/standards/ansi/ansiplatofl2025",
    "portable lighting products covered by the standard",
    "ANSI/PLATO FL 1-2025",
)
FDA_SOURCE = _source(
    "U.S. Food and Drug Administration",
    "https://www.fda.gov/food/environmental-contaminants-food/questions-and-answers-lead-glazed-traditional-pottery",
    "United States; pottery intended for cooking, serving, or storing food",
    "FDA consumer guidance, reviewed 2026-08-28",
)
EC_SOURCE = _source(
    "European Commission",
    "https://food.ec.europa.eu/food-safety/chemical-safety/food-contact-materials_en",
    "European Union food-contact materials",
    "Regulation (EC) No 1935/2004 framework, reviewed 2026-08-28",
)
FEDEX_UK_SOURCE = _source(
    "FedEx United Kingdom",
    "https://www.fedex.com/en-gb/how-to/pack/fragile-items.html",
    "FedEx UK fragile-item packing guidance",
    "How to pack fragile items, reviewed 2026-08-28",
)
DHL_UK_SOURCE = _source(
    "DHL Express United Kingdom",
    "https://mydhl.express.dhl/content/dam/downloads/gb/en/rate-guide/service_and_rate_guide_gb_en.pdf",
    "DHL Express UK shipments",
    "DHL Express Service & Rate Guide 2026: United Kingdom",
)
USB_IF_SOURCE = _source(
    "USB Implementers Forum",
    "https://www.usb.org/usb-charger-pd",
    "USB Power Delivery chargers and connected USB devices",
    "USB Charger (USB Power Delivery), reviewed 2026-08-28",
)
APPLE_SOURCE = _source(
    "Apple Support",
    "https://support.apple.com/en-ie/111805",
    "35W Dual USB-C Port Power Adapter; power distribution when two devices are connected",
    "Apple Support article 111805: How to use the 35W Dual USB-C Port Power Adapter; reviewed 2026-08-28",
)
GOOGLE_SOURCE = _source(
    "Google Pixel Phone Help",
    "https://support.google.com/pixelphone/answer/16479972?hl=en",
    "supported Pixel models, Pixelsnap Qi2 charging, and compatible magnetic cases",
    "Google Pixel Phone Help article 16479972: Use Pixelsnap accessories; reviewed 2026-08-28",
)
CDC_NIOSH_SOURCE = _source(
    "U.S. CDC / NIOSH",
    "https://www.cdc.gov/niosh/noise/prevent/ppe.html",
    "United States occupational hearing protection guidance",
    "Provide Hearing Protection, 2024-01-31; reviewed 2026-08-28",
)

OFFICIAL_SOURCES: dict[str, list[dict[str, str]]] = {
    "kc-travel-gear-selection-guide-01": [IATA_SOURCE],
    "kc-travel-gear-pitfall-01": [IATA_SOURCE],
    "kc-outdoor-sports-selection-guide-01": [ANSI_FL1_SOURCE],
    "kc-outdoor-sports-pitfall-01": [ANSI_FL1_SOURCE],
    "kc-home-living-selection-guide-01": [FDA_SOURCE, EC_SOURCE],
    "kc-home-living-pitfall-01": [FEDEX_UK_SOURCE, DHL_UK_SOURCE],
    "kc-digital-accessories-selection-guide-01": [
        USB_IF_SOURCE,
        APPLE_SOURCE,
        GOOGLE_SOURCE,
    ],
    "kc-digital-accessories-pitfall-01": [CDC_NIOSH_SOURCE],
}

VERIFIED_EVIDENCE: dict[str, list[str]] = {
    "kc-travel-gear-selection-guide-01": [
        "IATA 提醒旅客，随身行李额度会随舱等、航司和机型变化，应向承运航司核对。",
    ],
    "kc-travel-gear-pitfall-01": [
        "IATA 将随身行李尺寸和重量要求交由具体航司规则决定，因此商品英寸标称不能单独证明可随身携带。",
    ],
    "kc-outdoor-sports-selection-guide-01": [
        "ANSI/PLATO FL 1-2025 为便携照明的光输出、光束距离、峰值光强、续航、抗冲击和防水等性能声明提供统一口径。",
    ],
    "kc-outdoor-sports-pitfall-01": [
        "ANSI/PLATO FL 1-2025 以定义和测试方法支持可比较的照明性能声明；只有“超亮”宣传不能替代这些测试信息。",
    ],
    "kc-home-living-selection-guide-01": [
        "FDA 提醒食品用陶器与仅供装饰器皿应按用途和警示区分，不能仅凭外观或宣传判断可用于食品。",
        "欧盟委员会说明食品接触材料须符合其销售或使用法域的安全框架，且合规判断与材料、预期用途及经营者资料有关。",
    ],
    "kc-home-living-pitfall-01": [
        "FedEx UK 的易碎品指引要求使用坚固硬箱、逐件缓冲、填满空隙并紧密放置。",
        "DHL Express UK 2026 指南要求易碎物品远离外箱四角，并受具体运输条款约束。",
    ],
    "kc-digital-accessories-selection-guide-01": [
        "USB-IF 说明 USB Power Delivery 由供电端与所接设备协商供电能力。",
        "Apple 说明 35W 双 USB-C 端口电源适配器同时连接两台设备时，会根据设备功率需求自动分配电力。",
        "Google Pixelsnap 帮助页按具体 Pixel 型号、Qi2 充电支持和兼容磁吸保护壳说明适用条件。",
    ],
    "kc-digital-accessories-pitfall-01": [
        "NIOSH 说明主动降噪耳机或耳塞使用电子方式抵消声音；除非标有 NRR，否则不应视为听力保护器。",
    ],
}

APPLICABILITY_SCOPES: dict[str, str] = {
    "kc-latex-pillow-selection-guide-01": "旧淘宝中文目录的材质、功能和适用人群属性维度；不是市场分布或医疗证据。",
    "kc-children-study-chair-selection-guide-01": "旧淘宝中文目录的调节、材质和适用年龄属性维度；不是人体工学或安全认证。",
    "kc-tablet-stand-selection-guide-01": "旧淘宝中文目录的便携、调节和材质属性维度；具体承重和兼容性另核。",
    "kc-portable-power-station-selection-guide-01": "旧淘宝中文目录的场景、容量功率和电池类型属性维度；具体型号规格另核。",
    "kc-phone-live-fill-light-selection-guide-01": "旧淘宝中文目录的使用方式、功能和形态属性维度；不是光学性能测试。",
    "kc-car-ambient-light-selection-guide-01": "旧淘宝中文目录的灯光、安装和车型属性维度；道路合规、安装安全和车型兼容另核。",
    "kc-badminton-bag-selection-guide-01": "旧淘宝中文目录的容量、背负方式和品牌属性维度；不是市场份额或真伪证明。",
    "kc-neck-massager-selection-guide-01": "旧淘宝中文目录的按摩方式、功能和部位属性维度；不是医疗证据。",
    "kc-travel-gear-selection-guide-01": "国际航空随身行李的一般选购提示；具体航司、舱等、机型和行程规则优先。",
    "kc-travel-gear-pitfall-01": "国际航空随身行李尺寸提示；商品英寸标称不等同于航司接受资格。",
    "kc-outdoor-sports-selection-guide-01": "ANSI/PLATO FL 1-2025 覆盖的手持和便携照明性能声明。",
    "kc-outdoor-sports-pitfall-01": "ANSI/PLATO FL 1-2025 照明性能信息完整性提示。",
    "kc-home-living-selection-guide-01": "美国与欧盟食品接触材料的一般核对提示；具体销售和使用法域优先。",
    "kc-home-living-pitfall-01": "FedEx UK 与 DHL Express UK 的易碎品包装提示；具体承运合同优先。",
    "kc-digital-accessories-selection-guide-01": "USB 认证及 Apple、Google 设备配件兼容提示；精确产品型号和修订版本优先。",
    "kc-digital-accessories-pitfall-01": "ANC 与听力防护的区分提示；听力防护合规提示限美国 NIOSH/NRR 语境。",
}

EVAL_QUERIES: dict[str, str] = {
    "kc-latex-pillow-selection-guide-01": "侧睡想换枕头，只看护颈和助眠宣传够不够？",
    "kc-children-study-chair-selection-guide-01": "给小学生买写字椅，需要先确认哪些调节和适龄信息？",
    "kc-tablet-stand-selection-guide-01": "出差带平板支架，想折起来收纳还应该查什么？",
    "kc-portable-power-station-selection-guide-01": "给露营设备供电，便携储能要核对哪些铭牌规格？",
    "kc-phone-live-fill-light-selection-guide-01": "手机拍产品视频时，补光设备的使用方式和形态怎么比较？",
    "kc-car-ambient-light-selection-guide-01": "给特定年款汽车加氛围灯，安装前要确认哪些适配信息？",
    "kc-badminton-bag-selection-guide-01": "两支球拍和随身装备要装进球包，容量和背法怎么选？",
    "kc-neck-massager-selection-guide-01": "按摩设备写着舒缓肩颈，能不能当作治疗效果依据？",
    "kc-travel-gear-selection-guide-01": "短途飞行选旅行箱时，舱位和行李额度要怎么核对？",
    "kc-travel-gear-pitfall-01": "商品写二十寸登机箱，是不是任何航班都能带上飞机？",
    "kc-outdoor-sports-selection-guide-01": "露营照明写着续航很长，哪些标准化性能项值得比较？",
    "kc-outdoor-sports-pitfall-01": "头灯只有超亮宣传却没有测试信息，能判断性能吗？",
    "kc-home-living-selection-guide-01": "餐具写天然食品级就够了吗，还需按销售地检查哪些合规资料？",
    "kc-home-living-pitfall-01": "邮寄陶瓷茶具时，外箱和缓冲包装要确认什么？",
    "kc-digital-accessories-selection-guide-01": "多口充电器配磁吸无线充，怎样核对端口分配、手机和保护壳？",
    "kc-digital-accessories-pitfall-01": "耳机只宣传降噪，能确认是 ANC 并当作美国工作场所听力防护吗？",
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_entry(path: Path, records: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if records is not None:
        entry["records"] = records
    return entry


def _validate_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_candidates(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 16:
        raise ValueError(f"expected 16 approved records, got {len(rows)}")
    ids: set[str] = set()
    type_counts: Counter[str] = Counter()
    for row in rows:
        if set(row) != CANDIDATE_FIELDS:
            raise ValueError(
                f"candidate field mismatch: {sorted(set(row) ^ CANDIDATE_FIELDS)}"
            )
        card_id = _validate_text(row["candidate_id"], "candidate_id")
        if card_id in ids:
            raise ValueError(f"duplicate candidate_id: {card_id}")
        ids.add(card_id)
        _validate_text(row["category"], "category")
        card_type = _validate_text(row["candidate_type"], "candidate_type")
        type_counts[card_type] += 1
        summary = _validate_text(row["summary"], "summary")
        boundary = _validate_text(row["claim_scope"], "claim_scope")
        if row["review_status"] != "approved":
            raise ValueError(f"record is not approved: {card_id}")
        if card_id not in APPLICABILITY_SCOPES or card_id not in EVAL_QUERIES:
            raise ValueError(f"missing release metadata: {card_id}")
        for term in FORBIDDEN_RUNTIME_TERMS:
            if term in summary or term in boundary:
                raise ValueError(f"forbidden runtime term {term!r}: {card_id}")
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"approved confidence must be numeric: {card_id}")
        if card_id in OFFICIAL_SOURCES and float(confidence) != 0.0:
            raise ValueError(f"officially reviewed confidence must be 0.0: {card_id}")
        if card_id not in OFFICIAL_SOURCES and not 0 < float(confidence) <= 1:
            raise ValueError(f"catalog confidence must be within (0, 1]: {card_id}")
        evidence = row["raw_evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"raw_evidence is required: {card_id}")
        _validate_text(row["source_ref"], "source_ref")

    if dict(type_counts) != EXPECTED_TYPES:
        raise ValueError(f"unexpected approved type counts: {dict(type_counts)}")
    if set(OFFICIAL_SOURCES) != {
        value for value in ids if value.startswith("kc-") and value in OFFICIAL_SOURCES
    }:
        raise ValueError("official source mapping contains an unknown card id")
    if len(OFFICIAL_SOURCES) != 8:
        raise ValueError("exactly eight externally verified cards are required")
    if set(VERIFIED_EVIDENCE) != set(OFFICIAL_SOURCES):
        raise ValueError("verified evidence must cover exactly the official cards")
    for card_id, evidence in VERIFIED_EVIDENCE.items():
        if not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            raise ValueError(f"verified evidence must be non-empty: {card_id}")
    for row in rows[:8]:
        if not str(row["summary"]).startswith("可比较维度包括"):
            raise ValueError(
                f"legacy-derived summary must use comparison framing: {row['candidate_id']}"
            )
    categories = {str(row["candidate_id"]): str(row["category"]) for row in rows}
    if categories["kc-car-ambient-light-selection-guide-01"] != "汽车氛围灯":
        raise ValueError("car ambient light category is not normalized")
    if categories["kc-neck-massager-selection-guide-01"] != "颈椎按摩器":
        raise ValueError("neck massager category is not normalized")


def _validate_legacy_inputs(
    cards: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if len(cards) != 48 or len(provenance) != 48:
        raise ValueError("exactly 48 legacy cards and provenance rows are required")
    ids: set[str] = set()
    type_counts: Counter[str] = Counter()
    for card in cards:
        if set(card) != LEGACY_CARD_FIELDS:
            raise ValueError(
                f"legacy card field mismatch: {sorted(set(card) ^ LEGACY_CARD_FIELDS)}"
            )
        card_id = _validate_text(card["card_id"], "legacy.card_id")
        if card_id in ids:
            raise ValueError(f"duplicate legacy card_id: {card_id}")
        ids.add(card_id)
        _validate_text(card["category"], "legacy.category")
        card_type = _validate_text(card["card_type"], "legacy.card_type")
        type_counts[card_type] += 1
        _validate_text(card["summary"], "legacy.summary")
        _validate_text(card["last_updated"], "legacy.last_updated")
        evidence = card["raw_evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"legacy raw_evidence is required: {card_id}")
        confidence = card["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 < float(confidence) <= 1
        ):
            raise ValueError(f"legacy confidence must be within (0, 1]: {card_id}")
    if dict(type_counts) != EXPECTED_LEGACY_TYPES:
        raise ValueError(f"unexpected legacy type counts: {dict(type_counts)}")

    provenance_by_id: dict[str, dict[str, Any]] = {}
    for item in provenance:
        card_id = _validate_text(item.get("card_id"), "legacy provenance card_id")
        if card_id in provenance_by_id:
            raise ValueError(f"duplicate legacy provenance card_id: {card_id}")
        provenance_by_id[card_id] = item
    if set(provenance_by_id) != ids:
        raise ValueError("legacy card/provenance ids differ")
    return provenance_by_id


def _legacy_usage_boundary(
    card: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    card_type = str(card["card_type"])
    if card_type == "bestseller":
        return (
            "仅表示历史目录中的款型线索，不代表实时热销、销量排行或市场份额；"
            "具体商品供给和受欢迎程度需另行核对。"
        )
    if card_type == "attribute":
        claim_scope = _validate_text(provenance.get("claim_scope"), "claim_scope")
        scope_parts = [part.strip() for part in claim_scope.split("；") if part.strip()]
        retained_scope = (
            "；".join(scope_parts[2:]) if len(scope_parts) > 2 else claim_scope
        )
        return (
            "仅表示历史样本的标题或属性出现分布，不代表市场占比；"
            f"{retained_scope}。具体功效、兼容、适配或安全结论仍需逐商品核对。"
        )
    if card_type == "price_range":
        return (
            "仅表示人民币历史目录挂牌或规格价样本区间，不代表成交价或实时价格；"
            "当前价格与具体 SKU 价格以实时商品搜索为准。"
        )
    raise ValueError(f"unsupported legacy card type: {card_type}")


def _legacy_eval_query(card: dict[str, Any], provenance: dict[str, Any]) -> str:
    category = str(card["category"])
    card_type = str(card["card_type"])
    if card_type == "bestseller":
        first_form = str(card["summary"]).split("：", 1)[-1].split(" / ", 1)[0]
        return f"{category}历史目录中与{first_form}相近的款型线索有哪些？"
    if card_type == "attribute":
        attribute_name = _validate_text(
            provenance.get("attribute_name"), "attribute_name"
        )
        return f"{category}历史样本的{attribute_name}标题属性分布怎样？"
    if card_type == "price_range":
        return f"{category}历史人民币目录价格大致分为哪些区间？"
    raise ValueError(f"unsupported legacy card type: {card_type}")


def _runtime_markdown(card: dict[str, str]) -> str:
    return (
        f"# 品类：{card['category']}\n\n"
        f"类型：{card['card_type']}\n\n"
        f"正文：{card['summary']}\n\n"
        f"使用边界：{card['usage_boundary']}\n"
    )


def build(
    source_path: Path = DEFAULT_SOURCE,
    release_dir: Path = DEFAULT_RELEASE_DIR,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    legacy_cards_path: Path = DEFAULT_LEGACY_CARDS,
    legacy_provenance_path: Path = DEFAULT_LEGACY_PROVENANCE,
) -> dict[str, Any]:
    candidates = _read_jsonl(source_path)
    _validate_candidates(candidates)
    candidates = sorted(candidates, key=lambda row: str(row["candidate_id"]))
    legacy_cards = _read_jsonl(legacy_cards_path)
    legacy_provenance = _read_jsonl(legacy_provenance_path)
    legacy_provenance_by_id = _validate_legacy_inputs(legacy_cards, legacy_provenance)
    legacy_cards = sorted(legacy_cards, key=lambda row: str(row["card_id"]))

    approved_cards: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    eval_cases: list[dict[str, Any]] = []
    runtime_documents: dict[str, str] = {}
    for row in legacy_cards:
        card_id = str(row["card_id"])
        source_provenance = legacy_provenance_by_id[card_id]
        approved_card = {
            "card_id": card_id,
            "card_type": str(row["card_type"]),
            "category": str(row["category"]),
            "review_status": "approved",
            "summary": str(row["summary"]),
            "usage_boundary": _legacy_usage_boundary(row, source_provenance),
        }
        approved_cards.append(approved_card)
        provenance_item = dict(source_provenance)
        provenance_item.update(
            {
                "confidence": row["confidence"],
                "last_updated": row["last_updated"],
                "official_sources": [],
                "raw_evidence": row["raw_evidence"],
                "raw_evidence_status": "catalog_evidence_not_runtime",
                "verification_status": "catalog_evidence_reviewed",
            }
        )
        provenance.append(provenance_item)
        filename = f"{card_id}.md"
        runtime_documents[filename] = _runtime_markdown(approved_card)
        eval_cases.append(
            {
                "case_id": f"eval-{card_id}",
                "expected_card_id": card_id,
                "kind": str(row["card_type"]),
                "query": _legacy_eval_query(row, source_provenance),
                "relevant": [filename],
            }
        )

    for row in candidates:
        card_id = str(row["candidate_id"])
        approved_card = {
            "card_id": card_id,
            "card_type": str(row["candidate_type"]),
            "category": str(row["category"]),
            "review_status": "approved",
            "summary": str(row["summary"]),
            "usage_boundary": str(row["claim_scope"]),
        }
        if set(approved_card) != APPROVED_CARD_FIELDS:
            raise AssertionError("approved card schema changed")
        approved_cards.append(approved_card)
        provenance_item = {
            "applicability_scope": APPLICABILITY_SCOPES[card_id],
            "card_id": card_id,
            "confidence": float(row["confidence"]),
            "official_sources": OFFICIAL_SOURCES.get(card_id, []),
            "raw_evidence": row["raw_evidence"],
            "raw_evidence_status": (
                "legacy_unverified_not_runtime"
                if card_id in OFFICIAL_SOURCES
                else "catalog_evidence_not_runtime"
            ),
            "reviewed_at": REVIEWED_AT,
            "reviewer": "user_explicit_approval",
            "source_ref": str(row["source_ref"]),
            "source_reviewed_on": SOURCE_REVIEWED_ON,
            "verification_status": (
                "official_source_reviewed"
                if card_id in OFFICIAL_SOURCES
                else "catalog_evidence_reviewed"
            ),
        }
        if card_id in VERIFIED_EVIDENCE:
            provenance_item["verified_evidence"] = VERIFIED_EVIDENCE[card_id]
        provenance.append(provenance_item)
        filename = f"{card_id}.md"
        runtime_documents[filename] = _runtime_markdown(approved_card)
        eval_cases.append(
            {
                "case_id": f"eval-{card_id}",
                "expected_card_id": card_id,
                "kind": str(row["candidate_type"]),
                "query": EVAL_QUERIES[card_id],
                "relevant": [filename],
            }
        )

    release_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    expected_runtime_files = set(runtime_documents)
    for path in runtime_dir.glob("*.md"):
        if path.name not in expected_runtime_files:
            path.unlink()

    cards_path = release_dir / "approved_cards.jsonl"
    provenance_path = release_dir / "approved_card_provenance.jsonl"
    eval_path = release_dir / "retrieval_eval_cases.jsonl"
    _write_jsonl(cards_path, approved_cards)
    _write_jsonl(provenance_path, provenance)
    _write_jsonl(eval_path, eval_cases)
    for filename, content in runtime_documents.items():
        path = runtime_dir / filename
        path.write_text(content, encoding="utf-8")
        for term in FORBIDDEN_RUNTIME_TERMS:
            if term in content:
                raise AssertionError(f"forbidden runtime term {term!r}: {filename}")

    type_counts = Counter(str(row["card_type"]) for row in approved_cards)
    if dict(type_counts) != EXPECTED_RELEASE_TYPES:
        raise AssertionError(f"unexpected release type counts: {dict(type_counts)}")
    manifest: dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "counts": {
            "approved_cards": len(approved_cards),
            "approved_cards_by_type": dict(sorted(type_counts.items())),
            "legacy_cards": len(legacy_cards),
            "officially_sourced_cards": len(OFFICIAL_SOURCES),
            "retrieval_eval_cases": len(eval_cases),
            "runtime_markdown_documents": len(runtime_documents),
        },
        "generated_at": REVIEWED_AT,
        "inputs": {
            "approved_source": _file_entry(source_path, len(candidates)),
            "legacy_cards": _file_entry(legacy_cards_path, len(legacy_cards)),
            "legacy_provenance": _file_entry(
                legacy_provenance_path, len(legacy_provenance)
            ),
            "promotion_builder": _file_entry(Path(__file__)),
        },
        "outputs": {
            "approved_cards": _file_entry(cards_path, len(approved_cards)),
            "approved_card_provenance": _file_entry(provenance_path, len(provenance)),
            "retrieval_eval_cases": _file_entry(eval_path, len(eval_cases)),
            "runtime_documents": {
                filename: _file_entry(runtime_dir / filename)
                for filename in sorted(runtime_documents)
            },
        },
        "release_version": RELEASE_VERSION,
        "runtime": {
            "collection_default": "globex_category_kb_v1",
            "knowledge_directory": "knowledge/category-insight-v1",
            "markdown_contract": [
                "category",
                "card_type",
                "summary",
                "usage_boundary",
            ],
        },
    }
    manifest_path = release_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--legacy-cards", type=Path, default=DEFAULT_LEGACY_CARDS)
    parser.add_argument(
        "--legacy-provenance", type=Path, default=DEFAULT_LEGACY_PROVENANCE
    )
    args = parser.parse_args()
    manifest = build(
        args.source,
        args.release_dir,
        args.runtime_dir,
        args.legacy_cards,
        args.legacy_provenance,
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
