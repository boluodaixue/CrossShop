"""LLM-generated per-case rubric scoring for agent-flow query conversations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable
from globex_agent.eval.fact_guard import validate_final_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RULES_TEXT = (
    "系统运费/关税规则：寄往中国，单件基础运费 25 元；"
    "关税按商品小计超出免税额度的部分计算，个人物品免税额度内关税为 0。"
)

EVIDENCE_SCHEMA_VERSION = "judge-evidence-v2"
RUBRIC_SCHEMA_VERSION = "judge-rubric-v2"
JUDGE_RESULT_SCHEMA_VERSION = "judge-result-v2"
_EVIDENCE_MAX_BYTES = 120_000
_MAX_VARIANTS_PER_PRODUCT = 80
_MAX_CATEGORY_ITEMS = 24
_MAX_CATALOG_ITEMS = 1_200
_MAX_STRING_CHARS = 600
_MAX_TRANSCRIPT_CHARS = 40_000
_UNKNOWN = "unknown"

RUBRIC_GENERATOR_SYSTEM = (
    "你是电商 Agent 评测用例设计者。输入包含 query、Agent 对话和有界 evidence catalog。"
    "catalog 中每条证据都有唯一 evidence_id；你只能从 catalog 原样选择 evidence_refs，"
    "严禁编写路径、字段名、商品库推断或新的 evidence_id。unknown 不等于不存在。"
    "price_major=null 仍可能有 variant 价格；同一 item/evidence 的变体价格必须精确绑定。"
    "CategoryInsight 只支持品类 aggregate/reference，不证明具体 SKU、店铺或库存。"
    "P0 只放硬事实/安全，P1 放 query 满足度，P2 放表达。无法由 catalog 核验的规则不要生成。"
    "每条 criterion 必须有唯一 id、criterion、source_scope、evidence_refs、applies_if。"
    "只输出 JSON；evidence_refs 必须是 catalog evidence_id 的子集："
    '{"schema_version":"judge-rubric-v2","p0":[{"id":"p0-1","criterion":"...",'
    '"source_scope":"product|category|system|conversation|mixed",'
    '"evidence_refs":["product:..."],"applies_if":"..."}],"p1":[],"p2":[]}'
)

JUDGE_SYSTEM = (
    "你是严格但证据约束的电商 Agent 评测员。只使用本请求的 evidence catalog、对话和系统规则。"
    "unknown/未提供字段不等于不存在；只有与 catalog 明确矛盾才 fail，证据不足为 not_evaluable。"
    "price_major=null 不代表没有 variant price；变体必须在同一 item/evidence 内绑定。"
    "CategoryInsight 只证明 category aggregate/reference。"
    "必须逐等级覆盖 rubric 的全部 criterion id，恰好一次；不得缺失、重复或新增 id。"
    "结果 evidence_refs 只能选择 catalog 中的 evidence_id，不能写路径或自由编造 ID。"
    "P0 是硬事实/安全，P1 是 query 满足度，P2 是表达。只输出 JSON："
    '{"schema_version":"judge-result-v2","p0":[{"criterion_id":"p0-1",'
    '"status":"pass|fail|not_evaluable","reason":"...","evidence_refs":["product:..."]}],'
    '"p1":[],"p2":[]}'
)


class JudgeSchemaError(ValueError):
    """Provider JSON was valid JSON but not a usable rubric/result schema."""


class JudgeHTTPError(RuntimeError):
    """HTTP failure retaining safe response diagnostics for the report."""

    def __init__(
        self,
        status_code: int,
        body: str,
        headers: dict[str, str],
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.headers = headers
        super().__init__(f"HTTP {status_code}: {body}")


def render_rules() -> str:
    rates = ExchangeRateTable()
    rate_text = ", ".join(
        f"1 {currency} = {rate} CNY"
        for currency, rate in rates.rates_to_cny.items()
    )
    return f"{RULES_TEXT}\n系统汇率表：{rate_text}"


def _model() -> str:
    return (
        os.environ.get("EVAL_JUDGE_MODEL")
        or os.environ.get("LLM_JUDGE")
        or os.environ.get("LLM_MODEL")
        or "qwen-plus"
    )


def _base_url() -> str:
    return os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _bounded_text(value: object, limit: int = _MAX_STRING_CHARS) -> str:
    text = str(value) if value is not None else ""
    return text[:limit]


def _bounded_json_value(value: object, *, depth: int = 0) -> object:
    if isinstance(value, str):
        return _bounded_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= 2:
        return _bounded_text(value, 300)
    if isinstance(value, list):
        return [
            _bounded_json_value(item, depth=depth + 1)
            for item in value[:12]
        ]
    if isinstance(value, dict):
        return {
            _bounded_text(key, 80): _bounded_json_value(item, depth=depth + 1)
            for key, item in list(value.items())[:12]
        }
    return _bounded_text(value)


def _safe_response_text(value: str, limit: int = 1200) -> str:
    text = value.replace(_api_key(), "<REDACTED_KEY>") if _api_key() else value
    text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer <REDACTED>", text)
    return text[:limit]


def _response_headers(response: httpx.Response) -> dict[str, str]:
    keys = (
        "request-id",
        "x-request-id",
        "cf-ray",
        "server",
        "content-type",
        "retry-after",
    )
    return {key: response.headers[key] for key in keys if key in response.headers}


def _compact_options(options: object) -> list[dict[str, str | None]] | None:
    if not isinstance(options, list):
        return None
    compact: list[dict[str, str | None]] = []
    for option in options[:12]:
        if not isinstance(option, dict):
            continue
        compact.append(
            {
                key: _bounded_json_value(option.get(key))
                if option.get(key) is not None
                else None
                for key in ("code", "name", "value")
                if key in option
            }
        )
    return compact


def _compact_variant(variant: object) -> dict:
    if not isinstance(variant, dict):
        return {"display_name": _bounded_text(variant), "completeness": "unknown"}
    return {
        "variant_id": _bounded_text(variant.get("variant_id"))
        if variant.get("variant_id") is not None
        else _UNKNOWN,
        "display_name": _bounded_text(variant.get("display_name"))
        if variant.get("display_name") is not None
        else _UNKNOWN,
        "options": _compact_options(variant.get("options")),
        "price_major": variant.get("price_major", None),
        "currency": _bounded_text(variant.get("currency", "CNY")),
        "availability": _bounded_json_value(variant.get("availability", _UNKNOWN)),
    }


def _compact_product_fact(fact: dict) -> dict:
    raw_variants = fact.get("variants")
    variants = (
        [_compact_variant(variant) for variant in raw_variants[:_MAX_VARIANTS_PER_PRODUCT]]
        if isinstance(raw_variants, list)
        else None
    )
    raw_highlights = fact.get("highlights")
    highlights = (
        [_bounded_text(highlight) for highlight in raw_highlights[:8]]
        if isinstance(raw_highlights, list)
        else None
    )
    completeness = "complete"
    if raw_variants is None:
        completeness = "unknown"
    elif len(raw_variants) > _MAX_VARIANTS_PER_PRODUCT:
        completeness = "bounded_truncated"
    return {
        "item_id": _bounded_text(fact.get("item_id", _UNKNOWN)),
        "title": _bounded_text(fact.get("title", _UNKNOWN)),
        "category": _bounded_text(fact.get("category", _UNKNOWN)),
        "evidence_id": _bounded_text(fact.get("evidence_id", _UNKNOWN)),
        "content_hash": _bounded_text(fact.get("content_hash", _UNKNOWN)),
        "schema_version": _bounded_text(fact.get("schema_version", _UNKNOWN)),
        "price_source": _bounded_json_value(fact.get("price_source", _UNKNOWN)),
        "price_major": fact.get("price_major"),
        "price_min_major": fact.get("price_min_major"),
        "price_max_major": fact.get("price_max_major"),
        "currency": _bounded_text(fact.get("currency", "CNY")),
        "store": _bounded_text(
            fact.get("store", fact.get("shop", fact.get("store_name", _UNKNOWN)))
        ),
        "availability": _bounded_json_value(fact.get("availability", _UNKNOWN)),
        "highlights": highlights,
        "variants": variants,
        "completeness": {"variants": completeness},
    }


def _compact_category_result(result: dict) -> dict:
    raw_insights = result.get("insights")
    insights = raw_insights if isinstance(raw_insights, dict) else {}
    output: dict = {
        "category": _bounded_text(result.get("category", insights.get("category", _UNKNOWN))),
        "scope": _bounded_text(
            result.get("source_boundary", {}).get("scope", "category_aggregate_reference")
        ),
        "insights": {},
        "evidence_refs": result.get("evidence_refs", [])[:20]
        if isinstance(result.get("evidence_refs"), list)
        else [],
        "source_boundary": {
            key: _bounded_text(value)
            for key, value in (result.get("source_boundary") or {}).items()
            if key in {"scope", "price_tiers", "exclusions"}
        },
    }
    for key in ("category", "confidence"):
        if key in insights:
            output["insights"][key] = insights[key]
    for key in ("price_tiers", "attributes", "bestsellers"):
        value = insights.get(key)
        if isinstance(value, list):
            output["insights"][key] = [
                _bounded_json_value(item) for item in value[:_MAX_CATEGORY_ITEMS]
            ]
            if len(value) > _MAX_CATEGORY_ITEMS:
                output.setdefault("completeness", {})[key] = "bounded_truncated"
            else:
                output.setdefault("completeness", {})[key] = "complete"
    return output


def build_judge_evidence(item: dict) -> dict:
    """Build one bounded evidence object shared by rubric and Judge prompts."""

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "products": [_compact_product_fact(fact) for fact in item.get("facts", [])],
        "category_insights": [
            _compact_category_result(result)
            for result in item.get("category_results", [])
        ],
        "conversation": {
            "query": _bounded_text(item.get("query", _UNKNOWN), _MAX_TRANSCRIPT_CHARS),
            "answer": _bounded_text(
                item.get("transcript", _UNKNOWN), _MAX_TRANSCRIPT_CHARS
            ),
        },
        "field_policy": {
            "unknown_is_not_nonexistent": True,
            "variant_price_requires_same_item_and_evidence": True,
            "category_scope_is_reference_only": True,
        },
    }
    for _ in range(3):
        evidence.pop("evidence_catalog", None)
        evidence["evidence_catalog"] = build_evidence_catalog(evidence)
        encoded = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) <= _EVIDENCE_MAX_BYTES:
            break
        # Keep the evidence bounded without claiming the omitted tail is absent.
        for category in evidence["category_insights"]:
            for key in ("bestsellers", "attributes"):
                if isinstance(category.get("insights", {}).get(key), list):
                    category["insights"][key] = category["insights"][key][:8]
                    category.setdefault("completeness", {})[key] = "bounded_truncated"
        for product in evidence["products"]:
            if isinstance(product.get("variants"), list):
                product["variants"] = product["variants"][:32]
                product["completeness"]["variants"] = "bounded_truncated"
    return evidence


def render_evidence(evidence: dict) -> str:
    return (
        "## Evidence catalog (only these stable IDs may be cited; unknown != nonexistent)\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2)
    )


def _exposed_category_result(payload: dict) -> dict:
    """Keep parser input on the fields the CategoryInsight tool exposed."""

    raw_insights = payload.get("insights")
    insights: dict = {}
    if isinstance(raw_insights, dict):
        for field in ("category", "confidence"):
            if field in raw_insights:
                insights[field] = raw_insights[field]
        for field in ("price_tiers", "attributes", "bestsellers"):
            value = raw_insights.get(field)
            if isinstance(value, list):
                if field == "price_tiers":
                    insights[field] = [
                        {
                            key: tier[key]
                            for key in ("tier", "range_cny", "source_scope")
                            if key in tier
                        }
                        for tier in value
                        if isinstance(tier, dict)
                    ][: _MAX_CATEGORY_ITEMS]
                elif field == "attributes":
                    insights[field] = [
                        {
                            key: attribute[key]
                            for key in ("name", "distribution", "claim_scope")
                            if key in attribute
                        }
                        for attribute in value
                        if isinstance(attribute, dict)
                    ][: _MAX_CATEGORY_ITEMS]
                else:
                    insights[field] = [
                        {
                            key: bestseller[key]
                            for key in (
                                "name",
                                "typical_price_cny",
                                "why_popular",
                                "source_scope",
                            )
                            if key in bestseller
                        }
                        for bestseller in value
                        if isinstance(bestseller, dict)
                    ][: _MAX_CATEGORY_ITEMS]
    return {
        "category": payload.get("category", insights.get("category", "")),
        "insights": insights,
        "evidence_refs": [
            {
                field: ref[field]
                for field in (
                    "card_id",
                    "category",
                    "card_type",
                    "last_updated",
                    "confidence",
                )
                if field in ref
            }
            for ref in payload.get("evidence_refs", [])
            if isinstance(ref, dict)
        ][:20],
        "source_boundary": {
            field: payload.get("source_boundary", {}).get(field)
            for field in ("scope", "price_tiers", "exclusions")
            if isinstance(payload.get("source_boundary"), dict)
            and field in payload["source_boundary"]
        },
    }


def _api_key() -> str:
    return os.environ.get("LLM_API_KEY") or ""


def _stable_token(value: object, *, prefix: str = "x") -> str:
    text = str(value) if value is not None else _UNKNOWN
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    if safe and len(safe) <= 40:
        return safe
    import hashlib

    return f"{prefix}{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def _catalog_entry(
    evidence_id: str,
    *,
    source_scope: str,
    fact_type: str,
    value: object,
    item_id: object = None,
    snapshot_evidence_id: object = None,
    variant_id: object = None,
    summary: object = None,
) -> dict:
    return {
        "evidence_id": _bounded_text(evidence_id, 160),
        "source_scope": _bounded_text(source_scope, 80),
        "fact_type": _bounded_text(fact_type, 120),
        "value": _bounded_json_value(value),
        "summary": _bounded_text(summary if summary is not None else value, 300),
        "item_id": _bounded_text(item_id, 160) if item_id is not None else None,
        "snapshot_evidence_id": (
            _bounded_text(snapshot_evidence_id, 160)
            if snapshot_evidence_id is not None
            else None
        ),
        "variant_id": (
            _bounded_text(variant_id, 160) if variant_id is not None else None
        ),
    }


def _system_catalog_entries() -> list[dict]:
    rates = ExchangeRateTable().rates_to_cny
    entries = [
        _catalog_entry(
            "system:shipping:base-fee",
            source_scope="system",
            fact_type="shipping.base_fee_cny",
            value=25,
            summary="寄往中国单件基础运费25元",
        ),
        _catalog_entry(
            "system:tax:personal-duty",
            source_scope="system",
            fact_type="tax.personal_duty",
            value=0,
            summary="个人物品免税额度内关税为0",
        ),
        _catalog_entry(
            "system:tax:over-exemption",
            source_scope="system",
            fact_type="tax.over_exemption_rule",
            value="按商品小计超出免税额度的部分计算",
        ),
    ]
    entries.extend(
        _catalog_entry(
            f"system:exchange-rate:{_stable_token(currency, prefix='c')}",
            source_scope="system",
            fact_type="exchange_rate_to_cny",
            value=rate,
            summary=f"1 {currency} = {rate} CNY",
        )
        for currency, rate in sorted(rates.items())
    )
    return entries


def build_evidence_catalog(evidence: dict) -> list[dict]:
    """Create stable, bounded IDs from already-whitelisted exposed evidence."""

    entries: list[dict] = []
    for product in evidence.get("products") or []:
        item_id = product.get("item_id", _UNKNOWN)
        snapshot_id = product.get("evidence_id")
        product_identity = (
            item_id
            if item_id != _UNKNOWN
            else snapshot_id
            or json.dumps(product, ensure_ascii=False, sort_keys=True)
        )
        item_token = _stable_token(product_identity, prefix="p")
        base = f"product:{item_token}"
        fields = (
            ("title", "title"),
            ("category", "category"),
            ("price-major", "price_major"),
            ("price-min", "price_min_major"),
            ("price-max", "price_max_major"),
            ("store", "store"),
            ("availability", "availability"),
            ("currency", "currency"),
        )
        for suffix, key in fields:
            entries.append(
                _catalog_entry(
                    f"{base}:{suffix}",
                    source_scope="product",
                    fact_type=key,
                    value=product.get(key, _UNKNOWN),
                    item_id=item_id,
                    snapshot_evidence_id=snapshot_id,
                )
            )
        for highlight in product.get("highlights") or []:
            highlight_token = _stable_token(highlight, prefix="h")
            entries.append(
                _catalog_entry(
                    f"{base}:highlight:{highlight_token}",
                    source_scope="product",
                    fact_type="highlight",
                    value=highlight,
                    item_id=item_id,
                    snapshot_evidence_id=snapshot_id,
                    summary=highlight,
                )
            )
        for variant in product.get("variants") or []:
            variant_raw_id = variant.get("variant_id", _UNKNOWN)
            variant_identity = (
                variant_raw_id
                if variant_raw_id != _UNKNOWN
                else json.dumps(variant, ensure_ascii=False, sort_keys=True)
            )
            variant_token = _stable_token(variant_identity, prefix="v")
            variant_base = f"{base}:variant:{variant_token}"
            for suffix, key in (
                ("label", "display_name"),
                ("price", "price_major"),
                ("availability", "availability"),
            ):
                entries.append(
                    _catalog_entry(
                        f"{variant_base}:{suffix}",
                        source_scope="product",
                        fact_type=f"variant.{key}",
                        value=variant.get(key, _UNKNOWN),
                        item_id=item_id,
                        snapshot_evidence_id=snapshot_id,
                        variant_id=variant_raw_id,
                    )
                )
    for category in evidence.get("category_insights") or []:
        category_key = category.get("category", _UNKNOWN)
        refs = category.get("evidence_refs") or []
        ref_key = next(
            (ref.get("card_id") for ref in refs if isinstance(ref, dict) and ref.get("card_id")),
            json.dumps(category, ensure_ascii=False, sort_keys=True),
        )
        category_base = f"category:{_stable_token(ref_key, prefix='c')}"
        entries.append(
            _catalog_entry(
                f"{category_base}:category",
                source_scope="category",
                fact_type="category",
                value=category_key,
                summary=f"品类参考：{category_key}",
            )
        )
        insights = category.get("insights") or {}
        for tier in insights.get("price_tiers") or []:
            tier_key = "|".join(
                str(tier.get(key, _UNKNOWN))
                for key in ("tier", "range_cny", "source_scope")
            )
            entries.append(
                _catalog_entry(
                    (
                        f"{category_base}:price-tier:"
                    f"{_stable_token(tier_key, prefix='t')}"
                    ),
                    source_scope="category",
                    fact_type="price_tier",
                    value=tier,
                    summary=(
                        f"品类价格参考 {tier.get('tier', _UNKNOWN)} "
                        f"{tier.get('range_cny', _UNKNOWN)}"
                    ),
                )
            )
        for attribute in insights.get("attributes") or []:
            attribute_name = attribute.get("name", _UNKNOWN)
            attribute_key = json.dumps(attribute, ensure_ascii=False, sort_keys=True)
            entries.append(
                _catalog_entry(
                    f"{category_base}:attribute:{_stable_token(attribute_key, prefix='a')}",
                    source_scope="category",
                    fact_type="attribute",
                    value=attribute,
                    summary=f"品类属性参考 {attribute_name}",
                )
            )
        for bestseller in insights.get("bestsellers") or []:
            bestseller_name = bestseller.get("name", _UNKNOWN)
            bestseller_key = json.dumps(bestseller, ensure_ascii=False, sort_keys=True)
            entries.append(
                _catalog_entry(
                    f"{category_base}:bestseller:{_stable_token(bestseller_key, prefix='b')}",
                    source_scope="category",
                    fact_type="bestseller",
                    value=bestseller,
                    summary=f"品类常见商品参考 {bestseller_name}",
                )
            )
    entries.extend(_system_catalog_entries())
    for key, value in (
        ("query", evidence.get("conversation", {}).get("query", _UNKNOWN)),
        ("answer", evidence.get("conversation", {}).get("answer", _UNKNOWN)),
    ):
        entries.append(
            _catalog_entry(
                f"conversation:{key}",
                source_scope="conversation",
                fact_type=key,
                value=value,
                summary=value,
            )
        )
    unique: dict[str, dict] = {}
    for entry in entries:
        unique.setdefault(entry["evidence_id"], entry)
    return [unique[key] for key in sorted(unique)][: _MAX_CATALOG_ITEMS]


def _catalog_ids(evidence: dict) -> set[str]:
    return {
        entry["evidence_id"]
        for entry in evidence.get("evidence_catalog", [])
        if isinstance(entry, dict) and entry.get("evidence_id")
    }


def normalize_rubric(
    raw: dict,
    evidence: dict,
    *,
    strict_ids: bool = True,
) -> tuple[dict, list[str]]:
    """Normalize rubric and enforce the stable evidence-catalog ID contract."""

    if not isinstance(raw, dict):
        raise JudgeSchemaError("rubric root must be an object")
    normalized: dict = {"schema_version": RUBRIC_SCHEMA_VERSION}
    issues: list[str] = []
    for level in ("p0", "p1", "p2"):
        entries = raw.get(level, [])
        if not isinstance(entries, list):
            issues.append(f"{level} must be a list")
            normalized[level] = []
            continue
        output: list[dict] = []
        for index, entry in enumerate(entries, 1):
            if isinstance(entry, str):
                criterion = entry
                source_scope = "unknown"
                evidence_refs: list[str] = []
                applies_if = "always"
                criterion_id = f"{level}-{index}"
                if strict_ids:
                    issues.append(f"{level}-{index}: legacy criterion has no id/evidence_refs")
            elif isinstance(entry, dict):
                criterion = entry.get("criterion")
                source_scope = entry.get("source_scope", "unknown")
                criterion_id = entry.get("id")
                raw_refs = entry.get("evidence_refs", [])
                evidence_refs = (
                    [str(ref) for ref in raw_refs]
                    if isinstance(raw_refs, list)
                    else [str(raw_refs)]
                )
                applies_if = entry.get("applies_if", "always")
                if not criterion_id:
                    issues.append(f"{level}-{index}: missing criterion id")
                    criterion_id = f"{level}-{index}"
            else:
                issues.append(f"{level}-{index}: criterion must be string/object")
                continue
            if not isinstance(criterion, str) or not criterion.strip():
                issues.append(f"{level}-{index}: missing criterion")
                continue
            unknown_refs = [ref for ref in evidence_refs if ref not in _catalog_ids(evidence)]
            if strict_ids and not evidence_refs:
                issues.append(f"{criterion_id}: missing evidence_refs")
            if unknown_refs:
                issues.append(f"{criterion_id}: unknown evidence_refs={unknown_refs}")
            output.append(
                {
                    "id": str(criterion_id),
                    "criterion": criterion[:1200],
                    "source_scope": str(source_scope),
                    "evidence_refs": evidence_refs[:20],
                    "applies_if": _bounded_text(applies_if, 500),
                }
            )
        normalized[level] = output
    all_ids = [
        row["id"]
        for level in ("p0", "p1", "p2")
        for row in normalized[level]
    ]
    duplicate_ids = sorted({id_ for id_ in all_ids if all_ids.count(id_) > 1})
    if duplicate_ids:
        issues.append(f"duplicate criterion ids={duplicate_ids}")
    if not any(normalized[level] for level in ("p0", "p1", "p2")):
        issues.append("rubric has no criteria")
    return normalized, issues


def normalize_judge_result(
    raw: dict,
    *,
    rubric: dict | None = None,
    evidence: dict | None = None,
) -> dict:
    """Normalize results and, for new runs, enforce exact rubric ID coverage."""

    if not isinstance(raw, dict):
        raise JudgeSchemaError("judge result root must be an object")
    expected = {
        level: {
            str(row["id"])
            for row in (rubric or {}).get(level, [])
            if isinstance(row, dict) and row.get("id")
        }
        for level in ("p0", "p1", "p2")
    }
    criterion_text = {
        str(row["id"]): row.get("criterion", "")
        for level in ("p0", "p1", "p2")
        for row in (rubric or {}).get(level, [])
        if isinstance(row, dict) and row.get("id")
    }
    catalog_ids = _catalog_ids(evidence or {})
    normalized: dict = {"schema_version": JUDGE_RESULT_SCHEMA_VERSION}
    diagnostics: list[str] = []
    for level in ("p0", "p1", "p2"):
        if level not in raw and expected[level]:
            diagnostics.append(f"{level}: missing result level")
        entries = raw.get(level, [])
        if not isinstance(entries, list):
            diagnostics.append(f"{level}: result must be a list")
            entries = []
        output: list[dict] = []
        seen: list[str] = []
        for index, entry in enumerate(entries, 1):
            if not isinstance(entry, dict):
                diagnostics.append(f"{level}-{index}: result must be an object")
                continue
            raw_id = entry.get("criterion_id")
            if raw_id is None and rubric is None:
                raw_id = entry.get("id") or f"{level}-{index}"
            elif raw_id is None:
                raw_id = entry.get("id")
            if not raw_id:
                diagnostics.append(f"{level}-{index}: missing criterion_id")
                continue
            criterion_id = str(raw_id)
            seen.append(criterion_id)
            status = entry.get("status")
            if status is None and rubric is None and "pass" in entry:
                status = "pass" if entry.get("pass") is True else "fail"
            if status not in {"pass", "fail", "not_evaluable"}:
                diagnostics.append(f"{criterion_id}: invalid status")
                continue
            evidence_refs = entry.get("evidence_refs", [])
            if not isinstance(evidence_refs, list):
                diagnostics.append(f"{criterion_id}: evidence_refs must be a list")
                evidence_refs = []
            evidence_refs = [str(ref) for ref in evidence_refs]
            if rubric is not None and not evidence_refs:
                diagnostics.append(f"{criterion_id}: missing evidence_refs")
            unknown_refs = [ref for ref in evidence_refs if ref not in catalog_ids]
            if unknown_refs:
                diagnostics.append(f"{criterion_id}: unknown evidence_refs={unknown_refs}")
            output.append(
                {
                    "id": criterion_id,
                    "criterion": _bounded_text(
                        entry.get("criterion", criterion_text.get(criterion_id, "")),
                        1200,
                    ),
                    "status": status,
                    "pass": status == "pass",
                    "reason": _bounded_text(entry.get("reason", ""), 1600),
                    "evidence_refs": evidence_refs[:20],
                }
            )
        duplicates = sorted({id_ for id_ in seen if seen.count(id_) > 1})
        if duplicates:
            diagnostics.append(f"{level}: duplicate criterion_ids={duplicates}")
        if rubric is not None:
            missing = sorted(expected[level] - set(seen))
            extra = sorted(set(seen) - expected[level])
            if missing:
                diagnostics.append(f"{level}: missing criterion_ids={missing}")
            if extra:
                diagnostics.append(f"{level}: extra criterion_ids={extra}")
        normalized[level] = output
    if not any(normalized[level] for level in ("p0", "p1", "p2")):
        diagnostics.append("judge result has no criteria")
    if diagnostics:
        raise JudgeSchemaError("judge inconclusive: " + "; ".join(diagnostics))
    return normalized


def _has_not_evaluable(judged: dict) -> bool:
    return any(
        row.get("status") == "not_evaluable"
        for level in ("p0", "p1", "p2")
        for row in judged.get(level, [])
    )


def _format_error(error: Exception) -> str:
    if isinstance(error, JudgeHTTPError):
        metadata = ", ".join(f"{key}={value}" for key, value in error.headers.items())
        return f"HTTP {error.status_code}; body={error.body!r}; {metadata}".rstrip("; ")
    return f"{type(error).__name__}: {error}"


async def _chat_json(
    client: httpx.AsyncClient,
    system: str,
    user: str,
    *,
    temperature: float = 0,
    request_timeout: float = 180,
    max_attempts: int = 6,
) -> dict:
    payload = {
        "model": _model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    last_error: Exception | None = None
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            response = await client.post(
                f"{_base_url().rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {_api_key()}"},
                json=payload,
                timeout=request_timeout,
            )
            if response.is_error:
                raise JudgeHTTPError(
                    response.status_code,
                    _safe_response_text(response.text),
                    _response_headers(response),
                )
            try:
                content = response.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as err:
                raise JudgeSchemaError(
                    f"provider JSON content is not a JSON object: "
                    f"{_safe_response_text(response.text, 600)}"
                ) from err
            if not isinstance(parsed, dict):
                raise JudgeSchemaError("provider JSON root must be an object")
            return parsed
        except Exception as err:  # noqa: BLE001 - transient judge failures retried
            if attempt + 1 >= attempts:
                raise
            last_error = err
            print(f"   LLM 调用第 {attempt + 1} 次失败，稍后重试：{err}", flush=True)
            await asyncio.sleep(min(10 * (2**attempt), 30))
    raise last_error if last_error else RuntimeError("LLM 重试耗尽")


def parse_conversation(path: Path) -> dict:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    turns = [row for row in rows if row.get("kind") == "turn"]
    transcript_lines = [
        f"[{row['role']}] {row['content']}"
        for row in turns
        if row.get("content")
    ]
    facts: dict[str, dict] = {}
    snapshots: dict[str, dict] = {}
    category_results: list[dict] = []
    for row in rows:
        if row.get("kind") != "event" or row.get("type") != "tool.result":
            continue
        payload = row.get("payload") or {}
        if payload.get("tool") == "product_search_tool":
            for snapshot in payload.get("evidence_snapshots") or []:
                item_id = snapshot.get("item_id")
                if item_id:
                    snapshots[item_id] = snapshot
            for hit in payload.get("hits") or []:
                item_id = hit.get("item_id")
                if not item_id:
                    continue
                exposed = snapshots.get(item_id, {}).get("exposed_facts")
                fact = dict(exposed) if isinstance(exposed, dict) else {
                    "item_id": item_id,
                    "title": hit.get("title", ""),
                    "category": hit.get("category", ""),
                    "price_major": hit.get("price_major"),
                    "price_min_major": hit.get("price_min_major"),
                    "price_max_major": hit.get("price_max_major"),
                    "price": hit.get("price_major"),
                    "currency": hit.get("currency", "CNY"),
                    "variants": hit.get("variants", []),
                    "landed_price": hit.get("landed_price"),
                    "availability": hit.get("availability"),
                }
                # Price ranges and the evidence identity are card-bound fields;
                # retain them while all audit facts come from exposed_facts.
                fact.setdefault("item_id", item_id)
                fact.setdefault("price", fact.get("price_major"))
                fact.setdefault("currency", hit.get("currency", "CNY"))
                fact.setdefault("price_min_major", hit.get("price_min_major"))
                fact.setdefault("price_max_major", hit.get("price_max_major"))
                fact.setdefault("landed_price", hit.get("landed_price"))
                fact["evidence_id"] = snapshots.get(item_id, {}).get("evidence_id")
                fact["content_hash"] = snapshots.get(item_id, {}).get("content_hash")
                fact["schema_version"] = snapshots.get(item_id, {}).get("schema_version")
                facts[item_id] = fact
        elif payload.get("tool") == "category_insight_tool":
            category_results.append(_exposed_category_result(payload))

    query = next(
        (row["content"] for row in turns if row.get("role") == "buyer"),
        "",
    )
    transcript = "\n\n".join(transcript_lines)
    agent_turns = [row["content"] for row in turns if row.get("role") == "agent"]
    final_agent_text = agent_turns[-1] if agent_turns else ""
    fact_violations = [
        {
            "code": violation.code,
            "message": violation.message,
            "excerpt": violation.excerpt,
        }
        for violation in validate_final_response(
            final_agent_text,
            product_facts=facts.values(),
            category_insights=category_results,
        )
    ]
    return {
        "source": path.stem,
        "query": query,
        "transcript": transcript,
        "facts": list(facts.values()),
        "category_results": category_results,
        "fact_violations": fact_violations,
    }


def render_facts(facts: list[dict]) -> str:
    # Keep this historical helper aligned with structured evidence.  A null
    # major price must not hide exposed variant prices.
    return render_evidence(
        build_judge_evidence({"facts": facts, "category_results": []})
    )


def render_category_results(results: list[dict]) -> str:
    if not results:
        return "品类知识工具结果：无"
    lines = ["品类知识工具结果（仅 category aggregate/reference）："]
    for result in results:
        lines.append(
            json.dumps(
                {
                    "category": result.get("category"),
                    "insights": result.get("insights", {}),
                    "evidence_refs": result.get("evidence_refs", []),
                    "source_boundary": result.get("source_boundary", {}),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


async def generate_rubric(
    client: httpx.AsyncClient,
    item: dict,
    *,
    evidence: dict | None = None,
    request_timeout: float,
    max_attempts: int,
) -> dict:
    evidence = evidence or build_judge_evidence(item)
    user = (
        f"## Query\n{item['query']}\n\n"
        f"## Agent answer\n{item['transcript']}\n\n"
        f"{render_evidence(evidence)}\n\n"
        f"{render_rules()}"
    )
    raw = await _chat_json(
        client,
        RUBRIC_GENERATOR_SYSTEM,
        user,
        temperature=0.3,
        request_timeout=request_timeout,
        max_attempts=max_attempts,
    )
    normalized, issues = normalize_rubric(raw, evidence, strict_ids=True)
    if issues:
        raise JudgeSchemaError("rubric inconclusive: " + "; ".join(issues))
    return normalized


async def call_judge(
    client: httpx.AsyncClient,
    item: dict,
    rubric: dict,
    *,
    evidence: dict | None = None,
    request_timeout: float,
    max_attempts: int,
) -> dict:
    evidence = evidence or build_judge_evidence(item)
    user = (
        f"{render_evidence(evidence)}\n\n"
        f"{render_rules()}\n\n"
        f"## 对话记录\n{item['transcript']}\n\n"
        f"## 评分细则\n{json.dumps(rubric, ensure_ascii=False, indent=2)}"
    )
    raw = await _chat_json(
        client,
        JUDGE_SYSTEM,
        user,
        request_timeout=request_timeout,
        max_attempts=max_attempts,
    )
    return normalize_judge_result(raw, rubric=rubric, evidence=evidence)


def score_case(
    judged: dict,
    fact_violations: list[dict] | None = None,
    *,
    rubric: dict | None = None,
    evidence: dict | None = None,
) -> tuple[float | None, bool | None]:
    judged = normalize_judge_result(judged, rubric=rubric, evidence=evidence)
    if _has_not_evaluable(judged):
        return None, False

    def ratio(level: list) -> float:
        return (
            sum(1 for rule in level if rule.get("status") == "pass") / len(level)
            if level
            else 1.0
        )

    p0_rules = judged.get("p0", [])
    p0 = ratio(p0_rules)
    p1 = ratio(judged.get("p1", []))
    p2 = ratio(judged.get("p2", []))
    if rubric is not None and not any(
        rubric.get(level) for level in ("p0", "p1", "p2")
    ):
        raise JudgeSchemaError("rubric has no applicable criteria")
    weighted = 0.5 * p0 + 0.35 * p1 + 0.15 * p2
    if not p0_rules:
        p0_pass: bool | None = False if fact_violations else None
    else:
        p0_pass = p0 == 1.0 and not fact_violations
    return round(weighted, 3), p0_pass


def render_report(results: list[dict]) -> str:
    completed = [item for item in results if item.get("judge_status") == "completed"]
    inconclusive = [item for item in results if item.get("judge_status") == "inconclusive"]
    external_blocked = [
        item
        for item in results
        if item.get("judge_status") in {"external_blocked", "not_run_external_blocked"}
    ]
    deterministic_only = not any(item.get("judge_status") for item in results)
    if deterministic_only:
        pass_count = sum(1 for item in results if not item.get("fact_violations"))
    else:
        pass_count = sum(
            1
            for item in completed
            if item.get("p0_pass") is not False and item["score"] >= 0.7
        )
    avg = (
        round(sum(item["score"] for item in completed) / len(completed), 3)
        if completed
        else 0
    )
    lines = [
        (
            f"# Flow query 确定性事实门禁评测（{datetime.now().strftime('%Y-%m-%d %H:%M')}）"
            if deterministic_only
            else f"# Flow query LLM-as-Judge 评测（{datetime.now().strftime('%Y-%m-%d %H:%M')}）"
        ),
        "",
        (
            f"总览：{pass_count}/{len(results)} PASS，确定性 P0 总数 "
            f"{sum(len(item.get('fact_violations', [])) for item in results)}"
            if deterministic_only
            else f"总览：Judge 完成 {len(completed)}/{len(results)}，"
            f"完成项 PASS {pass_count}/{len(completed)}，平均分 {avg}；"
            f"inconclusive {len(inconclusive)} 项；外部阻塞/未运行 {len(external_blocked)} 项"
        ),
        "",
    ]
    for item in results:
        snapshot_count = len(item.get("facts", []))
        snapshot_refs = sum(
            bool(fact.get("evidence_id") and fact.get("content_hash"))
            for fact in item.get("facts", [])
        )
        category_tier_count = sum(
            len(result.get("insights", {}).get("price_tiers", []))
            for result in item.get("category_results", [])
        )
        if item.get("judge_status") == "completed":
            verdict = (
                "PASS"
                if item.get("p0_pass") is not False and item["score"] >= 0.7
                else "FAIL"
            )
            score_text = str(item["score"])
        elif item.get("judge_status") == "inconclusive":
            verdict = "inconclusive"
            score_text = "N/A"
        elif item.get("judge_status"):
            verdict = item["judge_status"]
            score_text = "未完成"
        else:
            verdict = "PASS" if item["p0_pass"] and item["score"] >= 0.7 else "FAIL"
            score_text = str(item["score"])
        lines.extend(
            [
                f"## {item['source']}",
                "",
                f"- Query：{item['query']}",
                (
                    f"- Snapshot：{snapshot_count} 个；"
                    f"evidence_id/hash 完整 {snapshot_refs} 个"
                ),
                f"- CategoryInsight price_tiers：{category_tier_count} 条",
                f"- 确定性事实门禁：{'PASS' if not item.get('fact_violations') else 'FAIL'}",
                (
                    "- Judge P0：N/A（无适用 criterion）"
                    if item.get("p0_pass") is None
                    else f"- Judge P0：{'PASS' if item.get('p0_pass') else 'FAIL'}"
                ),
                f"- 结果：{verdict}（{score_text}）",
                *(
                    [
                        f"- Judge 错误（{item.get('error_stage', 'unknown')}）："
                        f"{item['judge_error']}"
                    ]
                    if item.get("judge_error")
                    else []
                ),
                "- 生成 rubric：",
                "```json",
                json.dumps(item["rubric"], ensure_ascii=False, indent=2),
                "```",
                "- Judge 判定：",
            ]
        )
        for level in ("p0", "p1", "p2"):
            for rule in item["judged"].get(level, []):
                mark = rule.get("status")
                if mark is None:
                    mark = "PASS" if rule.get("pass") else "FAIL"
                mark = str(mark).upper()
                lines.append(
                    f"  - [{level.upper()}][{mark}] {rule.get('criterion', '')}"
                    f"：{rule.get('reason', '')}"
                )
        for violation in item.get("fact_violations", []):
            lines.append(
                f"  - [P0][FAIL][{violation['code']}] {violation['message']}："
                f"{violation['excerpt']}"
            )
        lines.append("")
    return "\n".join(lines)


def save_progress(results: list[dict], path: Path) -> None:
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conversations",
        default=str(PROJECT_ROOT / "data" / "conversations"),
    )
    parser.add_argument(
        "--source",
        default=None,
        help="评测单个既有会话文件；用于将每个 Flow 隔离为独立进程",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--progress",
        default=str(PROJECT_ROOT / "eval" / "flow-rubric-progress.json"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prefix", default="flow")
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--total-timeout", type=float, default=600)
    args = parser.parse_args()

    conv_dir = Path(args.conversations)
    paths = (
        [Path(args.source)]
        if args.source
        else sorted(conv_dir.glob(f"{args.prefix}-*.jsonl"))
    )
    if args.limit:
        paths = paths[: args.limit]
    items = [parse_conversation(path) for path in paths]
    items = [item for item in items if item["query"]]
    if not items:
        raise SystemExit("没有可评测的 flow 会话")

    if args.deterministic_only:
        for item in items:
            item["rubric"] = {}
            item["judged"] = {}
            item["score"] = 1.0 if not item.get("fact_violations") else 0.0
            item["p0_pass"] = not item.get("fact_violations")
        report = render_report(items)
        report_path = (
            Path(args.out)
            if args.out
            else PROJECT_ROOT
            / "output"
            / "eval"
            / f"deterministic-p0-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"\n确定性 P0 报告已写入：{report_path}")
        return

    progress_path = Path(args.progress)
    done: dict[str, dict] = {}
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        done = {
            item["source"]: item
            for item in progress
            if item.get("score") is not None and item.get("judged") is not None
        }

    print(f"== 共 {len(items)} 条 flow query", flush=True)
    deadline = asyncio.get_running_loop().time() + args.total_timeout
    async with httpx.AsyncClient() as client:
        for index, item in enumerate(items, 1):
            print(f"  [{index}/{len(items)}] {item['source']} ...", flush=True)
            if item["source"] in done:
                item.update(done[item["source"]])
                print(f"     -> 已跳过（{item.get('score')}）", flush=True)
            else:
                remaining = deadline - asyncio.get_running_loop().time()
                stage = "rubric"
                item["rubric"] = {}
                item["judged"] = {}
                item["judge_error"] = None
                item["error_stage"] = None
                try:
                    if remaining <= 0:
                        raise TimeoutError("Judge total timeout exceeded")
                    evidence = build_judge_evidence(item)
                    item["rubric"] = await asyncio.wait_for(
                        generate_rubric(
                            client,
                            item,
                            evidence=evidence,
                            request_timeout=args.request_timeout,
                            max_attempts=args.max_attempts,
                        ),
                        timeout=remaining,
                    )
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError("Judge total timeout exceeded")
                    stage = "score"
                    item["judged"] = await asyncio.wait_for(
                        call_judge(
                            client,
                            item,
                            item["rubric"],
                            evidence=evidence,
                            request_timeout=args.request_timeout,
                            max_attempts=args.max_attempts,
                        ),
                        timeout=remaining,
                    )
                    item["score"], item["p0_pass"] = score_case(
                        item["judged"],
                        item.get("fact_violations", []),
                        rubric=item["rubric"],
                        evidence=evidence,
                    )
                    item["judge_status"] = (
                        "inconclusive" if item["score"] is None else "completed"
                    )
                    print(
                        f"     -> {item['judge_status']}（{item['score']}）",
                        flush=True,
                    )
                except JudgeSchemaError as err:
                    item["judge_status"] = "inconclusive"
                    item["error_stage"] = stage
                    item["judge_error"] = _format_error(err)
                    item["score"] = None
                    item["p0_pass"] = False
                    print(f"     -> 评测不可判定：{item['judge_error']}", flush=True)
                except Exception as err:  # noqa: BLE001 - report external blockage
                    item["judge_status"] = "external_blocked"
                    item["error_stage"] = stage
                    item["judge_error"] = _format_error(err)
                    item["score"] = None
                    item["p0_pass"] = False
                    print(f"     -> 外部阻塞：{item['judge_error']}", flush=True)
            save_progress(items, progress_path)

    report = render_report(items)
    report_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "eval" / f"flow-rubric-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{report_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
