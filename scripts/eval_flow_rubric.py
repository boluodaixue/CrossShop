"""Two-stage offline Rubric Generator + Final LLM-as-Judge for online flows.

The Rubric Generator sees only the query and pre-execution flow contract.  The
Final Judge receives the generated rubric plus compact user-visible evidence
and execution/gate summaries.  Complete online artifacts remain in the local
report for audit, but are never sent to either offline model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from globex_agent.application.evidence_verification import sanitize_evidence_judge_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
_P2_DIMENSIONS = ("需求覆盖度", "场景洞察力", "决策建议价值")
_ALLOWED_EVIDENCE_SOURCES = {
    "query",
    "final_text",
    "final_cards",
    "execution.flow_type",
    "execution.tool_sequence",
    "execution.tool_counts",
    "execution.generation_attempts",
    "execution.final_result_present",
    "online_gate.verification_status",
    "online_gate.fact_guard_passed",
    "online_gate.evidence_judge_status",
}
_INVISIBLE_TOPICS = ("现实物流", "用户满意", "实际收货", "真实配送", "未来销量")
_P1_USER_REQUIREMENT_TERMS = (
    "预算",
    "防水",
    "性别",
    "颜色",
    "材质",
    "尺寸",
    "价格",
    "需求",
    "满足",
    "推荐",
    "商品",
)
_P1_INVALID_STATUS_TERMS = ("verified", "passed", "已验证", "通过")

RUBRIC_GENERATOR_SYSTEM = (
    "你负责在执行前生成评分规则，只生成规则，不评分，不读取最终回答或实际工具结果。\n"
    "输出契约：只输出 JSON {\"p0\":[],\"p1\":[],\"p2\":[]}，不得增加字段；"
    "P0/P1 每项为 {id,criterion,evidence_sources}，P2 每项为 "
    "{id,dimension,requirements,expectation,evidence_sources}。\n\n"
    "P0 业务红线：只写能由 query、final_text、final_cards 直接判断的严重越界、危险/违禁建议、"
    "内部协议或字段泄露等硬失败；没有明确红线时可为空。\n\n"
    "P1 只写 flow_expectations 可机械核验的执行规则：required tools、声明顺序、单工具/总调用上限、"
    "生成尝试次数、final 是否存在，以及 online gate。允许示例：必须调用 product_search_tool；"
    "工具调用次数不超过 2；verification_status 为 supported；fact_guard_passed 为 true。"
    "P1 的 evidence_sources 只能是 execution.* 或 online_gate.*，不得使用 query、final_text、"
    "final_cards。禁止示例：预算、防水、颜色、材质、尺寸、需求是否满足、商品是否合适、推荐质量；"
    "这些全部属于 P2。online gate 成功值只允许 supported/true，不得写 verified/passed。\n\n"
    "P2 必须恰好有三个 dimension：需求覆盖度、场景洞察力、决策建议价值；每项 requirements 必须是"
    "非空字符串列表。P2 必须 evidence-bounded：requirements/expectation 只能要求 query、"
    "final_text、final_cards 实际提供且可验证的内容，不得预设最终证据包含任何外部品类知识。"
    "每条 expectation 必须是条件式：若 final_text/final_cards 提供相关事实，则评价是否正确使用；"
    "若证据缺失，requirements 应"
    "改写为可评价行为，例如说明证据限制、明确无合格结果、避免以不合格商品替代、给出必要澄清；不得留空。"
    "决策建议价值不等于必须提供下一步、产品对比或额外建议：只有 query 明确需要且证据支持时才"
    "评价这些。"
    "没有合格商品证据时，诚实零命中且不推荐不合格替代品可以是高价值决策，不因没有规格、对比、价格或"
    "下一步而扣分。场景洞察力同理，不得无条件要求防水等级、照射范围或市场常识；只评价限制说明和避免臆测。"
    "不得因缺失未提供的知识本身扣分。"
    "不得评判现实物流、用户满意、实际收货、真实配送或未来销量。"
    "证据源只能从 available_evidence.paths 选择。"
)

FINAL_JUDGE_SYSTEM = (
    "你是最终评分器。只依据输入中的 query、rubric、final、execution、online_gate 评分，"
    "不补充外部事实，不自行计算总分，不扩展 rubric 的评分项。"
    "selection、variant、卡片同源、币种/目的国/数量、报价算术和"
    "零命中选择等后端门禁只读取 online_gate，不重复判断。\n"
    "P0 每项输出 triggered 布尔值和 reason；P1 每项输出 violated 布尔值和 reason；"
    "P2 每项输出 1-5 整数 score 和 reason。"
    "P2 锚点：5=完全满足且证据清楚；4=基本满足仅轻微遗漏；3=核心满足但有明显缺口；"
    "2=少量满足或建议价值弱；1=未满足、冲突或不可用。需求覆盖度只评显式/隐含需求与最终可见事实；"
    "场景洞察力评证据支持的场景取舍；决策建议价值评 rubric 条件要求的可执行性。不要把它解释成必须有"
    "下一步、产品对比或额外建议；只有 query 明确需要且证据支持时才评价这些。zero-hit 的真实无结果、"
    "明确限制且不推荐不合格替代品，可以获得高分。"
    "不要因同一问题跨 P1/P2 重复扣分。\n"
    "评分必须 evidence-bounded：输入没有提供品类知识、常见类型、市场价、规格或物流时，"
    "不得因回答没有"
    "补充或编造这些知识而扣分；只评价回答是否说明限制并避免臆测，只有 rubric 条件要求时才评价"
    "澄清/下一步。\n"
    "必须覆盖 rubric 每一个 id，只输出 JSON："
    "{\"p0_results\":[{\"rubric_id\":\"...\",\"triggered\":false,\"reason\":\"...\"}],"
    "\"p1_results\":[{\"rubric_id\":\"...\",\"violated\":false,\"reason\":\"...\"}],"
    "\"p2_results\":[{\"rubric_id\":\"...\",\"score\":1,\"reason\":\"...\"}],"
    "\"reason\":\"...\"}"
)


def _model() -> str:
    return os.environ.get("EVAL_FINAL_JUDGE_MODEL") or "deepseek-v4-flash"


def _rubric_model() -> str:
    return os.environ.get("EVAL_RUBRIC_GENERATOR_MODEL") or _model()


def _pass_threshold() -> float:
    raw = os.environ.get("EVAL_PASS_THRESHOLD", "0.85")
    value = float(raw)
    if not 0 <= value <= 1:
        raise ValueError("EVAL_PASS_THRESHOLD must be between 0 and 1")
    return value


def _base_url() -> str:
    return os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def _api_key() -> str:
    return os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def parse_conversation(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON flow-result file without inventing missing evidence."""

    path = Path(path)
    if path.suffix == ".jsonl":
        return _load_json_lines(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("results") or payload.get("flows") or []
        return [row for row in rows if isinstance(row, dict)]
    return []


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _sanitize_external(value: Any, *, key: str = "") -> Any:
    """Keep complete evidence while removing PII, tokens, and embeddings."""
    return sanitize_evidence_judge_payload(value, key=key)


def _tool_events(item: dict[str, Any]) -> list[dict[str, Any]]:
    events = item.get("events")
    return (
        [event for event in events if isinstance(event, dict)]
        if isinstance(events, list)
        else []
    )


def _tool_outputs(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the exact model-visible result, never audit-only fields."""

    outputs: list[dict[str, Any]] = []
    for event in _tool_events(item):
        if event.get("type") != "tool.result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if "model_output" not in payload:
            continue
        output = {
            key: payload[key]
            for key in ("tool", "tool_call_id", "model_output")
            if key in payload
        }
        output["model_output"] = _json_value(output["model_output"])
        outputs.append(_sanitize_external(output))
    return outputs


def _visible_payload(output: dict[str, Any]) -> dict[str, Any]:
    value = output.get("model_output")
    return value if isinstance(value, dict) else {}


def _search_contexts(item: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    """Bind each returned item to its matching product-search invocation."""

    by_call: dict[str, dict[str, str | None]] = {}
    for event in _tool_events(item):
        if event.get("type") != "tool.invoke":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("tool") != "product_search_tool":
            continue
        call_id = payload.get("tool_call_id")
        args = _json_value(payload.get("args"))
        if not call_id or not isinstance(args, dict):
            continue
        ship_to = args.get("ship_to")
        currency = args.get("target_currency")
        by_call[str(call_id)] = {
            "ship_to": str(ship_to).strip().upper() if ship_to else None,
            "target_currency": str(currency).strip().upper() if currency else None,
        }

    by_item: dict[str, dict[str, str | None]] = {}
    for output in _tool_outputs(item):
        if output.get("tool") != "product_search_tool":
            continue
        model_output = _visible_payload(output)
        context = by_call.get(str(output.get("tool_call_id")))
        if context is None:
            continue
        for hit in model_output.get("hits", []):
            if isinstance(hit, dict) and hit.get("item_id"):
                by_item.setdefault(str(hit["item_id"]), context)
    return by_item


def _called_tools(item: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    for event in _tool_events(item):
        if event.get("type") not in {"tool.invoke", "tool.result"}:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and payload.get("tool"):
            name = str(payload["tool"])
            if name not in tools:
                tools.append(name)
    return tools


def _latest_verification(item: dict[str, Any]) -> dict[str, Any] | None:
    latest = None
    for event in _tool_events(item):
        if event.get("type") == "evidence.verify" and isinstance(
            event.get("payload"), dict
        ):
            latest = event["payload"]
    return latest


def _final_schema(item: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    final = item.get("final_result")
    if not isinstance(final, dict):
        final = {
            "text": item.get("final_text"),
            "recommended_cards": item.get("recommended_cards"),
            "verification_status": item.get("verification_status"),
        }
    text = final.get("text")
    cards = final.get("recommended_cards")
    status = final.get("verification_status")
    valid = isinstance(text, str) and bool(text.strip()) and isinstance(cards, list) and status in {
        "supported",
        "unsupported",
        "unavailable",
    }
    return valid, final


def _flow_summary(item: dict[str, Any]) -> dict[str, Any]:
    valid_schema, final = _final_schema(item)
    events = _tool_events(item)
    verification = _latest_verification(item)
    outputs = _tool_outputs(item)
    cards = (
        final.get("recommended_cards", [])
        if isinstance(final.get("recommended_cards"), list)
        else []
    )
    product_hits = [
        _visible_payload(output).get("hits")
        for output in outputs
        if output.get("tool") == "product_search_tool"
        and isinstance(_visible_payload(output).get("hits"), list)
    ]
    product_hit = any(bool(hits) for hits in product_hits)
    rag_hit = any(
        output.get("tool") == "category_insight_tool"
        and isinstance(_visible_payload(output).get("insights"), dict)
        for output in outputs
    )
    product_item_ids = []
    for hits in product_hits:
        for hit in hits or []:
            if isinstance(hit, dict) and hit.get("item_id") not in product_item_ids:
                product_item_ids.append(hit.get("item_id"))
    rag_card_ids = _visible_card_ids(outputs)
    quote_diagnostic = _quote_diagnostic(cards, _search_contexts(item))
    expected_observed = _expected_observed(
        item,
        {
            "rag_hit": rag_hit,
            "rag_card_ids": rag_card_ids,
            "product_hit": product_hit,
            "product_item_ids": product_item_ids,
            "tools": _called_tools(item),
        },
    )
    gate_status = str(final.get("verification_status") or "unavailable")
    gate_trustworthy = bool(verification) and gate_status != "unavailable"
    event_types = {
        str(event.get("type"))
        for event in events
        if isinstance(event, dict) and event.get("type")
    }
    evidence_complete = (
        bool(events)
        and bool(outputs)
        and "evidence.verify" in event_types
        and "final.result" in event_types
        and not item.get("event_capture_error")
    )
    return {
        "flow_valid": valid_schema and evidence_complete,
        "gate_trustworthy": gate_trustworthy,
        "rag_hit": rag_hit,
        "rag_card_ids": rag_card_ids,
        "product_hit": product_hit,
        "product_item_ids": product_item_ids,
        "tools": _called_tools(item),
        "final_selection_count": len(cards),
        "final_item_ids": [card.get("item_id") for card in cards if isinstance(card, dict)],
        "final_selections": [
            {
                "item_id": card.get("item_id"),
                "variant_id": card.get("selected_variant_id"),
            }
            for card in cards
            if isinstance(card, dict)
        ],
        "cards_have_quotes": quote_diagnostic["status"]
        in {"supported", "not_requested", "not_applicable"},
        "quote_diagnostic": quote_diagnostic,
        "online_fact_guard": (
            (verification or {}).get("fact_guard_errors", []) if verification else None
        ),
        "online_evidence_judge": (
            (verification or {}).get("verification_status") if verification else None
        ),
        "verification_status": gate_status,
        "final": _sanitize_external(final),
        "tool_outputs": outputs,
        "expected_observed": expected_observed,
    }


def _visible_card_ids(outputs: list[dict[str, Any]]) -> list[str]:
    card_ids: list[str] = []
    for output in outputs:
        if output.get("tool") != "category_insight_tool":
            continue
        value = _visible_payload(output)
        for card_id in _walk_key_values(value, "card_id"):
            if isinstance(card_id, str) and card_id not in card_ids:
                card_ids.append(card_id)
    return card_ids


def _walk_key_values(value: Any, wanted_key: str):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == wanted_key:
                yield child
            yield from _walk_key_values(child, wanted_key)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child, wanted_key)


def _quote_diagnostic(
    cards: list[Any], contexts: dict[str, dict[str, str | None]]
) -> dict[str, Any]:
    """Report quote coverage from search args, without becoming an offline gate."""

    if not cards:
        return {"status": "not_applicable", "reasons": []}
    reasons: list[str] = []
    requested = False
    for card in cards:
        if not isinstance(card, dict):
            reasons.append("final card is not an object")
            continue
        item_id = str(card.get("item_id", ""))
        context = contexts.get(item_id)
        if context is None:
            reasons.append(f"missing product-search context: {item_id}")
            continue
        ship_to = context.get("ship_to")
        currency = context.get("target_currency")
        variants = card.get("variants") or []
        quotes = []
        if isinstance(variants, list) and variants:
            if card.get("landed_price") is not None:
                reasons.append(f"variant card has top-level quote: {item_id}")
            for variant in variants:
                if not isinstance(variant, dict):
                    reasons.append(f"variant is not an object: {item_id}")
                    continue
                if variant.get("landed_price") is not None:
                    quotes.append(variant["landed_price"])
                if (
                    ship_to
                    and variant.get("price_major") is not None
                    and str(variant.get("availability", "")).casefold()
                    in {"available", "in_stock", "可售"}
                    and not isinstance(variant.get("landed_price"), dict)
                ):
                    reasons.append(f"missing variant quote: {item_id}/{variant.get('variant_id')}")
        else:
            if card.get("landed_price") is not None:
                quotes.append(card.get("landed_price"))
            if (
                ship_to
                and card.get("price_major") is not None
                and str(card.get("availability", "")).casefold()
                in {"available", "in_stock", "可售"}
                and not isinstance(card.get("landed_price"), dict)
            ):
                reasons.append(f"missing top-level quote: {item_id}")
        if not ship_to:
            if quotes:
                reasons.append(f"quote returned without requested ship_to: {item_id}")
            continue
        requested = True
        for quote in quotes:
            if not isinstance(quote, dict):
                reasons.append(f"quote is not an object: {item_id}")
                continue
            if str(quote.get("ship_to", "")).upper() != str(ship_to).upper():
                reasons.append(f"quote destination mismatch: {item_id}")
            if currency and str(quote.get("currency", "")).upper() != str(currency).upper():
                reasons.append(f"quote currency mismatch: {item_id}")
            if quote.get("quantity") != 1:
                reasons.append(f"quote quantity mismatch: {item_id}")
            if "unavailable_reason" not in quote:
                try:
                    total = round(
                        float(quote["subtotal_major"])
                        + float(quote["freight_major"])
                        + float(quote["tariff_major"]),
                        2,
                    )
                    if total != round(float(quote["landed_total_major"]), 2):
                        reasons.append(f"quote arithmetic mismatch: {item_id}")
                except (KeyError, TypeError, ValueError):
                    reasons.append(f"quote arithmetic unavailable: {item_id}")
    if reasons:
        return {"status": "unsupported", "requested": requested, "reasons": reasons}
    return {
        "status": "supported" if requested else "not_requested",
        "requested": requested,
        "reasons": [],
    }


def _cards_have_quotes(cards: list[Any]) -> bool:
    """Compatibility helper for callers of the old report utility."""

    return not cards


def _expected_observed(item: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    knowledge = item.get("knowledge_expectation")
    product = item.get("product_expectation")
    tools = item.get("tool_expectation")
    knowledge = knowledge if isinstance(knowledge, dict) else {}
    product = product if isinstance(product, dict) else {}
    tools = tools if isinstance(tools, dict) else {}
    return {
        "expected": {
            "rag_hit": knowledge.get("rag_hit"),
            "rag_card_ids": knowledge.get("card_ids", []),
            "product_hit": product.get("product_hit"),
            "product_item_ids": product.get("item_ids", []),
            "required_tools": tools.get("required", []),
        },
        "observed": {
            "rag_hit": summary.get("rag_hit"),
            "rag_card_ids": summary.get("rag_card_ids", []),
            "product_hit": summary.get("product_hit"),
            "product_item_ids": summary.get("product_item_ids", []),
            "tools": summary.get("tools", []),
        },
    }


def _tool_timeline(item: dict[str, Any]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for event in _tool_events(item):
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        tool = payload.get("tool")
        if event.get("type") not in {"tool.invoke", "tool.result"} or not tool:
            continue
        timeline.append(
            {
                "type": event.get("type"),
                "tool": tool,
                "tool_call_id": payload.get("tool_call_id"),
                "has_model_output": "model_output" in payload,
            }
        )
    return timeline


def _flow_type(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_class": item.get("case_class"),
        "query_type": item.get("query_type"),
    }


def _default_tool_expectation(case_class: Any) -> dict[str, Any]:
    if case_class in {"product_only", "no_evidence"}:
        required = ["product_search_tool"]
        optional = ["category_insight_tool"]
    else:
        required = ["category_insight_tool", "product_search_tool"]
        optional = []
    return {"required": required, "optional": optional}


_REGRESSION_CONTRACTS: dict[str, dict[str, Any]] | None = None


def _regression_contract(item: dict[str, Any]) -> dict[str, Any]:
    """Load only the pre-execution contract when an online row omitted it."""

    global _REGRESSION_CONTRACTS
    if _REGRESSION_CONTRACTS is None:
        contracts: dict[str, dict[str, Any]] = {}
        path = PROJECT_ROOT / "data" / "eval" / "flow_regression_v2.jsonl"
        try:
            for row in _load_json_lines(path):
                case_id = row.get("case_id")
                if isinstance(case_id, str):
                    contracts[case_id] = row
        except (OSError, json.JSONDecodeError):
            contracts = {}
        _REGRESSION_CONTRACTS = contracts
    value = _REGRESSION_CONTRACTS.get(str(item.get("case_id")))
    return value if isinstance(value, dict) else {}


def _flow_expectations(item: dict[str, Any]) -> dict[str, Any]:
    tools = item.get("tool_expectation")
    if not isinstance(tools, dict):
        tools = _regression_contract(item).get("tool_expectation")
    if not isinstance(tools, dict):
        tools = _default_tool_expectation(item.get("case_class"))
    required = [str(value) for value in tools.get("required", []) if value]
    optional = [str(value) for value in tools.get("optional", []) if value]
    allowed_tools = list(dict.fromkeys([*required, *optional]))
    raw_limits = tools.get("tool_call_limits") or tools.get("call_limits") or {}
    limits = {
        tool: int(raw_limits[tool])
        for tool in allowed_tools
        if isinstance(raw_limits, dict)
        and type(raw_limits.get(tool)) is int
        and raw_limits[tool] >= 1
    }
    for tool in allowed_tools:
        limits.setdefault(tool, 2 if tool == "product_search_tool" else 1)
    max_total = tools.get("max_total_tool_calls", 3)
    max_attempts = tools.get("max_generation_attempts", 2)
    require_final = tools.get("require_final_result", True)
    require_verified = tools.get("require_verified_final", True)
    if type(max_total) is not int or max_total < 1:
        max_total = 3
    if type(max_attempts) is not int or max_attempts < 1:
        max_attempts = 2
    if type(require_final) is not bool:
        require_final = True
    if type(require_verified) is not bool:
        require_verified = True
    return {
        "required_tools": required,
        "optional_tools": optional,
        "tool_call_limits": limits,
        "max_total_tool_calls": max_total,
        "max_generation_attempts": max_attempts,
        "require_final_result": require_final,
        "require_verified_final": require_verified,
        "verified_final_success_status": "supported",
        "evidence_judge_success_status": "supported",
        "fact_guard_success_value": True,
    }


def _available_evidence() -> dict[str, Any]:
    return {
        "paths": sorted(_ALLOWED_EVIDENCE_SOURCES),
        "capabilities": {
            "query": "用户原始请求，可判断预算、违禁要求、性别和显式需求",
            "final": "最终用户可见回答及去内部 ID 的紧凑推荐卡，可判断 P0/P2",
            "execution": "Flow 类型、工具调用顺序/次数、生成尝试和 final 是否存在，只判断 P1",
            "online_gate": (
                "验证状态和 Evidence Judge 成功值固定为 supported，"
                "Fact Guard 成功值固定为 true；只判断 P1"
            ),
        },
    }


def _compact_options(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    options: list[dict[str, Any]] = []
    for option in value:
        if not isinstance(option, dict):
            continue
        clean = {
            key: option.get(key)
            for key in ("name", "value")
            if option.get(key) is not None
        }
        if clean:
            options.append(clean)
    return options


def _compact_landed_price(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value.get("unavailable_reason"):
        return {"available": False, "reason": value.get("unavailable_reason")}
    return {
        key: value.get(source)
        for key, source in (("total", "landed_total_major"), ("currency", "currency"))
        if value.get(source) is not None
    }


def _compact_variant(variant: dict[str, Any]) -> dict[str, Any]:
    options = _compact_options(variant.get("options"))
    clean: dict[str, Any] = {
        "options": options or variant.get("display_name"),
        "price": variant.get("price_major"),
        "currency": variant.get("currency"),
        "available": str(variant.get("availability", "")).casefold()
        in {"available", "in_stock", "可售"},
    }
    landed = _compact_landed_price(variant.get("landed_price"))
    if landed is not None:
        clean["landed_price"] = landed
    return clean


def _compact_highlights(value: Any) -> list[Any]:
    """Keep user-visible highlights while removing internal provenance labels."""

    if not isinstance(value, list):
        return []
    internal_prefixes = ("source_attributes:", "source_tag:")
    return [
        entry
        for entry in value
        if not (
            isinstance(entry, str)
            and entry.strip().casefold().startswith(internal_prefixes)
        )
    ]


def _variant_is_mentioned(answer_text: str, variant: dict[str, Any]) -> bool:
    candidates = [variant.get("display_name")]
    candidates.extend(
        option.get("value")
        for option in variant.get("options", [])
        if isinstance(option, dict)
    )
    return any(
        isinstance(value, str) and len(value.strip()) >= 2 and value.strip() in answer_text
        for value in candidates
    )


def _compact_card(card: Any, answer_text: str) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    clean = {
        key: card.get(key)
        for key in ("title", "category", "currency")
        if card.get(key) not in (None, "", [])
    }
    highlights = _compact_highlights(card.get("highlights"))
    if highlights:
        clean["highlights"] = highlights
    price = card.get("price_major")
    if price is None and card.get("price_min_major") == card.get("price_max_major"):
        price = card.get("price_min_major")
    if price is not None:
        clean["price"] = price
    landed = _compact_landed_price(card.get("landed_price"))
    if landed is not None:
        clean["landed_price"] = landed

    variants = [value for value in card.get("variants", []) if isinstance(value, dict)]
    selected_id = card.get("selected_variant_id")
    selected = next(
        (variant for variant in variants if variant.get("variant_id") == selected_id),
        None,
    )
    if selected is not None:
        clean["selected_variant"] = _compact_variant(selected)
    mentioned = [
        _compact_variant(variant)
        for variant in variants
        if variant is not selected and _variant_is_mentioned(answer_text, variant)
    ]
    if mentioned:
        clean["available_options"] = mentioned[:12]
    return clean


def _compact_recommended_cards(cards: Any, answer_text: str) -> list[dict[str, Any]]:
    if not isinstance(cards, list):
        return []
    compact = [_compact_card(card, answer_text) for card in cards]
    return [card for card in compact if card is not None]


def _execution_summary(item: dict[str, Any]) -> dict[str, Any]:
    sequence = [
        str(event["payload"]["tool"])
        for event in _tool_events(item)
        if event.get("type") == "tool.invoke"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("tool")
    ]
    counts = {tool: sequence.count(tool) for tool in dict.fromkeys(sequence)}
    attempts = [
        event["payload"].get("attempt")
        for event in _tool_events(item)
        if event.get("type") == "evidence.verify"
        and isinstance(event.get("payload"), dict)
        and type(event["payload"].get("attempt")) is int
    ]
    return {
        "flow_type": _flow_type(item),
        "tool_sequence": sequence,
        "tool_counts": counts,
        "generation_attempts": max(attempts, default=0),
        "final_result_present": any(
            event.get("type") == "final.result" for event in _tool_events(item)
        ),
    }


def _online_gate(item: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    verification = _latest_verification(item) or {}
    fact_guard_errors = verification.get("fact_guard_errors")
    return {
        "verification_status": summary.get("verification_status", "unavailable"),
        "fact_guard_passed": isinstance(fact_guard_errors, list)
        and not fact_guard_errors,
        "evidence_judge_status": verification.get(
            "verification_status", "unavailable"
        ),
    }


def build_rubric_input(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": _sanitize_external(item.get("query", "")),
        "flow_type": _flow_type(item),
        "flow_expectations": _flow_expectations(item),
        "available_evidence": _available_evidence(),
    }


def build_final_judge_input(
    item: dict[str, Any], rubric: dict[str, Any] | None = None
) -> dict[str, Any]:
    if rubric is None:
        raise ValueError("Final Judge requires a validated rubric")
    summary = _flow_summary(item)
    answer_text = str(summary["final"].get("text") or "")
    return {
        "query": _sanitize_external(item.get("query", "")),
        "rubric": rubric,
        "final": {
            "answer_text": _sanitize_external(answer_text),
            "compact_recommended_cards": _sanitize_external(
                _compact_recommended_cards(
                    summary["final"].get("recommended_cards", []), answer_text
                )
            ),
        },
        "execution": _execution_summary(item),
        "online_gate": _online_gate(item, summary),
    }


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _evidence_sources(entry: dict[str, Any], field: str) -> list[str]:
    sources = entry.get(field)
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{field} must be a non-empty list")
    if any(source not in _ALLOWED_EVIDENCE_SOURCES for source in sources):
        raise ValueError(f"{field} contains unavailable evidence source")
    return [str(source) for source in sources]


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return [_text(item, field) for item in value]


def _validate_p1_status_criterion(criterion: str) -> None:
    lowered = criterion.casefold()
    compact = re.sub(r"[\s_\-]", "", lowered)
    if "verificationstatus" in compact and "supported" not in lowered:
        if any(term in lowered for term in _P1_INVALID_STATUS_TERMS):
            raise ValueError("P1 status criteria must use supported/true, not verified/passed")
        raise ValueError("verification_status success value must be supported")
    if "evidencejudgestatus" in compact and "supported" not in lowered:
        if any(term in lowered for term in _P1_INVALID_STATUS_TERMS):
            raise ValueError("P1 status criteria must use supported/true, not verified/passed")
        raise ValueError("evidence_judge_status success value must be supported")
    if "factguardpassed" in compact and "true" not in lowered:
        raise ValueError("fact_guard_passed success value must be true")


def validate_rubric(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {"p0", "p1", "p2"}:
        raise ValueError("Rubric must contain only p0, p1 and p2 lists")
    if any(not isinstance(payload[level], list) for level in ("p0", "p1", "p2")):
        raise ValueError("Rubric levels must be lists")
    ids: set[str] = set()
    normalized_criteria: dict[str, str] = {}
    result: dict[str, list[dict[str, Any]]] = {"p0": [], "p1": [], "p2": []}
    for level in ("p0", "p1", "p2"):
        for entry in payload[level]:
            if not isinstance(entry, dict):
                raise ValueError(f"{level} rubric entry must be an object")
            rubric_id = _text(entry.get("id"), f"{level}.id")
            if rubric_id in ids:
                raise ValueError(f"duplicate rubric id: {rubric_id}")
            ids.add(rubric_id)
            sources = _evidence_sources(entry, "evidence_sources")
            if level == "p0" and any(
                source not in {"query", "final_text", "final_cards"}
                for source in sources
            ):
                raise ValueError("P0 may use only query and final user-visible evidence")
            if level == "p1" and any(
                not source.startswith(("execution.", "online_gate."))
                for source in sources
            ):
                raise ValueError("P1 may use only execution and online gate evidence")
            if level == "p2" and any(
                source not in {"query", "final_text", "final_cards"}
                for source in sources
            ):
                raise ValueError("P2 may use only query and final user-visible evidence")
            if level in {"p0", "p1"}:
                if set(entry) != {"id", "criterion", "evidence_sources"}:
                    raise ValueError(f"{level} rubric entry has unexpected fields")
                criterion = _text(entry.get("criterion"), f"{level}.criterion")
                lowered = criterion.casefold()
                if any(topic.casefold() in lowered for topic in _INVISIBLE_TOPICS):
                    raise ValueError(f"rubric uses unobservable topic: {criterion}")
                if level == "p1" and any(term in lowered for term in _P1_USER_REQUIREMENT_TERMS):
                    raise ValueError("P1 may not score user-requirement satisfaction")
                if level == "p1":
                    _validate_p1_status_criterion(criterion)
                normalized = "".join(ch for ch in lowered if ch.isalnum())
                if normalized in normalized_criteria:
                    raise ValueError("duplicate rubric criterion across levels")
                normalized_criteria[normalized] = level
                clean = {
                    "id": rubric_id,
                    "criterion": criterion,
                    "evidence_sources": sources,
                }
            else:
                if set(entry) != {
                    "id",
                    "dimension",
                    "requirements",
                    "expectation",
                    "evidence_sources",
                }:
                    raise ValueError("p2 rubric entry has unexpected fields")
                dimension = _text(entry.get("dimension"), "p2.dimension")
                if dimension not in _P2_DIMENSIONS:
                    raise ValueError(f"invalid P2 dimension: {dimension}")
                requirements = _string_list(entry.get("requirements"), "p2.requirements")
                expectation = _text(entry.get("expectation"), "p2.expectation")
                visible_text = " ".join([*requirements, expectation]).casefold()
                if any(topic.casefold() in visible_text for topic in _INVISIBLE_TOPICS):
                    raise ValueError(f"rubric uses unobservable topic: {expectation}")
                clean = {
                    "id": rubric_id,
                    "dimension": dimension,
                    "requirements": requirements,
                    "expectation": expectation,
                    "evidence_sources": sources,
                }
            result[level].append(clean)
    p2_by_dimension = {entry["dimension"]: entry for entry in result["p2"]}
    if len(p2_by_dimension) != len(result["p2"]):
        raise ValueError("P2 dimensions must be unique")
    if set(p2_by_dimension) != set(_P2_DIMENSIONS):
        raise ValueError("P2 must contain the three fixed dimensions exactly once")
    result["p2"] = [p2_by_dimension[dimension] for dimension in _P2_DIMENSIONS]
    return result


def validate_judge_result(payload: Any, rubric: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Final Judge returned non-object JSON")
    if any(key in payload for key in ("score", "final_score", "p0_score", "p1_score", "p2_score")):
        raise ValueError("Final Judge must not calculate aggregate scores")
    expected = {
        "p0_results": [(entry["id"], entry) for entry in rubric["p0"]],
        "p1_results": [(entry["id"], entry) for entry in rubric["p1"]],
        "p2_results": [(entry["id"], entry) for entry in rubric["p2"]],
    }
    result: dict[str, Any] = {}
    for field, entries in expected.items():
        actual = payload.get(field)
        if not isinstance(actual, list):
            raise ValueError(f"Final Judge missing {field}")
        by_id: dict[str, dict[str, Any]] = {}
        for item in actual:
            if not isinstance(item, dict):
                raise ValueError(f"{field} entry must be an object")
            rubric_id = _text(item.get("rubric_id"), f"{field}.rubric_id")
            if rubric_id in by_id:
                raise ValueError(f"duplicate Final Judge result: {rubric_id}")
            by_id[rubric_id] = item
        if set(by_id) != {rubric_id for rubric_id, _ in entries}:
            raise ValueError(f"Final Judge ids do not match {field}")
        normalized: list[dict[str, Any]] = []
        for rubric_id, _ in entries:
            item = by_id[rubric_id]
            reason = _text(item.get("reason"), f"{field}.reason")
            clean: dict[str, Any] = {"rubric_id": rubric_id, "reason": reason}
            if field == "p0_results":
                if type(item.get("triggered")) is not bool:
                    raise ValueError("P0 triggered must be boolean")
                clean["triggered"] = item["triggered"]
            elif field == "p1_results":
                if type(item.get("violated")) is not bool:
                    raise ValueError("P1 violated must be boolean")
                clean["violated"] = item["violated"]
            else:
                score = item.get("score")
                if type(score) is not int or not 1 <= score <= 5:
                    raise ValueError("P2 score must be an integer from 1 to 5")
                clean["score"] = score
            normalized.append(clean)
        result[field] = normalized
    result["reason"] = str(payload.get("reason") or "逐项评分完成")
    return result


def aggregate_scores(
    rubric: dict[str, Any],
    judged: dict[str, Any],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    p0_results = judged["p0_results"]
    p1_results = judged["p1_results"]
    p2_results = judged["p2_results"]
    p0_score = round(
        sum(not result["triggered"] for result in p0_results) / len(p0_results)
        if p0_results
        else 1.0,
        3,
    )
    p1_score = round(
        max(0.0, 10.0 - 2.0 * sum(result["violated"] for result in p1_results)) / 10.0,
        3,
    )
    p2_score = round(sum(result["score"] for result in p2_results) / len(p2_results) / 5.0, 3)
    final_score = round(0.30 * p0_score + 0.30 * p1_score + 0.40 * p2_score, 3)
    pass_threshold = _pass_threshold() if threshold is None else threshold
    if not 0 <= pass_threshold <= 1:
        raise ValueError("pass threshold must be between 0 and 1")
    quality_pass = (
        not any(result["triggered"] for result in p0_results)
        and final_score >= pass_threshold
    )
    return {
        "p0_score": p0_score,
        "p1_score": p1_score,
        "p2_score": p2_score,
        "final_score": final_score,
        "pass_threshold": pass_threshold,
        "quality_pass": quality_pass,
    }


async def _call_json_model(
    client: httpx.AsyncClient,
    *,
    model: str,
    system: str,
    user_payload: dict[str, Any],
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    api_key = _api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    response: httpx.Response | None = None
    for attempt in range(2):
        try:
            response = await client.post(
                f"{_base_url().rstrip('/')}/chat/completions",
                headers=headers,
                json=request,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            break
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt == 1:
                raise
            await asyncio.sleep(0.25)
        except httpx.HTTPStatusError as error:
            if error.response.status_code not in {429, 500, 502, 503, 504} or attempt == 1:
                raise
            await asyncio.sleep(0.25)
    if response is None:
        raise httpx.NetworkError("model request produced no response")
    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, str) and content.strip().startswith("```"):
        content = content.strip().split("\n", 1)[-1].rsplit("```", 1)[0]
    parsed = _json_value(content)
    if not isinstance(parsed, dict):
        raise ValueError("model returned non-object JSON")
    return parsed


async def call_rubric_generator(client: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, Any]:
    model_payload = await _call_json_model(
        client,
        model=_rubric_model(),
        system=RUBRIC_GENERATOR_SYSTEM,
        user_payload=build_rubric_input(item),
        max_tokens=1200,
        timeout_seconds=150,
    )
    if not isinstance(model_payload, dict):
        raise ValueError("Rubric Generator returned non-object JSON")
    rubric = validate_rubric(model_payload)
    return {"evaluation_status": "completed", "rubric": rubric}


async def call_final_judge(
    client: httpx.AsyncClient, item: dict[str, Any], rubric: dict[str, Any]
) -> dict[str, Any]:
    summary = _flow_summary(item)
    if not summary["flow_valid"] or not summary["gate_trustworthy"]:
        return {
            "evaluation_status": "inconclusive",
            "reason": "flow evidence or online gate is unavailable",
        }
    judged = validate_judge_result(
        await _call_json_model(
            client,
            model=_model(),
            system=FINAL_JUDGE_SYSTEM,
            user_payload=build_final_judge_input(item, rubric),
            max_tokens=1600,
            timeout_seconds=240,
        ),
        rubric,
    )
    aggregate = aggregate_scores(rubric, judged)
    return {"evaluation_status": "completed", "judge": judged, **aggregate}


def _progress_line(index: int, total: int, row: dict[str, Any], elapsed_seconds: float) -> str:
    status = row.get("evaluation_status", "inconclusive")
    case_id = str(row.get("case_id") or row.get("query_id") or f"flow-{index}")
    judge = row.get("final_judge") or {}
    if status == "completed":
        outcome = "PASS" if judge.get("quality_pass") is True else "FAIL"
        detail = f"score={judge.get('final_score', 'N/A')} {outcome}"
    else:
        reason = str(judge.get("reason") or "unknown")
        reason = " ".join(reason.split())[:120]
        detail = f"reason={reason}"
    return f"[{index}/{total}] {case_id} status={status} {detail} elapsed={elapsed_seconds:.1f}s"


async def evaluate_items(
    items: list[dict[str, Any]], *, run_judge: bool = True, concurrency: int = 2
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    # The evaluator must not inherit a machine-wide proxy when it makes the
    # explicitly opted-in outbound model calls.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    async with httpx.AsyncClient(trust_env=False) as client:
        total = len(items)
        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_one(index: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                item_started = time.perf_counter()
                row = {**item, "summary": _flow_summary(item)}
                if run_judge:
                    stage = "rubric_generator"
                    try:
                        rubric_result = await call_rubric_generator(client, row)
                        row["rubric"] = rubric_result.get("rubric")
                        if rubric_result.get("evaluation_status") != "completed":
                            row["final_judge"] = rubric_result
                        else:
                            stage = "final_judge"
                            row["final_judge"] = await call_final_judge(
                                client, row, rubric_result["rubric"]
                            )
                    except (
                        httpx.HTTPError,
                        KeyError,
                        IndexError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as err:
                        row["final_judge"] = {
                            "evaluation_status": "inconclusive",
                            "failure_stage": stage,
                            "reason": f"{type(err).__name__}: {err}".strip(),
                        }
                else:
                    row["rubric"] = None
                    row["final_judge"] = {
                        "evaluation_status": "inconclusive",
                        "reason": "judge not run",
                    }
                row["evaluation_status"] = row["final_judge"].get(
                    "evaluation_status", "inconclusive"
                )
                print(
                    _progress_line(index, total, row, time.perf_counter() - item_started),
                    flush=True,
                )
                return index, row

        tasks = [
            asyncio.create_task(evaluate_one(index, item))
            for index, item in enumerate(items, 1)
        ]
        completed = await asyncio.gather(*tasks)
    return [row for _, row in sorted(completed)]


def _judge_passes(judge: dict[str, Any]) -> bool:
    return (
        judge.get("evaluation_status") == "completed"
        and judge.get("quality_pass") is True
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def render_report(results: list[dict[str, Any]]) -> str:
    valid = sum(1 for row in results if row.get("summary", {}).get("flow_valid"))
    trustworthy = sum(1 for row in results if row.get("summary", {}).get("gate_trustworthy"))
    judged = sum(
        1
        for row in results
        if row.get("final_judge", {}).get("evaluation_status") == "completed"
    )
    pass_count = sum(
        1
        for row in results
        if _judge_passes(row.get("final_judge", {}))
    )
    scores = [
        float(row["final_judge"]["final_score"])
        for row in results
        if row.get("final_judge", {}).get("evaluation_status") == "completed"
        and isinstance(row.get("final_judge", {}).get("final_score"), (int, float))
    ]
    avg_score = round(sum(scores) / len(scores), 3) if scores else None
    authority = (
        len(results) == 14
        and valid == 14
        and trustworthy == 14
        and judged == 14
        and pass_count == 14
    )
    lines = [
        (
            "# 14 Flow 在线流程与 Rubric/Final LLM-as-Judge（"
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}）"
        ),
        "",
        (
            f"有效流程：{valid}/{len(results)}；在线门禁可信：{trustworthy}/{len(results)}；"
            f"Final Judge 可判定：{judged}/{len(results)}；质量 PASS：{pass_count}/{len(results)}；"
            f"可判定样本平均分：{avg_score if avg_score is not None else 'N/A'}。"
        ),
        f"权威性：{'可成为候选权威基线' if authority else '候选/诊断，不替代旧 8 Flow 基线'}。",
        "",
    ]
    for index, row in enumerate(results, 1):
        summary = row.get("summary", {})
        judge = row.get("final_judge", {})
        lines.extend(
            [
                f"## {index}. {row.get('case_id') or row.get('query_id') or row.get('source', '')}",
                "",
                f"- Query：{row.get('query', '')}",
                (
                    "- Expected vs observed："
                    f"{json.dumps(summary.get('expected_observed', {}), ensure_ascii=False)}"
                ),
                (
                    f"- RAG 实际命中：{summary.get('rag_hit')}；"
                    f"商品实际命中：{summary.get('product_hit')}；"
                    f"实际工具：{', '.join(summary.get('tools', [])) or '无'}"
                ),
                (
                    f"- 最终 selections：{summary.get('final_selections', [])}；"
                    f"item_ids：{summary.get('final_item_ids', [])}"
                ),
                (
                    f"- 卡片/报价：{summary.get('cards_have_quotes')}；诊断："
                    f"{summary.get('quote_diagnostic')}；在线 Fact Guard："
                    f"{summary.get('online_fact_guard')}；在线 Evidence Judge："
                    f"{summary.get('online_evidence_judge')}"
                ),
                (
                    f"- Flow：{'valid' if summary.get('flow_valid') else 'inconclusive'}；"
                    f"评测：{judge.get('evaluation_status')}；P0/P1/P2："
                    f"{judge.get('p0_score', 'N/A')}/{judge.get('p1_score', 'N/A')}/"
                    f"{judge.get('p2_score', 'N/A')}；final_score："
                    f"{judge.get('final_score', 'N/A')}；"
                    f"PASS：{('是' if _judge_passes(judge) else '否')}；"
                    f"原因：{judge.get('reason') or judge.get('judge', {}).get('reason', '')}"
                ),
                "",
            ]
        )
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> Path:
    input_path = Path(args.input) if args.input else None
    if input_path is None:
        candidates = sorted((PROJECT_ROOT / "output" / "eval").glob("flow-results-*.json"))
        if not candidates:
            raise SystemExit("未找到完整 flow-results JSON；先运行 scripts/eval_flow_queries.py")
        input_path = candidates[-1]
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    items = parse_conversation(input_path)
    if args.limit:
        items = items[: args.limit]
    results = await evaluate_items(
        items, run_judge=not args.no_judge, concurrency=args.concurrency
    )
    output = Path(args.out) if args.out else PROJECT_ROOT / "output" / "eval" / (
        f"final-judge-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(results), encoding="utf-8")
    json_output = output.with_suffix(".json")
    json_output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Final Judge 报告已写入：{output}")
    print(f"逐项 JSON 已写入：{json_output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=2,
        help="maximum number of cases evaluated concurrently (default: 2)",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
