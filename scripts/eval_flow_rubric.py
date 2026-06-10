"""Independent Final LLM-as-Judge for completed online agent flows.

This script is intentionally downstream of the online gate.  It does not run
an offline Fact Guard, Claim Ledger, per-claim verifier, evidence catalog, or
bounded-evidence transform.  Missing flow evidence, invalid final schema, or
an unavailable judge is reported as inconclusive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from globex_agent.application.evidence_verification import sanitize_evidence_judge_payload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
_VALID_VERDICTS = {"supported", "unsupported", "inconclusive"}

FINAL_JUDGE_SYSTEM = (
    "你是独立的 Globex 最终 LLM-as-Judge。输入包含 query、在线最终 draft、后端最终卡片、"
    "完整模型可见工具输出和在线验证状态。只依据这些输入评估 P0 硬事实/安全、P1 query 满足度、"
    "P2 表达。在线 Evidence Judge/Fact Guard 的可用性由本地编排器先行判断，"
    "不发送快照、provenance、确认信息或交易 token。"
    "证据缺失、在线门禁不可用、输入结构无效或语义无法判断时必须 inconclusive，不能猜测。"
    "按 P0/P1/P2 criterion 状态计算 score：P0 50%、P1 35%、P2 15%；"
    "supported=1、unsupported=0；任一 criterion 为 inconclusive 时本条不可判定。"
    "只输出 JSON：{\"p0\":[{\"status\":\"supported|unsupported|inconclusive\",\"reason\":\"...\"}],"
    "\"p1\":[...],\"p2\":[...],\"score\":0到1之间的数字,\"reason\":\"...\"}。"
)


def _model() -> str:
    return os.environ.get("EVAL_FINAL_JUDGE_MODEL") or "qwen-plus"


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
def build_final_judge_input(item: dict[str, Any]) -> dict[str, Any]:
    summary = _flow_summary(item)
    final_cards = summary["final"].get("recommended_cards", [])
    online_verification = _latest_verification(item) or {
        "verification_status": summary.get("verification_status", "unavailable")
    }
    return {
        "query": _sanitize_external(item.get("query", "")),
        "draft": {
            "answer_text": summary["final"].get("text", ""),
            "selections": summary.get("final_selections", []),
        },
        "recommended_cards": _sanitize_external(final_cards),
        "online_verification": _sanitize_external(online_verification),
        "tool_outputs": summary["tool_outputs"],
    }


def _computed_score(payload: dict[str, Any]) -> float | None:
    """Apply the documented rubric locally after validating Judge statuses."""

    ratios: list[float] = []
    for level in ("p0", "p1", "p2"):
        entries = payload.get(level, [])
        if any(entry.get("status") == "inconclusive" for entry in entries):
            return None
        ratios.append(
            sum(entry.get("status") == "supported" for entry in entries) / len(entries)
            if entries
            else 1.0
        )
    return round(0.50 * ratios[0] + 0.35 * ratios[1] + 0.15 * ratios[2], 3)


def _p0_pass(payload: dict[str, Any]) -> bool:
    return all(entry.get("status") == "supported" for entry in payload.get("p0", []))


async def call_final_judge(client: httpx.AsyncClient, item: dict[str, Any]) -> dict[str, Any]:
    summary = _flow_summary(item)
    if not summary["flow_valid"] or not summary["gate_trustworthy"]:
        return {"status": "inconclusive", "reason": "flow evidence or online gate is unavailable"}
    api_key = _api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = await client.post(
        f"{_base_url().rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": _model(),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": FINAL_JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(build_final_judge_input(item), ensure_ascii=False),
                },
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    payload = _json_value(content)
    if not isinstance(payload, dict):
        raise ValueError("Final Judge returned non-object JSON")
    for level in ("p0", "p1", "p2"):
        if not isinstance(payload.get(level), list):
            raise ValueError(f"Final Judge missing {level}")
        for entry in payload[level]:
            if not isinstance(entry, dict) or entry.get("status") not in _VALID_VERDICTS:
                raise ValueError(f"Final Judge invalid {level} status")
    score = payload.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not isfinite(float(score)):
        raise ValueError("Final Judge missing numeric score")
    if not 0 <= float(score) <= 1:
        raise ValueError("Final Judge score must be between 0 and 1")
    computed = _computed_score(payload)
    if computed is not None and abs(float(score) - computed) > 0.051:
        raise ValueError(
            f"Final Judge score disagrees with rubric: model={score}, computed={computed}"
        )
    result = {"status": "completed"}
    result.update({key: value for key, value in payload.items() if key != "status"})
    result["computed_score"] = computed
    if computed is None:
        result["status"] = "inconclusive"
        result["reason"] = payload.get("reason") or "Final Judge criterion is inconclusive"
    return result


async def evaluate_items(
    items: list[dict[str, Any]], *, run_judge: bool = True
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for item in items:
            row = {**item, "summary": _flow_summary(item)}
            if run_judge:
                try:
                    row["final_judge"] = await call_final_judge(client, row)
                except (
                    httpx.HTTPError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as err:
                    row["final_judge"] = {
                        "status": "inconclusive",
                        "reason": f"{type(err).__name__}: {err}".strip(),
                    }
            else:
                row["final_judge"] = {"status": "inconclusive", "reason": "judge not run"}
            results.append(row)
    return results


def _judge_passes(judge: dict[str, Any]) -> bool:
    return (
        judge.get("status") == "completed"
        and _p0_pass(judge)
        and isinstance(judge.get("score"), (int, float))
        and float(judge["score"]) >= 0.7
    )


def render_report(results: list[dict[str, Any]]) -> str:
    valid = sum(1 for row in results if row.get("summary", {}).get("flow_valid"))
    trustworthy = sum(1 for row in results if row.get("summary", {}).get("gate_trustworthy"))
    judged = sum(1 for row in results if row.get("final_judge", {}).get("status") == "completed")
    pass_count = sum(
        1
        for row in results
        if row.get("final_judge", {}).get("status") == "completed"
        and _p0_pass(row["final_judge"])
        and float(row["final_judge"].get("score", -1)) >= 0.7
    )
    scores = [
        float(row["final_judge"]["score"])
        for row in results
        if row.get("final_judge", {}).get("status") == "completed"
        and isinstance(row.get("final_judge", {}).get("score"), (int, float))
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
        f"# 14 Flow 在线流程与 Final LLM-as-Judge（{datetime.now().strftime('%Y-%m-%d %H:%M')}）",
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
                    f"Final Judge：{judge.get('status')}；score：{judge.get('score', 'N/A')}；"
                    f"PASS：{('是' if _judge_passes(judge) else '否')}；"
                    f"原因：{judge.get('reason', '')}"
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
    results = await evaluate_items(items, run_judge=not args.no_judge)
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
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
