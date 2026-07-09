"""Normalize local EventBus events and OTel spans into Rubric evidence.

The evaluator subscribes before an intent and passes the resulting local
events/spans here after the turn.  No LangFuse read API is involved: external
trace export remains optional, while P1 process scoring receives deterministic
local evidence.  Only allow-listed arguments, result summaries, and span
attributes survive normalization.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from app.application.context.models import FrozenSegment, StageSummary
from scripts.eval.rubric_contract import (
    AgentDispatchEvidence,
    ContextLifecycleEvidence,
    DisplayedProductEvidence,
    EvaluationEvidence,
    PreferenceStateEvidence,
    ProductReferenceEvidence,
    RuntimeSignalEvidence,
    StructuredStateEvidence,
    ToolCallEvidence,
    TraceSpanEvidence,
    TurnEvidence,
)

_SECRET_RE = re.compile(
    r"(?i)(bearer\s+[^\s,;]+|sk-[a-z0-9_-]{8,}|pk-lf-[a-z0-9_-]{8,}|sk-lf-[a-z0-9_-]{8,})",
)
_PHONE_RE = re.compile(
    r"(?<![\w])(?:"
    r"\+?\d{1,3}[-.\s]+(?:\(\d{2,4}\)|\d{2,4})(?:[-.\s]+\d{3,4}){1,3}"
    r"|\(\d{2,4}\)[-.\s]+\d{3,4}[-.\s]+\d{4}"
    r"|0\d{2,3}[-.\s]+\d{7,8}"
    r"|\+?[1-9]\d{9,14}"
    r")(?![\w])"
)
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_URL_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret)=)[^&#\s]+"
)
_NAMED_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*)[^\s,;]+"
)
_POSTAL_RE = re.compile(r"(?i)(邮编\s*[:：]?\s*)\d{6}")
_SHIPPING_CLAUSE_RE = re.compile(
    r"(?i)((?:寄到|寄往|送到|配送至)\s*[:：]?)\s*[^。！？\n]*",
)
_PERSONAL_DETAIL_LINE_RE = re.compile(
    r"(?im)^(\s*(?:[-*]\s*)?(?:收件人|收货人|收货地址|配送地址|详细地址|地址|"
    r"邮编|邮政编码|电话|手机号)\s*[:：])[^\n]*$",
)
_INLINE_PERSONAL_DETAIL_RE = re.compile(
    r"(?i)((?:收件人|收货人|收货地址|配送地址|详细地址|地址)\s*[:：])"
    r"[^,，;；。\n]*",
)
_TABLE_PERSONAL_DETAIL_RE = re.compile(
    r"(?im)^(\s*\|\s*(?:收件人|收货人|收货地址|配送地址|详细地址|地址|"
    r"邮编|邮政编码|电话|手机号)\s*\|\s*)[^|\n]*(\|\s*)$",
)
_CHINESE_ADDRESS_RE = re.compile(
    r"(?:[\u4e00-\u9fff]{2,}(?:省|自治区|自治州|市|区|县|镇|街道|路|街|巷)){2,}"
    r"[\u4e00-\u9fffA-Za-z0-9\- ]{0,30}(?:号|栋|单元|室)?"
)
_ENGLISH_ADDRESS_RE = re.compile(
    r"(?i)\b\d{1,6}\s+[A-Z0-9 .'-]{2,40}\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln)\b[^,;\n]*"
)
_SENSITIVE_HEALTH_RE = re.compile(r"(?:我|您)?对[^，。；;\n]{1,24}过敏")

_SAFE_ARGUMENT_FIELDS = (
    "normalized_query",
    "platform",
    "site_locale",
    "category",
    "ship_to",
    "locale",
    "top_k",
    "price_max_major",
    "target_currency",
    "order_id",
    "query",
)
_SAFE_SPAN_ATTRIBUTES = frozenset(
    {
        "gen_ai.request.model",
        "globex.agent",
        "globex.budget_tier",
        "globex.dependency.attempts",
        "globex.dependency.retried",
        "globex.dependency.retry_reason",
        "globex.embedding.batch_size",
        "globex.embedding.vector_count",
        "globex.opensearch.hit_count",
        "globex.platform",
        "globex.reranker.score_count",
        "globex.recommended_count",
        "globex.site_locale",
    },
)


class EvidenceNormalizationError(ValueError):
    """Local events or spans cannot form trustworthy evaluation evidence."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def redact_text(value: str) -> str:
    """Remove common credentials and personal identifiers from local artifacts."""

    cleaned = _SECRET_RE.sub("[redacted]", value)
    cleaned = _URL_SECRET_RE.sub(r"\1[redacted]", cleaned)
    cleaned = _NAMED_SECRET_RE.sub(r"\1[redacted]", cleaned)
    cleaned = _SENSITIVE_HEALTH_RE.sub("[sensitive health detail redacted]", cleaned)
    cleaned = _SHIPPING_CLAUSE_RE.sub(r"\1[shipping details redacted]", cleaned)
    cleaned = _PERSONAL_DETAIL_LINE_RE.sub(r"\1[redacted]", cleaned)
    cleaned = _INLINE_PERSONAL_DETAIL_RE.sub(r"\1[redacted]", cleaned)
    cleaned = _TABLE_PERSONAL_DETAIL_RE.sub(r"\1[redacted] \2", cleaned)
    cleaned = _PHONE_RE.sub("[redacted]", cleaned)
    cleaned = _EMAIL_RE.sub("[redacted]", cleaned)
    cleaned = _CHINESE_ADDRESS_RE.sub("[address redacted]", cleaned)
    cleaned = _ENGLISH_ADDRESS_RE.sub("[address redacted]", cleaned)
    return _POSTAL_RE.sub(r"\1[redacted]", cleaned)


def _event_row(event: Any) -> dict[str, Any]:
    if hasattr(event, "to_dict"):
        row = event.to_dict()
    elif isinstance(event, Mapping):
        row = dict(event)
    else:
        raise EvidenceNormalizationError(
            f"unsupported event type: {type(event).__name__}",
        )
    if not isinstance(row.get("type"), str):
        raise EvidenceNormalizationError("event.type must be a string")
    payload = row.get("payload")
    if payload is not None and not isinstance(payload, Mapping):
        raise EvidenceNormalizationError("event.payload must be an object or null")
    return {"type": row["type"], "payload": dict(payload or {})}


def _safe_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key in _SAFE_ARGUMENT_FIELDS:
        item = value.get(key)
        if item is None or isinstance(item, (str, bool, int, float)):
            if item is not None:
                safe[key] = redact_text(item) if isinstance(item, str) else item
    items = value.get("items")
    if isinstance(items, list):
        safe["items"] = [
            {
                key: item[key]
                for key in ("product_id", "sku_id", "quantity")
                if key in item and isinstance(item[key], (str, int))
            }
            for item in items
            if isinstance(item, Mapping)
        ]
    return safe


def _product_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        product_id = item.get("product_id")
        if isinstance(product_id, str) and product_id:
            result.append(product_id)
    return result


def _safe_landed_price(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    safe: dict[str, Any] = {}
    for key in (
        "ship_to",
        "subtotal_major",
        "freight_major",
        "tariff_major",
        "tariff_rate",
        "de_minimis_applied",
        "landed_total_major",
        "currency",
        "unavailable_reason",
    ):
        item = value.get(key)
        if isinstance(item, (str, bool, int, float)):
            safe[key] = redact_text(item) if isinstance(item, str) else item
    return safe or None


def _safe_product_summaries(value: Any, *, filtered: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        product_id = item.get("product_id")
        if not isinstance(product_id, str) or not product_id:
            continue
        safe: dict[str, Any] = {"product_id": product_id}
        for key in ("title", "category", "price_major", "currency"):
            field = item.get(key)
            if isinstance(field, (str, int, float)):
                safe[key] = redact_text(field) if isinstance(field, str) else field
        if filtered:
            reason = item.get("reason")
            if isinstance(reason, str):
                safe["reason"] = redact_text(reason)
        else:
            landed_price = _safe_landed_price(item.get("landed_price"))
            if landed_price is not None:
                safe["landed_price"] = landed_price
        result.append(safe)
    return result


def _safe_result_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "platform",
        "site_locale",
        "hit_count",
        "filtered_count",
        "recall_strategy",
        "search_result_available",
        "circuit",
        "harness",
        "elapsed_ms",
    ):
        value = payload.get(key)
        if isinstance(value, (str, bool, int, float)):
            summary[key] = value
    summary["hit_product_ids"] = _product_ids(payload.get("hits"))
    summary["filtered_product_ids"] = _product_ids(payload.get("filtered_out"))
    summary["hit_facts"] = _safe_product_summaries(
        payload.get("hits"),
        filtered=False,
    )
    summary["filtered_facts"] = _safe_product_summaries(
        payload.get("filtered_out"),
        filtered=True,
    )
    order = payload.get("order")
    if isinstance(order, Mapping):
        summary["order"] = {
            key: order[key]
            for key in ("order_id", "status", "total_amount_major", "currency")
            if key in order and isinstance(order[key], (str, int, float))
        }
    if "saved" in payload:
        summary["saved"] = bool(payload.get("saved"))
    if "deleted" in payload:
        summary["deleted"] = bool(payload.get("deleted"))
    return summary


def _error_type(payload: Mapping[str, Any]) -> str | None:
    for key in ("classification", "circuit", "harness"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return "tool_error" if payload.get("error") else None


def _dispatch_key(payload: Mapping[str, Any]) -> tuple[str, str]:
    agent = str(payload.get("agent") or "")
    correlation = str(payload.get("dispatch_correlation_id") or "")
    return agent, correlation


def _tool_key(payload: Mapping[str, Any]) -> tuple[str, str]:
    tool = str(payload.get("tool") or "")
    correlation = str(payload.get("dispatch_correlation_id") or "")
    return tool, correlation


def _normalize_events(
    raw_events: Iterable[Any],
) -> tuple[
    str,
    list[AgentDispatchEvidence],
    list[ToolCallEvidence],
    list[RuntimeSignalEvidence],
]:
    rows = [_event_row(event) for event in raw_events]
    dispatches: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    signals: list[RuntimeSignalEvidence] = []
    active_tools: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    active_dispatches: dict[tuple[str, str], deque[int]] = defaultdict(deque)

    for order, row in enumerate(rows, 1):
        event_type = row["type"]
        payload = row["payload"]
        if event_type == "agent.dispatch":
            agent = payload.get("agent")
            if agent not in {"search_agent", "trade_agent"}:
                continue
            dispatch = {
                "order": order,
                "agent": agent,
                "platform": payload.get("platform"),
                "site_locale": payload.get("site_locale"),
                "correlation_id": payload.get("dispatch_correlation_id"),
                "elapsed_ms": None,
                "finished_order": None,
            }
            dispatches.append(dispatch)
            active_dispatches[_dispatch_key(payload)].append(len(dispatches) - 1)
            continue

        if event_type == "tool.invoke":
            tool = payload.get("tool")
            if not isinstance(tool, str) or not tool:
                raise EvidenceNormalizationError("tool.invoke requires payload.tool")
            call = {
                "order": order,
                "call_id": str(payload.get("tool_call_id") or f"event-call-{order}"),
                "agent": "main",
                "tool": tool,
                "arguments": _safe_arguments(payload.get("args")),
                "result_summary": {},
                "correlation_id": payload.get("dispatch_correlation_id"),
                "dependency_attempt": 1,
                "status": "unknown",
                "error_type": None,
            }
            tools.append(call)
            active_tools[_tool_key(payload)].append(len(tools) - 1)
            continue

        if event_type == "tool.result":
            tool = payload.get("tool")
            if tool == "task_dispatch":
                key = _dispatch_key(payload)
                candidates = active_dispatches.get(key)
                if candidates:
                    dispatch = dispatches[candidates.popleft()]
                    dispatch["elapsed_ms"] = payload.get("elapsed_ms")
                    dispatch["finished_order"] = order
                continue
            if not isinstance(tool, str) or not tool:
                continue
            key = _tool_key(payload)
            candidates = active_tools.get(key)
            if candidates:
                call = tools[candidates.popleft()]
                call["result_summary"] = _safe_result_summary(payload)
                call["error_type"] = _error_type(payload)
                call["status"] = "error" if call["error_type"] else "success"
            else:
                error_type = _error_type(payload)
                tools.append(
                    {
                        "order": order,
                        "call_id": f"event-result-{order}",
                        "agent": "main",
                        "tool": tool,
                        "arguments": _safe_arguments(payload.get("args")),
                        "result_summary": _safe_result_summary(payload),
                        "correlation_id": payload.get("dispatch_correlation_id"),
                        "dependency_attempt": 1,
                        "status": "error" if error_type else "unknown",
                        "error_type": error_type,
                    },
                )
                signals.append(
                    RuntimeSignalEvidence(
                        order=order,
                        signal="orphan_tool_result",
                        details={"tool": tool, "error_type": error_type},
                    ),
                )
            if payload.get("circuit"):
                signals.append(
                    RuntimeSignalEvidence(
                        order=order,
                        signal="circuit",
                        details={
                            "tool": tool,
                            "state": str(payload["circuit"]),
                        },
                    ),
                )
            if payload.get("harness"):
                signals.append(
                    RuntimeSignalEvidence(
                        order=order,
                        signal="harness",
                        details={
                            "tool": tool,
                            "state": str(payload["harness"]),
                        },
                    ),
                )
            continue

        if event_type in {
            "error",
            "model.fallback",
            "cache.hit",
            "context.compressed",
        }:
            details = {
                key: payload[key]
                for key in (
                    "classification",
                    "retrying",
                    "tool",
                    "agent",
                    "action",
                    "before_tokens",
                    "after_tokens",
                    "source_segment_count",
                )
                if key in payload and isinstance(payload[key], (str, bool, int))
            }
            signals.append(
                RuntimeSignalEvidence(
                    order=order,
                    signal=event_type,
                    details=details,
                ),
            )

    for call in tools:
        correlation = str(call.get("correlation_id") or "")
        matched = next(
            (
                dispatch
                for dispatch in dispatches
                if str(dispatch.get("correlation_id") or "") == correlation
                and correlation
            ),
            None,
        )
        if matched is None:
            matched = next(
                (
                    dispatch
                    for dispatch in dispatches
                    if dispatch["agent"] == "trade_agent"
                    and dispatch["order"] < call["order"]
                    and (
                        dispatch["finished_order"] is None
                        or call["order"] < dispatch["finished_order"]
                    )
                ),
                None,
            )
        if matched is not None:
            call["agent"] = matched["agent"]

    agents = {dispatch["agent"] for dispatch in dispatches}
    route = (
        "main.direct"
        if not agents
        else f"main.dispatch.{next(iter(agents))}"
        if len(agents) == 1
        else "main.dispatch.mixed"
    )
    normalized_dispatches = [
        AgentDispatchEvidence.model_validate(
            {
                **{
                    key: value
                    for key, value in dispatch.items()
                    if key not in {"finished_order", "correlation_id"}
                },
                "correlation_id": None,
            },
        )
        for dispatch in dispatches
    ]
    normalized_tools = [
        ToolCallEvidence.model_validate(
            {
                **tool,
                "call_id": f"call-{index}",
                "correlation_id": None,
            }
        )
        for index, tool in enumerate(tools, start=1)
    ]
    return route, normalized_dispatches, normalized_tools, signals


def _safe_span_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return redact_text(value) if isinstance(value, str) else value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _safe_span_value(item)
            for item in value
            if isinstance(item, (str, bool, int, float))
        ]
    return redact_text(str(value)[:200])


def normalize_trace_spans(
    spans: Iterable[Any],
) -> tuple[str | None, list[TraceSpanEvidence]]:
    """Keep one trace's topology and allow-listed operational attributes."""

    collected = list(spans)
    trace_ids = {
        int(span.context.trace_id)
        for span in collected
        if getattr(getattr(span, "context", None), "trace_id", 0)
    }
    if len(trace_ids) > 1:
        raise EvidenceNormalizationError("one turn contains spans from multiple traces")
    ordered = sorted(
        collected,
        key=lambda item: (getattr(item, "start_time", 0), item.context.span_id),
    )
    local_ids = {
        int(span.context.span_id): f"{index:016x}"
        for index, span in enumerate(ordered, start=1)
    }
    normalized: list[TraceSpanEvidence] = []
    for span in ordered:
        parent = getattr(span, "parent", None)
        status_code = getattr(getattr(span, "status", None), "status_code", None)
        status = getattr(status_code, "name", "UNSET")
        if status not in {"UNSET", "OK", "ERROR"}:
            status = "UNSET"
        start_time = int(getattr(span, "start_time", 0) or 0)
        end_time = int(getattr(span, "end_time", start_time) or start_time)
        attributes = {
            key: _safe_span_value(value)
            for key, value in dict(getattr(span, "attributes", {}) or {}).items()
            if key in _SAFE_SPAN_ATTRIBUTES
        }
        normalized.append(
            TraceSpanEvidence(
                span_id=local_ids[int(span.context.span_id)],
                parent_span_id=(
                    local_ids.get(int(parent.span_id)) if parent is not None else None
                ),
                name=str(span.name),
                status=status,
                duration_ms=round(max(0, end_time - start_time) / 1_000_000, 3),
                attributes=attributes,
            ),
        )
    return None, normalized


def _state_product_refs(values: Any) -> list[ProductReferenceEvidence]:
    if not isinstance(values, list):
        return []
    result: list[ProductReferenceEvidence] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        card = value.get("card")
        product_id = card.get("product_id") if isinstance(card, Mapping) else None
        platform = value.get("platform")
        if not isinstance(product_id, str) or platform not in {
            "globex_reference",
            "taobao",
            "amazon",
        }:
            continue
        key = (platform, product_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            ProductReferenceEvidence(
                platform=platform,
                product_id=product_id,
                site_locale=value.get("site_locale"),
            )
        )
    return result


def build_structured_state_evidence(
    context_namespace: Any,
    *,
    raw_context_message_count: int = 0,
    preference_before: PreferenceStateEvidence,
    preference_after: PreferenceStateEvidence,
) -> StructuredStateEvidence:
    """Project L2/L3/L4 metadata without buyer, address, or raw text."""

    namespace = context_namespace if isinstance(context_namespace, Mapping) else {}
    nested_context = namespace.get("session_context")
    context = nested_context if isinstance(nested_context, Mapping) else namespace
    search = context.get("last_search")
    search = search if isinstance(search, Mapping) else {}
    arguments: list[dict[str, Any]] = []
    if isinstance(search.get("args"), Mapping):
        arguments.append(_safe_arguments(search["args"]))
    platform_results = search.get("platform_results")
    if isinstance(platform_results, list):
        for item in platform_results[:3]:
            if not isinstance(item, Mapping):
                continue
            safe = _safe_arguments(item.get("args"))
            if isinstance(item.get("platform"), str):
                safe["platform"] = item["platform"]
            if isinstance(item.get("site_locale"), str):
                safe["site_locale"] = item["site_locale"]
            arguments.append(safe)
    recommendation = context.get("last_recommendation")
    recommendation = recommendation if isinstance(recommendation, Mapping) else {}
    order = context.get("order")
    safe_order = (
        {
            key: order[key]
            for key in ("order_id", "status", "currency", "total_amount_major")
            if key in order and isinstance(order[key], (str, int, float))
        }
        if isinstance(order, Mapping)
        else None
    )
    history = recommendation.get("displayed_history")
    if not isinstance(history, list):
        history = recommendation.get("displayed_products")
    frozen: list[FrozenSegment] = []
    for item in namespace.get("frozen_segments") or []:
        if not isinstance(item, Mapping):
            continue
        try:
            frozen.append(FrozenSegment.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    stage_raw = namespace.get("stage_summary")
    stage: StageSummary | None = None
    if isinstance(stage_raw, Mapping):
        try:
            stage = StageSummary.from_dict(stage_raw)
        except (KeyError, TypeError, ValueError):
            stage = None
    frozen_hashes = {item.segment_id: item.content_hash for item in frozen}
    stage_verified = bool(stage) and all(
        frozen_hashes.get(source["segment_id"]) == source["content_hash"]
        for source in stage.source_segments
    )
    compression = namespace.get("context_compression")
    compression = compression if isinstance(compression, Mapping) else {}
    action = str(compression.get("action") or "none")
    if action not in {"none", "l3_stage_summary", "l3_no_gain"}:
        action = "none"
    budget = namespace.get("budget_report")
    budget = budget if isinstance(budget, Mapping) else {}
    return StructuredStateEvidence(
        schema_version=str(context.get("schema_version") or ""),
        revision=max(0, int(context.get("revision") or 0)),
        last_search_arguments=arguments[:3],
        last_search_product_refs=[
            str(item) for item in (search.get("product_refs") or [])[:20]
        ],
        last_search_filtered_refs=[
            str(item) for item in (search.get("filtered_product_refs") or [])[:20]
        ],
        current_recommendations=_state_product_refs(
            recommendation.get("displayed_products")
        )[:5],
        verified_display_history=_state_product_refs(history)[:20],
        order=safe_order or None,
        context_lifecycle=ContextLifecycleEvidence(
            raw_context_message_count=max(0, raw_context_message_count),
            freeze_cursor=max(0, int(namespace.get("freeze_cursor") or 0)),
            frozen_segment_count=len(frozen),
            stage_summary_present=isinstance(stage_raw, Mapping),
            stage_summary_source_count=(
                len(stage.source_segments) if stage is not None else 0
            ),
            stage_source_message_start=(
                stage.source_message_start if stage is not None else None
            ),
            stage_source_message_end=(
                stage.source_message_end if stage is not None else None
            ),
            stage_summary_hash=(stage.summary_hash if stage is not None else None),
            stage_summary_verified=stage_verified,
            compression_action=action,
            before_tokens=(
                int(compression["before_tokens"])
                if isinstance(compression.get("before_tokens"), int)
                else None
            ),
            after_tokens=(
                int(compression["after_tokens"])
                if isinstance(compression.get("after_tokens"), int)
                else None
            ),
            budget_decision=str(namespace.get("budget_decision") or ""),
            total_input_tokens=max(0, int(budget.get("total_input_tokens") or 0)),
            soft_limit_tokens=max(0, int(budget.get("soft_limit_tokens") or 0)),
        ),
        preference_before=preference_before,
        preference_after=preference_after,
    )


def _displayed_products(values: Iterable[Any]) -> list[DisplayedProductEvidence]:
    result: list[DisplayedProductEvidence] = []
    for raw in values:
        if not isinstance(raw, Mapping):
            raise EvidenceNormalizationError("displayed product must be an object")
        card = raw.get("card")
        product_id = card.get("product_id") if isinstance(card, Mapping) else None
        if not isinstance(product_id, str) or not product_id:
            raise EvidenceNormalizationError(
                "displayed product is missing card.product_id"
            )
        result.append(
            DisplayedProductEvidence(
                rank=raw.get("rank"),
                product_id=product_id,
                platform=raw.get("platform"),
                site_locale=raw.get("site_locale"),
            ),
        )
    return result


def build_turn_evidence(
    *,
    turn_index: int,
    user_input: str,
    final_text: str,
    displayed_products: Iterable[Any],
    events: Iterable[Any],
    structured_state: StructuredStateEvidence,
    spans: Iterable[Any] = (),
) -> TurnEvidence:
    route, dispatches, tool_calls, runtime_signals = _normalize_events(events)
    trace_id, trace_spans = normalize_trace_spans(spans)
    return TurnEvidence(
        turn_index=turn_index,
        user_input=redact_text(user_input),
        route=route,
        trace_id=trace_id,
        dispatches=dispatches,
        tool_calls=tool_calls,
        runtime_signals=runtime_signals,
        trace_spans=trace_spans,
        displayed_products=_displayed_products(displayed_products),
        structured_state=structured_state,
        final_text=redact_text(final_text),
    )


def build_evaluation_evidence(
    *,
    case_id: str,
    session_id: str,
    turns: list[TurnEvidence],
    execution_profile: str = "default",
) -> EvaluationEvidence:
    return EvaluationEvidence(
        case_id=case_id,
        execution_profile=execution_profile,
        session_id_hash=_digest(session_id),
        turns=turns,
    )
