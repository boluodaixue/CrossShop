"""Pure, idempotent projection of H4 events into bounded L4 context."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from .models import L4Context

_SEARCH_ARGS = {
    "normalized_query",
    "category",
    "ship_to",
    "locale",
    "top_k",
    "target_currency",
    "price_max_major",
    # H4 dispatch metadata; these select an index but are never ProductSearchSpec fields.
    "platform",
    "site_locale",
}
_ORDER_ITEM_FIELDS = {
    "product_id",
    "sku_id",
    "quantity",
    "title",
    "unit_price_major",
    "subtotal_major",
    "currency",
}
_MAX_ITEMS = 10
_MAX_REFS = 20
_MAX_STRING = 512
_PLATFORM_ORDER = {"globex_reference": 0, "taobao": 1, "amazon": 2}


def _event_parts(event: Any) -> tuple[str, Mapping[str, Any], str]:
    if isinstance(event, Mapping):
        payload = event.get("payload")
        return (
            str(event.get("type", "")),
            payload if isinstance(payload, Mapping) else {},
            str(event.get("occurred_at", "")),
        )
    payload = getattr(event, "payload", {})
    return (
        str(getattr(event, "type", "")),
        payload if isinstance(payload, Mapping) else {},
        str(getattr(event, "occurred_at", "")),
    )


def _value(intent: Any, key: str) -> Any:
    return (
        intent.get(key) if isinstance(intent, Mapping) else getattr(intent, key, None)
    )


def _parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value.strip())
    except (json.JSONDecodeError, AttributeError):
        return value


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return None
    if isinstance(value, str):
        return value[:_MAX_STRING]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in list(value.items())[:32]:
            bounded = _bounded(child, depth=depth + 1)
            if bounded is not None:
                result[str(key)] = bounded
        return result
    if isinstance(value, (list, tuple)):
        return [_bounded(child, depth=depth + 1) for child in list(value)[:_MAX_REFS]]
    return value


def _compact_mapping(value: Any, allowed: set[str]) -> dict[str, Any]:
    parsed = _parse(value)
    if not isinstance(parsed, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, child in parsed.items():
        if str(key) not in allowed:
            continue
        bounded = _bounded(child)
        if _present(bounded):
            result[str(key)] = bounded
    return result


def _compact_items(value: Any) -> list[dict[str, Any]]:
    parsed = _parse(value)
    if not isinstance(parsed, list):
        return []
    result: list[dict[str, Any]] = []
    for item in parsed[:_MAX_ITEMS]:
        compact = _compact_mapping(item, _ORDER_ITEM_FIELDS)
        if _present(compact.get("product_id")) and _present(compact.get("sku_id")):
            result.append(compact)
    return result


def _output(payload: Mapping[str, Any]) -> Any:
    for key in ("raw_output", "result", "model_output"):
        if key in payload:
            return _parse(payload[key])
    return payload if "hits" in payload else None


def reduce_l4(
    intent: Any,
    events: Iterable[Any] = (),
    previous: L4Context | Mapping[str, Any] | None = None,
    *,
    now: str | None = None,
) -> L4Context:
    """Project only paired successful tool results; replay is idempotent."""

    prior = (
        previous if isinstance(previous, L4Context) else L4Context.from_dict(previous)
    )
    state = prior.to_dict()
    for key in ("session", "request", "last_search", "last_recommendation", "order"):
        state[key] = copy.deepcopy(getattr(prior, key))
    session = state["session"]
    request = state["request"]
    for source in ("shopping_session_id", "buyer_id", "locale", "currency"):
        value = _value(intent, source)
        if _present(value):
            session[source] = _bounded(value)
    raw_query = _value(intent, "raw_query")
    if _present(raw_query):
        request.setdefault("session_origin_raw_query", _bounded(raw_query))
        request["current_raw_query"] = _bounded(raw_query)

    normalized = [_event_parts(event) for event in events]
    invokes: dict[str, tuple[str, dict[str, Any]]] = {}
    dispatch_searches: list[dict[str, Any]] = []
    for kind, payload, _occurred_at in normalized:
        if kind != "tool.invoke":
            continue
        call_id = str(payload.get("tool_call_id", ""))
        tool = str(payload.get("tool", ""))
        if not call_id or not tool or call_id in invokes:
            continue
        args = payload.get("args")
        invokes[call_id] = (tool, dict(args) if isinstance(args, Mapping) else {})

    for kind, payload, _occurred_at in normalized:
        if kind != "tool.result" or payload.get("error"):
            continue
        call_id = str(payload.get("tool_call_id", ""))
        paired = invokes.get(call_id)
        if paired is None:
            continue
        tool, args = paired
        if str(payload.get("tool", "")) != tool:
            continue
        output = _output(payload)
        if not isinstance(output, Mapping) or output.get("error"):
            continue
        if tool == "product_search_tool" and isinstance(output.get("hits"), list):
            hits = output["hits"]
            refs = [
                str(hit["product_id"])
                for hit in hits
                if isinstance(hit, Mapping) and _present(hit.get("product_id"))
            ][:_MAX_REFS]
            state["last_search"] = {
                "tool_call_id": call_id,
                "args": _compact_mapping(args, _SEARCH_ARGS),
                "status": "succeeded",
                "returned_count": len(hits),
                "product_refs": refs,
            }
        elif (
            tool == "task_dispatch"
            and output.get("agent") == "search_agent"
            and output.get("search_result_available") is True
            and isinstance(output.get("hits"), list)
            and isinstance(output.get("filtered_out"), list)
        ):
            hits = output["hits"]
            filtered_out = output["filtered_out"]
            refs = [
                str(hit["product_id"])
                for hit in hits
                if isinstance(hit, Mapping) and _present(hit.get("product_id"))
            ][:_MAX_REFS]
            filtered_refs = [
                str(hit["product_id"])
                for hit in filtered_out
                if isinstance(hit, Mapping) and _present(hit.get("product_id"))
            ][:_MAX_REFS]
            search_args = _compact_mapping(output.get("search_args"), _SEARCH_ARGS)
            if _present(output.get("platform")):
                search_args["platform"] = _bounded(output["platform"])
            if _present(output.get("site_locale")):
                search_args["site_locale"] = _bounded(output["site_locale"])
            dispatch_searches.append(
                {
                    "tool_call_id": call_id,
                    "platform": _bounded(output.get("platform")),
                    "site_locale": _bounded(output.get("site_locale")),
                    "args": search_args,
                    "returned_count": len(hits),
                    "filtered_count": len(filtered_out),
                    "product_refs": refs,
                    "filtered_product_refs": filtered_refs,
                },
            )
        elif tool in {"create_order_tool", "query_order_tool", "cancel_order_tool"}:
            order = copy.deepcopy(state.get("order") or {})
            for key in (
                "order_id",
                "status",
                "currency",
                "total_amount_major",
                "cancel_reason",
            ):
                if _present(output.get(key)):
                    order[key] = _bounded(output[key])
            items = _compact_items(output.get("items") or output.get("lines"))
            if not items and tool == "create_order_tool":
                items = _compact_items(args.get("items"))
            if items:
                order["items"] = items
            state["order"] = order or state.get("order")

    if dispatch_searches:
        dispatch_searches.sort(
            key=lambda item: (
                _PLATFORM_ORDER.get(str(item.get("platform")), 99),
                str(item.get("site_locale") or ""),
                str(item.get("tool_call_id") or ""),
            ),
        )
        product_refs = list(
            dict.fromkeys(
                ref for result in dispatch_searches for ref in result["product_refs"]
            ),
        )[:_MAX_REFS]
        filtered_refs = list(
            dict.fromkeys(
                ref
                for result in dispatch_searches
                for ref in result["filtered_product_refs"]
            ),
        )[:_MAX_REFS]
        state["last_search"] = {
            "tool_call_ids": [item["tool_call_id"] for item in dispatch_searches],
            "status": "succeeded",
            "returned_count": sum(
                int(item["returned_count"]) for item in dispatch_searches
            ),
            "filtered_count": sum(
                int(item["filtered_count"]) for item in dispatch_searches
            ),
            "product_refs": product_refs,
            "filtered_product_refs": filtered_refs,
            "platform_results": dispatch_searches,
        }

    # V2 has no authoritative structured selected-product result. Natural
    # language final text is deliberately not parsed into last_recommendation.
    candidate = L4Context.from_dict(state)
    if candidate.semantic_dict() == prior.semantic_dict():
        return prior
    state["revision"] = prior.revision + 1
    state["updated_at"] = next(
        (occurred for _kind, _payload, occurred in reversed(normalized) if occurred),
        now or datetime.now(UTC).isoformat(),
    )
    return L4Context.from_dict(state)


class L4Reducer:
    def reduce(
        self,
        intent: Any,
        events: Iterable[Any] = (),
        previous: L4Context | Mapping[str, Any] | None = None,
        *,
        now: str | None = None,
    ) -> L4Context:
        return reduce_l4(intent, events, previous, now=now)
