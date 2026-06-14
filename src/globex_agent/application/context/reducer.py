"""Pure projection of existing intent/trade events into bounded L4 state."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from .models import L4Context

_SEARCH_ARGS = {
    "normalized_query",
    "category",
    "ship_to",
    "platform",
    "locale",
    "brand",
    "top_k",
    "price_max_major",
    "target_currency",
    "material_include",
    "material_exclude",
    "material_unknown_policy",
}
_CARD_FIELDS = {
    "item_id",
    "variant_id",
    "title",
    "name",
    "price_major",
    "price_min_major",
    "price_max_major",
    "unit_price_major",
    "currency",
    "target_currency",
    "ship_to",
}
_ORDER_ITEM_FIELDS = {
    "item_id",
    "variant_id",
    "quantity",
    "title",
    "variant_display_name",
    "unit_price_major",
    "subtotal_major",
    "currency",
}
_MAX_ITEMS = 5
_MAX_REFS = 20
_MAX_STRING = 512


def _event_parts(event: Any) -> tuple[str, Mapping[str, Any], str]:
    if isinstance(event, Mapping):
        payload = event.get("payload")
        return str(event.get("type", "")), payload if isinstance(payload, Mapping) else {}, str(
            event.get("occurred_at", "")
        )
    payload = getattr(event, "payload", {})
    return str(getattr(event, "type", "")), payload if isinstance(payload, Mapping) else {}, str(
        getattr(event, "occurred_at", "")
    )


def _value(intent: Any, key: str) -> Any:
    if isinstance(intent, Mapping):
        return intent.get(key)
    return getattr(intent, key, None)


def _parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Keep L4 useful and bounded without inventing or normalizing facts."""

    if depth > 3:
        return None
    if isinstance(value, str):
        return value[:_MAX_STRING]
    if isinstance(value, Mapping):
        return {
            str(key): _bounded(child, depth=depth + 1)
            for key, child in list(value.items())[:32]
            if _bounded(child, depth=depth + 1) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(child, depth=depth + 1) for child in list(value)[:_MAX_REFS]]
    return value


def _compact_mapping(value: Any, allowed: set[str]) -> dict[str, Any]:
    parsed = _parse(value)
    if not isinstance(parsed, Mapping):
        return {}
    return {
        str(key): bounded
        for key, child in parsed.items()
        if str(key) in allowed
        and (bounded := _bounded(child)) is not None
        and _present(bounded)
    }


def _compact_cards(value: Any) -> list[dict[str, Any]]:
    parsed = _parse(value)
    if not isinstance(parsed, list):
        return []
    cards: list[dict[str, Any]] = []
    for card in parsed[:_MAX_ITEMS]:
        compact = _compact_mapping(card, _CARD_FIELDS)
        if _present(compact.get("item_id")):
            cards.append(compact)
    return cards


def _compact_items(value: Any) -> list[dict[str, Any]]:
    parsed = _parse(value)
    if not isinstance(parsed, list):
        return []
    items: list[dict[str, Any]] = []
    for item in parsed[:_MAX_ITEMS]:
        compact = _compact_mapping(item, _ORDER_ITEM_FIELDS)
        if _present(compact.get("item_id")):
            items.append(compact)
    return items


def _result_value(payload: Mapping[str, Any]) -> Any:
    output = payload.get("model_output")
    if output is None:
        output = payload.get("result")
    if output is None and "hits" in payload:
        output = payload
    return _parse(output)


def _occurred_at(events: list[tuple[str, Mapping[str, Any], str]], fallback: str) -> str:
    for _kind, _payload, occurred_at in reversed(events):
        if occurred_at:
            return occurred_at
    return fallback


def reduce_l4(
    intent: Any,
    events: Iterable[Any] = (),
    previous: L4Context | Mapping[str, Any] | None = None,
    *,
    now: str | None = None,
) -> L4Context:
    """Apply an intent and ordered events once, returning an idempotent snapshot.

    The function has no I/O and never mutates ``previous`` or any event payload.
    Replaying the same intent/events against its result returns the same
    revision and timestamp.
    """

    prior = previous if isinstance(previous, L4Context) else L4Context.from_dict(previous)
    state = prior.to_dict()
    state["session"] = copy.deepcopy(prior.session)
    state["request"] = copy.deepcopy(prior.request)
    state["last_search"] = copy.deepcopy(prior.last_search)
    state["last_recommendation"] = copy.deepcopy(prior.last_recommendation)
    state["order"] = copy.deepcopy(prior.order)

    session = state["session"]
    request = state["request"]
    for source_key, target_key in (
        ("shopping_session_id", "shopping_session_id"),
        ("buyer_id", "buyer_id"),
        ("locale", "locale"),
        ("currency", "currency"),
    ):
        value = _value(intent, source_key)
        if _present(value):
            session[target_key] = _bounded(value)
    raw_query = _value(intent, "raw_query")
    if _present(raw_query):
        if not _present(request.get("session_origin_raw_query")):
            request["session_origin_raw_query"] = _bounded(raw_query)
        request["current_raw_query"] = _bounded(raw_query)

    normalized = [_event_parts(event) for event in events]
    search_invokes: dict[str, dict[str, Any]] = {}
    order_invokes: dict[str, list[dict[str, Any]]] = {}
    order_counter = 0
    used_prepare_invokes: set[int] = set()
    for kind, payload, _occurred in normalized:
        if kind != "tool.invoke":
            continue
        tool = str(payload.get("tool", ""))
        if tool == "product_search_tool":
            call_id = payload.get("tool_call_id")
            args = _compact_mapping(payload.get("args"), _SEARCH_ARGS)
            if _present(call_id) and args:
                search_invokes[str(call_id)] = args
        elif tool in {"prepare_order_tool", "query_order_tool"}:
            call_id = str(payload.get("tool_call_id") or f"order-{order_counter}")
            order_counter += 1
            order_invokes.setdefault(tool, []).append(
                {"call_id": call_id, "args": _bounded(payload.get("args"))}
            )

    last_successful_search: dict[str, Any] | None = None
    # Results are read in event order, so completion order naturally handles
    # parallel calls; a result without its invocation is intentionally ignored.
    for kind, payload, _occurred in normalized:
        if kind != "tool.result":
            continue
        tool = str(payload.get("tool", ""))
        if tool == "product_search_tool":
            call_id = payload.get("tool_call_id")
            if not _present(call_id) or str(call_id) not in search_invokes:
                continue
            output = _result_value(payload)
            if payload.get("error") or not isinstance(output, Mapping) or not isinstance(
                output.get("hits"), list
            ):
                continue
            hits = output.get("hits", [])
            refs = [
                str(card.get("item_id"))
                for card in hits
                if isinstance(card, Mapping) and _present(card.get("item_id"))
            ][:_MAX_REFS]
            returned_count = payload.get("hit_count", len(hits))
            if not isinstance(returned_count, int):
                returned_count = len(hits)
            last_successful_search = {
                "tool_call_id": str(call_id),
                "args": search_invokes[str(call_id)],
                "status": "succeeded",
                "returned_count": max(0, returned_count),
                "item_refs": refs,
            }
        elif tool == "prepare_order_tool":
            if payload.get("error"):
                continue
            output = _result_value(payload)
            if not isinstance(output, Mapping) or output.get("error"):
                continue
            order = copy.deepcopy(state.get("order") or {})
            invocation = None
            call_id = payload.get("tool_call_id")
            if _present(call_id):
                for index, item in enumerate(order_invokes.get(tool, [])):
                    if index not in used_prepare_invokes and item["call_id"] == str(call_id):
                        used_prepare_invokes.add(index)
                        invocation = item
                        break
            if invocation is None:
                for index, item in enumerate(order_invokes.get(tool, [])):
                    if index not in used_prepare_invokes:
                        used_prepare_invokes.add(index)
                        invocation = item
                        break
            if invocation is not None:
                args = _parse(invocation.get("args"))
                if isinstance(args, Mapping):
                    invoked_items = _compact_items(args.get("items"))
                    if invoked_items and not order.get("items"):
                        order["items"] = invoked_items
                    address = args.get("shipping_address")
                    if isinstance(address, Mapping):
                        country = address.get("country", address.get("country_code"))
                        if _present(country):
                            order["shipping_country"] = _bounded(country)
            items = _compact_items(output.get("items"))
            if items:
                order["items"] = items
            confirmation_id = output.get("confirmation_id", payload.get("confirmation_id"))
            confirmation_required = output.get(
                "confirmation_required", payload.get("confirmation_required")
            )
            if _present(confirmation_id):
                order["confirmation_id"] = _bounded(confirmation_id)
            if _present(confirmation_required):
                order["confirmation_status"] = (
                    "required" if confirmation_required else "not_required"
                )
            state["order"] = order or state.get("order")
        elif tool == "query_order_tool":
            if payload.get("error"):
                continue
            output = _result_value(payload)
            if not isinstance(output, Mapping) or output.get("error"):
                continue
            order = copy.deepcopy(state.get("order") or {})
            for key in ("order_id", "status"):
                if _present(output.get(key)):
                    order[key] = _bounded(output[key])
            items = _compact_items(output.get("lines") or output.get("items"))
            if items:
                order["items"] = items
            state["order"] = order or state.get("order")

    if last_successful_search is not None:
        state["last_search"] = last_successful_search

    for kind, payload, _occurred in normalized:
        if kind != "final.result":
            continue
        result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
        recommendation: dict[str, Any] = {}
        if _present(result.get("verification_status")):
            recommendation["verification_status"] = _bounded(result["verification_status"])
        if "recommended_cards" in result:
            recommendation["items"] = _compact_cards(result.get("recommended_cards"))
        elif "items" in result:
            recommendation["items"] = _compact_cards(result.get("items"))
        if recommendation:
            state["last_recommendation"] = recommendation

    candidate = L4Context.from_dict(state)
    if candidate.without_revision_metadata() == prior.without_revision_metadata():
        return prior
    state["revision"] = prior.revision + 1
    state["updated_at"] = _occurred_at(normalized, now or datetime.now(timezone.utc).isoformat())
    return L4Context.from_dict(state)


class L4Reducer:
    """Small object wrapper for dependency injection and checkpoint callers."""

    def reduce(
        self,
        intent: Any,
        events: Iterable[Any] = (),
        previous: L4Context | Mapping[str, Any] | None = None,
        *,
        now: str | None = None,
    ) -> L4Context:
        return reduce_l4(intent, events, previous, now=now)
