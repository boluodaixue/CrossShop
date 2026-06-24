"""Deterministic L2 freezing and L3 summarization for AgentScope messages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentscope.message import Msg, TextBlock, ToolResultBlock

from .lifecycle import InteractionUnit
from .models import FrozenSegment, StageSummary, canonical_json

_MAX_TEXT = 2_000
_MAX_REFS = 20
_MAX_STAGE_ASSISTANT = 512


def _message_text(message: Msg) -> str:
    return (message.get_text_content() or "")[:_MAX_TEXT]


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return None
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, Mapping):
        return {
            str(key): child_bounded
            for key, child in list(value.items())[:32]
            if (child_bounded := _bounded(child, depth=depth + 1)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(child, depth=depth + 1) for child in value[:20]]
    return value


def _result_text(result: ToolResultBlock) -> str:
    if isinstance(result.output, str):
        return result.output
    return "".join(
        block.text for block in result.output if isinstance(block, TextBlock)
    )


def _result_summary(result: ToolResultBlock) -> tuple[str, int | None, list[str]]:
    text = _result_text(result).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    status = str(getattr(result.state, "value", result.state))
    count: int | None = None
    refs: list[str] = []
    if isinstance(parsed, Mapping):
        if parsed.get("error"):
            status = "error"
        for key in ("returned_count", "hit_count", "count"):
            if isinstance(parsed.get(key), int):
                count = int(parsed[key])
                break
        for key in ("hits", "filtered_out", "items", "lines"):
            rows = parsed.get(key)
            if not isinstance(rows, list):
                continue
            if count is None and key in {"hits", "items", "lines"}:
                count = len(rows)
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                for ref_key in ("product_id", "sku_id", "order_id", "confirmation_id"):
                    if row.get(ref_key) not in (None, ""):
                        refs.append(str(row[ref_key]))
            break
        for ref_key in ("product_id", "sku_id", "order_id", "confirmation_id"):
            if parsed.get(ref_key) not in (None, ""):
                refs.append(str(parsed[ref_key]))
    return status, count, list(dict.fromkeys(refs))[:_MAX_REFS]


def compact_interaction(
    unit: InteractionUnit, *, context_revision: int
) -> FrozenSegment:
    if not unit.complete or unit.final_answer is None:
        raise ValueError("only complete successful interaction units can be frozen")
    tools: list[dict[str, Any]] = []
    for interaction in unit.tools:
        status, returned_count, refs = _result_summary(interaction.result)
        fact: dict[str, Any] = {
            "tool": interaction.name,
            "tool_call_id": interaction.tool_call_id,
            "args": _bounded(interaction.args),
            "status": status,
        }
        if returned_count is not None:
            fact["returned_count"] = returned_count
        if refs:
            fact["refs"] = refs
        tools.append(fact)
    content = canonical_json(
        {
            "user": _message_text(unit.human),
            "tools": tools,
            "assistant": _message_text(unit.final_answer),
        },
    )
    return FrozenSegment(
        segment_id=f"frozen-{unit.start}-{unit.end}",
        content=content,
        source_message_start=unit.start,
        source_message_end=unit.end,
        context_revision=context_revision,
    )


def freeze_settled_units(
    messages: Sequence[Msg],
    units: Sequence[InteractionUnit],
    *,
    existing: Sequence[FrozenSegment] = (),
    freeze_cursor: int = 0,
    context_revision: int = 0,
) -> tuple[list[FrozenSegment], int]:
    del messages
    segments = list(existing)
    cursor = max(0, freeze_cursor)
    ranges = {
        (item.source_message_start, item.source_message_end): item for item in segments
    }
    for unit in units:
        if unit.end <= cursor:
            continue
        if unit.start != cursor:
            break
        key = (unit.start, unit.end)
        if key not in ranges:
            segment = compact_interaction(unit, context_revision=context_revision)
            segments.append(segment)
            ranges[key] = segment
        cursor = unit.end
    return segments, max(freeze_cursor, cursor)


def summarize_frozen_segments(
    segments: Sequence[FrozenSegment],
    *,
    existing: StageSummary | Mapping[str, Any] | None = None,
) -> StageSummary:
    current = (
        existing
        if isinstance(existing, StageSummary)
        else StageSummary.from_dict(existing)
        if isinstance(existing, Mapping)
        else None
    )
    sources = list(current.source_segments if current else ())
    requests = list(current.user_requests if current else ())
    facts = [dict(item) for item in (current.tool_facts if current else ())]
    outcomes = list(current.assistant_outcomes if current else ())
    known = {item["segment_id"]: item["content_hash"] for item in sources}
    fact_indexes = {_canonical_tool(item): index for index, item in enumerate(facts)}
    starts = (
        [current.source_message_start]
        if current and current.source_message_start is not None
        else []
    )
    ends = (
        [current.source_message_end]
        if current and current.source_message_end is not None
        else []
    )
    revisions = [current.context_revision] if current else []

    for segment in segments:
        known_hash = known.get(segment.segment_id)
        if known_hash is not None:
            if known_hash != segment.content_hash:
                raise ValueError(f"frozen segment hash changed: {segment.segment_id}")
            continue
        try:
            content = json.loads(segment.content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid frozen segment JSON: {segment.segment_id}"
            ) from exc
        if not isinstance(content, Mapping) or not isinstance(content.get("user"), str):
            raise TypeError(
                f"frozen segment lacks original user request: {segment.segment_id}"
            )
        requests.append(content["user"])
        for raw_fact in content.get("tools") or []:
            if not isinstance(raw_fact, Mapping):
                raise TypeError(f"invalid frozen tool fact: {segment.segment_id}")
            fact = dict(raw_fact)
            call_ids = _tool_call_ids(fact)
            fact.pop("tool_call_id", None)
            fact.pop("tool_call_ids", None)
            key = _canonical_tool(fact)
            index = fact_indexes.get(key)
            if index is None:
                if call_ids:
                    fact["tool_call_ids"] = call_ids
                facts.append(fact)
                fact_indexes[key] = len(facts) - 1
            elif call_ids:
                facts[index]["tool_call_ids"] = list(
                    dict.fromkeys([*_tool_call_ids(facts[index]), *call_ids])
                )
        assistant = content.get("assistant")
        if (
            assistant not in (None, "")
            and str(assistant)[:_MAX_STAGE_ASSISTANT] not in outcomes
        ):
            outcomes.append(str(assistant)[:_MAX_STAGE_ASSISTANT])
        sources.append(
            {"segment_id": segment.segment_id, "content_hash": segment.content_hash}
        )
        known[segment.segment_id] = segment.content_hash
        if segment.source_message_start is not None:
            starts.append(segment.source_message_start)
        if segment.source_message_end is not None:
            ends.append(segment.source_message_end)
        revisions.append(segment.context_revision)

    if not sources:
        raise ValueError("at least one frozen segment is required for L3")
    return StageSummary(
        source_segments=tuple(sources),
        user_requests=tuple(requests),
        tool_facts=tuple(facts),
        assistant_outcomes=tuple(outcomes),
        source_message_start=min(starts) if starts else None,
        source_message_end=max(ends) if ends else None,
        context_revision=max(revisions, default=0),
    )


def _canonical_tool(value: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            key: child
            for key, child in value.items()
            if key not in {"tool_call_id", "tool_call_ids"}
        }
    )


def _tool_call_ids(value: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    if value.get("tool_call_id") not in (None, ""):
        result.append(str(value["tool_call_id"]))
    raw = value.get("tool_call_ids")
    if isinstance(raw, (list, tuple)):
        result.extend(str(item) for item in raw if item not in (None, ""))
    return list(dict.fromkeys(result))
