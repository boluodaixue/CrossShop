"""Deterministic L2 compaction of already-settled interaction units."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from .lifecycle import InteractionUnit
from .models import FrozenSegment, StageSummary

_MAX_TEXT = 2000
_MAX_REFS = 20
_MAX_STAGE_ASSISTANT = 512


def _content_text(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content[:_MAX_TEXT]
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
            for part in content
        )[:_MAX_TEXT]
    return str(content)[:_MAX_TEXT]


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return None
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, Mapping):
        return {
            str(key): bounded
            for key, child in list(value.items())[:32]
            if (bounded := _bounded(child, depth=depth + 1)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(child, depth=depth + 1) for child in value[:20]]
    return value


def _parse(value: Any) -> Any:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        text = "".join(
            str(part.get("text", "")) if isinstance(part, Mapping) else str(part)
            for part in content
        ).strip()
    else:
        text = str(content).strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _result_summary(result: Any) -> tuple[str, int | None, list[str]]:
    parsed = _parse(result)
    status = str(getattr(result, "status", "success"))
    count: int | None = None
    refs: list[str] = []
    if isinstance(parsed, Mapping):
        if parsed.get("error"):
            status = "error"
        for key in ("returned_count", "hit_count", "count"):
            if isinstance(parsed.get(key), int):
                count = int(parsed[key])
                break
        for key in ("hits", "results", "items", "lines"):
            rows = parsed.get(key)
            if not isinstance(rows, list):
                continue
            if count is None:
                count = len(rows)
            refs.extend(
                str(row[ref_key])
                for row in rows
                if isinstance(row, Mapping)
                for ref_key in ("item_id", "order_id", "confirmation_id")
                if row.get(ref_key) not in (None, "")
            )
            break
        for ref_key in ("item_id", "order_id", "confirmation_id"):
            if parsed.get(ref_key) not in (None, ""):
                refs.append(str(parsed[ref_key]))
    elif isinstance(parsed, str) and parsed.casefold().startswith("[error]"):
        status = "error"
    return status, count, list(dict.fromkeys(refs))[:_MAX_REFS]


def compact_interaction(unit: InteractionUnit, *, context_revision: int) -> FrozenSegment:
    """Build one deterministic, bounded FrozenSegment without an LLM."""

    if not unit.complete or unit.final_answer is None:
        raise ValueError("only complete interaction units can be frozen")
    tools: list[dict[str, Any]] = []
    for interaction in unit.tools:
        status, returned_count, refs = _result_summary(interaction.result)
        summary: dict[str, Any] = {
            "tool": interaction.name,
            "tool_call_id": interaction.tool_call_id,
            "args": _bounded(interaction.args),
            "status": status,
        }
        if returned_count is not None:
            summary["returned_count"] = returned_count
        if refs:
            summary["refs"] = refs
        tools.append(summary)
    content = json.dumps(
        {
            "user": _content_text(unit.human),
            "tools": tools,
            "assistant": _content_text(unit.final_answer),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return FrozenSegment(
        segment_id=f"frozen-{unit.start}-{unit.end}",
        content=content,
        source_message_start=unit.start,
        source_message_end=unit.end,
        context_revision=context_revision,
    )


def freeze_settled_units(
    messages: Sequence[BaseMessage],
    units: Sequence[InteractionUnit],
    *,
    existing: Sequence[FrozenSegment] = (),
    freeze_cursor: int = 0,
    context_revision: int = 0,
) -> tuple[list[FrozenSegment], int]:
    """Append immutable segments and move the cursor forward monotonically."""

    del messages  # Unit message references already contain every compacted fact.
    segments = list(existing)
    cursor = max(0, freeze_cursor)
    existing_ranges = {
        (segment.source_message_start, segment.source_message_end): segment
        for segment in segments
    }
    for unit in units:
        if unit.end <= cursor:
            continue
        if unit.start != cursor:
            break
        key = (unit.start, unit.end)
        if key in existing_ranges:
            cursor = max(cursor, unit.end)
            continue
        segment = compact_interaction(unit, context_revision=context_revision)
        segments.append(segment)
        existing_ranges[key] = segment
        cursor = unit.end
    return segments, max(freeze_cursor, cursor)


def summarize_frozen_segments(
    segments: Sequence[FrozenSegment],
    *,
    existing: StageSummary | Mapping[str, Any] | None = None,
) -> StageSummary:
    """Create one deterministic L3 summary from frozen inputs only.

    The source segments remain immutable and are merely covered in the model
    view. An existing summary can be extended by newly frozen segments; replay
    of the same source id/hash set returns the exact same summary.
    """

    current = (
        existing
        if isinstance(existing, StageSummary)
        else StageSummary.from_dict(existing)
        if isinstance(existing, Mapping)
        else None
    )
    source_segments = list(current.source_segments if current else ())
    user_requests = list(current.user_requests if current else ())
    tool_facts = [dict(item) for item in (current.tool_facts if current else ())]
    assistant_outcomes = list(current.assistant_outcomes if current else ())
    known_sources = {item["segment_id"]: item["content_hash"] for item in source_segments}
    tool_indexes = {_canonical_tool(item): index for index, item in enumerate(tool_facts)}
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
        known_hash = known_sources.get(segment.segment_id)
        if known_hash is not None:
            if known_hash != segment.content_hash:
                raise ValueError(f"frozen segment hash changed: {segment.segment_id}")
            continue
        try:
            content = json.loads(segment.content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid frozen segment JSON: {segment.segment_id}") from exc
        if not isinstance(content, Mapping) or not isinstance(content.get("user"), str):
            raise ValueError(f"frozen segment lacks original user request: {segment.segment_id}")
        user_requests.append(content["user"])
        tools = content.get("tools") or []
        if not isinstance(tools, list):
            raise ValueError(f"invalid frozen tool facts: {segment.segment_id}")
        for tool_fact in tools:
            if not isinstance(tool_fact, Mapping):
                raise ValueError(f"invalid frozen tool fact: {segment.segment_id}")
            fact = dict(tool_fact)
            call_ids = _tool_call_ids(fact)
            fact.pop("tool_call_id", None)
            fact.pop("tool_call_ids", None)
            key = _canonical_tool(fact)
            existing_index = tool_indexes.get(key)
            if existing_index is None:
                if call_ids:
                    fact["tool_call_ids"] = call_ids
                tool_facts.append(fact)
                tool_indexes[key] = len(tool_facts) - 1
            elif call_ids:
                existing_fact = tool_facts[existing_index]
                existing_ids = _tool_call_ids(existing_fact)
                existing_fact["tool_call_ids"] = list(
                    dict.fromkeys([*existing_ids, *call_ids])
                )
        assistant = content.get("assistant")
        if assistant not in (None, ""):
            outcome = str(assistant)[:_MAX_STAGE_ASSISTANT]
            if outcome not in assistant_outcomes:
                assistant_outcomes.append(outcome)
        source = {
            "segment_id": segment.segment_id,
            "content_hash": segment.content_hash,
        }
        source_segments.append(source)
        known_sources[segment.segment_id] = segment.content_hash
        if segment.source_message_start is not None:
            starts.append(segment.source_message_start)
        if segment.source_message_end is not None:
            ends.append(segment.source_message_end)
        revisions.append(segment.context_revision)

    if not source_segments:
        raise ValueError("at least one frozen segment is required for L3")
    return StageSummary(
        source_segments=tuple(source_segments),
        user_requests=tuple(user_requests),
        tool_facts=tuple(tool_facts),
        assistant_outcomes=tuple(assistant_outcomes),
        source_message_start=min(starts) if starts else None,
        source_message_end=max(ends) if ends else None,
        context_revision=max(revisions, default=0),
    )


def _canonical_tool(value: Mapping[str, Any]) -> str:
    content = {
        key: child
        for key, child in value.items()
        if key not in {"tool_call_id", "tool_call_ids"}
    }
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _tool_call_ids(value: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    if value.get("tool_call_id") not in (None, ""):
        ids.append(str(value["tool_call_id"]))
    raw_ids = value.get("tool_call_ids")
    if isinstance(raw_ids, (list, tuple)):
        ids.extend(str(item) for item in raw_ids if item not in (None, ""))
    return list(dict.fromkeys(ids))
