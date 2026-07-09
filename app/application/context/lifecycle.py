"""AgentScope message lifecycle grouping for protocol-safe L2 freezing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentscope.message import (
    AssistantMsg,
    Msg,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
)

_NON_BUYER_USER_MESSAGE_NAMES = frozenset({"memory_hint"})
_STRUCTURED_OUTPUT_TOOL = "GenerateStructuredOutput"


@dataclass(frozen=True)
class ToolInteraction:
    name: str
    tool_call_id: str
    args: Any
    call_message_index: int
    result_message_index: int
    result: ToolResultBlock


@dataclass(frozen=True)
class InteractionUnit:
    start: int
    end: int
    human: Msg
    final_answer: Msg | None
    tools: tuple[ToolInteraction, ...]
    complete: bool
    incomplete_reason: str = ""


def _parse_args(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def group_interaction_units(messages: Sequence[Msg]) -> list[InteractionUnit]:
    """Group actual buyer turns without treating injected hints as turns.

    AgentScope represents the approved preference hint as
    ``UserMsg(name="memory_hint")`` so the provider can see it immediately
    before the buyer request. It is still part of the raw message range, but
    it must not become an interaction boundary of its own.
    """

    buyer_starts = [
        index
        for index, message in enumerate(messages)
        if _is_buyer_user_message(message)
    ]
    starts = [
        _turn_start(messages, buyer_index, previous_buyer_index)
        for buyer_index, previous_buyer_index in zip(
            buyer_starts,
            [None, *buyer_starts[:-1]],
            strict=True,
        )
    ]
    return [
        _build_unit(
            messages,
            start,
            buyer_starts[position],
            starts[position + 1] if position + 1 < len(starts) else len(messages),
        )
        for position, start in enumerate(starts)
    ]


def _is_buyer_user_message(message: Msg) -> bool:
    return message.role == "user" and message.name not in _NON_BUYER_USER_MESSAGE_NAMES


def _turn_start(
    messages: Sequence[Msg],
    buyer_index: int,
    previous_buyer_index: int | None,
) -> int:
    """Attach immediately preceding injected hints to the upcoming turn."""

    lower_bound = 0 if previous_buyer_index is None else previous_buyer_index + 1
    start = buyer_index
    while start > lower_bound:
        candidate = messages[start - 1]
        if (
            candidate.role != "user"
            or candidate.name not in _NON_BUYER_USER_MESSAGE_NAMES
        ):
            break
        start -= 1
    return start


def _build_unit(
    messages: Sequence[Msg],
    start: int,
    human_index: int,
    end: int,
) -> InteractionUnit:
    calls: dict[str, tuple[ToolCallBlock, int]] = {}
    call_order: list[str] = []
    results: dict[str, tuple[ToolResultBlock, int]] = {}
    reasons: list[str] = []
    final: Msg | None = None

    for index in range(human_index + 1, end):
        message = messages[index]
        for block in message.get_content_blocks("tool_call"):
            if not block.id or block.id in calls:
                reasons.append("duplicate_tool_call_id")
                continue
            calls[block.id] = (block, index)
            call_order.append(block.id)
        for block in message.get_content_blocks("tool_result"):
            if not block.id or block.id not in calls or block.id in results:
                reasons.append("orphan_tool_result")
                continue
            results[block.id] = (block, index)
            if block.state != ToolResultState.SUCCESS:
                reasons.append("failed_tool_result")
        if message.role == "assistant" and message.get_text_content():
            final = message

    missing = [call_id for call_id in call_order if call_id not in results]
    if missing:
        reasons.append("missing_tool_result")
    if final is None:
        final = _structured_final_answer(messages, calls, results, call_order)
    if final is None:
        reasons.append("missing_final_answer")
    interactions = tuple(
        ToolInteraction(
            name=calls[call_id][0].name,
            tool_call_id=call_id,
            args=_parse_args(calls[call_id][0].input),
            call_message_index=calls[call_id][1],
            result_message_index=results[call_id][1],
            result=results[call_id][0],
        )
        for call_id in call_order
        if call_id in results
    )
    return InteractionUnit(
        start=start,
        end=end,
        human=messages[human_index],
        final_answer=final,
        tools=interactions,
        complete=not reasons,
        incomplete_reason=",".join(dict.fromkeys(reasons)),
    )


def _structured_final_answer(
    messages: Sequence[Msg],
    calls: Mapping[str, tuple[ToolCallBlock, int]],
    results: Mapping[str, tuple[ToolResultBlock, int]],
    call_order: Sequence[str],
) -> Msg | None:
    """Project a successful structured-output call into a compactable answer.

    AgentScope stores ``final_text`` in the built-in tool call arguments and a
    success acknowledgement in the paired result. The projection is used only
    by L2/L3 compaction; raw AgentState messages remain untouched.
    """

    for call_id in reversed(call_order):
        call, message_index = calls[call_id]
        if call.name != _STRUCTURED_OUTPUT_TOOL or call_id not in results:
            continue
        result = results[call_id][0]
        if str(getattr(result.state, "value", result.state)) != "success":
            continue
        payload = _parse_args(call.input)
        if not isinstance(payload, Mapping):
            continue
        final_text = payload.get("final_text")
        if not isinstance(final_text, str) or not final_text.strip():
            continue
        source = messages[message_index]
        return AssistantMsg(name=source.name, content=final_text.strip())
    return None


def settled_prefix_units(
    messages: Sequence[Msg],
    *,
    freeze_cursor: int = 0,
    has_pending: bool = False,
    has_interrupt: bool = False,
) -> list[InteractionUnit]:
    """Return the continuous complete prefix before the current user turn."""

    if has_pending or has_interrupt:
        return []
    units = group_interaction_units(messages)
    if len(units) < 2:
        return []
    settled: list[InteractionUnit] = []
    cursor = max(0, freeze_cursor)
    for unit in units[:-1]:
        if unit.end <= cursor:
            continue
        if unit.start != cursor or not unit.complete:
            break
        settled.append(unit)
        cursor = unit.end
    return settled
