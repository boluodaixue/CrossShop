"""Message lifecycle grouping for complete, protocol-safe interaction units."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


@dataclass(frozen=True)
class ToolInteraction:
    name: str
    tool_call_id: str
    args: Any
    call_message_index: int
    result_message_index: int
    result: ToolMessage


@dataclass(frozen=True)
class InteractionUnit:
    """One human turn with complete Assistant/Tool protocol boundaries."""

    start: int
    end: int
    human: HumanMessage
    final_answer: AIMessage | None
    tools: tuple[ToolInteraction, ...]
    complete: bool
    incomplete_reason: str = ""


def _tool_call_parts(call: Any) -> tuple[str, str, Any]:
    if isinstance(call, dict):
        return str(call.get("name", "")), str(call.get("id", "")), call.get("args", {})
    return (
        str(getattr(call, "name", "")),
        str(getattr(call, "id", "")),
        getattr(call, "args", {}),
    )


def group_interaction_units(messages: Sequence[BaseMessage]) -> list[InteractionUnit]:
    """Group messages by HumanMessage without ever accepting orphan tool results."""

    starts = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
    units: list[InteractionUnit] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(messages)
        units.append(_build_unit(messages, start, end))
    return units


def _build_unit(
    messages: Sequence[BaseMessage],
    start: int,
    end: int,
) -> InteractionUnit:
    calls: dict[str, tuple[str, Any, int]] = {}
    call_order: list[str] = []
    results: dict[str, tuple[int, ToolMessage]] = {}
    duplicate_call = False
    orphan_result = False
    final_answer: AIMessage | None = None
    final_index = -1

    for index in range(start + 1, end):
        message = messages[index]
        if isinstance(message, AIMessage):
            if message.tool_calls:
                for call in message.tool_calls:
                    name, call_id, args = _tool_call_parts(call)
                    if not call_id or call_id in calls:
                        duplicate_call = True
                        continue
                    calls[call_id] = (name, args, index)
                    call_order.append(call_id)
            else:
                final_answer = message
                final_index = index
        elif isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id)
            if not call_id or call_id not in calls or call_id in results:
                orphan_result = True
                continue
            results[call_id] = (index, message)

    missing_results = [call_id for call_id in call_order if call_id not in results]
    unconsumed_results = [
        call_id
        for call_id, (result_index, _message) in results.items()
        if not any(
            isinstance(messages[index], AIMessage)
            for index in range(result_index + 1, end)
        )
    ]
    final_after_results = not results or (
        final_index > max(result_index for result_index, _message in results.values())
    )
    reasons: list[str] = []
    if duplicate_call:
        reasons.append("duplicate_tool_call_id")
    if orphan_result:
        reasons.append("orphan_tool_result")
    if missing_results:
        reasons.append("missing_tool_result")
    if unconsumed_results:
        reasons.append("unconsumed_tool_result")
    if final_answer is None or not final_after_results:
        reasons.append("missing_final_answer")

    interactions = tuple(
        ToolInteraction(
            name=calls[call_id][0],
            tool_call_id=call_id,
            args=calls[call_id][1],
            call_message_index=calls[call_id][2],
            result_message_index=results[call_id][0],
            result=results[call_id][1],
        )
        for call_id in call_order
        if call_id in results
    )
    return InteractionUnit(
        start=start,
        end=end,
        human=messages[start],
        final_answer=final_answer,
        tools=interactions,
        complete=not reasons,
        incomplete_reason=",".join(reasons),
    )


def settled_prefix_units(
    messages: Sequence[BaseMessage],
    *,
    freeze_cursor: int = 0,
    has_pending: bool = False,
    has_interrupt: bool = False,
) -> list[InteractionUnit]:
    """Return only the continuous complete prefix before the current human turn."""

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
