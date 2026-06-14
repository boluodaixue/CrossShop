"""Four-layer model-view assembly with a logical cache breakpoint."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .budget import (
    ContextBudgetExceeded,
    ContextBudgetPolicy,
    decide_budget,
    measure_budget,
)
from .compactor import freeze_settled_units, summarize_frozen_segments
from .lifecycle import settled_prefix_units
from .models import FrozenSegment, L4Context, StageSummary

CACHE_BREAKPOINT_TEXT = "<globex-cache-breakpoint version=\"v1\" />"


def _context_dict(value: L4Context | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, L4Context):
        return value.to_dict()
    return dict(value or {})


def assemble_llm_input_messages(
    *,
    messages: Sequence[BaseMessage],
    frozen_segments: Sequence[FrozenSegment],
    freeze_cursor: int,
    session_context: L4Context | Mapping[str, Any] | None,
    stage_summary: StageSummary | Mapping[str, Any] | str | None = None,
) -> list[BaseMessage]:
    """Assemble frozen prefix -> breakpoint -> L4 -> untouched active/current."""

    assembled: list[BaseMessage] = []
    covered_ids: set[str] = set()
    parsed_summary: StageSummary | None = None
    if stage_summary:
        if isinstance(stage_summary, StageSummary):
            summary_payload: Any = stage_summary.to_dict()
            parsed_summary = stage_summary
        else:
            summary_payload = stage_summary
            if isinstance(stage_summary, Mapping):
                try:
                    parsed_summary = StageSummary.from_dict(stage_summary)
                except ValueError:
                    parsed_summary = None
        if parsed_summary is not None:
            source_hashes = {
                segment.segment_id: segment.content_hash for segment in frozen_segments
            }
            for source in parsed_summary.source_segments:
                if source_hashes.get(source["segment_id"]) != source["content_hash"]:
                    raise ValueError(
                        f"stage summary source unavailable or changed: {source['segment_id']}"
                    )
            covered_ids = set(parsed_summary.source_segment_ids)
        summary_text = stage_summary if isinstance(stage_summary, str) else json.dumps(
            summary_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        assembled.append(SystemMessage(content=f"<stage-summary>{summary_text}</stage-summary>"))
    assembled.extend(
        SystemMessage(
            content=f"<frozen-interaction id=\"{segment.segment_id}\">"
            f"{segment.content}</frozen-interaction>"
        )
        for segment in frozen_segments
        if segment.segment_id not in covered_ids
    )
    assembled.append(SystemMessage(content=CACHE_BREAKPOINT_TEXT))
    context = _context_dict(session_context)
    if context:
        assembled.append(
            SystemMessage(
                content="<session-context>"
                + json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "</session-context>"
            )
        )
    assembled.extend(messages[max(0, freeze_cursor) :])
    return assembled


def _partition_active(
    messages: Sequence[BaseMessage],
) -> tuple[list[BaseMessage], list[BaseMessage], dict[str, list[BaseMessage]]]:
    active: list[BaseMessage] = []
    current: list[BaseMessage] = []
    tool_results: dict[str, list[BaseMessage]] = {}
    last_human = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=len(messages),
    )
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            call_id = str(getattr(message, "tool_call_id", "") or f"index-{index}")
            tool_name = str(getattr(message, "name", "") or "tool")
            tool_results[f"{tool_name}:{call_id}:{index}"] = [message]
        elif index < last_human:
            active.append(message)
        else:
            current.append(message)
    return active, current, tool_results


def _measure_model_view(
    *,
    model: Any,
    fixed_prompt: str,
    tool_schemas: Any,
    frozen_segments: Sequence[FrozenSegment],
    stage_summary: StageSummary | Mapping[str, Any] | str | None,
    session_context: Mapping[str, Any] | None,
    active_messages: Sequence[BaseMessage],
    policy: ContextBudgetPolicy,
):
    history_messages = assemble_llm_input_messages(
        messages=[],
        frozen_segments=frozen_segments,
        freeze_cursor=0,
        session_context=None,
        stage_summary=stage_summary,
    )
    l4_messages = []
    if session_context:
        l4_messages = assemble_llm_input_messages(
            messages=[],
            frozen_segments=[],
            freeze_cursor=0,
            session_context=session_context,
        )[1:]
    active, current, tool_results = _partition_active(active_messages)
    report = measure_budget(
        fixed_prompt=[SystemMessage(content=fixed_prompt)] if fixed_prompt else [],
        tool_schemas=tool_schemas,
        l4_candidate=l4_messages,
        history=history_messages,
        active=active,
        current=current,
        tool_results=tool_results,
        model_context_tokens=policy.model_context_tokens,
        reply_reserved_tokens=policy.reply_reserved_tokens,
        safety_margin_tokens=policy.safety_margin_tokens,
        model=model,
    )
    return decide_budget(
        report,
        policy=policy,
        has_frozen_history=bool(frozen_segments or stage_summary),
    )


def _stage_summary(value: Any) -> StageSummary | None:
    if isinstance(value, StageSummary):
        return value
    if isinstance(value, Mapping):
        return StageSummary.from_dict(value)
    return None


def _l3_candidates(
    frozen_segments: Sequence[FrozenSegment],
    stage_summary: StageSummary | None,
    *,
    keep_recent: int,
) -> list[FrozenSegment]:
    covered = set(stage_summary.source_segment_ids if stage_summary else ())
    uncovered = [segment for segment in frozen_segments if segment.segment_id not in covered]
    if keep_recent:
        return uncovered[:-keep_recent] if len(uncovered) > keep_recent else []
    return uncovered


def context_pre_model_hook(
    state: Mapping[str, Any],
    *,
    model: Any | None = None,
    fixed_prompt: str = "",
    tool_schemas: Any = "",
    budget_policy: ContextBudgetPolicy | None = None,
) -> dict[str, Any]:
    """Advance L2 only for settled units and return a non-destructive model view."""

    messages = list(state.get("messages") or [])
    cursor = max(0, int(state.get("freeze_cursor") or 0))
    existing = [
        item if isinstance(item, FrozenSegment) else FrozenSegment.from_dict(item)
        for item in state.get("frozen_segments") or []
        if isinstance(item, (FrozenSegment, Mapping))
    ]
    units = settled_prefix_units(
        messages,
        freeze_cursor=cursor,
        has_pending=bool(state.get("context_has_pending")),
        has_interrupt=bool(state.get("context_has_interrupt")),
    )
    context = state.get("session_context")
    context_revision = (
        int(context.get("revision", 0)) if isinstance(context, Mapping) else 0
    )
    frozen, next_cursor = freeze_settled_units(
        messages,
        units,
        existing=existing,
        freeze_cursor=cursor,
        context_revision=context_revision,
    )
    active_messages = messages[next_cursor:]
    stage_value = state.get("stage_summary")
    update: dict[str, Any] = {
        "llm_input_messages": assemble_llm_input_messages(
            messages=messages,
            frozen_segments=frozen,
            freeze_cursor=next_cursor,
            session_context=context if isinstance(context, Mapping) else None,
            stage_summary=stage_value,
        ),
        "frozen_segments": [segment.to_dict() for segment in frozen],
        "freeze_cursor": next_cursor,
    }
    if budget_policy is not None:
        report = _measure_model_view(
            model=model,
            fixed_prompt=fixed_prompt,
            tool_schemas=tool_schemas,
            frozen_segments=frozen,
            stage_summary=stage_value,
            session_context=context if isinstance(context, Mapping) else None,
            active_messages=active_messages,
            policy=budget_policy,
        )
        if (
            report.decision == "needs_l3"
            and budget_policy.mode == "enforce"
            and not report.estimated
            and budget_policy.soft_limit_tokens > 0
        ):
            current_summary = _stage_summary(stage_value) if stage_value else None
            candidates = _l3_candidates(
                frozen,
                current_summary,
                keep_recent=budget_policy.l3_keep_recent_frozen_segments,
            )
            if not candidates:
                report_payload = report.to_dict()
                report_payload["decision"] = "needs_l3_no_eligible"
                update["budget_report"] = report_payload
                update["budget_decision"] = "needs_l3_no_eligible"
            else:
                candidate_summary = summarize_frozen_segments(
                    candidates,
                    existing=current_summary,
                )
                recounted = _measure_model_view(
                    model=model,
                    fixed_prompt=fixed_prompt,
                    tool_schemas=tool_schemas,
                    frozen_segments=frozen,
                    stage_summary=candidate_summary,
                    session_context=context if isinstance(context, Mapping) else None,
                    active_messages=active_messages,
                    policy=budget_policy,
                )
                if recounted.total_input_tokens < report.total_input_tokens:
                    stage_value = candidate_summary.to_dict()
                    update["stage_summary"] = stage_value
                    update["llm_input_messages"] = assemble_llm_input_messages(
                        messages=messages,
                        frozen_segments=frozen,
                        freeze_cursor=next_cursor,
                        session_context=context if isinstance(context, Mapping) else None,
                        stage_summary=stage_value,
                    )
                    action_decision = recounted.decision
                    if action_decision == "needs_l3":
                        action_decision = "needs_l3_recent_protected"
                    report_payload = recounted.to_dict()
                    report_payload["decision"] = action_decision
                    update["budget_report"] = report_payload
                    update["budget_decision"] = action_decision
                    update["context_compression"] = {
                        "action": "l3_stage_summary",
                        "before_tokens": report.total_input_tokens,
                        "after_tokens": recounted.total_input_tokens,
                        "source_segment_ids": list(candidate_summary.source_segment_ids),
                        "summary_hash": candidate_summary.summary_hash,
                    }
                    report = recounted
                else:
                    report_payload = report.to_dict()
                    report_payload["decision"] = "needs_l3_no_gain"
                    update["budget_report"] = report_payload
                    update["budget_decision"] = "needs_l3_no_gain"
        else:
            update["budget_report"] = report.to_dict()
            update["budget_decision"] = report.decision
        if (
            budget_policy.mode == "enforce"
            and not report.estimated
            and report.decision in {"active_overflow", "fixed_overflow"}
        ):
            raise ContextBudgetExceeded(report.decision, report)
        if (
            budget_policy.mode == "enforce"
            and not report.estimated
            and report.total_input_tokens > report.available_input_tokens
        ):
            # L3 may only rewrite old frozen history.  If the exact recount still
            # exceeds the hard model window (including when the recent-history
            # fidelity window leaves no eligible segment), never pass an
            # oversized request to the model or touch active/L4 content.
            raise ContextBudgetExceeded("needs_l3_hard_overflow", report)
    return update


def build_context_pre_model_hook(
    *,
    model: Any,
    fixed_prompt: str,
    tool_schemas: Any,
    budget_policy: ContextBudgetPolicy,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Bind the exact main model, prompt, schemas and measured configuration."""

    def _hook(state: Mapping[str, Any]) -> dict[str, Any]:
        return context_pre_model_hook(
            state,
            model=model,
            fixed_prompt=fixed_prompt,
            tool_schemas=tool_schemas,
            budget_policy=budget_policy,
        )

    return _hook
