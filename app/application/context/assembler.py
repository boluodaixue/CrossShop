"""Four-layer AgentScope model view with a logical cache breakpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from agentscope.message import Msg, SystemMsg, ToolResultBlock

from .budget import ContextBudgetPolicy, decide_budget, measure_budget
from .compactor import freeze_settled_units, summarize_frozen_segments
from .lifecycle import settled_prefix_units
from .models import FrozenSegment, L4Context, StageSummary, canonical_json

CACHE_BREAKPOINT_TEXT = '<globex-cache-breakpoint version="v1" />'


def cache_prefix_sha256(messages: Sequence[Msg]) -> str:
    """Hash only fields that the OpenAI-compatible provider can observe.

    AgentScope message dumps also contain random ids and timestamps. Including
    them made the logical breakpoint change across identical provider prompts.
    L2/L3 prefix messages are text-only system messages, so role, name, and
    text are the complete provider-visible projection for this hash.
    """

    visible = [
        {
            "role": message.role,
            "name": message.name,
            "content": message.get_text_content() or "",
        }
        for message in messages
    ]
    return hashlib.sha256(canonical_json(visible).encode()).hexdigest()


def _l4(value: L4Context | Mapping[str, Any] | None) -> dict[str, Any]:
    return value.to_dict() if isinstance(value, L4Context) else dict(value or {})


def _history_payload(
    frozen: Sequence[FrozenSegment],
    stage: StageSummary | None,
) -> dict[str, Any]:
    covered = set(stage.source_segment_ids if stage else ())
    return {
        "stage_summary": stage.to_dict() if stage else None,
        "frozen_segments": [
            item.to_dict() for item in frozen if item.segment_id not in covered
        ],
    }


def assemble_llm_input_messages(
    *,
    fixed_messages: Sequence[Msg],
    active_messages: Sequence[Msg],
    frozen_segments: Sequence[FrozenSegment],
    session_context: L4Context | Mapping[str, Any] | None,
    stage_summary: StageSummary | Mapping[str, Any] | None = None,
) -> list[Msg]:
    """Build a non-destructive L3/L2/breakpoint/L4/active provider view."""

    assembled = list(fixed_messages)
    summary = (
        stage_summary
        if isinstance(stage_summary, StageSummary)
        else StageSummary.from_dict(stage_summary)
        if isinstance(stage_summary, Mapping)
        else None
    )
    covered: set[str] = set()
    if summary is not None:
        source_hashes = {item.segment_id: item.content_hash for item in frozen_segments}
        for source in summary.source_segments:
            if source_hashes.get(source["segment_id"]) != source["content_hash"]:
                raise ValueError(
                    f"stage summary source unavailable or changed: {source['segment_id']}"
                )
        covered = set(summary.source_segment_ids)
        assembled.append(
            SystemMsg(
                name="globex_context",
                content=f"<stage-summary>{canonical_json(summary.to_dict())}</stage-summary>",
            )
        )
    for segment in frozen_segments:
        if segment.segment_id not in covered:
            assembled.append(
                SystemMsg(
                    name="globex_context",
                    content=f'<frozen-interaction id="{segment.segment_id}">{segment.content}</frozen-interaction>',
                )
            )

    prefix_hash = cache_prefix_sha256(assembled)
    assembled.append(
        SystemMsg(
            name="globex_context",
            content=CACHE_BREAKPOINT_TEXT.replace(
                " />", f' prefix_sha256="{prefix_hash}" />'
            ),
        ),
    )
    context = _l4(session_context)
    if context:
        assembled.append(
            SystemMsg(
                name="globex_context",
                content=f"<session-context>{canonical_json(context)}</session-context>",
            )
        )
    assembled.extend(active_messages)
    return assembled


def advance_context_state(
    *,
    context_messages: Sequence[Msg],
    namespace: Mapping[str, Any] | None,
    fixed_messages: Sequence[Msg],
    tool_schemas: Any,
    policy: ContextBudgetPolicy,
    has_pending: bool,
    has_interrupt: bool,
) -> tuple[dict[str, Any], list[Msg]]:
    """Advance immutable L2/L3 records and return the next model-only view."""

    state = dict(namespace or {})
    cursor = max(0, int(state.get("freeze_cursor", 0)))
    existing = [
        item if isinstance(item, FrozenSegment) else FrozenSegment.from_dict(item)
        for item in state.get("frozen_segments") or []
        if isinstance(item, (FrozenSegment, Mapping))
    ]
    units = settled_prefix_units(
        context_messages,
        freeze_cursor=cursor,
        has_pending=has_pending,
        has_interrupt=has_interrupt,
    )
    session_context = state.get("session_context")
    revision = (
        int(session_context.get("revision", 0))
        if isinstance(session_context, Mapping)
        else 0
    )
    frozen, next_cursor = freeze_settled_units(
        context_messages,
        units,
        existing=existing,
        freeze_cursor=cursor,
        context_revision=revision,
    )
    stage_raw = state.get("stage_summary")
    stage = (
        StageSummary.from_dict(stage_raw) if isinstance(stage_raw, Mapping) else None
    )
    active = list(context_messages[next_cursor:])
    provisional = assemble_llm_input_messages(
        fixed_messages=fixed_messages,
        active_messages=active,
        frozen_segments=frozen,
        session_context=session_context
        if isinstance(session_context, Mapping)
        else None,
        stage_summary=stage,
    )
    tool_results: dict[str, Any] = {}
    for index, message in enumerate(active):
        for block in message.get_content_blocks("tool_result"):
            if isinstance(block, ToolResultBlock):
                tool_results[f"{block.name}:{block.id}:{index}"] = block.model_dump(
                    mode="json"
                )
    report = decide_budget(
        measure_budget(
            layers={
                "fixed_prompt": [
                    item.model_dump(mode="json") for item in fixed_messages
                ],
                "tool_schemas": tool_schemas,
                "l4_candidate": session_context or {},
                "history": _history_payload(frozen, stage),
                "active": [item.model_dump(mode="json") for item in active[:-1]],
                "current": active[-1].model_dump(mode="json") if active else {},
            },
            tool_results=tool_results,
            model_context_tokens=policy.model_context_tokens,
            reply_reserved_tokens=policy.reply_reserved_tokens,
            safety_margin_tokens=policy.safety_margin_tokens,
        ),
        policy=policy,
        has_frozen_history=bool(frozen),
    )
    if report.decision == "needs_l3":
        covered = set(stage.source_segment_ids if stage else ())
        uncovered = [item for item in frozen if item.segment_id not in covered]
        keep = policy.l3_keep_recent_frozen_segments
        eligible = (
            uncovered[:-keep]
            if keep and len(uncovered) > keep
            else ([] if keep else uncovered)
        )
        if eligible:
            before_tokens = report.total_input_tokens
            candidate_stage = summarize_frozen_segments(eligible, existing=stage)
            candidate_view = assemble_llm_input_messages(
                fixed_messages=fixed_messages,
                active_messages=active,
                frozen_segments=frozen,
                session_context=session_context
                if isinstance(session_context, Mapping)
                else None,
                stage_summary=candidate_stage,
            )
            candidate_report = decide_budget(
                measure_budget(
                    layers={
                        "fixed_prompt": [
                            item.model_dump(mode="json") for item in fixed_messages
                        ],
                        "tool_schemas": tool_schemas,
                        "l4_candidate": session_context or {},
                        "history": _history_payload(frozen, candidate_stage),
                        "active": [
                            item.model_dump(mode="json") for item in active[:-1]
                        ],
                        "current": (
                            active[-1].model_dump(mode="json") if active else {}
                        ),
                    },
                    tool_results=tool_results,
                    model_context_tokens=policy.model_context_tokens,
                    reply_reserved_tokens=policy.reply_reserved_tokens,
                    safety_margin_tokens=policy.safety_margin_tokens,
                ),
                policy=policy,
                has_frozen_history=bool(frozen),
            )
            if candidate_report.total_input_tokens < before_tokens:
                stage = candidate_stage
                provisional = candidate_view
                report = candidate_report
                state["context_compression"] = {
                    "action": "l3_stage_summary",
                    "before_tokens": before_tokens,
                    "after_tokens": report.total_input_tokens,
                    "source_segment_ids": list(stage.source_segment_ids),
                    "summary_hash": stage.summary_hash,
                }
            else:
                state["context_compression"] = {
                    "action": "l3_no_gain",
                    "before_tokens": before_tokens,
                    "candidate_tokens": candidate_report.total_input_tokens,
                }
    state.update(
        {
            "frozen_segments": [item.to_dict() for item in frozen],
            "freeze_cursor": next_cursor,
            "stage_summary": stage.to_dict() if stage else None,
            "budget_report": _budget_report_with_cache_prefix(
                report.to_dict(),
                provisional,
            ),
            "budget_decision": report.decision,
        },
    )
    return state, provisional


def _budget_report_with_cache_prefix(
    report: dict[str, Any],
    model_view: Sequence[Msg],
) -> dict[str, Any]:
    prefix: list[Msg] = []
    for message in model_view:
        if (message.get_text_content() or "").startswith(
            CACHE_BREAKPOINT_TEXT[:-3],
        ):
            break
        prefix.append(message)
    layers = report.get("layers")
    layers = layers if isinstance(layers, Mapping) else {}

    def _tokens(name: str) -> int:
        value = layers.get(name)
        if not isinstance(value, Mapping):
            return 0
        try:
            return max(0, int(value.get("tokens") or 0))
        except (TypeError, ValueError):
            return 0

    report["stable_prefix_sha256"] = cache_prefix_sha256(prefix)
    report["stable_prefix_estimated_tokens"] = (
        _tokens("fixed_prompt") + _tokens("tool_schemas") + _tokens("history")
    )
    return report
