"""Four-layer AgentScope model view with a logical cache breakpoint."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from agentscope.message import Msg, SystemMsg, ToolResultBlock

from .budget import ContextBudgetPolicy, count_tokens, decide_budget, measure_budget
from .compactor import freeze_settled_units, summarize_frozen_segments
from .lifecycle import settled_prefix_units
from .models import (
    STAGE_SUMMARY_SCHEMA_VERSION,
    FrozenSegment,
    L4Context,
    StageSummary,
    canonical_json,
)

CACHE_BREAKPOINT_TEXT = '<crossshop-cache-breakpoint version="v1" />'


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
        "stage_summary": stage.to_prompt_dict() if stage else None,
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
                name="crossshop_context",
                content=(
                    f"<stage-summary>{canonical_json(summary.to_prompt_dict())}"
                    "</stage-summary>"
                ),
            )
        )
    for segment in frozen_segments:
        if segment.segment_id not in covered:
            assembled.append(
                SystemMsg(
                    name="crossshop_context",
                    content=f'<frozen-interaction id="{segment.segment_id}">{segment.content}</frozen-interaction>',
                )
            )

    prefix_hash = cache_prefix_sha256(assembled)
    assembled.append(
        SystemMsg(
            name="crossshop_context",
            content=CACHE_BREAKPOINT_TEXT.replace(
                " />", f' prefix_sha256="{prefix_hash}" />'
            ),
        ),
    )
    context = _l4(session_context)
    if context:
        assembled.append(
            SystemMsg(
                name="crossshop_context",
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
    summary_builder: Callable[..., StageSummary | Mapping[str, Any]] = (
        summarize_frozen_segments
    ),
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
        hot_turns=policy.l2_hot_turns,
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
        keep = _recent_keep_count(uncovered, policy)
        eligible = uncovered[:-keep] if keep else uncovered
        if eligible:
            before_tokens = report.total_input_tokens
            try:
                summary_sources, summary_existing = _summary_rebuild_inputs(
                    frozen=frozen,
                    eligible=eligible,
                    existing=stage,
                )
                candidate_stage = _build_and_validate_summary(
                    summary_builder,
                    eligible=summary_sources,
                    existing=summary_existing,
                    policy=policy,
                )
                candidate_view, candidate_report = _candidate_context(
                    candidate_stage,
                    fixed_messages=fixed_messages,
                    active=active,
                    frozen=frozen,
                    session_context=session_context,
                    tool_schemas=tool_schemas,
                    tool_results=tool_results,
                    policy=policy,
                )
                max_after = int(
                    policy.model_context_tokens * policy.post_compression_max_ratio
                )
                target_after = int(
                    policy.model_context_tokens * policy.post_compression_target_ratio
                )
                # This is L3 recent-segment retention, separate from L2's hot
                # buyer-turn window. Respect its bounded floor on rebuild.
                min_keep = min(policy.l3_min_recent_frozen_segments, keep)
                while (
                    candidate_report.total_input_tokens > max_after and keep > min_keep
                ):
                    keep -= 1
                    eligible = uncovered[:-keep] if keep else uncovered
                    summary_sources, summary_existing = _summary_rebuild_inputs(
                        frozen=frozen,
                        eligible=eligible,
                        existing=stage,
                    )
                    candidate_stage = _build_and_validate_summary(
                        summary_builder,
                        eligible=summary_sources,
                        existing=summary_existing,
                        policy=policy,
                    )
                    candidate_view, candidate_report = _candidate_context(
                        candidate_stage,
                        fixed_messages=fixed_messages,
                        active=active,
                        frozen=frozen,
                        session_context=session_context,
                        tool_schemas=tool_schemas,
                        tool_results=tool_results,
                        policy=policy,
                    )
                if candidate_report.total_input_tokens >= before_tokens:
                    state["context_compression"] = {
                        "action": "l3_no_gain",
                        "before_tokens": before_tokens,
                        "candidate_tokens": candidate_report.total_input_tokens,
                    }
                elif candidate_report.total_input_tokens > max_after:
                    state["context_compression"] = {
                        "action": "l3_target_missed",
                        "before_tokens": before_tokens,
                        "candidate_tokens": candidate_report.total_input_tokens,
                        "post_compression_max_tokens": max_after,
                    }
                else:
                    stage = candidate_stage
                    provisional = candidate_view
                    report = candidate_report
                    state["context_compression"] = {
                        "action": "l3_stage_summary",
                        "before_tokens": before_tokens,
                        "after_tokens": report.total_input_tokens,
                        "compression_ratio": (
                            (before_tokens - report.total_input_tokens) / before_tokens
                            if before_tokens
                            else 0.0
                        ),
                        "post_compression_target_tokens": target_after,
                        "post_compression_max_tokens": max_after,
                        "post_compression_target_met": (
                            report.total_input_tokens <= target_after
                        ),
                        "recent_raw_segment_count": keep,
                        "source_segment_ids": list(stage.source_segment_ids),
                        "source_manifest_hash": stage.combined_source_hash,
                        "summary_hash": stage.summary_hash,
                        "summary_estimated_tokens": stage.estimated_tokens,
                        "summary_target_tokens": policy.summary_target_tokens,
                        "summary_hard_limit_tokens": (policy.summary_hard_limit_tokens),
                    }
            except Exception as exc:  # fail-safe: retain the last valid view/state
                state["context_compression"] = {
                    "action": "l3_failed",
                    "before_tokens": before_tokens,
                    "failure_type": type(exc).__name__,
                    "previous_summary_hash": stage.summary_hash if stage else None,
                }
    budget_payload = _budget_report_with_cache_prefix(
        report.to_dict(),
        provisional,
    )
    if stage is not None:
        budget_payload.update(
            {
                "stage_summary_model_visible_estimated_tokens": (
                    stage.estimated_tokens
                ),
                "stage_summary_source_manifest_count": len(stage.source_segments),
                "stage_summary_source_manifest_hash": stage.combined_source_hash,
            },
        )
    state.update(
        {
            "frozen_segments": [item.to_dict() for item in frozen],
            "freeze_cursor": next_cursor,
            "stage_summary": stage.to_dict() if stage else None,
            "budget_report": budget_payload,
            "budget_decision": report.decision,
        },
    )
    return state, provisional


def _recent_keep_count(
    uncovered: Sequence[FrozenSegment],
    policy: ContextBudgetPolicy,
) -> int:
    keep = min(
        len(uncovered),
        policy.l3_keep_recent_frozen_segments,
        policy.l3_max_recent_frozen_segments,
    )
    while (
        keep
        and sum(
            count_tokens(item.content, name="recent_frozen").tokens
            for item in uncovered[-keep:]
        )
        > policy.l3_recent_frozen_token_limit
    ):
        keep -= 1
    return keep


def _summary_rebuild_inputs(
    *,
    frozen: Sequence[FrozenSegment],
    eligible: Sequence[FrozenSegment],
    existing: StageSummary | None,
) -> tuple[list[FrozenSegment], StageSummary | None]:
    """Return lossless inputs for normal recompression or legacy migration.

    v1/v2 summaries sampled independent request/fact/outcome lists and therefore
    cannot recover exact authored turn numbers or reliable request-to-tool
    relationships.  Their source manifest still points at immutable L2
    interactions, so a schema upgrade rebuilds v3 once from those verified
    sources instead of guessing from the sampled projection.
    """

    if existing is None or existing.schema_version == STAGE_SUMMARY_SCHEMA_VERSION:
        return list(eligible), existing
    by_id = {item.segment_id: item for item in frozen}
    covered: list[FrozenSegment] = []
    for source in existing.source_segments:
        segment = by_id.get(source["segment_id"])
        if segment is None or segment.content_hash != source["content_hash"]:
            raise ValueError(
                f"legacy stage summary source unavailable or changed: "
                f"{source['segment_id']}"
            )
        covered.append(segment)
    return [*covered, *eligible], None


def _build_and_validate_summary(
    builder: Callable[..., StageSummary | Mapping[str, Any]],
    *,
    eligible: Sequence[FrozenSegment],
    existing: StageSummary | None,
    policy: ContextBudgetPolicy,
) -> StageSummary:
    raw = builder(
        eligible,
        existing=existing,
        target_tokens=policy.summary_target_tokens,
        hard_limit_tokens=policy.summary_hard_limit_tokens,
    )
    candidate = raw if isinstance(raw, StageSummary) else StageSummary.from_dict(raw)
    expected = [
        *(existing.source_segments if existing else ()),
        *(
            {"segment_id": item.segment_id, "content_hash": item.content_hash}
            for item in eligible
            if existing is None or item.segment_id not in existing.source_segment_ids
        ),
    ]
    if list(candidate.source_segments) != expected:
        raise ValueError(
            "stage summary source manifest does not match eligible sources"
        )
    prompt_tokens = count_tokens(
        candidate.to_prompt_dict(), name="stage_summary"
    ).tokens
    if prompt_tokens != candidate.estimated_tokens:
        raise ValueError("stage summary estimated token count is inconsistent")
    if prompt_tokens > policy.summary_hard_limit_tokens:
        raise ValueError("stage summary exceeds hard token limit")
    return candidate


def _candidate_context(
    stage: StageSummary,
    *,
    fixed_messages: Sequence[Msg],
    active: Sequence[Msg],
    frozen: Sequence[FrozenSegment],
    session_context: Any,
    tool_schemas: Any,
    tool_results: Mapping[str, Any],
    policy: ContextBudgetPolicy,
) -> tuple[list[Msg], Any]:
    view = assemble_llm_input_messages(
        fixed_messages=fixed_messages,
        active_messages=active,
        frozen_segments=frozen,
        session_context=(
            session_context if isinstance(session_context, Mapping) else None
        ),
        stage_summary=stage,
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
    return view, report


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
