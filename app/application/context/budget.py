"""Layered context-budget measurement with explicit estimated/exact provenance."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .models import BudgetReport, LayerTokenCount


class ContextBudgetExceeded(RuntimeError):
    def __init__(self, decision: str, report: BudgetReport) -> None:
        self.decision = decision
        self.report = report
        super().__init__(
            f"context budget exceeded: {decision}; input={report.total_input_tokens}, available={report.available_input_tokens}"
        )


@dataclass(frozen=True)
class ContextBudgetPolicy:
    model_context_tokens: int
    reply_reserved_tokens: int = 0
    safety_margin_tokens: int = 0
    soft_limit_tokens: int = 0
    mode: str = "observe_only"
    # Low-level callers retain legacy eager freezing; production passes its
    # explicit CONTEXT_L2_HOT_TURNS (default 1) through the middleware.
    l2_hot_turns: int = 1
    l3_keep_recent_frozen_segments: int = 10
    l3_min_recent_frozen_segments: int = 8
    l3_max_recent_frozen_segments: int = 12
    l3_recent_frozen_token_limit: int = 16_000
    summary_target_tokens: int = 8_000
    summary_hard_limit_tokens: int = 12_000
    post_compression_target_ratio: float = 0.55
    post_compression_max_ratio: float = 0.60

    def __post_init__(self) -> None:
        if self.l2_hot_turns < 1:
            raise ValueError("L2 hot window must include at least the current turn")
        if self.mode not in {"observe_only", "enforce"}:
            raise ValueError("context budget mode must be observe_only or enforce")
        if any(
            value < 0
            for value in (
                self.model_context_tokens,
                self.reply_reserved_tokens,
                self.safety_margin_tokens,
                self.soft_limit_tokens,
                self.l3_keep_recent_frozen_segments,
                self.l3_min_recent_frozen_segments,
                self.l3_max_recent_frozen_segments,
                self.l3_recent_frozen_token_limit,
                self.summary_target_tokens,
                self.summary_hard_limit_tokens,
            )
        ):
            raise ValueError("context budget token values cannot be negative")
        if not self.summary_target_tokens or not self.summary_hard_limit_tokens:
            raise ValueError("summary token limits must be positive")
        if self.summary_target_tokens > self.summary_hard_limit_tokens:
            raise ValueError("summary target cannot exceed hard limit")
        if not 0 < self.post_compression_target_ratio <= 1:
            raise ValueError("post-compression target ratio must be within (0, 1]")
        if not 0 < self.post_compression_max_ratio <= 1:
            raise ValueError("post-compression max ratio must be within (0, 1]")
        if self.post_compression_target_ratio > self.post_compression_max_ratio:
            raise ValueError("post-compression target cannot exceed max ratio")


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return str(value)


def count_tokens(value: Any, *, name: str = "value") -> LayerTokenCount:
    text = _text(value)
    if not text:
        return LayerTokenCount(name=name, tokens=0, method="empty", estimated=False)
    return LayerTokenCount(
        name=name,
        tokens=max(1, math.ceil(len(text.encode("utf-8")) / 4)),
        method="estimate.utf8_bytes_div_4",
        estimated=True,
    )


def measure_budget(
    *,
    layers: Mapping[str, Any],
    tool_results: Mapping[str, Any] | None,
    model_context_tokens: int,
    reply_reserved_tokens: int = 0,
    safety_margin_tokens: int = 0,
) -> BudgetReport:
    measured = {name: count_tokens(value, name=name) for name, value in layers.items()}
    results = {
        name: count_tokens(value, name=f"tool_result:{name}")
        for name, value in (tool_results or {}).items()
    }
    total = sum(item.tokens for item in measured.values()) + sum(
        item.tokens for item in results.values()
    )
    available = max(
        0, model_context_tokens - reply_reserved_tokens - safety_margin_tokens
    )
    return BudgetReport(
        model_context_tokens=max(0, model_context_tokens),
        reply_reserved_tokens=max(0, reply_reserved_tokens),
        safety_margin_tokens=max(0, safety_margin_tokens),
        layers=measured,
        tool_results=results,
        total_input_tokens=total,
        available_input_tokens=available,
        overflow_tokens=max(0, total - available),
        estimated=any(
            item.estimated for item in [*measured.values(), *results.values()]
        ),
        counter_method="estimate.utf8_bytes_div_4",
    )


def decide_budget(
    report: BudgetReport, *, policy: ContextBudgetPolicy, has_frozen_history: bool
) -> BudgetReport:
    fixed = sum(
        report.layers.get(name, LayerTokenCount(name, 0, "empty")).tokens
        for name in ("fixed_prompt", "tool_schemas", "l4_candidate")
    )
    active = sum(
        report.layers.get(name, LayerTokenCount(name, 0, "empty")).tokens
        for name in ("active", "current")
    ) + sum(item.tokens for item in report.tool_results.values())
    if fixed > report.available_input_tokens:
        decision = "fixed_overflow"
    elif fixed + active > report.available_input_tokens:
        decision = "active_overflow"
    elif report.total_input_tokens > report.available_input_tokens:
        decision = "needs_l3" if has_frozen_history else "active_overflow"
    elif (
        policy.soft_limit_tokens
        and report.total_input_tokens > policy.soft_limit_tokens
        and has_frozen_history
    ):
        decision = "needs_l3"
    else:
        decision = "ok"
    return replace(
        report,
        soft_limit_tokens=policy.soft_limit_tokens,
        decision=decision,
        enforcement_mode=policy.mode,
    )
