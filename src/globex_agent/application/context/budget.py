"""Layered token measurement for main-model context governance.

This module only measures.  It does not trim messages, choose a model, or
change any prompt text.  A deterministic UTF-8 byte estimate is used only
when neither a model nor tokenizer exposes a count method, and the report
marks that estimate explicitly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .models import BudgetReport, LayerTokenCount


class ContextBudgetExceeded(RuntimeError):
    """Raised before model invocation when an enforced immutable layer cannot fit."""

    def __init__(self, decision: str, report: BudgetReport) -> None:
        self.decision = decision
        self.report = report
        super().__init__(
            f"context budget exceeded: {decision}; "
            f"input={report.total_input_tokens}, available={report.available_input_tokens}"
        )


@dataclass(frozen=True)
class ContextBudgetPolicy:
    """Measured L1 limits. Zero soft limit means observation without L3 scheduling."""

    model_context_tokens: int
    reply_reserved_tokens: int = 0
    safety_margin_tokens: int = 0
    soft_limit_tokens: int = 0
    mode: str = "observe_only"
    l3_keep_recent_frozen_segments: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"observe_only", "enforce"}:
            raise ValueError("context budget mode must be observe_only or enforce")
        for value in (
            self.model_context_tokens,
            self.reply_reserved_tokens,
            self.safety_margin_tokens,
            self.soft_limit_tokens,
            self.l3_keep_recent_frozen_segments,
        ):
            if value < 0:
                raise ValueError("context budget token values cannot be negative")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "to_dict"):
        value = value.to_dict()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        # Some provider message objects expose only a useful string form.
        return str(value)


class TokenCounter:
    """Use exposed model/tokenizer counters before the explicit fallback."""

    def __init__(self, *, model: Any | None = None, tokenizer: Any | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def count(self, value: Any, *, name: str = "value") -> LayerTokenCount:
        message_count = self._model_message_count(value)
        if message_count is not None:
            return LayerTokenCount(
                name=name,
                tokens=message_count,
                method="model.get_num_tokens_from_messages",
                estimated=self._model_count_is_estimated(),
            )
        text = _as_text(value)
        if not text:
            return LayerTokenCount(name=name, tokens=0, method="empty", estimated=False)

        model_count = self._model_count(text)
        if model_count is not None:
            return LayerTokenCount(
                name=name,
                tokens=model_count,
                method="model.get_num_tokens",
                estimated=self._model_count_is_estimated(),
            )
        tokenizer_count = self._tokenizer_count(text)
        if tokenizer_count is not None:
            return LayerTokenCount(
                name=name,
                tokens=tokenizer_count,
                method="tokenizer.encode",
                estimated=False,
            )

        # This is a byte-based estimate, not a message-count approximation.
        estimated = max(1, math.ceil(len(text.encode("utf-8")) / 4))
        return LayerTokenCount(
            name=name,
            tokens=estimated,
            method="estimate.utf8_bytes_div_4",
            estimated=True,
        )

    def _model_message_count(self, value: Any) -> int | None:
        """Prefer provider-native message accounting for history/current layers."""

        if self.model is None or not isinstance(value, (list, tuple)):
            return None
        method = getattr(self.model, "get_num_tokens_from_messages", None)
        if not callable(method):
            return None
        if self._is_generic_langchain_counter(method):
            return None
        try:
            result = method(value)
            if isinstance(result, int) and result >= 0:
                return result
        except (TypeError, ValueError, AttributeError, NotImplementedError, OSError):
            return None
        return None

    def _model_count(self, text: str) -> int | None:
        if self.model is None:
            return None
        for method_name in ("get_num_tokens", "count_tokens"):
            method = getattr(self.model, method_name, None)
            if not callable(method):
                continue
            if self._is_generic_langchain_counter(method):
                continue
            try:
                value = method(text)
                if isinstance(value, int) and value >= 0:
                    return value
            except (TypeError, ValueError, AttributeError, NotImplementedError, OSError):
                continue
        return None

    def _model_count_is_estimated(self) -> bool:
        explicit = bool(
            getattr(self.model, "token_count_is_estimated", False)
            or getattr(self.model, "_globex_token_count_estimated", False)
        )
        method = getattr(self.model, "get_num_tokens", None)
        inherited_generic = str(getattr(method, "__module__", "")).startswith(
            "langchain_core."
        )
        return explicit or inherited_generic

    @staticmethod
    def _is_generic_langchain_counter(method: Any) -> bool:
        return str(getattr(method, "__module__", "")).startswith("langchain_core.")

    def _tokenizer_count(self, text: str) -> int | None:
        if self.tokenizer is None:
            return None
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            try:
                try:
                    value = encode(text, add_special_tokens=False)
                except TypeError:
                    value = encode(text)
                if isinstance(value, int):
                    return max(0, value)
                return len(value)
            except (TypeError, ValueError, AttributeError):
                pass
        for method_name in ("get_num_tokens", "count_tokens"):
            method = getattr(self.tokenizer, method_name, None)
            if callable(method):
                try:
                    value = method(text)
                    if isinstance(value, int) and value >= 0:
                        return value
                except (TypeError, ValueError, AttributeError):
                    continue
        return None


def count_tokens(
    value: Any,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> LayerTokenCount:
    return TokenCounter(model=model, tokenizer=tokenizer).count(value)


def measure_budget(
    *,
    fixed_prompt: Any = "",
    tool_schemas: Any = "",
    l4_candidate: Any = "",
    history: Any = "",
    active_current: Any = "",
    active: Any | None = None,
    current: Any | None = None,
    tool_results: Mapping[str, Any] | list[Any] | tuple[Any, ...] | None = None,
    model_context_tokens: int,
    reply_reserved_tokens: int = 0,
    safety_margin_tokens: int = 0,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> BudgetReport:
    """Measure every planned layer and each named tool result independently."""

    counter = TokenCounter(model=model, tokenizer=tokenizer)
    layer_values: list[tuple[str, Any]] = [
        ("fixed_prompt", fixed_prompt),
        ("tool_schemas", tool_schemas),
        ("l4_candidate", l4_candidate),
        ("history", history),
    ]
    if active is not None or current is not None:
        layer_values.extend((("active", active or ""), ("current", current or "")))
    else:
        layer_values.append(("active_current", active_current))
    layers = {
        name: counter.count(value, name=name)
        for name, value in layer_values
    }
    if isinstance(tool_results, Mapping):
        result_items = tool_results.items()
    else:
        result_items = enumerate(tool_results or ())
    results = {
        str(name): counter.count(value, name=f"tool_result:{name}")
        for name, value in result_items
    }
    total = sum(item.tokens for item in layers.values()) + sum(
        item.tokens for item in results.values()
    )
    available = max(0, model_context_tokens - reply_reserved_tokens - safety_margin_tokens)
    overflow = max(0, total - available)
    all_counts = [*layers.values(), *results.values()]
    methods = {item.method for item in all_counts if item.tokens}
    method = next(iter(methods), "empty") if len(methods) == 1 else "mixed"
    return BudgetReport(
        model_context_tokens=max(0, model_context_tokens),
        reply_reserved_tokens=max(0, reply_reserved_tokens),
        safety_margin_tokens=max(0, safety_margin_tokens),
        layers=layers,
        tool_results=results,
        total_input_tokens=total,
        available_input_tokens=available,
        overflow_tokens=overflow,
        estimated=any(item.estimated for item in all_counts),
        counter_method=method,
    )


def decide_budget(
    report: BudgetReport,
    *,
    policy: ContextBudgetPolicy,
    has_frozen_history: bool,
) -> BudgetReport:
    """Classify overflow without modifying any model-visible content."""

    layers = report.layers
    fixed_tokens = sum(
        layers.get(name, LayerTokenCount(name, 0, "empty")).tokens
        for name in ("fixed_prompt", "tool_schemas", "l4_candidate")
    )
    active_tokens = sum(
        layers.get(name, LayerTokenCount(name, 0, "empty")).tokens
        for name in ("active", "current", "active_current")
    ) + sum(item.tokens for item in report.tool_results.values())
    available = report.available_input_tokens
    if fixed_tokens > available:
        decision = "fixed_overflow"
    elif fixed_tokens + active_tokens > available:
        decision = "active_overflow"
    elif report.total_input_tokens > available:
        decision = "needs_l3" if has_frozen_history else "active_overflow"
    elif (
        policy.soft_limit_tokens > 0
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
