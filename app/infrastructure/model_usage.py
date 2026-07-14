"""Request-scoped model-usage capture for reproducible evaluations.

Production model calls always pass through this module, but records are only
retained while an evaluation explicitly opens :func:`capture_model_usage`.
No prompt, response, credential, buyer identifier, or session identifier is
stored here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any


def _field(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    try:
        return getattr(source, name, default)
    except Exception:  # noqa: BLE001 - provider objects may raise KeyError
        return default


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ModelUsageSample:
    """Privacy-safe usage for one completed upstream model invocation."""

    requested_model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    cache_creation_input_tokens: int
    elapsed_seconds: float

    @property
    def cache_hit(self) -> bool:
        return self.cached_input_tokens > 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cache_hit": self.cache_hit}


@dataclass
class ModelUsageCapture:
    """All successful model usages observed inside one evaluation scope."""

    samples: list[ModelUsageSample] = field(default_factory=list)

    def totals(self) -> dict[str, int | float | None]:
        input_tokens = sum(item.input_tokens for item in self.samples)
        output_tokens = sum(item.output_tokens for item in self.samples)
        cached_tokens = sum(item.cached_input_tokens for item in self.samples)
        total_tokens = sum(item.total_tokens for item in self.samples)
        return {
            "model_calls": len(self.samples),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_hit_calls": sum(item.cache_hit for item in self.samples),
            "token_weighted_cache_hit_ratio": (
                cached_tokens / input_tokens if input_tokens else None
            ),
        }


_CURRENT_CAPTURE: ContextVar[ModelUsageCapture | None] = ContextVar(
    "crossshop_model_usage_capture",
    default=None,
)


@contextmanager
def capture_model_usage() -> Iterator[ModelUsageCapture]:
    """Capture model usage for the current task and child asyncio tasks."""

    capture = ModelUsageCapture()
    token = _CURRENT_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CURRENT_CAPTURE.reset(token)


def record_model_usage(response: Any, *, requested_model: str) -> None:
    """Record normalized AgentScope ``ChatUsage`` when capture is active."""

    capture = _CURRENT_CAPTURE.get()
    if capture is None or response is None:
        return
    usage = _field(response, "usage")
    if usage is None:
        return
    input_tokens = _non_negative_int(
        _field(usage, "input_tokens", _field(usage, "prompt_tokens", 0)),
    )
    output_tokens = _non_negative_int(
        _field(usage, "output_tokens", _field(usage, "completion_tokens", 0)),
    )
    cached_input_tokens = min(
        input_tokens,
        _non_negative_int(
            _field(
                usage,
                "cache_input_tokens",
                _field(
                    _field(usage, "prompt_tokens_details", {}),
                    "cached_tokens",
                    0,
                ),
            ),
        ),
    )
    total_tokens = _non_negative_int(
        _field(usage, "total_tokens"),
        input_tokens + output_tokens,
    )
    capture.samples.append(
        ModelUsageSample(
            requested_model=requested_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=_non_negative_int(
                _field(usage, "cache_creation_input_tokens", 0),
            ),
            elapsed_seconds=max(0.0, float(_field(usage, "time", 0.0) or 0.0)),
        ),
    )
