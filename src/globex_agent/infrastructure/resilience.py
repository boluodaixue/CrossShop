"""Pure-Python timeout and circuit-breaker wrapper for async tool calls."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUTS: dict[str, float] = {
    # CPU FP32 batch=16 measured 65.6s for 100 real documents; 75s leaves
    # only the measured embedding/recall overhead and remains finite.
    "product_search_tool": 75.0,
    "category_insight_tool": 15.0,
    "web_search_tool": 20.0,
    "prepare_order_tool": 10.0,
    "query_order_tool": 10.0,
    "prepare_cancel_order_tool": 10.0,
    "remember_preference_tool": 10.0,
    "task_dispatch": 180.0,
}
_FALLBACK_TIMEOUT = 30.0


@dataclass
class _CircuitState:
    consecutive_failures: int = 0
    opened_at: float | None = None
    half_open_probing: bool = False

    @property
    def status(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.half_open_probing else "open"


@dataclass
class CircuitBreakerRegistry:
    """Process-local circuit state keyed by tool name."""

    failure_threshold: int = 3
    reset_seconds: float = 60.0
    _states: dict[str, _CircuitState] = field(default_factory=dict)

    def _state(self, tool_name: str) -> _CircuitState:
        return self._states.setdefault(tool_name, _CircuitState())

    def status(self, tool_name: str) -> str:
        return self._state(tool_name).status

    def allow(self, tool_name: str, now: float | None = None) -> bool:
        state = self._state(tool_name)
        if state.opened_at is None:
            return True
        elapsed = (now or time.monotonic()) - state.opened_at
        if elapsed < self.reset_seconds:
            return False
        state.half_open_probing = True
        return True

    def record_success(self, tool_name: str) -> None:
        self._states[tool_name] = _CircuitState()

    def record_failure(self, tool_name: str, now: float | None = None) -> None:
        state = self._state(tool_name)
        if state.half_open_probing:
            state.opened_at = now or time.monotonic()
            state.half_open_probing = False
            return
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.failure_threshold:
            state.opened_at = now or time.monotonic()


class ToolResilienceMiddleware:
    """Run one async tool call with timeout, circuit breaker, and event output."""

    def __init__(
        self,
        registry: CircuitBreakerRegistry,
        bus: TradeEventBus | None = None,
        timeouts: dict[str, float] | None = None,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._timeouts = timeouts or DEFAULT_TIMEOUTS

    def _timeout_for(self, tool_name: str) -> float:
        return self._timeouts.get(tool_name, _FALLBACK_TIMEOUT)

    def _publish_circuit(self, tool_name: str, circuit: str, detail: str) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            ShoppingContext.current_session_id(),
            "tool.result",
            {"tool": tool_name, "circuit": circuit, "error": detail},
        )

    async def run(
        self,
        tool_name: str,
        operation: Callable[[], Awaitable[Any]],
    ) -> Any:
        if not self._registry.allow(tool_name):
            detail = f"{tool_name} 连续失败已熔断，暂不可用，请稍后再试或改用其他方式"
            logger.warning("工具熔断短路：%s", tool_name)
            self._publish_circuit(tool_name, "open", detail)
            return f"[error] {detail}"

        timeout = self._timeout_for(tool_name)
        try:
            result = await asyncio.wait_for(operation(), timeout=timeout)
        except asyncio.TimeoutError:
            self._registry.record_failure(tool_name)
            detail = f"{tool_name} 执行超过 {timeout:.0f} 秒已中断"
            logger.warning("工具超时：%s（%.0fs）", tool_name, timeout)
            self._publish_circuit(tool_name, self._registry.status(tool_name), detail)
            return f"[error] {detail}"
        except Exception as err:  # noqa: BLE001 - resilience boundary converts to text
            self._registry.record_failure(tool_name)
            detail = f"{tool_name} 执行异常：{err}"
            logger.warning("工具异常：%s（%s）", tool_name, err)
            self._publish_circuit(tool_name, self._registry.status(tool_name), detail)
            return f"[error] {detail}"

        if isinstance(result, str) and result.startswith("[error]"):
            self._registry.record_failure(tool_name)
        else:
            self._registry.record_success(tool_name)
        return result
