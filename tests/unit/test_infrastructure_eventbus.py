"""Event bus routing and circuit-breaker/resilience behavior tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.resilience import (
    CircuitBreakerRegistry,
    ToolResilienceMiddleware,
)
from globex_agent.infrastructure.throttle import GatewayThrottle
from globex_agent.infrastructure.transient import is_transient_error


class TestEventBus:
    async def test_publish_routes_to_subscriber(self) -> None:
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        other = bus.subscribe("s2")
        bus.publish("s1", "final.result", {"text": "done"})

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "final.result"
        assert other.empty()

    def test_reject_unknown_event_type(self) -> None:
        with pytest.raises(ValueError, match="未知事件类型"):
            TradeEventBus().publish("s1", "not.a.type", {})


class TestTransientAndThrottle:
    def test_transient_markers(self) -> None:
        assert is_transient_error(RuntimeError("Too many concurrent requests"))
        assert is_transient_error(RuntimeError("Error code: 429"))
        assert not is_transient_error(RuntimeError("库存不足"))

    async def test_concurrency_cap(self) -> None:
        throttle = GatewayThrottle(max_concurrency=2, min_interval_seconds=0)
        in_flight = 0
        peak = 0

        async def worker() -> None:
            nonlocal in_flight, peak
            async with throttle.slot():
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.02)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(4)))
        assert peak == 2

    async def test_min_interval_spaces_out_starts(self) -> None:
        throttle = GatewayThrottle(max_concurrency=8, min_interval_seconds=0.03)
        starts: list[float] = []

        async def worker() -> None:
            async with throttle.slot():
                starts.append(time.monotonic())

        await asyncio.gather(*(worker() for _ in range(3)))
        starts.sort()
        gaps = [starts[index] - starts[index - 1] for index in range(1, len(starts))]
        assert gaps and all(gap >= 0.02 for gap in gaps)


class TestResilience:
    async def test_circuit_opens_after_failures(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=3, reset_seconds=60)
        middleware = ToolResilienceMiddleware(registry)

        async def fail() -> str:
            raise RuntimeError("boom")

        for _ in range(3):
            result = await middleware.run("product_search_tool", fail)
            assert result.startswith("[error]")
        assert registry.status("product_search_tool") == "open"
        assert await middleware.run("product_search_tool", fail) == (
            "[error] product_search_tool 连续失败已熔断，暂不可用，请稍后再试或改用其他方式"
        )

    async def test_timeout_returns_error_and_records_failure(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=1, reset_seconds=60)
        middleware = ToolResilienceMiddleware(
            registry,
            timeouts={"slow_tool": 0.01},
        )

        async def slow() -> str:
            await asyncio.sleep(0.2)
            return "never"

        result = await middleware.run("slow_tool", slow)
        assert result.startswith("[error]")
        assert registry.status("slow_tool") == "open"
