"""Event-level integration tests for fallback, compression, and circuit events."""

from __future__ import annotations

import asyncio

import pytest

from globex_agent.infrastructure.context import (
    ShoppingContext,
    ShoppingContextSnapshot,
)
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.llm import _FallbackRunnable
from globex_agent.infrastructure.resilience import (
    CircuitBreakerRegistry,
    ToolResilienceMiddleware,
)


class FailingPrimary:
    async def ainvoke(self, input, config=None, **kwargs):
        del input, config, kwargs
        raise RuntimeError("Too many concurrent requests")


class NonTransientPrimary:
    async def ainvoke(self, input, config=None, **kwargs):
        del input, config, kwargs
        raise RuntimeError("库存不足")


class WorkingFallback:
    async def ainvoke(self, input, config=None, **kwargs):
        del input, config, kwargs
        return "fallback-ok"


class TestModelFallbackEvent:
    async def test_transient_primary_error_emits_fallback_event(self) -> None:
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s1",
                buyer_id="b1",
                locale="zh-CN",
                currency="CNY",
            )
        )
        try:
            runnable = _FallbackRunnable(
                FailingPrimary(),
                WorkingFallback(),
                bus,
                "primary-model",
                "fallback-model",
            )
            result = await runnable.ainvoke({})
        finally:
            ShoppingContext.reset(token)

        assert result == "fallback-ok"
        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "model.fallback"
        assert event.payload["from"] == "primary-model"
        assert event.payload["to"] == "fallback-model"
        assert "Too many concurrent requests" in event.payload["reason"]

    async def test_non_transient_error_propagates_without_fallback_event(self) -> None:
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        runnable = _FallbackRunnable(
            NonTransientPrimary(),
            WorkingFallback(),
            bus,
            "primary-model",
            "fallback-model",
        )

        with pytest.raises(RuntimeError, match="库存不足"):
            await runnable.ainvoke({})
        assert queue.empty()


class TestCircuitEvent:
    async def test_open_circuit_publishes_tool_result_error(self) -> None:
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s1",
                buyer_id="b1",
                locale="zh-CN",
                currency="CNY",
            )
        )
        try:
            registry = CircuitBreakerRegistry(
                failure_threshold=1,
                reset_seconds=60,
            )
            middleware = ToolResilienceMiddleware(
                registry,
                bus,
                timeouts={"broken_tool": 1},
            )

            async def fail() -> str:
                raise RuntimeError("boom")

            result = await middleware.run("broken_tool", fail)
        finally:
            ShoppingContext.reset(token)

        assert result.startswith("[error]")
        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "tool.result"
        assert event.payload["circuit"] == "open"
        assert event.payload["tool"] == "broken_tool"
        assert "boom" in event.payload["error"]
