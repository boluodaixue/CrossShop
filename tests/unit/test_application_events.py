"""Event-level integration tests for fallback, compression, and circuit events."""

from __future__ import annotations

import asyncio

import pytest

from globex_agent.application.agents.identity import ThreadIdentity
from globex_agent.application.agents.orchestrator import (
    MainAgentOrchestrator,
    SubmitIntentInput,
)
from globex_agent.application.agents.session_lock import SessionLockRegistry
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


class FakeAgent:
    async def reply(self, query, *, thread_id=None, event_sink=None) -> str:
        del query, thread_id, event_sink
        return "done"

    async def compress_context(self, thread_id, max_messages) -> dict | None:
        del thread_id, max_messages
        return {"removed": 2, "kept": 4}

    def checkpoint_state(self, thread_id) -> str:
        del thread_id
        return "{}"


class FakeSessions:
    def __init__(self) -> None:
        self._identity = ThreadIdentity(environment="test", hmac_key="event-test")
        self._locks = SessionLockRegistry()

    def thread_id(self, shopping_session_id: str) -> str:
        return self._identity.main_thread_id(shopping_session_id)

    def session_lock(self, shopping_session_id: str):
        return self._locks.acquire(self.thread_id(shopping_session_id))

    async def get_or_create(self, shopping_session_id):
        del shopping_session_id
        return FakeAgent()

    async def persist(self, shopping_session_id) -> None:
        del shopping_session_id


class FakePreferences:
    async def list_by_buyer(self, buyer_id):
        del buyer_id
        return []


class FakeConversations:
    async def list_turns(self, session_id, limit=1):
        del session_id, limit
        return [object()]

    async def touch_session(self, session_id, buyer_id, locale, currency) -> None:
        del session_id, buyer_id, locale, currency

    async def append_turn(self, turn) -> None:
        del turn

    async def append_events(self, events) -> None:
        del events


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


class TestContextCompressedEvent:
    async def test_orchestrator_publishes_compression_event(self) -> None:
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        orchestrator = MainAgentOrchestrator(
            sessions=FakeSessions(),
            bus=bus,
            preference_store=FakePreferences(),
            conversation_store=FakeConversations(),
            context_size=128000,
        )
        await orchestrator.handle_intent(
            SubmitIntentInput(
                shopping_session_id="s1",
                buyer_id="b1",
                locale="zh-CN",
                currency="CNY",
                raw_query="第二句话",
            )
        )

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.type == "context.compressed"
        assert event.payload == {"removed": 2, "kept": 4}


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
