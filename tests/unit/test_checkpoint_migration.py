"""Unit coverage for Redis checkpoint wiring and stable runtime identities."""

from __future__ import annotations

import asyncio
import json

import pytest

import globex_agent.composition as composition
from globex_agent.application.agents.identity import ThreadIdentity, graph_config
from globex_agent.application.agents.session_lock import SessionLockRegistry
from globex_agent.application.tools.task_dispatch_tool import build_task_dispatch_tool
from globex_agent.infrastructure.checkpoint import (
    CheckpointConfigurationError,
    InMemoryDispatchResultStore,
    create_redis_checkpoint_resource,
)
from globex_agent.infrastructure.context import (
    ShoppingContext,
    ShoppingContextSnapshot,
)
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.settings import load_settings


def test_thread_identity_is_stable_anonymous_and_isolated() -> None:
    identity = ThreadIdentity(environment="test", hmac_key="unit-key")
    main = identity.main_thread_id("session-a")
    assert main == identity.main_thread_id("session-a")
    assert "session-a" not in main
    assert "unit-key" not in main

    dispatch_id = identity.dispatch_id(main, "tool-call-1", 0, "search_agent", "lamp")
    child = identity.child_thread_id(main, "search_agent", dispatch_id)
    assert child == identity.child_thread_id(main, "search_agent", dispatch_id)
    assert child != identity.child_thread_id(main, "trade_agent", dispatch_id)
    assert child != identity.child_thread_id(
        identity.main_thread_id("session-b"),
        "search_agent",
        dispatch_id,
    )


def test_graph_config_uses_root_checkpoint_namespace() -> None:
    config = graph_config("globex:v1:test:main:abc")
    assert config["configurable"] == {
        "thread_id": "globex:v1:test:main:abc",
        "checkpoint_ns": "",
    }


def test_task_dispatch_requires_explicit_completion_store() -> None:
    with pytest.raises(RuntimeError, match="explicit completion marker store"):
        build_task_dispatch_tool(object(), object(), TradeEventBus())


async def test_session_lock_serializes_same_key_and_cleans_up() -> None:
    registry = SessionLockRegistry()
    active = 0
    peak = 0

    async def enter(key: str) -> None:
        nonlocal active, peak
        async with registry.acquire(key):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(enter("same"), enter("same"))
    assert peak == 1
    assert registry.size == 0

    peak = 0
    await asyncio.gather(enter("a"), enter("b"))
    assert peak == 2
    assert registry.size == 0


class _FakeSaver:
    instances: list[_FakeSaver] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.setup_calls = 0
        self.close_calls = 0
        self.__class__.instances.append(self)

    async def asetup(self) -> None:
        self.setup_calls += 1

    async def __aexit__(self, *_args) -> None:
        self.close_calls += 1


async def test_checkpoint_resource_uses_shared_saver_and_ttl(monkeypatch) -> None:
    monkeypatch.setattr(
        "globex_agent.infrastructure.checkpoint.AsyncRedisSaver",
        _FakeSaver,
    )
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    monkeypatch.delenv("CHECKPOINT_REDIS_URL", raising=False)
    monkeypatch.setenv("GLOBEX_ENV", "local")
    monkeypatch.delenv("LANGGRAPH_AES_KEY", raising=False)
    monkeypatch.delenv("CHECKPOINT_REQUIRE_ENCRYPTION", raising=False)
    settings = load_settings()

    resource = await create_redis_checkpoint_resource(settings)
    assert resource.saver.kwargs["redis_url"] == "redis://unit-test"
    assert resource.saver.kwargs["ttl"] == {
        "default_ttl": 10080,
        "refresh_on_read": True,
    }
    assert resource.saver.setup_calls == 1
    await resource.close()
    assert resource.saver.close_calls == 1


async def test_checkpoint_resource_fails_without_redis(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("CHECKPOINT_REDIS_URL", raising=False)
    monkeypatch.setenv("GLOBEX_ENV", "local")
    monkeypatch.delenv("LANGGRAPH_AES_KEY", raising=False)
    settings = load_settings()
    with pytest.raises(CheckpointConfigurationError, match="InMemorySaver fallback"):
        await create_redis_checkpoint_resource(settings)


async def test_non_local_requires_encryption_key(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    monkeypatch.setenv("GLOBEX_ENV", "production")
    monkeypatch.delenv("LANGGRAPH_AES_KEY", raising=False)
    monkeypatch.delenv("CHECKPOINT_REQUIRE_ENCRYPTION", raising=False)
    settings = load_settings()
    with pytest.raises(CheckpointConfigurationError, match="LANGGRAPH_AES_KEY"):
        await create_redis_checkpoint_resource(settings)


async def test_non_local_cannot_disable_encryption_requirement(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    monkeypatch.setenv("GLOBEX_ENV", "production")
    monkeypatch.setenv("CHECKPOINT_REQUIRE_ENCRYPTION", "0")
    monkeypatch.delenv("LANGGRAPH_AES_KEY", raising=False)
    settings = load_settings()
    assert settings.checkpoint_require_encryption is True
    with pytest.raises(CheckpointConfigurationError, match="LANGGRAPH_AES_KEY"):
        await create_redis_checkpoint_resource(settings)


async def test_configured_key_is_rejected_when_official_saver_cannot_inject_serde(
    monkeypatch,
) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    monkeypatch.setenv("GLOBEX_ENV", "local")
    monkeypatch.setenv("LANGGRAPH_AES_KEY", "1234567890123456")
    settings = load_settings()
    with pytest.raises(CheckpointConfigurationError, match="does not expose serializer"):
        await create_redis_checkpoint_resource(settings)


class _CountingAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def reply(self, _query: str, *, thread_id: str, event_sink=None) -> str:
        del event_sink
        self.calls.append(thread_id)
        return "done"


class _CountingFactory:
    def __init__(self, agents: list[_CountingAgent]) -> None:
        self.agents = agents

    def build(self):
        agent = _CountingAgent()
        self.agents.append(agent)
        return agent


async def test_dispatch_replay_reuses_child_thread_and_done_marker() -> None:
    identity = ThreadIdentity(environment="test", hmac_key="unit-key")
    first_agents: list[_CountingAgent] = []
    second_agents: list[_CountingAgent] = []
    result_store = InMemoryDispatchResultStore()
    first_tool = build_task_dispatch_tool(
        _CountingFactory(first_agents),
        _CountingFactory(first_agents),
        TradeEventBus(),
        identity=identity,
        result_store=result_store,
    )
    second_tool = build_task_dispatch_tool(
        _CountingFactory(second_agents),
        _CountingFactory(second_agents),
        TradeEventBus(),
        identity=identity,
        result_store=result_store,
    )
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="session-a",
            buyer_id="buyer-a",
            locale="zh-CN",
            currency="CNY",
            main_thread_id=identity.main_thread_id("session-a"),
        )
    )
    dispatch = {
        "dispatches": [
            {"subagent_type": "search_agent", "demands": "camping lamp"}
        ]
    }
    try:
        first_text = await first_tool.arun(dispatch, tool_call_id="unit-dispatch-call")
        second_text = await second_tool.arun(dispatch, tool_call_id="unit-dispatch-call")
        first = json.loads(getattr(first_text, "content", first_text))
        second = json.loads(getattr(second_text, "content", second_text))
    finally:
        ShoppingContext.reset(token)

    first_item = first["dispatches"][0]
    second_item = second["dispatches"][0]
    assert first_item["dispatch_id"] == second_item["dispatch_id"]
    assert first_item["thread_id"] == second_item["thread_id"]
    assert second_item["cached"] is True
    assert len(first_agents) == 1
    assert first_agents[0].calls == [first_item["thread_id"]]
    assert second_agents == []


async def test_build_container_closes_checkpoint_resources_on_late_failure(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    monkeypatch.setenv("GLOBEX_ENV", "local")
    settings = load_settings()

    class _FakeCheckpointResource:
        saver = object()

        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    class _FakeDispatchResultStore:
        def __init__(self, *_args, **_kwargs) -> None:
            self.startup_calls = 0
            self.close_calls = 0

        async def startup(self) -> None:
            self.startup_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    checkpoint = _FakeCheckpointResource()
    dispatch_store: _FakeDispatchResultStore | None = None

    async def _create(_settings):
        return checkpoint

    def _make_dispatch(*args, **kwargs):
        nonlocal dispatch_store
        dispatch_store = _FakeDispatchResultStore(*args, **kwargs)
        return dispatch_store

    def _fail_vector_index():
        raise RuntimeError("vector setup failed")

    monkeypatch.setattr(composition, "load_settings", lambda: settings)
    monkeypatch.setattr(composition, "create_redis_checkpoint_resource", _create)
    monkeypatch.setattr(composition, "RedisDispatchResultStore", _make_dispatch)
    monkeypatch.setattr(composition, "_build_vector_index", _fail_vector_index)

    with pytest.raises(RuntimeError, match="vector setup failed"):
        await composition.build_container()

    assert dispatch_store is not None
    assert dispatch_store.startup_calls == 1
    assert dispatch_store.close_calls == 1
    assert checkpoint.close_calls == 1
