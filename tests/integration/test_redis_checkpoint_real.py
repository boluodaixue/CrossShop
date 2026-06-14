"""Opt-in integration coverage against a real Redis 8+/Stack server.

Run with ``GLOBEX_REQUIRE_REAL_REDIS=1`` to turn an unavailable endpoint into a
failure instead of a skip. The tests never use fakeredis.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TypedDict

import pytest
import redis.asyncio as aioredis
from langchain_core.messages import AIMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.agents.identity import ThreadIdentity
from globex_agent.application.agents.main_agent import MainAgentState
from globex_agent.application.agents.orchestrator import SubmitIntentInput
from globex_agent.application.context.assembler import context_pre_model_hook
from globex_agent.application.context.compactor import summarize_frozen_segments
from globex_agent.application.context.lifecycle import group_interaction_units
from globex_agent.application.context.models import FrozenSegment
from globex_agent.application.tools.task_dispatch_tool import build_task_dispatch_tool
from globex_agent.infrastructure.checkpoint import (
    CheckpointConfigurationError,
    RedisDispatchResultStore,
)
from globex_agent.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.settings import load_settings
from scripts.eval_context_management_phase_f import (
    _digest_messages,
    run_evaluation,
)

# RediSearch indexes are only supported in Redis DB 0; isolate cases with
# unique thread IDs instead of selecting a numbered database.
REAL_REDIS_URL = os.getenv("GLOBEX_REAL_REDIS_URL", "redis://127.0.0.1:6379/0")
REQUIRE_REAL_REDIS = os.getenv("GLOBEX_REQUIRE_REAL_REDIS") == "1"


class CounterState(TypedDict, total=False):
    count: int
    events: list[str]


def _config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }


def _counter_graph(saver: AsyncRedisSaver):
    def step(state: CounterState) -> CounterState:
        count = int(state.get("count", 0)) + 1
        return {"count": count, "events": [f"step-{count}"]}

    builder = StateGraph(CounterState)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    return builder.compile(checkpointer=saver)


def _child_graph(saver: AsyncRedisSaver):
    def step(state: MessagesState):
        del state
        return {"messages": [AIMessage(content="child-done")]}

    builder = StateGraph(MessagesState)
    builder.add_node("step", step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    return builder.compile(checkpointer=saver)


def _context_graph(saver: AsyncRedisSaver):
    def finish(state: MainAgentState):
        del state
        return {"messages": [AIMessage(content="done")]}

    builder = StateGraph(MainAgentState)
    builder.add_node("pre_model_hook", context_pre_model_hook)
    builder.add_node("agent", finish)
    builder.add_edge(START, "pre_model_hook")
    builder.add_edge("pre_model_hook", "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=saver)


def _phase_f_snapshot_graph(saver: AsyncRedisSaver):
    def persist(state: MainAgentState):
        del state
        return {}

    builder = StateGraph(MainAgentState)
    builder.add_node("persist", persist)
    builder.add_edge(START, "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=saver)


class _RecordingGraph:
    def __init__(self, graph) -> None:
        self._graph = graph
        self.inputs: list[dict | None] = []

    async def aget_state(self, config):
        return await self._graph.aget_state(config)

    async def astream(self, graph_input, **kwargs):
        self.inputs.append(graph_input)
        async for chunk in self._graph.astream(graph_input, **kwargs):
            yield chunk


def _approval_graph(saver: AsyncRedisSaver):
    async def ask_for_approval(state: MessagesState):
        del state
        return {"messages": [AIMessage(content="approved")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", ask_for_approval)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    # Static interrupts exercise a durable node-boundary checkpoint without
    # relying on dynamic interrupt() context propagation on Python 3.10.
    return builder.compile(checkpointer=saver, interrupt_before=["ask"])


async def _open_saver(*, ttl_minutes: int | None = None) -> AsyncRedisSaver:
    ttl = None
    if ttl_minutes is not None:
        ttl = {"default_ttl": ttl_minutes, "refresh_on_read": True}
    saver = AsyncRedisSaver(redis_url=REAL_REDIS_URL, ttl=ttl)
    await saver.asetup()
    return saver


async def _checkpoint_keys(client, thread_id: str) -> list[str]:
    keys: list[str] = []
    async for key in client.scan_iter(match="checkpoint*"):
        text = key.decode() if isinstance(key, bytes) else key
        if thread_id in text:
            keys.append(key)
    return keys


@pytest.fixture
async def real_redis():
    client = aioredis.from_url(REAL_REDIS_URL, decode_responses=True)
    try:
        await client.ping()
        command_info = await client.execute_command(
            "COMMAND",
            "INFO",
            "JSON.SET",
            "JSON.GET",
            "FT.SEARCH",
        )
        if not command_info or any(item is None for item in command_info):
            raise RuntimeError("RedisJSON/RediSearch commands are unavailable")
        probe_key = f"globex-real-test:probe:{os.getpid()}"
        await client.execute_command("JSON.SET", probe_key, "$", '{"ok":1}')
        assert await client.execute_command("JSON.GET", probe_key, "$.ok")
        await client.delete(probe_key)
        await client.execute_command("FT._LIST")
    except Exception as err:  # noqa: BLE001 - integration prerequisite
        await client.aclose()
        message = f"real Redis prerequisite unavailable at {REAL_REDIS_URL}: {err}"
        if REQUIRE_REAL_REDIS:
            pytest.fail(message)
        pytest.skip(message)
    try:
        yield client
    finally:
        await client.aclose()


async def test_real_redis_checkpoint_history_and_rebuild(real_redis) -> None:
    del real_redis
    thread_id = f"globex-real-test:counter:{os.getpid()}:{time.time_ns()}"
    config = _config(thread_id)
    saver = await _open_saver(ttl_minutes=1)
    try:
        graph = _counter_graph(saver)
        first = await graph.ainvoke({"count": 0, "events": []}, config=config)
        second = await graph.ainvoke({}, config=config)
        history = [item async for item in saver.alist(config)]
        assert first["count"] == 1
        assert second["count"] == 2
        assert len(history) > 1
        assert await saver.aget_tuple(config) is not None
        keys = await _checkpoint_keys(saver._redis, thread_id)
        assert keys
        ttls = [await saver._redis.ttl(key) for key in keys]
        assert all(ttl > 0 for ttl in ttls)
    finally:
        await saver.__aexit__(None, None, None)

    rebuilt = await _open_saver(ttl_minutes=1)
    try:
        restored = await _counter_graph(rebuilt).ainvoke({}, config=config)
        assert restored["count"] == 3
        history = [item async for item in rebuilt.alist(config)]
        assert len(history) > 2
    finally:
        await rebuilt.adelete_thread(thread_id)
        await rebuilt.__aexit__(None, None, None)


async def test_real_redis_l4_roundtrip_and_rebuild(real_redis) -> None:
    del real_redis
    thread_id = f"globex-real-test:l4:{os.getpid()}:{time.time_ns()}"
    intent = SubmitIntentInput(
        shopping_session_id="real-l4-session",
        buyer_id="real-buyer",
        locale="zh-CN",
        currency="CNY",
        raw_query="预算 500 元",
    )
    saver = await _open_saver(ttl_minutes=1)
    try:
        first = LangGraphAgent("main", _context_graph(saver), thread_id=thread_id)
        request_context = await first.prepare_session_context(intent)
        assert await first.reply(intent.raw_query, session_context=request_context) == "done"
        completed = await first.reduce_session_context(
            intent,
            [
                {
                    "type": "tool.invoke",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": "real-search",
                        "args": {"normalized_query": "bag", "ship_to": "US"},
                    },
                },
                {
                    "type": "tool.result",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": "real-search",
                        "model_output": {"hits": [{"item_id": "real-item"}]},
                    },
                },
            ],
        )
        assert completed.revision == 2
        replayed_request = await first.prepare_session_context(intent)
        assert await first.reply(intent.raw_query, session_context=replayed_request) == "done"
        frozen_state = await first._graph.aget_state(_config(thread_id))
        assert frozen_state.values["freeze_cursor"] == 2
        assert len(frozen_state.values["frozen_segments"]) == 1
        summary = summarize_frozen_segments(
            [FrozenSegment.from_dict(item) for item in frozen_state.values["frozen_segments"]]
        )
        await first._graph.aupdate_state(
            _config(thread_id),
            {"stage_summary": summary.to_dict()},
            as_node="agent",
        )
        frozen_state = await first._graph.aget_state(_config(thread_id))
    finally:
        await saver.__aexit__(None, None, None)

    rebuilt = await _open_saver(ttl_minutes=1)
    try:
        second = LangGraphAgent("main", _context_graph(rebuilt), thread_id=thread_id)
        restored = await second.get_session_context()
        assert restored == completed
        assert restored.last_search["tool_call_id"] == "real-search"
        restored_state = await second._graph.aget_state(_config(thread_id))
        assert restored_state.values["freeze_cursor"] == 2
        assert restored_state.values["frozen_segments"] == frozen_state.values["frozen_segments"]
        assert restored_state.values["stage_summary"] == summary.to_dict()
    finally:
        await rebuilt.adelete_thread(thread_id)
        await rebuilt.__aexit__(None, None, None)


async def test_real_redis_phase_f_long_session_rebuild(real_redis) -> None:
    del real_redis
    report = run_evaluation()
    expected = report["checkpoint_state"]
    thread_id = f"globex-real-test:phase-f:{os.getpid()}:{time.time_ns()}"
    config = _config(thread_id)
    saver = await _open_saver(ttl_minutes=1)
    try:
        await _phase_f_snapshot_graph(saver).ainvoke(expected, config=config)
        persisted = await _phase_f_snapshot_graph(saver).aget_state(config)
        assert persisted.values["session_context"]["revision"] == 40
    finally:
        await saver.__aexit__(None, None, None)

    rebuilt = await _open_saver(ttl_minutes=1)
    try:
        snapshot = await _phase_f_snapshot_graph(rebuilt).aget_state(config)
        values = snapshot.values
        assert values["session_context"] == expected["session_context"]
        assert values["freeze_cursor"] == expected["freeze_cursor"]
        assert values["frozen_segments"] == expected["frozen_segments"]
        assert values["stage_summary"] == expected["stage_summary"]
        assert values["budget_report"] == expected["budget_report"]
        assert values["budget_decision"] == expected["budget_decision"]
        assert values["context_compression"] == expected["context_compression"]
        assert len(values["messages"]) == len(expected["messages"])
        cursor = int(values["freeze_cursor"])
        assert _digest_messages(values["messages"][cursor:]) == report["active_digest"]
        assert len(values["frozen_segments"]) == 19
        assert len(values["stage_summary"]["source_segment_ids"]) == 18
        units = group_interaction_units(values["messages"])
        assert len(units) == 20
        assert all(unit.complete for unit in units)
    finally:
        await rebuilt.adelete_thread(thread_id)
        await rebuilt.__aexit__(None, None, None)


async def test_real_redis_thread_isolation_and_ttl_refresh(real_redis) -> None:
    saver = await _open_saver(ttl_minutes=1)
    identity = ThreadIdentity(environment="test", hmac_key="real-redis-test")
    main_a = identity.main_thread_id("real-session-a")
    main_b = identity.main_thread_id("real-session-b")
    dispatch = identity.dispatch_id(main_a, "real-tool-call", 0, "search_agent", "lamp")
    search = identity.child_thread_id(main_a, "search_agent", dispatch)
    trade = identity.child_thread_id(main_a, "trade_agent", dispatch)
    thread_ids = [main_a, main_b, search, trade]
    try:
        graph = _counter_graph(saver)
        for thread_id in thread_ids:
            result = await graph.ainvoke({"count": 0}, config=_config(thread_id))
            assert result["count"] == 1
        next_a = await graph.ainvoke({}, config=_config(main_a))
        assert next_a["count"] == 2
        for thread_id in thread_ids[1:]:
            state = await saver.aget_tuple(_config(thread_id))
            assert state is not None
            assert state.checkpoint["channel_values"]["count"] == 1

        keys = await _checkpoint_keys(saver._redis, main_a)
        assert keys
        await asyncio.sleep(2)
        stale_ttls = [await saver._redis.ttl(key) for key in keys]
        stale_ttl = min(stale_ttls)
        assert stale_ttl > 0
        await saver.aget_tuple(_config(main_a))
        refreshed_ttls = [await saver._redis.ttl(key) for key in keys]
        refreshed_ttl = min(refreshed_ttls)
        assert refreshed_ttl >= stale_ttl
    finally:
        for thread_id in thread_ids:
            await saver.adelete_thread(thread_id)
        await saver.__aexit__(None, None, None)


async def test_real_redis_child_reentry_and_done_marker(real_redis) -> None:
    del real_redis
    saver = await _open_saver(ttl_minutes=1)
    identity = ThreadIdentity(environment="test", hmac_key="real-redis-child")
    marker_store = RedisDispatchResultStore(
        REAL_REDIS_URL,
        identity,
        ttl_seconds=60,
    )
    await marker_store.startup()
    first_agents: list[LangGraphAgent] = []
    second_agents: list[LangGraphAgent] = []

    class AgentFactory:
        def __init__(self, agents: list[LangGraphAgent]) -> None:
            self.agents = agents

        def build(self):
            agent = LangGraphAgent("search", _child_graph(saver))
            self.agents.append(agent)
            return agent

    bus = TradeEventBus()
    tool_first = build_task_dispatch_tool(
        AgentFactory(first_agents),
        AgentFactory(first_agents),
        bus,
        identity=identity,
        result_store=marker_store,
    )
    tool_second = build_task_dispatch_tool(
        AgentFactory(second_agents),
        AgentFactory(second_agents),
        bus,
        identity=identity,
        result_store=marker_store,
    )
    session_id = "real-child-session"
    main_thread_id = identity.main_thread_id(session_id)
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id=session_id,
            buyer_id="real-buyer",
            locale="zh-CN",
            currency="CNY",
            main_thread_id=main_thread_id,
        )
    )
    dispatch = {
        "dispatches": [
            {"subagent_type": "search_agent", "demands": "camping lamp"}
        ]
    }
    try:
        first = json.loads(
            (await tool_first.arun(dispatch, tool_call_id="real-child-call")).content
        )
        second = json.loads(
            (await tool_second.arun(dispatch, tool_call_id="real-child-call")).content
        )
    finally:
        ShoppingContext.reset(token)
    first_item = first["dispatches"][0]
    second_item = second["dispatches"][0]
    try:
        assert first_item["thread_id"] == second_item["thread_id"]
        assert second_item["cached"] is True
        assert len(first_agents) == 1
        assert second_agents == []
        child_state = await saver.aget_tuple(_config(first_item["thread_id"]))
        assert child_state is not None
        assert await marker_store.get(first_item["dispatch_id"]) is not None
    finally:
        await saver.adelete_thread(first_item["thread_id"])
        await marker_store._client.delete(marker_store._key(first_item["dispatch_id"]))
        await marker_store.close()
        await saver.__aexit__(None, None, None)


async def test_real_redis_task_dispatch_pending_reentry(real_redis) -> None:
    del real_redis
    saver = await _open_saver(ttl_minutes=1)
    identity = ThreadIdentity(environment="test", hmac_key="real-redis-retry")
    marker_store = RedisDispatchResultStore(
        REAL_REDIS_URL,
        identity,
        ttl_seconds=60,
    )
    await marker_store.startup()
    first_agents: list[object] = []
    second_agents: list[object] = []

    class FlakyAgent:
        def __init__(self, graph, *, fail_once: bool) -> None:
            self.graph = graph
            self.agent = LangGraphAgent("search", graph)
            self.fail_once = fail_once

        async def reply(self, query: str, *, thread_id: str, event_sink=None) -> str:
            output = await self.agent.reply(
                query,
                thread_id=thread_id,
                event_sink=event_sink,
            )
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("simulated transient after checkpoint")
            return output

    class AgentFactory:
        def __init__(self, agents: list[object], *, fail_once: bool) -> None:
            self.agents = agents
            self.fail_once = fail_once

        def build(self):
            graph = _RecordingGraph(_approval_graph(saver))
            agent = FlakyAgent(graph, fail_once=self.fail_once)
            self.agents.append(agent)
            return agent

    bus = TradeEventBus()
    first_factory = AgentFactory(first_agents, fail_once=True)
    second_factory = AgentFactory(second_agents, fail_once=False)
    first_tool = build_task_dispatch_tool(
        first_factory,
        first_factory,
        bus,
        identity=identity,
        result_store=marker_store,
    )
    second_tool = build_task_dispatch_tool(
        second_factory,
        second_factory,
        bus,
        identity=identity,
        result_store=marker_store,
    )
    session_id = "real-retry-session"
    main_thread_id = identity.main_thread_id(session_id)
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id=session_id,
            buyer_id="real-buyer",
            locale="zh-CN",
            currency="CNY",
            main_thread_id=main_thread_id,
        )
    )
    dispatch = {
        "dispatches": [
            {"subagent_type": "search_agent", "demands": "approve lamp"}
        ]
    }
    try:
        first = json.loads(
            (await first_tool.arun(dispatch, tool_call_id="real-retry-call")).content
        )
        second = json.loads(
            (await second_tool.arun(dispatch, tool_call_id="real-retry-call")).content
        )
    finally:
        ShoppingContext.reset(token)

    first_item = first["dispatches"][0]
    second_item = second["dispatches"][0]
    try:
        assert first_item["thread_id"] == second_item["thread_id"]
        assert first_item["output"].startswith("[error] simulated transient")
        assert second_item["output"] == "approved"
        assert second_item.get("cached") is not True
        assert len(first_agents) == 1
        assert len(second_agents) == 1
        assert first_agents[0].graph.inputs == [
            {"messages": [("user", "approve lamp")]}
        ]
        assert second_agents[0].graph.inputs == [None]
        assert await marker_store.get(second_item["dispatch_id"]) is not None
    finally:
        await saver.adelete_thread(first_item["thread_id"])
        await marker_store._client.delete(marker_store._key(first_item["dispatch_id"]))
        await marker_store.close()
        await saver.__aexit__(None, None, None)


async def test_real_redis_interrupt_reentry(real_redis) -> None:
    del real_redis
    identity = ThreadIdentity(environment="test", hmac_key="real-redis-interrupt")
    main_thread_id = identity.main_thread_id("real-interrupt-session")
    dispatch_id = identity.dispatch_id(
        main_thread_id,
        "real-interrupt-call",
        0,
        "search_agent",
        "approve",
    )
    thread_id = identity.child_thread_id(main_thread_id, "search_agent", dispatch_id)
    config = _config(thread_id)
    saver = await _open_saver(ttl_minutes=1)
    try:
        first_agent = LangGraphAgent(
            "search",
            _approval_graph(saver),
            thread_id=thread_id,
        )
        interrupted = await first_agent.reply("approve", thread_id=thread_id)
        assert interrupted == "approve"
        snapshot = await first_agent._graph.aget_state(config)
        assert snapshot.next
        history = [item async for item in saver.alist(config)]
        assert len(history) > 1
    finally:
        await saver.__aexit__(None, None, None)

    rebuilt = await _open_saver(ttl_minutes=1)
    try:
        second_agent = LangGraphAgent(
            "search",
            _approval_graph(rebuilt),
            thread_id=thread_id,
        )
        resumed = await second_agent.reply("approve", thread_id=thread_id)
        assert resumed == "approved"
    finally:
        await rebuilt.adelete_thread(thread_id)
        await rebuilt.__aexit__(None, None, None)


async def test_real_client_startup_fails_fast_when_endpoint_is_down(monkeypatch) -> None:
    monkeypatch.setenv("GLOBEX_ENV", "local")
    monkeypatch.setenv("CHECKPOINT_REDIS_URL", "redis://127.0.0.1:63999/0")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("LANGGRAPH_AES_KEY", raising=False)
    settings = load_settings()
    with pytest.raises(CheckpointConfigurationError):
        await asyncio.wait_for(
            _create_checkpoint_for_test(settings),
            timeout=5,
        )


async def _create_checkpoint_for_test(settings) -> None:
    saver = AsyncRedisSaver(
        redis_url=settings.checkpoint_redis_url,
        connection_args={"socket_connect_timeout": 0.5, "socket_timeout": 0.5},
    )
    try:
        await saver.asetup()
    except Exception as err:  # noqa: BLE001 - expected unavailable endpoint
        raise CheckpointConfigurationError("real Redis startup failed fast") from err
    finally:
        await saver.__aexit__(None, None, None)
