from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.agents.identity import ThreadIdentity
from globex_agent.application.agents.main_agent import MainAgentState
from globex_agent.application.agents.orchestrator import (
    MainAgentOrchestrator,
    SubmitIntentInput,
)
from globex_agent.application.agents.session_lock import SessionLockRegistry
from globex_agent.application.context.models import L4Context
from globex_agent.application.context.reducer import reduce_l4
from globex_agent.application.evidence_verification import (
    EvidenceJudgeResult,
    FactGuardResult,
)
from globex_agent.infrastructure.eventbus import TradeEventBus


def _card(item_id: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "title": f"商品 {item_id}",
        "ships_to": ["US", "CA"],
        "variants": [],
    }


class _ContextAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.context = L4Context()
        self.fail = fail
        self.commits: list[list[dict[str, Any]]] = []

    async def prepare_session_context(self, intent: SubmitIntentInput) -> L4Context:
        return reduce_l4(intent, (), self.context, now="request-start")

    async def reply(
        self,
        query: str,
        *,
        thread_id: str,
        event_sink,
        session_context: L4Context,
    ) -> str:
        del query, thread_id
        # The real graph checkpointer applies this value as its input before
        # entering the model node.
        self.context = L4Context.from_dict(session_context.to_dict())
        await event_sink(
            "tool.invoke",
            {
                "tool": "product_search_tool",
                "tool_call_id": "search-a",
                "args": {"normalized_query": "bag", "ship_to": "US"},
            },
        )
        if self.fail:
            raise RuntimeError("model failed after tool invocation")
        await event_sink(
            "tool.invoke",
            {
                "tool": "product_search_tool",
                "tool_call_id": "search-b",
                "args": {"normalized_query": "backpack", "ship_to": "CA"},
            },
        )
        await event_sink(
            "tool.result",
            {
                "tool": "product_search_tool",
                "tool_call_id": "search-b",
                "model_output": {"hits": [_card("item-b")]},
            },
        )
        await event_sink(
            "tool.result",
            {
                "tool": "product_search_tool",
                "tool_call_id": "search-a",
                "model_output": {"hits": [_card("item-a")]},
            },
        )
        return '{"answer_text":"推荐 A","selections":[{"item_id":"item-a"}]}'

    async def reduce_session_context(
        self,
        intent: SubmitIntentInput,
        events: list[dict[str, Any]],
    ) -> L4Context:
        self.commits.append(list(events))
        self.context = reduce_l4(intent, events, self.context, now="turn-end")
        return self.context

class _Sessions:
    def __init__(self, agent: _ContextAgent) -> None:
        self.agent = agent
        self.identity = ThreadIdentity(environment="test", hmac_key="phase-b")
        self.locks = SessionLockRegistry()

    def thread_id(self, session_id: str) -> str:
        return self.identity.main_thread_id(session_id)

    def session_lock(self, session_id: str):
        return self.locks.acquire(self.thread_id(session_id))

    async def get_or_create(self, session_id: str) -> _ContextAgent:
        del session_id
        return self.agent


class _Preferences:
    async def list_by_buyer(self, buyer_id: str) -> list:
        del buyer_id
        return []


class _Verifier:
    async def verify(self, **kwargs):
        del kwargs
        return FactGuardResult("supported"), EvidenceJudgeResult(
            verdict="supported",
            reason="supported by frozen search",
        )


def _intent() -> SubmitIntentInput:
    return SubmitIntentInput(
        shopping_session_id="session-b",
        buyer_id="buyer-b",
        locale="zh-CN",
        currency="CNY",
        raw_query="预算 500 元的包",
    )


async def test_orchestrator_commits_request_start_then_ordered_success_events() -> None:
    agent = _ContextAgent()
    orchestrator = MainAgentOrchestrator(
        sessions=_Sessions(agent),
        bus=TradeEventBus(),
        preference_store=_Preferences(),
        evidence_verifier=_Verifier(),
    )

    result = await orchestrator.handle_intent(_intent())

    assert result.verification_status == "supported"
    assert agent.context.revision == 2
    assert agent.context.request == {
        "session_origin_raw_query": "预算 500 元的包",
        "current_raw_query": "预算 500 元的包",
    }
    assert agent.context.last_search["tool_call_id"] == "search-a"
    assert agent.context.last_search["args"]["ship_to"] == "US"
    assert len(agent.commits) == 1
    event_types = [event["type"] for event in agent.commits[0]]
    assert event_types.index("tool.invoke") < event_types.index("tool.result")
    assert event_types[-1] == "final.result"
    assert (await agent.prepare_session_context(_intent())).revision == 2


async def test_orchestrator_failure_keeps_only_request_start_checkpoint() -> None:
    agent = _ContextAgent(fail=True)
    orchestrator = MainAgentOrchestrator(
        sessions=_Sessions(agent),
        bus=TradeEventBus(),
        preference_store=_Preferences(),
        evidence_verifier=_Verifier(),
    )

    result = await orchestrator.handle_intent(_intent())

    assert result.verification_status == "unavailable"
    assert agent.context.revision == 1
    assert agent.context.last_search is None
    assert agent.commits == []


async def test_graph_input_checkpoints_request_context_before_node_failure() -> None:
    def fail(state: MainAgentState):
        del state
        raise RuntimeError("model node failed")

    builder = StateGraph(MainAgentState)
    builder.add_node("agent", fail)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    agent = LangGraphAgent(
        "main",
        builder.compile(checkpointer=InMemorySaver()),
        thread_id="phase-b-failure",
    )
    request_context = await agent.prepare_session_context(_intent())

    with pytest.raises(RuntimeError, match="model node failed"):
        await agent.reply(_intent().raw_query, session_context=request_context)

    checkpointed = await agent.get_session_context()
    assert checkpointed.revision == 1
    assert checkpointed.request["current_raw_query"] == _intent().raw_query
    assert checkpointed.last_search is None
