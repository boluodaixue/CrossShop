from __future__ import annotations

import asyncio

import pytest

from globex_agent.application.agents.orchestrator import (
    MainAgentOrchestrator,
    _top_k_cards_with_contexts,
)
from globex_agent.application.evidence_verification import (
    EvidenceJudgeResult,
    FactGuardResult,
)
from globex_agent.infrastructure.eventbus import TradeEventBus


def _card() -> dict:
    return {
        "item_id": "item-1",
        "title": "后端事实商品",
        "ships_to": [],
        "variants": [],
    }


def test_frozen_cards_bind_to_their_own_product_search_context() -> None:
    events = [
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-cn",
                "args": {"ship_to": "CN", "target_currency": "CNY"},
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-cn",
                "model_output": {"hits": [{**_card(), "item_id": "item-cn"}]},
                "evidence_snapshots": [{"secret": "audit-only"}],
            },
        },
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-us",
                "args": {"ship_to": "US", "target_currency": "USD"},
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "tool_call_id": "search-us",
                "model_output": {"hits": [{**_card(), "item_id": "item-us"}]},
                "evidence_snapshots": [{"secret": "audit-only"}],
            },
        },
    ]
    cards, contexts = _top_k_cards_with_contexts(events)
    assert [card["item_id"] for card in cards] == ["item-us"]
    assert contexts["item-us"].ship_to == "US"
    assert contexts["item-us"].target_currency == "USD"
    assert "evidence_snapshots" not in cards[0]


class _Sessions:
    def thread_id(self, session_id: str) -> str:
        return f"thread:{session_id}"


class _Agent:
    def __init__(self) -> None:
        self.calls = 0

    async def reply(self, query: str, *, thread_id: str, event_sink) -> str:
        del query, thread_id
        self.calls += 1
        await event_sink("token.delta", {"text": "unverified"})
        await event_sink(
            "tool.invoke",
            {
                "tool": "product_search_tool",
                "tool_call_id": "search-1",
                "args": {"ship_to": None, "target_currency": "CNY"},
            },
        )
        await event_sink(
            "tool.result",
            {
                "tool": "product_search_tool",
                "tool_call_id": "search-1",
                "model_output": {"hits": [_card()]},
            },
        )
        return '{"answer_text":"第一次草稿","selections":[{"item_id":"item-1"}]}'


class _Finalizer:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class _Verifier:
    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = verdicts
        self.calls = 0

    async def verify(self, **kwargs):
        del kwargs
        self.calls += 1
        verdict = self.verdicts.pop(0)
        return FactGuardResult("supported"), EvidenceJudgeResult(
            verdict=verdict,
            reason=f"judge-{self.calls}",
        )


class _SlowVerifier:
    async def verify(self, **kwargs):
        del kwargs
        await asyncio.sleep(1.05)
        return FactGuardResult("supported"), EvidenceJudgeResult(
            verdict="supported",
            reason="late",
        )


async def _run(verifier, finalizer=None):
    bus = TradeEventBus()
    trace = bus.subscribe("session-1")
    agent = _Agent()
    orchestrator = MainAgentOrchestrator(
        sessions=_Sessions(),
        bus=bus,
        preference_store=None,
        evidence_verifier=verifier,
        answer_finalizer=finalizer,
    )
    result = await orchestrator._reply_with_retry(
        session_id="session-1",
        agent=agent,
        query="推荐商品",
        trace=trace,
        captured_events=[],
    )
    return result, agent


@pytest.mark.asyncio
async def test_second_and_third_generation_reuse_frozen_evidence_and_hide_deltas() -> None:
    verifier = _Verifier(["unsupported", "unsupported", "supported"])
    finalizer = _Finalizer(
        [
            '{"answer_text":"修正一","selections":[{"item_id":"item-1"}]}',
            '{"answer_text":"修正二","selections":[{"item_id":"item-1"}]}',
        ]
    )
    result, agent = await _run(verifier, finalizer)

    assert agent.calls == 1
    assert verifier.calls == 3
    assert len(finalizer.prompts) == 2
    assert all("item-1" in prompt for prompt in finalizer.prompts)
    assert result[0] == "修正二"
    assert result[1][0]["item_id"] == "item-1"
    assert result[2] == "supported"


@pytest.mark.asyncio
async def test_judge_unavailable_returns_conservative_empty_cards_without_rewrite() -> None:
    verifier = _Verifier(["unavailable"])
    finalizer = _Finalizer(
        ['{"answer_text":"不应执行","selections":[{"item_id":"item-1"}]}']
    )
    result, agent = await _run(verifier, finalizer)

    assert agent.calls == 1
    assert verifier.calls == 1
    assert finalizer.prompts == []
    assert result[1] == []
    assert result[2] == "unavailable"


@pytest.mark.asyncio
async def test_evidence_verification_is_capped_by_turn_deadline() -> None:
    bus = TradeEventBus()
    trace = bus.subscribe("session-1")
    orchestrator = MainAgentOrchestrator(
        sessions=_Sessions(),
        bus=bus,
        preference_store=None,
        evidence_verifier=_SlowVerifier(),
        turn_timeout_seconds=1.0,
    )
    result = await orchestrator._reply_with_retry(
        session_id="session-1",
        agent=_Agent(),
        query="推荐商品",
        trace=trace,
        captured_events=[],
    )
    assert result[1] == []
    assert result[2] == "unavailable"
