"""Thin application adapter for the mandatory evidence verification service.

The main orchestrator calls the service unconditionally.  This adapter exists
for local integration tests and service boundaries; it is not registered as a
model-authorized bypass tool.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from globex_agent.application.evidence_verification import (
    EvidenceVerificationService,
    parse_recommendation_draft,
)
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def build_evidence_verify_tool(
    service: EvidenceVerificationService,
    bus: TradeEventBus,
):
    @tool
    async def evidence_verify_tool(
        query: str,
        draft: dict[str, Any],
        top_k_cards: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
        card_contexts: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        parsed, error = parse_recommendation_draft(draft)
        if parsed is None:
            result = {"verification_status": "unsupported", "reason": error or "invalid draft"}
        else:
            guard, judge = await service.verify(
                query=query,
                draft=parsed,
                top_k_cards=top_k_cards,
                tool_outputs=tool_outputs,
                card_contexts=card_contexts,
            )
            result = {
                "verification_status": judge.verdict if guard.passed else "unsupported",
                "reason": judge.reason or "; ".join(guard.errors),
                "unsupported_claims": [
                    claim.model_dump(mode="json") for claim in judge.unsupported_claims
                ],
                "fact_guard_errors": list(guard.errors),
            }
        bus.publish(ShoppingContext.current_session_id(), "evidence.verify", result)
        return json.dumps(result, ensure_ascii=False)

    return evidence_verify_tool
