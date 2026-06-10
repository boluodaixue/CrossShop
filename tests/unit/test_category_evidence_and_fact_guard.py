from __future__ import annotations

import json

from globex_agent.application.agents.orchestrator import _sanitize_audit_payload
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.category_insight_tool import build_category_insight_tool
from globex_agent.category_insight.models import CategoryEvidenceRef
from globex_agent.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from globex_agent.infrastructure.eventbus import TradeEventBus


class _Run:
    class _Output:
        def model_dump(self, mode: str) -> dict:
            assert mode == "json"
            return {
                "category": "乳胶枕",
                "components": [],
                "bestsellers": [],
                "attributes": [],
                "price_tiers": [{"tier": "budget", "range_cny": [100.0, 300.0]}],
                "confidence": 0.8,
            }

    output = _Output()
    diagnostics = type(
        "Diagnostics",
        (),
        {
            "evidence_refs": (
                CategoryEvidenceRef(
                    card_id="card-1",
                    category="乳胶枕",
                    card_type="price_range",
                    last_updated="2026-08-20T00:00:00+08:00",
                    confidence=0.8,
                ),
            )
        },
    )()


class _Service:
    def insight(self, category: str, *, depth: str):
        assert category == "乳胶枕"
        assert depth == "quick"
        return _Run()


async def test_category_tool_keeps_aggregate_boundary_and_exact_model_output() -> None:
    bus = TradeEventBus()
    queue = bus.subscribe("category-session")
    tool = build_category_insight_tool(_Service(), bus)
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="category-session",
            buyer_id="buyer-1",
            locale="zh-CN",
            currency="CNY",
        )
    )
    try:
        result = json.loads(await tool.ainvoke({"category": "乳胶枕", "depth": "quick"}))
    finally:
        ShoppingContext.reset(token)
    assert result["provenance"][0]["card_id"] == "card-1"
    assert result["source_boundary"]["scope"] == "category_aggregate_reference"
    assert "具体商品实时价格" in result["source_boundary"]["exclusions"]
    assert "category_result" not in result
    assert "evidence_catalog" not in result
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    result_event = next(event for event in events if event.type == "tool.result")
    assert result_event.payload["model_output"] == result


def test_audit_redacts_pii_but_keeps_complete_product_lists() -> None:
    payload = _sanitize_audit_payload(
        {
            "tool": "product_search_tool",
            "model_output": {
                "hits": [{"item_id": "sku-完整-001", "highlights": ["联系人 13800000000"]}],
                "variants": [{"variant_id": "very-long-sku-001"}],
            },
            "confirmation_token": "secret-token",
        }
    )
    assert payload["model_output"]["hits"][0]["item_id"] == "sku-完整-001"
    assert payload["model_output"]["variants"][0]["variant_id"] == "very-long-sku-001"
    assert "13800000000" not in json.dumps(payload, ensure_ascii=False)
    assert payload["confirmation_token"] == "[redacted]"


def test_prompt_requires_minimal_draft_and_backend_card_hydration() -> None:
    prompt = load_prompts()["main_agent"]["system_prompt"]
    assert "answer_text" in prompt
    assert "selections" in prompt
    assert "后端会从冻结工具输出" in prompt
    assert "evidence_verify" in prompt
