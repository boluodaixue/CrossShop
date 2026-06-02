from __future__ import annotations

import json

from globex_agent.application.agents.orchestrator import _sanitize_audit_payload
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.category_insight_tool import (
    build_category_insight_tool,
)
from globex_agent.category_insight.models import CategoryEvidenceRef
from globex_agent.eval.fact_guard import validate_final_response
from globex_agent.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from globex_agent.infrastructure.eventbus import TradeEventBus
from scripts.eval_flow_rubric import parse_conversation


class _Run:
    class _Output:
        def model_dump(self, mode: str) -> dict:
            assert mode == "json"
            return {
                "category": "乳胶枕",
                "components": [],
                "bestsellers": [],
                "attributes": [],
                "price_tiers": [
                    {"tier": "budget", "range_cny": [100.0, 300.0], "notes": "参考"}
                ],
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


async def test_category_tool_returns_compact_refs_and_source_boundary() -> None:
    bus = TradeEventBus()
    queue = bus.subscribe("evidence-session")
    tool = build_category_insight_tool(_Service(), bus)
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="evidence-session",
            buyer_id="buyer-1",
            locale="zh-CN",
            currency="CNY",
        )
    )
    try:
        result = json.loads(await tool.ainvoke({"category": "乳胶枕", "depth": "quick"}))
    finally:
        ShoppingContext.reset(token)

    assert result["evidence_refs"] == [
        {
            "card_id": "card-1",
            "category": "乳胶枕",
            "card_type": "price_range",
            "last_updated": "2026-08-20T00:00:00+08:00",
            "confidence": 0.8,
        }
    ]
    assert result["source_boundary"]["scope"] == "category_aggregate_reference"
    assert "具体商品实时价格" in result["source_boundary"]["exclusions"]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    result_event = next(event for event in events if event.type == "tool.result")
    assert result_event.payload["evidence_refs"] == result["evidence_refs"]
    assert "raw_evidence" not in result_event.payload


def test_fact_guard_distinguishes_product_and_category_prices() -> None:
    product = [{"item_id": "taobao:1", "price_major": 199, "currency": "CNY"}]
    category = [{"insights": {"price_tiers": [{"range_cny": [100, 300]}]}}]

    assert (
        validate_final_response(
            "商品价格是199元。",
            product_facts=product,
            category_insights=category,
        )
        == ()
    )
    codes = {
        violation.code
        for violation in validate_final_response(
            "这款商品价格是300元。",
            product_facts=product,
            category_insights=category,
        )
    }
    assert codes == {"category_price_as_product_fact"}

    assert (
        validate_final_response(
            "乳胶枕品类参考价约200元。",
            product_facts=product,
            category_insights=category,
        )
        == ()
    )
    assert {
        violation.code
        for violation in validate_final_response(
            "这款商品约200元。",
            product_facts=product,
            category_insights=category,
        )
    } == {"unsourced_specific_price"}


def test_fact_guard_detects_unsupported_price_store_and_inventory() -> None:
    violations = validate_final_response(
        "店铺现货，库存充足，价格是499元。",
        product_facts=[{"item_id": "taobao:1", "availability": None}],
        category_insights=[],
    )
    assert {violation.code for violation in violations} == {
        "unsourced_specific_price",
        "unsupported_store_or_inventory",
    }


def test_fact_guard_allows_store_verification_request_but_not_store_claim() -> None:
    facts = [{"item_id": "taobao:1", "availability": None}]

    assert (
        validate_final_response(
            "材质未确认，建议向店铺确认。",
            product_facts=facts,
        )
        == ()
    )
    violations = validate_final_response(
        "店铺现货，建议向店铺确认。",
        product_facts=facts,
    )
    assert {
        violation.code for violation in violations
    } == {"unsupported_store_or_inventory"}


def test_flow_parser_preserves_product_price_bounds_for_fact_guard(tmp_path) -> None:
    path = tmp_path / "conversation.jsonl"
    rows = [
        {"kind": "turn", "role": "buyer", "content": "找一个商品"},
        {"kind": "turn", "role": "agent", "content": "价格是149元。"},
        {
            "kind": "event",
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "hits": [
                    {
                        "item_id": "item-1",
                        "title": "商品",
                        "category": "品类",
                        "price_major": None,
                        "price_min_major": 149.0,
                        "price_max_major": 219.0,
                        "currency": "CNY",
                    }
                ],
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    parsed = parse_conversation(path)

    assert parsed["facts"][0]["price_min_major"] == 149.0
    assert parsed["facts"][0]["price_max_major"] == 219.0
    assert parsed["fact_violations"] == []

    rows[1]["content"] = "价格是220元。"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    assert {
        violation["code"] for violation in parse_conversation(path)["fact_violations"]
    } == {"unsourced_specific_price"}


def test_product_fact_boundaries_are_explicit_in_agent_prompts() -> None:
    prompts = load_prompts()
    main_prompt = prompts["main_agent"]["system_prompt"]
    search_prompt = prompts["sub_agents"]["search"]["system_prompt"]

    for prompt in (main_prompt, search_prompt):
        assert "shop/store/store_name" in prompt
        assert "区间内部价" in prompt


def test_audit_payload_redacts_pii_and_omits_raw_evidence() -> None:
    payload = _sanitize_audit_payload(
        {
            "phone": "13800000000",
            "raw_evidence": ["full source text"],
            "evidence_refs": [{"card_id": "card-1"}],
        }
    )
    assert payload["phone"] == "[redacted]"
    assert payload["raw_evidence"] == "[omitted: use evidence_refs]"
    assert payload["evidence_refs"] == [{"card_id": "card-1"}]
