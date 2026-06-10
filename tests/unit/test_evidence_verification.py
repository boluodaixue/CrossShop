from __future__ import annotations

import json

import httpx
import pytest

from globex_agent.application.evidence_verification import (
    EvidenceJudgeResult,
    EvidenceVerificationService,
    OpenAIEvidenceJudge,
    RecommendationDraft,
    SearchQuoteContext,
    build_semantic_evidence_bundle,
    fact_guard,
    hydrate_recommended_cards,
    normalize_evidence_judge_result,
    parse_recommendation_draft,
)


def _card(item_id: str = "item-1") -> dict:
    return {
        "item_id": item_id,
        "title": "真实商品",
        "price_major": 10,
        "currency": "CNY",
        "availability": "available",
        "variants": [
            {
                "variant_id": "sku-完整-001",
                "display_name": "颜色: 黑色",
                "price_major": 10,
                "currency": "CNY",
                "availability": "available",
                "landed_price": {
                    "ship_to": "CN",
                    "quantity": 1,
                    "subtotal_major": 10,
                    "freight_major": 25,
                    "tariff_major": 0,
                    "landed_total_major": 35,
                    "currency": "CNY",
                },
            }
        ],
    }


def test_minimal_draft_rejects_legacy_evidence_fields() -> None:
    draft, error = parse_recommendation_draft(
        '{"answer_text":"推荐","selections":[],"claims":[]}'
    )
    assert draft is None
    assert error and "Extra" in error


def test_fact_guard_checks_top_k_identity_variant_and_arithmetic() -> None:
    draft = RecommendationDraft(
        answer_text="推荐黑色款",
        selections=[{"item_id": "item-1", "variant_id": "sku-完整-001"}],
    )
    result = fact_guard(
        draft,
        top_k_cards=[_card()],
        expected_ship_to="CN",
        expected_currency="CNY",
    )
    assert result.passed
    hydrated = hydrate_recommended_cards([_card()], draft.selections)
    assert hydrated[0]["item_id"] == "item-1"
    assert hydrated[0]["selected_variant_id"] == "sku-完整-001"


def test_semantic_evidence_projects_only_selected_variant_and_keeps_speakable_facts() -> None:
    card = _card()
    card["highlights"] = ["轻便"]
    card["provenance"] = {"internal": True}
    card["evidence_snapshots"] = [{"internal": True}]
    bundle = build_semantic_evidence_bundle(
        query="推荐商品",
        draft=RecommendationDraft(
            answer_text="推荐黑色款",
            selections=[{"item_id": "item-1", "variant_id": "sku-完整-001"}],
        ),
        top_k_cards=[card],
        category_insight={
            "insights": {"price_tiers": ["参考"]},
            "source_boundary": {"scope": "category_aggregate_reference"},
            "provenance": [{"secret": "audit"}],
        },
    )
    product = bundle["products"][0]
    assert product["highlights"] == ["轻便"]
    assert product["selected_variant"]["variant_id"] == "sku-完整-001"
    assert "variants" not in product
    assert "provenance" not in product
    assert "evidence_snapshots" not in product
    assert bundle["category_insight"]["source_boundary"]["scope"] == (
        "category_aggregate_reference"
    )
    assert "provenance" not in bundle["category_insight"]


def test_fact_guard_rejects_out_of_top_k_and_bad_quote_without_fuzzy_binding() -> None:
    draft = RecommendationDraft(
        answer_text="两个同价商品都推荐",
        selections=[{"item_id": "item-2", "variant_id": None}],
    )
    result = fact_guard(draft, top_k_cards=[_card()])
    assert not result.passed
    assert any("frozen Top-K" in error for error in result.errors)
    broken = _card()
    broken["variants"][0]["landed_price"]["landed_total_major"] = 36
    result = fact_guard(
        RecommendationDraft(answer_text="推荐", selections=[]),
        top_k_cards=[broken],
    )
    assert any("does not equal" in error for error in result.errors)


def test_fact_guard_rejects_quote_without_ship_to_and_uses_card_context() -> None:
    result = fact_guard(
        RecommendationDraft(answer_text="推荐", selections=[]),
        top_k_cards=[_card()],
        card_contexts={"item-1": SearchQuoteContext(ship_to=None, target_currency="CNY")},
    )
    assert not result.passed
    assert any("without ship_to" in error for error in result.errors)

    wrong_destination = _card()
    quote = wrong_destination["variants"][0]["landed_price"]
    quote["ship_to"] = "US"
    quote["currency"] = "USD"
    result = fact_guard(
        RecommendationDraft(answer_text="推荐", selections=[]),
        top_k_cards=[wrong_destination],
        card_contexts={"item-1": {"ship_to": "CA", "target_currency": "CAD"}},
    )
    assert not result.passed
    assert any("destination" in error for error in result.errors)
    assert any("currency" in error for error in result.errors)


def test_fact_guard_requires_each_priced_available_variant_quote() -> None:
    card = _card()
    del card["variants"][0]["landed_price"]
    result = fact_guard(
        RecommendationDraft(answer_text="推荐", selections=[]),
        top_k_cards=[card],
        card_contexts={"item-1": {"ship_to": "CN", "target_currency": "CNY"}},
    )
    assert not result.passed
    assert any("missing quote" in error for error in result.errors)


def test_fact_guard_fails_closed_when_frozen_card_context_is_missing() -> None:
    result = fact_guard(
        RecommendationDraft(answer_text="推荐", selections=[]),
        top_k_cards=[_card()],
        card_contexts={},
    )
    assert not result.passed
    assert any("missing search quote context" in error for error in result.errors)


def test_fact_guard_compatibility_expected_context_still_checks_quotes() -> None:
    result = fact_guard(
        RecommendationDraft(answer_text="推荐", selections=[]),
        top_k_cards=[_card()],
        expected_ship_to="CN",
        expected_currency="CNY",
    )
    assert result.passed


def test_fact_guard_rejects_unavailable_selected_item() -> None:
    card = _card()
    card["availability"] = "unavailable"
    result = fact_guard(
        RecommendationDraft(
            answer_text="推荐",
            selections=[{"item_id": "item-1", "variant_id": "sku-完整-001"}],
        ),
        top_k_cards=[card],
    )
    assert not result.passed
    assert any("selected item is not available" in error for error in result.errors)


def test_evidence_judge_result_fails_closed_on_conflicting_claims() -> None:
    with pytest.raises(ValueError):
        EvidenceJudgeResult(
            verdict="supported",
            unsupported_claims=[{"text": "商品价格", "reason": "缺证据"}],
        )
    with pytest.raises(ValueError):
        EvidenceJudgeResult(
            verdict="unsupported",
            unsupported_claims=[{"text": "", "reason": "缺证据"}],
        )
    compatible = EvidenceJudgeResult(verdict="unsupported", reason="整体不支持")
    assert compatible.unsupported_claims == []


def test_evidence_judge_claim_alias_normalizes_to_standard_text() -> None:
    normalized = normalize_evidence_judge_result(
        {
            "verdict": "unsupported",
            "unsupported_claims": [{"claim": "商品价格", "reason": "缺证据"}],
            "reason": "部分不支持",
        }
    )
    result = EvidenceJudgeResult.model_validate(normalized)
    assert result.unsupported_claims[0].text == "商品价格"
    assert result.unsupported_claims[0].reason == "缺证据"


def test_evidence_judge_text_is_the_standard_output_key() -> None:
    result = EvidenceJudgeResult.model_validate(
        {
            "verdict": "unsupported",
            "unsupported_claims": [{"text": "商品价格", "reason": "缺证据"}],
            "reason": "部分不支持",
        }
    )
    assert result.unsupported_claims[0].text == "商品价格"


def test_evidence_judge_normalization_rejects_conflict_and_non_string_values() -> None:
    with pytest.raises(ValueError, match="both text and claim"):
        normalize_evidence_judge_result(
            {
                "verdict": "unsupported",
                "unsupported_claims": [
                    {"text": "商品价格", "claim": "另一个片段", "reason": "缺证据"}
                ],
            }
        )
    with pytest.raises(ValueError, match="must be a string"):
        normalize_evidence_judge_result(
            {
                "verdict": "unsupported",
                "unsupported_claims": [{"claim": 123, "reason": "缺证据"}],
            }
        )


def test_evidence_judge_normalization_keeps_unknown_fields_fail_closed() -> None:
    normalized = normalize_evidence_judge_result(
        {
            "verdict": "unsupported",
            "unsupported_claims": [
                {"claim": "商品价格", "reason": "缺证据", "extra": "拒绝"}
            ],
        }
    )
    with pytest.raises(ValueError):
        EvidenceJudgeResult.model_validate(normalized)


@pytest.mark.asyncio
async def test_online_judge_http_payload_redacts_pii_tokens_and_keeps_full_variants() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"verdict":"supported","reason":"整体支持"}'
                        }
                    }
                ]
            },
        )

    card = _card()
    card["variants"].append(
        {
            "variant_id": "sku-完整-002",
            "price_major": 11,
            "availability": "available",
        }
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await OpenAIEvidenceJudge(
            client,
            base_url="https://judge.invalid/v1",
            api_key="secret-key-not-for-payload",
            model="test-online-judge",
        ).judge(
            query="请联系 buyer@example.com 或 13800138000 推荐商品",
            draft=RecommendationDraft(
                answer_text="收货人张三，电话 13800138000，推荐真实商品",
                selections=[{"item_id": "item-1", "variant_id": "sku-完整-001"}],
            ),
            tool_outputs=[
                {
                    "tool": "product_search_tool",
                    "tool_call_id": "call-1",
                    "model_output": {
                        "hits": [card],
                        "recipient": "张三",
                        "phone": "13800138000",
                        "postal": "100000",
                        "shipping_summary": "北京市朝阳区某路",
                        "confirmation_token": "confirm-secret",
                        "token": "ordinary-secret",
                        "evidence_snapshots": [{"hidden": "audit"}],
                        "provenance": {"hidden": "audit"},
                    },
                }
            ],
        )
    finally:
        await client.aclose()

    assert result.verdict == "supported"
    body = captured["payload"]
    user_text = body["messages"][1]["content"]
    assert "buyer@example.com" not in user_text
    assert "13800138000" not in user_text
    assert "confirm-secret" not in user_text
    assert "ordinary-secret" not in user_text
    assert "evidence_snapshots" not in user_text
    assert "provenance" not in user_text
    assert "sku-完整-001" in user_text
    assert "sku-完整-002" in user_text


class _Judge:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    async def judge(self, **kwargs):
        del kwargs
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.asyncio
async def test_judge_transport_retries_without_generation_retry() -> None:
    judge = _Judge(
        [
            RuntimeError("ConnectError"),
            EvidenceJudgeResult(verdict="supported", reason="整体支持"),
        ]
    )
    service = EvidenceVerificationService(judge, judge_retries=1)
    guard, semantic = await service.verify(
        query="买真实商品",
        draft=RecommendationDraft(answer_text="推荐真实商品", selections=[{"item_id": "item-1"}]),
        top_k_cards=[_card()],
        tool_outputs=[{"hits": [_card()]}],
    )
    assert guard.passed
    assert semantic.verdict == "supported"
    assert judge.calls == 2
