"""Contract tests for the retained 14-case regression dataset."""

import json
from pathlib import Path

from scripts.eval_flow_queries import _load_queries
from scripts.eval_flow_rubric import (
    _flow_summary,
    _tool_outputs,
    build_final_judge_input,
    render_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "eval" / "flow_regression_v2.jsonl"


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_flow_regression_dataset_has_14_unique_cases_and_four_classes() -> None:
    rows = _jsonl(DATASET_PATH)
    assert len(rows) == 14
    assert len({row["case_id"] for row in rows}) == 14
    assert len({row["query_id"] for row in rows}) == 14
    assert {row["case_class"] for row in rows} == {
        "historical_baseline",
        "rag_hit",
        "product_only",
        "no_evidence",
    }


def test_case_contracts_keep_valid_knowledge_and_product_boundaries() -> None:
    for row in _jsonl(DATASET_PATH):
        assert row["query"].strip()
        if "tool_expectation" not in row:
            continue
        assert row["tool_expectation"]["required"]
        if "quality_expectation" in row:
            assert set(row["quality_expectation"]) == {"p0", "p1", "p2"}
        if "evidence_scope" not in row:
            continue
        assert set(row["evidence_scope"]) >= {"category_card_ids", "product_item_ids"}
        if row["case_class"] == "no_evidence":
            assert row["product_expectation"]["product_hit"] is False
            assert not row["evidence_scope"]["product_item_ids"]


def test_final_judge_marks_missing_online_evidence_inconclusive() -> None:
    item = {
        "query": "查询商品",
        "final_result": {
            "text": "证据不足",
            "recommended_cards": [],
            "verification_status": "unavailable",
        },
        "events": [],
    }
    summary = _flow_summary(item)
    assert summary["flow_valid"] is False
    assert summary["gate_trustworthy"] is False


def test_event_capture_error_makes_flow_inconclusive_even_with_partial_events() -> None:
    item = {
        "query": "查询商品",
        "final_result": {
            "text": "已完成",
            "recommended_cards": [],
            "verification_status": "supported",
        },
        "events": [
            {
                "type": "tool.result",
                "payload": {"model_output": {"hits": []}},
            }
        ],
        "event_capture_error": "RuntimeError: missing final.result",
    }
    summary = _flow_summary(item)
    assert summary["flow_valid"] is False


def test_report_keeps_candidate_authority_explicit() -> None:
    report = render_report([])
    assert "候选/诊断" in report


def test_final_judge_module_only_exposes_final_judge_protocol() -> None:
    from scripts import eval_flow_rubric

    assert not hasattr(eval_flow_rubric, "verify_claims")


def test_query_loader_preserves_flow_regression_contract_fields() -> None:
    rows = _load_queries(DATASET_PATH)
    row = next(item for item in rows if item["split"] == "regression_v2")
    for key in (
        "knowledge_expectation",
        "product_expectation",
        "tool_expectation",
        "quality_expectation",
        "evidence_scope",
    ):
        assert key in row


def test_final_judge_payload_uses_only_model_output_and_keeps_full_variants() -> None:
    card = {
        "item_id": "item-1",
        "title": "完整商品",
        "variants": [
            {"variant_id": "sku-1", "price_major": 1, "availability": "available"},
            {"variant_id": "sku-2", "price_major": 2, "availability": "available"},
        ],
    }
    item = {
        "query": "推荐商品",
        "knowledge_expectation": {"rag_hit": False, "card_ids": []},
        "product_expectation": {"product_hit": True, "item_ids": ["item-1"]},
        "tool_expectation": {"required": ["product_search_tool"]},
        "final_result": {
            "text": "推荐",
            "recommended_cards": [card],
            "verification_status": "supported",
        },
        "events": [
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "call-1",
                    "args": {"ship_to": None, "target_currency": "CNY"},
                },
            },
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "call-1",
                    "hits": [],
                    "provenance": {"hidden": True},
                    "evidence_snapshots": [{"hidden": True}],
                    "model_output": {"hits": [card]},
                },
            },
            {
                "type": "evidence.verify",
                "payload": {"verification_status": "supported"},
            },
        ],
    }
    outputs = _tool_outputs(item)
    assert len(outputs) == 1
    assert "evidence_snapshots" not in outputs[0]
    assert "provenance" not in outputs[0]
    assert len(outputs[0]["model_output"]["hits"][0]["variants"]) == 2
    external = build_final_judge_input(item)
    assert set(external) == {
        "query",
        "draft",
        "recommended_cards",
        "online_verification",
        "tool_outputs",
    }
    assert external["recommended_cards"][0]["item_id"] == "item-1"
    assert external["online_verification"]["verification_status"] == "supported"
    external_text = json.dumps(external, ensure_ascii=False)
    assert "evidence_snapshots" not in external_text
    assert '"hidden"' not in external_text


def test_final_judge_payload_redacts_free_text_pii_and_structured_shipping_fields() -> None:
    item = {
        "query": "请给 buyer@example.com 拨打 13800138000",
        "final_result": {
            "text": "收货人张三，电话 13800138000，推荐",
            "recommended_cards": [],
            "verification_status": "supported",
        },
        "events": [
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "tool_call_id": "call-1",
                    "model_output": {
                        "hits": [
                            {
                                "item_id": "item-full",
                                "variants": [
                                    {"variant_id": "sku-full-001"},
                                    {"variant_id": "sku-full-002"},
                                ],
                            }
                        ],
                        "address": "北京市朝阳区某路",
                        "recipient": "张三",
                        "phone": "13800138000",
                        "postal": "100000",
                        "shipping_summary": "收货地址摘要",
                        "confirmation_token": "confirm-secret",
                        "token": "token-secret",
                        "evidence_snapshots": [{"audit": True}],
                        "provenance": {"internal": True},
                    },
                },
            },
            {
                "type": "evidence.verify",
                "payload": {"verification_status": "supported"},
            },
        ],
    }
    external_text = json.dumps(build_final_judge_input(item), ensure_ascii=False)
    for secret in (
        "buyer@example.com",
        "13800138000",
        "confirm-secret",
        "token-secret",
        "evidence_snapshots",
        "provenance",
    ):
        assert secret not in external_text
    assert "sku-full-001" in external_text
    assert "sku-full-002" in external_text


def test_final_judge_score_and_pass_are_separate_from_completed() -> None:
    from scripts.eval_flow_rubric import _computed_score, _p0_pass

    payload = {
        "p0": [{"status": "supported"}],
        "p1": [{"status": "unsupported"}],
        "p2": [{"status": "supported"}],
        "score": 0.65,
    }
    assert _computed_score(payload) == 0.65
    assert _p0_pass(payload)

    inconclusive = {
        "p0": [{"status": "inconclusive"}],
        "p1": [],
        "p2": [],
        "score": 0.0,
    }
    assert _computed_score(inconclusive) is None
