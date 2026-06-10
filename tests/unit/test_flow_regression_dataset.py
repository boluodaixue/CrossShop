"""Contract tests for the retained 14-case regression dataset."""

import json
from pathlib import Path

from scripts.eval_flow_queries import _load_queries
from scripts.eval_flow_rubric import (
    _flow_summary,
    _progress_line,
    _tool_outputs,
    aggregate_scores,
    build_final_judge_input,
    build_rubric_input,
    render_report,
    validate_judge_result,
    validate_rubric,
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


def test_rubric_contract_falls_back_to_same_case_pre_execution_rules() -> None:
    rubric_input = build_rubric_input(
        {
            "case_id": "product-only-floodlight",
            "query": "户外防水投光灯",
            "case_class": "product_only",
            "query_type": "buying_intent",
        }
    )
    assert rubric_input["flow_expectations"]["required_tools"] == [
        "product_search_tool"
    ]
    assert rubric_input["flow_expectations"]["optional_tools"] == [
        "category_insight_tool"
    ]
    assert rubric_input["flow_expectations"]["verified_final_success_status"] == "supported"
    assert rubric_input["flow_expectations"]["evidence_judge_success_status"] == "supported"
    assert rubric_input["flow_expectations"]["fact_guard_success_value"] is True


def test_p1_status_criterion_rejects_nonexistent_success_values() -> None:
    rubric = _rubric()
    rubric["p1"][0]["criterion"] = "verification_status 应为 verified"
    try:
        validate_rubric(rubric)
    except ValueError as error:
        assert "supported/true" in str(error)
    else:
        raise AssertionError("verified status criterion was accepted")

    rubric = _rubric()
    rubric["p1"][0]["criterion"] = "evidence_judge_status 应为 passed"
    try:
        validate_rubric(rubric)
    except ValueError as error:
        assert "supported/true" in str(error)
    else:
        raise AssertionError("passed status criterion was accepted")


def test_progress_line_is_compact_and_redacts_multiline_reason() -> None:
    completed = _progress_line(
        1,
        2,
        {
            "case_id": "case-a",
            "evaluation_status": "completed",
            "final_judge": {"final_score": 0.9, "quality_pass": True},
        },
        1.25,
    )
    assert completed == "[1/2] case-a status=completed score=0.9 PASS elapsed=1.2s"
    inconclusive = _progress_line(
        2,
        2,
        {
            "case_id": "case-b",
            "evaluation_status": "inconclusive",
            "final_judge": {"reason": "Schema\ninvalid"},
        },
        2.0,
    )
    assert inconclusive == "[2/2] case-b status=inconclusive reason=Schema invalid elapsed=2.0s"


def test_two_stage_inputs_hide_post_execution_data_and_compact_final_evidence() -> None:
    card = {
        "item_id": "item-1",
        "title": "完整商品",
        "selected_variant_id": "sku-1",
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
                "payload": {
                    "attempt": 1,
                    "verification_status": "supported",
                    "fact_guard_errors": [],
                },
            },
            {"type": "final.result", "payload": {"text": "推荐"}},
        ],
    }
    outputs = _tool_outputs(item)
    assert len(outputs) == 1
    assert "evidence_snapshots" not in outputs[0]
    assert "provenance" not in outputs[0]
    assert len(outputs[0]["model_output"]["hits"][0]["variants"]) == 2
    rubric_input = build_rubric_input(item)
    assert set(rubric_input) == {
        "query",
        "flow_type",
        "flow_expectations",
        "available_evidence",
    }
    rubric_text = json.dumps(rubric_input, ensure_ascii=False)
    for forbidden in (
        "answer_text",
        "recommended_cards",
        "tool_outputs",
        "tool_timeline",
        "online_verification",
        "item-1",
        "sku-1",
    ):
        assert forbidden not in rubric_text
    assert len(rubric_text.encode("utf-8")) < 4096

    external = build_final_judge_input(item, _rubric())
    assert set(external) == {
        "query",
        "rubric",
        "final",
        "execution",
        "online_gate",
    }
    compact_card = external["final"]["compact_recommended_cards"][0]
    assert compact_card == {
        "title": "完整商品",
        "selected_variant": {
            "options": None,
            "price": 1,
            "currency": None,
            "available": True,
        },
    }
    assert external["execution"] == {
        "flow_type": {"case_class": None, "query_type": None},
        "tool_sequence": ["product_search_tool"],
        "tool_counts": {"product_search_tool": 1},
        "generation_attempts": 1,
        "final_result_present": True,
    }
    assert external["online_gate"] == {
        "verification_status": "supported",
        "fact_guard_passed": True,
        "evidence_judge_status": "supported",
    }
    external_text = json.dumps(external, ensure_ascii=False)
    for forbidden in (
        "item-1",
        "sku-1",
        "sku-2",
        "evidence_snapshots",
        '"hidden"',
        "model_output",
        "provenance",
    ):
        assert forbidden not in external_text


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
    external_text = json.dumps(build_final_judge_input(item, _rubric()), ensure_ascii=False)
    for secret in (
        "buyer@example.com",
        "13800138000",
        "confirm-secret",
        "token-secret",
        "evidence_snapshots",
        "provenance",
    ):
        assert secret not in external_text
    assert "sku-full-001" not in external_text
    assert "sku-full-002" not in external_text


def _rubric() -> dict:
    return validate_rubric(
        {
            "p0": [
                {
                    "id": "P0-1",
                    "criterion": "不得推荐超过预算1.5倍的商品",
                    "evidence_sources": ["query", "final_cards"],
                }
            ],
            "p1": [
                {
                    "id": "P1-1",
                    "criterion": (
                        "最终结果必须存在，verification_status 和 evidence_judge_status "
                        "为 supported，Fact Guard 为 true"
                    ),
                    "evidence_sources": [
                        "execution.final_result_present",
                        "online_gate.verification_status",
                    ],
                }
            ],
            "p2": [
                {
                    "id": f"P2-{index}",
                    "dimension": dimension,
                    "requirements": [f"{dimension}相关要求"],
                    "expectation": f"回答应体现{dimension}。",
                    "evidence_sources": ["query", "final_text", "final_cards"],
                }
                for index, dimension in enumerate(
                    ("需求覆盖度", "场景洞察力", "决策建议价值"), 1
                )
            ],
        }
    )


def _judge(*, p0_triggered: bool = False, p1_violated: bool = False) -> dict:
    return validate_judge_result(
        {
            "p0_results": [
                {"rubric_id": "P0-1", "triggered": p0_triggered, "reason": "证据判断"}
            ],
            "p1_results": [
                {"rubric_id": "P1-1", "violated": p1_violated, "reason": "证据判断"}
            ],
            "p2_results": [
                {"rubric_id": f"P2-{index}", "score": score, "reason": "证据判断"}
                for index, score in enumerate((5, 5, 5), 1)
            ],
            "reason": "完成",
        },
        _rubric(),
    )


def test_new_score_is_normalized_to_one_and_uses_documented_weights() -> None:
    result = aggregate_scores(_rubric(), _judge())
    assert result["p0_score"] == 1.0
    assert result["p1_score"] == 1.0
    assert result["p2_score"] == 1.0
    assert result["final_score"] == 1.0
    assert result["pass_threshold"] == 0.85
    assert result["quality_pass"] is True


def test_p0_trigger_is_fail_even_when_numeric_score_is_high() -> None:
    result = aggregate_scores(_rubric(), _judge(p0_triggered=True))
    assert result["final_score"] == 0.7
    assert result["quality_pass"] is False


def test_p1_one_violation_deducts_0_2_within_p1_score() -> None:
    result = aggregate_scores(_rubric(), _judge(p1_violated=True))
    assert result["p1_score"] == 0.8
    assert result["final_score"] == 0.94


def test_p2_one_to_five_scores_are_normalized() -> None:
    rubric = _rubric()
    judged = validate_judge_result(
        {
            "p0_results": [{"rubric_id": "P0-1", "triggered": False, "reason": "无"}],
            "p1_results": [{"rubric_id": "P1-1", "violated": False, "reason": "无"}],
            "p2_results": [
                {"rubric_id": f"P2-{index}", "score": score, "reason": "部分满足"}
                for index, score in enumerate((1, 3, 5), 1)
            ],
        },
        rubric,
    )
    result = aggregate_scores(rubric, judged)
    assert result["p2_score"] == 0.6
    assert result["final_score"] == 0.84
    assert result["quality_pass"] is False


def test_rubric_accepts_p2_in_any_order_and_normalizes_output() -> None:
    rubric = _rubric()
    rubric["p2"] = list(reversed(rubric["p2"]))
    normalized = validate_rubric(rubric)
    assert [entry["dimension"] for entry in normalized["p2"]] == [
        "需求覆盖度",
        "场景洞察力",
        "决策建议价值",
    ]


def test_rubric_rejects_duplicate_or_missing_p2_dimension() -> None:
    duplicate = _rubric()
    duplicate["p2"][1]["dimension"] = duplicate["p2"][0]["dimension"]
    try:
        validate_rubric(duplicate)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate P2 dimension was accepted")

    missing = _rubric()
    missing["p2"] = missing["p2"][:2]
    try:
        validate_rubric(missing)
    except ValueError as error:
        assert "exactly once" in str(error)
    else:
        raise AssertionError("missing P2 dimension was accepted")


def test_rubric_rejects_unobservable_and_cross_level_mixing() -> None:
    rubric = _rubric()
    rubric["p1"][0]["criterion"] = "用户满意度必须很高"
    try:
        validate_rubric(rubric)
    except ValueError as error:
        assert "unobservable" in str(error)
    else:
        raise AssertionError("unobservable rubric was accepted")

    duplicate = _rubric()
    duplicate["p1"][0]["evidence_sources"] = ["query"]
    try:
        validate_rubric(duplicate)
    except ValueError as error:
        assert "P1 may use only execution" in str(error)
    else:
        raise AssertionError("cross-level duplicate rubric was accepted")


def test_old_report_shape_remains_readable_without_becoming_new_score() -> None:
    item = {
        "query": "旧报告",
        "final_result": {
            "text": "推荐",
            "recommended_cards": [],
            "verification_status": "supported",
        },
        "events": [],
        "final_judge": {"status": "completed", "score": 0.65},
    }
    assert _flow_summary(item)["flow_valid"] is False
    assert _flow_summary(item)["final"]["text"] == "推荐"
