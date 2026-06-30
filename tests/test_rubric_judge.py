"""Tests for the evidence-grounded Rubric judge protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval.rubric_cases import load_case_suite
from scripts.eval.rubric_contract import (
    EvaluationEvidence,
    P2CriterionSpec,
    PreferenceStateEvidence,
    RubricProtocolError,
    RubricSpec,
    StructuredStateEvidence,
    ToolCallEvidence,
    TurnEvidence,
)


def _state() -> StructuredStateEvidence:
    empty = PreferenceStateEvidence(count=0, content_hash="0" * 64)
    return StructuredStateEvidence(
        preference_before=empty,
        preference_after=empty,
    )


from scripts.eval.rubric_ground_truth import ProductFactWindow
from scripts.eval.rubric_judge import (
    RubricGroundTruthError,
    build_judge_messages,
    build_judge_request,
    parse_judge_output,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rubric() -> RubricSpec:
    return RubricSpec(
        p0=["不得编造价格"],
        p1=["使用真实工具证据"],
        p2=[
            P2CriterionSpec(
                criterion="决策价值",
                score_1_anchor="没有建议",
                score_5_anchor="建议清晰可执行",
            ),
        ],
    )


def _evidence() -> EvaluationEvidence:
    return EvaluationEvidence(
        case_id="case-one",
        session_id_hash="0123456789abcdef",
        turns=[
            TurnEvidence(
                turn_index=1,
                user_input="推荐露营灯",
                route="main.direct",
                tool_calls=[
                    ToolCallEvidence(
                        order=1,
                        call_id="call-1",
                        agent="main",
                        tool="product_search_tool",
                        result_summary={"hit_count": 1},
                        status="success",
                    ),
                ],
                structured_state=_state(),
                final_text="推荐 LumenGo 露营灯。",
            ),
        ],
    )


def _facts(*, missing: list[str] | None = None) -> ProductFactWindow:
    return ProductFactWindow(
        requested_product_ids=missing or [],
        products=[],
        missing_product_ids=missing or [],
        facts_sha256="0" * 64,
    )


def _request():
    return build_judge_request(
        case_id="case-one",
        description="测试",
        prior_context="",
        rubric=_rubric(),
        evidence=_evidence(),
        product_facts=_facts(),
    )


def _valid_output() -> dict:
    return {
        "p0": [
            {
                "criterion": "不得编造价格",
                "reason": "回答没有价格数字",
                "pass": True,
                "evidence_refs": ["turn:1:final"],
            },
        ],
        "p1": [
            {
                "criterion": "使用真实工具证据",
                "reason": "有真实检索调用",
                "pass": True,
                "evidence_refs": ["turn:1:tool:1"],
            },
        ],
        "p2": [
            {
                "criterion": "决策价值",
                "reason": "有明确推荐",
                "score": 4,
                "evidence_refs": ["turn:1:final"],
            },
        ],
    }


def test_v2_case_file_validates_and_dependencies_are_ordered() -> None:
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v2.yaml")

    assert suite.schema_version == "rubric-cases-v2"
    assert len(suite.cases) == 13
    memory_recall = next(case for case in suite.cases if case.id == "memory-recall")
    assert memory_recall.depends_on == ["memory-write"]
    assert memory_recall.rubric.p2[0].score_1_anchor
    compare = next(case for case in suite.cases if case.id == "compare-two")
    assert "展示币种小计" in compare.rubric.p0[0]
    assert "目录原币标价" in compare.rubric.p0[0]


def test_judge_prompt_contains_refs_schema_and_no_weighted_score_instruction() -> None:
    messages = build_judge_messages(_request())
    user_payload = json.loads(messages[1]["content"])

    assert "turn:1:tool:1" in user_payload["allowed_evidence_refs"]
    assert "output_schema" in user_payload
    assert "不计算总分" in messages[0]["content"]
    assert "顶层 p0、p1、p2 各且仅出现一次" in messages[0]["content"]
    assert "50%" not in messages[0]["content"]
    assert "session_id_hash" not in user_payload["evidence"]
    assert "0123456789abcdef" not in messages[1]["content"]


def test_valid_judge_output_is_scored_with_separate_levels() -> None:
    judgement, scorecard = parse_judge_output(_valid_output(), _request())

    assert judgement.p0[0].passed is True
    assert scorecard.verdict == "scored"
    assert scorecard.p1_deduction_points == 0
    assert scorecard.p2_average == 4


def test_unknown_or_missing_evidence_refs_are_protocol_errors() -> None:
    unknown = _valid_output()
    unknown["p1"][0]["evidence_refs"] = ["turn:9:tool:99"]
    with pytest.raises(RubricProtocolError, match="unknown refs"):
        parse_judge_output(unknown, _request())

    missing = _valid_output()
    missing["p2"][0]["evidence_refs"] = []
    with pytest.raises(RubricProtocolError, match="no evidence refs"):
        parse_judge_output(missing, _request())


def test_invalid_json_and_missing_repository_facts_are_errors_not_failures() -> None:
    with pytest.raises(RubricProtocolError, match="invalid JSON"):
        parse_judge_output("not-json", _request())

    with pytest.raises(RubricGroundTruthError, match="P404"):
        build_judge_request(
            case_id="case-one",
            description="测试",
            prior_context="",
            rubric=_rubric(),
            evidence=_evidence(),
            product_facts=_facts(missing=["P404"]),
        )


def test_duplicate_json_key_is_an_explicit_protocol_error() -> None:
    raw = '{"p0":[],"p1":[],"p1":[],"p2":[]}'

    with pytest.raises(RubricProtocolError, match="duplicate JSON key: p1"):
        parse_judge_output(raw, _request())


def test_observed_product_cannot_be_omitted_from_fact_window() -> None:
    evidence = _evidence()
    evidence.turns[0].tool_calls[0].result_summary["hit_product_ids"] = ["P1008"]

    with pytest.raises(RubricGroundTruthError, match="omitted.*P1008"):
        build_judge_request(
            case_id="case-one",
            description="测试",
            prior_context="",
            rubric=_rubric(),
            evidence=evidence,
            product_facts=_facts(),
        )
