"""Offline tests for the Rubric v3 contract and criterion scorecard."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.eval.rubric_contract import (
    BinaryCriterionJudgement,
    DisplayedProductEvidence,
    EvaluationEvidence,
    P2CriterionSpec,
    PreferenceStateEvidence,
    QualityWeightPolicy,
    QualityCriterionJudgement,
    RubricCaseSuite,
    RubricJudgement,
    RubricProtocolError,
    RubricSpec,
    StructuredStateEvidence,
    ToolCallEvidence,
    TurnEvidence,
    score_rubric,
)
from scripts.eval.rubric_cases import load_case_suite

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _state() -> StructuredStateEvidence:
    empty = PreferenceStateEvidence(count=0, content_hash="0" * 64)
    return StructuredStateEvidence(
        preference_before=empty,
        preference_after=empty,
    )


def _spec() -> RubricSpec:
    return RubricSpec(
        p0=["不得编造价格", "不得违反预算"],
        p1=["商品检索工具最多调用两次", "最终给出明确结论"],
        p2=[
            P2CriterionSpec(
                criterion="需求覆盖度",
                score_1_anchor="只复述需求",
                score_5_anchor="覆盖全部显式需求并解释取舍",
            ),
            P2CriterionSpec(
                criterion="决策建议价值",
                score_1_anchor="没有可执行建议",
                score_5_anchor="给出清晰对比和下一步建议",
            ),
        ],
    )


def _binary(criterion: str, passed: bool) -> BinaryCriterionJudgement:
    return BinaryCriterionJudgement.model_validate(
        {"criterion": criterion, "reason": "依据 evidence-1 判定", "pass": passed},
    )


def test_scorecard_keeps_three_sections_separate() -> None:
    judgement = RubricJudgement(
        p0=[_binary("不得编造价格", True), _binary("不得违反预算", True)],
        p1=[
            _binary("商品检索工具最多调用两次", False),
            _binary("最终给出明确结论", True),
        ],
        p2=[
            QualityCriterionJudgement(
                criterion="需求覆盖度",
                reason="覆盖大部分需求",
                score=4,
            ),
            QualityCriterionJudgement(
                criterion="决策建议价值",
                reason="建议可执行但缺少对比",
                score=3,
            ),
        ],
    )

    scorecard = score_rubric(_spec(), judgement)

    assert scorecard.verdict == "scored"
    assert (scorecard.p0_passed, scorecard.p0_total) == (2, 2)
    assert (scorecard.p1_failed, scorecard.p1_total) == (1, 2)
    assert scorecard.p1_deduction_points == 2
    assert scorecard.p2_scores == {"需求覆盖度": 4, "决策建议价值": 3}
    assert scorecard.p2_average == 3.5
    assert "weighted" not in scorecard.model_dump()


def test_any_p0_failure_is_a_redline_failure() -> None:
    judgement = RubricJudgement(
        p0=[_binary("不得编造价格", False), _binary("不得违反预算", True)],
        p1=[
            _binary("商品检索工具最多调用两次", True),
            _binary("最终给出明确结论", True),
        ],
        p2=[
            QualityCriterionJudgement(
                criterion="需求覆盖度",
                reason="完全覆盖",
                score=5,
            ),
            QualityCriterionJudgement(
                criterion="决策建议价值",
                reason="建议清楚",
                score=5,
            ),
        ],
    )

    scorecard = score_rubric(_spec(), judgement)

    assert scorecard.verdict == "redline_fail"
    assert scorecard.p2_average == 5.0


@pytest.mark.parametrize("score", [0, 6])
def test_p2_score_is_strictly_one_to_five(score: int) -> None:
    with pytest.raises(ValidationError):
        QualityCriterionJudgement(
            criterion="需求覆盖度",
            reason="非法分数",
            score=score,
        )


def test_missing_or_extra_judge_criteria_is_protocol_error() -> None:
    judgement = RubricJudgement(
        p0=[_binary("不得编造价格", True)],
        p1=[
            _binary("商品检索工具最多调用两次", True),
            _binary("最终给出明确结论", True),
        ],
        p2=[
            QualityCriterionJudgement(
                criterion="需求覆盖度",
                reason="覆盖需求",
                score=4,
            ),
            QualityCriterionJudgement(
                criterion="决策建议价值",
                reason="建议清楚",
                score=4,
            ),
        ],
    )

    with pytest.raises(RubricProtocolError, match="missing"):
        score_rubric(_spec(), judgement)


def test_duplicate_rubric_criterion_is_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate criteria"):
        RubricSpec(
            p0=["不得编造", "不得编造"],
            p1=["工具正确"],
            p2=[
                P2CriterionSpec(
                    criterion="回答质量",
                    score_1_anchor="差",
                    score_5_anchor="好",
                ),
            ],
        )


@pytest.mark.parametrize("empty_level", ["p0", "p1", "p2"])
def test_rubric_requires_all_three_non_empty_levels(empty_level: str) -> None:
    values = _spec().model_dump()
    values[empty_level] = []

    with pytest.raises(ValidationError, match="at least 1 item"):
        RubricSpec.model_validate(values)


def test_quality_weights_must_sum_to_one_hundred() -> None:
    with pytest.raises(ValidationError, match="sum to 100"):
        QualityWeightPolicy(p0=30, p1=30, p2=30)


def test_official_quality_policy_id_is_bound_to_approved_weights() -> None:
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml")
    payload = suite.evaluation_policy.model_dump(mode="json")
    payload["quality_weights"] = {"p0": 20, "p1": 40, "p2": 40}

    with pytest.raises(ValidationError, match="requires P0/P1/P2=25/35/40"):
        type(suite.evaluation_policy).model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scenario_family", "missing-family", "unknown scenario family"),
        ("coverage_points", ["missing-point"], "unknown coverage points"),
    ],
)
def test_suite_rejects_unknown_coverage_contract(
    field: str,
    value: object,
    message: str,
) -> None:
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml")
    payload = suite.model_dump(mode="json")
    payload["cases"][0][field] = value

    with pytest.raises(ValidationError, match=message):
        RubricCaseSuite.model_validate(payload)


def test_suite_rejects_non_hundred_family_weights_and_duplicate_points() -> None:
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml")
    bad_weight = suite.model_dump(mode="json")
    bad_weight["evaluation_policy"]["scenario_families"]["chitchat"]["weight"] = 6
    with pytest.raises(ValidationError, match="family weights must sum to 100"):
        RubricCaseSuite.model_validate(bad_weight)

    duplicate = suite.model_dump(mode="json")
    point = duplicate["cases"][0]["coverage_points"][0]
    duplicate["cases"][0]["coverage_points"].append(point)
    with pytest.raises(ValidationError, match="contains duplicates"):
        RubricCaseSuite.model_validate(duplicate)


def test_suite_rejects_unknown_top_level_fields() -> None:
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml")
    payload = suite.model_dump(mode="json")
    payload["unexpected"] = "must fail"

    with pytest.raises(ValidationError, match="unexpected"):
        RubricCaseSuite.model_validate(payload)


def test_evidence_requires_contiguous_turns_and_unique_tool_order() -> None:
    tool = ToolCallEvidence(
        order=1,
        call_id="call-1",
        agent="main",
        tool="product_search_tool",
        arguments={"platform": "reference_seed"},
        result_summary={"hit_count": 5},
    )
    turn = TurnEvidence(
        turn_index=1,
        user_input="只看示例平台露营灯",
        route="main.direct",
        tool_calls=[tool],
        displayed_products=[
            DisplayedProductEvidence(
                rank=1,
                product_id="reference_seed:123",
                platform="reference_seed",
            ),
        ],
        structured_state=_state(),
        final_text="找到一款符合要求的露营灯。",
    )

    evidence = EvaluationEvidence(
        case_id="single-platform",
        session_id_hash="session-hash",
        turns=[turn],
    )

    assert evidence.schema_version == "rubric-evidence-v2"
    assert evidence.turns[0].tool_calls[0].result_summary["hit_count"] == 5

    with pytest.raises(ValidationError, match="contiguous"):
        EvaluationEvidence(
            case_id="bad-turn-order",
            session_id_hash="session-hash",
            turns=[turn.model_copy(update={"turn_index": 2})],
        )

    with pytest.raises(ValidationError, match="tool call order"):
        TurnEvidence(
            turn_index=1,
            user_input="查询商品",
            route="main.direct",
            tool_calls=[tool, tool.model_copy(update={"call_id": "call-2"})],
            structured_state=_state(),
            final_text="完成。",
        )


def test_displayed_products_require_unique_contiguous_identity() -> None:
    common = {
        "turn_index": 1,
        "user_input": "推荐两件商品",
        "route": "main.direct",
        "structured_state": _state(),
        "final_text": "完成。",
    }
    with pytest.raises(ValidationError, match="contiguous"):
        TurnEvidence(
            **common,
            displayed_products=[
                DisplayedProductEvidence(
                    rank=2,
                    product_id="P1",
                    platform="crossshop_reference",
                ),
            ],
        )
    with pytest.raises(ValidationError, match="identities"):
        TurnEvidence(
            **common,
            displayed_products=[
                DisplayedProductEvidence(
                    rank=1,
                    product_id="P1",
                    platform="crossshop_reference",
                ),
                DisplayedProductEvidence(
                    rank=2,
                    product_id="P1",
                    platform="crossshop_reference",
                ),
            ],
        )


def test_judge_contract_rejects_unknown_fields() -> None:
    payload = {
        "criterion": "不得编造价格",
        "reason": "有证据",
        "pass": True,
        "unexpected": "must fail",
    }
    with pytest.raises(ValidationError, match="unexpected"):
        BinaryCriterionJudgement.model_validate(payload)
