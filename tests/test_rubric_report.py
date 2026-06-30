"""Tests for Rubric FAIL/ERROR separation and unweighted reports."""

from __future__ import annotations

from datetime import datetime

import pytest

from scripts.eval.rubric_contract import (
    BinaryCriterionJudgement,
    QualityCriterionJudgement,
    RubricJudgement,
    RubricScorecard,
)
from scripts.eval.rubric_report import (
    CaseEvaluationResult,
    EvaluationFailure,
    completed_result,
    error_result,
    render_markdown_report,
)


def _judgement(*, p0_pass: bool) -> RubricJudgement:
    return RubricJudgement(
        p0=[
            BinaryCriterionJudgement(
                criterion="不编造",
                reason="有证据",
                **{"pass": p0_pass},
                evidence_refs=["turn:1:final"],
            ),
        ],
        p1=[
            BinaryCriterionJudgement(
                criterion="工具正确",
                reason="调用顺序错误",
                **{"pass": False},
                evidence_refs=["turn:1:tool:1"],
            ),
        ],
        p2=[
            QualityCriterionJudgement(
                criterion="决策价值",
                reason="基本清楚",
                score=3,
                evidence_refs=["turn:1:final"],
            ),
        ],
    )


def _scorecard(*, redline: bool) -> RubricScorecard:
    return RubricScorecard(
        verdict="redline_fail" if redline else "scored",
        p0_passed=0 if redline else 1,
        p0_total=1,
        p1_failed=1,
        p1_total=1,
        p1_deduction_points=2,
        p2_scores={"决策价值": 3},
        p2_average=3,
    )


def test_p0_redline_is_fail_but_completed_non_redline_is_only_scored() -> None:
    failed = completed_result(
        case_id="redline",
        description="红线",
        judgement=_judgement(p0_pass=False),
        scorecard=_scorecard(redline=True),
    )
    scored = completed_result(
        case_id="quality",
        description="质量",
        judgement=_judgement(p0_pass=True),
        scorecard=_scorecard(redline=False),
    )

    assert failed.outcome == "FAIL"
    assert scored.outcome == "SCORED"


def test_infrastructure_failure_is_error_and_cannot_carry_fake_score() -> None:
    result = error_result(
        case_id="judge-down",
        description="Judge 不可用",
        failure=EvaluationFailure(
            phase="judge_transport",
            error_type="TimeoutError",
            message="Judge request timed out",
        ),
    )

    assert result.outcome == "ERROR"
    with pytest.raises(ValueError, match="ERROR requires failure"):
        CaseEvaluationResult(
            case_id="bad",
            description="错误合同",
            outcome="ERROR",
            judgement=_judgement(p0_pass=False),
            scorecard=_scorecard(redline=True),
            failure=result.failure,
        )


def test_report_has_separate_levels_and_no_weighted_average_or_pass() -> None:
    scored = completed_result(
        case_id="quality",
        description="质量",
        judgement=_judgement(p0_pass=True),
        scorecard=_scorecard(redline=False),
    )
    error = error_result(
        case_id="judge-down",
        description="Judge 不可用",
        failure=EvaluationFailure(
            phase="judge_transport",
            error_type="TimeoutError",
            message="Judge request timed out",
        ),
    )

    report = render_markdown_report(
        [scored, error],
        generated_at=datetime(2026, 8, 29, 12, 0),
    )

    assert "SCORED 1，FAIL 0，ERROR 1" in report
    assert "扣 2" in report
    assert "3.000/5" in report
    assert "加权" not in report
    assert "平均分 0." not in report
    assert "不等同于 PASS" in report
