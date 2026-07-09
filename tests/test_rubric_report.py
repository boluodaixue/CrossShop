"""Tests for Rubric quality scoring, coverage, and FAIL/ERROR separation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.eval.rubric_contract import (
    BinaryCriterionJudgement,
    EvaluationPolicy,
    P2CriterionSpec,
    QualityWeightPolicy,
    QualityCriterionJudgement,
    RubricCaseSpec,
    RubricJudgement,
    RubricScorecard,
    RubricSpec,
    ScenarioFamilyPolicy,
)
from scripts.eval.rubric_cases import load_case_suite
from scripts.eval.rubric_report import (
    CaseEvaluationResult,
    EvaluationFailure,
    build_evaluation_summary,
    completed_result,
    error_result,
    render_markdown_report,
    summarize_coverage,
    summarize_quality,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _policy() -> EvaluationPolicy:
    return EvaluationPolicy(
        quality_policy_id="test-quality-v1",
        coverage_policy_id="test-coverage-v1",
        quality_weights=QualityWeightPolicy(p0=25, p1=35, p2=40),
        scenario_families={
            "single-turn": ScenarioFamilyPolicy(
                label="单轮推荐",
                weight=100,
                required_points={
                    "quality": "回答质量",
                    "error": "错误路径",
                },
            ),
        },
    )


def _case(case_id: str, point: str) -> RubricCaseSpec:
    return RubricCaseSpec(
        id=case_id,
        description=case_id,
        queries=["测试"],
        scenario_family="single-turn",
        coverage_points=[point],
        rubric=RubricSpec(
            p0=["不编造"],
            p1=["工具正确"],
            p2=[
                P2CriterionSpec(
                    criterion="决策价值",
                    score_1_anchor="差",
                    score_5_anchor="好",
                ),
            ],
        ),
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


def test_report_has_weighted_quality_gate_and_separate_coverage() -> None:
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

    cases = [_case("quality", "quality"), _case("judge-down", "error")]
    report = render_markdown_report(
        [scored, error],
        design_cases=cases,
        run_cases=cases,
        policy=_policy(),
        generated_at=datetime(2026, 8, 29, 12, 0),
    )

    assert "SCORED 1，FAIL 0，ERROR 1" in report
    assert "扣 2" in report
    assert "3.000/5" in report
    assert "存在 ERROR，批次不完整" in report
    assert "P0 门禁：**无法判定（批次不完整）**" in report
    assert "场景覆盖" in report
    assert "test-quality-v1" in report
    assert "不等同于 PASS" in report


def test_approved_quality_formula_matches_final_calibration_fixture() -> None:
    results = []
    for index in range(13):
        p0_total = 2 if index == 0 else 1
        p1_total = 1 if index < 2 else 2
        p2_score = 4 if index == 0 else 5
        results.append(
            SimpleNamespace(
                outcome="SCORED",
                scorecard=SimpleNamespace(
                    p0_passed=p0_total,
                    p0_total=p0_total,
                    p1_failed=0,
                    p1_total=p1_total,
                    p2_scores={"quality": p2_score},
                ),
            ),
        )

    summary = summarize_quality(results, policy=_policy())  # type: ignore[arg-type]

    assert (summary.p0_passed, summary.p0_total) == (14, 14)
    assert (summary.p1_earned_points, summary.p1_max_points) == (48, 48)
    assert (summary.p2_earned_points, summary.p2_max_points) == (64, 65)
    assert summary.p2_percentage == 98.462
    assert summary.weighted_quality_score == 99.385
    assert summary.p0_gate_passed is True
    assert summary.p0_gate_status == "PASS"
    assert summary.batch_complete is True


def test_p0_failure_blocks_gate_even_when_numeric_score_exists() -> None:
    result = SimpleNamespace(
        outcome="FAIL",
        scorecard=SimpleNamespace(
            p0_passed=0,
            p0_total=1,
            p1_failed=0,
            p1_total=1,
            p2_scores={"quality": 5},
        ),
    )

    summary = summarize_quality([result], policy=_policy())  # type: ignore[arg-type]

    assert summary.weighted_quality_score == 75.0
    assert summary.p0_gate_passed is False
    assert summary.p0_gate_status == "FAIL"


def test_error_makes_quality_incomplete_and_does_not_count_as_executed() -> None:
    cases = [_case("quality", "quality"), _case("judge-down", "error")]
    scored = completed_result(
        case_id="quality",
        description="质量",
        judgement=_judgement(p0_pass=True),
        scorecard=_scorecard(redline=False),
    )
    error = error_result(
        case_id="judge-down",
        description="错误",
        failure=EvaluationFailure(
            phase="judge_transport",
            error_type="TimeoutError",
            message="timeout",
        ),
    )

    summary = build_evaluation_summary(
        [scored, error],
        design_cases=cases,
        run_cases=cases,
        policy=_policy(),
    )

    assert summary.quality.weighted_quality_score is None
    assert summary.quality.p0_gate_passed is None
    assert summary.quality.p0_gate_status == "INCOMPLETE"
    assert summary.quality.batch_complete is False
    assert summary.schema_version == "rubric-summary-v1"
    assert len(summary.evaluation_policy_sha256) == 64
    assert summary.quality_weights.model_dump() == {
        "p0": 25.0,
        "p1": 35.0,
        "p2": 40.0,
    }
    assert summary.coverage.designed_score == 100.0
    assert summary.coverage.executed_score == 50.0


def test_observed_p0_failure_takes_precedence_over_another_case_error() -> None:
    cases = [_case("redline", "quality"), _case("judge-down", "error")]
    failed = completed_result(
        case_id="redline",
        description="红线",
        judgement=_judgement(p0_pass=False),
        scorecard=_scorecard(redline=True),
    )
    error = error_result(
        case_id="judge-down",
        description="错误",
        failure=EvaluationFailure(
            phase="judge_transport",
            error_type="TimeoutError",
            message="timeout",
        ),
    )

    summary = build_evaluation_summary(
        [failed, error],
        design_cases=cases,
        run_cases=cases,
        policy=_policy(),
    )

    assert summary.quality.weighted_quality_score is None
    assert summary.quality.batch_complete is False
    assert summary.quality.p0_gate_passed is False
    assert summary.quality.p0_gate_status == "FAIL"


def test_summary_rejects_missing_or_duplicate_case_results() -> None:
    cases = [_case("quality", "quality"), _case("judge-down", "error")]
    scored = completed_result(
        case_id="quality",
        description="质量",
        judgement=_judgement(p0_pass=True),
        scorecard=_scorecard(redline=False),
    )

    with pytest.raises(ValueError, match="run_cases/results mismatch"):
        build_evaluation_summary(
            [scored],
            design_cases=cases,
            run_cases=cases,
            policy=_policy(),
        )
    with pytest.raises(ValueError, match="duplicate results ids"):
        build_evaluation_summary(
            [scored, scored],
            design_cases=cases[:1],
            run_cases=cases[:1],
            policy=_policy(),
        )


def test_report_escapes_judge_markdown_structure() -> None:
    judgement = _judgement(p0_pass=True)
    judgement.p2[0].reason = "正常说明\n# 伪造标题 | 表格"
    scored = completed_result(
        case_id="quality",
        description="质量 | 注入",
        judgement=judgement,
        scorecard=_scorecard(redline=False),
    )
    case = _case("quality", "quality")

    report = render_markdown_report(
        [scored],
        design_cases=[case],
        run_cases=[case],
        policy=_policy(),
        generated_at=datetime(2026, 8, 29, 12, 0),
    )

    assert "\n# 伪造标题" not in report
    assert "\\# 伪造标题 \\| 表格" in report
    assert "质量 \\| 注入" in report


def test_coverage_counts_unique_points_and_v3_suite_has_expected_gap() -> None:
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml")
    coverage = summarize_coverage(
        suite.cases,
        suite.cases,
        [],
        policy=suite.evaluation_policy,
    )

    assert coverage.designed_score == 85.143
    assert coverage.executed_score == 0.0
    short_multi = next(
        item
        for item in coverage.families
        if item.family_id == "short-multi-turn-recommendation"
    )
    assert short_multi.coverage_percentage == 100.0
    assert short_multi.missing_points == []
    missing = {
        (family.family_id, point)
        for family in coverage.families
        for point in family.missing_points
    }
    assert missing == {
        ("trade-order", "ownership-idempotency-failure"),
        ("exception-degradation", "opensearch-unavailable"),
        ("exception-degradation", "repository-inconsistency"),
        ("exception-degradation", "model-or-embedding-timeout"),
        ("exception-degradation", "reranker-or-platform-fallback"),
    }


def test_duplicate_cases_do_not_increase_coverage_for_the_same_point() -> None:
    first = _case("first", "quality")
    duplicate = _case("duplicate", "quality")
    coverage = summarize_coverage(
        [first, duplicate],
        [first, duplicate],
        [],
        policy=_policy(),
    )

    assert coverage.designed_score == 50.0
