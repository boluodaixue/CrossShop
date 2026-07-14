"""Render separate P0/P1/P2 results without an uncalibrated total score."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.eval.rubric_contract import (
    EvaluationPolicy,
    QualityWeightPolicy,
    RubricCaseSpec,
    RubricJudgement,
    RubricScorecard,
)

_P1_POINTS_PER_CRITERION = 2


class EvaluationFailure(BaseModel):
    """Sanitized infrastructure/protocol failure, never a product-quality fail."""

    phase: Literal[
        "case_contract",
        "dependency",
        "artifact_validation",
        "intent_execution",
        "evidence_collection",
        "ground_truth",
        "judge_transport",
        "judge_protocol",
    ]
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class CaseEvaluationResult(BaseModel):
    """One case outcome with deliberately conservative verdict semantics.

    SCORED means the protocol completed and P0 did not fail.  It is not called
    PASS until a human calibration run freezes P1/P2 release thresholds.
    """

    case_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    outcome: Literal["SCORED", "FAIL", "ERROR"]
    judgement: RubricJudgement | None = None
    scorecard: RubricScorecard | None = None
    failure: EvaluationFailure | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> CaseEvaluationResult:
        if self.outcome == "ERROR":
            if (
                self.failure is None
                or self.judgement is not None
                or self.scorecard is not None
            ):
                raise ValueError("ERROR requires failure and no judgement/scorecard")
            return self
        if self.failure is not None or self.judgement is None or self.scorecard is None:
            raise ValueError("SCORED/FAIL require judgement and scorecard only")
        expected = "FAIL" if self.scorecard.verdict == "redline_fail" else "SCORED"
        if self.outcome != expected:
            raise ValueError(
                f"outcome {self.outcome} contradicts scorecard verdict "
                f"{self.scorecard.verdict}",
            )
        return self


class QualityScoreSummary(BaseModel):
    """Aggregate quality score plus the non-compensating P0 release gate."""

    model_config = ConfigDict(extra="forbid")

    p0_passed: int = Field(ge=0)
    p0_total: int = Field(ge=0)
    p0_percentage: float = Field(ge=0, le=100)
    p0_gate_passed: bool | None
    p0_gate_status: Literal["PASS", "FAIL", "INCOMPLETE"]
    batch_complete: bool
    p1_earned_points: int = Field(ge=0)
    p1_max_points: int = Field(ge=0)
    p1_percentage: float = Field(ge=0, le=100)
    p2_earned_points: int = Field(ge=0)
    p2_max_points: int = Field(ge=0)
    p2_percentage: float = Field(ge=0, le=100)
    weighted_quality_score: float | None = Field(default=None, ge=0, le=100)
    error_count: int = Field(ge=0)


class ScenarioFamilyCoverage(BaseModel):
    """Authored and actually executed coverage for one user-journey family."""

    model_config = ConfigDict(extra="forbid")

    family_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    weight: float = Field(gt=0, le=100)
    covered_points: list[str]
    executed_points: list[str]
    required_points: list[str] = Field(min_length=1)
    missing_points: list[str]
    coverage_percentage: float = Field(ge=0, le=100)
    executed_coverage_percentage: float = Field(ge=0, le=100)


class ScenarioCoverageSummary(BaseModel):
    """Weighted completeness of the scenario catalog, separate from quality."""

    model_config = ConfigDict(extra="forbid")

    designed_score: float = Field(ge=0, le=100)
    executed_score: float = Field(ge=0, le=100)
    families: list[ScenarioFamilyCoverage] = Field(min_length=1)


class EvaluationSummary(BaseModel):
    """Machine-readable batch summary persisted beside the detailed evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["rubric-summary-v1"] = "rubric-summary-v1"
    quality_policy_id: str = Field(min_length=1)
    coverage_policy_id: str = Field(min_length=1)
    evaluation_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_weights: QualityWeightPolicy
    quality: QualityScoreSummary
    coverage: ScenarioCoverageSummary


def _duplicates(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def _validate_batch_identity(
    design_cases: list[RubricCaseSpec],
    run_cases: list[RubricCaseSpec],
    results: list[CaseEvaluationResult],
) -> None:
    design_ids = [case.id for case in design_cases]
    run_ids = [case.id for case in run_cases]
    result_ids = [result.case_id for result in results]
    for label, values in (
        ("design_cases", design_ids),
        ("run_cases", run_ids),
        ("results", result_ids),
    ):
        duplicates = _duplicates(values)
        if duplicates:
            raise ValueError(f"duplicate {label} ids: {', '.join(duplicates)}")
    unknown_run_ids = sorted(set(run_ids) - set(design_ids))
    if unknown_run_ids:
        raise ValueError(
            "run_cases are missing from design_cases: " + ", ".join(unknown_run_ids),
        )
    if set(run_ids) != set(result_ids):
        missing = sorted(set(run_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(run_ids))
        raise ValueError(
            f"run_cases/results mismatch; missing={missing}, extra={extra}",
        )


def completed_result(
    *,
    case_id: str,
    description: str,
    judgement: RubricJudgement,
    scorecard: RubricScorecard,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case_id=case_id,
        description=description,
        outcome=("FAIL" if scorecard.verdict == "redline_fail" else "SCORED"),
        judgement=judgement,
        scorecard=scorecard,
    )


def error_result(
    *,
    case_id: str,
    description: str,
    failure: EvaluationFailure,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case_id=case_id,
        description=description,
        outcome="ERROR",
        failure=failure,
    )


def _p0_summary(result: CaseEvaluationResult) -> str:
    scorecard = result.scorecard
    if scorecard is None:
        return "—"
    return f"{scorecard.p0_passed}/{scorecard.p0_total}"


def _p1_summary(result: CaseEvaluationResult) -> str:
    scorecard = result.scorecard
    if scorecard is None:
        return "—"
    return (
        f"{scorecard.p1_failed}/{scorecard.p1_total} 失败，"
        f"扣 {scorecard.p1_deduction_points}"
    )


def _p2_summary(result: CaseEvaluationResult) -> str:
    scorecard = result.scorecard
    if scorecard is None or scorecard.p2_average is None:
        return "—"
    return f"{scorecard.p2_average:.3f}/5"


def _percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator * 100.0, 3)


def summarize_quality(
    results: list[CaseEvaluationResult],
    *,
    policy: EvaluationPolicy,
) -> QualityScoreSummary:
    """Aggregate criteria without averaging easy and hard cases equally."""

    scorecards = [result.scorecard for result in results if result.scorecard]
    p0_passed = sum(item.p0_passed for item in scorecards)
    p0_total = sum(item.p0_total for item in scorecards)
    p1_failed = sum(item.p1_failed for item in scorecards)
    p1_total = sum(item.p1_total for item in scorecards)
    p1_max_points = p1_total * _P1_POINTS_PER_CRITERION
    p1_earned_points = p1_max_points - (p1_failed * _P1_POINTS_PER_CRITERION)
    p2_scores = [
        score for scorecard in scorecards for score in scorecard.p2_scores.values()
    ]
    p2_earned_points = sum(p2_scores)
    p2_max_points = len(p2_scores) * 5
    p0_ratio = p0_passed / p0_total if p0_total else 0.0
    p1_ratio = p1_earned_points / p1_max_points if p1_max_points else 0.0
    p2_ratio = p2_earned_points / p2_max_points if p2_max_points else 0.0
    p0_percentage = round(p0_ratio * 100.0, 3)
    p1_percentage = round(p1_ratio * 100.0, 3)
    p2_percentage = round(p2_ratio * 100.0, 3)
    error_count = sum(result.outcome == "ERROR" for result in results)
    has_all_sections = p0_total > 0 and p1_max_points > 0 and p2_max_points > 0
    weighted_quality_score = None
    if error_count == 0 and has_all_sections:
        weights = policy.quality_weights
        weighted_quality_score = round(
            p0_ratio * weights.p0 + p1_ratio * weights.p1 + p2_ratio * weights.p2,
            3,
        )
    p0_gate_passed: bool | None
    p0_gate_status: Literal["PASS", "FAIL", "INCOMPLETE"]
    if p0_total > 0 and p0_passed < p0_total:
        p0_gate_passed = False
        p0_gate_status = "FAIL"
    elif error_count:
        p0_gate_passed = None
        p0_gate_status = "INCOMPLETE"
    else:
        p0_gate_passed = p0_total > 0 and p0_passed == p0_total
        p0_gate_status = "PASS" if p0_gate_passed else "FAIL"
    return QualityScoreSummary(
        p0_passed=p0_passed,
        p0_total=p0_total,
        p0_percentage=p0_percentage,
        p0_gate_passed=p0_gate_passed,
        p0_gate_status=p0_gate_status,
        batch_complete=(error_count == 0),
        p1_earned_points=p1_earned_points,
        p1_max_points=p1_max_points,
        p1_percentage=p1_percentage,
        p2_earned_points=p2_earned_points,
        p2_max_points=p2_max_points,
        p2_percentage=p2_percentage,
        weighted_quality_score=weighted_quality_score,
        error_count=error_count,
    )


def summarize_coverage(
    design_cases: list[RubricCaseSpec],
    run_cases: list[RubricCaseSpec],
    results: list[CaseEvaluationResult],
    *,
    policy: EvaluationPolicy,
) -> ScenarioCoverageSummary:
    """Score unique required journey points, so duplicate easy cases cannot game it."""

    executed_case_ids = {
        result.case_id for result in results if result.outcome != "ERROR"
    }
    families: list[ScenarioFamilyCoverage] = []
    designed_score = 0.0
    executed_score = 0.0
    for family_id, family_policy in policy.scenario_families.items():
        family_cases = [
            case for case in design_cases if case.scenario_family == family_id
        ]
        run_family_cases = [
            case for case in run_cases if case.scenario_family == family_id
        ]
        covered = sorted(
            {point for case in family_cases for point in case.coverage_points},
        )
        executed = sorted(
            {
                point
                for case in run_family_cases
                if case.id in executed_case_ids
                for point in case.coverage_points
            },
        )
        required = list(family_policy.required_points)
        coverage_percentage = _percentage(len(covered), len(required))
        executed_percentage = _percentage(len(executed), len(required))
        designed_score += family_policy.weight * len(covered) / len(required)
        executed_score += family_policy.weight * len(executed) / len(required)
        families.append(
            ScenarioFamilyCoverage(
                family_id=family_id,
                label=family_policy.label,
                weight=family_policy.weight,
                covered_points=covered,
                executed_points=executed,
                required_points=required,
                missing_points=sorted(set(required) - set(covered)),
                coverage_percentage=coverage_percentage,
                executed_coverage_percentage=executed_percentage,
            ),
        )
    return ScenarioCoverageSummary(
        designed_score=round(designed_score, 3),
        executed_score=round(executed_score, 3),
        families=families,
    )


def build_evaluation_summary(
    results: list[CaseEvaluationResult],
    *,
    design_cases: list[RubricCaseSpec],
    run_cases: list[RubricCaseSpec],
    policy: EvaluationPolicy,
) -> EvaluationSummary:
    _validate_batch_identity(design_cases, run_cases, results)
    policy_json = json.dumps(
        policy.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return EvaluationSummary(
        quality_policy_id=policy.quality_policy_id,
        coverage_policy_id=policy.coverage_policy_id,
        evaluation_policy_sha256=hashlib.sha256(policy_json).hexdigest(),
        quality_weights=policy.quality_weights,
        quality=summarize_quality(results, policy=policy),
        coverage=summarize_coverage(
            design_cases,
            run_cases,
            results,
            policy=policy,
        ),
    )


def _case_quality_score(
    result: CaseEvaluationResult,
    *,
    policy: EvaluationPolicy,
) -> str:
    scorecard = result.scorecard
    if scorecard is None:
        return "—"
    p0_percentage = _percentage(scorecard.p0_passed, scorecard.p0_total)
    p1_max = scorecard.p1_total * _P1_POINTS_PER_CRITERION
    p1_earned = p1_max - (scorecard.p1_failed * _P1_POINTS_PER_CRITERION)
    p1_percentage = _percentage(p1_earned, p1_max)
    p2_percentage = (
        round(scorecard.p2_average / 5.0 * 100.0, 3)
        if scorecard.p2_average is not None
        else 0.0
    )
    weights = policy.quality_weights
    total = (
        p0_percentage * weights.p0
        + p1_percentage * weights.p1
        + p2_percentage * weights.p2
    ) / 100.0
    return f"{total:.3f}/100"


def _markdown_text(value: str) -> str:
    """Keep Judge-authored text inline and unable to create report structure."""

    text = " ".join(value.replace("\r", "\n").splitlines())
    text = text.replace("\\", "\\\\")
    for char in ("`", "*", "_", "[", "]", "<", ">", "#", "|"):
        text = text.replace(char, f"\\{char}")
    return text


def render_markdown_report(
    results: list[CaseEvaluationResult],
    *,
    design_cases: list[RubricCaseSpec],
    run_cases: list[RubricCaseSpec],
    policy: EvaluationPolicy,
    generated_at: datetime,
) -> str:
    """Render quality and coverage separately; P0 remains a hard gate."""

    counts = {
        outcome: sum(result.outcome == outcome for result in results)
        for outcome in ("SCORED", "FAIL", "ERROR")
    }
    summary = build_evaluation_summary(
        results,
        design_cases=design_cases,
        run_cases=run_cases,
        policy=policy,
    )
    quality = summary.quality
    coverage = summary.coverage
    quality_score = (
        f"{quality.weighted_quality_score:.3f}/100"
        if quality.weighted_quality_score is not None
        else "—（存在 ERROR，批次不完整）"
    )
    p0_gate_label = {
        "PASS": "通过",
        "FAIL": "未通过",
        "INCOMPLETE": "无法判定（批次不完整）",
    }[quality.p0_gate_status]
    weights = policy.quality_weights
    lines = [
        f"# CrossShop Rubric 评测报告（{generated_at:%Y-%m-%d %H:%M}）",
        "",
        (
            f"评分协议：`{policy.quality_policy_id}`；"
            f"覆盖协议：`{policy.coverage_policy_id}`。"
        ),
        "",
        (
            f"总览：SCORED {counts['SCORED']}，FAIL {counts['FAIL']}，"
            f"ERROR {counts['ERROR']}，共 {len(results)} 个用例。"
        ),
        "",
        (
            f"质量总分：**{quality_score}**（P0/P1/P2="
            f"{weights.p0:g}/{weights.p1:g}/{weights.p2:g}）；"
            f"P0 门禁：**{p0_gate_label}**。"
        ),
        "",
        (
            f"场景覆盖：设计 **{coverage.designed_score:.3f}/100**，"
            f"本次实际执行 **{coverage.executed_score:.3f}/100**。"
        ),
        "",
        "> 质量总分只评价已经纳入题库的场景；场景覆盖分评价题库完整度。SCORED 表示评测协议完整执行且没有触发 P0 红线，在发布阈值完成人工校准前不等同于 PASS。",
        "",
        "## 质量评分",
        "",
        "| case | 描述 | P0 | P1 | P2 平均 | 单题质量分 | 结果 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.case_id} | {_markdown_text(result.description)} | "
            f"{_p0_summary(result)} | {_p1_summary(result)} | "
            f"{_p2_summary(result)} | "
            f"{_case_quality_score(result, policy=policy)} | {result.outcome} |",
        )

    lines.extend(
        [
            "",
            "## 场景覆盖",
            "",
            "| 场景族 | 权重 | 已设计 | 已执行 | 必要场景点 | 尚缺 |",
            "|---|---:|---:|---:|---:|---:|",
        ],
    )
    for family in coverage.families:
        lines.append(
            f"| {family.label} | {family.weight:g}% | "
            f"{family.coverage_percentage:.3f}% | "
            f"{family.executed_coverage_percentage:.3f}% | "
            f"{len(family.required_points)} | {len(family.missing_points)} |",
        )

    lines.extend(["", "### 尚缺的必要场景点", ""])
    family_policies = policy.scenario_families
    for family in coverage.families:
        if not family.missing_points:
            continue
        labels = family_policies[family.family_id].required_points
        missing = "；".join(
            f"`{point}`（{labels[point]}）" for point in family.missing_points
        )
        lines.append(f"- **{family.label}**：{missing}")

    for result in results:
        lines.extend(["", f"## {result.case_id}（{result.outcome}）", ""])
        if result.failure is not None:
            lines.append(
                f"- 评测错误阶段：`{result.failure.phase}`；"
                f"类型：`{result.failure.error_type}`；{result.failure.message}",
            )
            continue
        assert result.judgement is not None
        for level, items in (
            ("P0", result.judgement.p0),
            ("P1", result.judgement.p1),
            ("P2", result.judgement.p2),
        ):
            lines.append(f"### {level}")
            lines.append("")
            for item in items:
                if level == "P2":
                    mark = f"{item.score}/5"
                else:
                    mark = "PASS" if item.passed else "FAIL"
                refs = ", ".join(f"`{ref}`" for ref in item.evidence_refs)
                lines.append(
                    f"- [{mark}] {_markdown_text(item.criterion)}："
                    f"{_markdown_text(item.reason)}（证据：{refs}）",
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
