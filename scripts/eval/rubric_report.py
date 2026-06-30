"""Render separate P0/P1/P2 results without an uncalibrated total score."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from scripts.eval.rubric_contract import RubricJudgement, RubricScorecard


class EvaluationFailure(BaseModel):
    """Sanitized infrastructure/protocol failure, never a product-quality fail."""

    phase: Literal[
        "case_contract",
        "dependency",
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


def render_markdown_report(
    results: list[CaseEvaluationResult],
    *,
    generated_at: datetime,
) -> str:
    """Render an auditable report with no weighted score or invented PASS."""

    counts = {
        outcome: sum(result.outcome == outcome for result in results)
        for outcome in ("SCORED", "FAIL", "ERROR")
    }
    lines = [
        f"# Globex Rubric 评测报告（{generated_at:%Y-%m-%d %H:%M}）",
        "",
        (
            f"总览：SCORED {counts['SCORED']}，FAIL {counts['FAIL']}，"
            f"ERROR {counts['ERROR']}，共 {len(results)} 个用例。"
        ),
        "",
        "> SCORED 表示评测协议完整执行且没有触发 P0 红线；在 P1/P2 阈值完成人工校准前，不等同于 PASS。",
        "",
        "| case | 描述 | P0 | P1 | P2 平均 | 结果 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.case_id} | {result.description} | "
            f"{_p0_summary(result)} | {_p1_summary(result)} | "
            f"{_p2_summary(result)} | {result.outcome} |",
        )

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
                    f"- [{mark}] {item.criterion}：{item.reason}（证据：{refs}）",
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
