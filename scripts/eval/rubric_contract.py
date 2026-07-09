"""Rubric evaluation contracts shared by offline runners and tests.

This module deliberately contains no model, network, LangFuse, or application
runtime dependency.  It defines the evidence envelope and the P0/P1/P2
scorecard described by the V2 course:

* P0 is an essential red-line gate.
* P1 records important process violations at two deduction points each.
* P2 scores answer quality from one to five per dimension.

The judge returns the three sections separately.  The deterministic report
layer applies the versioned, user-approved 25/35/40 policy and keeps the P0
red-line gate non-compensating.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_P1_DEDUCTION_POINTS = 2


class RubricProtocolError(ValueError):
    """The judge output cannot be compared with the requested rubric."""


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


class P2CriterionSpec(BaseModel):
    """One scored quality dimension with explicit one/five-point anchors."""

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(min_length=1)
    score_1_anchor: str = Field(min_length=1)
    score_5_anchor: str = Field(min_length=1)


class RubricSpec(BaseModel):
    """Human-reviewed rubric sent to the offline judge for one eval case."""

    model_config = ConfigDict(extra="forbid")

    p0: list[str] = Field(min_length=1)
    p1: list[str] = Field(min_length=1)
    p2: list[P2CriterionSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_criteria(self) -> RubricSpec:
        for level, criteria in (
            ("p0", self.p0),
            ("p1", self.p1),
            ("p2", [item.criterion for item in self.p2]),
        ):
            duplicates = _duplicates(criteria)
            if duplicates:
                raise ValueError(
                    f"{level} contains duplicate criteria: {', '.join(duplicates)}",
                )
        return self


class QualityWeightPolicy(BaseModel):
    """Versioned weights for the human-readable quality score."""

    model_config = ConfigDict(extra="forbid")

    p0: float = Field(gt=0, le=100)
    p1: float = Field(gt=0, le=100)
    p2: float = Field(gt=0, le=100)

    @model_validator(mode="after")
    def require_one_hundred_percent(self) -> QualityWeightPolicy:
        if abs((self.p0 + self.p1 + self.p2) - 100.0) > 1e-9:
            raise ValueError("quality weights must sum to 100")
        return self


class ScenarioFamilyPolicy(BaseModel):
    """One user-journey family and its non-gameable required coverage points."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    weight: float = Field(gt=0, le=100)
    required_points: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_point_ids(self) -> ScenarioFamilyPolicy:
        invalid = sorted(
            point_id
            for point_id in self.required_points
            if not point_id
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in point_id
            )
        )
        if invalid:
            raise ValueError(
                "coverage point ids must use lowercase letters, numbers, and hyphens: "
                + ", ".join(invalid),
            )
        if any(not label.strip() for label in self.required_points.values()):
            raise ValueError("coverage point labels must be non-empty")
        return self


class EvaluationPolicy(BaseModel):
    """Frozen score and scenario-coverage policy for one case suite."""

    model_config = ConfigDict(extra="forbid")

    quality_policy_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    coverage_policy_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    quality_weights: QualityWeightPolicy
    scenario_families: dict[str, ScenarioFamilyPolicy] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_family_weights(self) -> EvaluationPolicy:
        invalid = sorted(
            family_id
            for family_id in self.scenario_families
            if not family_id
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for char in family_id
            )
        )
        if invalid:
            raise ValueError(
                "scenario family ids must use lowercase letters, numbers, and hyphens: "
                + ", ".join(invalid),
            )
        total = sum(family.weight for family in self.scenario_families.values())
        if abs(total - 100.0) > 1e-9:
            raise ValueError("scenario family weights must sum to 100")
        if self.quality_policy_id == "globex-quality-v1" and (
            self.quality_weights.model_dump() != {"p0": 25.0, "p1": 35.0, "p2": 40.0}
        ):
            raise ValueError("globex-quality-v1 requires P0/P1/P2=25/35/40")
        return self


class RubricCaseSpec(BaseModel):
    """One versioned, human-reviewed end-to-end regression case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str = Field(min_length=1)
    queries: list[str] = Field(min_length=1)
    buyer_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    prior_context: str = ""
    execution_profile: Literal["default", "force-context-summary"] = "default"
    ground_truth_product_ids: list[str] = Field(default_factory=list)
    scenario_family: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    coverage_points: list[str] = Field(min_length=1)
    rubric: RubricSpec

    @model_validator(mode="after")
    def validate_case_contract(self) -> RubricCaseSpec:
        if any(not query.strip() for query in self.queries):
            raise ValueError("queries must contain non-empty strings")
        if len(self.ground_truth_product_ids) != len(
            set(self.ground_truth_product_ids)
        ):
            raise ValueError("ground_truth_product_ids must be unique")
        if self.id in self.depends_on:
            raise ValueError("case cannot depend on itself")
        duplicates = _duplicates(self.coverage_points)
        if duplicates:
            raise ValueError(
                "coverage_points contains duplicates: " + ", ".join(duplicates),
            )
        return self


class RubricCaseSuite(BaseModel):
    """The versioned case-file contract loaded by the regression runner."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["rubric-cases-v3"]
    evaluation_policy: EvaluationPolicy
    cases: list[RubricCaseSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_suite(self) -> RubricCaseSuite:
        case_ids = [case.id for case in self.cases]
        duplicates = _duplicates(case_ids)
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(duplicates)}")
        known: set[str] = set()
        all_ids = set(case_ids)
        for case in self.cases:
            unknown = sorted(set(case.depends_on) - all_ids)
            if unknown:
                raise ValueError(
                    f"case {case.id} has unknown dependencies: {', '.join(unknown)}",
                )
            not_earlier = sorted(set(case.depends_on) - known)
            if not_earlier:
                raise ValueError(
                    f"case {case.id} dependencies must appear earlier: "
                    f"{', '.join(not_earlier)}",
                )
            known.add(case.id)
            family = self.evaluation_policy.scenario_families.get(
                case.scenario_family,
            )
            if family is None:
                raise ValueError(
                    f"case {case.id} has unknown scenario family: "
                    f"{case.scenario_family}",
                )
            unknown_points = sorted(
                set(case.coverage_points) - set(family.required_points),
            )
            if unknown_points:
                raise ValueError(
                    f"case {case.id} has unknown coverage points for "
                    f"{case.scenario_family}: {', '.join(unknown_points)}",
                )
        return self


class BinaryCriterionJudgement(BaseModel):
    """Judge result for a P0 or P1 criterion."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    criterion: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    passed: bool = Field(alias="pass")
    evidence_refs: list[str] = Field(default_factory=list)


class QualityCriterionJudgement(BaseModel):
    """Judge result for a P2 dimension."""

    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    score: int = Field(ge=1, le=5)
    evidence_refs: list[str] = Field(default_factory=list)


class RubricJudgement(BaseModel):
    """Structured output returned by the offline rubric judge."""

    model_config = ConfigDict(extra="forbid")

    p0: list[BinaryCriterionJudgement] = Field(default_factory=list)
    p1: list[BinaryCriterionJudgement] = Field(default_factory=list)
    p2: list[QualityCriterionJudgement] = Field(default_factory=list)


class RubricScorecard(BaseModel):
    """Separate, explainable P0/P1/P2 results without a fabricated total."""

    verdict: Literal["redline_fail", "scored"]
    p0_passed: int = Field(ge=0)
    p0_total: int = Field(ge=0)
    p1_failed: int = Field(ge=0)
    p1_total: int = Field(ge=0)
    p1_deduction_points: int = Field(ge=0)
    p2_scores: dict[str, int] = Field(default_factory=dict)
    p2_average: float | None = Field(default=None, ge=1, le=5)


class PreferenceStateEvidence(BaseModel):
    """Count and one-way content fingerprint; preference text never leaves locally."""

    count: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ProductReferenceEvidence(BaseModel):
    """Identity-only verified reference stored in L4 context."""

    platform: Literal["globex_reference", "taobao", "amazon"]
    product_id: str = Field(min_length=1)
    site_locale: Literal["us", "es", "jp"] | None = None


class ContextLifecycleEvidence(BaseModel):
    """Privacy-safe proof that the real L2/L3 lifecycle advanced."""

    raw_context_message_count: int = Field(default=0, ge=0)
    freeze_cursor: int = Field(default=0, ge=0)
    frozen_segment_count: int = Field(default=0, ge=0)
    stage_summary_present: bool = False
    stage_summary_source_count: int = Field(default=0, ge=0)
    stage_source_message_start: int | None = Field(default=None, ge=0)
    stage_source_message_end: int | None = Field(default=None, ge=0)
    stage_summary_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    stage_summary_verified: bool = False
    compression_action: Literal["none", "l3_stage_summary", "l3_no_gain"] = "none"
    before_tokens: int | None = Field(default=None, ge=0)
    after_tokens: int | None = Field(default=None, ge=0)
    budget_decision: str = ""
    total_input_tokens: int = Field(default=0, ge=0)
    soft_limit_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_successful_summary(self) -> ContextLifecycleEvidence:
        if self.compression_action != "l3_stage_summary":
            return self
        if self.before_tokens is None or self.after_tokens is None:
            raise ValueError("successful L3 compression requires token counts")
        if self.after_tokens >= self.before_tokens:
            raise ValueError("successful L3 compression must reduce tokens")
        if not self.stage_summary_present or not self.stage_summary_verified:
            raise ValueError("successful L3 compression requires a verified summary")
        if self.stage_summary_source_count < 1 or not self.stage_summary_hash:
            raise ValueError("successful L3 compression requires source evidence")
        return self


class StructuredStateEvidence(BaseModel):
    """Bounded privacy-safe context lifecycle, L4, and preference snapshot."""

    schema_version: str = ""
    revision: int = Field(default=0, ge=0)
    last_search_arguments: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=3,
    )
    last_search_product_refs: list[str] = Field(default_factory=list, max_length=20)
    last_search_filtered_refs: list[str] = Field(default_factory=list, max_length=20)
    current_recommendations: list[ProductReferenceEvidence] = Field(
        default_factory=list,
        max_length=5,
    )
    verified_display_history: list[ProductReferenceEvidence] = Field(
        default_factory=list,
        max_length=20,
    )
    order: dict[str, Any] | None = None
    context_lifecycle: ContextLifecycleEvidence = Field(
        default_factory=ContextLifecycleEvidence,
    )
    preference_before: PreferenceStateEvidence
    preference_after: PreferenceStateEvidence


def _criterion_names(items: list[Any]) -> list[str]:
    return [str(item.criterion) for item in items]


def _assert_same_criteria(
    level: str,
    expected: list[str],
    actual: list[str],
) -> None:
    if len(actual) != len(set(actual)):
        raise RubricProtocolError(f"judge returned duplicate {level} criteria")
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RubricProtocolError(
            f"judge {level} criteria mismatch; missing={missing}, extra={extra}",
        )


def score_rubric(
    spec: RubricSpec,
    judgement: RubricJudgement,
) -> RubricScorecard:
    """Validate judge coverage and build the unweighted rubric scorecard."""

    _assert_same_criteria("p0", spec.p0, _criterion_names(judgement.p0))
    _assert_same_criteria("p1", spec.p1, _criterion_names(judgement.p1))
    _assert_same_criteria(
        "p2",
        [item.criterion for item in spec.p2],
        _criterion_names(judgement.p2),
    )

    p0_passed = sum(item.passed for item in judgement.p0)
    p1_failed = sum(not item.passed for item in judgement.p1)
    p2_scores = {item.criterion: item.score for item in judgement.p2}
    p2_average = (
        round(sum(p2_scores.values()) / len(p2_scores), 3) if p2_scores else None
    )
    return RubricScorecard(
        verdict=("scored" if p0_passed == len(judgement.p0) else "redline_fail"),
        p0_passed=p0_passed,
        p0_total=len(judgement.p0),
        p1_failed=p1_failed,
        p1_total=len(judgement.p1),
        p1_deduction_points=p1_failed * _P1_DEDUCTION_POINTS,
        p2_scores=p2_scores,
        p2_average=p2_average,
    )


class DisplayedProductEvidence(BaseModel):
    """Identity-only product selection retained for factual verification."""

    rank: int = Field(ge=1, le=5)
    product_id: str = Field(min_length=1)
    platform: Literal["globex_reference", "taobao", "amazon"]
    site_locale: Literal["us", "es", "jp"] | None = None


class AgentDispatchEvidence(BaseModel):
    """One fixed sub-agent dispatch observed in the local event stream."""

    order: int = Field(ge=1)
    agent: Literal["search_agent", "trade_agent"]
    platform: Literal["globex_reference", "taobao", "amazon"] | None = None
    site_locale: Literal["us", "es", "jp"] | None = None
    correlation_id: str | None = None
    elapsed_ms: float | None = Field(default=None, ge=0)


class TraceSpanEvidence(BaseModel):
    """Privacy-safe topology and operational attributes from one local span."""

    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    parent_span_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{16}$")
    name: str = Field(min_length=1)
    status: Literal["UNSET", "OK", "ERROR"]
    duration_ms: float = Field(ge=0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class RuntimeSignalEvidence(BaseModel):
    """A bounded retry, fallback, circuit, harness, cache, or error signal."""

    order: int = Field(ge=1)
    signal: Literal[
        "error",
        "model.fallback",
        "cache.hit",
        "context.compressed",
        "circuit",
        "harness",
        "orphan_tool_result",
    ]
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCallEvidence(BaseModel):
    """Bounded local evidence for one completed or failed tool call."""

    order: int = Field(ge=1)
    call_id: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    dependency_attempt: int = Field(default=1, ge=1)
    status: Literal["success", "error", "unknown"] = "unknown"
    error_type: str | None = None


class TurnEvidence(BaseModel):
    """Evidence collected for exactly one buyer turn."""

    turn_index: int = Field(ge=1)
    user_input: str = Field(min_length=1)
    route: str = Field(min_length=1)
    trace_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    dispatches: list[AgentDispatchEvidence] = Field(default_factory=list)
    tool_calls: list[ToolCallEvidence] = Field(default_factory=list)
    runtime_signals: list[RuntimeSignalEvidence] = Field(default_factory=list)
    trace_spans: list[TraceSpanEvidence] = Field(default_factory=list)
    displayed_products: list[DisplayedProductEvidence] = Field(
        default_factory=list,
        max_length=5,
    )
    structured_state: StructuredStateEvidence
    final_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_turn_identity(self) -> TurnEvidence:
        orders = [item.order for item in self.tool_calls]
        if len(orders) != len(set(orders)):
            raise ValueError("tool call order must be unique within a turn")
        ranks = [item.rank for item in self.displayed_products]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("displayed product ranks must be unique and contiguous")
        identities = [
            (item.platform, item.product_id) for item in self.displayed_products
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("displayed product identities must be unique")
        return self


class EvaluationEvidence(BaseModel):
    """Local source of truth for one offline end-to-end evaluation case."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["rubric-evidence-v2"] = "rubric-evidence-v2"
    case_id: str = Field(min_length=1)
    execution_profile: Literal["default", "force-context-summary"] = "default"
    session_id_hash: str = Field(min_length=1)
    turns: list[TurnEvidence] = Field(min_length=1)

    @model_validator(mode="after")
    def require_ordered_turns(self) -> EvaluationEvidence:
        expected = list(range(1, len(self.turns) + 1))
        actual = [turn.turn_index for turn in self.turns]
        if actual != expected:
            raise ValueError(
                f"turn_index must be contiguous from 1; expected={expected}, "
                f"actual={actual}",
            )
        return self
