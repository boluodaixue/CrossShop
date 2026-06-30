"""Run V2 Rubric cases against the real in-process Globex composition.

The runner subscribes to the same local EventBus used by the application and
installs a privacy-safe local OTel exporter before the container is built.
LangFuse remains an optional external observability destination; evaluation
never reads it back and never depends on its availability.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from opentelemetry.sdk.trace.export import SpanExportResult

from app.application.agents.orchestrator import SubmitIntentInput
from app.composition import Container, build_container
from app.infrastructure.settings import Settings, load_settings
from app.infrastructure.tracing import setup_tracing
from app.infrastructure.transient import is_transient_error
from scripts.eval.rubric_cases import load_case_suite
from scripts.eval.rubric_contract import (
    EvaluationEvidence,
    PreferenceStateEvidence,
    RubricCaseSpec,
    TurnEvidence,
)
from scripts.eval.rubric_evidence import (
    build_evaluation_evidence,
    build_structured_state_evidence,
    build_turn_evidence,
    redact_text,
)
from scripts.eval.rubric_ground_truth import (
    ProductFactWindow,
    build_product_fact_window,
    collect_case_product_ids,
)
from scripts.eval.rubric_judge import (
    RubricJudgeRequest,
    build_judge_messages,
    build_judge_request,
    parse_judge_output,
)
from scripts.eval.rubric_report import (
    CaseEvaluationResult,
    EvaluationFailure,
    completed_result,
    error_result,
    render_markdown_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = PROJECT_ROOT / "eval" / "rubric_cases_v2.yaml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "rubric"
_JUDGE_MAX_ATTEMPTS = 2


class EvaluationRunError(RuntimeError):
    """Attach a stable phase and all safe artifacts completed before failure."""

    def __init__(
        self,
        phase: str,
        cause: Exception,
        *,
        evidence: EvaluationEvidence | None = None,
        product_facts: ProductFactWindow | None = None,
        judge_request: RubricJudgeRequest | None = None,
        raw_judgement: str | None = None,
    ) -> None:
        super().__init__(phase)
        self.phase = phase
        self.cause = cause
        self.evidence = evidence
        self.product_facts = product_facts
        self.judge_request = judge_request
        self.raw_judgement = raw_judgement


class LocalSpanCollector:
    """Thread-safe OTel exporter used only by an evaluation process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: list[Any] = []

    def export(self, spans: Sequence[Any]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def snapshot(self) -> list[Any]:
        with self._lock:
            return list(self._spans)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True

    def shutdown(self) -> None:
        return None


@dataclass(frozen=True)
class CompletedCaseArtifacts:
    evidence: EvaluationEvidence
    product_facts: ProductFactWindow
    judge_request: RubricJudgeRequest
    raw_judgement: str
    result: CaseEvaluationResult


def select_cases(
    cases: list[RubricCaseSpec],
    requested_ids: list[str],
) -> list[RubricCaseSpec]:
    """Select requested cases and all earlier declared dependencies."""

    if not requested_ids:
        return cases
    by_id = {case.id: case for case in cases}
    unknown = sorted(set(requested_ids) - set(by_id))
    if unknown:
        raise ValueError(f"unknown case ids: {', '.join(unknown)}")
    selected: set[str] = set()

    def include(case_id: str) -> None:
        if case_id in selected:
            return
        for dependency in by_id[case_id].depends_on:
            include(dependency)
        selected.add(case_id)

    for case_id in requested_ids:
        include(case_id)
    return [case for case in cases if case.id in selected]


def _turn_spans(collector: LocalSpanCollector, offset: int) -> list[Any]:
    spans = collector.snapshot()[offset:]
    roots = [span for span in spans if span.name == "globex.shopping_intent"]
    if len(roots) != 1:
        raise RuntimeError(
            f"expected one globex.shopping_intent span, observed {len(roots)}",
        )
    trace_id = roots[0].context.trace_id
    selected = [span for span in spans if span.context.trace_id == trace_id]
    if len(selected) != len(spans):
        raise RuntimeError("turn emitted spans from multiple trace ids")
    return selected


async def collect_case_evidence(
    container: Container,
    collector: LocalSpanCollector,
    case: RubricCaseSpec,
    *,
    buyer_id: str,
) -> EvaluationEvidence:
    """Execute all turns and capture complete local events plus OTel spans."""

    session_id = f"rubric-{case.id}-{uuid.uuid4().hex[:10]}"
    turns: list[TurnEvidence] = []
    for turn_index, query in enumerate(case.queries, start=1):
        queue = container.bus.subscribe(session_id)
        span_offset = len(collector.snapshot())
        try:
            preference_before = await _preference_state(container, buyer_id)
            output = await container.orchestrator.handle_intent(
                SubmitIntentInput(
                    shopping_session_id=session_id,
                    buyer_id=buyer_id,
                    locale="zh-CN",
                    currency="CNY",
                    raw_query=query,
                ),
            )
            events = []
            while not queue.empty():
                events.append(queue.get_nowait())
            preference_after = await _preference_state(container, buyer_id)
            session_context = await _session_context(container, session_id)
            turns.append(
                build_turn_evidence(
                    turn_index=turn_index,
                    user_input=query,
                    final_text=output.final_text,
                    displayed_products=list(output.displayed_products),
                    events=events,
                    structured_state=build_structured_state_evidence(
                        session_context,
                        preference_before=preference_before,
                        preference_after=preference_after,
                    ),
                    spans=_turn_spans(collector, span_offset),
                ),
            )
        finally:
            container.bus.unsubscribe(session_id, queue)
    return build_evaluation_evidence(
        case_id=case.id,
        session_id=session_id,
        turns=turns,
    )


async def _preference_state(
    container: Container,
    buyer_id: str,
) -> PreferenceStateEvidence:
    preferences = await container.orchestrator._preference_store.list_by_buyer(  # noqa: SLF001
        buyer_id,
    )
    canonical = json.dumps(
        sorted((item.kind, item.statement.strip()) for item in preferences),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return PreferenceStateEvidence(
        count=len(preferences),
        content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


async def _session_context(container: Container, session_id: str) -> Any:
    agent = await container.orchestrator._sessions.get_or_create(session_id)  # noqa: SLF001
    namespace = agent.state.middle_context.get("globex_context_v1", {})
    if not isinstance(namespace, dict):
        return {}
    context = namespace.get("session_context")
    return context if isinstance(context, dict) else {}


async def call_judge(
    client: httpx.AsyncClient,
    settings: Settings,
    request: RubricJudgeRequest,
    *,
    model: str,
) -> str:
    """Call the configured OpenAI-compatible judge with one exact retry."""

    payload = {
        "model": model,
        "messages": build_judge_messages(request),
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    for attempt in range(1, _JUDGE_MAX_ATTEMPTS + 1):
        try:
            response = await client.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            body = response.json()
            return str(body["choices"][0]["message"]["content"])
        except Exception as err:
            if attempt == _JUDGE_MAX_ATTEMPTS or not is_transient_error(err):
                raise
            await asyncio.sleep(1.0)
    raise AssertionError("judge retry loop exhausted")


async def evaluate_case(
    container: Container,
    collector: LocalSpanCollector,
    client: httpx.AsyncClient,
    settings: Settings,
    case: RubricCaseSpec,
    *,
    judge_model: str,
    buyer_id: str,
    prior_context: str,
) -> CompletedCaseArtifacts:
    try:
        evidence = await collect_case_evidence(
            container,
            collector,
            case,
            buyer_id=buyer_id,
        )
    except Exception as err:
        raise EvaluationRunError("evidence_collection", err) from err
    try:
        product_ids = collect_case_product_ids(
            case.ground_truth_product_ids,
            evidence,
        )
        product_facts = await build_product_fact_window(
            container.product_repo,
            product_ids,
        )
        request = build_judge_request(
            case_id=case.id,
            description=case.description,
            prior_context=prior_context,
            rubric=case.rubric,
            evidence=evidence,
            product_facts=product_facts,
        )
    except Exception as err:
        raise EvaluationRunError(
            "ground_truth",
            err,
            evidence=evidence,
        ) from err
    try:
        raw_judgement = await call_judge(
            client,
            settings,
            request,
            model=judge_model,
        )
    except Exception as err:
        raise EvaluationRunError(
            "judge_transport",
            err,
            evidence=evidence,
            product_facts=product_facts,
            judge_request=request,
        ) from err
    try:
        judgement, scorecard = parse_judge_output(raw_judgement, request)
    except Exception as err:
        raise EvaluationRunError(
            "judge_protocol",
            err,
            evidence=evidence,
            product_facts=product_facts,
            judge_request=request,
            raw_judgement=raw_judgement,
        ) from err
    result = completed_result(
        case_id=case.id,
        description=case.description,
        judgement=judgement,
        scorecard=scorecard,
    )
    return CompletedCaseArtifacts(
        evidence=evidence,
        product_facts=product_facts,
        judge_request=request,
        raw_judgement=raw_judgement,
        result=result,
    )


def _safe_failure(
    case: RubricCaseSpec,
    phase: str,
    error: Exception,
) -> CaseEvaluationResult:
    return error_result(
        case_id=case.id,
        description=case.description,
        failure=EvaluationFailure(
            phase=phase,
            error_type=type(error).__name__,
            message="评测未能在该阶段形成可信结果；详细异常仅保留在本地运行日志。",
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_judge_request(path: Path, request: RubricJudgeRequest) -> None:
    value = request.model_dump(mode="json")
    value["evidence"].pop("session_id_hash", None)
    _write_json(path, value)


def write_case_artifacts(
    run_dir: Path,
    artifacts: CompletedCaseArtifacts,
) -> None:
    case_dir = run_dir / artifacts.result.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "evidence.json", artifacts.evidence)
    _write_json(case_dir / "product_facts.json", artifacts.product_facts)
    _write_judge_request(case_dir / "judge_request.json", artifacts.judge_request)
    (case_dir / "judge_response_redacted.txt").write_text(
        redact_text(artifacts.raw_judgement),
        encoding="utf-8",
    )
    _write_json(case_dir / "result.json", artifacts.result)


def write_failed_case_artifacts(
    run_dir: Path,
    case: RubricCaseSpec,
    result: CaseEvaluationResult,
    error: Exception,
) -> None:
    """Persist every trustworthy intermediate artifact without inventing a score."""

    case_dir = run_dir / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(error, EvaluationRunError):
        if error.evidence is not None:
            _write_json(case_dir / "evidence.json", error.evidence)
        if error.product_facts is not None:
            _write_json(case_dir / "product_facts.json", error.product_facts)
        if error.judge_request is not None:
            _write_judge_request(case_dir / "judge_request.json", error.judge_request)
        if error.raw_judgement is not None:
            (case_dir / "judge_response_redacted.txt").write_text(
                redact_text(error.raw_judgement),
                encoding="utf-8",
            )
    _write_json(case_dir / "result.json", result)


def _scoped_buyer_id(run_scope: str, logical_buyer_id: str) -> str:
    digest = hashlib.sha256(f"{run_scope}:{logical_buyer_id}".encode()).hexdigest()
    return f"rubric-buyer-{digest[:20]}"


def dependency_is_ready(artifacts: CompletedCaseArtifacts) -> bool:
    scorecard = artifacts.result.scorecard
    return bool(
        artifacts.result.outcome == "SCORED"
        and scorecard is not None
        and scorecard.p0_passed == scorecard.p0_total
        and scorecard.p1_failed == 0
    )


def build_dependency_context(
    dependencies: list[CompletedCaseArtifacts],
) -> str:
    """Build bounded context only from completed, locally captured dependencies."""

    rows: list[dict[str, Any]] = []
    for artifacts in dependencies:
        turns = []
        for turn in artifacts.evidence.turns:
            turns.append(
                {
                    "turn_index": turn.turn_index,
                    "user_input": turn.user_input,
                    "final_text": turn.final_text,
                    "tool_calls": [
                        {
                            "tool": call.tool,
                            "status": call.status,
                            "result_summary": call.result_summary,
                        }
                        for call in turn.tool_calls
                    ],
                    "preference_before": (
                        turn.structured_state.preference_before.model_dump()
                    ),
                    "preference_after": (
                        turn.structured_state.preference_after.model_dump()
                    ),
                },
            )
        rows.append(
            {
                "case_id": artifacts.result.case_id,
                "outcome": artifacts.result.outcome,
                "turns": turns,
            },
        )
    return redact_text(json.dumps(rows, ensure_ascii=False))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree_sha256(cases_path: Path) -> str:
    candidates = [PROJECT_ROOT / "pyproject.toml", cases_path]
    candidates.extend((PROJECT_ROOT / "app").rglob("*.py"))
    candidates.extend((PROJECT_ROOT / "app").rglob("*.yml"))
    candidates.extend((PROJECT_ROOT / "app").rglob("*.yaml"))
    candidates.extend((PROJECT_ROOT / "scripts" / "eval").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in candidates if item.is_file()}):
        try:
            label = path.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            label = f"external/{path.name}"
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_run_manifest(
    *,
    settings: Settings,
    cases_path: Path,
    judge_model: str,
    run_scope: str,
    startup_checks: dict[str, str],
) -> dict[str, Any]:
    catalog_manifest = settings.product_catalog_root / "manifest.json"
    category_release = (
        PROJECT_ROOT / "data" / "category_insight" / "releases" / "category-insight-v1"
    )
    category_manifest = category_release / "manifest.json"
    category_cards = category_release / "approved_cards.jsonl"
    return {
        "schema_version": "rubric-run-manifest-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "git_head": _git_value("rev-parse", "HEAD"),
        "source_tree_sha256": _source_tree_sha256(cases_path),
        "case_file_sha256": _file_sha256(cases_path),
        "catalog_manifest_sha256": (
            _file_sha256(catalog_manifest) if catalog_manifest.is_file() else None
        ),
        "category_release_manifest_sha256": (
            _file_sha256(category_manifest) if category_manifest.is_file() else None
        ),
        "category_approved_cards_sha256": (
            _file_sha256(category_cards) if category_cards.is_file() else None
        ),
        "run_scope_sha256": hashlib.sha256(run_scope.encode()).hexdigest(),
        "judge_model": judge_model,
        "application_model": settings.llm_model,
        "product_embedding_model": settings.product_embedding_model,
        "product_embedding_dim": settings.product_embedding_dim,
        "reranker_model": settings.reranker_model or None,
        "reranker_configured": bool(settings.reranker_base_url),
        "category_collection": settings.category_kb_collection,
        "qdrant_mode": "remote" if settings.qdrant_url else "embedded",
        "database_mode": (
            "json_file" if settings.database_url == "file" else "configured_database"
        ),
        "semantic_cache_enabled": settings.semantic_cache_enabled,
        "queue_active": bool(settings.redis_url and settings.queue_enabled),
        "langfuse_enabled": settings.langfuse_enabled,
        "product_search_startup_checks": dict(sorted(startup_checks.items())),
    }


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    if settings.semantic_cache_enabled and not args.allow_semantic_cache:
        raise RuntimeError(
            "semantic cache is enabled; rerun with SEMANTIC_CACHE_ENABLED=0",
        )
    suite = load_case_suite(Path(args.cases))
    cases = select_cases(suite.cases, args.only)
    judge_model = (
        args.judge_model or os.getenv("EVAL_JUDGE_MODEL") or settings.llm_model
    )
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_scope = uuid.uuid4().hex
    buyer_aliases: dict[str, str] = {}

    collector = LocalSpanCollector()
    setup_tracing(settings, exporter=collector)
    container = await build_container()
    results: list[CaseEvaluationResult] = []
    completed_cases: dict[str, CompletedCaseArtifacts] = {}
    await container.startup()
    _write_json(
        run_dir / "run_manifest.json",
        build_run_manifest(
            settings=settings,
            cases_path=Path(args.cases),
            judge_model=judge_model,
            run_scope=run_scope,
            startup_checks=container.product_search_startup_check,
        ),
    )
    try:
        async with httpx.AsyncClient() as client:
            for case in cases:
                print(f"== Rubric {case.id}", flush=True)
                dependency_artifacts = [
                    completed_cases[case_id]
                    for case_id in case.depends_on
                    if case_id in completed_cases
                ]
                unavailable = [
                    case_id
                    for case_id in case.depends_on
                    if case_id not in completed_cases
                    or not dependency_is_ready(completed_cases[case_id])
                ]
                if unavailable:
                    cause = RuntimeError(
                        "required dependency did not produce a ready P0/P1 result",
                    )
                    result = _safe_failure(case, "dependency", cause)
                    write_failed_case_artifacts(run_dir, case, result, cause)
                    results.append(result)
                    print("   ERROR dependency/RuntimeError", flush=True)
                    continue
                logical_buyer_id = case.buyer_id or case.id
                buyer_id = buyer_aliases.setdefault(
                    logical_buyer_id,
                    _scoped_buyer_id(run_scope, logical_buyer_id),
                )
                dynamic_context = build_dependency_context(dependency_artifacts)
                prior_context = "\n".join(
                    item
                    for item in (case.prior_context, dynamic_context)
                    if item.strip()
                )
                try:
                    artifacts = await evaluate_case(
                        container,
                        collector,
                        client,
                        settings,
                        case,
                        judge_model=judge_model,
                        buyer_id=buyer_id,
                        prior_context=prior_context,
                    )
                except Exception as err:  # keep batch evidence; never fake a score
                    cause = err.cause if isinstance(err, EvaluationRunError) else err
                    phase = (
                        err.phase
                        if isinstance(err, EvaluationRunError)
                        else "intent_execution"
                    )
                    print(
                        f"   ERROR {phase}/{type(cause).__name__}",
                        flush=True,
                    )
                    if phase in {
                        "evidence_collection",
                        "ground_truth",
                        "judge_protocol",
                    }:
                        print(f"   detail: {redact_text(str(cause))[:500]}", flush=True)
                    result = _safe_failure(case, phase, cause)
                    write_failed_case_artifacts(run_dir, case, result, err)
                else:
                    write_case_artifacts(run_dir, artifacts)
                    result = artifacts.result
                    completed_cases[case.id] = artifacts
                    print(f"   {result.outcome}", flush=True)
                results.append(result)
    finally:
        await container.shutdown()

    report = render_markdown_report(results, generated_at=datetime.now())
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    _write_json(
        run_dir / "results.json",
        [item.model_dump(mode="json", by_alias=True) for item in results],
    )
    print(f"报告：{run_dir / 'report.md'}", flush=True)
    return 1 if any(result.outcome in {"FAIL", "ERROR"} for result in results) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Globex V2 Rubric evaluation")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--allow-semantic-cache", action="store_true")
    return parser


def main() -> None:
    try:
        exit_code = asyncio.run(run(build_parser().parse_args()))
    except Exception as err:  # startup/configuration must not print secrets or URLs
        print(
            f"ERROR startup/{type(err).__name__}: "
            "评测未启动；详细异常仅保留在本地受控日志。",
            flush=True,
        )
        raise SystemExit(2) from None
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
