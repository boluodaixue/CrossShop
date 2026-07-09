"""Tests for Rubric v3 runner selection, retry, and safe failures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts.eval.rubric_cases import load_case_suite
from scripts.eval.rubric_contract import (
    EvaluationEvidence,
    P2CriterionSpec,
    PreferenceStateEvidence,
    RubricSpec,
    StructuredStateEvidence,
    TurnEvidence,
)
from scripts.eval.rubric_ground_truth import ProductFactWindow
from scripts.eval.rubric_judge import build_judge_request
from scripts.eval.rubric_runner import (
    CompletedCaseArtifacts,
    EvaluationRunError,
    _safe_failure,
    _scoped_buyer_id,
    build_dependency_context,
    build_run_manifest,
    call_judge,
    dependency_is_ready,
    select_cases,
    write_failed_case_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _state(before: int = 0, after: int = 0) -> StructuredStateEvidence:
    return StructuredStateEvidence(
        preference_before=PreferenceStateEvidence(
            count=before,
            content_hash="0" * 64,
        ),
        preference_after=PreferenceStateEvidence(
            count=after,
            content_hash="a" * 64,
        ),
    )


def _request():
    return build_judge_request(
        case_id="chat",
        description="普通聊天",
        prior_context="",
        rubric=RubricSpec(
            p0=["不得编造"],
            p1=["不得调用业务工具"],
            p2=[
                P2CriterionSpec(
                    criterion="闲聊体验",
                    score_1_anchor="答非所问",
                    score_5_anchor="简洁友好",
                ),
            ],
        ),
        evidence=EvaluationEvidence(
            case_id="chat",
            session_id_hash="0123456789abcdef",
            turns=[
                TurnEvidence(
                    turn_index=1,
                    user_input="你好",
                    route="main.direct",
                    structured_state=_state(),
                    final_text="你好",
                ),
            ],
        ),
        product_facts=ProductFactWindow(
            requested_product_ids=[],
            products=[],
            missing_product_ids=[],
            facts_sha256="0" * 64,
        ),
    )


def test_case_selection_includes_declared_dependency_in_file_order() -> None:
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml")

    selected = select_cases(suite.cases, ["memory-recall"])

    assert [case.id for case in selected] == ["memory-write", "memory-recall"]
    with pytest.raises(ValueError, match="unknown case ids"):
        select_cases(suite.cases, ["missing"])


def test_run_scoped_buyer_is_stable_within_run_and_isolated_between_runs() -> None:
    first = _scoped_buyer_id("run-one", "logical-buyer")

    assert first == _scoped_buyer_id("run-one", "logical-buyer")
    assert first != _scoped_buyer_id("run-one", "other-buyer")
    assert first != _scoped_buyer_id("run-two", "logical-buyer")
    assert "logical-buyer" not in first


def _completed_artifacts(*, p1_failed: int = 0) -> CompletedCaseArtifacts:
    request = _request()
    result = SimpleNamespace(
        case_id="memory-write",
        outcome="SCORED",
        scorecard=SimpleNamespace(
            p0_passed=1,
            p0_total=1,
            p1_failed=p1_failed,
        ),
    )
    request.evidence.turns[0].structured_state = _state(before=0, after=1)
    return CompletedCaseArtifacts(
        evidence=request.evidence,
        product_facts=request.product_facts,
        judge_request=request,
        raw_judgement='{"p0":[],"p1":[],"p2":[]}',
        result=result,  # type: ignore[arg-type]
    )


def test_dependency_gate_requires_clean_p0_and_p1() -> None:
    assert dependency_is_ready(_completed_artifacts()) is True
    assert dependency_is_ready(_completed_artifacts(p1_failed=1)) is False


def test_dependency_context_is_derived_from_evidence_without_buyer_identity() -> None:
    context = build_dependency_context([_completed_artifacts()])
    payload = json.loads(context)

    assert payload[0]["case_id"] == "memory-write"
    assert payload[0]["turns"][0]["final_text"] == "你好"
    assert payload[0]["turns"][0]["preference_after"]["count"] == 1
    assert "logical-buyer" not in context


def test_run_manifest_has_source_identity_but_no_endpoint_or_secret(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        product_catalog_root=tmp_path,
        llm_model="application-model",
        product_embedding_model="BAAI/bge-m3",
        product_embedding_dim=1024,
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_base_url="http://private-reranker.invalid",
        category_kb_collection="globex_category_kb_v1",
        qdrant_url="http://private-qdrant.invalid",
        database_url="file",
        semantic_cache_enabled=False,
        redis_url="redis://private.invalid",
        queue_enabled=True,
        langfuse_enabled=True,
    )

    manifest = build_run_manifest(
        settings=settings,
        cases_path=PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml",
        judge_model="judge-model",
        run_scope="private-run-scope",
        startup_checks={"taobao": "ready"},
        evaluation_policy=load_case_suite(
            PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml",
        ).evaluation_policy,
    )
    serialized = json.dumps(manifest)

    assert manifest["schema_version"] == "rubric-run-manifest-v2"
    assert manifest["quality_policy_id"] == "globex-quality-v1"
    assert manifest["coverage_policy_id"] == "globex-scenario-coverage-v1"
    assert len(manifest["evaluation_policy_sha256"]) == 64
    assert len(manifest["source_tree_sha256"]) == 64
    assert len(manifest["category_release_manifest_sha256"]) == 64
    assert len(manifest["category_approved_cards_sha256"]) == 64
    assert manifest["product_search_startup_checks"] == {"taobao": "ready"}
    assert "private-reranker" not in serialized
    assert "private-qdrant" not in serialized
    assert "private-run-scope" not in serialized


@pytest.mark.asyncio
async def test_judge_retries_one_transient_response_with_identical_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": '{"p0":[],"p1":[],"p2":[]}'}}]},
        )

    settings = SimpleNamespace(
        llm_base_url="https://judge.invalid/v1",
        llm_api_key="private-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        output = await call_judge(
            client,
            settings,
            _request(),
            model="judge-model",
        )

    assert json.loads(output) == {"p0": [], "p1": [], "p2": []}
    assert len(requests) == 2
    assert requests[0].content == requests[1].content


def test_failure_report_does_not_copy_exception_message() -> None:
    case = load_case_suite(
        PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml",
    ).cases[0]

    result = _safe_failure(
        case,
        "judge_transport",
        RuntimeError("secret upstream body sk-do-not-copy"),
    )

    assert result.outcome == "ERROR"
    assert result.failure is not None
    assert result.failure.error_type == "RuntimeError"
    assert "sk-do-not-copy" not in result.model_dump_json()


def test_protocol_error_keeps_safe_intermediate_artifacts(tmp_path: Path) -> None:
    case = load_case_suite(
        PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml",
    ).cases[0]
    request = _request()
    result = _safe_failure(case, "judge_protocol", ValueError("bad schema"))
    error = EvaluationRunError(
        "judge_protocol",
        ValueError("bad schema"),
        evidence=request.evidence,
        product_facts=request.product_facts,
        judge_request=request,
        raw_judgement='{"unexpected": true}',
    )

    write_failed_case_artifacts(tmp_path, case, result, error)

    case_dir = tmp_path / case.id
    assert (case_dir / "evidence.json").exists()
    assert (case_dir / "product_facts.json").exists()
    assert (case_dir / "judge_request.json").exists()
    saved_request = json.loads(
        (case_dir / "judge_request.json").read_text(encoding="utf-8")
    )
    assert "session_id_hash" not in saved_request["evidence"]
    assert (case_dir / "judge_response_redacted.txt").read_text(encoding="utf-8") == (
        '{"unexpected": true}'
    )
    saved_result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    assert saved_result["outcome"] == "ERROR"
