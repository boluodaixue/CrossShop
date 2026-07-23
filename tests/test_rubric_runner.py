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
    ToolCallEvidence,
    TurnEvidence,
)
from scripts.eval.rubric_context_artifact import (
    ContextCompressionArtifactError,
    load_context_compression_artifact,
)
from scripts.eval.rubric_ground_truth import ProductFactWindow
from scripts.eval.rubric_judge import build_judge_request
from scripts.eval.rubric_fault_injection import RubricFaultInjection
from scripts.eval.rubric_runner import (
    CompletedCaseArtifacts,
    EvaluationRunError,
    _safe_failure,
    _scoped_buyer_id,
    bind_dependency_inputs,
    build_dependency_context,
    build_run_manifest,
    call_judge,
    dependency_is_ready,
    select_cases,
    validate_case_execution_contract,
    write_failed_case_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_context_artifact(root: Path, *, repeated: bool = False) -> None:
    lifecycle = {
        "raw_context_message_count": 172,
        "freeze_cursor": 170,
        "frozen_segment_count": 85,
        "stage_summary_present": True,
        "stage_summary_source_count": 84,
        "stage_source_message_start": 0,
        "stage_source_message_end": 168,
        "stage_summary_hash": "1" * 64,
        "stage_summary_verified": True,
        "compression_action": "l3_stage_summary",
        "before_tokens": 89626,
        "after_tokens": 74085,
        "budget_decision": "ok",
        "total_input_tokens": 74805,
        "soft_limit_tokens": 89600,
    }
    recovery_lifecycle = {
        **lifecycle,
        "raw_context_message_count": 174,
        "freeze_cursor": 172,
        "frozen_segment_count": 86,
        "total_input_tokens": 75041,
    }
    manifest = {
        "schema_version": "context-compression-scenario-v1",
        "profile": "resume-clone",
        "application_model": "deepseek-v4-flash",
        "context_size": 128000,
        "semantic_cache_enabled": False,
        "source_artifact_sha256": "a" * 64,
        "anchor_artifact_sha256": "b" * 64,
        "source_session_hash": "c" * 16,
        "target_session_hash": "d" * 16,
    }
    checks = {
        "anchor_available": True,
        "anchor_title_recovered": True,
        "anchor_price_recovered": True,
        "anchor_currency_recovered": True,
        "no_product_search": True,
        "stage_summary_verified": True,
    }
    summary = {
        "source_turns": 85,
        "after_context": recovery_lifecycle,
        "compression": {"observed": True, "trigger_turn": 86},
        "recovery": {
            "passed": True,
            "probes": [
                {"probe_id": "recover-first-product", "passed": True, "checks": checks}
            ],
        },
        "source_state_unchanged": True,
        "l2_repair_passed": True,
        "l3_recovery_complete": True,
        "status": "L3_RECOVERY_PASS",
        "passed": True,
    }
    event = {
        "type": "context.compressed",
        "action": "l3_stage_summary",
        "before_tokens": 89626,
        "after_tokens": 74085,
        "source_segment_count": 84,
        "summary_hash": "1" * 64,
    }
    turns = [
        {
            "turn_index": 86,
            "phase": "resume-trigger",
            "user_input": "继续真实长会话直到自然触发压缩。",
            "final_text": "已继续处理当前购物需求。",
            "events": [event],
            "compression_event": event,
            "context": lifecycle,
        },
    ]
    if repeated:
        second_lifecycle = {
            **lifecycle,
            "raw_context_message_count": 192,
            "freeze_cursor": 190,
            "frozen_segment_count": 95,
            "stage_summary_source_count": 94,
            "stage_source_message_end": 188,
            "stage_summary_hash": "2" * 64,
            "before_tokens": 90010,
            "after_tokens": 76100,
            "total_input_tokens": 77000,
        }
        second_event = {
            "type": "context.compressed",
            "action": "l3_stage_summary",
            "before_tokens": 90010,
            "after_tokens": 76100,
            "source_segment_count": 94,
            "summary_hash": "2" * 64,
        }
        recovery_lifecycle.update(second_lifecycle)
        recovery_lifecycle.update(
            {
                "raw_context_message_count": 194,
                "freeze_cursor": 192,
                "frozen_segment_count": 96,
            }
        )
        summary["compression"] = {
            "observed": True,
            "trigger_turn": 86,
            "trigger_turns": [86, 96],
        }
        turns.append(
            {
                "turn_index": 96,
                "phase": "resume-trigger",
                "user_input": "继续真实长会话直到第二次自然压缩。",
                "final_text": "第二次压缩后继续处理。",
                "events": [second_event],
                "compression_event": second_event,
                "context": second_lifecycle,
            },
        )
        third_lifecycle = {
            **second_lifecycle,
            "raw_context_message_count": 212,
            "freeze_cursor": 210,
            "frozen_segment_count": 105,
            "stage_summary_source_count": 104,
            "stage_source_message_end": 208,
            "stage_summary_hash": "3" * 64,
            "stage_source_manifest_hash": "a" * 64,
            # A v4 rebuild from a legacy summary intentionally starts at one.
            "stage_summary_generation": 1,
            "stage_summary_estimated_tokens": 3_900,
            "stage_summary_target_tokens": 8_000,
            "stage_summary_hard_limit_tokens": 12_000,
            "before_tokens": 90_050,
            "after_tokens": 70_000,
            "compression_ratio": (90_050 - 70_000) / 90_050,
            "recent_raw_segment_count": 10,
            "post_compression_target_tokens": 70_400,
            "post_compression_max_tokens": 76_800,
            "post_compression_target_met": True,
            "total_input_tokens": 70_000,
        }
        third_event = {
            "type": "context.compressed",
            "action": "l3_stage_summary",
            "before_tokens": 90_050,
            "after_tokens": 70_000,
            "source_segment_count": 104,
            "source_manifest_hash": "a" * 64,
            "summary_hash": "3" * 64,
        }
        recovery_lifecycle.update(third_lifecycle)
        recovery_lifecycle.update(
            {
                "raw_context_message_count": 214,
                "freeze_cursor": 212,
                "frozen_segment_count": 106,
            }
        )
        summary["compression"]["trigger_turns"] = [86, 96, 106]
        turns.append(
            {
                "turn_index": 106,
                "phase": "resume-trigger",
                "user_input": "继续真实长会话直到第三次自然压缩。",
                "final_text": "第三次压缩后继续处理。",
                "events": [third_event],
                "compression_event": third_event,
                "context": third_lifecycle,
            },
        )
    turns.append(
        {
            "turn_index": 107 if repeated else 87,
            "phase": "resume-recovery:recover-first-product",
            "user_input": "不要搜索，恢复最开始展示的商品。",
            "final_text": "最开始展示的是 LumenGo 便携露营灯 可充电，价格 89 CNY。",
            "events": [],
            "compression_event": None,
            "context": recovery_lifecycle,
        },
    )
    for name, value in (
        ("manifest.json", manifest),
        ("summary.json", summary),
        ("turns.json", turns),
    ):
        (root / name).write_text(json.dumps(value), encoding="utf-8")


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


def test_natural_context_artifact_is_bounded_and_judge_compatible(
    tmp_path: Path,
) -> None:
    _write_context_artifact(tmp_path)
    suite = load_case_suite(PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml")
    case = next(
        item
        for item in suite.cases
        if item.id == "natural-context-compression-recovery"
    )

    loaded = load_context_compression_artifact(tmp_path, case=case)

    assert len(loaded.evidence.turns) == 2
    trigger = loaded.evidence.turns[0]
    recovery = loaded.evidence.turns[1]
    assert trigger.runtime_signals[0].signal == "context.compressed"
    assert trigger.runtime_signals[0].details["original_turn_index"] == 86
    assert trigger.structured_state.context_lifecycle.before_tokens == 89626
    assert recovery.runtime_signals[0].details["recovery_probe"]["passed"] is True
    assert recovery.final_text.endswith("89 CNY。")
    assert set(loaded.file_sha256) == {
        "manifest.json",
        "summary.json",
        "turns.json",
    }


def test_natural_context_artifact_rejects_forced_soft_limit(tmp_path: Path) -> None:
    _write_context_artifact(tmp_path)
    turns_path = tmp_path / "turns.json"
    turns = json.loads(turns_path.read_text(encoding="utf-8"))
    turns[0]["context"]["soft_limit_tokens"] = 1
    turns_path.write_text(json.dumps(turns), encoding="utf-8")
    case = next(
        item
        for item in load_case_suite(
            PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml"
        ).cases
        if item.id == "natural-context-compression-recovery"
    )

    with pytest.raises(ContextCompressionArtifactError, match="non-default soft limit"):
        load_context_compression_artifact(tmp_path, case=case)


def test_fault_injection_restores_search_usecase_fields() -> None:
    original_embedder = object()
    original_index = object()
    original_reranker = object()
    usecase = SimpleNamespace(
        _embedder=original_embedder,
        _vector_index=original_index,
        _reranker=original_reranker,
    )
    container = SimpleNamespace(
        catalog_searches={"crossshop_reference": usecase},
        product_repo=object(),
    )

    with RubricFaultInjection(  # type: ignore[arg-type]
        container,
        "fault-reranker-fallback",
    ) as injection:
        assert usecase._embedder is not original_embedder
        assert usecase._vector_index is not original_index
        assert usecase._reranker is not original_reranker
        details = injection.details()
        assert details["profile"] == "fault-reranker-fallback"
        assert details["additional_recall_calls"] == 0

    assert usecase._embedder is original_embedder
    assert usecase._vector_index is original_index
    assert usecase._reranker is original_reranker


def test_repeated_context_artifact_requires_three_growing_summaries(
    tmp_path: Path,
) -> None:
    _write_context_artifact(tmp_path, repeated=True)
    case = next(
        item
        for item in load_case_suite(
            PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml"
        ).cases
        if item.id == "natural-context-compression-recovery"
    )

    loaded = load_context_compression_artifact(
        tmp_path,
        case=case,
        require_repeated=True,
    )

    compression_signals = [
        turn.runtime_signals[0]
        for turn in loaded.evidence.turns
        if turn.runtime_signals[0].signal == "context.compressed"
    ]
    assert [item.details["original_turn_index"] for item in compression_signals] == [
        86,
        96,
        106,
    ]
    assert [
        turn.structured_state.context_lifecycle.stage_summary_source_count
        for turn in loaded.evidence.turns[:3]
    ] == [84, 94, 104]


def test_repeated_full_run_artifact_accepts_native_phase_names(tmp_path: Path) -> None:
    _write_context_artifact(tmp_path, repeated=True)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "profile": "full",
            "scenario_sha256": "e" * 64,
            "langfuse_trace_session_hash": "f" * 16,
        },
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    turns_path = tmp_path / "turns.json"
    turns = json.loads(turns_path.read_text(encoding="utf-8"))
    for turn in turns:
        if turn["phase"] == "resume-trigger":
            turn["phase"] = "natural-long-session"
        elif turn["phase"].startswith("resume-recovery:"):
            turn["phase"] = turn["phase"].removeprefix("resume-")
    turns_path.write_text(json.dumps(turns), encoding="utf-8")
    case = next(
        item
        for item in load_case_suite(
            PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml"
        ).cases
        if item.id == "repeated-context-compression-recovery"
    )

    loaded = load_context_compression_artifact(
        tmp_path,
        case=case,
        require_repeated=True,
    )

    assert len(loaded.evidence.turns) == 4


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


def _order_dependency_artifacts() -> CompletedCaseArtifacts:
    artifacts = _completed_artifacts()
    turn = artifacts.evidence.turns[0].model_copy(
        update={
            "tool_calls": [
                ToolCallEvidence(
                    order=1,
                    call_id="create-order-1",
                    agent="main",
                    tool="create_order_tool",
                    status="success",
                    result_summary={
                        "order": {
                            "order_id": "GBX-000123",
                            "status": "CONFIRMED",
                        },
                    },
                ),
            ],
        },
    )
    evidence = artifacts.evidence.model_copy(update={"turns": [turn]})
    return CompletedCaseArtifacts(
        evidence=evidence,
        product_facts=artifacts.product_facts,
        judge_request=artifacts.judge_request,
        raw_judgement=artifacts.raw_judgement,
        result=artifacts.result,
    )


def _order_ownership_case():
    return next(
        item
        for item in load_case_suite(
            PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml",
        ).cases
        if item.id == "order-ownership-idempotency-write-failure"
    )


def test_order_dependency_id_is_bound_into_query_and_mechanically_checked() -> None:
    case = _order_ownership_case()
    dependency = _order_dependency_artifacts()
    bound = bind_dependency_inputs(case, [dependency])
    query_call = (
        dependency.evidence.turns[0]
        .tool_calls[0]
        .model_copy(
            update={
                "tool": "query_order_tool",
                "arguments": {"order_id": "GBX-000123"},
                "status": "error",
            },
        )
    )
    turn = dependency.evidence.turns[0].model_copy(
        update={"user_input": bound.queries[0], "tool_calls": [query_call]},
    )
    evidence = dependency.evidence.model_copy(
        update={"case_id": case.id, "turns": [turn]},
    )

    assert "GBX-000123" in bound.queries[0]
    assert "{{dependency.latest_order_id}}" not in bound.queries[0]
    validate_case_execution_contract(bound, evidence)


def test_order_ownership_case_rejects_missing_real_query_call() -> None:
    case = _order_ownership_case()
    dependency = _order_dependency_artifacts()
    bound = bind_dependency_inputs(case, [dependency])
    turn = dependency.evidence.turns[0].model_copy(
        update={"user_input": bound.queries[0], "tool_calls": []},
    )
    evidence = dependency.evidence.model_copy(
        update={"case_id": case.id, "turns": [turn]},
    )

    with pytest.raises(ValueError, match="must invoke query_order_tool"):
        validate_case_execution_contract(bound, evidence)


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
        category_kb_collection="crossshop_category_kb_v1",
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
        startup_checks={"reference_seed": "ready"},
        evaluation_policy=load_case_suite(
            PROJECT_ROOT / "eval" / "rubric_cases_v3.yaml",
        ).evaluation_policy,
    )
    serialized = json.dumps(manifest)

    assert manifest["schema_version"] == "rubric-run-manifest-v2"
    assert manifest["quality_policy_id"] == "crossshop-quality-v1"
    assert manifest["coverage_policy_id"] == "crossshop-scenario-coverage-v1"
    assert len(manifest["evaluation_policy_sha256"]) == 64
    assert len(manifest["source_tree_sha256"]) == 64
    category_release = PROJECT_ROOT / "data" / "category_insight" / "releases" / "category-insight-v1"
    if category_release.exists():
        assert len(manifest["category_release_manifest_sha256"]) == 64
        assert len(manifest["category_approved_cards_sha256"]) == 64
    else:
        assert manifest["category_release_manifest_sha256"] is None
        assert manifest["category_approved_cards_sha256"] is None
    assert manifest["product_search_startup_checks"] == {"reference_seed": "ready"}
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
