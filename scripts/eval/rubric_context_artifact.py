"""Validate and adapt a natural context-compression run for Rubric v3.

The long session is intentionally executed by ``context_resume_runner`` and
imported here.  A normal Rubric batch can therefore score the already captured
86/87-turn boundary without silently paying for the same long model run again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.eval.rubric_contract import (
    ContextLifecycleEvidence,
    EvaluationEvidence,
    PreferenceStateEvidence,
    RubricCaseSpec,
    RuntimeSignalEvidence,
    StructuredStateEvidence,
    TurnEvidence,
)
from scripts.eval.rubric_evidence import redact_text

_DEFAULT_CONTEXT_SIZE = 128_000
_DEFAULT_SOFT_LIMIT_RATIO = 0.70
_EMPTY_PREFERENCE = PreferenceStateEvidence(count=0, content_hash="0" * 64)


class ContextCompressionArtifactError(ValueError):
    """The supplied artifact cannot prove the authored Rubric scenario."""


@dataclass(frozen=True)
class LoadedContextCompressionArtifact:
    evidence: EvaluationEvidence
    file_sha256: dict[str, str]


def _read_json(path: Path, expected: type) -> Any:
    if not path.is_file():
        raise ContextCompressionArtifactError(f"missing artifact file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        raise ContextCompressionArtifactError(
            f"invalid artifact JSON: {path.name}",
        ) from err
    if not isinstance(value, expected):
        raise ContextCompressionArtifactError(
            f"artifact {path.name} has unexpected top-level type",
        )
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContextCompressionArtifactError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContextCompressionArtifactError(f"{label} must be an object")
    return value


def _lifecycle(value: Any, label: str) -> ContextLifecycleEvidence:
    try:
        return ContextLifecycleEvidence.model_validate(_mapping(value, label))
    except Exception as err:
        raise ContextCompressionArtifactError(
            f"{label} is not valid context lifecycle evidence",
        ) from err


def _safe_state(context: ContextLifecycleEvidence) -> StructuredStateEvidence:
    return StructuredStateEvidence(
        context_lifecycle=context,
        preference_before=_EMPTY_PREFERENCE,
        preference_after=_EMPTY_PREFERENCE,
    )


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    require_repeated: bool,
) -> int:
    _require(
        manifest.get("schema_version") == "context-compression-scenario-v1",
        "artifact schema_version is not context-compression-scenario-v1",
    )
    profile = manifest.get("profile")
    if require_repeated:
        _require(
            profile in {"full", "resume-clone"},
            "repeated compression artifact must use full or resume-clone profile",
        )
    else:
        _require(
            profile == "resume-clone", "artifact must use the resume-clone profile"
        )
    _require(
        manifest.get("semantic_cache_enabled") is False,
        "artifact must disable semantic cache",
    )
    context_size = manifest.get("context_size")
    _require(
        context_size == _DEFAULT_CONTEXT_SIZE,
        "artifact must use the production 128K context size",
    )
    if profile == "resume-clone":
        for name in ("source_artifact_sha256", "anchor_artifact_sha256"):
            _require(_is_hex(manifest.get(name), 64), f"manifest {name} is invalid")
        for name in ("source_session_hash", "target_session_hash"):
            _require(_is_hex(manifest.get(name), 16), f"manifest {name} is invalid")
    else:
        _require(
            _is_hex(manifest.get("scenario_sha256"), 64),
            "manifest scenario_sha256 is invalid",
        )
        _require(
            _is_hex(manifest.get("langfuse_trace_session_hash"), 16),
            "manifest trace session hash is invalid",
        )
    return int(context_size)


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    context_size: int,
    require_repeated: bool,
) -> tuple[list[int], Mapping[str, Any]]:
    _require(summary.get("status") == "L3_RECOVERY_PASS", "L3 recovery did not pass")
    _require(summary.get("passed") is True, "artifact summary is not passed")
    _require(
        summary.get("source_state_unchanged") is True,
        "source session changed during resume evaluation",
    )
    _require(
        summary.get("l2_repair_passed") is True, "L2 lifecycle repair did not pass"
    )
    _require(
        summary.get("l3_recovery_complete") is True,
        "L3 compression and recovery are not complete",
    )
    compression = _mapping(summary.get("compression"), "summary.compression")
    _require(
        compression.get("observed") is True, "natural L3 compression was not observed"
    )
    trigger_turns = compression.get("trigger_turns")
    if not isinstance(trigger_turns, list):
        legacy_turn = compression.get("trigger_turn") or compression.get(
            "first_trigger_turn"
        )
        trigger_turns = [legacy_turn] if isinstance(legacy_turn, int) else []
    _require(
        trigger_turns
        and all(isinstance(item, int) and item >= 1 for item in trigger_turns),
        "compression trigger turns are missing or invalid",
    )
    _require(
        trigger_turns == sorted(set(trigger_turns)),
        "compression trigger turns must be unique and ordered",
    )
    if require_repeated:
        _require(
            len(trigger_turns) >= 3,
            "artifact does not prove at least three natural compressions",
        )
    source_turns = summary.get("source_turns")
    if isinstance(source_turns, int):
        _require(source_turns >= 10, "artifact does not contain a real long session")
        _require(
            trigger_turns[0] == source_turns + 1,
            "compression trigger turn is inconsistent with source session length",
        )
    else:
        _require(
            trigger_turns[0] >= 10,
            "artifact does not contain a real long session",
        )
    recovery = _mapping(summary.get("recovery"), "summary.recovery")
    probes = recovery.get("probes")
    _require(
        recovery.get("passed") is True and isinstance(probes, list) and probes,
        "post-compression recovery probes did not pass",
    )
    for probe in probes:
        probe_map = _mapping(probe, "summary.recovery.probe")
        checks = _mapping(probe_map.get("checks"), "summary.recovery.probe.checks")
        required_checks = {"no_product_search", "stage_summary_verified"}
        fact_checks = {
            "anchor_available",
            "anchor_title_recovered",
            "anchor_price_recovered",
            "anchor_currency_recovered",
        }
        _require(probe_map.get("passed") is True, "a recovery probe failed")
        _require(
            required_checks.issubset(checks)
            and all(checks[name] is True for name in required_checks),
            "recovery probe is missing a verified-summary or no-search check",
        )
        _require(
            (
                fact_checks.issubset(checks)
                and all(checks[name] is True for name in fact_checks)
            )
            or checks.get("required_text_fragments_recovered") is True,
            "recovery probe is missing exact fact or request-fragment proof",
        )
    after = _lifecycle(summary.get("after_context"), "summary.after_context")
    expected_soft_limit = int(context_size * _DEFAULT_SOFT_LIMIT_RATIO)
    _require(
        after.soft_limit_tokens == expected_soft_limit,
        "artifact did not use the default 70% soft limit",
    )
    _require(
        after.compression_action == "l3_stage_summary",
        "summary does not retain successful L3 evidence",
    )
    return [int(item) for item in trigger_turns], recovery


def _validate_turns(
    turns: list[Any],
    *,
    trigger_turns: list[int],
    context_size: int,
    recovery: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    rows = [_mapping(item, "turns[]") for item in turns]
    expected_soft_limit = int(context_size * _DEFAULT_SOFT_LIMIT_RATIO)
    trigger_rows: list[Mapping[str, Any]] = []
    summary_hashes: list[str] = []
    source_counts: list[int] = []
    for trigger_turn in trigger_turns:
        matches = [item for item in rows if item.get("turn_index") == trigger_turn]
        _require(len(matches) == 1, "artifact must contain every trigger turn once")
        trigger = matches[0]
        _require(
            trigger.get("phase") in {"resume-trigger", "natural-long-session"},
            "trigger turn has wrong phase",
        )
        event = _mapping(trigger.get("compression_event"), "trigger.compression_event")
        _require(
            event.get("type") == "context.compressed"
            and event.get("action") == "l3_stage_summary",
            "trigger turn has no successful context.compressed event",
        )
        context = _lifecycle(trigger.get("context"), "trigger.context")
        _require(
            context.soft_limit_tokens == expected_soft_limit,
            "trigger used a non-default soft limit",
        )
        _require(
            context.before_tokens is not None
            and context.before_tokens >= expected_soft_limit
            and context.after_tokens is not None
            and context.after_tokens < context.before_tokens,
            "trigger did not cross the default threshold and reduce context tokens",
        )
        _require(
            event.get("before_tokens") == context.before_tokens
            and event.get("after_tokens") == context.after_tokens
            and event.get("source_segment_count") == context.stage_summary_source_count
            and event.get("summary_hash") == context.stage_summary_hash,
            "context.compressed event does not match the persisted lifecycle state",
        )
        trigger_rows.append(trigger)
        summary_hashes.append(str(context.stage_summary_hash))
        source_counts.append(context.stage_summary_source_count)
    _require(
        len(summary_hashes) == len(set(summary_hashes)),
        "repeated compressions must produce distinct summary hashes",
    )
    _require(
        source_counts == sorted(source_counts)
        and len(source_counts) == len(set(source_counts)),
        "StageSummary source count must grow across repeated compressions",
    )
    if len(trigger_rows) >= 3:
        latest_event = _mapping(
            trigger_rows[-1].get("compression_event"),
            "latest_trigger.compression_event",
        )
        latest = _lifecycle(
            trigger_rows[-1].get("context"),
            "latest_trigger.context",
        )
        _require(
            latest.stage_summary_generation >= 1,
            "latest transition is missing StageSummary generation metadata",
        )
        _require(
            bool(latest.stage_source_manifest_hash)
            and latest_event.get("source_manifest_hash")
            == latest.stage_source_manifest_hash,
            "latest transition is missing a matching audit manifest hash",
        )
        _require(
            0
            < latest.stage_summary_estimated_tokens
            <= latest.stage_summary_hard_limit_tokens,
            "latest model-visible summary is not bounded by its hard limit",
        )
        _require(
            latest.after_tokens is not None
            and latest.after_tokens <= int(context_size * 0.60),
            "latest compression did not reach the 60% post-compression ceiling",
        )
    recovery_rows = [
        item
        for item in rows
        if isinstance(item.get("phase"), str)
        and str(item["phase"]).startswith(("resume-recovery:", "recovery:"))
    ]
    probes = recovery.get("probes")
    _require(
        isinstance(probes, list) and len(recovery_rows) == len(probes),
        "recovery turn count does not match recovery probe count",
    )
    _require(
        all(
            isinstance(item.get("turn_index"), int)
            and item["turn_index"] > trigger_turns[-1]
            for item in recovery_rows
        ),
        "recovery must run after natural compression",
    )
    for row in recovery_rows:
        _require(
            bool(str(row.get("user_input") or "").strip()), "recovery input is missing"
        )
        _require(
            bool(str(row.get("final_text") or "").strip()), "recovery answer is missing"
        )
        recovery_context = _lifecycle(row.get("context"), "recovery.context")
        _require(
            recovery_context.stage_summary_verified,
            "recovery lost verified StageSummary",
        )
        events = row.get("events")
        _require(isinstance(events, list), "recovery events must be a list")
        used_search = any(
            isinstance(item, Mapping)
            and (
                item.get("tool") == "product_search_tool"
                or item.get("agent") == "search_agent"
            )
            for item in events
        )
        _require(not used_search, "recovery reran product search")
    return trigger_rows, recovery_rows


def load_context_compression_artifact(
    artifact_dir: Path,
    *,
    case: RubricCaseSpec,
    require_repeated: bool = False,
) -> LoadedContextCompressionArtifact:
    """Load one resume artifact and emit bounded, Judge-compatible evidence."""

    root = artifact_dir.resolve()
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    turns_path = root / "turns.json"
    manifest = _read_json(manifest_path, dict)
    summary = _read_json(summary_path, dict)
    turns = _read_json(turns_path, list)
    context_size = _validate_manifest(manifest, require_repeated=require_repeated)
    trigger_turns, recovery = _validate_summary(
        summary,
        context_size=context_size,
        require_repeated=require_repeated,
    )
    triggers, recovery_rows = _validate_turns(
        turns,
        trigger_turns=trigger_turns,
        context_size=context_size,
        recovery=recovery,
    )
    file_hashes = {
        "manifest.json": _sha256(manifest_path),
        "summary.json": _sha256(summary_path),
        "turns.json": _sha256(turns_path),
    }
    evidence_turns: list[TurnEvidence] = []
    for index, (trigger, original_turn) in enumerate(
        zip(triggers, trigger_turns, strict=True),
        start=1,
    ):
        trigger_context = _lifecycle(trigger.get("context"), "trigger.context")
        evidence_turns.append(
            TurnEvidence(
                turn_index=index,
                user_input=redact_text(str(trigger.get("user_input"))),
                route="artifact.context-compression.trigger",
                runtime_signals=[
                    RuntimeSignalEvidence(
                        order=1,
                        signal="context.compressed",
                        details={
                            "original_turn_index": original_turn,
                            "context_size": context_size,
                            "soft_limit_ratio": _DEFAULT_SOFT_LIMIT_RATIO,
                            "source_turns": summary.get("source_turns"),
                            "source_state_unchanged": summary.get(
                                "source_state_unchanged",
                                True,
                            ),
                            **dict(
                                _mapping(
                                    trigger.get("compression_event"),
                                    "trigger.compression_event",
                                )
                            ),
                        },
                    ),
                ],
                structured_state=_safe_state(trigger_context),
                final_text=redact_text(str(trigger.get("final_text"))),
            ),
        )
    probe_rows = list(_mapping(recovery, "summary.recovery").get("probes") or [])
    for index, (row, probe) in enumerate(
        zip(recovery_rows, probe_rows, strict=True), start=len(evidence_turns) + 1
    ):
        recovery_context = _lifecycle(row.get("context"), "recovery.context")
        evidence_turns.append(
            TurnEvidence(
                turn_index=index,
                user_input=redact_text(str(row.get("user_input"))),
                route="artifact.context-compression.recovery",
                runtime_signals=[
                    RuntimeSignalEvidence(
                        order=1,
                        signal="harness",
                        details={
                            "original_turn_index": row.get("turn_index"),
                            "recovery_probe": probe,
                            "artifact_file_sha256": file_hashes,
                        },
                    ),
                ],
                structured_state=_safe_state(recovery_context),
                final_text=redact_text(str(row.get("final_text"))),
            ),
        )
    identity = json.dumps(file_hashes, sort_keys=True, separators=(",", ":"))
    evidence = EvaluationEvidence(
        case_id=case.id,
        session_id_hash=hashlib.sha256(identity.encode()).hexdigest()[:16],
        turns=evidence_turns,
    )
    return LoadedContextCompressionArtifact(
        evidence=evidence,
        file_sha256=file_hashes,
    )
