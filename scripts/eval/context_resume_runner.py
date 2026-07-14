"""Resume context-compression evaluation from a cloned persisted session.

The source session is read-only. A deep AgentState copy is saved under a new
evaluation session id, then only unconsumed natural scenario turns are executed
until the next L3 transition. A private, Git-ignored checkpoint keeps the raw
local session handle and scenario cursor so interruption never forces replay.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from agentscope.state import AgentState
from sqlalchemy import select

from app.application.context import group_interaction_units
from app.composition import Container, build_container
from app.infrastructure.model_usage import ModelUsageSample
from app.infrastructure.settings import load_settings
from app.infrastructure.tracing import text_digest
from app.infrastructure.persistence.json_file_stores import JsonFileSessionStore
from app.infrastructure.persistence.sql.repositories import SqlSessionStore
from app.infrastructure.persistence.sql.tables import AgentSessionStateRow
from scripts.eval.context_compression_runner import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_SCENARIO,
    ContextScenario,
    _anchor_fact,
    _checkpoint_progress,
    _context_snapshot,
    _execute_turn,
    _write_json,
    evaluate_recovery_probe,
    load_scenario,
)

_PRIVATE_CHECKPOINT = "resume_checkpoint.private.json"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _compression_history(
    artifact_dir: Path,
) -> tuple[int, list[int], list[dict[str, Any]]]:
    """Return original source size plus bounded prior L3 trigger evidence."""

    summary = _read_json_object(artifact_dir / "summary.json")
    source_turns, _ = _load_source_turns(artifact_dir)
    compression = summary.get("compression")
    compression = compression if isinstance(compression, dict) else {}
    trigger_turns = compression.get("trigger_turns")
    if not isinstance(trigger_turns, list):
        legacy = compression.get("trigger_turn") or compression.get(
            "first_trigger_turn",
        )
        trigger_turns = [legacy] if isinstance(legacy, int) else []
    normalized = sorted(
        {
            int(item)
            for item in trigger_turns
            if isinstance(item, int) and not isinstance(item, bool) and item >= 1
        },
    )
    trigger_rows = [
        turn
        for turn in source_turns
        if int(turn.get("turn_index") or 0) in normalized
        and isinstance(turn.get("compression_event"), dict)
    ]
    if len(trigger_rows) != len(normalized):
        raise ValueError("source artifact does not contain every prior L3 trigger")
    base_source_turns = summary.get("source_turns")
    if not isinstance(base_source_turns, int):
        base_source_turns = normalized[0] - 1 if normalized else 0
    return base_source_turns, normalized, trigger_rows


def natural_scenario_cursor(
    artifact_dir: Path,
    *,
    completed_session_turns: int,
) -> int:
    """Count authored shopping turns without counting recovery probes."""

    summary = _read_json_object(artifact_dir / "summary.json")
    source_turns = summary.get("source_turns")
    added_turns = summary.get("trigger_turns_executed")
    if isinstance(source_turns, int) and isinstance(added_turns, int):
        return min(completed_session_turns, source_turns + added_turns)
    turns, _ = _load_source_turns(artifact_dir)
    authored = sum(
        str(turn.get("phase") or "") in {"natural-long-session", "resume-trigger"}
        for turn in turns
    )
    return min(completed_session_turns, authored or completed_session_turns)


def recovery_query_for_session(
    query: str,
    *,
    authored_turn_index: int,
    source_scenario_cursor: int,
    completed_source_turns: int,
) -> tuple[str, int]:
    """Map an authored shopping-turn number across inserted recovery turns."""

    actual_turn_index = authored_turn_index
    if authored_turn_index > source_scenario_cursor:
        actual_turn_index = completed_source_turns + (
            authored_turn_index - source_scenario_cursor
        )
    return (
        query.replace(
            f"第{authored_turn_index}轮",
            f"第{actual_turn_index}轮",
        ),
        actual_turn_index,
    )


def _usage_samples_from_turns(
    turns: list[dict[str, Any]],
) -> list[ModelUsageSample]:
    fields = {
        "requested_model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "elapsed_seconds",
    }
    samples: list[ModelUsageSample] = []
    for turn in turns:
        for value in turn.get("model_usage") or []:
            if not isinstance(value, dict):
                continue
            samples.append(
                ModelUsageSample(**{name: value[name] for name in fields}),
            )
    return samples


def _write_private_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path / _PRIVATE_CHECKPOINT, payload)


def _match_session_id_by_hash(
    session_ids: list[str],
    expected_hash: str,
) -> str:
    matches = [item for item in session_ids if text_digest(item) == expected_hash]
    if len(matches) != 1:
        raise ValueError(
            "source session hash did not resolve to exactly one local session",
        )
    return matches[0]


async def _resolve_source_session_id(
    container: Container,
    expected_hash: str,
) -> str:
    registry = container.orchestrator._sessions  # noqa: SLF001
    store = registry._session_store  # noqa: SLF001
    if isinstance(store, JsonFileSessionStore):
        session_ids = [path.stem for path in store._dir.glob("*.json")]  # noqa: SLF001
    elif isinstance(store, SqlSessionStore):
        async with store._session_factory() as database:  # noqa: SLF001
            session_ids = list(
                await database.scalars(select(AgentSessionStateRow.session_id)),
            )
    else:
        raise ValueError("configured session store cannot resolve a session hash")
    return _match_session_id_by_hash(session_ids, expected_hash)


def clone_session_state_json(
    source_state_json: str,
    *,
    target_session_id: str,
) -> tuple[str, str]:
    """Clone one AgentState and retarget only session ownership identifiers."""

    state = AgentState.model_validate_json(source_state_json)
    cloned = state.model_copy(deep=True, update={"session_id": target_session_id})
    namespace = cloned.middle_context.get("crossshop_context_v1")
    namespace = namespace if isinstance(namespace, dict) else {}
    cloned.middle_context["crossshop_context_v1"] = namespace
    intent = namespace.get("current_intent")
    intent = intent if isinstance(intent, dict) else {}
    session_context = namespace.get("session_context")
    session_context = session_context if isinstance(session_context, dict) else {}
    session = session_context.get("session")
    session = session if isinstance(session, dict) else {}
    buyer_id = str(intent.get("buyer_id") or session.get("buyer_id") or "").strip()
    if not buyer_id:
        buyer_id = next(
            (
                message.name
                for message in cloned.context
                if message.role == "user" and message.name != "memory_hint"
            ),
            "",
        )
    if not buyer_id:
        raise ValueError("source session has no recoverable buyer identity")
    intent["shopping_session_id"] = target_session_id
    intent["buyer_id"] = buyer_id
    session["shopping_session_id"] = target_session_id
    session["buyer_id"] = buyer_id
    session_context["session"] = session
    namespace["current_intent"] = intent
    namespace["current_turn_events"] = []
    namespace["session_context"] = session_context
    return cloned.model_dump_json(), buyer_id


def completed_turns_from_state_json(state_json: str) -> int:
    """Count the continuous complete buyer-turn prefix in persisted state."""

    state = AgentState.model_validate_json(state_json)
    completed = 0
    for unit in group_interaction_units(state.context):
        if not unit.complete:
            break
        completed += 1
    return completed


def remaining_natural_queries(
    scenario: ContextScenario,
    *,
    completed_turns: int,
    limit: int,
) -> list[str]:
    queries = [
        *scenario.anchor_turns,
        *(query for journey in scenario.long_journeys for query in journey),
    ]
    return queries[completed_turns : completed_turns + limit]


def _load_source_turns(
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], Path]:
    for name in ("turns.json", "turns.partial.json"):
        path = artifact_dir / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)], path
    raise FileNotFoundError("source artifact has no turns.json or turns.partial.json")


def _load_checkpoint_turns(artifact_dir: Path) -> list[dict[str, Any]]:
    path = artifact_dir / "turns.partial.json"
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("turns.partial.json must contain a JSON array")
    return [item for item in value if isinstance(item, dict)]


async def _clone_persisted_session(
    container: Container,
    *,
    source_session_id: str,
    target_session_id: str,
) -> tuple[str, str, int]:
    registry = container.orchestrator._sessions  # noqa: SLF001
    store = registry._session_store  # noqa: SLF001
    source = await store.load(source_session_id)
    if source is None:
        raise ValueError("source session was not found")
    cloned, buyer_id = clone_session_state_json(
        source,
        target_session_id=target_session_id,
    )
    await store.save(target_session_id, cloned)
    return (
        buyer_id,
        hashlib.sha256(source.encode()).hexdigest(),
        completed_turns_from_state_json(source),
    )


def _usage_totals(samples: list[ModelUsageSample]) -> dict[str, Any]:
    input_tokens = sum(item.input_tokens for item in samples)
    cached_tokens = sum(item.cached_input_tokens for item in samples)
    return {
        "model_calls": len(samples),
        "input_tokens": input_tokens,
        "output_tokens": sum(item.output_tokens for item in samples),
        "total_tokens": sum(item.total_tokens for item in samples),
        "cached_input_tokens": cached_tokens,
        "cache_token_hit_ratio": (
            cached_tokens / input_tokens if input_tokens else None
        ),
    }


def classify_resume_status(
    *,
    source_unchanged: bool,
    l2_repair_passed: bool,
    compression_observed: bool,
    recovery_passed: bool,
) -> str:
    """Keep partial L2 success distinct from complete L3 recovery."""

    if not source_unchanged:
        return "SOURCE_MUTATED"
    if not l2_repair_passed:
        return "L2_NOT_ADVANCED"
    if not compression_observed:
        return "L2_PASS_L3_NOT_REACHED"
    if recovery_passed:
        return "L3_RECOVERY_PASS"
    return "L3_RECOVERY_FAIL"


def evaluate_warm_cache_window(
    *,
    turns: list[dict[str, Any]],
    source_turns: list[dict[str, Any]],
    before_freeze_cursor: int,
    expected_turns: int,
    minimum_turn_ratio: float,
    minimum_weighted_ratio: float,
    source_unchanged: bool,
) -> dict[str, Any]:
    """Evaluate a bounded Main-only warm-cache window without prompt text."""

    latest_displayed = next(
        (
            turn.get("displayed_products")
            for turn in reversed(source_turns)
            if turn.get("displayed_products")
        ),
        [],
    )
    allowed_cards = {
        str(item.get("card", {}).get("product_id")): str(
            item.get("card", {}).get("title") or "",
        )
        for item in latest_displayed
        if isinstance(item, Mapping)
        and isinstance(item.get("card"), Mapping)
        and item.get("card", {}).get("product_id")
    }
    ratios: list[float] = []
    cursor = before_freeze_cursor
    cursor_advanced = True
    no_product_search = True
    one_main_call_each = True
    no_unknown_products = True
    known_titles_recalled = bool(allowed_cards)
    for turn in turns:
        usage = turn.get("turn_usage")
        usage = usage if isinstance(usage, Mapping) else {}
        try:
            ratios.append(float(usage.get("token_weighted_cache_hit_ratio")))
        except (TypeError, ValueError):
            ratios.append(0.0)
        one_main_call_each = (
            one_main_call_each
            and int(
                usage.get("model_calls") or 0,
            )
            == 1
        )
        events = turn.get("events")
        events = events if isinstance(events, list) else []
        no_product_search = no_product_search and not any(
            isinstance(event, Mapping)
            and event.get("type") == "tool.invoke"
            and event.get("tool") == "product_search_tool"
            for event in events
        )
        context = turn.get("context")
        context = context if isinstance(context, Mapping) else {}
        next_cursor = int(context.get("freeze_cursor") or 0)
        cursor_advanced = cursor_advanced and next_cursor > cursor
        cursor = next_cursor
        displayed = turn.get("displayed_products")
        displayed = displayed if isinstance(displayed, list) else []
        observed_ids = {
            str(item.get("card", {}).get("product_id"))
            for item in displayed
            if isinstance(item, Mapping) and isinstance(item.get("card"), Mapping)
        }
        no_unknown_products = no_unknown_products and observed_ids.issubset(
            allowed_cards,
        )
        final_text = str(turn.get("final_text") or "")
        known_titles_recalled = known_titles_recalled and all(
            title in final_text for title in allowed_cards.values() if title
        )
    input_tokens = sum(
        int(turn.get("turn_usage", {}).get("input_tokens") or 0) for turn in turns
    )
    cached_tokens = sum(
        int(turn.get("turn_usage", {}).get("cached_input_tokens") or 0)
        for turn in turns
    )
    weighted_ratio = cached_tokens / input_tokens if input_tokens else 0.0
    checks = {
        "turn_count": len(turns) == expected_turns,
        "one_main_call_each": one_main_call_each,
        "no_product_search": no_product_search,
        "freeze_cursor_advanced_each_turn": cursor_advanced,
        "no_unknown_products": no_unknown_products,
        "known_titles_recalled_each_turn": known_titles_recalled,
        "minimum_turn_ratio": bool(ratios) and min(ratios) >= minimum_turn_ratio,
        "weighted_ratio": weighted_ratio >= minimum_weighted_ratio,
        "source_state_unchanged": source_unchanged,
        "no_l3_transition": all(
            turn.get("compression_event") is None for turn in turns
        ),
    }
    return {
        "turn_ratios": ratios,
        "minimum_observed_turn_ratio": min(ratios) if ratios else None,
        "weighted_cache_hit_ratio": weighted_ratio,
        "target_minimum_turn_ratio": minimum_turn_ratio,
        "target_weighted_ratio": minimum_weighted_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    compression = summary["compression"]
    recovery = summary["recovery"]
    lines = [
        "# Resumed Context Compression Evaluation",
        "",
        f"- 源会话轮数：{summary['source_turns']}",
        f"- 克隆后新增自然轮数：{summary['trigger_turns_executed']}",
        f"- 冻结游标：{summary['before_context'].get('freeze_cursor')} → {summary['after_context'].get('freeze_cursor')}",
        f"- 冻结段：{summary['before_context'].get('frozen_segment_count')} → {summary['after_context'].get('frozen_segment_count')}",
        f"- 评测状态：{summary['status']}",
        f"- L2 冻结修复：{'PASS' if summary['l2_repair_passed'] else 'FAIL'}",
        f"- L3 压缩：{'PASS' if compression['observed'] else '未触发'}",
        f"- 压缩后恢复：{'PASS' if recovery['passed'] else ('FAIL' if compression['observed'] else '未执行')}",
        f"- 源状态未改变：{'PASS' if summary['source_state_unchanged'] else 'FAIL'}",
    ]
    warm = summary.get("warm_cache")
    if isinstance(warm, Mapping):
        lines.extend(
            [
                f"- Warm Cache 加权命中率：{float(warm['weighted_cache_hit_ratio']):.2%}",
                f"- Warm Cache 最低单轮命中率：{float(warm['minimum_observed_turn_ratio']):.2%}",
                f"- Warm Cache 稳定性：{'PASS' if warm['passed'] else 'FAIL'}",
            ],
        )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    if settings.semantic_cache_enabled:
        raise RuntimeError("set SEMANTIC_CACHE_ENABLED=0 for prompt-cache evaluation")
    scenario = load_scenario(Path(args.scenario))
    continuing = bool(args.continue_artifact_dir)
    checkpoint: dict[str, Any] = {}
    if continuing:
        run_dir = Path(args.continue_artifact_dir).resolve()
        checkpoint = _read_json_object(run_dir / _PRIVATE_CHECKPOINT)
        source_artifact = Path(str(checkpoint["source_artifact_dir"]))
        anchor_artifact = Path(str(checkpoint["anchor_artifact_dir"]))
        source_session_id = str(checkpoint["source_session_id"])
        target_session_id = str(checkpoint["target_session_id"])
        buyer_id = str(checkpoint["buyer_id"])
        source_hash = str(checkpoint["source_state_sha256"])
        completed_source_turns = int(checkpoint["completed_source_turns"])
        scenario_cursor = int(checkpoint["scenario_cursor"])
        before_context = dict(checkpoint["before_context"])
        max_trigger_turns = int(checkpoint["max_trigger_turns"])
        turns = _load_checkpoint_turns(run_dir)
        all_samples = _usage_samples_from_turns(turns)
    else:
        if (
            not (args.source_session_id or args.source_session_hash)
            or not args.source_artifact_dir
        ):
            raise ValueError(
                "a source session id/hash and source-artifact-dir are required",
            )
        source_artifact = Path(args.source_artifact_dir).resolve()
        anchor_artifact = Path(
            args.anchor_artifact_dir or args.source_artifact_dir,
        ).resolve()
        source_session_id = args.source_session_id or ""
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.artifact_root) / f"resume-{run_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
        target_session_id = f"context-resume-{uuid.uuid4().hex[:12]}"
        buyer_id = ""
        source_hash = ""
        completed_source_turns = 0
        scenario_cursor = 0
        before_context: dict[str, Any] = {}
        max_trigger_turns = args.max_trigger_turns
        turns: list[dict[str, Any]] = []
        all_samples: list[ModelUsageSample] = []

    source_turns, source_turns_path = _load_source_turns(source_artifact)
    anchor_turns, anchor_turns_path = _load_source_turns(anchor_artifact)
    first_anchor = _anchor_fact(anchor_turns)
    if first_anchor is None:
        raise ValueError("anchor artifact has no recoverable anchor product")
    base_source_turns, prior_trigger_turns, prior_trigger_rows = _compression_history(
        source_artifact
    )
    compression_turn = next(
        (
            int(turn["turn_index"])
            for turn in turns
            if isinstance(turn.get("compression_event"), dict)
        ),
        None,
    )

    container = await build_container()
    await container.startup()
    try:
        registry = container.orchestrator._sessions  # noqa: SLF001
        store = registry._session_store  # noqa: SLF001
        if continuing:
            target_state = await store.load(target_session_id)
            if target_state is None:
                raise ValueError("checkpoint target session was not found")
            expected_turns = completed_source_turns + len(turns)
            if completed_turns_from_state_json(target_state) != expected_turns:
                raise ValueError(
                    "checkpoint and persisted target session have diverged; refusing replay",
                )
        else:
            if not source_session_id:
                source_session_id = await _resolve_source_session_id(
                    container,
                    args.source_session_hash,
                )
            (
                buyer_id,
                source_hash,
                completed_source_turns,
            ) = await _clone_persisted_session(
                container,
                source_session_id=source_session_id,
                target_session_id=target_session_id,
            )
            scenario_cursor = natural_scenario_cursor(
                source_artifact,
                completed_session_turns=completed_source_turns,
            )
            before_context = await _context_snapshot(container, target_session_id)
        executed_trigger_turns = sum(
            turn.get("phase") in {"resume-trigger", "warm-cache"} for turn in turns
        )
        remaining_limit = max(0, max_trigger_turns - executed_trigger_turns)
        queries = (
            list(scenario.warm_cache_queries)[executed_trigger_turns:]
            if args.warm_cache
            else remaining_natural_queries(
                scenario,
                completed_turns=scenario_cursor,
                limit=remaining_limit,
            )
        )
        if not queries and compression_turn is None:
            raise ValueError("scenario has no remaining natural query for resume")
        checkpoint_payload = {
            "schema_version": "context-resume-checkpoint-v1",
            "status": "running",
            "source_artifact_dir": str(source_artifact),
            "anchor_artifact_dir": str(anchor_artifact),
            "source_session_id": source_session_id,
            "target_session_id": target_session_id,
            "buyer_id": buyer_id,
            "source_state_sha256": source_hash,
            "completed_source_turns": completed_source_turns,
            "source_scenario_cursor": natural_scenario_cursor(
                source_artifact,
                completed_session_turns=completed_source_turns,
            ),
            "scenario_cursor": scenario_cursor,
            "before_context": before_context,
            "max_trigger_turns": max_trigger_turns,
        }
        _write_private_checkpoint(run_dir, checkpoint_payload)
        for query in queries:
            if compression_turn is not None:
                break
            turn, samples = await _execute_turn(
                container,
                session_id=target_session_id,
                buyer_id=buyer_id,
                locale=scenario.locale,
                currency=scenario.currency,
                turn_index=completed_source_turns + len(turns) + 1,
                query=query,
                phase="warm-cache" if args.warm_cache else "resume-trigger",
            )
            turns.append(turn)
            all_samples.extend(samples)
            if not args.warm_cache:
                scenario_cursor += 1
            if turn["compression_event"] is not None:
                compression_turn = int(turn["turn_index"])
            _checkpoint_progress(
                run_dir,
                turns=turns,
                samples=all_samples,
                compression_turn=compression_turn,
            )
            checkpoint_payload["scenario_cursor"] = scenario_cursor
            _write_private_checkpoint(run_dir, checkpoint_payload)

        recovery_results: list[dict[str, Any]] = []
        all_trigger_turns = [
            *prior_trigger_turns,
            *([compression_turn] if compression_turn is not None else []),
        ]
        if compression_turn is not None and not args.warm_cache:
            for probe in scenario.recovery_probes:
                if probe.minimum_compressions > len(all_trigger_turns):
                    continue
                phase = f"resume-recovery:{probe.id}"
                probe_query, actual_anchor_turn = recovery_query_for_session(
                    probe.query,
                    authored_turn_index=probe.anchor_turn_index,
                    source_scenario_cursor=int(
                        checkpoint_payload["source_scenario_cursor"],
                    ),
                    completed_source_turns=completed_source_turns,
                )
                existing = next(
                    (turn for turn in turns if turn.get("phase") == phase),
                    None,
                )
                if existing is not None:
                    probe_anchor = (
                        first_anchor
                        if probe.anchor_turn_index == 1
                        else _anchor_fact(
                            anchor_turns,
                            turn_index=probe.anchor_turn_index,
                        )
                    )
                    existing_result = evaluate_recovery_probe(
                        existing,
                        probe=probe,
                        anchor=probe_anchor,
                    )
                    if existing_result["passed"]:
                        existing_result["authored_anchor_turn"] = (
                            probe.anchor_turn_index
                        )
                        existing_result["actual_anchor_turn"] = actual_anchor_turn
                        recovery_results.append(existing_result)
                        continue
                    existing["phase"] = f"resume-diagnostic-failed:{probe.id}"
                turn, samples = await _execute_turn(
                    container,
                    session_id=target_session_id,
                    buyer_id=buyer_id,
                    locale=scenario.locale,
                    currency=scenario.currency,
                    turn_index=completed_source_turns + len(turns) + 1,
                    query=probe_query,
                    phase=phase,
                )
                turns.append(turn)
                all_samples.extend(samples)
                probe_anchor = (
                    first_anchor
                    if probe.anchor_turn_index == 1
                    else _anchor_fact(
                        anchor_turns,
                        turn_index=probe.anchor_turn_index,
                    )
                )
                probe_result = evaluate_recovery_probe(
                    turn,
                    probe=probe,
                    anchor=probe_anchor,
                )
                probe_result["authored_anchor_turn"] = probe.anchor_turn_index
                probe_result["actual_anchor_turn"] = actual_anchor_turn
                recovery_results.append(probe_result)
                _checkpoint_progress(
                    run_dir,
                    turns=turns,
                    samples=all_samples,
                    compression_turn=compression_turn,
                )
                _write_private_checkpoint(run_dir, checkpoint_payload)

        after_context = await _context_snapshot(container, target_session_id)
        source_after = await store.load(source_session_id)
        source_unchanged = (
            bool(source_after)
            and hashlib.sha256(
                source_after.encode(),
            ).hexdigest()
            == source_hash
        )
        summary = {
            "source_turns": base_source_turns,
            "resume_source_turns": completed_source_turns,
            "scenario_cursor": scenario_cursor,
            "source_artifact_turns": len(source_turns),
            "trigger_turns_executed": sum(
                turn["phase"] in {"resume-trigger", "warm-cache"} for turn in turns
            ),
            "before_context": before_context,
            "after_context": after_context,
            "usage": _usage_totals(all_samples),
            "compression": {
                "observed": compression_turn is not None,
                "trigger_turn": compression_turn,
                "trigger_turns": all_trigger_turns,
                "observed_count": len(all_trigger_turns),
            },
            "recovery": {
                "probes": recovery_results,
                "passed": bool(recovery_results)
                and all(item["passed"] for item in recovery_results),
            },
            "source_state_unchanged": source_unchanged,
        }
        summary["l2_repair_passed"] = bool(
            source_unchanged
            and after_context.get("freeze_cursor", 0)
            > before_context.get("freeze_cursor", 0)
        )
        summary["l3_recovery_complete"] = bool(
            summary["compression"]["observed"] and summary["recovery"]["passed"]
        )
        if args.warm_cache:
            summary["warm_cache"] = evaluate_warm_cache_window(
                turns=turns,
                source_turns=source_turns,
                before_freeze_cursor=int(before_context.get("freeze_cursor") or 0),
                expected_turns=scenario.targets.warm_cache_turns,
                minimum_turn_ratio=scenario.targets.min_warm_turn_cache_hit_ratio,
                minimum_weighted_ratio=scenario.targets.min_rolling_cache_hit_ratio,
                source_unchanged=source_unchanged,
            )
            summary["status"] = (
                "WARM_CACHE_PASS"
                if summary["warm_cache"]["passed"]
                else "WARM_CACHE_FAIL"
            )
            summary["passed"] = summary["warm_cache"]["passed"]
        else:
            summary["status"] = classify_resume_status(
                source_unchanged=source_unchanged,
                l2_repair_passed=summary["l2_repair_passed"],
                compression_observed=summary["compression"]["observed"],
                recovery_passed=summary["recovery"]["passed"],
            )
            summary["passed"] = summary["status"] == "L3_RECOVERY_PASS"
        manifest = {
            "schema_version": scenario.schema_version,
            "profile": "warm-cache" if args.warm_cache else "resume-clone",
            "application_model": settings.llm_model,
            "context_size": settings.context_size,
            "semantic_cache_enabled": settings.semantic_cache_enabled,
            "source_artifact_sha256": hashlib.sha256(
                source_turns_path.read_bytes(),
            ).hexdigest(),
            "anchor_artifact_sha256": hashlib.sha256(
                anchor_turns_path.read_bytes(),
            ).hexdigest(),
            "source_session_hash": text_digest(source_session_id),
            "target_session_hash": text_digest(target_session_id),
        }
        _write_json(run_dir / "manifest.json", manifest)
        combined_turns = [*prior_trigger_rows, *turns]
        _write_json(run_dir / "turns.json", combined_turns)
        _write_json(run_dir / "summary.json", summary)
        (run_dir / "report.md").write_text(
            _render_report(summary),
            encoding="utf-8",
        )
        checkpoint_payload.update(
            {
                "status": "complete" if summary["passed"] else "resumable",
                "scenario_cursor": scenario_cursor,
            },
        )
        _write_private_checkpoint(run_dir, checkpoint_payload)
        print(f"报告：{run_dir / 'report.md'}", flush=True)
        return 0 if summary["passed"] else 1
    finally:
        await container.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume context evaluation from a cloned persisted session",
    )
    parser.add_argument("--source-session-id")
    parser.add_argument(
        "--source-session-hash",
        help="resolve one existing local session by its privacy-safe artifact hash",
    )
    parser.add_argument("--source-artifact-dir")
    parser.add_argument(
        "--anchor-artifact-dir",
        help="Artifact containing the original recovery anchor; defaults to source artifact",
    )
    parser.add_argument(
        "--max-trigger-turns",
        type=int,
        choices=range(1, 61),
        default=2,
        metavar="1..60",
    )
    parser.add_argument(
        "--continue-artifact-dir",
        help=(
            "continue an interrupted resume run from its private local checkpoint; "
            "source arguments are read from that checkpoint"
        ),
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="Run the fixed five-turn Main-only warm-cache stability window",
    )
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    return parser


def main() -> None:
    try:
        exit_code = asyncio.run(run(build_parser().parse_args()))
    except Exception as err:
        print(
            f"ERROR startup/{type(err).__name__}: 恢复评测未启动；详细异常仅保留在本地受控日志。",
            flush=True,
        )
        raise SystemExit(2) from None
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
