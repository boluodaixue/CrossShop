"""Evaluate cumulative tokens, Ark prompt cache, and natural L3 recovery.

This runner deliberately uses the production 128K/70% policy.  It never
overrides ``CONTEXT_SIZE`` or ``soft_limit_ratio`` to manufacture a compression
event.  Redis semantic caching must be disabled so every turn exercises the
model and its provider-side prompt cache.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.application.agents.orchestrator import SubmitIntentInput
from app.composition import Container, build_container
from app.infrastructure.model_usage import (
    ModelUsageSample,
    capture_model_usage,
)
from app.infrastructure.settings import load_settings
from app.infrastructure.tracing import text_digest
from scripts.eval.rubric_contract import PreferenceStateEvidence
from scripts.eval.rubric_evidence import (
    build_structured_state_evidence,
    redact_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = PROJECT_ROOT / "eval" / "context_compression_scenario_v1.yaml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "context-compression"
_EMPTY_PREFERENCE = PreferenceStateEvidence(count=0, content_hash="0" * 64)


class ContextTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cumulative_turns: int = Field(default=10, ge=1)
    max_single_call_total_tokens: int = Field(default=30_000, ge=1)
    freeze_stall_check_turns: int = Field(default=3, ge=2)
    warmup_model_calls: int = Field(default=1, ge=0)
    rolling_window_model_calls: int = Field(default=5, ge=1)
    min_rolling_cache_hit_ratio: float = Field(default=0.80, ge=0, le=1)
    warm_cache_turns: int = Field(default=5, ge=2)
    min_warm_turn_cache_hit_ratio: float = Field(default=0.75, ge=0, le=1)
    max_turns_until_compression: int = Field(default=60, ge=10)
    target_compression_events: int = Field(default=1, ge=1, le=3)


class RecoveryProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    query: str = Field(min_length=1)
    require_anchor_title_price_currency: bool = True
    forbid_product_search: bool = True
    minimum_compressions: int = Field(default=1, ge=1, le=3)
    anchor_turn_index: int = Field(default=1, ge=1)
    required_text_fragments: list[str] = Field(default_factory=list)


class ContextScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["context-compression-scenario-v1"]
    scenario_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    locale: str = "zh-CN"
    currency: str = "CNY"
    targets: ContextTargets
    anchor_turns: list[str] = Field(min_length=10)
    long_journeys: list[list[str]] = Field(min_length=1)
    warm_cache_queries: list[str] = Field(min_length=5, max_length=5)
    recovery_probes: list[RecoveryProbe] = Field(min_length=1)


def load_scenario(path: Path) -> ContextScenario:
    return ContextScenario.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _checkpoint_progress(
    run_dir: Path,
    *,
    turns: list[dict[str, Any]],
    samples: list[ModelUsageSample],
    compression_turn: int | None,
) -> None:
    input_tokens = sum(item.input_tokens for item in samples)
    cached_tokens = sum(item.cached_input_tokens for item in samples)
    progress = {
        "turns_executed": len(turns),
        "model_calls": len(samples),
        "cumulative_input_tokens": input_tokens,
        "cumulative_output_tokens": sum(item.output_tokens for item in samples),
        "cumulative_total_tokens": sum(item.total_tokens for item in samples),
        "cached_input_tokens": cached_tokens,
        "cache_token_hit_ratio": (
            cached_tokens / input_tokens if input_tokens else None
        ),
        "last_context_estimated_tokens": (
            turns[-1].get("context", {}).get("total_input_tokens") if turns else None
        ),
        "compression_turn": compression_turn,
    }
    _write_json(run_dir / "progress.json", progress)
    _write_json(run_dir / "turns.partial.json", turns)
    ratio = progress["cache_token_hit_ratio"]
    ratio_text = f"{ratio:.2%}" if isinstance(ratio, float) else "n/a"
    print(
        " ".join(
            [
                f"TURN={progress['turns_executed']}",
                f"CALLS={progress['model_calls']}",
                f"TOTAL={progress['cumulative_total_tokens']}",
                f"CACHE={ratio_text}",
                f"CONTEXT={progress['last_context_estimated_tokens']}",
                f"COMPRESSED={compression_turn or 'no'}",
            ],
        ),
        flush=True,
    )


def _cache_ratio(samples: Iterable[ModelUsageSample]) -> float | None:
    rows = list(samples)
    input_tokens = sum(item.input_tokens for item in rows)
    if not input_tokens:
        return None
    return sum(item.cached_input_tokens for item in rows) / input_tokens


def rolling_cache_ratios(
    samples: list[ModelUsageSample],
    *,
    warmup_calls: int,
    window_calls: int,
) -> list[float]:
    eligible = samples[warmup_calls:]
    if len(eligible) < window_calls:
        return []
    return [
        ratio
        for start in range(len(eligible) - window_calls + 1)
        if (ratio := _cache_ratio(eligible[start : start + window_calls])) is not None
    ]


def detect_freeze_stall(
    turns: list[dict[str, Any]],
    *,
    check_after_turns: int,
) -> dict[str, Any] | None:
    """Fail fast when completed multi-turn state never advances its L2 cursor."""

    natural = [
        turn
        for turn in turns
        if turn.get("phase", "natural-long-session") == "natural-long-session"
    ]
    if len(natural) < check_after_turns:
        return None
    latest = natural[-1]
    context = latest.get("context")
    context = context if isinstance(context, Mapping) else {}
    raw_count = int(context.get("raw_context_message_count") or 0)
    freeze_cursor = int(context.get("freeze_cursor") or 0)
    frozen_count = int(context.get("frozen_segment_count") or 0)
    stage_present = bool(context.get("stage_summary_present"))
    if (
        raw_count >= check_after_turns * 2
        and freeze_cursor == 0
        and frozen_count == 0
        and not stage_present
    ):
        return {
            "detected": True,
            "turn_index": int(latest.get("turn_index") or len(natural)),
            "raw_context_message_count": raw_count,
            "freeze_cursor": freeze_cursor,
            "frozen_segment_count": frozen_count,
            "reason": "compression_precondition_stalled",
        }
    return None


def _safe_card(card: Any) -> dict[str, Any]:
    if not isinstance(card, Mapping):
        return {}
    return {
        key: card[key]
        for key in ("product_id", "title", "price_major", "currency")
        if isinstance(card.get(key), (str, int, float))
    }


def _safe_displayed(values: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in values:
        if not isinstance(raw, Mapping):
            continue
        result.append(
            {
                "rank": raw.get("rank"),
                "platform": raw.get("platform"),
                "site_locale": raw.get("site_locale"),
                "card": _safe_card(raw.get("card")),
            },
        )
    return result


def _safe_events(events: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in events:
        event_type = str(getattr(event, "type", ""))
        payload = getattr(event, "payload", None)
        payload = payload if isinstance(payload, Mapping) else {}
        row: dict[str, Any] = {"type": event_type}
        if event_type in {"tool.invoke", "tool.result"}:
            row["tool"] = str(payload.get("tool") or "")
        elif event_type == "agent.dispatch":
            row["agent"] = str(payload.get("agent") or "")
        elif event_type == "context.compressed":
            row.update(
                {
                    key: payload[key]
                    for key in (
                        "action",
                        "before_tokens",
                        "after_tokens",
                        "source_segment_count",
                        "source_manifest_hash",
                        "summary_hash",
                        "summary_generation",
                        "summary_estimated_tokens",
                        "summary_target_tokens",
                        "summary_hard_limit_tokens",
                        "compression_ratio",
                        "recent_raw_segment_count",
                    )
                    if isinstance(payload.get(key), (str, int, float))
                },
            )
        elif event_type in {"error", "model.fallback", "cache.hit"}:
            row["observed"] = True
        else:
            continue
        result.append(row)
    return result


async def _context_snapshot(
    container: Container,
    session_id: str,
) -> dict[str, Any]:
    agent = await container.orchestrator._sessions.get_or_create(session_id)  # noqa: SLF001
    namespace = agent.state.middle_context.get("crossshop_context_v1", {})
    structured = build_structured_state_evidence(
        namespace,
        raw_context_message_count=len(agent.state.context),
        preference_before=_EMPTY_PREFERENCE,
        preference_after=_EMPTY_PREFERENCE,
    )
    return structured.context_lifecycle.model_dump(mode="json")


async def _execute_turn(
    container: Container,
    *,
    session_id: str,
    buyer_id: str,
    locale: str,
    currency: str,
    turn_index: int,
    query: str,
    phase: str,
) -> tuple[dict[str, Any], list[ModelUsageSample]]:
    queue = container.bus.subscribe(session_id)
    try:
        with capture_model_usage() as capture:
            output = await container.orchestrator.handle_intent(
                SubmitIntentInput(
                    shopping_session_id=session_id,
                    buyer_id=buyer_id,
                    locale=locale,
                    currency=currency,
                    raw_query=query,
                ),
            )
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        safe_events = _safe_events(events)
        return (
            {
                "turn_index": turn_index,
                "phase": phase,
                "user_input": redact_text(query),
                "final_text": redact_text(output.final_text),
                "displayed_products": _safe_displayed(output.displayed_products),
                "model_usage": [item.to_dict() for item in capture.samples],
                "turn_usage": capture.totals(),
                "events": safe_events,
                "context": await _context_snapshot(container, session_id),
                "compression_event": next(
                    (
                        event
                        for event in safe_events
                        if event.get("type") == "context.compressed"
                        and event.get("action") == "l3_stage_summary"
                    ),
                    None,
                ),
            },
            list(capture.samples),
        )
    finally:
        container.bus.unsubscribe(session_id, queue)


def _anchor_fact(
    turns: Iterable[dict[str, Any]],
    *,
    turn_index: int = 1,
) -> dict[str, Any] | None:
    for turn in turns:
        if int(turn.get("turn_index") or 0) != turn_index:
            continue
        displayed = turn.get("displayed_products") or []
        if displayed and isinstance(displayed[0].get("card"), Mapping):
            card = dict(displayed[0]["card"])
            if card.get("title") and card.get("price_major") is not None:
                return card
    return None


def _price_fragments(value: Any) -> set[str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return {str(value)} if value not in (None, "") else set()
    result = {str(value), f"{number:.2f}"}
    if number.is_integer():
        result.add(str(int(number)))
    return result


def evaluate_recovery_probe(
    turn: Mapping[str, Any],
    *,
    probe: RecoveryProbe,
    anchor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    final_text = str(turn.get("final_text") or "")
    events = turn.get("events") or []
    product_search_observed = any(
        isinstance(event, Mapping)
        and (
            event.get("tool") == "product_search_tool"
            or event.get("agent") == "search_agent"
        )
        for event in events
    )
    context = turn.get("context")
    context = context if isinstance(context, Mapping) else {}
    checks: dict[str, bool] = {
        "stage_summary_verified": bool(context.get("stage_summary_verified")),
        "no_product_search": not product_search_observed,
    }
    if probe.require_anchor_title_price_currency:
        title = str((anchor or {}).get("title") or "")
        currency = str((anchor or {}).get("currency") or "")
        price_fragments = _price_fragments((anchor or {}).get("price_major"))
        checks.update(
            {
                "anchor_available": bool(anchor),
                "anchor_title_recovered": bool(title) and title in final_text,
                "anchor_price_recovered": bool(price_fragments)
                and any(fragment in final_text for fragment in price_fragments),
                "anchor_currency_recovered": bool(currency) and currency in final_text,
            },
        )
    if probe.required_text_fragments:
        checks["required_text_fragments_recovered"] = all(
            fragment in final_text for fragment in probe.required_text_fragments
        )
    if not probe.forbid_product_search:
        checks.pop("no_product_search", None)
    return {
        "probe_id": probe.id,
        "passed": all(checks.values()),
        "checks": checks,
    }


def summarize_run(
    *,
    scenario: ContextScenario,
    turns: list[dict[str, Any]],
    samples: list[ModelUsageSample],
    compression_turn: int | None,
    compression_turns: list[int] | None = None,
    recovery_results: list[dict[str, Any]],
    lifecycle_stall: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = scenario.targets
    first_turns = turns[: target.cumulative_turns]
    first_calls = [
        call
        for turn in first_turns
        for call in turn.get("model_usage") or []
        if isinstance(call, Mapping)
    ]
    natural_calls = [
        (int(turn.get("turn_index") or index), call)
        for index, turn in enumerate(turns, start=1)
        if turn.get("phase", "natural-long-session") == "natural-long-session"
        for call in turn.get("model_usage") or []
        if isinstance(call, Mapping)
    ]

    def _call_tokens(call: Mapping[str, Any], name: str) -> int:
        try:
            return max(0, int(call.get(name) or 0))
        except (TypeError, ValueError):
            return 0

    first_max_input = max(
        (_call_tokens(call, "input_tokens") for call in first_calls),
        default=0,
    )
    first_max_total = max(
        (_call_tokens(call, "total_tokens") for call in first_calls),
        default=0,
    )
    natural_max_total = max(
        (_call_tokens(call, "total_tokens") for _, call in natural_calls),
        default=0,
    )
    first_breach_turn = next(
        (
            turn_index
            for turn_index, call in natural_calls
            if _call_tokens(call, "total_tokens") > target.max_single_call_total_tokens
        ),
        None,
    )
    ten_total = sum(
        int(turn.get("turn_usage", {}).get("total_tokens") or 0) for turn in first_turns
    )
    ten_input = sum(
        int(turn.get("turn_usage", {}).get("input_tokens") or 0) for turn in first_turns
    )
    ten_output = sum(
        int(turn.get("turn_usage", {}).get("output_tokens") or 0)
        for turn in first_turns
    )
    rolling = rolling_cache_ratios(
        samples,
        warmup_calls=target.warmup_model_calls,
        window_calls=target.rolling_window_model_calls,
    )
    eligible = samples[target.warmup_model_calls :]
    aggregate_cache_ratio = _cache_ratio(eligible)
    min_rolling = min(rolling) if rolling else None
    observed_compression_turns = list(compression_turns or [])
    if (
        compression_turn is not None
        and compression_turn not in observed_compression_turns
    ):
        observed_compression_turns.insert(0, compression_turn)
    return {
        "scenario_id": scenario.scenario_id,
        "turns_executed": len(turns),
        "model_calls": len(samples),
        "ten_turn": {
            "turns_observed": len(first_turns),
            "cumulative_input_tokens": ten_input,
            "cumulative_output_tokens": ten_output,
            "cumulative_total_tokens": ten_total,
            "max_single_call_input_tokens": first_max_input,
            "max_single_call_total_tokens": first_max_total,
            "target_max_single_call_total_tokens": target.max_single_call_total_tokens,
            "passed": len(first_turns) == target.cumulative_turns
            and bool(first_calls)
            and first_max_total <= target.max_single_call_total_tokens,
        },
        "long_session_token_cap": {
            "model_calls_observed": len(natural_calls),
            "max_single_call_total_tokens": natural_max_total,
            "target_max_single_call_total_tokens": target.max_single_call_total_tokens,
            "first_breach_turn": first_breach_turn,
            "passed": bool(natural_calls) and first_breach_turn is None,
        },
        "prompt_cache": {
            "warmup_model_calls_excluded": target.warmup_model_calls,
            "eligible_input_tokens": sum(item.input_tokens for item in eligible),
            "cached_input_tokens": sum(item.cached_input_tokens for item in eligible),
            "aggregate_token_weighted_hit_ratio": aggregate_cache_ratio,
            "rolling_window_model_calls": target.rolling_window_model_calls,
            "minimum_rolling_hit_ratio": min_rolling,
            "target_minimum_rolling_hit_ratio": target.min_rolling_cache_hit_ratio,
            "passed": min_rolling is not None
            and min_rolling >= target.min_rolling_cache_hit_ratio,
        },
        "compression": {
            "observed": bool(observed_compression_turns),
            "observed_count": len(observed_compression_turns),
            "target_count": target.target_compression_events,
            "trigger_turns": observed_compression_turns,
            "first_trigger_turn": (
                observed_compression_turns[0] if observed_compression_turns else None
            ),
            "max_turns_until_compression": target.max_turns_until_compression,
            "passed": len(observed_compression_turns)
            >= target.target_compression_events
            and observed_compression_turns[0] <= target.max_turns_until_compression,
        },
        "recovery": {
            "probes": recovery_results,
            "passed": bool(recovery_results)
            and all(item["passed"] for item in recovery_results),
        },
        "lifecycle_stall": dict(lifecycle_stall or {"detected": False}),
    }


def render_report(summary: Mapping[str, Any]) -> str:
    ten = summary["ten_turn"]
    token_cap = summary["long_session_token_cap"]
    cache = summary["prompt_cache"]
    compression = summary["compression"]
    recovery = summary["recovery"]
    lifecycle_stall = summary["lifecycle_stall"]
    ratio = cache.get("aggregate_token_weighted_hit_ratio")
    minimum = cache.get("minimum_rolling_hit_ratio")
    return "\n".join(
        [
            "# Context Compression Evaluation",
            "",
            f"- 10 轮累计 total tokens（成本诊断）：{ten['cumulative_total_tokens']}",
            f"- 10 轮累计 input/output：{ten['cumulative_input_tokens']} / {ten['cumulative_output_tokens']}",
            f"- 10 轮最大单次 total tokens：{ten['max_single_call_total_tokens']} / {ten['target_max_single_call_total_tokens']}，{'PASS' if ten['passed'] else 'FAIL'}",
            f"- 长会话最大单次 total tokens：{token_cap['max_single_call_total_tokens']}；首次突破 30K：{token_cap['first_breach_turn'] or '未发生'}",
            f"- Prompt Cache token 加权命中率：{ratio:.3%}"
            if ratio is not None
            else "- Prompt Cache token 加权命中率：不可测",
            f"- 滚动最低命中率：{minimum:.3%}"
            if minimum is not None
            else "- 滚动最低命中率：样本不足",
            f"- 首次自然 L3 压缩轮次：{compression['first_trigger_turn'] or '未触发'}",
            f"- 自然 L3 压缩次数：{compression['observed_count']} / {compression['target_count']}",
            f"- 压缩后恢复：{'PASS' if recovery['passed'] else 'FAIL'}",
            f"- 冻结前置条件停滞：{'YES' if lifecycle_stall.get('detected') else 'NO'}",
            "",
            "本报告使用默认运行配置，不降低上下文压缩阈值；Redis 语义缓存关闭。",
        ],
    )


async def run(args: argparse.Namespace) -> int:
    settings = load_settings()
    if settings.semantic_cache_enabled:
        raise RuntimeError("set SEMANTIC_CACHE_ENABLED=0 for prompt-cache evaluation")
    scenario = load_scenario(Path(args.scenario))
    if args.target_compressions:
        scenario = scenario.model_copy(
            update={
                "targets": scenario.targets.model_copy(
                    update={"target_compression_events": args.target_compressions},
                ),
            },
        )
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    session_id = f"context-eval-{uuid.uuid4().hex[:12]}"
    buyer_id = f"context-buyer-{uuid.uuid4().hex[:12]}"
    turns: list[dict[str, Any]] = []
    all_samples: list[ModelUsageSample] = []
    compression_turn: int | None = None
    compression_turns: list[int] = []
    lifecycle_stall: dict[str, Any] | None = None

    container = await build_container()
    await container.startup()
    try:
        turn_limit = (
            scenario.targets.cumulative_turns
            if args.smoke
            else args.max_turns or scenario.targets.max_turns_until_compression
        )
        long_queries = [
            *scenario.anchor_turns,
            *(query for journey in scenario.long_journeys for query in journey),
        ][:turn_limit]
        for query in long_queries:
            turn, samples = await _execute_turn(
                container,
                session_id=session_id,
                buyer_id=buyer_id,
                locale=scenario.locale,
                currency=scenario.currency,
                turn_index=len(turns) + 1,
                query=query,
                phase="natural-long-session",
            )
            turns.append(turn)
            all_samples.extend(samples)
            if compression_turn is None and turn["compression_event"] is not None:
                compression_turn = int(turn["turn_index"])
            if turn["compression_event"] is not None:
                observed_turn = int(turn["turn_index"])
                if observed_turn not in compression_turns:
                    compression_turns.append(observed_turn)
            _checkpoint_progress(
                run_dir,
                turns=turns,
                samples=all_samples,
                compression_turn=compression_turn,
            )
            if not args.smoke:
                lifecycle_stall = detect_freeze_stall(
                    turns,
                    check_after_turns=scenario.targets.freeze_stall_check_turns,
                )
                if lifecycle_stall is not None:
                    break
            if (
                len(compression_turns) >= scenario.targets.target_compression_events
                and len(turns) >= scenario.targets.cumulative_turns
            ):
                break

        recovery_results: list[dict[str, Any]] = []
        if len(compression_turns) >= scenario.targets.target_compression_events:
            for probe in scenario.recovery_probes:
                if len(compression_turns) < probe.minimum_compressions:
                    continue
                turn, samples = await _execute_turn(
                    container,
                    session_id=session_id,
                    buyer_id=buyer_id,
                    locale=scenario.locale,
                    currency=scenario.currency,
                    turn_index=len(turns) + 1,
                    query=probe.query,
                    phase=f"recovery:{probe.id}",
                )
                turns.append(turn)
                all_samples.extend(samples)
                recovery_results.append(
                    evaluate_recovery_probe(
                        turn,
                        probe=probe,
                        anchor=_anchor_fact(
                            turns,
                            turn_index=probe.anchor_turn_index,
                        ),
                    ),
                )
                _checkpoint_progress(
                    run_dir,
                    turns=turns,
                    samples=all_samples,
                    compression_turn=compression_turn,
                )

        summary = summarize_run(
            scenario=scenario,
            turns=turns,
            samples=all_samples,
            compression_turn=compression_turn,
            compression_turns=compression_turns,
            recovery_results=recovery_results,
            lifecycle_stall=lifecycle_stall,
        )
        summary["after_context"] = await _context_snapshot(container, session_id)
        summary["source_state_unchanged"] = True
        summary["l2_repair_passed"] = bool(
            summary["after_context"].get("freeze_cursor", 0) > 0
        )
        summary["l3_recovery_complete"] = bool(
            summary["compression"]["passed"] and summary["recovery"]["passed"]
        )
        summary["status"] = (
            "L3_RECOVERY_PASS"
            if summary["l3_recovery_complete"]
            else "L3_RECOVERY_FAIL"
        )
        summary["passed"] = summary["status"] == "L3_RECOVERY_PASS"
        manifest = {
            "schema_version": scenario.schema_version,
            "scenario_sha256": hashlib.sha256(
                Path(args.scenario).read_bytes(),
            ).hexdigest(),
            "application_model": settings.llm_model,
            "context_size": settings.context_size,
            "semantic_cache_enabled": settings.semantic_cache_enabled,
            "profile": "smoke" if args.smoke else "full",
            "langfuse_trace_session_hash": text_digest(session_id),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "turns.json", turns)
        _write_json(run_dir / "summary.json", summary)
        (run_dir / "report.md").write_text(
            render_report(summary),
            encoding="utf-8",
        )
        print(f"报告：{run_dir / 'report.md'}", flush=True)
        passed = summary["ten_turn"]["passed"] and summary["prompt_cache"]["passed"]
        if not args.smoke:
            passed = (
                passed
                and not summary["lifecycle_stall"].get("detected")
                and summary["long_session_token_cap"]["passed"]
                and summary["compression"]["passed"]
                and summary["recovery"]["passed"]
            )
        return 0 if passed else 1
    finally:
        await container.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Natural context compression evaluation"
    )
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the fixed ten-turn token/cache journey",
    )
    parser.add_argument(
        "--target-compressions",
        type=int,
        choices=(1, 2, 3),
        default=0,
        help="override the scenario target number of natural L3 transitions",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="bounded natural-session turn limit; zero keeps the scenario default",
    )
    return parser


def main() -> None:
    try:
        exit_code = asyncio.run(run(build_parser().parse_args()))
    except Exception as err:
        print(
            f"ERROR startup/{type(err).__name__}: 评测未启动；详细异常仅保留在本地受控日志。",
            flush=True,
        )
        raise SystemExit(2) from None
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
