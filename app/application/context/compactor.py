"""Deterministic L2 freezing and L3 summarization for AgentScope messages."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from agentscope.message import Msg, TextBlock, ToolResultBlock

from .lifecycle import InteractionUnit
from .models import FrozenSegment, StageSummary, canonical_json

_MAX_TEXT = 2_000
_MAX_REFS = 20
_MAX_JOURNEY_REQUEST = 320
_MAX_JOURNEY_FOCUS = 160
_MAX_JOURNEY_DECISION = 200
_MAX_ANCHOR_TITLE = 180
_STRUCTURED_OUTPUT_TOOLS = {"generatestructuredoutput"}
_SEARCH_TOOLS = {"product_search_tool"}
_JOURNEY_START = re.compile(
    r"(?:switch\s+to|改看|切换到|重新看|再看|继续看|最后看|回到|"
    r"(?:帮我|现在|先|实际)?找\s*\d*\s*(?:款|件|条装)?|推荐\s*\d*\s*(?:款|件)?)",
    re.IGNORECASE,
)
_NOISE_REQUEST = re.compile(
    r"(?:不要重新搜索|仍然?不要搜索|不要增加(?:新)?商品|不要增加候选|"
    r"只依据已有(?:商品)?信息|只基于已展示信息回答|一句话回答|"
    r"用表格|用简短清单|保留准确商品名称和价格|保留准确名称和价格(?:与币种)?)",
    re.IGNORECASE,
)
_PLATFORM_PATTERNS = (
    (re.compile(r"Amazon\s*美国站", re.IGNORECASE), ("amazon", "us")),
    (re.compile(r"Amazon\s*日本站", re.IGNORECASE), ("amazon", "jp")),
    (re.compile(r"示例平台", re.IGNORECASE), ("reference_seed", "cn")),
    (re.compile(r"京东|JD", re.IGNORECASE), ("jd", "cn")),
    (re.compile(r"CrossShop(?:\s*参考目录)?", re.IGNORECASE), ("crossshop_reference", None)),
)
_BUDGET_VALUE = re.compile(
    r"(?:预算|上限|最多|不超过|以内)\s*(?:为|是|[:=])?\s*"
    r"[¥￥$€£]?\s*(\d+(?:\.\d+)?)\s*(元|美元|日元|CNY|USD|JPY)?",
    re.IGNORECASE,
)
_STATE_REPLACEMENT = re.compile(
    r"(?:预算|上限|改成|换成|replace\s+with|排除|取消)",
    re.IGNORECASE,
)
_RECOVERY_PROBE = re.compile(
    r"(?:本次会话最开始展示|本次会话第\s*\d+\s*轮开始寻找|"
    r"准确告诉我本次会话|请说明本次会话第\s*\d+\s*轮)",
    re.IGNORECASE,
)
_PRODUCT_LINE = re.compile(
    r"(?m)^\s*\d+[\.、]\s*([^|\n]{1,80})\s*\|\s*([^\n]{1,240})\n"
    r"(?:价格\s*[:：]?\s*)?([¥￥$€£]?\s*\d+(?:\.\d+)?(?:\s*[–—~-]\s*\d+(?:\.\d+)?)?)"
    r"\s*(CNY|USD|JPY|EUR|GBP|人民币|美元|日元)?",
    re.IGNORECASE,
)


class SummaryBudgetError(RuntimeError):
    """Raised when a structurally valid summary cannot meet its hard limit."""


def _message_text(message: Msg) -> str:
    return (message.get_text_content() or "")[:_MAX_TEXT]


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if depth > 3:
        return None
    if isinstance(value, str):
        return value[:512]
    if isinstance(value, Mapping):
        return {
            str(key): child_bounded
            for key, child in list(value.items())[:32]
            if (child_bounded := _bounded(child, depth=depth + 1)) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_bounded(child, depth=depth + 1) for child in value[:20]]
    return value


def _result_text(result: ToolResultBlock) -> str:
    if isinstance(result.output, str):
        return result.output
    return "".join(
        block.text for block in result.output if isinstance(block, TextBlock)
    )


def _result_summary(result: ToolResultBlock) -> tuple[str, int | None, list[str]]:
    text = _result_text(result).strip()
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    status = str(getattr(result.state, "value", result.state))
    count: int | None = None
    refs: list[str] = []
    if isinstance(parsed, Mapping):
        if parsed.get("error"):
            status = "error"
        for key in ("returned_count", "hit_count", "count"):
            if isinstance(parsed.get(key), int):
                count = int(parsed[key])
                break
        for key in ("hits", "filtered_out", "items", "lines"):
            rows = parsed.get(key)
            if not isinstance(rows, list):
                continue
            if count is None and key in {"hits", "items", "lines"}:
                count = len(rows)
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                for ref_key in ("product_id", "sku_id", "order_id", "confirmation_id"):
                    if row.get(ref_key) not in (None, ""):
                        refs.append(str(row[ref_key]))
            break
        for ref_key in ("product_id", "sku_id", "order_id", "confirmation_id"):
            if parsed.get(ref_key) not in (None, ""):
                refs.append(str(parsed[ref_key]))
    return status, count, list(dict.fromkeys(refs))[:_MAX_REFS]


def compact_interaction(
    unit: InteractionUnit, *, context_revision: int
) -> FrozenSegment:
    if not unit.complete or unit.final_answer is None:
        raise ValueError("only protocol-complete interaction units can be frozen")
    tools: list[dict[str, Any]] = []
    for interaction in unit.tools:
        status, returned_count, refs = _result_summary(interaction.result)
        fact: dict[str, Any] = {
            "tool": interaction.name,
            "tool_call_id": interaction.tool_call_id,
            "args": _bounded(interaction.args),
            "status": status,
        }
        if returned_count is not None:
            fact["returned_count"] = returned_count
        if refs:
            fact["refs"] = refs
        tools.append(fact)
    content = canonical_json(
        {
            "user": _message_text(unit.human),
            "tools": tools,
            "assistant": _message_text(unit.final_answer),
            **({"outcome": unit.terminal_status} if unit.terminal_status != "completed" else {}),
        },
    )
    return FrozenSegment(
        segment_id=f"frozen-{unit.start}-{unit.end}",
        content=content,
        source_message_start=unit.start,
        source_message_end=unit.end,
        context_revision=context_revision,
    )


def freeze_settled_units(
    messages: Sequence[Msg],
    units: Sequence[InteractionUnit],
    *,
    existing: Sequence[FrozenSegment] = (),
    freeze_cursor: int = 0,
    context_revision: int = 0,
) -> tuple[list[FrozenSegment], int]:
    del messages
    segments = list(existing)
    cursor = max(0, freeze_cursor)
    ranges = {
        (item.source_message_start, item.source_message_end): item for item in segments
    }
    for unit in units:
        if unit.end <= cursor:
            continue
        if unit.start != cursor:
            break
        key = (unit.start, unit.end)
        if key not in ranges:
            segment = compact_interaction(unit, context_revision=context_revision)
            segments.append(segment)
            ranges[key] = segment
        cursor = unit.end
    return segments, max(freeze_cursor, cursor)


def _semantic_text(value: str, *, limit: int) -> str:
    """Remove turn-control prose while retaining the buyer's business meaning."""

    text = _NOISE_REQUEST.sub("", value)
    text = re.sub(r"\s+", " ", text).strip(" ，。；;:\n")
    text = re.sub(r"[，；;]{2,}", "；", text)
    return text[:limit]


def _platform_from_request(request: str) -> tuple[str | None, str | None]:
    for pattern, result in _PLATFORM_PATTERNS:
        if pattern.search(request):
            return result
    return None, None


def _budget_from_request(request: str) -> tuple[float | None, str | None]:
    match = _BUDGET_VALUE.search(request)
    if match is None:
        match = re.search(
            r"[¥￥$€£]?\s*(\d+(?:\.\d+)?)\s*(元|美元|日元|CNY|USD|JPY)?\s*以内",
            request,
            re.IGNORECASE,
        )
    if match is None:
        return None, None
    amount = float(match.group(1))
    unit = str(match.group(2) or "").upper()
    currency = {
        "元": "CNY",
        "人民币": "CNY",
        "美元": "USD",
        "日元": "JPY",
    }.get(unit, unit or None)
    return amount, currency


def _anchor_product(assistant: str) -> dict[str, Any] | None:
    match = _PRODUCT_LINE.search(assistant)
    if match is None:
        return None
    price = re.sub(r"\s+", "", match.group(3))
    currency = str(match.group(4) or "").upper()
    currency = {"人民币": "CNY", "美元": "USD", "日元": "JPY"}.get(
        currency,
        currency,
    )
    return {
        "product_id": match.group(1).strip(),
        "title": match.group(2).strip()[:_MAX_ANCHOR_TITLE],
        "price": price,
        "currency": currency or None,
    }


def _decision_summary(assistant: str) -> str:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？!?])\s*|\n+", assistant)
        if item.strip()
    ]
    for sentence in sentences:
        if re.search(r"结论|推荐|更适合|优先|排序", sentence):
            return sentence[:_MAX_JOURNEY_DECISION]
    return ""


def _is_structured_output_fact(fact: Mapping[str, Any]) -> bool:
    tool = re.sub(r"[^a-z]", "", str(fact.get("tool", "")).lower())
    return tool in _STRUCTURED_OUTPUT_TOOLS


def _is_search_fact(fact: Mapping[str, Any]) -> bool:
    tool = str(fact.get("tool", ""))
    if tool in _SEARCH_TOOLS:
        return True
    args = fact.get("args")
    return bool(
        tool == "task_dispatch"
        and isinstance(args, Mapping)
        and str(args.get("subagent_type", "")) == "search_agent"
    )


def _compact_summary_fact(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Keep authoritative tool state, never duplicated final-answer prose."""

    if _is_structured_output_fact(raw):
        return None
    tool = str(raw.get("tool", ""))
    args = raw.get("args")
    args = args if isinstance(args, Mapping) else {}
    allowed_args = {
        "normalized_query",
        "category",
        "platform",
        "site_locale",
        "ship_to",
        "target_currency",
        "price_max_major",
        "top_k",
        "order_id",
        "product_id",
        "sku_id",
        "quantity",
        "reason",
        "subagent_type",
    }
    compact_args = {
        str(key): _bounded(value)
        for key, value in args.items()
        if str(key) in allowed_args and value not in (None, "")
    }
    fact: dict[str, Any] = {
        "tool": tool,
        "status": str(raw.get("status", "")),
    }
    if compact_args:
        fact["args"] = compact_args
    if isinstance(raw.get("returned_count"), int):
        fact["returned_count"] = int(raw["returned_count"])
    refs = raw.get("refs")
    if isinstance(refs, (list, tuple)):
        fact["refs"] = [str(item) for item in refs[:_MAX_REFS]]
    call_ids = _tool_call_ids(raw)
    if call_ids:
        fact["tool_call_ids"] = call_ids
    return fact


def _fact_search_fields(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    refs: list[str] = []
    for fact in facts:
        if not _is_search_fact(fact):
            continue
        args = fact.get("args")
        if isinstance(args, Mapping):
            for key in (
                "normalized_query",
                "category",
                "platform",
                "site_locale",
                "target_currency",
                "price_max_major",
            ):
                if args.get(key) not in (None, ""):
                    result[key] = _bounded(args[key])
        raw_refs = fact.get("refs")
        if isinstance(raw_refs, (list, tuple)):
            refs.extend(str(item) for item in raw_refs if item not in (None, ""))
    if refs:
        result["product_refs"] = list(dict.fromkeys(refs))[:5]
    return result


def _new_journey(
    request: str,
    *,
    journey_index: int,
    source_message_start: int | None,
    authored_turn_start: int | None,
    facts: Sequence[Mapping[str, Any]],
    assistant: str,
) -> dict[str, Any]:
    platform, site_locale = _platform_from_request(request)
    budget, currency = _budget_from_request(request)
    search = _fact_search_fields(facts)
    journey: dict[str, Any] = {
        "journey_id": f"journey-{journey_index + 1}",
        "request": _semantic_text(request, limit=_MAX_JOURNEY_REQUEST),
    }
    if source_message_start is not None:
        journey["source_message_start"] = source_message_start
    if authored_turn_start is not None:
        journey["authored_turn_start"] = authored_turn_start
        journey["authored_turn_end"] = authored_turn_start
    for key in ("normalized_query", "category"):
        if search.get(key) not in (None, ""):
            journey[key] = search[key]
    effective_platform = search.get("platform") or platform
    effective_site = search.get("site_locale") or site_locale
    if effective_platform:
        journey["platform"] = effective_platform
    if effective_site:
        journey["site_locale"] = effective_site
    effective_budget = search.get("price_max_major")
    if effective_budget in (None, ""):
        effective_budget = budget
    if effective_budget is not None:
        journey["budget_max_major"] = effective_budget
    effective_currency = search.get("target_currency") or currency
    if effective_currency:
        journey["currency"] = effective_currency
    if search.get("product_refs"):
        journey["product_refs"] = search["product_refs"]
    anchor = _anchor_product(assistant)
    if anchor:
        journey["anchor_product"] = anchor
    decision = _decision_summary(assistant)
    if decision:
        journey["decision"] = decision
    return journey


def _merge_journey_turn(
    journeys: list[dict[str, Any]],
    request: str,
    *,
    source_message_start: int | None,
    authored_turn: int | None,
    facts: Sequence[Mapping[str, Any]],
    assistant: str,
    force_new: bool = False,
) -> None:
    starts_journey = force_new or bool(_JOURNEY_START.search(request)) or not journeys
    if starts_journey:
        journeys.append(
            _new_journey(
                request,
                journey_index=len(journeys),
                source_message_start=source_message_start,
                authored_turn_start=authored_turn,
                facts=facts,
                assistant=assistant,
            )
        )
        return
    current = journeys[-1]
    if authored_turn is not None:
        current["authored_turn_end"] = authored_turn
    focus = _semantic_text(request, limit=_MAX_JOURNEY_FOCUS)
    budget, currency = _budget_from_request(request)
    platform, site_locale = _platform_from_request(request)
    if _STATE_REPLACEMENT.search(request):
        # A state-changing request is represented once as the journey's latest
        # effective request.  Keeping both old and new budget/product prose in
        # the model view would reintroduce the conflict this layer must resolve.
        current["request"] = _semantic_text(
            request,
            limit=_MAX_JOURNEY_REQUEST,
        )
    if budget is not None:
        current["budget_max_major"] = budget
    if currency:
        current["currency"] = currency
    if platform:
        current["platform"] = platform
    if site_locale:
        current["site_locale"] = site_locale
    if focus:
        focuses = list(current.get("focus") or [])
        if focus not in focuses:
            focuses.append(focus)
        current["focus"] = focuses
    search = _fact_search_fields(facts)
    if search.get("product_refs"):
        current["product_refs"] = search["product_refs"]
    anchor = _anchor_product(assistant)
    if anchor:
        current["anchor_product"] = anchor
    decision = _decision_summary(assistant)
    if decision:
        current["decision"] = decision


def _legacy_journeys(summary: StageSummary) -> list[dict[str, Any]]:
    """Upgrade v1/v2 extractive lists into semantic shopping journeys."""

    journeys: list[dict[str, Any]] = []
    outcomes = list(summary.assistant_outcomes)
    for index, request in enumerate(summary.user_requests):
        if _RECOVERY_PROBE.search(request):
            continue
        assistant = outcomes[index] if index < len(outcomes) else ""
        _merge_journey_turn(
            journeys,
            request,
            source_message_start=index * 2,
            # Extractive legacy lists do not preserve each selected request's
            # original turn number.  Production migration rebuilds from L2;
            # this fallback must omit the number instead of inventing one.
            authored_turn=None,
            facts=(),
            assistant=assistant,
        )
    # v1/v2 stored requests, facts, and outcomes as independent lists.  Their
    # positions are not a relational key: one turn may emit several tools or
    # none.  Positional zipping corrupted platform/budget across journeys in a
    # real 100-turn migration, so legacy facts remain in ``tool_facts`` and are
    # never guessed onto a journey.  New v3 segments attach facts while the
    # original interaction boundary is still available.
    return journeys


def summarize_frozen_segments(
    segments: Sequence[FrozenSegment],
    *,
    existing: StageSummary | Mapping[str, Any] | None = None,
    target_tokens: int = 8_000,
    hard_limit_tokens: int = 12_000,
) -> StageSummary:
    """Recompress old memory into one semantic shopping-journey summary.

    Every journey keeps its core request, platform/budget, verified references,
    follow-up focus, anchor product, and latest decision.  Unlike the legacy
    v2 implementation, this function never samples whole requests merely to
    hit a token target.  If semantic aggregation cannot stay below the hard
    limit, callers retain the previous valid summary and uncovered raw context.
    """

    if target_tokens <= 0 or hard_limit_tokens <= 0:
        raise ValueError("summary token limits must be positive")
    if target_tokens > hard_limit_tokens:
        raise ValueError("summary target cannot exceed hard limit")
    current = (
        existing
        if isinstance(existing, StageSummary)
        else StageSummary.from_dict(existing)
        if isinstance(existing, Mapping)
        else None
    )
    sources = list(current.source_segments if current else ())
    journeys = (
        [dict(item) for item in current.shopping_journeys]
        if current and current.shopping_journeys
        else _legacy_journeys(current)
        if current
        else []
    )
    authored_turn: int | None = (
        None
        if current and not current.shopping_journeys
        else max(
            (
                int(
                    item.get("authored_turn_end")
                    or item.get("authored_turn_start")
                    or 0
                )
                for item in journeys
            ),
            default=0,
        )
    )
    facts: list[dict[str, Any]] = []
    ignored_structured_outputs = 0
    if current:
        for raw_fact in current.tool_facts:
            compact = _compact_summary_fact(raw_fact)
            if compact is None:
                ignored_structured_outputs += 1
                continue
            facts.append(compact)
    known = {item["segment_id"]: item["content_hash"] for item in sources}
    fact_indexes = {_tool_state_key(item): index for index, item in enumerate(facts)}
    starts = (
        [current.source_message_start]
        if current and current.source_message_start is not None
        else []
    )
    ends = (
        [current.source_message_end]
        if current and current.source_message_end is not None
        else []
    )
    revisions = [current.context_revision] if current else []
    added_source = False

    for segment in segments:
        known_hash = known.get(segment.segment_id)
        if known_hash is not None:
            if known_hash != segment.content_hash:
                raise ValueError(f"frozen segment hash changed: {segment.segment_id}")
            continue
        try:
            content = json.loads(segment.content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid frozen segment JSON: {segment.segment_id}"
            ) from exc
        if not isinstance(content, Mapping) or not isinstance(content.get("user"), str):
            raise TypeError(
                f"frozen segment lacks original user request: {segment.segment_id}"
            )
        segment_facts: list[dict[str, Any]] = []
        for raw_fact in content.get("tools") or []:
            if not isinstance(raw_fact, Mapping):
                raise TypeError(f"invalid frozen tool fact: {segment.segment_id}")
            fact = _compact_summary_fact(raw_fact)
            if fact is None:
                ignored_structured_outputs += 1
                continue
            segment_facts.append(fact)
            key = _tool_state_key(fact)
            index = fact_indexes.get(key)
            if index is None:
                facts.append(fact)
                fact_indexes[key] = len(facts) - 1
            else:
                previous_call_ids = _tool_call_ids(facts[index])
                call_ids = _tool_call_ids(fact)
                if call_ids or previous_call_ids:
                    fact["tool_call_ids"] = list(
                        dict.fromkeys([*previous_call_ids, *call_ids])
                    )
                # The later source is authoritative, including cancellation,
                # replacement, removal, exclusion, and error states.
                facts[index] = fact
        assistant = str(content.get("assistant") or "")
        if not _RECOVERY_PROBE.search(content["user"]):
            if authored_turn is not None:
                authored_turn += 1
            _merge_journey_turn(
                journeys,
                content["user"],
                source_message_start=segment.source_message_start,
                authored_turn=authored_turn,
                facts=segment_facts,
                assistant=assistant,
                force_new=any(_is_search_fact(fact) for fact in segment_facts),
            )
        sources.append(
            {"segment_id": segment.segment_id, "content_hash": segment.content_hash}
        )
        added_source = True
        known[segment.segment_id] = segment.content_hash
        if segment.source_message_start is not None:
            starts.append(segment.source_message_start)
        if segment.source_message_end is not None:
            ends.append(segment.source_message_end)
        revisions.append(segment.context_revision)

    if not sources:
        raise ValueError("at least one frozen segment is required for L3")
    if current is not None and not added_source:
        return current
    generation = (current.generation + 1) if current else 1
    source_kwargs = {
        "source_segments": tuple(sources),
        "source_message_start": min(starts) if starts else None,
        "source_message_end": max(ends) if ends else None,
        "context_revision": max(revisions, default=0),
        "generation": generation,
        "target_tokens": target_tokens,
    }
    candidate = _build_sized_summary(
        **source_kwargs,
        user_requests=(),
        tool_facts=facts,
        assistant_outcomes=(),
        shopping_journeys=journeys,
        refinement="semantic_journey_aggregation",
        omitted_counts={
            "duplicate_structured_outputs": ignored_structured_outputs,
        },
    )
    if candidate.estimated_tokens <= target_tokens:
        return candidate

    if candidate.estimated_tokens > hard_limit_tokens:
        raise SummaryBudgetError(
            "semantic journey summary remains above hard limit: "
            f"estimated={candidate.estimated_tokens}, hard={hard_limit_tokens}"
        )
    return _build_sized_summary(
        **source_kwargs,
        user_requests=(),
        tool_facts=facts,
        assistant_outcomes=(),
        shopping_journeys=journeys,
        refinement="semantic_journey_target_unmet_but_within_hard_limit",
        omitted_counts={
            "duplicate_structured_outputs": ignored_structured_outputs,
        },
    )


def _canonical_tool(value: Mapping[str, Any]) -> str:
    return canonical_json(
        {
            key: child
            for key, child in value.items()
            if key not in {"tool_call_id", "tool_call_ids"}
        }
    )


def _tool_state_key(value: Mapping[str, Any]) -> str:
    """Return an identity key whose later fact replaces earlier state."""

    args = value.get("args")
    args = args if isinstance(args, Mapping) else {}
    identity: dict[str, Any] = {}
    for key in (
        "order_id",
        "confirmation_id",
        "sku_id",
        "product_id",
        "preference_id",
    ):
        if args.get(key) not in (None, ""):
            identity[key] = args[key]
        elif value.get(key) not in (None, ""):
            identity[key] = value[key]
    refs = value.get("refs")
    if not identity and isinstance(refs, (list, tuple)) and refs:
        identity["refs"] = list(refs)
    if identity:
        return canonical_json({"tool": value.get("tool"), "identity": identity})
    return _canonical_tool(value)


def _build_sized_summary(
    *,
    source_segments: tuple[dict[str, str], ...],
    source_message_start: int | None,
    source_message_end: int | None,
    context_revision: int,
    generation: int,
    target_tokens: int,
    user_requests: Sequence[str],
    tool_facts: Sequence[Mapping[str, Any]],
    assistant_outcomes: Sequence[str],
    shopping_journeys: Sequence[Mapping[str, Any]],
    refinement: str,
    omitted_counts: Mapping[str, int],
) -> StageSummary:
    estimate = 0
    summary: StageSummary | None = None
    for _ in range(4):
        summary = StageSummary(
            source_segments=source_segments,
            user_requests=tuple(user_requests),
            tool_facts=tuple(dict(item) for item in tool_facts),
            assistant_outcomes=tuple(assistant_outcomes),
            shopping_journeys=tuple(dict(item) for item in shopping_journeys),
            source_message_start=source_message_start,
            source_message_end=source_message_end,
            context_revision=context_revision,
            generation=generation,
            target_tokens=target_tokens,
            estimated_tokens=estimate,
            refinement=refinement,
            omitted_counts=dict(omitted_counts),
        )
        next_estimate = max(
            1,
            (len(canonical_json(summary.to_prompt_dict()).encode("utf-8")) + 3) // 4,
        )
        if next_estimate == estimate:
            break
        estimate = next_estimate
    assert summary is not None
    return summary


def _tool_call_ids(value: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    if value.get("tool_call_id") not in (None, ""):
        result.append(str(value["tool_call_id"]))
    raw = value.get("tool_call_ids")
    if isinstance(raw, (list, tuple)):
        result.extend(str(item) for item in raw if item not in (None, ""))
    return list(dict.fromkeys(result))
