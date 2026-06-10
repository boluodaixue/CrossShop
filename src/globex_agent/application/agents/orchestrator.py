"""Main-agent orchestration and the mandatory online evidence gate."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from globex_agent.application.agents.main_agent import SessionRegistry
from globex_agent.application.evidence_verification import (
    EvidenceVerificationService,
    RecommendationDraft,
    SearchQuoteContext,
    build_semantic_evidence_bundle,
    hydrate_recommended_cards,
    parse_recommendation_draft,
)
from globex_agent.domain.buyer.preference import PreferenceStore
from globex_agent.domain.session.ports.conversation_store import (
    ConversationEventRecord,
    ConversationStore,
    ConversationTurn,
)
from globex_agent.infrastructure.cache.semantic_cache import SemanticCache
from globex_agent.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from globex_agent.infrastructure.eventbus import TradeEvent, TradeEventBus
from globex_agent.infrastructure.transient import is_transient_error

logger = logging.getLogger(__name__)

_MAX_GENERATION_ATTEMPTS = 3
_MAX_TURN_RETRIES = 2
_RETRY_BASE_SECONDS = 1.0


@dataclass(frozen=True)
class SubmitIntentInput:
    shopping_session_id: str
    buyer_id: str
    locale: str
    currency: str
    raw_query: str


@dataclass(frozen=True)
class SubmitIntentOutput:
    shopping_session_id: str
    final_text: str
    recommended_cards: list[dict[str, Any]]
    verification_status: str


class MainAgentOrchestrator:
    """Run generation, freeze first-turn evidence, and publish only verified output."""

    def __init__(
        self,
        sessions: SessionRegistry,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        conversation_store: ConversationStore | None = None,
        semantic_cache: SemanticCache | None = None,
        context_size: int = 128000,
        evidence_verifier: EvidenceVerificationService | None = None,
        answer_finalizer: Any | None = None,
        max_generation_attempts: int = _MAX_GENERATION_ATTEMPTS,
        turn_timeout_seconds: float = 85.0,
        generation_timeout_seconds: float = 45.0,
        rewrite_timeout_seconds: float = 25.0,
    ) -> None:
        self._sessions = sessions
        self._bus = bus
        self._preference_store = preference_store
        self._conversation_store = conversation_store
        self._semantic_cache = semantic_cache
        self._context_size = context_size
        self._evidence_verifier = evidence_verifier or EvidenceVerificationService(None)
        self._answer_finalizer = answer_finalizer
        self._max_generation_attempts = max(1, min(3, max_generation_attempts))
        self._turn_timeout_seconds = max(1.0, turn_timeout_seconds)
        self._generation_timeout_seconds = max(1.0, generation_timeout_seconds)
        self._rewrite_timeout_seconds = max(1.0, rewrite_timeout_seconds)

    async def handle_intent(self, intent: SubmitIntentInput) -> SubmitIntentOutput:
        session_id = intent.shopping_session_id
        snapshot = ShoppingContextSnapshot(
            shopping_session_id=session_id,
            buyer_id=intent.buyer_id,
            locale=intent.locale,
            currency=intent.currency,
            main_thread_id=self._sessions.thread_id(session_id),
        )
        token = ShoppingContext.set(snapshot)
        try:
            async with self._sessions.session_lock(session_id):
                return await self._handle_intent_locked(intent)
        finally:
            ShoppingContext.reset(token)

    async def _handle_intent_locked(self, intent: SubmitIntentInput) -> SubmitIntentOutput:
        session_id = intent.shopping_session_id
        started_at = time.monotonic()
        trace = self._bus.subscribe(session_id)
        captured_events: list[dict[str, Any]] = []
        final_text = ""
        recommended_cards: list[dict[str, Any]] = []
        verification_status = "unavailable"
        try:
            has_history = await self._has_history(session_id)
            agent = await self._sessions.get_or_create(session_id)
            if has_history:
                compressed = await agent.compress_context(
                    self._sessions.thread_id(session_id),
                    max_messages=max(4, self._context_size // 1000),
                )
                if compressed:
                    self._bus.publish(session_id, "context.compressed", compressed)
            query = await self._build_query(intent)
            captured_events.extend(_drain_trace(trace))
            result = await self._reply_with_retry(
                session_id=session_id,
                agent=agent,
                query=query,
                trace=trace,
                captured_events=captured_events,
            )
            final_text, recommended_cards, verification_status = result
            self._bus.publish(
                session_id,
                "final.result",
                {
                    "text": final_text,
                    "recommended_cards": recommended_cards,
                    "verification_status": verification_status,
                },
            )
            return SubmitIntentOutput(
                session_id,
                final_text,
                recommended_cards,
                verification_status,
            )
        except Exception as err:  # noqa: BLE001 - turn errors must remain observable
            logger.exception("MainAgent 异常")
            self._bus.publish(session_id, "error", {"message": _format_exception(err)})
            final_text = "本轮处理未完成，证据不可用，暂不展示商品卡。请稍后重试。"
            verification_status = "unavailable"
            self._bus.publish(
                session_id,
                "final.result",
                {
                    "text": final_text,
                    "recommended_cards": [],
                    "verification_status": verification_status,
                },
            )
            return SubmitIntentOutput(session_id, final_text, [], verification_status)
        finally:
            captured_events.extend(_drain_trace(trace))
            await self._record_conversation(
                intent,
                final_text,
                int((time.monotonic() - started_at) * 1000),
                trace,
                captured_events,
            )

    async def _has_history(self, session_id: str) -> bool:
        if self._conversation_store is None:
            return False
        try:
            turns = await self._conversation_store.list_turns(session_id, limit=1)
        except Exception as err:  # noqa: BLE001 - memory read must not block a turn
            logger.warning("读取会话历史失败，按无历史处理：%s", err)
            return False
        return bool(turns)

    async def _build_query(self, intent: SubmitIntentInput) -> str:
        try:
            preferences = await self._preference_store.list_by_buyer(intent.buyer_id)
        except Exception as err:  # noqa: BLE001 - preference read is best effort
            logger.warning("读取买家偏好失败：%s", err)
            preferences = []
        if not preferences:
            return intent.raw_query
        rendered = "\n".join(f"- [{p.kind}] {p.statement}" for p in preferences)
        return f"<buyer-preferences>\n{rendered}\n</buyer-preferences>\n{intent.raw_query}"

    async def _reply_with_retry(
        self,
        *,
        session_id: str,
        agent: Any,
        query: str,
        trace: asyncio.Queue,
        captured_events: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], str]:
        frozen_events: list[dict[str, Any]] | None = None
        frozen_tool_outputs: list[dict[str, Any]] = []
        frozen_top_k: list[dict[str, Any]] = []
        frozen_card_contexts: dict[str, SearchQuoteContext] = {}
        frozen_category_insight: dict[str, Any] | None = None
        previous_raw: str | None = None
        previous_draft: RecommendationDraft | None = None
        feedback = ""
        deadline = asyncio.get_running_loop().time() + self._turn_timeout_seconds

        for attempt in range(self._max_generation_attempts):
            if attempt == 0:
                try:
                    raw = await self._run_with_deadline(
                        self._call_agent_with_transient_retries(
                            session_id,
                            agent,
                            query,
                            allow_tools=True,
                        ),
                        deadline,
                        self._generation_timeout_seconds,
                    )
                except TimeoutError:
                    reason = "首轮生成超过服务端时间预算"
                    self._bus.publish(
                        session_id,
                        "error",
                        {"phase": "generation", "error_type": "TimeoutError", "message": reason},
                    )
                    return self._safe_fallback(reason), [], "unavailable"
                attempt_events = _drain_trace(trace)
                captured_events.extend(attempt_events)
                frozen_events = list(attempt_events)
                frozen_tool_outputs = _model_visible_tool_outputs(attempt_events)
                frozen_top_k, frozen_card_contexts = _top_k_cards_with_contexts(attempt_events)
                frozen_category_insight = _last_category_insight(attempt_events)
                authoritative = _last_successful_product_search(attempt_events)
                if authoritative is not None:
                    call_id, args, output = authoritative
                    self._bus.publish(
                        session_id,
                        "evidence.freeze",
                        {
                            "authoritative_tool_call_id": call_id,
                            "authoritative_search_args": args,
                            "frozen_item_ids": [
                                str(card.get("item_id")) for card in frozen_top_k
                            ],
                            "returned_count": len(output.get("hits", [])),
                        },
                    )
            else:
                if self._answer_finalizer is None or frozen_events is None:
                    break
                try:
                    raw = await self._run_with_deadline(
                        self._finalize_without_tools(
                            query=query,
                            previous_raw=previous_raw,
                            previous_draft=previous_draft,
                            feedback=feedback,
                            frozen_top_k=frozen_top_k,
                            category_insight=frozen_category_insight,
                        ),
                        deadline,
                        self._rewrite_timeout_seconds,
                    )
                except TimeoutError:
                    reason = f"第{attempt + 1}次无工具重写超过服务端时间预算"
                    self._bus.publish(
                        session_id,
                        "error",
                        {"phase": "rewrite", "error_type": "TimeoutError", "message": reason},
                    )
                    return self._safe_fallback(reason), [], "unavailable"
            previous_raw = _content_text(raw)
            draft, parse_error = parse_recommendation_draft(raw)
            if draft is None:
                feedback = f"输出协议错误：{parse_error}。只输出 answer_text 和 selections。"
                self._publish_verification(
                    session_id,
                    attempt + 1,
                    "unsupported",
                    feedback,
                )
                continue
            previous_draft = draft
            try:
                guard, judge = await self._run_with_deadline(
                    self._evidence_verifier.verify(
                        query=query,
                        draft=draft,
                        top_k_cards=frozen_top_k,
                        tool_outputs=frozen_tool_outputs,
                        card_contexts=frozen_card_contexts,
                        category_insight=frozen_category_insight,
                    ),
                    deadline,
                    self._turn_timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError):
                reason = "在线证据验证超过服务端整轮时间预算"
                self._bus.publish(
                    session_id,
                    "error",
                    {
                        "phase": "evidence_verification",
                        "error_type": "TimeoutError",
                        "message": reason,
                    },
                )
                return self._safe_fallback(reason), [], "unavailable"
            status = judge.verdict if guard.passed else "unsupported"
            reason = judge.reason or "; ".join(guard.errors)
            self._publish_verification(
                session_id,
                attempt + 1,
                status,
                reason,
                unsupported_claims=[
                    claim.model_dump(mode="json") for claim in judge.unsupported_claims
                ],
                fact_guard_errors=list(guard.errors),
            )
            if status == "supported" and guard.passed:
                cards = hydrate_recommended_cards(frozen_top_k, draft.selections)
                return draft.answer_text, cards, "supported"
            if status == "unavailable":
                return self._safe_fallback("在线证据验证不可用"), [], "unavailable"
            feedback = _verification_feedback(guard.errors, judge.reason, judge.unsupported_claims)

        return self._safe_fallback("本轮回答未通过在线事实验证"), [], "unsupported"

    async def _run_with_deadline(
        self,
        awaitable: Any,
        deadline: float,
        phase_budget: float,
    ) -> Any:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("turn deadline exceeded")
        return await asyncio.wait_for(awaitable, timeout=min(remaining, phase_budget))

    async def _call_agent_with_transient_retries(
        self,
        session_id: str,
        agent: Any,
        query: str,
        *,
        allow_tools: bool,
    ) -> Any:
        del allow_tools  # the first graph call has tools; retry calls use the same graph turn

        async def sink(event_type: str, payload: dict) -> None:
            # LangGraph emits token.delta before the final draft is verified.
            # Non-token events remain available to the timeline and audit.
            if event_type != "token.delta":
                self._bus.publish(session_id, event_type, payload)

        last_error: Exception | None = None
        for attempt in range(_MAX_TURN_RETRIES + 1):
            try:
                return await agent.reply(
                    query,
                    thread_id=self._sessions.thread_id(session_id),
                    event_sink=sink,
                )
            except Exception as err:  # noqa: BLE001 - transient model boundary
                if not is_transient_error(err) or attempt >= _MAX_TURN_RETRIES:
                    raise
                last_error = err
                self._bus.publish(
                    session_id,
                    "error",
                    {
                        "message": f"上游瞬时故障，正在重试：{_format_exception(err)}",
                        "retrying": True,
                    },
                )
                await asyncio.sleep(_RETRY_BASE_SECONDS * (3**attempt))
        raise last_error if last_error else RuntimeError("reply 重试耗尽")

    async def _finalize_without_tools(
        self,
        *,
        query: str,
        previous_raw: str | None,
        previous_draft: RecommendationDraft | None,
        feedback: str,
        frozen_top_k: list[dict[str, Any]],
        category_insight: dict[str, Any] | None,
    ) -> str:
        previous_text = (
            previous_draft.model_dump_json()
            if previous_draft
            else previous_raw or "{}"
        )
        semantic_evidence = build_semantic_evidence_bundle(
            query=query,
            draft=previous_draft
            or RecommendationDraft(answer_text="证据不足", selections=[]),
            top_k_cards=frozen_top_k,
            category_insight=category_insight,
        )
        prompt = (
            "只输出严格 JSON，不要 Markdown、解释或额外字段。JSON 只能是 "
            '{"answer_text":"...","selections":[{"item_id":"...","variant_id":"...或null"}]}。'
            "这是同一轮已冻结的候选；不要调用工具，不要换商品，不要写商品卡事实字段。"
            "所有被具体推荐、比较或描述的商品都必须进入 selections；"
            "具体 SKU 陈述必须提供 variant_id。"
            "只根据 semantic evidence 修正自然语言和 selections。"
            "若证据不足，诚实说明并将 selections 置为空。\n"
            f"Query:\n{query}\n"
            "允许的冻结 item_id:\n"
            f"{json.dumps([card.get('item_id') for card in frozen_top_k], ensure_ascii=False)}\n"
            f"semantic evidence:\n{json.dumps(semantic_evidence, ensure_ascii=False)}\n"
            f"上次 draft:\n{previous_text}\n"
            f"验证反馈:\n{feedback}\n"
        )
        result = await self._answer_finalizer.ainvoke(prompt)
        return _content_text(result)

    def _publish_verification(
        self,
        session_id: str,
        attempt: int,
        status: str,
        reason: str,
        *,
        unsupported_claims: list[dict[str, Any]] | None = None,
        fact_guard_errors: list[str] | None = None,
    ) -> None:
        self._bus.publish(
            session_id,
            "evidence.verify",
            {
                "attempt": attempt,
                "verification_status": status,
                "reason": reason,
                "unsupported_claims": unsupported_claims or [],
                "fact_guard_errors": fact_guard_errors or [],
            },
        )

    @staticmethod
    def _safe_fallback(reason: str) -> str:
        return f"{reason}，暂不展示商品卡。请补充具体品类、规格或收货国家后重试。"

    async def _record_conversation(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        latency_ms: int,
        trace: asyncio.Queue,
        captured_events: list[dict[str, Any]],
    ) -> None:
        if self._conversation_store is None:
            self._bus.unsubscribe(intent.shopping_session_id, trace)
            return
        self._bus.unsubscribe(intent.shopping_session_id, trace)
        events = [
            ConversationEventRecord(
                session_id=intent.shopping_session_id,
                type=str(event.get("type", "")),
                payload=_sanitize_audit_payload(event.get("payload", {})),
                occurred_at=str(event.get("occurred_at", "")),
            )
            for event in captured_events
            if event.get("type") != "token.delta"
        ]
        try:
            await self._conversation_store.touch_session(
                intent.shopping_session_id,
                intent.buyer_id,
                intent.locale,
                intent.currency,
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=intent.shopping_session_id,
                    buyer_id=intent.buyer_id,
                    role="buyer",
                    content=intent.raw_query,
                )
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=intent.shopping_session_id,
                    buyer_id=intent.buyer_id,
                    role="agent",
                    content=final_text,
                    latency_ms=latency_ms,
                )
            )
            await self._conversation_store.append_events(events)
        except Exception as err:  # noqa: BLE001 - audit persistence is best effort
            logger.warning("对话记录写入失败：%s（%s）", intent.shopping_session_id, err)


_AUDIT_REDACT_KEYS = {
    "address",
    "address_line",
    "phone",
    "postal_code",
    "recipient",
    "recipient_name",
    "shipping_address",
    "shipping_summary",
    "confirmation_token",
    "token",
}
_PHONE_RE = re.compile(r"(?<!\d)1\d{10}(?!\d)")


def _sanitize_audit_payload(value: Any, *, key: str = "") -> Any:
    """Redact PII/tokens while retaining complete relevant tool output."""

    if key.casefold() in _AUDIT_REDACT_KEYS:
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_audit_payload(child, key=str(child_key))
            for child_key, child in value.items()
            if str(child_key) != "content_vector"
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_audit_payload(child, key=key) for child in value]
    if isinstance(value, str):
        return _PHONE_RE.sub("[redacted]", value)
    return value


def _drain_trace(trace: asyncio.Queue) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while not trace.empty():
        event = trace.get_nowait()
        if isinstance(event, TradeEvent):
            events.append(
                {
                    "type": event.type,
                    "payload": event.payload,
                    "occurred_at": event.occurred_at,
                }
            )
    return events


def _model_visible_tool_outputs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only the authoritative search and relevant category insight.

    The event stream remains the complete audit record. This compact list is
    only the online verification input and must not grow with exploratory
    ProductSearch calls.
    """
    outputs: list[dict[str, Any]] = []
    last_product: dict[str, Any] | None = None
    last_category: dict[str, Any] | None = None
    authoritative = _last_successful_product_search(events)
    if authoritative is not None:
        call_id, _args, parsed = authoritative
        last_product = {
            "tool": "product_search_tool",
            "tool_call_id": call_id,
            "model_output": _sanitize_audit_payload(parsed),
        }
    for event in events:
        if event.get("type") != "tool.result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        # Only the exact model_output is evidence for the online Judge.  The
        # surrounding event may contain audit-only snapshots and provenance.
        output = payload.get("model_output")
        if output is None:
            continue
        parsed = _parse_json_value(output)
        if not isinstance(parsed, dict):
            continue
        tool = str(payload.get("tool", ""))
        if tool == "category_insight_tool" and isinstance(parsed.get("insights"), dict):
            last_category = {
                "tool": tool,
                "tool_call_id": payload.get("tool_call_id"),
                "model_output": _sanitize_audit_payload(parsed),
            }
    if last_product is not None:
        outputs.append(last_product)
    if last_category is not None:
        outputs.append(last_category)
    return outputs


def _top_k_cards(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards, _ = _top_k_cards_with_contexts(events)
    return cards


def _top_k_cards_with_contexts(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, SearchQuoteContext]]:
    """Freeze cards and bind each one to its own product-search invocation."""

    cards: list[dict[str, Any]] = []
    card_contexts: dict[str, SearchQuoteContext] = {}
    invocation_contexts: dict[str, SearchQuoteContext] = {}
    for event in events:
        if event.get("type") != "tool.invoke":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("tool") != "product_search_tool":
            continue
        call_id = payload.get("tool_call_id")
        args = _parse_json_value(payload.get("args"))
        if call_id and isinstance(args, dict):
            invocation_contexts[str(call_id)] = SearchQuoteContext.from_mapping(args)
    authoritative = _last_successful_product_search(events)
    if authoritative is None:
        return [], {}
    call_id, _args, output = authoritative
    context = invocation_contexts.get(str(call_id))
    for card in output.get("hits", [])[:5]:
        if not isinstance(card, dict):
            continue
        item_id = str(card.get("item_id", ""))
        if not item_id:
            continue
        cards.append(card)
        if context is not None:
            card_contexts[item_id] = context
    return cards, card_contexts


def _last_successful_product_search(
    events: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    """Find the last successful non-empty ProductSearch, without backfilling."""

    invocation_args: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("type") != "tool.invoke":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("tool") != "product_search_tool":
            continue
        call_id = payload.get("tool_call_id")
        args = _parse_json_value(payload.get("args"))
        if not call_id or not isinstance(args, dict):
            continue
        invocation_args[str(call_id)] = args
    for event in events:
        if event.get("type") != "tool.result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("tool") != "product_search_tool":
            continue
        model_output = payload.get("model_output")
        if model_output is None or payload.get("error"):
            continue
        parsed = _parse_json_value(model_output)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("hits"), list):
            continue
        hits = [card for card in parsed["hits"] if isinstance(card, dict) and card.get("item_id")]
        if not hits:
            continue
        call_id = str(payload.get("tool_call_id", ""))
        if not call_id or call_id not in invocation_args:
            continue
        args = invocation_args[call_id]
        yield_value = (call_id, args, {**parsed, "hits": hits})
        # The loop intentionally keeps walking so the last valid result wins.
        last = yield_value
    return last if "last" in locals() else None


def _last_category_insight(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for event in events:
        if event.get("type") != "tool.result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("tool") != "category_insight_tool":
            continue
        parsed = _parse_json_value(payload.get("model_output"))
        if isinstance(parsed, dict) and isinstance(parsed.get("insights"), dict):
            latest = parsed
    return latest


def _walk_values(value: Any):
    if isinstance(value, str):
        parsed = _parse_json_value(value)
        if parsed is not value:
            yield from _walk_values(parsed)
        return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _content_text(value: Any) -> str:
    content = getattr(value, "content", value)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _format_exception(err: BaseException) -> str:
    """Keep the failure class when HTTP/async timeout text is empty."""

    detail = str(err).strip()
    return f"{type(err).__name__}: {detail}" if detail else type(err).__name__


def _verification_feedback(errors: tuple[str, ...], reason: str, unsupported_claims: list) -> str:
    details = list(errors)
    if reason:
        details.append(reason)
    details.extend(
        f"{claim.text}: {claim.reason}"
        for claim in unsupported_claims
        if getattr(claim, "text", "") or getattr(claim, "reason", "")
    )
    return "；".join(details) or "自然语言没有被当前冻结工具证据整体支持"
