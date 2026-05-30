"""MainAgent orchestrator: context, preferences, cache, and event publishing."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from globex_agent.application.agents.main_agent import SessionRegistry
from globex_agent.domain.buyer.preference import PreferenceStore
from globex_agent.domain.session.ports.conversation_store import (
    ConversationEventRecord,
    ConversationStore,
    ConversationTurn,
)
from globex_agent.infrastructure.cache.semantic_cache import SemanticCache
from globex_agent.infrastructure.context import (
    ShoppingContext,
    ShoppingContextSnapshot,
)
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.transient import is_transient_error

logger = logging.getLogger(__name__)

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


class MainAgentOrchestrator:
    def __init__(
        self,
        sessions: SessionRegistry,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        conversation_store: ConversationStore | None = None,
        semantic_cache: SemanticCache | None = None,
        context_size: int = 128000,
    ) -> None:
        self._sessions = sessions
        self._bus = bus
        self._preference_store = preference_store
        self._conversation_store = conversation_store
        self._semantic_cache = semantic_cache
        self._context_size = context_size

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

    async def _handle_intent_locked(
        self,
        intent: SubmitIntentInput,
    ) -> SubmitIntentOutput:
        """Run one stateful turn while the session lock is held."""

        session_id = intent.shopping_session_id
        started_at = time.monotonic()
        trace = self._bus.subscribe(session_id) if self._conversation_store else None
        final_text = ""
        try:
            has_history = await self._has_history(session_id)
            cached = await self._lookup_cache(intent, has_history)
            if cached is not None:
                final_text = cached
                self._bus.publish(session_id, "final.result", {"text": final_text})
                return SubmitIntentOutput(session_id, final_text)

            agent = await self._sessions.get_or_create(session_id)
            if has_history:
                compressed = await agent.compress_context(
                    self._sessions.thread_id(session_id),
                    max_messages=max(4, self._context_size // 1000),
                )
                if compressed:
                    self._bus.publish(
                        session_id,
                        "context.compressed",
                        compressed,
                    )
            query = await self._build_query(intent)
            final_text = await self._reply_with_retry(
                session_id,
                agent,
                query,
            )
            self._bus.publish(session_id, "final.result", {"text": final_text})
            await self._remember_cache(intent, final_text, has_history)
            return SubmitIntentOutput(session_id, final_text)
        except Exception as err:  # noqa: BLE001 - orchestrator must not fail silently
            logger.exception("MainAgent 异常")
            self._bus.publish(session_id, "error", {"message": str(err)})
            final_text = f"[error] {err}"
            return SubmitIntentOutput(session_id, final_text)
        finally:
            await self._record_conversation(
                intent,
                final_text,
                int((time.monotonic() - started_at) * 1000),
                trace,
            )
    async def _has_history(self, session_id: str) -> bool:
        if self._conversation_store is None:
            return False
        try:
            turns = await self._conversation_store.list_turns(session_id, limit=1)
        except Exception as err:  # noqa: BLE001 - cache safety must not block reply
            logger.warning("读取会话历史失败，按无历史处理：%s", err)
            return False
        return bool(turns)

    async def _build_query(self, intent: SubmitIntentInput) -> str:
        try:
            preferences = await self._preference_store.list_by_buyer(
                intent.buyer_id
            )
        except Exception as err:  # noqa: BLE001 - memory read must not block
            logger.warning("读取买家偏好失败：%s", err)
            preferences = []
        if not preferences:
            return intent.raw_query
        rendered = "\n".join(f"- [{p.kind}] {p.statement}" for p in preferences)
        return (
            "<buyer-preferences>\n"
            f"{rendered}\n"
            "</buyer-preferences>\n"
            f"{intent.raw_query}"
        )

    async def _reply_with_retry(
        self,
        session_id: str,
        agent,
        query: str,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(_MAX_TURN_RETRIES + 1):
            try:
                return await agent.reply(
                    query,
                    thread_id=self._sessions.thread_id(session_id),
                    event_sink=self._publish_event,
                )
            except Exception as err:  # noqa: BLE001
                if not is_transient_error(err) or attempt >= _MAX_TURN_RETRIES:
                    raise
                last_error = err
                delay = _RETRY_BASE_SECONDS * (3**attempt)
                self._bus.publish(
                    session_id,
                    "error",
                    {
                        "message": f"上游瞬时故障，正在重试：{err}",
                        "retrying": True,
                    },
                )
                await asyncio.sleep(delay)
        raise last_error if last_error else RuntimeError("reply 重试耗尽")

    async def _publish_event(self, event_type: str, payload: dict) -> None:
        session_id = ShoppingContext.current_session_id()
        self._bus.publish(session_id, event_type, payload)

    async def _lookup_cache(self, intent: SubmitIntentInput, has_history: bool) -> str | None:
        if self._semantic_cache is None:
            return None
        hit = await self._semantic_cache.lookup(
            intent.buyer_id,
            intent.raw_query,
            has_history,
        )
        if hit is None:
            return None
        self._bus.publish(
            intent.shopping_session_id,
            "cache.hit",
            {"similarity": hit.similarity, "matched_query": hit.matched_query},
        )
        return hit.reply

    async def _remember_cache(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        has_history: bool,
    ) -> None:
        if self._semantic_cache is None:
            return
        await self._semantic_cache.remember(
            intent.buyer_id,
            intent.raw_query,
            final_text,
            has_history,
        )

    async def _record_conversation(
        self,
        intent: SubmitIntentInput,
        final_text: str,
        latency_ms: int,
        trace: asyncio.Queue | None,
    ) -> None:
        if self._conversation_store is None:
            return
        session_id = intent.shopping_session_id
        events: list[ConversationEventRecord] = []
        if trace is not None:
            self._bus.unsubscribe(session_id, trace)
            while not trace.empty():
                event = trace.get_nowait()
                if event.type != "token.delta":
                    events.append(
                        ConversationEventRecord(
                            session_id=session_id,
                            type=event.type,
                            payload=(
                                event.payload
                                if isinstance(event.payload, dict)
                                else {"value": event.payload}
                            ),
                            occurred_at=event.occurred_at,
                        )
                    )
        try:
            await self._conversation_store.touch_session(
                session_id,
                intent.buyer_id,
                intent.locale,
                intent.currency,
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=session_id,
                    buyer_id=intent.buyer_id,
                    role="buyer",
                    content=intent.raw_query,
                )
            )
            await self._conversation_store.append_turn(
                ConversationTurn(
                    session_id=session_id,
                    buyer_id=intent.buyer_id,
                    role="agent",
                    content=final_text,
                    latency_ms=latency_ms,
                )
            )
            await self._conversation_store.append_events(events)
        except Exception as err:  # noqa: BLE001 - conversation persistence is best effort
            logger.warning("对话记录写入失败：%s（%s）", session_id, err)
