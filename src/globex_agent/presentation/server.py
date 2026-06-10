"""FastAPI entry point for the Globex commerce agent."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from globex_agent.application.agents.orchestrator import SubmitIntentInput
from globex_agent.application.tools.order_tools import _address
from globex_agent.application.usecases.confirmation_usecases import ConfirmationError
from globex_agent.application.usecases.order_usecases import OrderItemInput
from globex_agent.composition import Container, build_container
from globex_agent.domain.queue.ports.task_queue import IntentTask, TaskStatus
from globex_agent.infrastructure.settings import load_settings
from globex_agent.presentation.connection import ConnectionManager
from globex_agent.presentation.dto import (
    CancelOrderRequest,
    ConfirmCancelRequest,
    ConfirmOrderRequest,
    PrepareOrderRequest,
    SubmitIntentRequest,
    SubmitIntentResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_IDEMPOTENCY_TTL_SECONDS = 600


def build_app() -> FastAPI:
    state: dict = {}

    async def _forward_remote_events(container: Container) -> None:
        if container.backplane is None:
            return
        try:
            async for event in container.backplane.listen():
                container.bus.deliver_local(event)
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001 - backplane must not kill API
            logger.warning("事件背板监听中断：%s", err)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        container = await build_container()
        state["c"] = container
        state["connections"] = ConnectionManager(container.bus)
        api.state.container = container
        api.state.connections = state["connections"]
        await container.startup()
        if container.backplane is not None:
            state["forwarder"] = asyncio.create_task(
                _forward_remote_events(container)
            )
        try:
            yield
        finally:
            forwarder = state.pop("forwarder", None)
            if forwarder is not None:
                forwarder.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await forwarder
            state.pop("c", None)
            await container.shutdown()

    api = FastAPI(title="Globex 跨境电商 Agent", version="0.4.0", lifespan=lifespan)

    def container() -> Container:
        if "c" not in state:
            raise HTTPException(status_code=503, detail="服务尚未就绪")
        return state["c"]

    settings = load_settings()
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    async def health() -> dict:
        current = container()
        database = "disabled"
        if current.db_engine is not None:
            try:
                async with current.db_engine.connect() as conn:
                    await conn.execute(text("select 1"))
                database = current.db_engine.url.get_backend_name()
            except Exception as err:  # noqa: BLE001
                database = f"error: {err}"
        redis_state = "disabled"
        if current.cache.enabled:
            redis_state = "ok" if await current.cache.ping() else "error"
        return {
            "status": "ok",
            "model": current.settings.llm_model,
            "database": database,
            "redis": redis_state,
            "semantic_cache": current.semantic_cache.enabled,
            "queue": "enabled" if current.task_queue is not None else "disabled",
            "queue_depth": (
                await current.task_queue.depth()
                if current.task_queue is not None
                else 0
            ),
        }

    @api.post("/commerce/intents", response_model=SubmitIntentResponse)
    async def submit_intent(body: SubmitIntentRequest) -> SubmitIntentResponse:
        current = container()
        session_id = body.shopping_session_id or f"session-{uuid.uuid4().hex[:8]}"
        intent = SubmitIntentInput(
            shopping_session_id=session_id,
            buyer_id=body.buyer_id,
            locale=body.locale,
            currency=body.currency,
            raw_query=body.raw_query,
        )
        if current.task_queue is None:
            result = await current.orchestrator.handle_intent(intent)
            return SubmitIntentResponse(
                shopping_session_id=result.shopping_session_id,
                text=result.final_text,
                final_text=result.final_text,
                recommended_cards=result.recommended_cards,
                verification_status=result.verification_status,
            )
        task_id = await _enqueue(current, intent)
        queued_result = await _await_result(current, task_id, session_id)
        return SubmitIntentResponse(
            shopping_session_id=session_id,
            text=str(queued_result.get("text", queued_result.get("final_text", ""))),
            final_text=str(queued_result.get("text", queued_result.get("final_text", ""))),
            recommended_cards=queued_result.get("recommended_cards", []),
            verification_status=queued_result.get("verification_status", "unavailable"),
        )

    @api.post("/commerce/intents/async")
    async def submit_intent_async(body: SubmitIntentRequest) -> dict:
        current = container()
        session_id = body.shopping_session_id or f"session-{uuid.uuid4().hex[:8]}"
        intent = SubmitIntentInput(
            shopping_session_id=session_id,
            buyer_id=body.buyer_id,
            locale=body.locale,
            currency=body.currency,
            raw_query=body.raw_query,
        )
        if current.task_queue is None:
            raise HTTPException(
                status_code=503,
                detail="队列未启用，请使用 /commerce/intents",
            )
        task_id = await _enqueue(current, intent)
        return {"shopping_session_id": session_id, "task_id": task_id, "state": "queued"}

    @api.get("/commerce/tasks/{task_id}")
    async def get_task(task_id: str) -> dict:
        current = container()
        if current.task_queue is None:
            raise HTTPException(status_code=503, detail="队列未启用")
        status = await current.task_queue.get_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"任务不存在或已过期：{task_id}")
        return {
            "task_id": status.task_id,
            "state": status.state,
            "text": status.text or status.final_text,
            "final_text": status.text or status.final_text,
            "recommended_cards": status.recommended_cards,
            "verification_status": status.verification_status,
            "error": status.error,
            "queue_position": status.queue_position,
        }

    @api.websocket("/commerce/events")
    async def commerce_events(websocket: WebSocket) -> None:
        await state["connections"].serve(websocket)

    @api.get("/commerce/orders/{order_id}")
    async def get_order(order_id: str) -> dict:
        try:
            return await container().query_order.execute(order_id)
        except ValueError as err:
            raise HTTPException(status_code=404, detail=str(err)) from err

    @api.post("/commerce/orders/prepare")
    async def prepare_order_endpoint(body: PrepareOrderRequest) -> dict:
        current = container()
        try:
            return await current.confirmation_service.prepare_order(
                session_id=body.shopping_session_id,
                user_id=body.buyer_id,
                items=[
                    OrderItemInput(
                        item_id=item.item_id,
                        variant_id=item.variant_id,
                        quantity=item.quantity,
                    )
                    for item in body.items
                ],
                shipping_address=_address(body.shipping_address),
                idempotency_key=body.idempotency_key,
                include_token=True,
            )
        except (ValueError, ConfirmationError) as err:
            raise _confirmation_http_error(err) from err

    @api.post("/commerce/orders/confirm")
    async def confirm_order_endpoint(body: ConfirmOrderRequest) -> dict:
        try:
            return await container().confirmation_service.confirm_order(
                confirmation_id=body.confirmation_id,
                token=body.confirmation_token,
                session_id=body.shopping_session_id,
            )
        except (ValueError, ConfirmationError) as err:
            raise _confirmation_http_error(err) from err

    @api.post("/commerce/orders/{order_id}/cancel")
    async def cancel_order_endpoint(order_id: str, body: CancelOrderRequest) -> dict:
        try:
            return await container().confirmation_service.prepare_cancel(
                session_id=body.shopping_session_id,
                user_id=body.buyer_id,
                order_id=order_id,
                reason=body.reason,
                idempotency_key=body.idempotency_key,
                include_token=True,
            )
        except (ValueError, ConfirmationError) as err:
            raise _confirmation_http_error(err) from err

    @api.post("/commerce/orders/cancel/confirm")
    async def confirm_cancel_endpoint(body: ConfirmCancelRequest) -> dict:
        try:
            return await container().confirmation_service.confirm_cancel(
                confirmation_id=body.confirmation_id,
                token=body.confirmation_token,
                session_id=body.shopping_session_id,
            )
        except (ValueError, ConfirmationError) as err:
            raise _confirmation_http_error(err) from err

    return api


def _confirmation_http_error(error: Exception) -> HTTPException:
    code = getattr(error, "code", "invalid_confirmation")
    status_code = 409 if code in {
        "already_used",
        "in_progress",
        "expired",
        "invalidated",
        "price_changed",
        "spec_changed",
        "shipping_changed",
        "unavailable",
        "order_changed",
        "idempotency_conflict",
    } else 400
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": str(error)},
    )


async def _enqueue(container: Container, intent: SubmitIntentInput) -> str:
    assert container.task_queue is not None
    fingerprint = hashlib.sha256(
        f"{intent.shopping_session_id}\n{intent.raw_query}".encode()
    ).hexdigest()[:32]
    idem_key = f"idem:{fingerprint}"
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    acquired = await container.cache.set_if_absent(
        idem_key,
        task_id,
        _IDEMPOTENCY_TTL_SECONDS,
    )
    if not acquired:
        previous = await container.cache.get_raw(idem_key)
        if previous:
            return previous
    await container.task_queue.enqueue(
        IntentTask(
            task_id=task_id,
            shopping_session_id=intent.shopping_session_id,
            buyer_id=intent.buyer_id,
            locale=intent.locale,
            currency=intent.currency,
            raw_query=intent.raw_query,
        )
    )
    await container.task_queue.set_status(
        TaskStatus(task_id=task_id, state="queued")
    )
    container.bus.publish(
        intent.shopping_session_id,
        "task.queued",
        {"task_id": task_id},
    )
    return task_id


async def _await_result(
    container: Container,
    task_id: str,
    session_id: str,
) -> dict:
    assert container.task_queue is not None
    queue = container.bus.subscribe(session_id)
    deadline = time.monotonic() + container.settings.queue_wait_seconds
    try:
        while time.monotonic() < deadline:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                status = await container.task_queue.get_status(task_id)
                if status is not None and status.state == "done":
                    text_value = status.text or status.final_text
                    return {
                        "text": text_value,
                        "final_text": text_value,
                        "recommended_cards": status.recommended_cards,
                        "verification_status": status.verification_status,
                    }
                if status is not None and status.state == "failed":
                    return {
                        "text": f"[error] {status.error}",
                        "final_text": f"[error] {status.error}",
                        "recommended_cards": [],
                        "verification_status": "unavailable",
                    }
                continue
            if event.type == "final.result":
                return dict(event.payload)
        return {
            "text": "[error] 处理超时，请稍后重试或改用异步接口查询任务状态",
            "final_text": "[error] 处理超时，请稍后重试或改用异步接口查询任务状态",
            "recommended_cards": [],
            "verification_status": "unavailable",
        }
    finally:
        container.bus.unsubscribe(session_id, queue)


app = build_app()
