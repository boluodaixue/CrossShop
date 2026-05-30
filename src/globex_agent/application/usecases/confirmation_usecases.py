"""Trusted, one-time confirmation gates for user-visible order writes."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import uuid
from asyncio import Lock
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from globex_agent.application.usecases.order_usecases import (
    OrderItemInput,
    PlaceOrderUseCase,
    _build_order_lines,
)
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.domain.order.address import Address
from globex_agent.domain.order.order import Order
from globex_agent.domain.order.ports.order_repository import OrderRepository


class ConfirmationStatus(str, Enum):
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    CONSUMED = "CONSUMED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class ConfirmationError(ValueError):
    """Stable application error for trusted confirmation clients."""

    def __init__(self, code: str, message: str, *, status: ConfirmationStatus | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class ConfirmationIntent:
    confirmation_id: str
    token_hash: str
    session_id: str
    user_id: str
    action: str
    canonical_payload: dict[str, Any]
    payload_sha256: str
    preview: dict[str, Any]
    status: ConfirmationStatus
    expires_at: datetime
    idempotency_key: str | None
    created_at: datetime
    consumed_at: datetime | None = None
    result: dict[str, Any] | None = None


class InMemoryConfirmationIntentStore:
    """Process-local atomic intent store.

    The lock makes the claim/consume transition single-use within the current
    application process.  Multi-process durability belongs to P1 storage work.
    """

    def __init__(self) -> None:
        self._intents: dict[str, ConfirmationIntent] = {}
        self._by_idempotency: dict[tuple[str, str, str], str] = {}
        self._lock = Lock()

    async def create(self, intent: ConfirmationIntent) -> ConfirmationIntent:
        async with self._lock:
            if intent.idempotency_key:
                key = (intent.session_id, intent.action, intent.idempotency_key)
                previous_id = self._by_idempotency.get(key)
                if previous_id is not None:
                    previous = self._intents[previous_id]
                    if previous.payload_sha256 != intent.payload_sha256:
                        raise ConfirmationError(
                            "idempotency_conflict",
                            "幂等键已用于不同的确认内容",
                        )
                    return previous
                self._by_idempotency[key] = intent.confirmation_id
            self._intents[intent.confirmation_id] = intent
            return intent

    async def get(self, confirmation_id: str) -> ConfirmationIntent | None:
        async with self._lock:
            return self._intents.get(confirmation_id)

    async def claim(
        self,
        confirmation_id: str,
        token: str,
        *,
        session_id: str,
        action: str,
    ) -> ConfirmationIntent:
        async with self._lock:
            intent = self._intents.get(confirmation_id)
            if intent is None:
                raise ConfirmationError("not_found", "确认意图不存在")
            if intent.session_id != session_id:
                raise ConfirmationError("session_mismatch", "确认意图不属于当前会话")
            if intent.action != action:
                raise ConfirmationError("action_mismatch", "确认动作不匹配")
            if not hmac.compare_digest(intent.token_hash, _hash_token(token)):
                raise ConfirmationError("invalid_token", "确认令牌无效")
            if _payload_sha256(intent.canonical_payload) != intent.payload_sha256:
                intent.status = ConfirmationStatus.INVALIDATED
                raise ConfirmationError(
                    "payload_tampered",
                    "确认内容摘要校验失败，请重新预览",
                    status=intent.status,
                )
            now = _now()
            if intent.status is ConfirmationStatus.PENDING and now >= intent.expires_at:
                intent.status = ConfirmationStatus.EXPIRED
                raise ConfirmationError(
                    "expired",
                    "确认已过期，请重新预览",
                    status=intent.status,
                )
            if intent.status is ConfirmationStatus.EXECUTING:
                raise ConfirmationError("in_progress", "确认正在执行，请勿重复提交")
            if intent.status is ConfirmationStatus.CONSUMED:
                raise ConfirmationError("already_used", "确认已使用")
            if intent.status is ConfirmationStatus.EXPIRED:
                raise ConfirmationError("expired", "确认已过期，请重新预览", status=intent.status)
            if intent.status is ConfirmationStatus.INVALIDATED:
                raise ConfirmationError(
                    "invalidated",
                    "确认已失效，请重新预览",
                    status=intent.status,
                )
            intent.status = ConfirmationStatus.EXECUTING
            return intent

    async def consumed(self, confirmation_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            intent = self._intents[confirmation_id]
            intent.status = ConfirmationStatus.CONSUMED
            intent.consumed_at = _now()
            intent.result = result

    async def invalidate(self, confirmation_id: str) -> None:
        async with self._lock:
            intent = self._intents.get(confirmation_id)
            if intent is not None and intent.status is ConfirmationStatus.EXECUTING:
                intent.status = ConfirmationStatus.INVALIDATED


class OrderConfirmationService:
    """Prepare and execute place/cancel actions behind trusted confirmation."""

    def __init__(
        self,
        item_repo: ItemRepository,
        order_repo: OrderRepository,
        place_order: PlaceOrderUseCase,
        cancel_order,
        store: InMemoryConfirmationIntentStore | None = None,
        ttl_seconds: int = 300,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("confirmation TTL must be positive")
        self._item_repo = item_repo
        self._order_repo = order_repo
        self._place_order = place_order
        self._cancel_order = cancel_order
        self._store = store or InMemoryConfirmationIntentStore()
        self._ttl = timedelta(seconds=ttl_seconds)

    @property
    def store(self) -> InMemoryConfirmationIntentStore:
        return self._store

    async def prepare_order(
        self,
        *,
        session_id: str,
        user_id: str,
        items: list[OrderItemInput],
        shipping_address: Address,
        idempotency_key: str | None = None,
        include_token: bool = False,
    ) -> dict[str, Any]:
        lines, facts = await _build_order_lines(self._item_repo, items)
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "items": facts,
            "shipping_address": _address_payload(shipping_address),
        }
        total_minor = sum(line.subtotal().amount_in_minor_units for line in lines)
        currency = lines[0].unit_price.currency
        expires_at = _now() + self._ttl
        confirmation_id = f"confirm-{uuid.uuid4().hex}"
        token = secrets.token_urlsafe(32)
        preview = {
            "confirmation_required": True,
            "confirmation_id": confirmation_id,
            "action": "place_order",
            "expires_at": expires_at.isoformat(),
            "items": [
                {
                    "item_id": line.item_id,
                    "variant_id": line.variant_id,
                    "title": line.title,
                    "variant_display_name": line.variant_display_name,
                    "unit_price_major": line.unit_price.to_major_units(),
                    "quantity": line.quantity,
                    "subtotal_major": line.subtotal().to_major_units(),
                    "currency": line.unit_price.currency,
                }
                for line in lines
            ],
            "shipping_summary": shipping_address.one_line(),
            "total_amount_major": total_minor / 100,
            "currency": currency,
        }
        intent = await self._store.create(
            ConfirmationIntent(
                confirmation_id=confirmation_id,
                token_hash=_hash_token(token),
                session_id=session_id,
                user_id=user_id,
                action="place_order",
                canonical_payload=payload,
                payload_sha256=_payload_sha256(payload),
                preview=preview,
                status=ConfirmationStatus.PENDING,
                expires_at=expires_at,
                idempotency_key=idempotency_key,
                created_at=_now(),
            )
        )
        result = dict(intent.preview)
        if include_token and intent.confirmation_id == confirmation_id:
            result["confirmation_token"] = token
        return result

    async def confirm_order(
        self,
        *,
        confirmation_id: str,
        token: str,
        session_id: str,
    ) -> dict[str, Any]:
        intent = await self._store.claim(
            confirmation_id,
            token,
            session_id=session_id,
            action="place_order",
        )
        try:
            items = [
                OrderItemInput(
                    item_id=str(row["item_id"]),
                    variant_id=row.get("variant_id"),
                    quantity=int(row["quantity"]),
                )
                for row in intent.canonical_payload["items"]
            ]
            address = Address(**intent.canonical_payload["shipping_address"])
            try:
                lines, facts = await _build_order_lines(self._item_repo, items)
            except ValueError as err:
                await self._store.invalidate(confirmation_id)
                raise ConfirmationError(
                    _catalog_error_code(str(err)),
                    "商品当前不可售或规格已变化，请重新预览",
                    status=ConfirmationStatus.INVALIDATED,
                ) from err
            current_payload = {
                "user_id": intent.user_id,
                "session_id": intent.session_id,
                "items": facts,
                "shipping_address": _address_payload(address),
            }
            if current_payload != intent.canonical_payload:
                await self._store.invalidate(confirmation_id)
                raise ConfirmationError(
                    _payload_change_code(intent.canonical_payload, current_payload),
                    "商品规格、价格、数量、配送或可售状态已变化，请重新预览",
                    status=ConfirmationStatus.INVALIDATED,
                )
            snapshot = await self._place_order._execute_confirmed(
                buyer_id=intent.user_id,
                items=items,
                shipping_address=address,
            )
            await self._store.consumed(confirmation_id, snapshot)
            return snapshot
        except ConfirmationError:
            raise
        except Exception:
            await self._store.invalidate(confirmation_id)
            raise

    async def prepare_cancel(
        self,
        *,
        session_id: str,
        user_id: str,
        order_id: str,
        reason: str,
        idempotency_key: str | None = None,
        include_token: bool = False,
    ) -> dict[str, Any]:
        order = await self._order_repo.find_by_id(order_id)
        if order is None:
            raise ConfirmationError("order_not_found", f"订单不存在：{order_id}")
        if order.buyer_id != user_id:
            raise ConfirmationError("user_mismatch", "订单不属于当前用户")
        if order.status.value != "CONFIRMED":
            raise ConfirmationError("not_cancellable", "仅 CONFIRMED 订单可取消")
        if not reason.strip():
            raise ConfirmationError("invalid_reason", "取消原因不能为空")
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "order_id": order_id,
            "reason": reason.strip(),
            "order_signature": _order_signature(order),
        }
        expires_at = _now() + self._ttl
        confirmation_id = f"confirm-{uuid.uuid4().hex}"
        token = secrets.token_urlsafe(32)
        preview = {
            "confirmation_required": True,
            "confirmation_id": confirmation_id,
            "action": "cancel_order",
            "expires_at": expires_at.isoformat(),
            "order_id": order_id,
            "reason": reason.strip(),
            "status": order.status.value,
            "total_amount_major": order.total_amount().to_major_units(),
            "currency": order.total_amount().currency,
        }
        intent = await self._store.create(
            ConfirmationIntent(
                confirmation_id=confirmation_id,
                token_hash=_hash_token(token),
                session_id=session_id,
                user_id=user_id,
                action="cancel_order",
                canonical_payload=payload,
                payload_sha256=_payload_sha256(payload),
                preview=preview,
                status=ConfirmationStatus.PENDING,
                expires_at=expires_at,
                idempotency_key=idempotency_key,
                created_at=_now(),
            )
        )
        result = dict(intent.preview)
        if include_token and intent.confirmation_id == confirmation_id:
            result["confirmation_token"] = token
        return result

    async def confirm_cancel(
        self,
        *,
        confirmation_id: str,
        token: str,
        session_id: str,
    ) -> dict[str, Any]:
        intent = await self._store.claim(
            confirmation_id,
            token,
            session_id=session_id,
            action="cancel_order",
        )
        try:
            order_id = str(intent.canonical_payload["order_id"])
            order = await self._order_repo.find_by_id(order_id)
            if order is None:
                await self._store.invalidate(confirmation_id)
                raise ConfirmationError("order_not_found", f"订单不存在：{order_id}")
            if order.buyer_id != intent.user_id:
                await self._store.invalidate(confirmation_id)
                raise ConfirmationError("user_mismatch", "订单不属于当前用户")
            if _order_signature(order) != intent.canonical_payload["order_signature"]:
                await self._store.invalidate(confirmation_id)
                raise ConfirmationError(
                    "order_changed",
                    "订单状态已变化，请重新预览",
                    status=ConfirmationStatus.INVALIDATED,
                )
            snapshot = await self._cancel_order._execute_confirmed(
                order_id,
                str(intent.canonical_payload["reason"]),
            )
            await self._store.consumed(confirmation_id, snapshot)
            return snapshot
        except ConfirmationError:
            raise
        except Exception:
            await self._store.invalidate(confirmation_id)
            raise


def _address_payload(address: Address) -> dict[str, str]:
    return {
        "recipient_name": address.recipient_name,
        "country": address.country,
        "state": address.state,
        "city": address.city,
        "address_line": address.address_line,
        "postal_code": address.postal_code,
        "phone": address.phone,
    }


def _order_signature(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "buyer_id": order.buyer_id,
        "status": order.status.value,
        "lines": [
            {
                "item_id": line.item_id,
                "variant_id": line.variant_id,
                "unit_price_minor": line.unit_price.amount_in_minor_units,
                "currency": line.unit_price.currency,
                "quantity": line.quantity,
                "variant_display_name": line.variant_display_name,
            }
            for line in order.lines
        ],
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _payload_change_code(previous: dict[str, Any], current: dict[str, Any]) -> str:
    previous_items = previous.get("items", [])
    current_items = current.get("items", [])
    if previous_items != current_items:
        if any(
            old.get("unit_price_minor") != new.get("unit_price_minor")
            for old, new in zip(previous_items, current_items, strict=True)
        ):
            return "price_changed"
        if any(
            old.get("availability") != new.get("availability")
            for old, new in zip(previous_items, current_items, strict=True)
        ):
            return "unavailable"
        return "spec_changed"
    if previous.get("shipping_address") != current.get("shipping_address"):
        return "shipping_changed"
    return "payload_changed"


def _catalog_error_code(message: str) -> str:
    if "不可售" in message:
        return "unavailable"
    if "不存在" in message:
        return "spec_changed"
    return "payload_changed"


def _now() -> datetime:
    return datetime.now(timezone.utc)
