"""P0-003 trusted order confirmation gates."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from globex_agent.application.tools.order_tools import (
    build_cancel_order_tool,
    build_create_order_tool,
)
from globex_agent.application.usecases.confirmation_usecases import (
    ConfirmationError,
    ConfirmationStatus,
    OrderConfirmationService,
)
from globex_agent.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
)
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.order.address import Address
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
    InMemoryOrderRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


def _address() -> Address:
    return Address(
        recipient_name="张三",
        country="CN",
        state="浙江",
        city="杭州",
        address_line="西湖区某路 1 号",
        postal_code="310000",
        phone="13800000000",
    )


@pytest.fixture()
def fixture_service():
    catalog = LocalCatalog.from_jsonl(PRODUCTS_PATH, strict=True).catalog
    item_repo = InMemoryItemRepository(list(catalog.items))
    order_repo = InMemoryOrderRepository()
    place = PlaceOrderUseCase(item_repo, order_repo)
    cancel = CancelOrderUseCase(order_repo)
    service = OrderConfirmationService(item_repo, order_repo, place, cancel)
    return service, item_repo, order_repo


async def _prepare(service: OrderConfirmationService, **kwargs):
    return await service.prepare_order(
        session_id=kwargs.pop("session_id", "s1"),
        user_id=kwargs.pop("user_id", "b1"),
        items=kwargs.pop(
            "items",
            [OrderItemInput("amazon:aqp-1001", None, 1)],
        ),
        shipping_address=kwargs.pop("shipping_address", _address()),
        include_token=True,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_token_is_hashed_and_confirm_is_one_time(fixture_service) -> None:
    service, _, order_repo = fixture_service
    prepared = await _prepare(service)
    intent = await service.store.get(prepared["confirmation_id"])
    assert intent is not None
    assert prepared["confirmation_token"] not in intent.__dict__.values()
    assert intent.token_hash != prepared["confirmation_token"]

    created = await service.confirm_order(
        confirmation_id=prepared["confirmation_id"],
        token=prepared["confirmation_token"],
        session_id="s1",
    )
    assert created["status"] == "CONFIRMED"
    assert len(order_repo._orders) == 1
    with pytest.raises(ConfirmationError, match="确认已使用") as replay:
        await service.confirm_order(
            confirmation_id=prepared["confirmation_id"],
            token=prepared["confirmation_token"],
            session_id="s1",
        )
    assert replay.value.code == "already_used"


@pytest.mark.asyncio
async def test_invalid_token_and_cross_session_do_not_consume(fixture_service) -> None:
    service, _, _ = fixture_service
    prepared = await _prepare(service)
    with pytest.raises(ConfirmationError) as wrong:
        await service.confirm_order(
            confirmation_id=prepared["confirmation_id"],
            token="wrong",
            session_id="s1",
        )
    assert wrong.value.code == "invalid_token"
    with pytest.raises(ConfirmationError) as cross:
        await service.confirm_order(
            confirmation_id=prepared["confirmation_id"],
            token=prepared["confirmation_token"],
            session_id="other-session",
        )
    assert cross.value.code == "session_mismatch"
    created = await service.confirm_order(
        confirmation_id=prepared["confirmation_id"],
        token=prepared["confirmation_token"],
        session_id="s1",
    )
    assert created["status"] == "CONFIRMED"


@pytest.mark.asyncio
async def test_expiry_and_payload_tamper_are_terminal(fixture_service) -> None:
    service, _, _ = fixture_service
    expired = await _prepare(service)
    expired_intent = await service.store.get(expired["confirmation_id"])
    assert expired_intent is not None
    expired_intent.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    with pytest.raises(ConfirmationError) as expired_error:
        await service.confirm_order(
            confirmation_id=expired["confirmation_id"],
            token=expired["confirmation_token"],
            session_id="s1",
        )
    assert expired_error.value.code == "expired"
    assert expired_intent.status is ConfirmationStatus.EXPIRED

    tampered = await _prepare(service)
    tampered_intent = await service.store.get(tampered["confirmation_id"])
    assert tampered_intent is not None
    tampered_intent.canonical_payload["items"][0]["quantity"] = 99
    with pytest.raises(ConfirmationError) as tamper_error:
        await service.confirm_order(
            confirmation_id=tampered["confirmation_id"],
            token=tampered["confirmation_token"],
            session_id="s1",
        )
    assert tamper_error.value.code == "payload_tampered"
    assert tampered_intent.status is ConfirmationStatus.INVALIDATED


@pytest.mark.asyncio
async def test_price_and_availability_changes_invalidate(fixture_service) -> None:
    service, item_repo, _ = fixture_service
    prepared = await _prepare(service)
    item = await item_repo.find_by_id("amazon:aqp-1001")
    assert item is not None
    item.price_cny = item.price_cny + 1
    with pytest.raises(ConfirmationError) as price_error:
        await service.confirm_order(
            confirmation_id=prepared["confirmation_id"],
            token=prepared["confirmation_token"],
            session_id="s1",
        )
    assert price_error.value.code == "price_changed"

    prepared = await _prepare(service)
    item.availability = "unavailable"
    with pytest.raises(ConfirmationError) as availability_error:
        await service.confirm_order(
            confirmation_id=prepared["confirmation_id"],
            token=prepared["confirmation_token"],
            session_id="s1",
        )
    assert availability_error.value.code == "unavailable"


@pytest.mark.asyncio
async def test_idempotency_and_concurrent_double_click(fixture_service) -> None:
    service, _, order_repo = fixture_service
    first = await _prepare(service, idempotency_key="checkout-1")
    same = await _prepare(service, idempotency_key="checkout-1")
    assert same["confirmation_id"] == first["confirmation_id"]
    assert "confirmation_token" not in same

    results = await asyncio.gather(
        service.confirm_order(
            confirmation_id=first["confirmation_id"],
            token=first["confirmation_token"],
            session_id="s1",
        ),
        service.confirm_order(
            confirmation_id=first["confirmation_id"],
            token=first["confirmation_token"],
            session_id="s1",
        ),
        return_exceptions=True,
    )
    successes = [result for result in results if isinstance(result, dict)]
    errors = [result for result in results if isinstance(result, ConfirmationError)]
    assert len(successes) == 1
    assert len(errors) == 1
    assert errors[0].code in {"in_progress", "already_used"}
    assert successes[0]["status"] == "CONFIRMED"
    assert order_repo._orders.keys() == {successes[0]["order_id"]}


@pytest.mark.asyncio
async def test_legacy_write_tools_are_non_mutating_stubs(fixture_service) -> None:
    _service, item_repo, order_repo = fixture_service
    bus = TradeEventBus()
    create = build_create_order_tool(
        PlaceOrderUseCase(item_repo, order_repo),
        bus,
    )
    cancel = build_cancel_order_tool(CancelOrderUseCase(order_repo), bus)
    create_result = await create.ainvoke({"items": [], "shipping_address": {}})
    cancel_result = await cancel.ainvoke({"order_id": "GBX-1", "reason": "x"})
    assert "直接 create_order_tool 已禁用" in create_result
    assert "直接 cancel_order_tool 已禁用" in cancel_result
    assert order_repo._orders == {}
