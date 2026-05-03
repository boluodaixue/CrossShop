"""JSON and SQL persistence tests for preferences, sessions, and orders."""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from globex_agent.domain.buyer.preference import BuyerPreference
from globex_agent.domain.catalog.money import Money
from globex_agent.domain.order.address import Address
from globex_agent.domain.order.order import Order
from globex_agent.domain.order.order_line import OrderLine
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryOrderRepository,
)
from globex_agent.infrastructure.persistence.json_file_stores import (
    JsonFilePreferenceStore,
    JsonFileSessionStore,
)
from globex_agent.infrastructure.persistence.sql.repositories import (
    SqlOrderRepository,
    bootstrap_schema,
)


def _order(order_id: str = "GBX-000001") -> Order:
    return Order.place(
        order_id=order_id,
        buyer_id="buyer-1",
        shipping_address=Address(
            recipient_name="Pan",
            country="US",
            state="CA",
            city="San Jose",
            address_line="1 Market St",
            postal_code="95110",
            phone="+1-555-0100",
        ),
        lines=[
            OrderLine(
                item_id="amazon:item-1",
                variant_id="amazon:item-1:black",
                title="LumenGo 便携露营灯 可充电",
                unit_price=Money(amount_in_minor_units=8900, currency="CNY"),
                quantity=2,
            ),
        ],
    )


class TestJsonFileStores:
    async def test_preference_append_and_dedupe(self) -> None:
        store = JsonFilePreferenceStore(_scratch_dir())
        preference = BuyerPreference(
            buyer_id="b1",
            kind="dislike",
            statement="不要塑料材质",
        )
        await store.append(preference)
        await store.append(preference)
        assert len(await store.list_by_buyer("b1")) == 1
        assert await store.list_by_buyer("b2") == []

    async def test_session_roundtrip_and_sanitized_name(self) -> None:
        tmp = _scratch_dir()
        store = JsonFileSessionStore(tmp)
        await store.save("s1", '{"session_id":"s1"}')
        assert await store.load("s1") == '{"session_id":"s1"}'
        assert await store.load("missing") is None
        await store.save("../evil", "{}")
        files = list((tmp / "sessions").iterdir())
        assert {path.name for path in files} == {"s1.json", "evil.json"}
        assert all(path.parent == tmp / "sessions" for path in files)


class TestOrderRepositories:
    async def test_in_memory_order_roundtrip(self) -> None:
        repo = InMemoryOrderRepository()
        order_id = await repo.next_order_id()
        await repo.save(_order(order_id))
        restored = await repo.find_by_id("GBX-000001")
        assert restored is not None
        assert restored.lines[0].item_id == "amazon:item-1"
        assert restored.lines[0].variant_id == "amazon:item-1:black"
        assert await repo.next_order_id() == "GBX-000002"

    async def test_sql_order_roundtrip_uses_item_and_variant(self) -> None:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        await bootstrap_schema(engine)
        try:
            repo = SqlOrderRepository(engine)
            await repo.save(_order())
            restored = await repo.find_by_id("GBX-000001")
            assert restored is not None
            assert restored.lines[0].item_id == "amazon:item-1"
            assert restored.lines[0].variant_id == "amazon:item-1:black"
            assert restored.total_amount().amount_in_minor_units == 17800
        finally:
            await engine.dispose()


def _scratch_dir() -> Path:
    root = Path(__file__).resolve().parents[2] / ".pytest_tmp"
    path = root / f"persistence-{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True)
    return path
