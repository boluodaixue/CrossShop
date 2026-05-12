"""Application tool tests including parallel task dispatch."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from globex_agent.application.tools.order_tools import (
    build_create_order_tool,
)
from globex_agent.application.tools.product_search_tool import (
    build_product_search_tool,
)
from globex_agent.application.tools.task_dispatch_tool import (
    build_task_dispatch_tool,
)
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.application.usecases.order_usecases import PlaceOrderUseCase
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.catalog.models import StandardItem
from globex_agent.infrastructure.context import (
    ShoppingContext,
    ShoppingContextSnapshot,
)
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
    InMemoryOrderRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


class FakeAgent:
    def __init__(self, name: str, delay: float = 0.1) -> None:
        self.name = name
        self.delay = delay

    async def reply(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        event_sink=None,
    ) -> str:
        await asyncio.sleep(self.delay)
        return f"{self.name}-done"


class FakeSearchFactory:
    def __init__(self) -> None:
        self._delay = 0.1

    def build_tools(self):
        return []

    def build(self):
        return FakeAgent("search", self._delay)


class FakeTradeFactory:
    def build_tools(self):
        return []

    def build(self):
        return FakeAgent("trade", 0.1)


class TestProductSearchTool:
    async def test_direct_invoke_returns_cards_and_events(self) -> None:
        catalog = LocalCatalog.from_jsonl(PRODUCTS_PATH, strict=True).catalog
        usecase = CatalogSearchUseCase(
            InMemoryItemRepository(list(catalog.items))
        )
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_product_search_tool(usecase, bus)
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s1",
                buyer_id="b1",
                locale="zh-CN",
                currency="CNY",
            )
        )
        try:
            text = await tool.ainvoke(
                {"normalized_query": "降噪耳机", "top_k": 2}
            )
        finally:
            ShoppingContext.reset(token)
        payload = json.loads(text)
        assert payload["hits"]
        assert payload["hits"][0]["item_id"]
        assert queue.qsize() == 2


class TestOrderTools:
    async def test_create_order_accepts_chinese_address_aliases(self) -> None:
        item = StandardItem(
            item_id="taobao:test:001",
            same_group_id="test-group-001",
            platform="taobao",
            locale="cn",
            language="zh",
            title="测试头戴式耳机",
            category_path=["数码", "耳机"],
            price_cny="68.00",
            currency_raw="CNY",
            variants=[
                {
                    "variant_id": "v1",
                    "price_major": "68.00",
                    "currency": "CNY",
                    "is_available": True,
                }
            ],
            ingested_at=datetime.now(timezone.utc),
            provenance={
                "kind": "synthetic",
                "source": "unit-test",
                "generated_at": datetime.now(timezone.utc),
                "notes": "test fixture",
            },
        )
        variant_id = str(item.variants[0]["variant_id"])
        order_repo = InMemoryOrderRepository()
        tool = build_create_order_tool(
            PlaceOrderUseCase(
                InMemoryItemRepository([item]),
                order_repo,
            ),
            TradeEventBus(),
        )
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="order-tool",
                buyer_id="buyer-001",
                locale="zh-CN",
                currency="CNY",
            )
        )
        try:
            text = await tool.ainvoke(
                {
                    "items": [
                        {
                            "item_id": item.item_id,
                            "variant_id": variant_id,
                            "quantity": 1,
                        }
                    ],
                    "shipping_address": {
                        "recipient": "张三",
                        "country": "中国",
                        "address": "浙江省杭州市西湖区某路1号",
                        "postal_code": "310000",
                        "phone": "13800000000",
                    },
                }
            )
        finally:
            ShoppingContext.reset(token)

        snapshot = json.loads(text)
        assert snapshot["order_id"].startswith("GBX-")
        assert snapshot["status"] == "CONFIRMED"
        assert "杭州市" in snapshot["shipping_address"]


class TestTaskDispatchParallel:
    async def test_multiple_search_agents_overlap(self) -> None:
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_task_dispatch_tool(FakeSearchFactory(), FakeTradeFactory(), bus)
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s1",
                buyer_id="b1",
                locale="zh-CN",
                currency="CNY",
            )
        )
        try:
            text = await tool.ainvoke(
                {
                    "dispatches": [
                        {"subagent_type": "search_agent", "demands": "露营灯"},
                        {"subagent_type": "search_agent", "demands": "登山杖"},
                    ]
                }
            )
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(text)
        assert len(payload["dispatches"]) == 2
        all_events = []
        while not queue.empty():
            all_events.append(queue.get_nowait())
        plan_events = [
            event for event in all_events if event.type == "plan.update"
        ]
        assert plan_events
        assert plan_events[0].payload["tasks"][0]["state"] == "queued"
        events = [
            event.payload
            for event in all_events
            if event.type == "tool.result"
            and event.payload.get("tool") == "task_dispatch"
        ]
        assert len(events) == 2
        starts = [datetime.fromisoformat(event["started_at"]) for event in events]
        ends = [datetime.fromisoformat(event["finished_at"]) for event in events]
        assert starts[0] < ends[1] and starts[1] < ends[0]
