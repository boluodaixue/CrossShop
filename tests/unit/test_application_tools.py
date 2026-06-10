"""Application tool tests including parallel task dispatch."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from globex_agent.application.tools.order_tools import (
    _address,
    build_prepare_order_tool,
    build_query_order_tool,
)
from globex_agent.application.tools.product_search_tool import (
    build_product_search_tool,
)
from globex_agent.application.tools.task_dispatch_tool import (
    build_task_dispatch_tool,
)
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.application.usecases.confirmation_usecases import (
    OrderConfirmationService,
)
from globex_agent.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.catalog.models import StandardItem
from globex_agent.infrastructure.checkpoint import InMemoryDispatchResultStore
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
        events = [queue.get_nowait(), queue.get_nowait()]
        result_event = next(event for event in events if event.type == "tool.result")
        snapshots = {
            snapshot["item_id"]: snapshot
            for snapshot in result_event.payload["evidence_snapshots"]
        }
        for hit in result_event.payload["hits"]:
            assert hit["variants"] == snapshots[hit["item_id"]]["exposed_facts"][
                "variants"
            ]


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
            price_cny=None,
            price_source="unavailable",
            availability="available",
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
        variant_id = item.variants[0].variant_id
        order_repo = InMemoryOrderRepository()
        bus = TradeEventBus()
        item_repo = InMemoryItemRepository([item])
        place = PlaceOrderUseCase(item_repo, order_repo)
        cancel = CancelOrderUseCase(order_repo)
        service = OrderConfirmationService(
            item_repo,
            order_repo,
            place,
            cancel,
        )
        tool = build_prepare_order_tool(service, bus)
        query_tool = build_query_order_tool(
            QueryOrderUseCase(order_repo),
            bus,
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
            preview = json.loads(text)
            trusted_preview = await service.prepare_order(
                session_id="order-tool",
                user_id="buyer-001",
                items=[
                    OrderItemInput(
                        item_id=item.item_id,
                        variant_id=variant_id,
                        quantity=1,
                    )
                ],
                shipping_address=_address({
                    "recipient": "张三",
                    "country": "中国",
                    "address": "浙江省杭州市西湖区某路1号",
                    "postal_code": "310000",
                    "phone": "13800000000",
                }),
                include_token=True,
            )
            snapshot = await service.confirm_order(
                confirmation_id=trusted_preview["confirmation_id"],
                token=trusted_preview["confirmation_token"],
                session_id="order-tool",
            )
            query_text = await query_tool.ainvoke({"order_id": snapshot["order_id"]})
            cancel_preview = await service.prepare_cancel(
                session_id="order-tool",
                user_id="buyer-001",
                order_id=snapshot["order_id"],
                reason="买家改主意",
                include_token=True,
            )
            cancelled = await service.confirm_cancel(
                confirmation_id=cancel_preview["confirmation_id"],
                token=cancel_preview["confirmation_token"],
                session_id="order-tool",
            )
        finally:
            ShoppingContext.reset(token)

        assert preview["confirmation_required"] is True
        assert "confirmation_token" not in preview
        assert snapshot["order_id"].startswith("GBX-")
        assert snapshot["status"] == "CONFIRMED"
        assert "杭州市" in snapshot["shipping_address"]
        assert json.loads(query_text)["status"] == "CONFIRMED"
        assert cancelled["status"] == "CANCELLED"


class TestTaskDispatchParallel:
    async def test_multiple_search_agents_overlap(self) -> None:
        bus = TradeEventBus()
        queue = bus.subscribe("s1")
        tool = build_task_dispatch_tool(
            FakeSearchFactory(),
            FakeTradeFactory(),
            bus,
            result_store=InMemoryDispatchResultStore(),
        )
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="s1",
                buyer_id="b1",
                locale="zh-CN",
                currency="CNY",
            )
        )
        try:
            text = await tool.arun(
                {
                    "dispatches": [
                        {"subagent_type": "search_agent", "demands": "露营灯"},
                        {"subagent_type": "search_agent", "demands": "登山杖"},
                    ]
                },
                tool_call_id="unit-dispatch-call",
            )
        finally:
            ShoppingContext.reset(token)

        payload = json.loads(getattr(text, "content", text))
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
        assert len(events) >= 2
        per_dispatch = [event for event in events if event.get("agent")]
        assert len(per_dispatch) == 2
        starts = [datetime.fromisoformat(event["started_at"]) for event in per_dispatch]
        ends = [datetime.fromisoformat(event["finished_at"]) for event in per_dispatch]
        assert starts[0] < ends[1] and starts[1] < ends[0]
