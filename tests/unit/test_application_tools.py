"""Application tool tests including parallel task dispatch."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from globex_agent.application.tools.product_search_tool import (
    build_product_search_tool,
)
from globex_agent.application.tools.task_dispatch_tool import (
    build_task_dispatch_tool,
)
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.catalog import LocalCatalog
from globex_agent.infrastructure.context import (
    ShoppingContext,
    ShoppingContextSnapshot,
)
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


class FakeAgent:
    def __init__(self, name: str, delay: float = 0.1) -> None:
        self.name = name
        self.delay = delay

    async def reply(self, query: str, *, thread_id: str | None = None) -> str:
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
        events = []
        while not queue.empty():
            event = queue.get_nowait()
            if event.type == "tool.result" and event.payload.get("tool") == "task_dispatch":
                events.append(event.payload)
        assert len(events) == 2
        starts = [datetime.fromisoformat(event["started_at"]) for event in events]
        ends = [datetime.fromisoformat(event["finished_at"]) for event in events]
        assert starts[0] < ends[1] and starts[1] < ends[0]
