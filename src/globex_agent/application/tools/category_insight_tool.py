"""Category insight tool backed by the existing CategoryInsightService."""

from __future__ import annotations

import asyncio
import json

from langchain_core.tools import tool

from globex_agent.category_insight.service import CategoryInsightService
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def build_category_insight_tool(
    service: CategoryInsightService,
    bus: TradeEventBus,
):
    @tool
    async def category_insight_tool(category: str, depth: str = "quick") -> str:
        """查询品类洞察知识库：热卖款型、属性口径、价格区间、避坑点。

        Args:
            category: 品类名或带约束的品类短语。
            depth: quick 或 deep，deep 额外返回属性分布。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "category_insight_tool", "args": {"category": category, "depth": depth}},
        )
        try:
            run = await asyncio.to_thread(service.insight, category, depth=depth)
            payload = run.output.model_dump(mode="json")
        except Exception as err:  # noqa: BLE001 - knowledge base may be down
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "category_insight_tool", "error": str(err)},
            )
            return f"[error] 品类知识库不可用：{err}"
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "category_insight_tool", "category": payload.get("category")},
        )
        return json.dumps({"insights": payload}, ensure_ascii=False)

    return category_insight_tool
