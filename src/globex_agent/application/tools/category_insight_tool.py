"""Category insight tool backed by the existing CategoryInsightService."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from langchain_core.tools import tool

from globex_agent.category_insight.service import CategoryInsightService
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus

_SOURCE_BOUNDARY = {
    "scope": "category_aggregate_reference",
    "category": "结果是品类聚合/参考知识，不是具体商品事实。",
    "bestsellers": "bestsellers 是目录高频款型代理/高频形态，不是真实销量榜。",
    "attributes": (
        "attributes 百分比是当前目录样本中的标题/属性出现率，不代表市场规律；"
        "护颈、助眠、矫正等功效词仅表示商品标题声明，未验证功效。"
    ),
    "price_tiers": "price_tiers 仅表示该品类的参考价格区间。",
    "exclusions": (
        "CategoryInsight 不证明具体商品实时价格、店铺、库存或具体 SKU；"
        "这些事实只能引用 product_search_tool。"
    ),
}


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
        tool_call_id = f"category-insight-{uuid4().hex}"
        bus.publish(
            session_id,
            "tool.invoke",
            {
                "tool": "category_insight_tool",
                "tool_call_id": tool_call_id,
                "args": {"category": category, "depth": depth},
            },
        )
        try:
            run = await asyncio.to_thread(service.insight, category, depth=depth)
            payload = run.output.model_dump(mode="json")
            evidence_refs = [ref.model_dump(mode="json") for ref in run.diagnostics.evidence_refs]
            category = str(payload.get("category") or category)
            result = {
                "insights": payload,
                # Provenance is an audit/source boundary, not a field-level
                # evidence catalog and not a model-generated citation list.
                "provenance": evidence_refs,
                "source_boundary": _SOURCE_BOUNDARY,
            }
        except Exception as err:  # noqa: BLE001 - knowledge base may be down
            bus.publish(
                session_id,
                "tool.result",
                {
                    "tool": "category_insight_tool",
                    "tool_call_id": tool_call_id,
                    "error": str(err),
                    "model_output": {"error": str(err)},
                },
            )
            return f"[error] 品类知识库不可用：{err}"
        bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "category_insight_tool",
                "tool_call_id": tool_call_id,
                "category": payload.get("category"),
                "insights": payload,
                "provenance": result["provenance"],
                "source_boundary": _SOURCE_BOUNDARY,
                "model_output": result,
            },
        )
        return json.dumps(result, ensure_ascii=False)

    return category_insight_tool
