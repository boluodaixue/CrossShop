"""Product search tool factory backed by CatalogSearchUseCase."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def build_product_search_tool(
    usecase: CatalogSearchUseCase,
    bus: TradeEventBus,
):
    @tool
    async def product_search_tool(
        normalized_query: str,
        category: str | None = None,
        ship_to: str | None = None,
        top_k: int = 5,
        price_max_major: float | None = None,
        target_currency: str = "CNY",
    ) -> str:
        """检索跨境商品库（embedding+rerank 二阶段召回），返回 Top-K 商品卡 JSON。

        Args:
            normalized_query: 标准化检索词，保留品类词与关键属性词。
            category: 品类槽位，可选。
            ship_to: 收货国家二位码，可选；传入后过滤不可送达商品并内联到手价。
            top_k: 返回候选数量，默认 5。
            price_max_major: 价格上限（target_currency 主单位），可选。
            target_currency: 价格口径币种，默认 CNY。
        """
        session_id = ShoppingContext.current_session_id()
        args = {
            "normalized_query": normalized_query,
            "category": category,
            "ship_to": ship_to,
            "top_k": top_k,
            "price_max_major": price_max_major,
            "target_currency": target_currency,
        }
        bus.publish(session_id, "tool.invoke", {"tool": "product_search_tool", "args": args})
        try:
            spec = ProductSearchSpec(
                normalized_query=normalized_query,
                category=category,
                ship_to=ship_to,
                top_k=top_k,
                price_max_major=price_max_major,
                target_currency=target_currency,
            )
            result = await usecase.execute(spec)
        except ValueError as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "product_search_tool", "error": str(err)},
            )
            return f"[error] {err}"
        bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "product_search_tool",
                "hit_count": len(result["hits"]),
                "recall_strategy": result["recall_strategy"],
                "hits": result["hits"],
            },
        )
        return json.dumps(result, ensure_ascii=False)

    return product_search_tool
