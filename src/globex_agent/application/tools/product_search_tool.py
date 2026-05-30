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
        platform: str | None = None,
        locale: str | None = None,
        brand: str | None = None,
        top_k: int = 5,
        price_max_major: float | None = None,
        target_currency: str = "CNY",
        material_include: list[str] | None = None,
        material_exclude: list[str] | None = None,
        material_unknown_policy: str = "warn",
    ) -> str:
        """检索跨境商品库（embedding+rerank 二阶段召回），返回 Top-K 商品卡 JSON。

        Args:
            normalized_query: 标准化检索词，保留品类词与关键属性词。
            category: 品类槽位，可选。
            ship_to: 收货国家二位码，可选；传入后过滤不可送达商品并内联到手价。
            platform: 用户明确指定的平台，可选。
            locale: 用户明确指定的商品站点 locale，可选（如 cn / us）。
            brand: 用户明确指定的品牌，可选。
            top_k: 返回候选数量，默认 5。
            price_max_major: 价格上限（target_currency 主单位），可选。
            target_currency: 价格口径币种，默认 CNY。
        """
        session_id = ShoppingContext.current_session_id()
        args = {
            "normalized_query": normalized_query,
            "category": category,
            "ship_to": ship_to,
            "platform": platform,
            "locale": locale,
            "brand": brand,
            "top_k": top_k,
            "price_max_major": price_max_major,
            "target_currency": target_currency,
            "material_include": material_include or [],
            "material_exclude": material_exclude or [],
            "material_unknown_policy": material_unknown_policy,
        }
        bus.publish(session_id, "tool.invoke", {"tool": "product_search_tool", "args": args})
        try:
            spec = ProductSearchSpec(
                normalized_query=normalized_query,
                category=category,
                ship_to=ship_to,
                platform=platform,
                locale=locale,
                brand=brand,
                top_k=top_k,
                price_max_major=price_max_major,
                target_currency=target_currency,
                material_include=tuple(material_include or ()),
                material_exclude=tuple(material_exclude or ()),
                material_unknown_policy=material_unknown_policy,
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
