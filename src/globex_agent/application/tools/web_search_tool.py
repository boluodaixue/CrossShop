"""Optional Tavily web search tool."""

from __future__ import annotations

import json

import httpx
from langchain_core.tools import tool

from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.settings import Settings

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


def build_web_search_tool(settings: Settings, bus: TradeEventBus):
    @tool
    async def web_search_tool(query: str, max_results: int = 5) -> str:
        """联网搜索外部实时资料（跨境政策 / 关税规则 / 清关限制 / 评测趋势）。

        Args:
            query: 搜索关键词。
            max_results: 返回结果条数，默认 5。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "web_search_tool", "args": {"query": query}},
        )
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    _TAVILY_ENDPOINT,
                    json={
                        "api_key": settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
                response.raise_for_status()
                body = response.json()
            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:500],
                }
                for item in body.get("results", [])
            ]
        except (httpx.HTTPError, ValueError) as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "web_search_tool", "error": str(err)},
            )
            return f"[error] web 搜索失败：{err}"
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "web_search_tool", "hit_count": len(results)},
        )
        return json.dumps({"results": results}, ensure_ascii=False)

    return web_search_tool
