# -*- coding: utf-8 -*-
"""category_insight_tool

品类洞察工具（RAG）：目录参考、选购兼容性与有来源的跨境运输注意事项。
与 product_search_tool（当前商品清单）分工明确，运输资料不构成当次承运许可。

注意：本模块不能用 `from __future__ import annotations`（AgentScope schema 生成依赖运行时注解）。
"""

import json

from agentscope.message import TextBlock, ToolResultState
from agentscope.rag import KnowledgeBase
from agentscope.tool import ToolChunk

from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus


def _chunk_text(content) -> str:
    """Chunk.content 是 TextBlock / DataBlock 而非纯字符串，统一归一为可序列化文本。"""
    if isinstance(content, str):
        return content
    text = getattr(content, "text", None)
    if text is not None:
        return text
    if isinstance(content, dict):
        return content.get("text") or str(content)
    return str(content)


def build_category_insight_tool(knowledge_base: KnowledgeBase, bus: TradeEventBus):
    async def category_insight_tool(question: str, top_k: int = 3) -> ToolChunk:
        """查询知识库：目录参考、选购兼容性、包装、电池寄运及清关资料注意事项。

        目录样本不代表实时热销、销量排行、市场份额或实时价格；需要当前商品清单与价格时
        使用 product_search_tool。运输片段须连同来源、地区和查阅日期使用；不承诺实时
        承运许可、免税额度、税率或到货日，缺少目的地/服务信息时先澄清或查官方最新说明。

        Args:
            question (`str`):
                自然语言问题，建议带上品类词，如"旅行装备有哪些历史款型线索"。
            top_k (`int`):
                返回知识片段数量，默认 3。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {
                "tool": "category_insight_tool",
                "args": {"question": question, "top_k": top_k},
            },
        )
        try:
            if not question.strip() or not 1 <= top_k <= 5:
                raise ValueError("question不能为空，top_k必须为1..5")
            results = await knowledge_base.search(queries=[question], top_k=top_k)
        except Exception as err:  # noqa: BLE001 —— 知识库不可用时如实降级，不编造洞察
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "category_insight_tool", "error": str(err)},
            )
            return ToolChunk(
                content=[
                    TextBlock(type="text", text=f"[error] 品类知识库不可用：{err}")
                ],
                state=ToolResultState.ERROR,
            )

        insights = [
            {
                "content": _chunk_text(item.chunk.content),
                "source": item.chunk.metadata.get("source", item.document_id)
                if item.chunk.metadata
                else item.document_id,
                "score": round(item.score, 4),
                "document_id": item.document_id,
                "provenance": {
                    key: item.chunk.metadata[key]
                    for key in ("url", "publisher", "checked_on", "scope", "section",
                                "source_kind", "index", "rerank_status", "score_kind")
                    if item.chunk.metadata and key in item.chunk.metadata
                },
            }
            for item in results
        ]
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "category_insight_tool", "hit_count": len(insights)},
        )
        return ToolChunk(
            content=[
                TextBlock(
                    type="text",
                    text=json.dumps({"insights": insights}, ensure_ascii=False),
                )
            ],
            state=ToolResultState.SUCCESS,
        )

    return category_insight_tool
