"""Long-term buyer preference write tool."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool

from globex_agent.domain.buyer.preference import BuyerPreference, PreferenceStore
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def build_remember_preference_tool(store: PreferenceStore, bus: TradeEventBus):
    @tool
    async def remember_preference_tool(
        kind: Literal["like", "dislike"],
        statement: str,
    ) -> str:
        """记住买家的一条长期偏好（跨会话生效）。

        Args:
            kind: like 或 dislike。
            statement: 一句话偏好陈述。
        """
        snapshot = ShoppingContext.current()
        buyer_id = snapshot.buyer_id if snapshot else "anonymous"
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {
                "tool": "remember_preference_tool",
                "args": {"kind": kind, "statement": statement},
            },
        )
        try:
            await store.append(
                BuyerPreference(
                    buyer_id=buyer_id,
                    kind=kind,
                    statement=statement,
                )
            )
        except ValueError as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "remember_preference_tool", "error": str(err)},
            )
            return f"[error] {err}"
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "remember_preference_tool", "saved": statement},
        )
        return f"已记住买家偏好：[{kind}] {statement}"

    return remember_preference_tool
