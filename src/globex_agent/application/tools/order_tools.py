"""Order tool factories using item_id + variant_id."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from globex_agent.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from globex_agent.domain.order.address import Address
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def _buyer_id() -> str:
    snapshot = ShoppingContext.current()
    return snapshot.buyer_id if snapshot else "anonymous"


def _address(payload: dict) -> Address:
    return Address(
        recipient_name=payload.get("recipient_name", ""),
        country=payload.get("country", ""),
        state=payload.get("state", ""),
        city=payload.get("city", ""),
        address_line=payload.get("address_line", ""),
        postal_code=payload.get("postal_code", ""),
        phone=payload.get("phone", ""),
    )


def build_create_order_tool(usecase: PlaceOrderUseCase, bus: TradeEventBus):
    @tool
    async def create_order_tool(items: list[dict], shipping_address: dict) -> str:
        """创建订单（直接进入 CONFIRMED 态），买家身份由系统会话上下文自动注入。

        Args:
            items: 订单行列表，每项含 item_id / variant_id / quantity。
            shipping_address: 收货地址字典。
        """
        session_id = ShoppingContext.current_session_id()
        buyer_id = _buyer_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {
                "tool": "create_order_tool",
                "args": {"buyer_id": buyer_id, "items": items},
            },
        )
        try:
            order_items = [
                OrderItemInput(
                    item_id=str(item["item_id"]),
                    variant_id=str(item.get("variant_id", "")),
                    quantity=int(item.get("quantity", 1)),
                )
                for item in items
            ]
            snapshot = await usecase.execute(
                buyer_id=buyer_id,
                items=order_items,
                shipping_address=_address(shipping_address),
            )
        except (ValueError, KeyError) as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "create_order_tool", "error": str(err)},
            )
            return f"[error] {err}"
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "create_order_tool", "order": snapshot},
        )
        return json.dumps(snapshot, ensure_ascii=False)

    return create_order_tool


def build_query_order_tool(usecase: QueryOrderUseCase, bus: TradeEventBus):
    @tool
    async def query_order_tool(order_id: str) -> str:
        """查询订单详情。

        Args:
            order_id: 订单号。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {"tool": "query_order_tool", "args": {"order_id": order_id}},
        )
        try:
            snapshot = await usecase.execute(order_id)
        except ValueError as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "query_order_tool", "error": str(err)},
            )
            return f"[error] {err}"
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "query_order_tool", "order": snapshot},
        )
        return json.dumps(snapshot, ensure_ascii=False)

    return query_order_tool


def build_cancel_order_tool(usecase: CancelOrderUseCase, bus: TradeEventBus):
    @tool
    async def cancel_order_tool(order_id: str, reason: str) -> str:
        """取消订单（仅 CONFIRMED 态可取消）。

        Args:
            order_id: 订单号。
            reason: 取消原因，必填。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {
                "tool": "cancel_order_tool",
                "args": {"order_id": order_id, "reason": reason},
            },
        )
        try:
            snapshot = await usecase.execute(order_id, reason)
        except ValueError as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "cancel_order_tool", "error": str(err)},
            )
            return f"[error] {err}"
        bus.publish(
            session_id,
            "tool.result",
            {"tool": "cancel_order_tool", "order": snapshot},
        )
        return json.dumps(snapshot, ensure_ascii=False)

    return cancel_order_tool
