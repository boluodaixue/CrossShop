"""Order tool factories using item_id + variant_id."""

from __future__ import annotations

import json
import re

from langchain_core.tools import tool

from globex_agent.application.usecases.confirmation_usecases import (
    ConfirmationError,
    OrderConfirmationService,
)
from globex_agent.application.usecases.order_usecases import (
    OrderItemInput,
    QueryOrderUseCase,
)
from globex_agent.domain.order.address import Address
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def _buyer_id() -> str:
    snapshot = ShoppingContext.current()
    return snapshot.buyer_id if snapshot else "anonymous"


_COUNTRY_ALIASES = {
    "中国": "CN",
    "中华人民共和国": "CN",
}
_CITY_PATTERN = re.compile(r"([\u4e00-\u9fff]+?市)")
_DISTRICT_PATTERN = re.compile(r"([\u4e00-\u9fff]+?区|[\u4e00-\u9fff]+?县)")


def _pick(payload: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _split_city_and_state(address_line: str) -> tuple[str, str]:
    """Extract a Chinese city and province/state from a full address line."""

    city_match = _CITY_PATTERN.search(address_line)
    if city_match is None:
        return "", ""
    city = city_match.group(1)
    prefix = address_line[: city_match.start()].strip()
    district_match = _DISTRICT_PATTERN.search(address_line, city_match.end())
    if district_match is not None:
        district = district_match.group(1)
        if district not in prefix:
            prefix = f"{prefix} {district}".strip()
    return city, prefix


def _address(payload: dict) -> Address:
    recipient_name = _pick(payload, ("recipient_name", "recipient", "name"))
    country = _pick(payload, ("country", "country_code"))
    state = _pick(payload, ("state", "province", "region"))
    city = _pick(payload, ("city", "city_name"))
    address_line = _pick(
        payload,
        (
            "address_line",
            "address",
            "street",
            "street_address",
            "address1",
            "detail_address",
        ),
    )
    district = _pick(payload, ("district", "area"))
    if district and district not in state:
        state = f"{state} {district}".strip()
    if not city and address_line:
        parsed_city, parsed_state = _split_city_and_state(address_line)
        if parsed_city:
            city = parsed_city
        if parsed_state and not state:
            state = parsed_state
    return Address(
        recipient_name=recipient_name,
        country=_COUNTRY_ALIASES.get(country, country),
        state=state,
        city=city,
        address_line=address_line,
        postal_code=_pick(
            payload,
            ("postal_code", "postcode", "zip", "zip_code"),
        ),
        phone=_pick(
            payload,
            ("phone", "telephone", "mobile", "phone_number"),
        ),
    )


def build_prepare_order_tool(
    service: OrderConfirmationService,
    bus: TradeEventBus,
):
    @tool
    async def prepare_order_tool(items: list[dict], shipping_address: dict) -> str:
        """准备订单预览；最终下单必须由用户通过可信确认入口完成。

        Args:
            items: 订单行列表，每项含 item_id / variant_id / quantity。
            shipping_address: 收货地址字典。
                推荐字段：recipient_name / country / province / city /
                district / address_line / postal_code / phone。
        """
        session_id = ShoppingContext.current_session_id()
        buyer_id = _buyer_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {
                "tool": "prepare_order_tool",
                "args": {
                    "buyer_id": buyer_id,
                    "items": items,
                    "shipping_address": shipping_address,
                },
            },
        )
        try:
            order_items = [
                OrderItemInput(
                    item_id=str(item["item_id"]),
                    variant_id=(
                        str(item["variant_id"])
                        if item.get("variant_id") not in (None, "")
                        else None
                    ),
                    quantity=int(item.get("quantity", 1)),
                )
                for item in items
            ]
            preview = await service.prepare_order(
                session_id=session_id,
                user_id=buyer_id,
                items=order_items,
                shipping_address=_address(shipping_address),
                include_token=False,
            )
        except (ValueError, KeyError, ConfirmationError) as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "prepare_order_tool", "error": str(err)},
            )
            return f"[error] {err}"
        bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "prepare_order_tool",
                "confirmation_required": True,
                "confirmation_id": preview["confirmation_id"],
            },
        )
        return json.dumps(preview, ensure_ascii=False)

    return prepare_order_tool


def build_create_order_tool(usecase, bus: TradeEventBus):
    """Legacy name retained as a prepare-only compatibility wrapper."""

    if isinstance(usecase, OrderConfirmationService):
        return build_prepare_order_tool(usecase, bus)

    @tool
    async def create_order_tool(items: list[dict], shipping_address: dict) -> str:
        """Compatibility stub; direct order creation is disabled."""

        del items, shipping_address
        return "[error] 直接 create_order_tool 已禁用，请通过可信 PrepareOrder/ConfirmOrder API"

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


def build_prepare_cancel_tool(
    service: OrderConfirmationService,
    bus: TradeEventBus,
):
    @tool
    async def prepare_cancel_order_tool(order_id: str, reason: str) -> str:
        """准备取消订单预览；最终取消必须由用户通过可信确认入口完成。

        Args:
            order_id: 订单号。
            reason: 取消原因，必填。
        """
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "tool.invoke",
            {
                "tool": "prepare_cancel_order_tool",
                "args": {"order_id": order_id, "reason": reason},
            },
        )
        try:
            preview = await service.prepare_cancel(
                session_id=session_id,
                user_id=_buyer_id(),
                order_id=order_id,
                reason=reason,
                include_token=False,
            )
        except (ValueError, ConfirmationError) as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "prepare_cancel_order_tool", "error": str(err)},
            )
            return f"[error] {err}"
        bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "prepare_cancel_order_tool",
                "confirmation_required": True,
                "confirmation_id": preview["confirmation_id"],
            },
        )
        return json.dumps(preview, ensure_ascii=False)

    return prepare_cancel_order_tool


def build_cancel_order_tool(usecase, bus: TradeEventBus):
    """Legacy name retained as a prepare-only compatibility wrapper."""

    if isinstance(usecase, OrderConfirmationService):
        return build_prepare_cancel_tool(usecase, bus)

    @tool
    async def cancel_order_tool(order_id: str, reason: str) -> str:
        """Compatibility stub; direct cancellation is disabled."""

        del order_id, reason
        return "[error] 直接 cancel_order_tool 已禁用，请通过可信 PrepareCancel/ConfirmCancel API"

    return cancel_order_tool
