"""A no-network scripted chat model for reproducible AgentLoop tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from globex_agent.domain import (
    ItemPickerOutput,
    ItemSearchOutput,
    PriceCompareOutput,
    ShippingCalcOutput,
    ShoppingSummaryOutput,
)
from globex_agent.tools.planner import PlannerOutput


class ScriptedShoppingModel(BaseChatModel):
    """Emit tool calls from prior ToolMessages without using a remote LLM.

    This model is only the deterministic test double described in the engineering
    plan. The AgentLoop, ToolNode, message appending, checkpoint, and termination
    behavior are still provided by LangGraph.
    """

    _bound_tool_names: set[str] = PrivateAttr(default_factory=set)

    @property
    def _llm_type(self) -> str:
        return "globex-scripted-shopping-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        del tool_choice, kwargs
        self._bound_tool_names = {_tool_name(tool) for tool in tools}
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        message = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        current_turn = _current_turn(messages)
        query = _user_query(current_turn)
        tool_messages = [message for message in current_turn if isinstance(message, ToolMessage)]
        completed_names = [message.name for message in tool_messages]

        if not tool_messages:
            if _looks_like_shopping(query):
                return self._tool_call("planner", {"user_input": query}, 1)
            return self._tool_call("chat_fallback", {"user_input": query}, 1)

        last = tool_messages[-1]
        if last.name == "chat_fallback":
            return AIMessage(content=_text_content(last))
        if last.name == "shopping_summary":
            summary = ShoppingSummaryOutput.model_validate_json(_text_content(last))
            return AIMessage(content=summary.final_text)

        planner_message = _last_named(tool_messages, "planner")
        if planner_message is None:
            return AIMessage(content="Planner 未返回结构化结果，AgentLoop 已安全结束。")
        plan = PlannerOutput.model_validate_json(_text_content(planner_message))

        search_messages = [message for message in tool_messages if message.name == "item_search"]
        if len(search_messages) < len(plan.platforms):
            platform = plan.platforms[len(search_messages)]
            return self._tool_call(
                "item_search",
                {
                    "query": query,
                    "platform": platform.value,
                    "top_k": plan.top_k,
                },
                len(tool_messages) + 1,
            )

        if "price_compare" not in completed_names:
            candidates = [
                candidate.model_dump(mode="json")
                for message in search_messages
                for candidate in ItemSearchOutput.model_validate_json(
                    _text_content(message)
                ).candidates
            ]
            return self._tool_call(
                "price_compare",
                {"candidates": candidates, "base_currency": "CNY", "top_n": 12},
                len(tool_messages) + 1,
            )

        if "shipping_calc" not in completed_names:
            prices = PriceCompareOutput.model_validate_json(
                _text_content(_last_named_required(tool_messages, "price_compare"))
            )
            return self._tool_call(
                "shipping_calc",
                {
                    "points": [point.model_dump(mode="json") for point in prices.ranked],
                    "destination": "CN",
                },
                len(tool_messages) + 1,
            )

        if "item_picker" not in completed_names:
            shipping = ShippingCalcOutput.model_validate_json(
                _text_content(_last_named_required(tool_messages, "shipping_calc"))
            )
            return self._tool_call(
                "item_picker",
                {
                    "landed": [item.model_dump(mode="json") for item in shipping.items],
                    "user_preferences": plan.soft_preferences,
                    "top_n": min(plan.top_k, 3),
                },
                len(tool_messages) + 1,
            )

        selection = ItemPickerOutput.model_validate_json(
            _text_content(_last_named_required(tool_messages, "item_picker"))
        )
        return self._tool_call(
            "shopping_summary",
            {
                "picks": [pick.model_dump(mode="json") for pick in selection.picks],
                "user_query": query,
                "new_preferences": [],
            },
            len(tool_messages) + 1,
        )

    def _tool_call(self, name: str, args: dict[str, Any], index: int) -> AIMessage:
        if self._bound_tool_names and name not in self._bound_tool_names:
            raise RuntimeError(f"script requested unbound tool: {name}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"script-call-{index}",
                    "type": "tool_call",
                }
            ],
        )


def _tool_name(tool: object) -> str:
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", getattr(tool, "__name__", "")))


def _user_query(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _text_content(message)
    return ""


def _current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = index
    return messages[start:]


def _text_content(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return json.dumps(message.content, ensure_ascii=False)


def _last_named(messages: list[ToolMessage], name: str) -> ToolMessage | None:
    return next((message for message in reversed(messages) if message.name == name), None)


def _last_named_required(messages: list[ToolMessage], name: str) -> ToolMessage:
    message = _last_named(messages, name)
    if message is None:
        raise RuntimeError(f"missing scripted tool result: {name}")
    return message


def _looks_like_shopping(query: str) -> bool:
    keywords = {
        "买",
        "预算",
        "商品",
        "价格",
        "比价",
        "耳机",
        "背包",
        "键盘",
        "推荐",
        "搜索",
    }
    return any(keyword in query for keyword in keywords)
