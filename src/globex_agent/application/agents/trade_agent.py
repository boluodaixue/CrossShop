"""Trade agent factory: order create, query, and cancel tools."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.order_tools import (
    build_cancel_order_tool,
    build_create_order_tool,
    build_query_order_tool,
)
from globex_agent.application.usecases.order_usecases import (
    CancelOrderUseCase,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.llm import create_chat_model
from globex_agent.infrastructure.settings import Settings


class TradeAgentFactory:
    def __init__(
        self,
        settings: Settings,
        place_order: PlaceOrderUseCase,
        query_order: QueryOrderUseCase,
        cancel_order: CancelOrderUseCase,
        bus: TradeEventBus,
        circuit_registry=None,
    ) -> None:
        self._settings = settings
        self._place_order = place_order
        self._query_order = query_order
        self._cancel_order = cancel_order
        self._bus = bus
        self._circuit_registry = circuit_registry

    def build_tools(self):
        return [
            build_create_order_tool(self._place_order, self._bus),
            build_query_order_tool(self._query_order, self._bus),
            build_cancel_order_tool(self._cancel_order, self._bus),
        ]

    def build(self, model: BaseChatModel | None = None) -> LangGraphAgent:
        prompts = load_prompts()["sub_agents"]["trade"]
        graph = create_react_agent(
            model or create_chat_model(self._settings),
            tools=self.build_tools(),
            prompt=prompts["system_prompt"],
        )
        return LangGraphAgent(prompts["name"], graph)
