"""Trade agent factory: order create, query, and cancel tools."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.order_tools import (
    build_prepare_cancel_tool,
    build_prepare_order_tool,
    build_query_order_tool,
)
from globex_agent.application.tools.resilient import wrap_tool
from globex_agent.application.usecases.confirmation_usecases import (
    OrderConfirmationService,
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
        confirmation_service: OrderConfirmationService | None = None,
        *,
        checkpointer=None,
    ) -> None:
        self._settings = settings
        self._place_order = place_order
        self._query_order = query_order
        self._cancel_order = cancel_order
        self._confirmation_service = confirmation_service or OrderConfirmationService(
            place_order._item_repo,
            cancel_order._order_repo,
            place_order,
            cancel_order,
        )
        self._bus = bus
        self._circuit_registry = circuit_registry
        self._checkpointer = checkpointer

    def build_tools(self):
        tools = [
            build_prepare_order_tool(self._confirmation_service, self._bus),
            build_query_order_tool(self._query_order, self._bus),
            build_prepare_cancel_tool(self._confirmation_service, self._bus),
        ]
        return tools

    def build(
        self,
        model: BaseChatModel | None = None,
        *,
        thread_id: str | None = None,
    ) -> LangGraphAgent:
        if self._checkpointer is None:
            raise RuntimeError(
                "TradeAgentFactory requires a shared checkpointer; "
                "production must not fall back to InMemorySaver"
            )
        prompts = load_prompts()["sub_agents"]["trade"]
        tools = self.build_tools()
        if self._circuit_registry is not None:
            tools = [
                wrap_tool(tool, self._circuit_registry, self._bus)
                for tool in tools
            ]
        graph = create_react_agent(
            model or create_chat_model(self._settings, self._bus),
            tools=tools,
            prompt=prompts["system_prompt"],
            checkpointer=self._checkpointer,
        )
        return LangGraphAgent(prompts["name"], graph, thread_id=thread_id)
