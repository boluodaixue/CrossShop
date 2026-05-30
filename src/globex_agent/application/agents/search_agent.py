"""Search agent factory: product search, category insight, and optional web search."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.category_insight_tool import (
    build_category_insight_tool,
)
from globex_agent.application.tools.product_search_tool import (
    build_product_search_tool,
)
from globex_agent.application.tools.resilient import wrap_tool
from globex_agent.application.tools.web_search_tool import build_web_search_tool
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.category_insight.service import CategoryInsightService
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.llm import create_chat_model
from globex_agent.infrastructure.settings import Settings


class SearchAgentFactory:
    def __init__(
        self,
        settings: Settings,
        catalog_search: CatalogSearchUseCase,
        bus: TradeEventBus,
        category_insight_service: CategoryInsightService,
        circuit_registry=None,
        *,
        checkpointer=None,
    ) -> None:
        self._settings = settings
        self._catalog_search = catalog_search
        self._bus = bus
        self._category_insight_service = category_insight_service
        self._circuit_registry = circuit_registry
        self._checkpointer = checkpointer

    def build_tools(self):
        tools = [
            build_product_search_tool(self._catalog_search, self._bus),
            build_category_insight_tool(
                self._category_insight_service,
                self._bus,
            ),
        ]
        if self._settings.tavily_api_key:
            tools.append(build_web_search_tool(self._settings, self._bus))
        return tools

    def build(
        self,
        model: BaseChatModel | None = None,
        *,
        thread_id: str | None = None,
    ) -> LangGraphAgent:
        if self._checkpointer is None:
            raise RuntimeError(
                "SearchAgentFactory requires a shared checkpointer; "
                "production must not fall back to InMemorySaver"
            )
        prompts = load_prompts()["sub_agents"]["search"]
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
