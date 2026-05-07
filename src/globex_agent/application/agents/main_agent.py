"""Main agent factory and session registry."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.agents.search_agent import SearchAgentFactory
from globex_agent.application.agents.trade_agent import TradeAgentFactory
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.remember_preference_tool import (
    build_remember_preference_tool,
)
from globex_agent.application.tools.task_dispatch_tool import (
    build_task_dispatch_tool,
)
from globex_agent.domain.buyer.preference import PreferenceStore
from globex_agent.domain.session.ports.session_store import SessionStore
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.llm import create_chat_model
from globex_agent.infrastructure.settings import Settings


class MainAgentFactory:
    def __init__(
        self,
        settings: Settings,
        search_factory: SearchAgentFactory,
        trade_factory: TradeAgentFactory,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        circuit_registry=None,
    ) -> None:
        self._settings = settings
        self._search_factory = search_factory
        self._trade_factory = trade_factory
        self._bus = bus
        self._preference_store = preference_store
        self._circuit_registry = circuit_registry

    def build(self, model: BaseChatModel | None = None) -> LangGraphAgent:
        prompts = load_prompts()["main_agent"]
        tools = [
            *self._search_factory.build_tools(),
            *self._trade_factory.build_tools(),
            build_task_dispatch_tool(
                self._search_factory,
                self._trade_factory,
                self._bus,
            ),
            build_remember_preference_tool(
                self._preference_store,
                self._bus,
            ),
        ]
        graph = create_react_agent(
            model or create_chat_model(self._settings),
            tools=tools,
            prompt=prompts["system_prompt"],
            checkpointer=InMemorySaver(),
        )
        return LangGraphAgent(prompts["name"], graph)


class SessionRegistry:
    """Cache one MainAgent per shopping session."""

    def __init__(
        self,
        main_factory: MainAgentFactory,
        session_store: SessionStore,
    ) -> None:
        self._main_factory = main_factory
        self._session_store = session_store
        self._agents: dict[str, LangGraphAgent] = {}

    async def get_or_create(self, shopping_session_id: str) -> LangGraphAgent:
        if shopping_session_id not in self._agents:
            self._agents[shopping_session_id] = self._main_factory.build()
        return self._agents[shopping_session_id]

    async def persist(self, shopping_session_id: str) -> None:
        """Persist a lightweight marker; message state lives in the graph checkpointer."""

        if shopping_session_id in self._agents:
            await self._session_store.save(
                shopping_session_id,
                '{"session_id":' f'"{shopping_session_id}"' "}",
            )
