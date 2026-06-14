"""Main agent factory and session registry."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.channels import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import create_react_agent
from typing_extensions import NotRequired, TypedDict

from globex_agent.application.agents.base import LangGraphAgent
from globex_agent.application.agents.identity import ThreadIdentity
from globex_agent.application.agents.search_agent import SearchAgentFactory
from globex_agent.application.agents.session_lock import SessionLockRegistry
from globex_agent.application.agents.trade_agent import TradeAgentFactory
from globex_agent.application.context.assembler import build_context_pre_model_hook
from globex_agent.application.context.budget import ContextBudgetPolicy
from globex_agent.application.context.tool_output import (
    ToolArtifactStore,
    apply_tool_output_contract,
)
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.remember_preference_tool import (
    build_remember_preference_tool,
)
from globex_agent.application.tools.resilient import wrap_tool
from globex_agent.application.tools.task_dispatch_tool import (
    build_task_dispatch_tool,
)
from globex_agent.domain.buyer.preference import PreferenceStore
from globex_agent.infrastructure.checkpoint import DispatchResultStore
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.llm import create_chat_model
from globex_agent.infrastructure.settings import Settings


class MainAgentState(TypedDict):
    """Checkpointed main-agent context, lifecycle and L1 observation state."""

    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]
    session_context: NotRequired[dict[str, Any]]
    frozen_segments: NotRequired[list[dict[str, Any]]]
    freeze_cursor: NotRequired[int]
    stage_summary: NotRequired[dict[str, Any] | None]
    budget_report: NotRequired[dict[str, Any]]
    budget_decision: NotRequired[str]
    context_compression: NotRequired[dict[str, Any]]
    llm_input_messages: NotRequired[
        Annotated[list[BaseMessage], EphemeralValue]
    ]


class MainAgentFactory:
    def __init__(
        self,
        settings: Settings,
        search_factory: SearchAgentFactory,
        trade_factory: TradeAgentFactory,
        bus: TradeEventBus,
        preference_store: PreferenceStore,
        circuit_registry=None,
        *,
        checkpointer=None,
        identity: ThreadIdentity | None = None,
        dispatch_result_store: DispatchResultStore | None = None,
    ) -> None:
        self._settings = settings
        self._search_factory = search_factory
        self._trade_factory = trade_factory
        self._bus = bus
        self._preference_store = preference_store
        self._circuit_registry = circuit_registry
        self._checkpointer = checkpointer
        self._identity = identity or ThreadIdentity(
            environment=settings.checkpoint_environment,
            hmac_key=settings.thread_id_hmac_key,
        )
        self._dispatch_result_store = dispatch_result_store

    @property
    def identity(self) -> ThreadIdentity:
        return self._identity

    def build(
        self,
        model: BaseChatModel | None = None,
        *,
        thread_id: str | None = None,
    ) -> LangGraphAgent:
        if self._checkpointer is None:
            raise RuntimeError(
                "MainAgentFactory requires a shared checkpointer; "
                "production must not fall back to InMemorySaver"
            )
        if self._dispatch_result_store is None:
            raise RuntimeError(
                "MainAgentFactory requires a dispatch result store; "
                "production must not use an in-memory completion marker"
            )
        prompts = load_prompts()["main_agent"]
        tools = [
            *self._search_factory.build_tools(),
            *self._trade_factory.build_tools(),
        ]
        tools.append(
            build_task_dispatch_tool(
                self._search_factory,
                self._trade_factory,
                self._bus,
                identity=self._identity,
                result_store=self._dispatch_result_store,
            )
        )
        tools.append(
            build_remember_preference_tool(
                self._preference_store,
                self._bus,
            )
        )
        artifact_store = ToolArtifactStore(
            self._settings.data_dir / "runtime" / "tool_artifacts"
        )
        tools = [
            apply_tool_output_contract(
                tool,
                artifact_store=artifact_store,
                configured_limit=self._settings.tool_result_limit,
            )
            for tool in tools
        ]
        if self._circuit_registry is not None:
            tools = [
                wrap_tool(tool, self._circuit_registry, self._bus)
                for tool in tools
        ]
        chat_model = model or create_chat_model(self._settings, self._bus)
        tool_schemas = [convert_to_openai_tool(tool) for tool in tools]
        pre_model_hook = build_context_pre_model_hook(
            model=chat_model,
            fixed_prompt=prompts["system_prompt"],
            tool_schemas=tool_schemas,
            budget_policy=ContextBudgetPolicy(
                model_context_tokens=self._settings.context_size,
                reply_reserved_tokens=self._settings.reply_token_budget,
                safety_margin_tokens=self._settings.context_safety_margin_tokens,
                soft_limit_tokens=self._settings.context_soft_limit_tokens,
                mode=self._settings.context_budget_mode,
                l3_keep_recent_frozen_segments=(
                    self._settings.l3_keep_recent_frozen_segments
                ),
            ),
        )
        graph = create_react_agent(
            chat_model,
            tools=tools,
            prompt=prompts["system_prompt"],
            pre_model_hook=pre_model_hook,
            state_schema=MainAgentState,
            checkpointer=self._checkpointer,
        )
        return LangGraphAgent(prompts["name"], graph, thread_id=thread_id)


class SessionRegistry:
    """Cache one MainAgent per shopping session."""

    def __init__(
        self,
        main_factory: MainAgentFactory,
        identity: ThreadIdentity | None = None,
    ) -> None:
        self._main_factory = main_factory
        self._identity = identity or main_factory.identity
        self._agents: dict[str, LangGraphAgent] = {}
        self._locks = SessionLockRegistry()

    def thread_id(self, shopping_session_id: str) -> str:
        return self._identity.main_thread_id(shopping_session_id)

    async def get_or_create(self, shopping_session_id: str) -> LangGraphAgent:
        if shopping_session_id not in self._agents:
            agent = self._main_factory.build(thread_id=self.thread_id(shopping_session_id))
            self._agents[shopping_session_id] = agent
        return self._agents[shopping_session_id]

    def session_lock(self, shopping_session_id: str):
        """Return the process-local lock for one anonymous main thread."""

        return self._locks.acquire(self.thread_id(shopping_session_id))

    @property
    def lock_count(self) -> int:
        return self._locks.size
