"""Main agent LangGraph tests for solo and dispatch paths."""

from __future__ import annotations

import asyncio
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import PrivateAttr

from globex_agent.application.agents.main_agent import (
    MainAgentFactory,
    SessionRegistry,
)
from globex_agent.application.agents.search_agent import SearchAgentFactory
from globex_agent.application.agents.trade_agent import TradeAgentFactory
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.application.usecases.order_usecases import (
    CancelOrderUseCase,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from globex_agent.catalog import LocalCatalog
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
    InMemoryOrderRepository,
)
from globex_agent.infrastructure.persistence.json_file_stores import (
    JsonFilePreferenceStore,
)
from globex_agent.infrastructure.settings import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


class ScriptedMigrationModel(BaseChatModel):
    """Emit one scripted tool call, then answer after its result."""

    _bound_tool_names: set[str] = PrivateAttr(default_factory=set)

    @property
    def _llm_type(self) -> str:
        return "globex-scripted-migration-model"

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: str | None = None,
        **kwargs,
    ) -> Runnable:
        del tool_choice, kwargs
        self._bound_tool_names = {
            tool.name if hasattr(tool, "name") else str(tool)
            for tool in tools
        }
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        last_tool = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ToolMessage)
            ),
            None,
        )
        if last_tool is not None:
            if last_tool.name == "product_search_tool":
                return AIMessage(content="单干完成")
            if last_tool.name == "task_dispatch":
                return AIMessage(content="派发完成")
        query = next(
            (
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        if "下单" in str(query):
            return self._call(
                "task_dispatch",
                {
                    "dispatches": [
                        {
                            "subagent_type": "trade_agent",
                            "demands": "创建订单",
                        }
                    ]
                },
                1,
            )
        if "派发" in str(query):
            return self._call(
                "task_dispatch",
                {
                    "dispatches": [
                        {
                            "subagent_type": "search_agent",
                            "demands": "露营灯",
                        }
                    ]
                },
                1,
            )
        return self._call(
            "product_search_tool",
            {"normalized_query": "降噪耳机", "top_k": 1},
            1,
        )

    def _call(self, name: str, args: dict, index: int) -> AIMessage:
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


class RecallChatModel(BaseChatModel):
    """Return the previous assistant message on the second turn."""

    @property
    def _llm_type(self) -> str:
        return "globex-recall-model"

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: str | None = None,
        **kwargs,
    ) -> Runnable:
        del tools, tool_choice, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=f"recalled:{message.content}"
                            )
                        )
                    ]
                )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="first"))]
        )


class FirstHumanRecallModel(BaseChatModel):
    """Return the first human message from history to prove checkpoint isolation."""

    @property
    def _llm_type(self) -> str:
        return "globex-first-human-recall-model"

    def bind_tools(
        self,
        tools,
        *,
        tool_choice: str | None = None,
        **kwargs,
    ) -> Runnable:
        del tools, tool_choice, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        humans = [
            message.content
            for message in messages
            if isinstance(message, HumanMessage)
        ]
        if humans:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"recalled:{humans[0]}")
                    )
                ]
            )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="first"))]
        )


class FakeSubAgent:
    def __init__(self, name: str) -> None:
        self.name = name

    async def reply(self, query: str, *, thread_id: str | None = None) -> str:
        return f"{self.name}-done"


class DictSessionStore:
    def __init__(self, data: dict[str, str] | None = None) -> None:
        self._data = data if data is not None else {}

    async def save(self, session_id: str, state_json: str) -> None:
        self._data[session_id] = state_json

    async def load(self, session_id: str) -> str | None:
        return self._data.get(session_id)


def _build_main_factory(model: ScriptedMigrationModel) -> MainAgentFactory:
    settings = load_settings()
    bus = TradeEventBus()
    catalog = LocalCatalog.from_jsonl(PRODUCTS_PATH, strict=True).catalog
    item_repo = InMemoryItemRepository(list(catalog.items))
    order_repo = InMemoryOrderRepository()
    catalog_search = CatalogSearchUseCase(item_repo)
    place_order = PlaceOrderUseCase(item_repo, order_repo)
    query_order = QueryOrderUseCase(order_repo)
    cancel_order = CancelOrderUseCase(order_repo)
    search_factory = SearchAgentFactory(
        settings,
        catalog_search,
        bus,
        object(),  # CategoryInsightService is not invoked by the scripted path
    )
    trade_factory = TradeAgentFactory(
        settings,
        place_order,
        query_order,
        cancel_order,
        bus,
    )
    search_factory.build = lambda: FakeSubAgent("search")
    trade_factory.build = lambda: FakeSubAgent("trade")
    return MainAgentFactory(
        settings,
        search_factory,
        trade_factory,
        bus,
        JsonFilePreferenceStore(PROJECT_ROOT / "data"),
    )


class TestMainAgentPaths:
    async def test_solo_path(self) -> None:
        agent = _build_main_factory(ScriptedMigrationModel()).build(
            model=ScriptedMigrationModel()
        )
        assert await agent.reply("帮我找降噪耳机") == "单干完成"

    async def test_dispatch_search_path(self) -> None:
        agent = _build_main_factory(ScriptedMigrationModel()).build(
            model=ScriptedMigrationModel()
        )
        assert await agent.reply("派发检索子代理") == "派发完成"

    async def test_dispatch_trade_path(self) -> None:
        agent = _build_main_factory(ScriptedMigrationModel()).build(
            model=ScriptedMigrationModel()
        )
        assert await agent.reply("派发下单子代理") == "派发完成"

    async def test_multiturn_context_retained_with_checkpointer(self) -> None:
        agent = _build_main_factory(RecallChatModel()).build(
            model=RecallChatModel()
        )
        assert await agent.reply("first", thread_id="same-thread") == "first"
        assert (
            await agent.reply("second", thread_id="same-thread")
            == "recalled:first"
        )

    async def test_checkpoint_roundtrip_restores_context(self) -> None:
        first = _build_main_factory(RecallChatModel()).build(
            model=RecallChatModel()
        )
        assert await first.reply("first", thread_id="restore-thread") == "first"

        state_json = first.checkpoint_state("restore-thread")
        second = _build_main_factory(RecallChatModel()).build(
            model=RecallChatModel()
        )
        second.restore_checkpoint_state(state_json, "restore-thread")
        assert (
            await second.reply("second", thread_id="restore-thread")
            == "recalled:first"
        )

    async def test_stream_reply_emits_token_delta(self) -> None:
        agent = _build_main_factory(ScriptedMigrationModel()).build(
            model=ScriptedMigrationModel()
        )
        events: list[tuple[str, dict]] = []

        async def sink(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        await agent.reply("帮我找降噪耳机", thread_id="s1", event_sink=sink)
        assert any(event_type == "token.delta" for event_type, _ in events)

    async def test_compress_context_trims_old_messages(self) -> None:
        agent = _build_main_factory(RecallChatModel()).build(
            model=RecallChatModel()
        )
        for index in range(6):
            await agent.reply(f"turn-{index}", thread_id="compress-thread")

        result = agent.compress_context("compress-thread", max_messages=4)
        assert result is not None
        assert result["removed"] > 0

        state = await agent._graph.aget_state(
            {"configurable": {"thread_id": "compress-thread"}}
        )
        assert len(state.values["messages"]) <= 4


class TestSessionRegistryPersistence:
    async def test_restart_restores_session_context(self) -> None:
        shared: dict[str, str] = {}
        factory = _build_main_factory(RecallChatModel())
        original_build = factory.build
        factory.build = lambda model=None: original_build(
            model or RecallChatModel()
        )
        first_registry = SessionRegistry(factory, DictSessionStore(shared))
        first_agent = await first_registry.get_or_create("restart-session")
        assert (
            await first_agent.reply("first", thread_id="restart-session")
            == "first"
        )
        await first_registry.persist("restart-session")

        second_registry = SessionRegistry(factory, DictSessionStore(shared))
        restored_agent = await second_registry.get_or_create("restart-session")
        assert (
            await restored_agent.reply("second", thread_id="restart-session")
            == "recalled:first"
        )

    async def test_concurrent_sessions_do_not_cross_talk(self) -> None:
        factory = _build_main_factory(FirstHumanRecallModel())
        original_build = factory.build
        factory.build = lambda model=None: original_build(
            model or FirstHumanRecallModel()
        )
        registry = SessionRegistry(factory, DictSessionStore())
        agent_a = await registry.get_or_create("session-a")
        agent_b = await registry.get_or_create("session-b")
        assert agent_a is not agent_b

        await agent_a.reply("alpha", thread_id="session-a")
        await agent_b.reply("beta", thread_id="session-b")
        results = await asyncio.gather(
            agent_a.reply("again-a", thread_id="session-a"),
            agent_b.reply("again-b", thread_id="session-b"),
        )
        assert results == ["recalled:alpha", "recalled:beta"]
