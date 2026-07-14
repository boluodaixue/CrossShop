"""H5 cross-platform SearchAgent result propagation into Main L4 context."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from agentscope.message import (
    AssistantMsg,
    ToolCallBlock,
    ToolCallState,
    UserMsg,
)
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolResponse

from app.application.context import reduce_l4
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import (
    SearchDispatchContext,
    ShoppingContext,
    ShoppingContextSnapshot,
)
from app.infrastructure.context_middleware import (
    CONTEXT_NAMESPACE,
    ContextLifecycleMiddleware,
)
from app.infrastructure.eventbus import TradeEventBus


class _SearchWorker:
    def __init__(self, bus: TradeEventBus, platform: str, site_locale: str | None):
        self._bus = bus
        self._platform = platform
        self._site_locale = site_locale

    async def reply(self, inputs):
        del inputs
        await asyncio.sleep(
            {"crossshop_reference": 0.003, "reference_seed": 0.002, "amazon": 0.001}[
                self._platform
            ],
        )
        product_id = f"{self._platform}-hit"
        filtered_id = f"{self._platform}-filtered"
        self._bus.publish(
            "cross-session",
            "tool.result",
            {
                "tool": "product_search_tool",
                "platform": self._platform,
                "site_locale": self._site_locale,
                "args": {
                    "normalized_query": "咖啡杯",
                    "platform": self._platform,
                    "site_locale": self._site_locale,
                    "ship_to": "CN",
                    "top_k": 5,
                },
                "dispatch_correlation_id": SearchDispatchContext.current(),
                "hits": [{"product_id": product_id, "title": product_id}],
                "filtered_out": [
                    {
                        "product_id": filtered_id,
                        "reason": "over_price_cap",
                    },
                ],
                "recall_strategy": "embedding_rerank",
            },
        )
        return AssistantMsg(
            "catalog_search_agent",
            f"{self._platform} natural prose; deliberately not JSON",
        )


class _SearchFactory:
    def __init__(self, bus: TradeEventBus) -> None:
        self._bus = bus

    def build(self, *, platform=None, site_locale=None):
        return _SearchWorker(self._bus, platform, site_locale)


class _NoSearchWorker:
    async def reply(self, inputs):
        del inputs
        return AssistantMsg("catalog_search_agent", "检索服务不可用")


class _NoSearchFactory:
    def build(self, **_kwargs):
        return _NoSearchWorker()


class _RetrySearchWorker:
    def __init__(self, bus: TradeEventBus) -> None:
        self._bus = bus

    async def reply(self, inputs):
        del inputs
        for query, product_id in (
            ("咖啡器具", "first-hit"),
            ("咖啡杯", "second-hit"),
        ):
            self._bus.publish(
                "cross-session",
                "tool.result",
                {
                    "tool": "product_search_tool",
                    "platform": "reference_seed",
                    "site_locale": None,
                    "args": {
                        "normalized_query": query,
                        "platform": "reference_seed",
                        "top_k": 5,
                    },
                    "dispatch_correlation_id": SearchDispatchContext.current(),
                    "hits": [{"product_id": product_id}],
                    "filtered_out": [],
                    "recall_strategy": "embedding_rerank",
                },
            )
        return AssistantMsg("catalog_search_agent", "两次检索已完成")


class _RetrySearchFactory:
    def __init__(self, bus: TradeEventBus) -> None:
        self._bus = bus

    def build(self, **_kwargs):
        return _RetrySearchWorker(self._bus)


class _UnusedTradeFactory:
    def build(self):  # pragma: no cover - search-only acceptance
        raise AssertionError("trade agent must not be built")


class _CatalogUseCase:
    async def execute(self, spec: ProductSearchSpec) -> dict:
        assert spec.normalized_query == "咖啡杯"
        return {
            "hits": [{"product_id": "reference_seed-hit"}],
            "filtered_out": [
                {"product_id": "reference_seed-filtered", "reason": "over_price_cap"},
            ],
            "recall_strategy": "embedding_rerank",
        }


class _InterleavedUseCase:
    async def execute(self, spec: ProductSearchSpec) -> dict:
        await asyncio.sleep(0.02 if spec.normalized_query == "咖啡杯" else 0.005)
        return {
            "hits": [{"product_id": f"{spec.normalized_query}-hit"}],
            "filtered_out": [
                {
                    "product_id": f"{spec.normalized_query}-filtered",
                    "reason": "over_price_cap",
                },
            ],
            "recall_strategy": "embedding_rerank",
        }


class _ActualProductWorker:
    def __init__(self, bus: TradeEventBus) -> None:
        self._tool = build_product_search_tool(
            {"reference_seed": _InterleavedUseCase()},  # type: ignore[arg-type]
            bus,
            fixed_platform="reference_seed",
        )

    async def reply(self, inputs):
        demands = inputs.get_text_content() or ""
        query = "咖啡杯" if "咖啡" in demands else "茶杯"
        await self._tool(
            normalized_query=query,
            ship_to="CN",
            price_max_major=100,
        )
        return AssistantMsg("catalog_search_agent", f"{query} 检索完成")


class _ActualProductFactory:
    def __init__(self, bus: TradeEventBus) -> None:
        self._bus = bus

    def build(self, **_kwargs):
        return _ActualProductWorker(self._bus)


def _snapshot_token():
    return ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="cross-session",
            buyer_id="buyer-1",
            locale="zh-CN",
            currency="CNY",
        ),
    )


async def _dispatch_three_platforms():
    bus = TradeEventBus()
    tool = build_task_dispatch_tool(
        _SearchFactory(bus),  # type: ignore[arg-type]
        _UnusedTradeFactory(),  # type: ignore[arg-type]
        bus,
    )
    token = _snapshot_token()
    try:
        chunks = await asyncio.gather(
            tool(
                subagent_type="search_agent",
                demands="CrossShop 搜咖啡杯",
                platform="crossshop_reference",
            ),
            tool(
                subagent_type="search_agent",
                demands="示例平台搜咖啡杯",
                platform="reference_seed",
            ),
            tool(
                subagent_type="search_agent",
                demands="Amazon 搜咖啡杯",
                platform="amazon",
            ),
        )
    finally:
        ShoppingContext.reset(token)
    return chunks


async def test_concurrent_dispatch_returns_exact_structured_platform_results() -> None:
    chunks = await _dispatch_three_platforms()
    outputs = [json.loads(chunk.content[0].text) for chunk in chunks]

    assert [output["platform"] for output in outputs] == [
        "crossshop_reference",
        "reference_seed",
        "amazon",
    ]
    for output in outputs:
        platform = output["platform"]
        assert output["agent"] == "search_agent"
        assert output["search_result_available"] is True
        assert output["hits"] == [
            {"product_id": f"{platform}-hit", "title": f"{platform}-hit"},
        ]
        assert output["filtered_out"][0]["product_id"] == f"{platform}-filtered"
        assert output["search_args"]["platform"] == platform
        assert output["agent_output"].endswith(
            "natural prose; deliberately not JSON",
        )


async def test_product_tool_event_contains_exact_dispatch_contract() -> None:
    bus = TradeEventBus()
    queue = bus.subscribe("cross-session")
    tool = build_product_search_tool(
        {"reference_seed": _CatalogUseCase()},  # type: ignore[arg-type]
        bus,
    )
    token = _snapshot_token()
    try:
        await tool(
            normalized_query="咖啡杯",
            platform="reference_seed",
            ship_to="CN",
            price_max_major=100,
        )
    finally:
        ShoppingContext.reset(token)

    queue.get_nowait()  # tool.invoke
    event = queue.get_nowait()
    assert event.payload["args"]["normalized_query"] == "咖啡杯"
    assert event.payload["args"]["platform"] == "reference_seed"
    assert event.payload["hits"] == [{"product_id": "reference_seed-hit"}]
    assert event.payload["filtered_out"] == [
        {"product_id": "reference_seed-filtered", "reason": "over_price_cap"},
    ]


async def test_missing_product_result_does_not_fabricate_l4_search() -> None:
    bus = TradeEventBus()
    tool = build_task_dispatch_tool(
        _NoSearchFactory(),  # type: ignore[arg-type]
        _UnusedTradeFactory(),  # type: ignore[arg-type]
        bus,
    )
    token = _snapshot_token()
    try:
        chunk = await tool(
            subagent_type="search_agent",
            demands="示例平台搜咖啡杯",
            platform="reference_seed",
        )
    finally:
        ShoppingContext.reset(token)
    output = json.loads(chunk.content[0].text)
    assert output["search_result_available"] is False

    state = reduce_l4(
        {"raw_query": "跨平台找咖啡杯"},
        [
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": "task_dispatch",
                    "tool_call_id": "dispatch-reference_seed",
                    "args": {"platform": "reference_seed"},
                },
            },
            {
                "type": "tool.result",
                "payload": {
                    "tool": "task_dispatch",
                    "tool_call_id": "dispatch-reference_seed",
                    "raw_output": chunk.content[0].text,
                },
            },
        ],
    )
    assert state.last_search is None


async def test_dispatch_uses_latest_structured_result_after_allowed_rewrite() -> None:
    bus = TradeEventBus()
    tool = build_task_dispatch_tool(
        _RetrySearchFactory(bus),  # type: ignore[arg-type]
        _UnusedTradeFactory(),  # type: ignore[arg-type]
        bus,
    )
    token = _snapshot_token()
    try:
        chunk = await tool(
            subagent_type="search_agent",
            demands="示例平台搜咖啡杯",
            platform="reference_seed",
        )
    finally:
        ShoppingContext.reset(token)
    output = json.loads(chunk.content[0].text)
    assert output["search_args"]["normalized_query"] == "咖啡杯"
    assert output["hits"] == [{"product_id": "second-hit"}]


async def test_same_platform_concurrent_dispatches_are_correlated_exactly() -> None:
    bus = TradeEventBus()
    tool = build_task_dispatch_tool(
        _ActualProductFactory(bus),  # type: ignore[arg-type]
        _UnusedTradeFactory(),  # type: ignore[arg-type]
        bus,
    )
    token = _snapshot_token()
    try:
        coffee_chunk, tea_chunk = await asyncio.gather(
            tool(
                subagent_type="search_agent",
                demands="示例平台搜咖啡杯",
                platform="reference_seed",
            ),
            tool(
                subagent_type="search_agent",
                demands="示例平台搜茶杯",
                platform="reference_seed",
            ),
        )
    finally:
        ShoppingContext.reset(token)

    coffee = json.loads(coffee_chunk.content[0].text)
    tea = json.loads(tea_chunk.content[0].text)
    assert (
        "dispatch_correlation_id" not in FunctionTool(tool).input_schema["properties"]
    )
    assert "dispatch_correlation_id" not in coffee
    assert "dispatch_correlation_id" not in tea
    assert coffee["search_args"]["normalized_query"] == "咖啡杯"
    assert coffee["hits"] == [{"product_id": "咖啡杯-hit"}]
    assert coffee["filtered_out"][0]["product_id"] == "咖啡杯-filtered"
    assert tea["search_args"]["normalized_query"] == "茶杯"
    assert tea["hits"] == [{"product_id": "茶杯-hit"}]
    assert tea["filtered_out"][0]["product_id"] == "茶杯-filtered"


async def test_dispatch_results_update_main_l4_without_parsing_agent_prose(
    tmp_path,
) -> None:
    chunks = await _dispatch_three_platforms()
    middleware = ContextLifecycleMiddleware(
        artifact_root=tmp_path / "artifacts",
        model_context_tokens=8_000,
        tool_result_limit=16_000,
    )
    agent = SimpleNamespace(
        state=AgentState(session_id="cross-session"),
        name="commerce_concierge",
    )
    calls = [
        ToolCallBlock(
            id=f"dispatch-{platform}",
            name="task_dispatch",
            input=json.dumps(
                {
                    "subagent_type": "search_agent",
                    "demands": f"{platform} prose demand",
                    "platform": platform,
                },
            ),
            state=ToolCallState.ALLOWED,
        )
        for platform in ("crossshop_reference", "reference_seed", "amazon")
    ]
    model_outputs: list[dict] = []

    async def reply_handler(**_kwargs):
        for call, chunk in zip(calls, chunks):

            async def acting_handler(response_chunk=chunk, **_acting_kwargs):
                yield ToolResponse(
                    content=response_chunk.content,
                    state=response_chunk.state,
                    metadata=response_chunk.metadata,
                )

            async for formatted in middleware.on_acting(
                agent,
                {"tool_call": call},
                acting_handler,
            ):
                model_outputs.append(json.loads(formatted.content[0].text))
        yield AssistantMsg("commerce_concierge", "最终自然语言不参与 L4 投影")

    token = _snapshot_token()
    try:
        async for _ in middleware.on_reply(
            agent,
            {"inputs": UserMsg("buyer-1", "跨平台找咖啡杯")},
            reply_handler,
        ):
            pass
    finally:
        ShoppingContext.reset(token)

    last_search = agent.state.middle_context[CONTEXT_NAMESPACE]["session_context"][
        "last_search"
    ]
    assert last_search["returned_count"] == 3
    assert last_search["filtered_count"] == 3
    assert last_search["product_refs"] == [
        "crossshop_reference-hit",
        "reference_seed-hit",
        "amazon-hit",
    ]
    assert last_search["filtered_product_refs"] == [
        "crossshop_reference-filtered",
        "reference_seed-filtered",
        "amazon-filtered",
    ]
    assert [item["platform"] for item in last_search["platform_results"]] == [
        "crossshop_reference",
        "reference_seed",
        "amazon",
    ]
    assert all(
        "natural prose" not in json.dumps(item)
        for item in last_search["platform_results"]
    )
    assert [output["platform"] for output in model_outputs] == [
        "crossshop_reference",
        "reference_seed",
        "amazon",
    ]
    assert all(output["hits"] and output["filtered_out"] for output in model_outputs)
