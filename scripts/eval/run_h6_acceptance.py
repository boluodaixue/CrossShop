"""H6 reproducible acceptance for multi-turn routing and failure boundaries.

The contract section injects deterministic dependency outcomes into the existing
CatalogSearchUseCase and resilience middleware.  It proves classification and
degradation behavior, not live retrieval quality.  The optional online section
uses the production composition root, the configured model, BGE-M3 services,
OpenSearch indexes, repositories, prompts, and session lifecycle.

Evidence is deliberately compact and redacted: current test inputs, tool names,
search arguments, product references, filtered reasons, L4 state, and final text
are retained; credentials, correlation IDs, raw vectors, and full documents are
never written.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, patch

import httpx
from agentscope.credential import OpenAICredential
from agentscope.message import AssistantMsg, TextBlock, ToolResultState
from agentscope.tool import FunctionTool, ToolChunk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application.agents.orchestrator import (
    MainAgentOrchestrator,
    SubmitIntentInput,
)
from app.application.context.models import L4Context
from app.application.memory.preference_selector import PreferenceSelector
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import (
    CatalogRetrievalError,
    CatalogSearchUseCase,
)
from app.catalog.opensearch_product_h1 import (
    INDEX_NAMES,
    RRF_PIPELINE_NAME,
    TEXT_FIELDS,
    stock_filter,
)
from app.composition import Container, build_container
from app.domain.buyer.preference import BuyerPreference
from app.domain.catalog.money import Money
from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.sku import Sku
from app.infrastructure.context import (
    SearchDispatchContext,
    ShoppingContext,
    ShoppingContextSnapshot,
)
from app.infrastructure.context_middleware import CONTEXT_NAMESPACE
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.llm import ThrottledChatModel
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryProductRepository,
)
from app.infrastructure.resilience import (
    CircuitBreakerRegistry,
    ToolResilienceMiddleware,
)
from app.infrastructure.throttle import GatewayThrottle
from app.infrastructure.transient import retry_dependency_once
from scripts.index.smoke_h4_agent import (
    _RecordingEmbedder,
    _RecordingReranker,
    _atomic_json,
    _atomic_text,
    _dispatch_overlap,
    _mentioned_recommendations,
    _print_json_terminal_safe,
    _product_search_calls,
    _validated_output_root,
    run_deterministic_protocol,
    run_real_infrastructure,
)


@dataclass(frozen=True)
class LiveTurn:
    name: str
    session: str
    buyer: str
    query: str
    expectation: str


@dataclass(frozen=True)
class FaultObservation:
    name: str
    classification: Literal[
        "true_empty",
        "constraint_filtered",
        "data_consistency_failure",
        "transient_dependency_failure",
        "non_transient_dependency_failure",
        "degraded_success",
        "timeout",
        "circuit_open",
    ]
    passed: bool
    request: dict[str, Any]
    result: dict[str, Any]


class _Embedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return [1.0]


class _Index:
    def __init__(
        self,
        product_ids: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.product_ids = product_ids or []
        self.error = error
        self.calls = 0

    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        del query, embedding, top_n
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [
            VectorHit(product_id=product_id, score=1.0 - position / 10)
            for position, product_id in enumerate(self.product_ids)
        ]


class _Reranker:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        del query
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [float(len(documents) - index) for index in range(len(documents))]


class _InjectedChatModel(ThrottledChatModel):
    def __init__(self, behaviors: list[Any]) -> None:
        super().__init__(
            credential=OpenAICredential(
                api_key="test",
                base_url="http://127.0.0.1:9/v1",
            ),
            model="h6-injected-model",
            throttle=GatewayThrottle(max_concurrency=1, min_interval_seconds=0),
            max_transient_retries=2,
            retry_base_seconds=0.01,
        )
        self.behaviors = list(behaviors)
        self.calls = 0

    async def _invoke_upstream(
        self,
        messages: Any,
        tools: Any,
        tool_choice: Any,
        **kwargs: Any,
    ) -> Any:
        del messages, tools, tool_choice, kwargs
        self.calls += 1
        behavior = self.behaviors.pop(0)
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior


class _InjectedSessions:
    """Small production-orchestrator seam for public error-boundary checks."""

    def __init__(self) -> None:
        self.agent = SimpleNamespace(
            state=SimpleNamespace(summary=None, context=[], middle_context={}),
        )
        self.persisted: list[str] = []

    async def get_or_create(self, session_id: str) -> Any:
        del session_id
        return self.agent

    async def persist(self, session_id: str) -> None:
        self.persisted.append(session_id)


class _InjectedErrorOrchestrator(MainAgentOrchestrator):
    def __init__(self, error: Exception, bus: TradeEventBus) -> None:
        self._injected_error = error
        self.sessions = _InjectedSessions()
        super().__init__(
            self.sessions,  # type: ignore[arg-type]
            bus,
            SimpleNamespace(list_by_buyer=None),  # type: ignore[arg-type]
        )

    async def _build_inputs(self, intent: SubmitIntentInput) -> list[Any]:
        del intent
        return []

    async def _reply_with_retry(
        self,
        session_id: str,
        agent: Any,
        inputs: list[Any],
    ) -> str:
        del session_id, agent, inputs
        raise self._injected_error


class _InjectedDispatchWorker:
    def __init__(self, platform: str, bus: TradeEventBus, *, fail: bool) -> None:
        self._platform = platform
        self._bus = bus
        self._fail = fail

    async def reply(self, _message: Any) -> AssistantMsg:
        session_id = ShoppingContext.current_session_id()
        correlation_id = SearchDispatchContext.current()
        if self._fail:
            self._bus.publish(
                session_id,
                "tool.result",
                {
                    "tool": "product_search_tool",
                    "platform": self._platform,
                    "site_locale": None,
                    "dispatch_correlation_id": correlation_id,
                    "error": "injected platform outage",
                },
            )
            return AssistantMsg("search_agent", "[error] 商品 Hybrid 检索不可用")
        self._bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "product_search_tool",
                "platform": self._platform,
                "site_locale": None,
                "dispatch_correlation_id": correlation_id,
                "args": {
                    "normalized_query": "H6 故障边界商品",
                    "platform": self._platform,
                    "top_k": 5,
                },
                "hits": [
                    {
                        "product_id": f"{self._platform}:h6",
                        "title": f"{self._platform} H6 候选",
                    },
                ],
                "filtered_out": [],
                "recall_strategy": "embedding_rerank",
                "hit_count": 1,
            },
        )
        return AssistantMsg("search_agent", f"{self._platform} 查询成功")


class _InjectedSearchFactory:
    def __init__(self, bus: TradeEventBus, failed_platform: str) -> None:
        self._bus = bus
        self._failed_platform = failed_platform

    def build(self, *, platform: str, site_locale: str | None = None) -> Any:
        del site_locale
        return _InjectedDispatchWorker(
            platform,
            self._bus,
            fail=platform == self._failed_platform,
        )


def _product(product_id: str, *, price: float = 100.0) -> Product:
    return Product(
        product_id=product_id,
        title=f"H6 商品 {product_id}",
        brand="CrossShop",
        category="H6 测试品类",
        origin_country="CN",
        description="H6 deterministic fault acceptance",
        ships_to=["CN"],
        skus=[
            Sku(
                sku_id=f"{product_id}-SKU",
                spec="标准款",
                price=Money.from_major_units(price, "CNY"),
                stock=2,
            ),
        ],
    )


async def _call_function_tool(tool: FunctionTool, **kwargs: Any) -> ToolChunk:
    result = await tool(**kwargs)
    if hasattr(result, "__aiter__"):
        chunks = [chunk async for chunk in result]
        return chunks[-1]
    return result


async def run_fault_contracts() -> dict[str, Any]:
    """Exercise the frozen error taxonomy without external services."""
    spec = ProductSearchSpec(normalized_query="H6 测试商品", price_max_major=90)
    observations: list[FaultObservation] = []

    empty_index = _Index([])
    empty = await CatalogSearchUseCase(
        InMemoryProductRepository([]),
        embedder=_Embedder(),
        vector_index=empty_index,
        allow_keyword_fallback=False,
    ).execute(spec)
    observations.append(
        FaultObservation(
            name="hybrid-empty-is-not-system-failure",
            classification="true_empty",
            passed=empty["hits"] == [] and empty_index.calls == 1,
            request=asdict(spec),
            result={
                "hits": empty["hits"],
                "recall_strategy": empty["recall_strategy"],
                "rerank_applied": empty["rerank_applied"],
                "index_calls": empty_index.calls,
                "keyword_fallback_calls": 0,
            },
        ),
    )

    unchanged_request = {
        "dependency": "opensearch",
        "query": "H6 原参数重试",
        "body_sha256": hashlib.sha256(b"h6-identical-request").hexdigest(),
    }
    dependency_attempts: list[dict[str, Any]] = []

    async def transient_dependency() -> str:
        dependency_attempts.append(dict(unchanged_request))
        if len(dependency_attempts) == 1:
            raise httpx.ConnectError("injected transient connection failure")
        return "ok"

    with patch("app.infrastructure.transient.asyncio.sleep", new=AsyncMock()):
        dependency_result = await retry_dependency_once(
            transient_dependency,
            dependency="opensearch",
        )
    observations.append(
        FaultObservation(
            name="approved-dependency-retry-repeats-identical-request-once",
            classification="transient_dependency_failure",
            passed=(
                dependency_result == "ok"
                and len(dependency_attempts) == 2
                and dependency_attempts[0] == dependency_attempts[1]
            ),
            request=unchanged_request,
            result={
                "dependency_attempts": len(dependency_attempts),
                "requests_identical": dependency_attempts[0] == dependency_attempts[1],
                "h4_query_rewrite_calls": 0,
            },
        ),
    )

    platform_registry = CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60)

    async def product_search_tool(platform: str) -> ToolChunk:
        """Injected single product tool with platform-scoped dependency health."""
        if platform == "reference_seed":
            raise httpx.ConnectError("injected reference_seed connection failure")
        return ToolChunk(
            content=[TextBlock(type="text", text=f"{platform}:ok")],
            state=ToolResultState.SUCCESS,
        )

    platform_tool = FunctionTool(
        product_search_tool,
        middlewares=[ToolResilienceMiddleware(platform_registry)],
    )
    await _call_function_tool(platform_tool, platform="reference_seed")
    await _call_function_tool(platform_tool, platform="reference_seed")
    reference_seed_short_circuit = await _call_function_tool(
        platform_tool,
        platform="reference_seed",
    )
    amazon_after_reference_seed_failure = await _call_function_tool(
        platform_tool,
        platform="amazon",
    )
    observations.append(
        FaultObservation(
            name="product-circuit-is-platform-scoped",
            classification="circuit_open",
            passed=(
                platform_registry.status("product_search_tool:reference_seed") == "open"
                and platform_registry.status("product_search_tool:amazon") == "closed"
                and "已熔断" in reference_seed_short_circuit.content[0].text
                and amazon_after_reference_seed_failure.state == ToolResultState.SUCCESS
            ),
            request={
                "tool": "product_search_tool",
                "failed_platform": "reference_seed",
                "healthy_platform": "amazon",
            },
            result={
                "reference_seed_circuit": platform_registry.status(
                    "product_search_tool:reference_seed"
                ),
                "amazon_circuit": platform_registry.status(
                    "product_search_tool:amazon"
                ),
                "amazon_available": (
                    amazon_after_reference_seed_failure.state == ToolResultState.SUCCESS
                ),
            },
        ),
    )

    over_cap = _product("P-OVER", price=120)
    filtered_reranker = _Reranker()
    filtered_index = _Index([over_cap.product_id])
    filtered = await CatalogSearchUseCase(
        InMemoryProductRepository([over_cap]),
        embedder=_Embedder(),
        vector_index=filtered_index,
        reranker=filtered_reranker,
        allow_keyword_fallback=False,
    ).execute(spec)
    observations.append(
        FaultObservation(
            name="all-filtered-is-not-system-failure",
            classification="constraint_filtered",
            passed=(
                filtered["hits"] == []
                and filtered["filtered_out"][0]["reason"] == "over_price_cap"
                and filtered_reranker.calls == 0
                and filtered_index.calls == 1
            ),
            request=asdict(spec),
            result={
                "hits": filtered["hits"],
                "filtered_out": filtered.get("filtered_out", []),
                "recall_strategy": filtered["recall_strategy"],
                "reranker_calls": filtered_reranker.calls,
                "index_calls": filtered_index.calls,
            },
        ),
    )

    destination_blocked = _product("P-SHIP-BLOCKED", price=80)
    destination_blocked.ships_to = ["US"]
    shipping = await CatalogSearchUseCase(
        InMemoryProductRepository([destination_blocked]),
        embedder=_Embedder(),
        vector_index=_Index([destination_blocked.product_id]),
        reranker=_Reranker(),
        allow_keyword_fallback=False,
    ).execute(
        ProductSearchSpec(
            normalized_query="H6 配送约束商品",
            ship_to="CN",
        ),
    )
    observations.append(
        FaultObservation(
            name="ship-to-unavailable-is-explicit-constraint-filter",
            classification="constraint_filtered",
            passed=(
                shipping["hits"] == []
                and shipping["filtered_out"][0]["reason"] == "ship_to_unavailable"
            ),
            request={"normalized_query": "H6 配送约束商品", "ship_to": "CN"},
            result={
                "hits": shipping["hits"],
                "filtered_out": shipping.get("filtered_out", []),
                "recall_strategy": shipping["recall_strategy"],
            },
        ),
    )

    for name, embed_error, index_error in (
        ("embedding-error", RuntimeError("injected embedding failure"), None),
        ("opensearch-error", None, RuntimeError("injected OpenSearch failure")),
    ):
        embedder = _Embedder(embed_error)
        index = _Index([], index_error)
        caught = None
        try:
            await CatalogSearchUseCase(
                InMemoryProductRepository([]),
                embedder=embedder,
                vector_index=index,
                allow_keyword_fallback=False,
            ).execute(spec)
        except CatalogRetrievalError as error:
            caught = f"{type(error).__name__}: {error}"
        observations.append(
            FaultObservation(
                name=name,
                classification="transient_dependency_failure",
                passed=caught == "CatalogRetrievalError: 商品 Hybrid 检索不可用",
                request=asdict(spec),
                result={
                    "error": caught,
                    "embedding_calls": len(embedder.calls),
                    "index_calls": index.calls,
                    "keyword_fallback_calls": 0,
                },
            ),
        )

    missing_index = _Index(["P-MISSING"])
    caught = None
    try:
        await CatalogSearchUseCase(
            InMemoryProductRepository([]),
            embedder=_Embedder(),
            vector_index=missing_index,
            allow_keyword_fallback=False,
        ).execute(spec)
    except RuntimeError as error:
        caught = f"{type(error).__name__}: {error}"
    observations.append(
        FaultObservation(
            name="repository-cannot-hydrate-hit",
            classification="data_consistency_failure",
            passed=bool(caught and "P-MISSING" in caught and "无法精确还原" in caught),
            request=asdict(spec),
            result={"error": caught, "index_calls": missing_index.calls},
        ),
    )

    first = _product("P-FIRST", price=80)
    second = _product("P-SECOND", price=70)
    reranker = _Reranker(RuntimeError("injected reranker failure"))
    rerank_index = _Index([first.product_id, second.product_id])
    degraded = await CatalogSearchUseCase(
        InMemoryProductRepository([first, second]),
        embedder=_Embedder(),
        vector_index=rerank_index,
        reranker=reranker,
        allow_keyword_fallback=False,
    ).execute(ProductSearchSpec(normalized_query="H6 测试商品"))
    observations.append(
        FaultObservation(
            name="reranker-error-keeps-vector-order",
            classification="degraded_success",
            passed=(
                degraded["recall_strategy"] == "embedding_only"
                and degraded["rerank_applied"] is False
                and [item["product_id"] for item in degraded["hits"]]
                == ["P-FIRST", "P-SECOND"]
                and rerank_index.calls == 1
                and reranker.calls == 1
            ),
            request={"normalized_query": "H6 测试商品"},
            result={
                "product_refs": [item["product_id"] for item in degraded["hits"]],
                "recall_strategy": degraded["recall_strategy"],
                "rerank_applied": degraded["rerank_applied"],
                "index_calls": rerank_index.calls,
                "reranker_calls": reranker.calls,
                "additional_recall_calls": 0,
            },
        ),
    )

    async def slow_tool() -> ToolChunk:
        """H6 injected slow dependency."""
        await asyncio.sleep(0.08)
        return ToolChunk(
            content=[TextBlock(type="text", text="late")],
            state=ToolResultState.SUCCESS,
        )

    timeout_registry = CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60)
    slow = FunctionTool(
        slow_tool,
        middlewares=[
            ToolResilienceMiddleware(
                timeout_registry,
                timeouts={"slow_tool": 0.01},
            ),
        ],
    )
    first_timeout = await _call_function_tool(slow)
    second_timeout = await _call_function_tool(slow)
    short_circuit = await _call_function_tool(slow)
    observations.extend(
        [
            FaultObservation(
                name="tool-timeout-is-explicit",
                classification="timeout",
                passed=(
                    first_timeout.state == ToolResultState.ERROR
                    and "超过" in first_timeout.content[0].text
                ),
                request={"tool": "slow_tool", "timeout_seconds": 0.01},
                result={
                    "state": str(first_timeout.state),
                    "message": first_timeout.content[0].text,
                    "automatic_retry_calls": 0,
                },
            ),
            FaultObservation(
                name="repeated-timeout-opens-existing-circuit",
                classification="circuit_open",
                passed=(
                    second_timeout.state == ToolResultState.ERROR
                    and "已熔断" in short_circuit.content[0].text
                    and timeout_registry.status("slow_tool") == "open"
                ),
                request={"tool": "slow_tool", "failure_threshold": 2},
                result={
                    "second_state": str(second_timeout.state),
                    "circuit_state": timeout_registry.status("slow_tool"),
                    "short_circuit_message": short_circuit.content[0].text,
                },
            ),
        ],
    )

    async def unexpected_tool() -> ToolChunk:
        """H6 injected unexpected exception carrying private-looking details."""
        raise RuntimeError(
            "unexpected http://opensearch:9200/private H6_TOOL_INTERNAL_VALUE",
        )

    unexpected_bus = TradeEventBus()
    unexpected_registry = CircuitBreakerRegistry(
        failure_threshold=3,
        reset_seconds=60,
    )
    unexpected = FunctionTool(
        unexpected_tool,
        middlewares=[
            ToolResilienceMiddleware(unexpected_registry, unexpected_bus),
        ],
    )
    unexpected_queue = unexpected_bus.subscribe("h6-unexpected-tool")
    unexpected_token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="h6-unexpected-tool",
            buyer_id="h6-buyer",
            locale="zh-CN",
            currency="CNY",
        ),
    )
    try:
        unexpected_chunk = await _call_function_tool(unexpected)
    finally:
        ShoppingContext.reset(unexpected_token)
    unexpected_events = []
    while not unexpected_queue.empty():
        unexpected_events.append(unexpected_queue.get_nowait().to_dict())
    unexpected_bus.unsubscribe("h6-unexpected-tool", unexpected_queue)
    unexpected_surface = json.dumps(
        {
            "text": unexpected_chunk.content[0].text,
            "events": unexpected_events,
        },
        ensure_ascii=False,
    )
    observations.append(
        FaultObservation(
            name="unexpected-tool-error-does-not-echo-internals",
            classification="non_transient_dependency_failure",
            passed=(
                unexpected_chunk.state == ToolResultState.ERROR
                and "H6_TOOL_INTERNAL_VALUE" not in unexpected_surface
                and "http://opensearch:9200/private" not in unexpected_surface
                and unexpected_registry.status("unexpected_tool") == "closed"
            ),
            request={"tool": "unexpected_tool", "fault": "unexpected exception"},
            result={
                "state": str(unexpected_chunk.state),
                "public_message": unexpected_chunk.content[0].text,
                "internal_details_exposed": False,
                "circuit_state": unexpected_registry.status("unexpected_tool"),
            },
        ),
    )

    model = _InjectedChatModel([RuntimeError("model_not_found")])
    model_error = None
    try:
        await model([])
    except RuntimeError as error:
        model_error = str(error)
    observations.append(
        FaultObservation(
            name="model-business-error-is-not-retried",
            classification="non_transient_dependency_failure",
            passed=model_error == "model_not_found" and model.calls == 1,
            request={"model": "h6-injected-model", "fault": "model_not_found"},
            result={"error": model_error, "upstream_calls": model.calls},
        ),
    )

    model_timeout = _InjectedChatModel(
        [RuntimeError("request timeout") for _ in range(3)],
    )
    timeout_error = None
    try:
        await model_timeout([])
    except RuntimeError as error:
        timeout_error = str(error)
    observations.append(
        FaultObservation(
            name="model-timeout-uses-existing-bounded-retry",
            classification="timeout",
            passed=timeout_error == "request timeout" and model_timeout.calls == 3,
            request={"model": "h6-injected-model", "fault": "request timeout"},
            result={
                "public_error": "request timeout",
                "upstream_calls": model_timeout.calls,
            },
        ),
    )

    public_bus = TradeEventBus()
    secret_marker = "H6_INTERNAL_SECRET_VALUE"
    internal_url = "http://opensearch:9200/private"
    public_orchestrator = _InjectedErrorOrchestrator(
        RuntimeError(f"model_not_found {internal_url} {secret_marker}"),
        public_bus,
    )
    public_queue = public_bus.subscribe("h6-public-error")
    public_output = await public_orchestrator.handle_intent(
        SubmitIntentInput(
            shopping_session_id="h6-public-error",
            buyer_id="h6-public-error-buyer",
            locale="zh-CN",
            currency="CNY",
            raw_query="H6 模型错误脱敏验收",
        ),
    )
    public_events = []
    while not public_queue.empty():
        public_events.append(public_queue.get_nowait().to_dict())
    public_bus.unsubscribe("h6-public-error", public_queue)
    public_surface = json.dumps(
        {"final_text": public_output.final_text, "events": public_events},
        ensure_ascii=False,
    )
    observations.append(
        FaultObservation(
            name="orchestrator-error-boundary-does-not-echo-internals",
            classification="non_transient_dependency_failure",
            passed=(
                public_output.final_text.startswith("[error]")
                and internal_url not in public_surface
                and secret_marker not in public_surface
                and "model_not_found" not in public_surface
                and public_orchestrator.sessions.persisted == ["h6-public-error"]
            ),
            request={"fault": "non-transient model configuration error"},
            result={
                "final_text": public_output.final_text,
                "event_classifications": [
                    event["payload"].get("classification")
                    for event in public_events
                    if event["type"] == "error"
                ],
                "internal_details_exposed": False,
                "session_persisted": bool(public_orchestrator.sessions.persisted),
            },
        ),
    )

    retry_bus = TradeEventBus()
    retry_sessions = _InjectedSessions()
    retry_orchestrator = MainAgentOrchestrator(
        retry_sessions,  # type: ignore[arg-type]
        retry_bus,
        SimpleNamespace(list_by_buyer=None),  # type: ignore[arg-type]
    )
    retry_calls = 0
    transient_secret = "H6_TRANSIENT_INTERNAL_VALUE"
    transient_url = "http://opensearch:9200/retry"

    async def fail_during_stream(
        _session_id: str,
        _agent: Any,
        _inputs: list[Any],
    ) -> str:
        nonlocal retry_calls
        retry_calls += 1
        raise RuntimeError(
            f"request timeout {transient_url} {transient_secret}",
        )

    retry_orchestrator._consume_reply = fail_during_stream  # type: ignore[method-assign]
    retry_queue = retry_bus.subscribe("h6-transient-public-error")
    retry_error = None
    with patch(
        "app.application.agents.orchestrator.asyncio.sleep",
        new=AsyncMock(),
    ):
        try:
            await retry_orchestrator._reply_with_retry(  # noqa: SLF001
                "h6-transient-public-error",
                retry_sessions.agent,  # type: ignore[arg-type]
                [],
            )
        except RuntimeError as error:
            retry_error = str(error)
    retry_events = []
    while not retry_queue.empty():
        retry_events.append(retry_queue.get_nowait().to_dict())
    retry_bus.unsubscribe("h6-transient-public-error", retry_queue)
    retry_surface = json.dumps(retry_events, ensure_ascii=False)
    observations.append(
        FaultObservation(
            name="transient-retry-events-do-not-echo-internals",
            classification="timeout",
            passed=(
                retry_calls == 3
                and retry_error is not None
                and transient_url not in retry_surface
                and transient_secret not in retry_surface
                and len(retry_events) == 2
                and all(
                    event["payload"].get("classification") == "transient"
                    and event["payload"].get("retrying") is True
                    for event in retry_events
                )
            ),
            request={"fault": "transient stream timeout with internal details"},
            result={
                "attempts": retry_calls,
                "retry_event_count": len(retry_events),
                "internal_details_exposed": False,
                "retry_classifications": [
                    event["payload"].get("classification") for event in retry_events
                ],
            },
        ),
    )

    platform_bus = TradeEventBus()
    injected_search_factory = _InjectedSearchFactory(platform_bus, "reference_seed")
    injected_trade_factory = SimpleNamespace()
    dispatch = build_task_dispatch_tool(
        injected_search_factory,  # type: ignore[arg-type]
        injected_trade_factory,  # type: ignore[arg-type]
        platform_bus,
    )
    platform_token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="h6-single-platform-fault",
            buyer_id="h6-buyer",
            locale="zh-CN",
            currency="CNY",
        ),
    )
    try:
        platform_chunks = await asyncio.gather(
            *(
                dispatch(
                    subagent_type="search_agent",
                    demands=f"只查 {platform} 的 H6 故障边界商品",
                    platform=platform,
                    site_locale=None,
                )
                for platform in ("crossshop_reference", "reference_seed", "amazon")
            ),
        )
    finally:
        ShoppingContext.reset(platform_token)
    platform_states = {
        platform: {
            "state": str(chunk.state),
            "dispatch_result": json.loads(chunk.content[0].text),
        }
        for platform, chunk in zip(
            ("crossshop_reference", "reference_seed", "amazon"),
            platform_chunks,
            strict=True,
        )
    }
    observations.append(
        FaultObservation(
            name="single-platform-failure-does-not-cancel-peer-results",
            classification="degraded_success",
            passed=(
                platform_states["crossshop_reference"]["state"] == "success"
                and platform_states["crossshop_reference"]["dispatch_result"][
                    "search_result_available"
                ]
                is True
                and platform_states["reference_seed"]["state"] == "success"
                and platform_states["reference_seed"]["dispatch_result"][
                    "search_result_available"
                ]
                is False
                and platform_states["amazon"]["state"] == "success"
                and platform_states["amazon"]["dispatch_result"][
                    "search_result_available"
                ]
                is True
            ),
            request={
                "platforms": ["crossshop_reference", "reference_seed", "amazon"],
                "injected_failure": "reference_seed",
            },
            result={
                "platform_results": platform_states,
                "boundary": "production task_dispatch correlation and aggregation",
                "failed_dispatch_outer_state": platform_states["reference_seed"]["state"],
                "automatic_retry_calls": 0,
            },
        ),
    )

    class _UnavailableIndex:
        async def ensure_ready(self, vector_dim: int) -> None:
            del vector_dim
            raise RuntimeError("injected OpenSearch startup outage")

    startup_probe = SimpleNamespace(
        db_engine=None,
        task_queue=None,
        product_indexes={"crossshop_reference": _UnavailableIndex()},
        product_search_startup_check={},
        knowledge_base=None,
    )
    await Container.startup(startup_probe)  # type: ignore[arg-type]
    observations.append(
        FaultObservation(
            name="opensearch-unavailable-does-not-block-chat-trade-startup",
            classification="transient_dependency_failure",
            passed=(
                startup_probe.product_search_startup_check.get("crossshop_reference")
                == "unavailable:RuntimeError"
            ),
            request={"operation": "Container.startup"},
            result={
                "error": "RuntimeError",
                "startup_completed": True,
                "chat_trade_available_in_same_container": True,
                "product_search_startup_check": (
                    startup_probe.product_search_startup_check
                ),
                "behavior_changed_post_h6_with_user_approval": True,
            },
        ),
    )

    return {
        "evidence_type": "deterministic_fault_injection_against_production_usecase",
        "passed": all(observation.passed for observation in observations),
        "observations": [asdict(observation) for observation in observations],
        "approved_retry_statement": (
            "Product embedding and OpenSearch may repeat the exact HTTP request once "
            "for timeout/connect/429/502/503/504. Repository hydration and Reranker "
            "are not retried; no extra recall or query rewrite is introduced."
        ),
        "known_boundaries": [
            "OpenSearch startup readiness is an explicit product-search capability "
            "status and no longer prevents chat or Trade startup.",
            "product_search_tool breaker state is isolated by platform; Amazon locales "
            "share the Amazon platform scope.",
            "The post-H6 retry is a dependency attempt, not an H4 query rewrite.",
        ],
    }


def _compact_card(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in (
            "product_id",
            "title",
            "brand",
            "price_major",
            "currency",
            "reason",
        )
        if key in value
    }


def _safe_event(event: Any) -> dict[str, Any] | None:
    row = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    event_type = row.get("type")
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    if event_type not in {
        "agent.dispatch",
        "tool.invoke",
        "tool.result",
        "error",
        "context.compressed",
        "model.fallback",
        "final.result",
    }:
        return None
    safe: dict[str, Any] = {}
    for key in (
        "tool",
        "agent",
        "platform",
        "site_locale",
        "hit_count",
        "filtered_count",
        "recall_strategy",
        "search_result_available",
        "circuit",
        "retrying",
        "elapsed_ms",
        "started_at",
        "finished_at",
        "message",
        "error",
        "harness",
        "classification",
    ):
        if key in payload:
            safe[key] = payload[key]
    args = payload.get("args")
    if isinstance(args, dict):
        safe["args"] = {
            key: args[key]
            for key in (
                "normalized_query",
                "platform",
                "site_locale",
                "category",
                "ship_to",
                "locale",
                "top_k",
                "price_max_major",
                "target_currency",
                "order_id",
                "reason",
            )
            if key in args
        }
        if isinstance(args.get("items"), list):
            safe["args"]["items"] = [
                {
                    key: item[key]
                    for key in ("product_id", "sku_id", "quantity")
                    if key in item
                }
                for item in args["items"]
                if isinstance(item, dict)
            ]
    if isinstance(payload.get("hits"), list):
        safe["hits"] = [_compact_card(item) for item in payload["hits"][:5]]
    if isinstance(payload.get("filtered_out"), list):
        safe["filtered_out"] = [
            _compact_card(item) for item in payload["filtered_out"][:3]
        ]
    if isinstance(payload.get("displayed_products"), list):
        safe["displayed_products"] = [
            {
                "rank": item.get("rank"),
                "platform": item.get("platform"),
                "site_locale": item.get("site_locale"),
                "card": _compact_card(item.get("card")),
            }
            for item in payload["displayed_products"][:5]
            if isinstance(item, dict)
        ]
    if isinstance(payload.get("order"), dict):
        safe["order"] = {
            key: payload["order"][key]
            for key in (
                "order_id",
                "status",
                "total_amount_major",
                "currency",
                "cancel_reason",
            )
            if key in payload["order"]
        }
    return {
        "type": event_type,
        "occurred_at": row.get("occurred_at", ""),
        "payload": safe,
    }


def _product_invokes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["type"] == "tool.invoke"
        and event["payload"].get("tool") == "product_search_tool"
    ]


def _search_dispatches(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event["type"] == "agent.dispatch"
        and event["payload"].get("agent") == "search_agent"
    ]


def _tool_names(events: list[dict[str, Any]]) -> list[str]:
    return [
        str(event["payload"].get("tool"))
        for event in events
        if event["type"] == "tool.invoke"
    ]


def _recommendation_display_mentions(
    final_text: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Count identifiable displayed products without mapping generic type words.

    The shared H4 matcher deliberately accepts four-character unique title
    fragments so it can recover loose ordinal references.  That is too broad
    for H6's display-cap assertion: phrases such as ``蓝牙音箱`` or ``挂颈风扇``
    can describe excluded noise without naming a specific card.  For the cap,
    keep exact identities and brands, but require a longer shortened-title
    anchor before treating prose as an additional displayed product.
    """

    def strict_mentions(text: str) -> list[dict[str, Any]]:
        mentions = [
            item
            for item in _mentioned_recommendations(text, events)
            if item.get("evidence", {}).get("kind") != "unique_event_title_fragment"
            or len(
                _normalized_text(item.get("evidence", {}).get("value", "")),
            )
            >= 7
        ]
        strong = [
            item
            for item in mentions
            if item.get("evidence", {}).get("kind") != "event_brand"
        ]
        # A row containing an exact title/id must not also map its shared brand
        # to a second returned SKU (for example AeroHush Pro vs AeroHush Lite).
        return strong or mentions

    list_rows: list[dict[str, Any]] = []
    for line in final_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or re.fullmatch(r"[|:\-\s]+", stripped):
            continue
        list_rows.extend(strict_mentions(stripped))
    if list_rows:
        deduplicated = {item["product_id"]: item for item in list_rows}
        return list(deduplicated.values())
    return strict_mentions(final_text)


async def _l4_snapshot(container: Container, session_id: str) -> dict[str, Any]:
    registry = container.orchestrator._sessions  # noqa: SLF001 - acceptance probe
    agent = await registry.get_or_create(session_id)
    namespace = agent.state.middle_context.get(CONTEXT_NAMESPACE, {})
    raw = namespace.get("session_context") if isinstance(namespace, dict) else None
    return L4Context.from_dict(raw).to_dict()


async def _run_turn(
    container: Container,
    turn: LiveTurn,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    queue = container.bus.subscribe(turn.session)
    prior_state = await _l4_snapshot(container, turn.session)
    started = time.perf_counter()
    try:
        output = await asyncio.wait_for(
            container.orchestrator.handle_intent(
                SubmitIntentInput(
                    shopping_session_id=turn.session,
                    buyer_id=turn.buyer,
                    locale="zh-CN",
                    currency="CNY",
                    raw_query=turn.query,
                ),
            ),
            timeout=timeout_seconds,
        )
        final_text = output.final_text
        displayed_products = list(output.displayed_products)
        exception = None
    except Exception as error:  # noqa: BLE001 - record real online outcome
        final_text = ""
        displayed_products = []
        exception = f"{type(error).__name__}: {error}"
    raw_events = []
    while not queue.empty():
        raw_events.append(queue.get_nowait())
    container.bus.unsubscribe(turn.session, queue)
    events = [safe for event in raw_events if (safe := _safe_event(event))]
    state = await _l4_snapshot(container, turn.session)
    return {
        "name": turn.name,
        "session_label": turn.session.rsplit("-", 1)[0],
        "buyer_label": turn.buyer.rsplit("-", 1)[0],
        "input": turn.query,
        "expectation": turn.expectation,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "exception": exception,
        "events": events,
        "prior_structured_state": prior_state,
        "structured_state": state,
        "final_output": final_text,
        "displayed_products": displayed_products,
    }


def _strict_displayed_identity_failures(turn: dict[str, Any]) -> list[str]:
    """Validate the authoritative selection against tool facts and exact prose."""

    displayed = turn.get("displayed_products") or []
    if len(displayed) > 5:
        return ["structured final display exceeded five products"]
    candidates: dict[tuple[str, str | None, str], dict[str, Any]] = {}
    prior_recommendation = (
        turn.get("prior_structured_state", {}).get("last_recommendation") or {}
    )
    for prior in prior_recommendation.get("displayed_products", []):
        if not isinstance(prior, dict) or not isinstance(prior.get("card"), dict):
            continue
        prior_card = prior["card"]
        prior_id = prior_card.get("product_id")
        if isinstance(prior_id, str):
            candidates[(prior.get("platform"), prior.get("site_locale"), prior_id)] = (
                prior_card
            )
    for event in turn["events"]:
        payload = event["payload"]
        if (
            event["type"] != "tool.result"
            or payload.get("tool") != "product_search_tool"
        ):
            continue
        if payload.get("error"):
            continue
        for hit in payload.get("hits", []):
            product_id = hit.get("product_id")
            if isinstance(product_id, str):
                candidates[
                    (payload.get("platform"), payload.get("site_locale"), product_id)
                ] = hit
    failures: list[str] = []
    displayed_product_ids = [
        item.get("card", {}).get("product_id")
        for item in displayed
        if isinstance(item, dict) and isinstance(item.get("card"), dict)
    ]
    identity_segments: list[str] = []
    for line in turn["final_output"].splitlines():
        if not line.strip():
            continue
        id_positions = sorted(
            (
                position,
                product_id,
            )
            for product_id in displayed_product_ids
            if isinstance(product_id, str)
            for position in [line.find(product_id)]
            if position >= 0
        )
        if len(id_positions) <= 1:
            identity_segments.append(line)
            continue
        for index, (position, _) in enumerate(id_positions):
            end = (
                id_positions[index + 1][0]
                if index + 1 < len(id_positions)
                else len(line)
            )
            identity_segments.append(line[position:end])
    seen: set[tuple[str, str | None, str]] = set()
    for expected_rank, item in enumerate(displayed, start=1):
        card = item.get("card") if isinstance(item, dict) else None
        if not isinstance(card, dict):
            failures.append(f"displayed product {expected_rank} has no card")
            continue
        product_id = card.get("product_id")
        title = card.get("title")
        key = (item.get("platform"), item.get("site_locale"), product_id)
        if item.get("rank") != expected_rank:
            failures.append(f"displayed product rank mismatch at {expected_rank}")
        if key in seen:
            failures.append(f"duplicate displayed product: {product_id}")
        seen.add(key)
        if key not in candidates:
            # A prior verified card may be intentionally referenced without a
            # new search. L4 keeps that evidence; current-turn tool facts are
            # mandatory only when this turn actually searched.
            current_searched = bool(_product_invokes(turn["events"]))
            if current_searched:
                failures.append(f"displayed product not in successful hits: {key}")
        elif candidates[key].get("title") != title:
            failures.append(
                f"displayed product title differs from tool hit: {product_id}"
            )
        if not isinstance(product_id, str) or product_id not in turn["final_output"]:
            failures.append(f"final text omitted exact product_id: {product_id}")
        if not isinstance(title, str) or title not in turn["final_output"]:
            failures.append(f"final text omitted exact title: {product_id}")
        if (
            isinstance(product_id, str)
            and isinstance(title, str)
            and product_id in turn["final_output"]
            and title in turn["final_output"]
        ):
            if not any(
                product_id in segment and title in segment
                for segment in identity_segments
            ):
                failures.append(
                    f"final text did not pair exact id and title: {product_id}"
                )
    return failures


def _evaluate_turn(turn: dict[str, Any]) -> list[str]:
    expectation = turn["expectation"]
    events = turn["events"]
    invokes = _product_invokes(events)
    dispatches = _search_dispatches(events)
    tools = _tool_names(events)
    failures: list[str] = []
    if turn["exception"] or turn["final_output"].startswith("[error]"):
        failures.append(turn["exception"] or turn["final_output"])
    if any(event["payload"].get("harness") == "loop_detected" for event in events):
        failures.append("legal H6 route triggered loop_detected")
    if expectation == "ordinary_chat":
        forbidden = {
            "product_search_tool",
            "category_insight_tool",
            "create_order_tool",
            "query_order_tool",
            "cancel_order_tool",
            "task_dispatch",
        }
        used = sorted(forbidden & set(tools))
        if used or dispatches:
            failures.append(f"ordinary chat used shopping route: {used}")
    elif expectation == "category_only":
        if "category_insight_tool" not in tools:
            failures.append("pure category question did not query category knowledge")
        if invokes or dispatches:
            failures.append(
                "pure category question incorrectly triggered product search"
            )
    elif expectation == "broad_product":
        if not invokes:
            failures.append("broad product request did not search")
        if dispatches:
            failures.append(
                "broad request without multi-platform intent dispatched agents"
            )
    elif expectation == "cross_platform":
        platforms = {event["payload"].get("platform") for event in dispatches}
        if platforms != {"crossshop_reference", "reference_seed", "amazon"}:
            failures.append(f"cross-platform dispatch mismatch: {sorted(platforms)}")
        overlap = _dispatch_overlap(events)
        if set(overlap["complete_platforms"]) != platforms:
            failures.append("cross-platform dispatch timing evidence incomplete")
        if overlap["overlapped"] is not True:
            failures.append(f"cross-platform dispatch did not overlap: {overlap}")
        protocol_failures, _ = _protocol(events, platforms)
        failures.extend(protocol_failures)
    elif expectation.startswith("direct:"):
        expected = expectation.split(":", 1)[1]
        observed = [
            event["payload"].get("args", {}).get("platform") for event in invokes
        ]
        if not 1 <= len(invokes) <= 2 or set(observed) != {expected}:
            failures.append(f"direct route mismatch: {observed}")
        if dispatches:
            failures.append("single-platform request dispatched SearchAgent")
        protocol_failures, _ = _protocol(events, {expected})
        failures.extend(protocol_failures)
    elif expectation == "amazon_jp":
        sites = {
            event["payload"].get("args", {}).get("site_locale") for event in invokes
        }
        if sites != {"jp"}:
            failures.append(f"Amazon JP site_locale mismatch: {sorted(sites, key=str)}")
        if dispatches:
            failures.append("single Amazon JP request dispatched SearchAgent")
    elif expectation == "trade_query":
        if "query_order_tool" not in tools:
            failures.append("order query did not use query_order_tool")
        if invokes or dispatches:
            failures.append("order query incorrectly entered product SearchAgent route")
    elif expectation == "trade_create":
        if "create_order_tool" not in tools:
            failures.append("confirmed order did not use create_order_tool")
        if invokes or dispatches:
            failures.append("confirmed order incorrectly entered product search route")
    elif expectation == "trade_confirm":
        if "create_order_tool" in tools or "cancel_order_tool" in tools:
            failures.append("order confirmation card mutated order before confirmation")
        if dispatches:
            failures.append("order confirmation card dispatched product SearchAgent")
        if "确认" not in turn["final_output"]:
            failures.append("order preparation did not present a confirmation step")
    elif expectation == "trade_cancel_confirm":
        if "query_order_tool" not in tools:
            failures.append("cancellation confirmation did not verify order status")
        if "cancel_order_tool" in tools:
            failures.append(
                "cancellation executed before explicit follow-up confirmation"
            )
        if invokes or dispatches:
            failures.append("cancellation confirmation entered product search route")
    elif expectation == "trade_cancel":
        if "cancel_order_tool" not in tools:
            failures.append("confirmed cancellation did not cancel order")
        if invokes or dispatches:
            failures.append(
                "order cancellation incorrectly entered product search route"
            )
    elif expectation == "ordinal_reference":
        if invokes or dispatches:
            failures.append("ordinal reference unnecessarily re-ran product search")
    elif expectation == "constraint_followup":
        if not invokes:
            failures.append("constraint follow-up did not search")
        for event in invokes:
            args = event["payload"].get("args", {})
            if args.get("ship_to") != "CN" or args.get("price_max_major") != 100:
                failures.append(f"constraint follow-up lost hard constraints: {args}")
    elif expectation == "category_switch":
        if not invokes:
            failures.append("category switch did not search new category")
        for event in invokes:
            args = event["payload"].get("args", {})
            if (
                args.get("ship_to") is not None
                or args.get("price_max_major") is not None
            ):
                failures.append(f"category switch inherited old constraints: {args}")
            normalized_query = re.sub(
                r"\s+", "", str(args.get("normalized_query") or "")
            )
            if normalized_query != "降噪耳机":
                failures.append(f"category switch did not form new request: {args}")
    elif expectation == "conditional_rewrite_filtered":
        protocol_failures, evidence = _protocol(events, {"reference_seed"})
        failures.extend(protocol_failures)
        calls = evidence["calls_by_platform"]["reference_seed"]
        if len(calls) != 2:
            failures.append(f"expected one legal rewrite, observed {len(calls)} calls")
        elif not all(
            (call.get("result") or {}).get("hit_count") == 0
            and (call.get("result") or {}).get("filtered_out")
            for call in calls
        ):
            failures.append("rewrite scenario was not a filtered-only result")
        reasons = {
            item.get("reason")
            for call in calls
            for item in (call.get("result") or {}).get("filtered_out", [])
        }
        if "ship_to_unavailable" not in reasons:
            failures.append(f"shipping filter reason missing: {sorted(reasons)}")
    elif expectation == "more_recommendations":
        # It is valid to reuse still-unshown candidates from the immediately
        # preceding tool result. Detailed provenance/no-repeat checks run once
        # both turns are available below.
        pass
    else:
        failures.append(f"unknown expectation: {expectation}")
    if any(event["payload"].get("args", {}).get("top_k", 5) > 5 for event in invokes):
        failures.append("product search top_k exceeded 5")
    failures.extend(_strict_displayed_identity_failures(turn))
    final_mentions = _recommendation_display_mentions(turn["final_output"], events)
    turn["final_product_mention_evidence"] = final_mentions
    if len(final_mentions) > 5:
        failures.append(
            f"final output named {len(final_mentions)} products; maximum is 5"
        )
    return failures


def _protocol(
    events: list[dict[str, Any]],
    platforms: set[str],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    calls = _product_search_calls(events)
    by_platform: dict[str, list[dict[str, Any]]] = {
        platform: [] for platform in platforms
    }
    for call in calls:
        platform = call["args"].get("platform")
        if platform in by_platform:
            by_platform[platform].append(call)
    for platform, platform_calls in by_platform.items():
        if not platform_calls:
            failures.append(f"{platform} made no product search")
            continue
        if len(platform_calls) > 2:
            failures.append(f"{platform} made more than two product searches")
        for position, call in enumerate(platform_calls, start=1):
            result = call.get("result")
            if not isinstance(result, dict):
                failures.append(f"{platform} call {position} has no paired result")
            elif result.get("error"):
                failures.append(
                    f"{platform} call {position} returned error: {result['error']}"
                )
        if len(platform_calls) == 2:
            first = platform_calls[0]
            first_count = (first.get("result") or {}).get("hit_count")
            if not isinstance(first_count, int) or first_count >= 3:
                failures.append(
                    f"{platform} rewrote after first hit_count={first_count!r}"
                )
            changed = sorted(
                key
                for key in set(first["args"]) | set(platform_calls[1]["args"])
                if first["args"].get(key) != platform_calls[1]["args"].get(key)
            )
            if changed != ["normalized_query"]:
                failures.append(f"{platform} rewrite changed {changed}")
    return failures, {"calls_by_platform": by_platform}


def _previous_product_refs(turn: dict[str, Any]) -> set[str]:
    return {
        str(hit["product_id"])
        for event in turn["events"]
        if event["type"] == "tool.result"
        and event["payload"].get("tool") == "product_search_tool"
        for hit in event["payload"].get("hits", [])
        if hit.get("product_id")
    }


def _normalized_text(value: Any) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _displayed_product_refs(turn: dict[str, Any]) -> list[str]:
    """Use authoritative structured order; retain prose matching for old artifacts."""
    structured = [
        str(item.get("card", {}).get("product_id"))
        for item in turn.get("displayed_products", [])
        if isinstance(item, dict) and item.get("card", {}).get("product_id")
    ]
    if structured:
        return structured
    mentions = _mentioned_recommendations(turn["final_output"], turn["events"])
    normalized_output = _normalized_text(turn["final_output"])
    normalized_titles = {
        item["product_id"]: _normalized_text(item["title"]) for item in mentions
    }
    positions: list[tuple[int, str]] = []
    for item in mentions:
        product_id = item["product_id"]
        title = normalized_titles[product_id]
        position = len(normalized_output) + 1
        for width in range(min(12, len(title)), 1, -1):
            candidates = {
                title[start : start + width]
                for start in range(len(title) - width + 1)
                if all(
                    title[start : start + width] not in other_title
                    for other_id, other_title in normalized_titles.items()
                    if other_id != product_id
                )
            }
            found = [
                normalized_output.find(candidate)
                for candidate in candidates
                if candidate in normalized_output
            ]
            if found:
                position = min(found)
                break
        positions.append((position, product_id))
    return [product_id for _, product_id in sorted(positions)]


def _numbered_product_refs(turn: dict[str, Any]) -> dict[int, str]:
    """Map display ranks to cards, using verified structured output first."""
    structured = {
        int(item["rank"]): str(item.get("card", {}).get("product_id"))
        for item in turn.get("displayed_products", [])
        if isinstance(item, dict)
        and isinstance(item.get("rank"), int)
        and item.get("card", {}).get("product_id")
    }
    if structured:
        return structured
    numbered: dict[int, str] = {}
    for match in re.finditer(r"\*\*(\d+)\.\s*(.*?)\*\*", turn["final_output"]):
        mentions = _mentioned_recommendations(match.group(2), turn["events"])
        if len(mentions) == 1:
            numbered[int(match.group(1))] = mentions[0]["product_id"]
    return numbered


async def _run_recorded_retrieval_turn(
    container: Container,
    turn: LiveTurn,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record the exact query crossing both real retrieval model boundaries."""
    usecase = container.catalog_searches["reference_seed"]
    original_embedder = usecase._embedder
    original_reranker = usecase._reranker
    embedder = _RecordingEmbedder(original_embedder)
    reranker = _RecordingReranker(original_reranker)
    usecase._embedder = embedder
    usecase._reranker = reranker
    try:
        result = await _run_turn(
            container,
            turn,
            timeout_seconds=timeout_seconds,
        )
    finally:
        usecase._embedder = original_embedder
        usecase._reranker = original_reranker
    evidence = {
        "embedding_queries": [call["text"] for call in embedder.calls],
        "embedding_dimensions": [call["dimension"] for call in embedder.calls],
        "reranker_queries": [call["query"] for call in reranker.calls],
        "reranker_document_counts": [call["document_count"] for call in reranker.calls],
    }
    result["retrieval_query_evidence"] = evidence
    return result, evidence


def _created_order_id(turn: dict[str, Any]) -> str | None:
    for event in turn["events"]:
        payload = event["payload"]
        if (
            event["type"] == "tool.result"
            and payload.get("tool") == "create_order_tool"
            and isinstance(payload.get("order"), dict)
        ):
            order_id = payload["order"].get("order_id")
            if isinstance(order_id, str):
                return order_id
    return None


async def run_online_acceptance(
    container: Container,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    stamp = str(int(time.time()))
    preference_embedding_probe = await _run_real_preference_embedding_probe(container)
    independent = [
        LiveTurn(
            "ordinary-chat",
            f"h6-chat-{stamp}",
            f"h6-chat-buyer-{stamp}",
            "你好，今天过得怎么样？请只和我普通聊天。",
            "ordinary_chat",
        ),
        LiveTurn(
            "category-only",
            f"h6-category-{stamp}",
            f"h6-category-buyer-{stamp}",
            "请查品类知识库告诉我选降噪耳机要看哪些维度；先不要推荐具体商品。",
            "category_only",
        ),
        LiveTurn(
            "broad-product",
            f"h6-broad-{stamp}",
            f"h6-broad-buyer-{stamp}",
            "帮我找降噪耳机，直接搜索并给商品，不用先追问。",
            "broad_product",
        ),
        LiveTurn(
            "explicit-cross-platform",
            f"h6-cross-{stamp}",
            f"h6-cross-buyer-{stamp}",
            "同时比较 CrossShop、示例平台和 Amazon 的降噪耳机，三个平台都要查。",
            "cross_platform",
        ),
        LiveTurn(
            "single-crossshop",
            f"h6-crossshop-{stamp}",
            f"h6-crossshop-buyer-{stamp}",
            "只看 CrossShop，帮我找轻便旅行背包，最多 5 件。",
            "direct:crossshop_reference",
        ),
        LiveTurn(
            "single-reference_seed",
            f"h6-reference_seed-{stamp}",
            f"h6-reference_seed-buyer-{stamp}",
            "只看示例平台，帮我找轻便旅行背包，最多 5 件。",
            "direct:reference_seed",
        ),
        LiveTurn(
            "single-amazon",
            f"h6-amazon-{stamp}",
            f"h6-amazon-buyer-{stamp}",
            "只看 Amazon，帮我找 noise cancelling headphones，未指定站点。",
            "direct:amazon",
        ),
        LiveTurn(
            "single-amazon-jp",
            f"h6-amazon-jp-{stamp}",
            f"h6-amazon-jp-buyer-{stamp}",
            "只看 Amazon 日本站，帮我找ノイズキャンセリング ヘッドホン。",
            "amazon_jp",
        ),
        LiveTurn(
            "trade-query",
            f"h6-trade-{stamp}",
            f"h6-trade-buyer-{stamp}",
            "查询订单 GBX-H6-NOT-FOUND 的状态。",
            "trade_query",
        ),
        LiveTurn(
            "shipping-filtered-rewrite",
            f"h6-shipping-filter-{stamp}",
            f"h6-shipping-filter-buyer-{stamp}",
            "只看示例平台，找旅行背包，必须配送到南极洲 AQ。如果首次顶层原始 hits 少于 3，按既有规则只改 normalized_query 再查一次；之后停止并如实说明 filtered_out，不要报系统故障。",
            "conditional_rewrite_filtered",
        ),
    ]
    turns = [
        await _run_turn(container, turn, timeout_seconds=timeout_seconds)
        for turn in independent
    ]

    constraint_session = f"h6-constraint-{stamp}"
    constraint_buyer = f"h6-constraint-buyer-{stamp}"
    for turn in (
        LiveTurn(
            "constraint-base",
            constraint_session,
            constraint_buyer,
            "只看示例平台，帮我找轻便旅行背包。",
            "direct:reference_seed",
        ),
        LiveTurn(
            "constraint-followup",
            constraint_session,
            constraint_buyer,
            "还是刚才的旅行背包，这轮加上预算 100 元并配送到中国。",
            "constraint_followup",
        ),
        LiveTurn(
            "category-switch",
            constraint_session,
            constraint_buyer,
            "改看降噪耳机，不要沿用刚才的预算和配送条件。请把本轮工具返回的 5 件候选全部按 1–5 顺序展示；不完全匹配的也要保留并如实说明差异。",
            "category_switch",
        ),
        LiveTurn(
            "ordinal-reference",
            constraint_session,
            constraint_buyer,
            "刚才第三个和第五个分别是什么？不要重新搜索。",
            "ordinal_reference",
        ),
    ):
        turns.append(await _run_turn(container, turn, timeout_seconds=timeout_seconds))

    more_session = f"h6-more-{stamp}"
    more_buyer = f"h6-more-buyer-{stamp}"
    first_more = await _run_turn(
        container,
        LiveTurn(
            "quantity-three",
            more_session,
            more_buyer,
            "只看示例平台找旅行背包，明确推荐 3 件。",
            "direct:reference_seed",
        ),
        timeout_seconds=timeout_seconds,
    )
    second_more = await _run_turn(
        container,
        LiveTurn(
            "more-recommendations",
            more_session,
            more_buyer,
            "再给我更多推荐，避免无故重复；如果当前候选没有新的就如实说。",
            "more_recommendations",
        ),
        timeout_seconds=timeout_seconds,
    )
    turns.extend([first_more, second_more])

    trade_session = f"h6-trade-lifecycle-{stamp}"
    trade_buyer = f"h6-trade-lifecycle-buyer-{stamp}"
    trade_search = await _run_turn(
        container,
        LiveTurn(
            "trade-product-search",
            trade_session,
            trade_buyer,
            "只看 CrossShop，搜索 QuietEar 硅胶隔音耳塞，我后面要买。",
            "direct:crossshop_reference",
        ),
        timeout_seconds=timeout_seconds,
    )
    trade_confirm = await _run_turn(
        container,
        LiveTurn(
            "trade-create-confirm-card",
            trade_session,
            trade_buyer,
            "我要买刚才的 QuietEar 耳塞 1 件（product_id=P1053，sku_id=P1053-S1）。收件人 H6验收，中国上海市浦东新区 H6 测试地址，邮编 200120，电话 00000000000。请先给我确认卡，暂不创建订单，等我下一轮确认。",
            "trade_confirm",
        ),
        timeout_seconds=timeout_seconds,
    )
    trade_create = await _run_turn(
        container,
        LiveTurn(
            "trade-create-confirmed",
            trade_session,
            trade_buyer,
            "我确认按上一条确认卡现在创建订单。",
            "trade_create",
        ),
        timeout_seconds=timeout_seconds,
    )
    order_id = _created_order_id(trade_create)
    trade_query_created = await _run_turn(
        container,
        LiveTurn(
            "trade-query-created",
            trade_session,
            trade_buyer,
            f"查询刚创建的订单 {order_id or 'GBX-H6-MISSING'} 状态。",
            "trade_query",
        ),
        timeout_seconds=timeout_seconds,
    )
    trade_cancel_confirm = await _run_turn(
        container,
        LiveTurn(
            "trade-cancel-confirm-card",
            trade_session,
            trade_buyer,
            f"我想取消订单 {order_id or 'GBX-H6-MISSING'}，原因是 H6 验收。请先查询状态并给取消确认卡，这一轮暂不取消。",
            "trade_cancel_confirm",
        ),
        timeout_seconds=timeout_seconds,
    )
    trade_cancel = await _run_turn(
        container,
        LiveTurn(
            "trade-cancel-confirmed",
            trade_session,
            trade_buyer,
            f"我确认取消订单 {order_id or 'GBX-H6-MISSING'}，原因仍是 H6 验收，请现在执行。",
            "trade_cancel",
        ),
        timeout_seconds=timeout_seconds,
    )
    turns.extend(
        [
            trade_search,
            trade_confirm,
            trade_create,
            trade_query_created,
            trade_cancel_confirm,
            trade_cancel,
        ],
    )

    preference_store = container.orchestrator._preference_store  # noqa: SLF001
    preferred_buyer = f"h6-pref-buyer-{stamp}"
    plain_buyer = f"h6-plain-buyer-{stamp}"
    statement = "不喜欢塑料材质"
    await preference_store.append(
        BuyerPreference(
            buyer_id=preferred_buyer,
            kind="dislike",
            statement=statement,
        ),
    )
    try:
        preference_query = (
            "只看示例平台，帮我找轻便旅行背包；本轮 normalized_query "
            "严格使用“轻便旅行背包”，不添加同义词。"
        )
        preferred, preferred_retrieval = await _run_recorded_retrieval_turn(
            container,
            LiveTurn(
                "with-history-preference",
                f"h6-pref-{stamp}",
                preferred_buyer,
                preference_query,
                "direct:reference_seed",
            ),
            timeout_seconds=timeout_seconds,
        )
        plain, plain_retrieval = await _run_recorded_retrieval_turn(
            container,
            LiveTurn(
                "without-history-preference",
                f"h6-plain-{stamp}",
                plain_buyer,
                preference_query,
                "direct:reference_seed",
            ),
            timeout_seconds=timeout_seconds,
        )
    finally:
        await preference_store.delete(preferred_buyer, statement)
    turns.extend([preferred, plain])

    for turn in turns:
        failures = _evaluate_turn(turn)
        turn["passed"] = not failures
        turn["failures"] = failures

    cross_turn = next(
        turn for turn in turns if turn["name"] == "explicit-cross-platform"
    )
    cross_turn["dispatch_overlap"] = _dispatch_overlap(cross_turn["events"])

    prior_refs = _previous_product_refs(
        next(turn for turn in turns if turn["name"] == "category-switch"),
    )
    ordinal = next(turn for turn in turns if turn["name"] == "ordinal-reference")
    ordinal_mentions = _mentioned_recommendations(
        ordinal["final_output"],
        next(turn for turn in turns if turn["name"] == "category-switch")["events"],
    )
    ordinal["reference_evidence"] = {
        "previous_product_refs": sorted(prior_refs),
        "mentioned_prior_products": ordinal_mentions,
        "mentioned_prior_count": len(ordinal_mentions),
    }
    category_switch_turn = next(
        turn for turn in turns if turn["name"] == "category-switch"
    )
    numbered_refs = _numbered_product_refs(category_switch_turn)
    displayed_refs = [numbered_refs[key] for key in sorted(numbered_refs)]
    if len(displayed_refs) < 5:
        displayed_refs = _displayed_product_refs(category_switch_turn)
    expected_ordinal_refs = [
        product_ref
        for position in (3, 5)
        if (product_ref := numbered_refs.get(position)) is not None
    ]
    ordinal["reference_evidence"]["expected_positions"] = [3, 5]
    ordinal["reference_evidence"]["prior_display_order"] = displayed_refs
    ordinal["reference_evidence"]["prior_numbered_refs"] = numbered_refs
    ordinal["reference_evidence"]["expected_product_refs"] = expected_ordinal_refs
    mentioned_ordinal_refs = {item["product_id"] for item in ordinal_mentions}
    ordinal["reference_evidence"]["expected_refs_mentioned"] = set(
        expected_ordinal_refs
    ).issubset(mentioned_ordinal_refs)
    if (
        not expected_ordinal_refs
        or not ordinal["reference_evidence"]["expected_refs_mentioned"]
    ):
        ordinal["passed"] = False
        ordinal["failures"].append(
            "ordinal answer did not recover exact positions three and five"
        )

    first_mentions = _mentioned_recommendations(
        first_more["final_output"], first_more["events"]
    )
    second_mentions = _mentioned_recommendations(
        second_more["final_output"], first_more["events"]
    )
    first_ids = {item["product_id"] for item in first_mentions}
    second_ids = {item["product_id"] for item in second_mentions}
    first_candidate_refs = _previous_product_refs(first_more)
    previously_unshown_refs = first_candidate_refs - first_ids
    duplicate_ids = sorted(first_ids & second_ids)
    second_more["recommendation_evidence"] = {
        "first_turn_ids": sorted(first_ids),
        "second_turn_ids": sorted(second_ids),
        "duplicate_ids": duplicate_ids,
        "first_display_count": len(first_ids),
        "second_display_count": len(second_ids),
        "first_candidate_refs": sorted(first_candidate_refs),
        "previously_unshown_refs": sorted(previously_unshown_refs),
    }
    second_product_invokes = _product_invokes(second_more["events"])
    reference_only_reuse = not second_product_invokes and bool(second_ids)
    second_more["recommendation_evidence"]["reference_only_reuse"] = (
        reference_only_reuse
    )
    second_candidate_refs = _previous_product_refs(second_more)
    new_candidate_refs = second_candidate_refs - first_ids
    new_mentions = _mentioned_recommendations(
        second_more["final_output"], second_more["events"]
    )
    new_mentioned_refs = {
        item["product_id"]
        for item in new_mentions
        if item["product_id"] in new_candidate_refs
    }
    second_more["recommendation_evidence"].update(
        {
            "second_candidate_refs": sorted(second_candidate_refs),
            "new_candidate_refs": sorted(new_candidate_refs),
            "new_mentioned_refs": sorted(new_mentioned_refs),
        },
    )
    transparent_repeat_handling = any(
        phrase in second_more["final_output"]
        for phrase in (
            "不再重列",
            "不重复",
            "已展示过",
            "尚未展示",
            "上一轮",
            "上轮",
        )
    )
    if second_product_invokes and (
        not new_candidate_refs
        or not new_mentioned_refs
        or not transparent_repeat_handling
    ):
        second_more["passed"] = False
        second_more["failures"].append(
            "more-recommendations did not surface new candidates with transparent repeat handling"
        )
    if len(first_ids) > 5 or len(second_ids) > 5:
        second_more["passed"] = False
        second_more["failures"].append("final display exceeded five products")
    if not second_ids:
        second_more["passed"] = False
        second_more["failures"].append(
            "more-recommendations answer returned no verifiable prior or new product"
        )
    if len(first_ids) != 3:
        first_more["passed"] = False
        first_more["failures"].append(
            f"explicit quantity requested 3 but final mentioned {len(first_ids)}"
        )
    if not second_product_invokes:
        if previously_unshown_refs and not previously_unshown_refs.issubset(second_ids):
            second_more["passed"] = False
            second_more["failures"].append(
                "more-recommendations skipped previously recalled unshown candidates"
            )
        if not previously_unshown_refs and not any(
            phrase in second_more["final_output"]
            for phrase in (
                "没有新",
                "没有更多",
                "没有未展示",
                "没有其他未展示的新候选",
                "暂无新",
                "无新",
                "当前候选",
            )
        ):
            second_more["passed"] = False
            second_more["failures"].append(
                "more-recommendations reused context without transparently stating no new candidates"
            )

    preferred_calls = _product_search_calls(preferred["events"])
    plain_calls = _product_search_calls(plain["events"])
    preferred_queries = [
        call["args"].get("normalized_query") for call in preferred_calls
    ]
    plain_queries = [call["args"].get("normalized_query") for call in plain_calls]
    preference_equivalence = {
        "current_input_equal": preferred["input"] == plain["input"],
        "preferred_normalized_queries": preferred_queries,
        "plain_normalized_queries": plain_queries,
        "normalized_query_equal": preferred_queries == plain_queries,
        "preferred_retrieval": preferred_retrieval,
        "plain_retrieval": plain_retrieval,
        "embedding_query_equal": (
            preferred_retrieval["embedding_queries"]
            == plain_retrieval["embedding_queries"]
        ),
        "reranker_query_equal": (
            preferred_retrieval["reranker_queries"]
            == plain_retrieval["reranker_queries"]
        ),
    }
    preference_equivalence["passed"] = bool(
        preferred_queries
        and preferred_queries == plain_queries
        and preferred_retrieval["embedding_queries"]
        and preferred_retrieval["embedding_queries"]
        == plain_retrieval["embedding_queries"]
        and preferred_retrieval["reranker_queries"]
        and preferred_retrieval["reranker_queries"]
        == plain_retrieval["reranker_queries"]
    )
    if not preference_equivalence["passed"]:
        preferred["passed"] = False
        plain["passed"] = False
        preferred["failures"].append("history preference changed normalized_query")
        plain["failures"].append("history preference changed normalized_query")

    same_platform_turns = {
        turn["name"]: turn
        for turn in turns
        if turn["name"]
        in {
            "with-history-preference",
            "without-history-preference",
        }
    }
    session_refs = {
        name: (turn["structured_state"].get("last_search") or {}).get(
            "product_refs", []
        )
        for name, turn in same_platform_turns.items()
    }
    owner_checks = {
        name: (
            turn["structured_state"].get("session", {}).get("buyer_id")
            == (preferred_buyer if name == "with-history-preference" else plain_buyer)
            and turn["structured_state"].get("session", {}).get("shopping_session_id")
            == (
                f"h6-pref-{stamp}"
                if name == "with-history-preference"
                else f"h6-plain-{stamp}"
            )
            and turn["structured_state"].get("request", {}).get("current_raw_query")
            == preference_query
        )
        for name, turn in same_platform_turns.items()
    }
    isolation = {
        "platform": "reference_seed",
        "session_product_refs": session_refs,
        "all_sessions_populated": all(session_refs.values()),
        "owner_and_request_checks": owner_checks,
        "owner_and_request_isolated": all(owner_checks.values()),
        "same_query_may_correctly_return_same_products": True,
    }

    trade_lifecycle = {
        "created_order_id": order_id,
        "create_state": trade_create["structured_state"].get("order"),
        "query_state": trade_query_created["structured_state"].get("order"),
        "cancel_state": trade_cancel["structured_state"].get("order"),
        "passed": bool(
            order_id
            and (trade_create["structured_state"].get("order") or {}).get("status")
            == "CONFIRMED"
            and (trade_query_created["structured_state"].get("order") or {}).get(
                "status"
            )
            == "CONFIRMED"
            and (trade_cancel["structured_state"].get("order") or {}).get("status")
            == "CANCELLED"
        ),
    }
    if not trade_lifecycle["passed"]:
        for trade_turn in (trade_create, trade_query_created, trade_cancel):
            trade_turn["passed"] = False
            trade_turn["failures"].append(
                "Trade lifecycle structured order state mismatch"
            )

    return {
        "evidence_type": "real_online_main_agent_and_production_composition",
        "passed": all(turn["passed"] for turn in turns)
        and preference_embedding_probe["passed"]
        and preference_equivalence["passed"]
        and isolation["all_sessions_populated"]
        and isolation["owner_and_request_isolated"]
        and trade_lifecycle["passed"],
        "turns": turns,
        "preference_embedding_probe": preference_embedding_probe,
        "preference_query_equivalence": preference_equivalence,
        "multi_session_isolation": isolation,
        "trade_lifecycle": trade_lifecycle,
    }


async def _run_real_preference_embedding_probe(container: Container) -> dict[str, Any]:
    """Prove the distinct preference client can use the local product BGE service."""

    target = "喜欢轻便旅行背包"
    preferences = [
        BuyerPreference("h6-preference-probe", "like", target),
        BuyerPreference("h6-preference-probe", "like", "喜欢手冲咖啡杯"),
    ]
    try:
        selected = await PreferenceSelector(container.preference_embedder).select(
            preferences,
            query=target,
            top_k=1,
        )
    except Exception as error:  # noqa: BLE001 - persist only the safe error class
        return {
            "passed": False,
            "error_class": type(error).__name__,
            "selected_likes": [],
            "raw_vectors_persisted": False,
        }
    selected_likes = [item.statement for item in selected if item.kind == "like"]
    preference_embedder = container.preference_embedder
    product_embedder = container.product_embedder

    def _raw_embedder(embedder: Any) -> Any:
        while hasattr(embedder, "_inner"):
            embedder = embedder._inner  # noqa: SLF001
        return embedder

    raw_preference = _raw_embedder(preference_embedder)
    raw_product = _raw_embedder(product_embedder)
    return {
        "passed": selected_likes == [target],
        "selected_likes": selected_likes,
        "top_k": 1,
        "logical_client_distinct": preference_embedder is not product_embedder,
        "physical_endpoint_shared": (
            getattr(raw_preference, "_base_url", None)
            == getattr(raw_product, "_base_url", None)
        ),
        "model": getattr(raw_preference, "_model", None),
        "raw_vectors_persisted": False,
    }


async def run_real_empty_result_probes(
    container: Container,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Prove live ANN/BM25/Hybrid can return zero hits without an error."""
    impossible_id = "__h6_product_id_that_does_not_exist__"
    impossible_query = "h6nomatchtoken7f9c3a61e24b"
    vector = await container.product_embedder.embed(impossible_query)
    impossible_filter = {
        "bool": {
            "filter": [
                stock_filter(),
                {"term": {"product_id": impossible_id}},
            ],
        },
    }
    bodies = {
        "ann": {
            "size": 5,
            "query": {
                "knn": {
                    "content_vector": {
                        "vector": vector,
                        "k": 5,
                        "filter": impossible_filter,
                    },
                },
            },
        },
        "bm25": {
            "size": 5,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": impossible_query,
                                "fields": list(TEXT_FIELDS),
                            },
                        },
                    ],
                    "filter": [stock_filter()],
                },
            },
        },
        "hybrid": {
            "size": 5,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "knn": {
                                "content_vector": {
                                    "vector": vector,
                                    "k": 5,
                                    "filter": impossible_filter,
                                },
                            },
                        },
                        {
                            "bool": {
                                "must": [
                                    {
                                        "multi_match": {
                                            "query": impossible_query,
                                            "fields": list(TEXT_FIELDS),
                                        },
                                    },
                                ],
                                "filter": [impossible_filter],
                            },
                        },
                    ],
                },
            },
        },
    }
    results: dict[str, Any] = {}
    async with httpx.AsyncClient(
        base_url=container.settings.opensearch_endpoint.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        for name, body in bodies.items():
            params = (
                {"search_pipeline": RRF_PIPELINE_NAME} if name == "hybrid" else None
            )
            response = await client.post(
                f"/{INDEX_NAMES['crossshop_reference']}/_search",
                params=params,
                json=body,
            )
            response.raise_for_status()
            hits = response.json()["hits"]["hits"]
            results[name] = {
                "http_status": response.status_code,
                "hit_count": len(hits),
                "classification": "true_empty",
                "passed": hits == [],
            }
    return {
        "query_label": "redacted-impossible-token",
        "index": INDEX_NAMES["crossshop_reference"],
        "embedding_dimension": len(vector),
        "results": results,
        "passed": all(item["passed"] for item in results.values()),
    }


def _render_readme(report: dict[str, Any], args: argparse.Namespace) -> str:
    fault_rows = [
        f"| {item['name']} | {item['classification']} | "
        f"{'PASS' if item['passed'] else 'FAIL'} |"
        for item in report["fault_contracts"]["observations"]
    ]
    online = report.get("online")
    real_empty = report.get("real_empty_results")
    if real_empty is None:
        empty_text = "本次未请求真实 OpenSearch 空结果探针。"
    else:
        empty_text = "\n".join(
            f"- `{name}`：HTTP {item['http_status']}，hits={item['hit_count']}，"
            f"{'PASS' if item['passed'] else 'FAIL'}"
            for name, item in real_empty["results"].items()
        )
    if online is None:
        online_text = "本次未请求真实在线 Main Agent 场景。"
        preference_text = "本次未请求真实偏好 Embedding 探针。"
    else:
        online_rows = [
            f"| {turn['name']} | {turn['expectation']} | "
            f"{'PASS' if turn['passed'] else 'FAIL'} | "
            f"{turn['elapsed_seconds']:.3f} 秒 |"
            for turn in online["turns"]
        ]
        online_text = "\n".join(
            [
                "| 场景 | 验收点 | 结果 | 耗时 |",
                "|---|---|---|---:|",
                *online_rows,
            ],
        )
        preference_probe = online["preference_embedding_probe"]
        preference_text = (
            f"- 结果：{'PASS' if preference_probe['passed'] else 'FAIL'}\n"
            f"- 逻辑客户端独立：{preference_probe.get('logical_client_distinct')}\n"
            f"- 物理 BGE endpoint 与商品 Query 共用："
            f"{preference_probe.get('physical_endpoint_shared')}\n"
            f"- 真实 like Top-K：{preference_probe.get('selected_likes', [])}"
        )
    command = (
        ".venv\\Scripts\\python.exe scripts/eval/run_h6_acceptance.py "
        f"--output-root {args.output_root}" + (" --online" if args.online else "")
    )
    return f"""# H6 多轮、故障、降级与在线验收

本目录由 `scripts/eval/run_h6_acceptance.py` 原子生成。故障矩阵对生产 UseCase 与
既有韧性中间件做确定性故障注入；真实在线部分使用生产装配与本机服务。两类证据严格
分开，不把桩测试称为真实模型或真实检索质量。

## 实测命令

```powershell
{command}
```

## 故障与降级矩阵

| 场景 | 分类 | 结果 |
|---|---|---|
{chr(10).join(fault_rows)}

## 真实 OpenSearch 空结果

{empty_text}

## 真实在线 Main Agent

{online_text}

## 真实偏好 Embedding

{preference_text}

完整的脱敏输入、关键工具事件、结构化 L4 状态与最终输出保存在
`acceptance-report.json`。证据不含密钥、向量、完整文档或内部 correlation ID。

本验收未迁移品类 RAG、未部署 LangFuse、未扩大粗召回，也未新增 Agent、工具或架构层。
仅对商品 Embedding/OpenSearch 增加了获批的原参数一次瞬时依赖重试；不重试 Reranker，
也不占 H4 query 改写次数。
"""


def _overall_passed(
    *,
    online_requested: bool,
    fault_contracts: dict[str, Any],
    real_empty_results: dict[str, Any] | None,
    online: dict[str, Any] | None,
) -> bool:
    """H6 cannot pass without the mandatory real online acceptance."""
    return bool(
        online_requested
        and fault_contracts["passed"]
        and real_empty_results is not None
        and real_empty_results["passed"]
        and online is not None
        and online["passed"]
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _validated_output_root(args.output_root)
    fault_contracts = await run_fault_contracts()
    protocol_contracts = await run_deterministic_protocol()
    infrastructure = None
    real_empty_results = None
    online = None
    if args.online:
        container = await build_container()
        try:
            await container.startup()
            infrastructure = await run_real_infrastructure(
                container,
                timeout_seconds=args.timeout_seconds,
                output_root=output_root / "real-infrastructure",
            )
            real_empty_results = await run_real_empty_result_probes(
                container,
                timeout_seconds=args.timeout_seconds,
            )
            online = await run_online_acceptance(
                container,
                timeout_seconds=args.main_timeout_seconds,
            )
        finally:
            await container.shutdown()
    report = {
        "executed_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "source_fingerprint": {
            "acceptance_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "fault_contracts": fault_contracts,
        "protocol_contracts": protocol_contracts,
        "real_infrastructure": infrastructure,
        "real_empty_results": real_empty_results,
        "online": online,
        "passed": _overall_passed(
            online_requested=args.online,
            fault_contracts=fault_contracts,
            real_empty_results=real_empty_results,
            online=online,
        ),
        "scope": {
            "h6": True,
            "category_rag_migrated": False,
            "langfuse_deployed": False,
            "bounded_embedding_opensearch_retry": True,
            "reranker_retry": False,
            "empty_result_expansion": False,
        },
    }
    _atomic_json(output_root / "acceptance-report.json", report)
    _atomic_text(output_root / "README.md", _render_readme(report, args))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "h6-acceptance",
    )
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--main-timeout-seconds", type=float, default=180.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(run(args))
    _print_json_terminal_safe(report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
