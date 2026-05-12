"""OpenAI-compatible chat model factory for all agents."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI

from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.settings import Settings
from globex_agent.infrastructure.throttle import GatewayThrottle
from globex_agent.infrastructure.transient import is_transient_error


class _ThrottledRunnable(Runnable[Any, Any]):
    """Enforce gateway concurrency and start-interval limits on async calls."""

    def __init__(self, inner: Runnable[Any, Any], throttle: GatewayThrottle) -> None:
        self._inner = inner
        self._throttle = throttle

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._inner.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        async with self._throttle.slot():
            return await self._inner.ainvoke(input, config=config, **kwargs)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        yield from self._inner.stream(input, config=config, **kwargs)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async with self._throttle.slot():
            async for item in self._inner.astream(
                input,
                config=config,
                **kwargs,
            ):
                yield item


class _FallbackRunnable(Runnable[Any, Any]):
    """Call the primary model and fall back only on transient upstream errors."""

    def __init__(
        self,
        primary: Runnable[Any, Any],
        fallback: Runnable[Any, Any],
        bus: TradeEventBus | None,
        primary_model: str,
        fallback_model: str,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._bus = bus
        self._primary_model = primary_model
        self._fallback_model = fallback_model

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            return self._primary.invoke(input, config=config, **kwargs)
        except Exception as err:  # noqa: BLE001 - fallback boundary
            if not is_transient_error(err):
                raise
            self._publish_fallback(err)
            return self._fallback.invoke(input, config=config, **kwargs)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            return await self._primary.ainvoke(input, config=config, **kwargs)
        except Exception as err:  # noqa: BLE001 - fallback boundary
            if not is_transient_error(err):
                raise
            self._publish_fallback(err)
            return await self._fallback.ainvoke(input, config=config, **kwargs)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        try:
            yield from self._primary.stream(input, config=config, **kwargs)
        except Exception as err:  # noqa: BLE001 - fallback boundary
            if not is_transient_error(err):
                raise
            self._publish_fallback(err)
            yield from self._fallback.stream(input, config=config, **kwargs)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        try:
            async for item in self._primary.astream(
                input,
                config=config,
                **kwargs,
            ):
                yield item
        except Exception as err:  # noqa: BLE001 - fallback boundary
            if not is_transient_error(err):
                raise
            self._publish_fallback(err)
            async for item in self._fallback.astream(
                input,
                config=config,
                **kwargs,
            ):
                yield item

    def _publish_fallback(self, error: Exception) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            ShoppingContext.current_session_id(),
            "model.fallback",
            {
                "from": self._primary_model,
                "to": self._fallback_model,
                "reason": str(error),
            },
        )


class FallbackChatModel:
    """Minimal model facade used by ``create_react_agent``."""

    def __init__(
        self,
        primary: ChatOpenAI,
        fallback: ChatOpenAI | None,
        bus: TradeEventBus | None,
        throttle: GatewayThrottle,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._bus = bus
        self._throttle = throttle

    def bind_tools(self, tools, **kwargs: Any) -> Runnable[Any, Any]:
        primary = _ThrottledRunnable(
            self._primary.bind_tools(tools, **kwargs),
            self._throttle,
        )
        if self._fallback is None:
            return primary
        fallback = _ThrottledRunnable(
            self._fallback.bind_tools(tools, **kwargs),
            self._throttle,
        )
        return _FallbackRunnable(
            primary,
            fallback,
            self._bus,
            self._primary.model,
            self._fallback.model,
        )


def create_chat_model(
    settings: Settings,
    bus: TradeEventBus | None = None,
) -> Any:
    """Create the primary OpenAI-compatible Chat Completions model."""

    common = dict(
        api_key=settings.llm_api_key or "not-configured",
        base_url=settings.llm_base_url,
        temperature=0.2,
        max_retries=settings.llm_max_retries,
    )
    primary = ChatOpenAI(model=settings.llm_model, **common)
    fallback = None
    if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model:
        fallback = ChatOpenAI(model=settings.llm_fallback_model, **common)
    throttle = GatewayThrottle(
        max_concurrency=settings.llm_max_concurrency,
        min_interval_seconds=settings.llm_min_interval_seconds,
    )
    return FallbackChatModel(primary, fallback, bus, throttle)
