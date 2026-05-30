"""Strict shared resources for LangGraph Redis checkpoints and app markers."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from globex_agent.application.agents.identity import ThreadIdentity
from globex_agent.infrastructure.settings import Settings


class CheckpointConfigurationError(RuntimeError):
    """Raised when checkpoint persistence cannot honor application settings."""


class DispatchResultStore(Protocol):
    async def get(self, dispatch_id: str) -> dict[str, Any] | None: ...

    async def put(self, dispatch_id: str, result: dict[str, Any]) -> None: ...


class InMemoryDispatchResultStore:
    """Explicit test double; production composition always uses Redis."""

    def __init__(self) -> None:
        self._values: dict[str, dict[str, Any]] = {}

    async def get(self, dispatch_id: str) -> dict[str, Any] | None:
        return self._values.get(dispatch_id)

    async def put(self, dispatch_id: str, result: dict[str, Any]) -> None:
        self._values[dispatch_id] = dict(result)


class RedisDispatchResultStore:
    """Strict app-owned marker storage, separate from saver-managed keys."""

    def __init__(
        self,
        redis_url: str,
        identity: ThreadIdentity,
        *,
        ttl_seconds: int,
    ) -> None:
        if not redis_url:
            raise CheckpointConfigurationError(
                "CHECKPOINT_REDIS_URL or REDIS_URL is required for dispatch markers"
            )
        if ttl_seconds <= 0:
            raise ValueError("dispatch marker TTL must be greater than zero")
        import redis.asyncio as aioredis

        self._client = aioredis.from_url(redis_url, decode_responses=True)
        self._identity = identity
        self._ttl_seconds = ttl_seconds
        self._prefix = (
            f"globex:{identity.environment}:v{identity.version.lstrip('v')}:"
            "dispatch_result:"
        )

    def _key(self, dispatch_id: str) -> str:
        return self._prefix + self._identity.digest("result:" + dispatch_id)

    async def startup(self) -> None:
        try:
            await self._client.ping()
        except Exception as err:  # noqa: BLE001 - checkpoint startup is strict
            await self.close()
            raise CheckpointConfigurationError(
                f"dispatch marker Redis is unavailable: {err}"
            ) from err

    async def get(self, dispatch_id: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._key(dispatch_id))
        if raw is None:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise CheckpointConfigurationError("invalid dispatch marker payload")
        return value

    async def put(self, dispatch_id: str, result: dict[str, Any]) -> None:
        await self._client.set(
            self._key(dispatch_id),
            json.dumps(result, ensure_ascii=False),
            ex=self._ttl_seconds,
        )

    async def close(self) -> None:
        with suppress(Exception):
            await self._client.aclose()


@dataclass
class RedisCheckpointResource:
    """Own one official saver for the entire FastAPI process."""

    saver: AsyncRedisSaver
    _closed: bool = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # AsyncRedisSaver 0.5.x exposes lifecycle through the async context
        # manager, not a public aclose() method.
        await self.saver.__aexit__(None, None, None)


async def create_redis_checkpoint_resource(settings: Settings) -> RedisCheckpointResource:
    """Construct and initialize the official async saver, failing fast on errors."""

    redis_url = settings.checkpoint_redis_url
    if not redis_url:
        raise CheckpointConfigurationError(
            "CHECKPOINT_REDIS_URL or REDIS_URL is required; refusing InMemorySaver fallback"
        )
    if settings.checkpoint_require_encryption and not settings.checkpoint_encryption_key:
        raise CheckpointConfigurationError(
            "LANGGRAPH_AES_KEY is required when CHECKPOINT_REQUIRE_ENCRYPTION is enabled"
        )
    if settings.checkpoint_encryption_key:
        # AsyncRedisSaver 0.5.2 hard-codes JsonPlusRedisSerializer and its
        # public constructor has no serde parameter. Do not claim encryption
        # when a key would be ignored by the official Redis implementation.
        raise CheckpointConfigurationError(
            "LANGGRAPH_AES_KEY is configured, but langgraph-checkpoint-redis "
            "0.5.x does not expose serializer injection for AsyncRedisSaver"
        )

    saver = AsyncRedisSaver(
        redis_url=redis_url,
        ttl={
            "default_ttl": settings.checkpoint_ttl_minutes,
            "refresh_on_read": settings.checkpoint_refresh_on_read,
        },
    )
    resource = RedisCheckpointResource(saver)
    try:
        await saver.asetup()
    except Exception as err:  # noqa: BLE001 - startup must fail fast
        with suppress(Exception):
            await resource.close()
        raise CheckpointConfigurationError(
            f"LangGraph Redis Checkpointer setup failed: {err}"
        ) from err
    return resource
