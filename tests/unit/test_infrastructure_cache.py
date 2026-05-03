"""Cache fallback and semantic-cache safety tests."""

from __future__ import annotations

from globex_agent.infrastructure.cache.cached_embedding_client import (
    CachedEmbeddingClient,
)
from globex_agent.infrastructure.cache.redis_cache import RedisCache
from globex_agent.infrastructure.cache.semantic_cache import (
    SemanticCache,
    is_cacheable_query,
)


class InMemoryCache(RedisCache):
    def __init__(self) -> None:
        super().__init__("")
        self._store: dict = {}

    @property
    def enabled(self) -> bool:
        return True

    async def get_json(self, key):
        return self._store.get(key)

    async def set_json(self, key, value, ttl_seconds):
        self._store[key] = value

    async def delete(self, key):
        self._store.pop(key, None)

    async def set_if_absent(self, key, value, ttl_seconds):
        if key in self._store:
            return False
        self._store[key] = value
        return True


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = sum(ord(ch) for ch in text) % 97
        return [1.0, seed / 97.0, 0.1]


class TestCachedEmbedding:
    async def test_second_call_hits_cache(self) -> None:
        inner = CountingEmbedder()
        client = CachedEmbeddingClient(inner, InMemoryCache(), "bge-m3")
        first = await client.embed("露营灯")
        second = await client.embed("露营灯")
        assert first == second
        assert inner.calls == 1
        assert (client.hits, client.misses) == (1, 1)

    async def test_absent_redis_short_circuits(self) -> None:
        client = CachedEmbeddingClient(CountingEmbedder(), RedisCache(""), "m")
        assert await client.embed("x") == CountingEmbedder._vector("x")


class TestSemanticCache:
    def test_unsafe_queries_are_not_cacheable(self) -> None:
        assert is_cacheable_query("帮我下单这款露营灯") is False
        assert is_cacheable_query("刚才那款多少钱") is False
        assert is_cacheable_query("露营灯推荐") is True

    async def test_hit_and_buyer_isolation(self) -> None:
        cache = InMemoryCache()
        sem = SemanticCache(cache, CountingEmbedder(), threshold=0.95)
        await sem.remember("b1", "露营灯推荐", "推荐 LumenGo 89 元", has_history=False)
        hit = await sem.lookup("b1", "露营灯推荐", has_history=False)
        assert hit is not None and hit.reply == "推荐 LumenGo 89 元"
        assert await sem.lookup("b2", "露营灯推荐", has_history=False) is None

    async def test_disabled_without_redis(self) -> None:
        sem = SemanticCache(RedisCache(""), CountingEmbedder())
        assert sem.enabled is False
        assert await sem.lookup("b1", "露营灯推荐", has_history=False) is None
