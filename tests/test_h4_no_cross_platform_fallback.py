"""H4 platform failures must never fall back to the shared full catalog."""

import pytest

from app.application.usecases.catalog_search import (
    CatalogRetrievalError,
    CatalogSearchUseCase,
)
from app.domain.catalog.product_search_spec import ProductSearchSpec


class _RecordingSharedRepository:
    def __init__(self) -> None:
        self.list_all_calls = 0

    async def find_by_ids(self, product_ids: list[str]) -> list:
        del product_ids
        return []

    async def list_all(self) -> list:
        self.list_all_calls += 1
        raise AssertionError("共享全量 Repository 不得用于平台故障降级")


class _QueryEmbedder:
    async def embed(self, text: str) -> list[float]:
        del text
        return [0.0] * 1024


class _FailingPlatformIndex:
    async def search(self, *, query: str, embedding: list[float], top_n: int):
        del query, embedding, top_n
        raise RuntimeError("OpenSearch unavailable")


class _EmptyPlatformIndex:
    async def search(self, *, query: str, embedding: list[float], top_n: int):
        del query, embedding, top_n
        return []


@pytest.mark.parametrize("route", ["taobao", "amazon:jp"])
async def test_platform_failure_never_scans_shared_catalog(route: str) -> None:
    repository = _RecordingSharedRepository()
    usecases = {
        route: CatalogSearchUseCase(
            repository,  # type: ignore[arg-type]
            embedder=_QueryEmbedder(),  # type: ignore[arg-type]
            vector_index=_FailingPlatformIndex(),  # type: ignore[arg-type]
            allow_keyword_fallback=False,
        ),
    }

    with pytest.raises(CatalogRetrievalError, match="商品 Hybrid 检索不可用"):
        await usecases[route].execute(ProductSearchSpec(normalized_query="旅行背包"))

    assert repository.list_all_calls == 0


async def test_normal_empty_hybrid_result_is_a_true_empty_without_second_query() -> (
    None
):
    repository = _RecordingSharedRepository()
    usecase = CatalogSearchUseCase(
        repository,  # type: ignore[arg-type]
        embedder=_QueryEmbedder(),  # type: ignore[arg-type]
        vector_index=_EmptyPlatformIndex(),  # type: ignore[arg-type]
        allow_keyword_fallback=False,
    )

    result = await usecase.execute(ProductSearchSpec(normalized_query="不存在的商品"))

    assert result == {
        "hits": [],
        "total_candidates": 0,
        "recall_strategy": "embedding_only",
        "rerank_applied": False,
    }
    assert repository.list_all_calls == 0
