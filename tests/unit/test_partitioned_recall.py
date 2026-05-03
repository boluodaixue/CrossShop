import importlib.util
from pathlib import Path

import pytest

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import MarketLocale, Platform
from globex_agent.infrastructure.recall import (
    CatalogPartition,
    LocalizedFusionSearchBackend,
    PartitionedSearchBackendRouter,
    RecallHit,
    RecallResult,
    embedding_document_text,
    standard_item_to_search_document,
)


class RecordingBackend:
    def __init__(self, backend_id: str, document_id: str) -> None:
        self._backend_id = backend_id
        self._document_id = document_id
        self.queries: list[str] = []

    @property
    def backend_id(self) -> str:
        return self._backend_id

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        self.queries.append(query)
        return RecallResult(
            backend_id=self.backend_id,
            hits=(RecallHit(self._document_id, 1.0, 1),),
            total_recall=1,
            truncated=False,
        )


def test_router_requires_locale_for_multi_locale_platform() -> None:
    us = RecordingBackend("us", "amazon:us:1")
    es = RecordingBackend("es", "amazon:es:1")
    router = PartitionedSearchBackendRouter(
        {
            CatalogPartition(Platform.AMAZON, MarketLocale.US): us,
            CatalogPartition(Platform.AMAZON, MarketLocale.ES): es,
        }
    )

    assert router.backend_for(Platform.AMAZON, MarketLocale.ES) is es
    with pytest.raises(ValueError, match="locale is required"):
        router.backend_for(Platform.AMAZON, None)


def test_localized_fusion_never_translates_the_semantic_query() -> None:
    semantic = RecordingBackend("semantic", "amazon:us:semantic")
    lexical = RecordingBackend("lexical", "amazon:us:lexical")
    backend = LocalizedFusionSearchBackend(
        lexical,
        semantic,
        lexical_query_localizer=lambda query: "wireless headphones",
    )

    result = backend.search("无线耳机", top_k=2)

    assert semantic.queries == ["无线耳机"]
    assert lexical.queries == ["wireless headphones"]
    assert {hit.document_id for hit in result.hits} == {
        "amazon:us:semantic",
        "amazon:us:lexical",
    }


def test_cuda_worker_reproduces_project_item_text(
    demo_catalog: LocalCatalog,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    worker_path = (
        project_root
        / "scripts"
        / "index"
        / "encode_catalog_items_cuda.py"
    )
    spec = importlib.util.spec_from_file_location("catalog_cuda_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ids, texts = module._load_item_texts(project_root / "data" / "demo" / "products.jsonl")
    expected = []
    for item in demo_catalog.items:
        document = standard_item_to_search_document(item)
        expected.append(embedding_document_text(document.title, document.body))

    assert ids == [item.item_id for item in demo_catalog.items]
    assert texts == expected
