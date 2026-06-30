"""H4 online Product OpenSearch composition wiring tests."""

from pathlib import Path
from types import SimpleNamespace

from app.catalog.opensearch_product_h1 import INDEX_NAMES, VECTOR_DIMENSION
from app.infrastructure.settings import Settings, load_settings
from app.composition import Container


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://llm.invalid/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        port=8000,
        log_level="info",
        embedding_base_url="http://rag-embedding.invalid/v1",
        embedding_api_key="rag-key",
        embedding_model="text-embedding-v4",
        embedding_dim=1024,
        qdrant_url="",
        qdrant_collection="unused-product-collection",
        reranker_base_url="http://reranker.invalid",
        reranker_model="BAAI/bge-reranker-v2-m3",
        tavily_api_key="",
        otlp_endpoint="",
        data_dir=tmp_path,
        category_kb_collection="test-category-kb",
        context_size=128000,
        tool_result_limit=20000,
        reply_token_budget=0,
        tool_failure_threshold=3,
        tool_circuit_reset_seconds=60.0,
        cors_origins=["http://localhost:5173"],
        database_url="file",
        redis_url="",
        product_catalog_root=tmp_path / "catalogs-v2",
        opensearch_endpoint="http://opensearch.invalid:9200",
        opensearch_timeout_seconds=17.0,
        product_embedding_base_url="http://bge-query.invalid/v1",
        product_embedding_api_key="product-key",
        product_embedding_model="BAAI/bge-m3",
        product_embedding_dim=1024,
    )


class _FakeProductRepository:
    def __init__(self, catalog_root: Path) -> None:
        self.catalog_root = catalog_root


class _FakeOpenSearchProductIndex:
    def __init__(
        self,
        endpoint: str,
        index_name: str,
        *,
        timeout_seconds: float,
        site_locale: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.index_name = index_name
        self.timeout_seconds = timeout_seconds
        self.site_locale = site_locale
        self.ensure_calls: list[int] = []
        self.closed = False

    async def ensure_ready(self, vector_dim: int) -> None:
        self.ensure_calls.append(vector_dim)

    async def close(self) -> None:
        self.closed = True


async def test_opensearch_startup_failure_is_a_non_blocking_capability_state() -> None:
    class Unavailable:
        async def ensure_ready(self, vector_dim: int) -> None:
            del vector_dim
            raise RuntimeError("dependency down")

    container = SimpleNamespace(
        db_engine=None,
        task_queue=None,
        product_indexes={"taobao": Unavailable()},
        product_search_startup_check={},
        knowledge_base=None,
    )

    await Container.startup(container)

    assert container.product_search_startup_check == {
        "taobao": "unavailable:RuntimeError",
    }


async def test_build_container_wires_three_read_only_product_indexes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from app import composition

    settings = _settings(tmp_path)
    monkeypatch.setattr(composition, "load_settings", lambda: settings)
    monkeypatch.setattr(composition, "setup_tracing", lambda _: None)
    monkeypatch.setattr(composition, "JsonlProductRepository", _FakeProductRepository)
    monkeypatch.setattr(
        composition,
        "OpenSearchProductIndex",
        _FakeOpenSearchProductIndex,
    )
    monkeypatch.setattr(
        composition,
        "build_category_knowledge_base",
        lambda _: object(),
    )

    container = await composition.build_container()

    assert container.product_repo.catalog_root == settings.product_catalog_root
    expected_routes = {
        *INDEX_NAMES,
        "amazon:us",
        "amazon:es",
        "amazon:jp",
    }
    assert set(container.product_indexes) == expected_routes
    assert set(container.catalog_searches) == expected_routes
    assert {
        platform: index.index_name
        for platform, index in container.product_indexes.items()
        if platform in INDEX_NAMES
    } == INDEX_NAMES
    assert {
        route: index.site_locale
        for route, index in container.product_indexes.items()
        if route.startswith("amazon:")
    } == {"amazon:us": "us", "amazon:es": "es", "amazon:jp": "jp"}
    assert all(
        index.endpoint == settings.opensearch_endpoint
        and index.timeout_seconds == settings.opensearch_timeout_seconds
        for index in container.product_indexes.values()
    )

    usecases = list(container.catalog_searches.values())
    assert all(usecase._product_repo is container.product_repo for usecase in usecases)
    assert all(usecase._embedder is container.product_embedder for usecase in usecases)
    assert all(usecase._reranker is usecases[0]._reranker for usecase in usecases)
    assert all(usecase._allow_keyword_fallback is False for usecase in usecases)
    assert {usecase._vector_index.index_name for usecase in usecases} == set(
        INDEX_NAMES.values()
    )
    assert container.product_embedder._model == "BAAI/bge-m3"
    assert container.preference_embedder._model == "BAAI/bge-m3"
    assert container.preference_embedder._base_url == "http://bge-query.invalid/v1"
    assert container.embedder._model == "text-embedding-v4"

    async def _skip_knowledge_bootstrap(_knowledge_base) -> int:
        return 0

    monkeypatch.setattr(
        composition,
        "bootstrap_category_knowledge",
        _skip_knowledge_bootstrap,
    )
    await container.startup()
    assert all(
        container.product_indexes[platform].ensure_calls == [VECTOR_DIMENSION]
        for platform in INDEX_NAMES
    )
    assert all(
        container.product_indexes[f"amazon:{site}"].ensure_calls == []
        for site in ("us", "es", "jp")
    )

    await container.shutdown()
    assert all(index.closed for index in container.product_indexes.values())


def test_load_settings_separates_product_query_embedding(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("PRODUCT_EMBEDDING_BASE_URL", "http://127.0.0.1:8002/v1")
    monkeypatch.setenv("PRODUCT_EMBEDDING_API_KEY", "product-key")
    monkeypatch.setenv("PRODUCT_EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("PRODUCT_EMBEDDING_DIM", "1024")
    monkeypatch.setenv("PRODUCT_CATALOG_ROOT", str(tmp_path / "catalogs-v2"))
    monkeypatch.setenv("OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200")
    monkeypatch.setenv("OPENSEARCH_TIMEOUT_SECONDS", "19")

    settings = load_settings()

    assert settings.embedding_model == "text-embedding-v4"
    assert settings.product_embedding_model == "BAAI/bge-m3"
    assert settings.product_embedding_dim == VECTOR_DIMENSION
    assert settings.product_embedding_base_url == "http://127.0.0.1:8002/v1"
    assert settings.product_embedding_api_key == "product-key"
    assert settings.preference_embedding_model == "BAAI/bge-m3"
    assert settings.preference_embedding_base_url == "http://127.0.0.1:8002/v1"
    assert settings.preference_embedding_api_key == "product-key"
    assert settings.product_catalog_root == tmp_path / "catalogs-v2"
    assert settings.opensearch_endpoint == "http://127.0.0.1:9200"
    assert settings.opensearch_timeout_seconds == 19.0
