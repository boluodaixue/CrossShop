"""Composition root shared by the API server and the optional queue worker."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from globex_agent.application.agents.main_agent import (
    MainAgentFactory,
    SessionRegistry,
)
from globex_agent.application.agents.orchestrator import MainAgentOrchestrator
from globex_agent.application.agents.search_agent import SearchAgentFactory
from globex_agent.application.agents.trade_agent import TradeAgentFactory
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.application.usecases.order_usecases import (
    CancelOrderUseCase,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from globex_agent.catalog import LocalCatalog
from globex_agent.category_insight import (
    CategoryInsightService,
    CategoryTaxonomy,
)
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.infrastructure.cache.redis_cache import RedisCache
from globex_agent.infrastructure.cache.semantic_cache import SemanticCache
from globex_agent.infrastructure.embedding.bge_m3_embedding import BgeM3EmbeddingClient
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
    InMemoryOrderRepository,
)
from globex_agent.infrastructure.persistence.json_file_stores import (
    JsonFileConversationStore,
    JsonFilePreferenceStore,
    JsonFileSessionStore,
)
from globex_agent.infrastructure.persistence.sql.repositories import (
    SqlConversationStore,
    SqlOrderRepository,
    SqlPreferenceStore,
    SqlSessionStore,
    bootstrap_schema,
    create_engine,
)
from globex_agent.infrastructure.persistence.sqlite_item_repository import (
    SqliteItemRepository,
)
from globex_agent.infrastructure.queue.redis_stream_queue import (
    RedisEventBackplane,
    RedisStreamTaskQueue,
)
from globex_agent.infrastructure.recall.category_kb import (
    OpenSearchCategoryKnowledgeBase,
    OpenSearchHttpClient,
)
from globex_agent.infrastructure.recall.embedding import SentenceTransformerTextEncoder
from globex_agent.infrastructure.recall.index import FaissHNSWIndex
from globex_agent.infrastructure.recall.reranker import (
    CrossEncoderReranker,
    SubprocessCrossEncoderReranker,
)
from globex_agent.infrastructure.rerank.bge_reranker import BgeReranker
from globex_agent.infrastructure.resilience import CircuitBreakerRegistry
from globex_agent.infrastructure.settings import Settings, load_settings
from globex_agent.infrastructure.vector.faiss_product_index import (
    FaissProductIndex,
    PartitionedFaissProductIndex,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_PRODUCTS_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"
_PARTITIONS = (
    ("amazon", "us"),
    ("amazon", "es"),
    ("amazon", "jp"),
    ("taobao", "cn"),
)


@dataclass
class Container:
    settings: Settings
    bus: TradeEventBus
    orchestrator: MainAgentOrchestrator
    cache: RedisCache
    semantic_cache: SemanticCache
    task_queue: RedisStreamTaskQueue | None
    backplane: RedisEventBackplane | None
    query_order: QueryOrderUseCase
    cancel_order: CancelOrderUseCase
    item_repo: ItemRepository
    embedder: BgeM3EmbeddingClient
    vector_index: FaissProductIndex | PartitionedFaissProductIndex
    db_engine: object | None

    async def startup(self) -> None:
        if self.db_engine is not None:
            try:
                await bootstrap_schema(self.db_engine)
            except Exception as err:  # noqa: BLE001 - startup must stay resilient
                logger.warning("数据库建表失败，持久化能力不可用：%s", err)
        if isinstance(self.task_queue, RedisStreamTaskQueue):
            try:
                await self.task_queue.ensure_group()
            except Exception as err:  # noqa: BLE001
                logger.warning("队列消费者组创建失败：%s", err)

    async def shutdown(self) -> None:
        await self.vector_index.close()
        await self.cache.close()
        if self.db_engine is not None:
            await self.db_engine.dispose()


async def build_container() -> Container:
    settings = load_settings()
    bus = TradeEventBus()
    cache = RedisCache(settings.redis_url)
    embedder = BgeM3EmbeddingClient(
        model_name=settings.bge_m3_model,
        local_files_only=settings.models_local_only,
    )
    vector_index = _build_vector_index()
    reranker = BgeReranker(
        model_name=settings.bge_reranker_model,
        local_files_only=settings.models_local_only,
    )

    item_repo = _build_item_repository()
    catalog_search = CatalogSearchUseCase(
        item_repo,
        embedder=embedder,
        vector_index=vector_index,
        reranker=reranker,
    )

    use_database = settings.database_url != "file"
    db_engine = create_engine(settings.database_url) if use_database else None
    if db_engine is not None:
        order_repo = SqlOrderRepository(db_engine)
        preference_store = SqlPreferenceStore(db_engine)
        session_store = SqlSessionStore(db_engine)
        conversation_store = SqlConversationStore(db_engine)
    else:
        order_repo = InMemoryOrderRepository()
        preference_store = JsonFilePreferenceStore(settings.data_dir)
        session_store = JsonFileSessionStore(settings.data_dir)
        conversation_store = JsonFileConversationStore(settings.data_dir)

    place_order = PlaceOrderUseCase(item_repo, order_repo)
    query_order = QueryOrderUseCase(order_repo)
    cancel_order = CancelOrderUseCase(order_repo)

    circuit_registry = CircuitBreakerRegistry(
        failure_threshold=settings.tool_failure_threshold,
        reset_seconds=settings.tool_circuit_reset_seconds,
    )
    semantic_cache = SemanticCache(
        cache,
        embedder,
        threshold=settings.semantic_cache_threshold,
        enabled=settings.semantic_cache_enabled,
        namespace=f"{settings.llm_model}:prompt-v1",
    )

    task_queue: RedisStreamTaskQueue | None = None
    backplane: RedisEventBackplane | None = None
    if cache.enabled and settings.queue_enabled:
        task_queue = RedisStreamTaskQueue(cache.client)
        backplane = RedisEventBackplane(cache.client)
        bus.attach_backplane(backplane)

    search_factory = SearchAgentFactory(
        settings,
        catalog_search,
        bus,
        _build_category_insight_service(settings),
        circuit_registry,
    )
    trade_factory = TradeAgentFactory(
        settings,
        place_order,
        query_order,
        cancel_order,
        bus,
        circuit_registry,
    )
    main_factory = MainAgentFactory(
        settings,
        search_factory,
        trade_factory,
        bus,
        preference_store,
        circuit_registry,
    )
    sessions = SessionRegistry(main_factory, session_store)
    orchestrator = MainAgentOrchestrator(
        sessions,
        bus,
        preference_store,
        conversation_store,
        semantic_cache,
    )
    return Container(
        settings=settings,
        bus=bus,
        orchestrator=orchestrator,
        cache=cache,
        semantic_cache=semantic_cache,
        task_queue=task_queue,
        backplane=backplane,
        query_order=query_order,
        cancel_order=cancel_order,
        item_repo=item_repo,
        embedder=embedder,
        vector_index=vector_index,
        db_engine=db_engine,
    )


def _build_item_repository() -> ItemRepository:
    database_paths = [
        PROJECT_ROOT / "data" / "processed" / "databases" / "taobao" / "catalog.sqlite3",
        PROJECT_ROOT / "data" / "processed" / "databases" / "amazon" / "catalog.sqlite3",
    ]
    existing = [path for path in database_paths if path.exists()]
    if existing:
        return SqliteItemRepository(existing)
    catalog = LocalCatalog.from_jsonl(DEMO_PRODUCTS_PATH, strict=True).catalog
    return InMemoryItemRepository(list(catalog.items))


def _build_vector_index() -> FaissProductIndex | PartitionedFaissProductIndex:
    vector_index = PartitionedFaissProductIndex()
    loaded = False
    for platform, locale in _PARTITIONS:
        partition_id = f"{platform}:{locale}"
        faiss_path = (
            PROJECT_ROOT
            / "output"
            / "index"
            / "catalog"
            / platform
            / locale
            / "bge-m3-hnsw-ip.faiss"
        )
        embeddings_path = (
            PROJECT_ROOT
            / "output"
            / "embeddings"
            / "catalog"
            / platform
            / locale
            / "bge-m3-items.npz"
        )
        if not faiss_path.exists() or not embeddings_path.exists():
            continue
        with np.load(embeddings_path, allow_pickle=False) as payload:
            document_ids = payload["document_ids"].astype(str).tolist()
        index = FaissHNSWIndex.load(faiss_path, document_ids)
        vector_index.add_partition(partition_id, index)
        loaded = True
    return vector_index if loaded else FaissProductIndex()


def _build_category_insight_service(settings: Settings) -> CategoryInsightService:
    taxonomy_path = settings.category_taxonomy
    if not taxonomy_path.is_absolute():
        taxonomy_path = PROJECT_ROOT / taxonomy_path
    encoder = SentenceTransformerTextEncoder(
        local_files_only=settings.models_local_only
    )
    knowledge_base = OpenSearchCategoryKnowledgeBase(
        OpenSearchHttpClient(settings.opensearch_endpoint, timeout=30),
        index_name=settings.category_index,
    )
    reranker = None
    if settings.category_reranker_enabled:
        if settings.category_reranker_python:
            reranker = SubprocessCrossEncoderReranker(
                Path(settings.category_reranker_python),
                device=settings.category_reranker_device,
                batch_size=settings.category_reranker_batch_size,
                use_fp16=not settings.category_reranker_fp32,
                local_files_only=settings.models_local_only,
            )
        else:
            reranker = CrossEncoderReranker(
                device="cpu",
                batch_size=1,
                local_files_only=settings.models_local_only,
            )
    return CategoryInsightService(
        CategoryTaxonomy.from_json(taxonomy_path),
        encoder,
        knowledge_base,
        reranker,
        reranker_document_mode=settings.category_reranker_document_mode,
    )
