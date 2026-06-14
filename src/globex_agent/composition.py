"""Composition root shared by the API server and the optional queue worker."""

from __future__ import annotations

import json
import logging
import ntpath
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from globex_agent.application.agents.identity import ThreadIdentity
from globex_agent.application.agents.main_agent import (
    MainAgentFactory,
    SessionRegistry,
)
from globex_agent.application.agents.orchestrator import MainAgentOrchestrator
from globex_agent.application.agents.search_agent import SearchAgentFactory
from globex_agent.application.agents.trade_agent import TradeAgentFactory
from globex_agent.application.evidence_verification import (
    EvidenceVerificationService,
    OpenAIEvidenceJudge,
)
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.application.usecases.confirmation_usecases import (
    OrderConfirmationService,
)
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
from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable
from globex_agent.domain.catalog.ports.item_repository import ItemRepository
from globex_agent.domain.catalog.ports.retrieval_ports import EmbeddingClient
from globex_agent.domain.shipping.tariff_schedule import TariffSchedule
from globex_agent.infrastructure.cache.redis_cache import RedisCache
from globex_agent.infrastructure.cache.semantic_cache import SemanticCache
from globex_agent.infrastructure.checkpoint import (
    RedisCheckpointResource,
    RedisDispatchResultStore,
    create_redis_checkpoint_resource,
)
from globex_agent.infrastructure.embedding.bge_m3_embedding import (
    BgeM3EmbeddingClient,
    SubprocessBgeM3EmbeddingClient,
)
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.llm import create_chat_model
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
    InMemoryOrderRepository,
)
from globex_agent.infrastructure.persistence.json_file_stores import (
    JsonFileConversationStore,
    JsonFilePreferenceStore,
)
from globex_agent.infrastructure.persistence.sql.repositories import (
    SqlConversationStore,
    SqlOrderRepository,
    SqlPreferenceStore,
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
from globex_agent.infrastructure.recall.persistence import index_manifest_compatible
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
DEMO_PRODUCTS_PATH = PROJECT_ROOT / "data" / "demo" / "products-v2.jsonl"
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
    confirmation_service: OrderConfirmationService
    item_repo: ItemRepository
    embedder: EmbeddingClient
    reranker: BgeReranker
    vector_index: FaissProductIndex | PartitionedFaissProductIndex
    db_engine: object | None
    checkpoint_resource: RedisCheckpointResource
    dispatch_result_store: RedisDispatchResultStore
    evidence_judge_client: httpx.AsyncClient | None = None

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
        if self.evidence_judge_client is not None:
            await self.evidence_judge_client.aclose()
        await self.vector_index.close()
        close = getattr(self.embedder, "close", None)
        if close is not None:
            close()
        close = getattr(self.reranker, "close", None)
        if close is not None:
            close()
        await self.cache.close()
        if self.db_engine is not None:
            await self.db_engine.dispose()
        await self.dispatch_result_store.close()
        await self.checkpoint_resource.close()


async def build_container() -> Container:
    settings = load_settings()
    identity = ThreadIdentity(
        environment=settings.checkpoint_environment,
        hmac_key=settings.thread_id_hmac_key,
    )
    checkpoint_resource = await create_redis_checkpoint_resource(settings)
    dispatch_result_store: RedisDispatchResultStore | None = None
    try:
        dispatch_result_store = RedisDispatchResultStore(
            settings.checkpoint_redis_url,
            identity,
            ttl_seconds=settings.checkpoint_ttl_minutes * 60,
        )
        await dispatch_result_store.startup()
        bus = TradeEventBus()
        cache = RedisCache(settings.redis_url)
        vector_index = _build_vector_index()
        _validate_catalog_index_contract(
            settings.bge_m3_model,
            max_seq_length=settings.bge_m3_max_seq_length,
        )
        embedder = _build_query_embedder(settings)
        reranker = _build_product_reranker(settings)

        item_repo = _build_item_repository()
        tariff_schedule = TariffSchedule(rates=ExchangeRateTable())
        catalog_search = CatalogSearchUseCase(
            item_repo,
            embedder=embedder,
            vector_index=vector_index,
            reranker=reranker,
            tariff_schedule=tariff_schedule,
        )

        use_database = settings.database_url != "file"
        db_engine = create_engine(settings.database_url) if use_database else None
        if db_engine is not None:
            order_repo = SqlOrderRepository(db_engine)
            preference_store = SqlPreferenceStore(db_engine)
            conversation_store = SqlConversationStore(db_engine)
        else:
            order_repo = InMemoryOrderRepository()
            preference_store = JsonFilePreferenceStore(settings.data_dir)
            conversation_store = JsonFileConversationStore(settings.data_dir)

        place_order = PlaceOrderUseCase(item_repo, order_repo, tariff_schedule=tariff_schedule)
        query_order = QueryOrderUseCase(order_repo)
        cancel_order = CancelOrderUseCase(order_repo)
        confirmation_service = OrderConfirmationService(
            item_repo,
            order_repo,
            place_order,
            cancel_order,
            tariff_schedule=tariff_schedule,
            ttl_seconds=settings.order_confirmation_ttl_seconds,
        )

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
            checkpointer=checkpoint_resource.saver,
        )
        trade_factory = TradeAgentFactory(
            settings,
            place_order,
            query_order,
            cancel_order,
            bus,
            circuit_registry,
            confirmation_service=confirmation_service,
            checkpointer=checkpoint_resource.saver,
        )
        main_factory = MainAgentFactory(
            settings,
            search_factory,
            trade_factory,
            bus,
            preference_store,
            circuit_registry,
            checkpointer=checkpoint_resource.saver,
            identity=identity,
            dispatch_result_store=dispatch_result_store,
        )
        sessions = SessionRegistry(main_factory, identity)
        evidence_judge_client = None
        evidence_judge = None
        if settings.online_evidence_judge_model and settings.llm_base_url:
            evidence_judge_client = httpx.AsyncClient()
            evidence_judge = OpenAIEvidenceJudge(
                evidence_judge_client,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                model=settings.online_evidence_judge_model,
                timeout_seconds=settings.evidence_judge_timeout_seconds,
            )
        # Retry/fallback turns are plain answer finalization and must not have
        # access to the production tools or durable Agent graph.
        answer_finalizer = create_chat_model(settings, bus).bind_tools([])
        orchestrator = MainAgentOrchestrator(
            sessions,
            bus,
            preference_store,
            conversation_store,
            semantic_cache,
            evidence_verifier=EvidenceVerificationService(
                evidence_judge,
                judge_timeout_seconds=settings.evidence_judge_timeout_seconds,
                judge_total_timeout_seconds=settings.evidence_judge_total_timeout_seconds,
            ),
            answer_finalizer=answer_finalizer,
            max_generation_attempts=3,
            turn_timeout_seconds=settings.turn_timeout_seconds,
            generation_timeout_seconds=settings.generation_timeout_seconds,
            rewrite_timeout_seconds=settings.rewrite_timeout_seconds,
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
            confirmation_service=confirmation_service,
            item_repo=item_repo,
            embedder=embedder,
            reranker=reranker,
            vector_index=vector_index,
            db_engine=db_engine,
            checkpoint_resource=checkpoint_resource,
            dispatch_result_store=dispatch_result_store,
            evidence_judge_client=evidence_judge_client,
        )
    except Exception:
        if dispatch_result_store is not None:
            with suppress(Exception):
                await dispatch_result_store.close()
        with suppress(Exception):
            await checkpoint_resource.close()
        raise


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
        mapping_path = faiss_path.with_suffix(".mapping.json")
        if not faiss_path.exists() or not (mapping_path.exists() or embeddings_path.exists()):
            continue
        compatible, diagnostic = index_manifest_compatible(faiss_path)
        if not compatible:
            logger.warning("跳过过期商品索引 %s：%s", faiss_path, diagnostic)
            continue
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            document_ids = [str(value) for value in mapping["document_ids"]]
        else:
            with np.load(embeddings_path, allow_pickle=False) as payload:
                document_ids = payload["document_ids"].astype(str).tolist()
        index = FaissHNSWIndex.load(faiss_path, document_ids)
        vector_index.add_partition(partition_id, index)
        loaded = True
    return vector_index if loaded else FaissProductIndex()


def _build_query_embedder(settings: Settings) -> EmbeddingClient:
    """Build the configured persistent query encoder and preserve GPU mode."""

    if settings.bge_m3_max_seq_length != 512:
        raise RuntimeError("formal BGE-M3 query max_seq_length must be 512")
    python_value = settings.bge_m3_python.strip()
    if python_value:
        python_executable = Path(python_value).expanduser()
        embedder = SubprocessBgeM3EmbeddingClient(
            python_executable,
            model_name=settings.bge_m3_model,
            device=settings.bge_m3_device,
            batch_size=settings.bge_m3_batch_size,
            max_seq_length=settings.bge_m3_max_seq_length,
            local_files_only=settings.models_local_only,
        )
        embedder.ensure_ready()
        warmup = getattr(embedder, "warmup", None)
        if warmup is not None:
            warmup()
        return embedder

    if settings.bge_m3_device == "cpu":
        embedder = BgeM3EmbeddingClient(
            model_name=settings.bge_m3_model,
            device="cpu",
            batch_size=settings.bge_m3_batch_size,
            max_seq_length=settings.bge_m3_max_seq_length,
            local_files_only=settings.models_local_only,
        )
        embedder.ensure_ready()
        warmup = getattr(embedder, "warmup", None)
        if warmup is not None:
            warmup()
        return embedder
    if not settings.bge_m3_device.startswith("cuda"):
        raise RuntimeError(
            "formal BGE-M3 query encoder device must be cpu or CUDA; "
            f"got BGE_M3_DEVICE={settings.bge_m3_device!r}"
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "CUDA query encoding requires torch or a configured BGE_M3_PYTHON worker"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in the Agent environment; configure BGE_M3_PYTHON "
            "to a CUDA-enabled Python environment"
        )
    embedder = BgeM3EmbeddingClient(
        model_name=settings.bge_m3_model,
        device=settings.bge_m3_device,
        batch_size=settings.bge_m3_batch_size,
        max_seq_length=settings.bge_m3_max_seq_length,
        local_files_only=settings.models_local_only,
    )
    embedder.ensure_ready()
    warmup = getattr(embedder, "warmup", None)
    if warmup is not None:
        warmup()
    return embedder


def _build_product_reranker(settings: Settings) -> BgeReranker:
    """Use the configured persistent GPU worker when one is available."""

    python_value = settings.bge_reranker_python.strip()
    if python_value:
        worker = SubprocessCrossEncoderReranker(
            Path(python_value).expanduser(),
            model_name=settings.bge_reranker_model,
            device=settings.bge_reranker_device,
            batch_size=settings.bge_reranker_batch_size,
            max_length=settings.bge_reranker_max_length,
            use_fp16=not settings.bge_reranker_fp32,
            local_files_only=settings.models_local_only,
        )
        worker.ensure_ready()
        warmup = getattr(worker, "warmup", None)
        if warmup is not None:
            warmup()
        return BgeReranker(reranker=worker)
    return BgeReranker(
        model_name=settings.bge_reranker_model,
        local_files_only=settings.models_local_only,
    )


def _validate_catalog_index_contract(model_name: str, *, max_seq_length: int) -> None:
    """Ensure query settings match every canonical item-index manifest."""

    manifest_paths = sorted(
        (PROJECT_ROOT / "output" / "index" / "catalog").glob("*/*/bge-m3-hnsw-ip.manifest.json")
    )
    if not manifest_paths:
        raise RuntimeError("formal catalog Faiss manifests are missing")
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid catalog index manifest: {manifest_path}") from exc
        expected = {
            "encoder_model": model_name,
            "dimension": 1024,
            "max_seq_length": max_seq_length,
            "text_format_version": "item-text-v5-catalog-schema-v2",
            "item_encoding_inference_dtype": "float16",
            "item_encoding_pooling": "cls",
        }
        mismatches = {
            key: (expected_value, manifest.get(key))
            for key, expected_value in expected.items()
            if not _model_identifiers_equal(expected_value, manifest.get(key))
        }
        if manifest.get("index_parameters", {}).get("normalized") is not True:
            mismatches["index_parameters.normalized"] = (
                True,
                manifest.get("index_parameters", {}).get("normalized"),
            )
        if mismatches:
            raise RuntimeError(
                f"catalog query/index contract mismatch in {manifest_path}: {mismatches}"
            )


def _model_identifiers_equal(left: object, right: object) -> bool:
    """Compare local model paths canonically, but keep repo IDs exact."""

    if not isinstance(left, str) or not isinstance(right, str):
        return left == right
    if _looks_like_local_model_path(left) and _looks_like_local_model_path(right):
        return _normalize_local_model_path(left) == _normalize_local_model_path(right)
    return left == right


def _looks_like_local_model_path(value: str) -> bool:
    text = value.strip()
    drive, tail = ntpath.splitdrive(text)
    return bool(
        Path(text).is_absolute()
        or (drive and tail.startswith(("\\", "/")))
        or text.startswith(("/", "\\\\", "//", "./", ".\\", "../", "..\\"))
    )


def _normalize_local_model_path(value: str) -> str:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path.absolute()
    return os.path.normcase(str(resolved))


def _build_category_insight_service(settings: Settings) -> CategoryInsightService:
    taxonomy_path = settings.category_taxonomy
    if not taxonomy_path.is_absolute():
        taxonomy_path = PROJECT_ROOT / taxonomy_path
    encoder = SentenceTransformerTextEncoder(
        model_name=settings.bge_m3_model,
        device="cpu",
        batch_size=settings.bge_m3_batch_size,
        max_seq_length=settings.bge_m3_max_seq_length,
        local_files_only=settings.models_local_only,
    )
    encoder.ensure_ready()
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
