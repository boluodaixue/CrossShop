"""Environment-backed application settings for the LangGraph/DDD architecture."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _local_model_dir(name: str, fallback: str) -> str:
    path = Path(name)
    return str(path) if path.is_dir() else fallback


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_fallback_model: str
    llm_judge: str
    port: int
    log_level: str
    data_dir: Path
    database_url: str
    redis_url: str
    queue_enabled: bool
    queue_wait_seconds: float
    worker_concurrency: int
    semantic_cache_enabled: bool
    semantic_cache_threshold: float
    context_size: int
    tool_result_limit: int
    reply_token_budget: int
    tool_failure_threshold: int
    tool_circuit_reset_seconds: float
    cors_origins: list[str]
    llm_max_concurrency: int
    llm_min_interval_seconds: float
    llm_max_retries: int
    tavily_api_key: str
    opensearch_endpoint: str
    category_index: str
    category_taxonomy: Path
    models_local_only: bool
    category_reranker_enabled: bool
    category_reranker_python: str
    category_reranker_device: str
    category_reranker_batch_size: int
    category_reranker_fp32: bool
    category_reranker_document_mode: str
    bge_m3_model: str
    bge_reranker_model: str
    bge_reranker_python: str
    faiss_index_path: Path


def load_settings() -> Settings:
    """Load all settings from the environment with local-development defaults."""

    data_dir = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    llm_base_url = os.getenv(
        "LLM_BASE_URL",
        os.getenv(
            "OPENAI_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )
    llm_api_key = os.getenv(
        "LLM_API_KEY",
        os.getenv("OPENAI_API_KEY", ""),
    )
    llm_model = os.getenv("LLM_MODEL", os.getenv("LLM_MAIN", "qwen3-max"))
    return Settings(
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL", "qwen-plus"),
        llm_judge=os.getenv("LLM_JUDGE", ""),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
        data_dir=data_dir,
        database_url=os.getenv("DATABASE_URL", "file"),
        redis_url=os.getenv("REDIS_URL", ""),
        queue_enabled=_env_flag("QUEUE_ENABLED", default=False),
        queue_wait_seconds=float(os.getenv("QUEUE_WAIT_SECONDS", "300")),
        worker_concurrency=int(os.getenv("WORKER_CONCURRENCY", "2")),
        semantic_cache_enabled=_env_flag("SEMANTIC_CACHE_ENABLED", default=True),
        semantic_cache_threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95")),
        context_size=int(os.getenv("CONTEXT_SIZE", "128000")),
        tool_result_limit=int(os.getenv("TOOL_RESULT_LIMIT", "20000")),
        reply_token_budget=int(os.getenv("REPLY_TOKEN_BUDGET", "0")),
        tool_failure_threshold=int(os.getenv("TOOL_FAILURE_THRESHOLD", "3")),
        tool_circuit_reset_seconds=float(
            os.getenv("TOOL_CIRCUIT_RESET_SECONDS", "60")
        ),
        cors_origins=[
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        ],
        llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "2")),
        llm_min_interval_seconds=float(
            os.getenv("LLM_MIN_INTERVAL_SECONDS", "1.0")
        ),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        opensearch_endpoint=os.getenv(
            "GLOBEX_OPENSEARCH_ENDPOINT",
            "http://127.0.0.1:9200",
        ),
        category_index=os.getenv(
            "GLOBEX_CATEGORY_INDEX",
            "globex_category_kb",
        ),
        category_taxonomy=Path(
            os.getenv(
                "GLOBEX_CATEGORY_TAXONOMY",
                str(data_dir / "category_insight" / "category_taxonomy.json"),
            )
        ),
        models_local_only=_env_flag("GLOBEX_MODELS_LOCAL_ONLY", default=True),
        category_reranker_enabled=_env_flag(
            "GLOBEX_CATEGORY_RERANKER_ENABLED",
            default=False,
        ),
        category_reranker_python=os.getenv(
            "GLOBEX_CATEGORY_RERANKER_PYTHON",
            "",
        ),
        category_reranker_device=os.getenv(
            "GLOBEX_CATEGORY_RERANKER_DEVICE",
            "cuda:0",
        ),
        category_reranker_batch_size=int(
            os.getenv("GLOBEX_CATEGORY_RERANKER_BATCH_SIZE", "1")
        ),
        category_reranker_fp32=_env_flag(
            "GLOBEX_CATEGORY_RERANKER_FP32",
            default=False,
        ),
        category_reranker_document_mode=os.getenv(
            "GLOBEX_CATEGORY_RERANKER_DOCUMENT_MODE",
            "contextual",
        ),
        bge_m3_model=os.getenv(
            "BGE_M3_MODEL",
            _local_model_dir("D:/models/bge-m3", "BAAI/bge-m3"),
        ),
        bge_reranker_model=os.getenv(
            "BGE_RERANKER_MODEL",
            _local_model_dir(
                "D:/models/bge-reranker-v2-m3",
                "BAAI/bge-reranker-v2-m3",
            ),
        ),
        bge_reranker_python=os.getenv("BGE_RERANKER_PYTHON", ""),
        faiss_index_path=Path(
            os.getenv(
                "FAISS_INDEX_PATH",
                str(data_dir / "indexes" / "globex_items.faiss"),
            )
        ),
    )
