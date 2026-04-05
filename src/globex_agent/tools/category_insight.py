"""LangChain adapter and default local dependencies for CategoryInsight."""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from globex_agent.category_insight import (
    CategoryInsightOutput,
    CategoryInsightService,
    CategoryTaxonomy,
)
from globex_agent.recall.category_kb import (
    DEFAULT_CATEGORY_INDEX,
    OpenSearchCategoryKnowledgeBase,
    OpenSearchHttpClient,
)
from globex_agent.recall.embedding import SentenceTransformerTextEncoder
from globex_agent.recall.reranker import (
    CrossEncoderReranker,
    SubprocessCrossEncoderReranker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_category_insight(
    category: str,
    *,
    depth: Literal["quick", "deep"] = "quick",
    service: CategoryInsightService | None = None,
) -> CategoryInsightOutput:
    """Return the compressed course output, keeping diagnostics out of Agent context."""

    active_service = service or get_default_category_insight_service()
    return active_service.insight(category, depth=depth).output


@tool("category_insight")
async def category_insight_tool(
    category: str,
    depth: Literal["quick", "deep"] = "quick",
) -> str:
    """Get category-level product forms, popular examples, attributes, and price tiers.

    Args:
        category: A normalized category or a category phrase with constraints.
        depth: quick returns popular examples and price tiers; deep also returns attributes.
    """

    output = await asyncio.to_thread(get_category_insight, category, depth=depth)
    return output.model_dump_json()


@lru_cache(maxsize=1)
def get_default_category_insight_service() -> CategoryInsightService:
    taxonomy_path = Path(
        os.getenv(
            "GLOBEX_CATEGORY_TAXONOMY",
            str(
                PROJECT_ROOT
                / "data"
                / "category_insight"
                / "category_taxonomy.json"
            ),
        )
    )
    if not taxonomy_path.is_absolute():
        taxonomy_path = PROJECT_ROOT / taxonomy_path
    endpoint = os.getenv("GLOBEX_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200")
    index_name = os.getenv("GLOBEX_CATEGORY_INDEX", DEFAULT_CATEGORY_INDEX)
    local_files_only = _env_flag("GLOBEX_MODELS_LOCAL_ONLY", default=True)
    encoder = SentenceTransformerTextEncoder(local_files_only=local_files_only)
    knowledge_base = OpenSearchCategoryKnowledgeBase(
        OpenSearchHttpClient(endpoint, timeout=30),
        index_name=index_name,
    )
    reranker = None
    if _env_flag("GLOBEX_CATEGORY_RERANKER_ENABLED", default=False):
        reranker_python = os.getenv("GLOBEX_CATEGORY_RERANKER_PYTHON", "").strip()
        if reranker_python:
            reranker = SubprocessCrossEncoderReranker(
                Path(reranker_python),
                device=os.getenv("GLOBEX_CATEGORY_RERANKER_DEVICE", "cuda:0"),
                batch_size=int(
                    os.getenv("GLOBEX_CATEGORY_RERANKER_BATCH_SIZE", "1")
                ),
                use_fp16=not _env_flag(
                    "GLOBEX_CATEGORY_RERANKER_FP32",
                    default=False,
                ),
                local_files_only=local_files_only,
            )
        else:
            reranker = CrossEncoderReranker(
                device="cpu",
                batch_size=1,
                local_files_only=local_files_only,
            )
    return CategoryInsightService(
        CategoryTaxonomy.from_json(taxonomy_path),
        encoder,
        knowledge_base,
        reranker,
        reranker_document_mode=os.getenv(
            "GLOBEX_CATEGORY_RERANKER_DOCUMENT_MODE", "contextual"
        ),
    )


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}
