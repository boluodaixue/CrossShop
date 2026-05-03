"""Per-run dependencies available to LangChain tool adapters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import (
    ItemPickerOutput,
    ItemSearchOutput,
    PriceCompareOutput,
    SearchRequest,
    ShippingCalcOutput,
    ShoppingSummaryOutput,
    UserProfile,
)
from globex_agent.infrastructure.recall import (
    ListingLanguage,
    PairReranker,
    PartitionedSearchBackendRouter,
    SearchBackend,
)


@dataclass
class AgentArtifacts:
    """Structured outputs collected while one AgentLoop is running."""

    searches: list[ItemSearchOutput] = field(default_factory=list)
    price_comparison: PriceCompareOutput | None = None
    shipping: ShippingCalcOutput | None = None
    selection: ItemPickerOutput | None = None
    summary: ShoppingSummaryOutput | None = None
    fallback_text: str | None = None


@dataclass
class AgentRuntime:
    """Local services hidden from the model-facing tool schemas."""

    catalog: LocalCatalog
    request: SearchRequest
    profile: UserProfile | None = None
    planner_model: BaseChatModel | None = None
    product_search_backend: SearchBackend | None = None
    product_search_router: PartitionedSearchBackendRouter | None = None
    product_reranker: PairReranker | None = None
    listing_languages: Mapping[str, ListingLanguage] | None = None
    artifacts: AgentArtifacts = field(default_factory=AgentArtifacts)


_runtime_var: ContextVar[AgentRuntime | None] = ContextVar(
    "globex_agent_runtime",
    default=None,
)


def get_agent_runtime() -> AgentRuntime:
    runtime = _runtime_var.get()
    if runtime is None:
        raise RuntimeError("Agent tool called outside agent_runtime_scope")
    return runtime


@contextmanager
def agent_runtime_scope(runtime: AgentRuntime) -> Iterator[None]:
    token = _runtime_var.set(runtime)
    try:
        yield
    finally:
        _runtime_var.reset(token)
