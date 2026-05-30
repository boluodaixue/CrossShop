"""CategoryInsight knowledge-card contracts and admission checks."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from globex_agent.category_insight.admission import AdmissionResult, admit_card
from globex_agent.category_insight.models import (
    AttributeDist,
    Bestseller,
    CategoryCard,
    CategoryCardType,
    CategoryEvidenceRef,
    CategoryInsightDepth,
    CategoryInsightOutput,
    PriceTier,
)

_LAZY_EXPORTS = {
    "CategoryRerankResult": (
        "globex_agent.category_insight.reranking",
        "CategoryRerankResult",
    ),
    "RerankerDocumentMode": (
        "globex_agent.category_insight.reranking",
        "RerankerDocumentMode",
    ),
    "rerank_category_hits": (
        "globex_agent.category_insight.reranking",
        "rerank_category_hits",
    ),
    "CategoryInsightDiagnostics": (
        "globex_agent.category_insight.service",
        "CategoryInsightDiagnostics",
    ),
    "CategoryInsightRun": (
        "globex_agent.category_insight.service",
        "CategoryInsightRun",
    ),
    "CategoryInsightService": (
        "globex_agent.category_insight.service",
        "CategoryInsightService",
    ),
    "CategoryTaxonomy": (
        "globex_agent.category_insight.service",
        "CategoryTaxonomy",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})

__all__ = [
    "AdmissionResult",
    "AttributeDist",
    "Bestseller",
    "CategoryCard",
    "CategoryCardType",
    "CategoryEvidenceRef",
    "CategoryInsightDepth",
    "CategoryInsightDiagnostics",
    "CategoryInsightOutput",
    "CategoryInsightRun",
    "CategoryInsightService",
    "CategoryRerankResult",
    "RerankerDocumentMode",
    "CategoryTaxonomy",
    "PriceTier",
    "admit_card",
    "rerank_category_hits",
]
