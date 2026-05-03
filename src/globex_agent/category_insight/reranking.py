"""Course-specific Top-30 to Top-K reranking for knowledge cards."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from globex_agent.infrastructure.recall.category_kb import CategoryCardHit, category_retrieval_text
from globex_agent.infrastructure.recall.reranker import RawTextPairReranker

RERANK_BYPASS_TOP_SCORE = 0.92
RerankerDocumentMode = Literal["summary_only", "contextual"]


@dataclass(frozen=True, slots=True)
class CategoryRerankResult:
    hits: tuple[CategoryCardHit, ...]
    reranked: bool
    bypass_reason: str | None = None


def rerank_category_hits(
    query: str,
    coarse_hits: tuple[CategoryCardHit, ...],
    reranker: RawTextPairReranker,
    *,
    final_k: int,
    bypass_top_score: float | None = RERANK_BYPASS_TOP_SCORE,
    document_mode: RerankerDocumentMode = "summary_only",
) -> CategoryRerankResult:
    if final_k < 1:
        raise ValueError("final_k must be positive")
    if document_mode not in ("summary_only", "contextual"):
        raise ValueError("document_mode must be summary_only or contextual")
    if not coarse_hits:
        return CategoryRerankResult((), False, "empty coarse recall")
    if len(coarse_hits) <= final_k:
        return CategoryRerankResult(
            _renumber(coarse_hits[:final_k]),
            False,
            "candidate count is not greater than final_k",
        )
    if (
        bypass_top_score is not None
        and coarse_hits[0].score >= bypass_top_score
    ):
        return CategoryRerankResult(
            _renumber(coarse_hits[:final_k]),
            False,
            f"coarse top score >= {bypass_top_score:g}",
        )
    documents = (
        [hit.card.summary for hit in coarse_hits]
        if document_mode == "summary_only"
        else [_contextual_text(query, hit) for hit in coarse_hits]
    )
    try:
        scores = reranker.score_texts(query, documents)
    except Exception as exc:  # noqa: BLE001 - course requires coarse fallback
        return CategoryRerankResult(
            _renumber(coarse_hits[:final_k]),
            False,
            f"reranker failed: {type(exc).__name__}",
        )
    if len(scores) != len(coarse_hits):
        return CategoryRerankResult(
            _renumber(coarse_hits[:final_k]),
            False,
            "reranker returned the wrong number of scores",
        )
    ranked = sorted(
        zip(scores, coarse_hits, strict=True),
        key=lambda pair: (-pair[0], pair[1].card.card_id),
    )
    hits = tuple(
        CategoryCardHit(
            card=hit.card,
            score=float(score),
            rank=rank,
            retrieval_text=hit.retrieval_text,
            retrieval_text_en=hit.retrieval_text_en,
            retrieval_text_zh=hit.retrieval_text_zh,
        )
        for rank, (score, hit) in enumerate(ranked[:final_k], start=1)
    )
    return CategoryRerankResult(hits, True)


def _renumber(hits: tuple[CategoryCardHit, ...]) -> tuple[CategoryCardHit, ...]:
    return tuple(
        CategoryCardHit(
            card=hit.card,
            score=hit.score,
            rank=rank,
            retrieval_text=hit.retrieval_text,
            retrieval_text_en=hit.retrieval_text_en,
            retrieval_text_zh=hit.retrieval_text_zh,
        )
        for rank, hit in enumerate(hits, start=1)
    )


def _contextual_text(query: str, hit: CategoryCardHit) -> str:
    if re.search(r"[\u3400-\u9fff]", query) and hit.retrieval_text_zh:
        return hit.retrieval_text_zh
    if hit.retrieval_text_en:
        return hit.retrieval_text_en
    return hit.retrieval_text or category_retrieval_text(hit.card)
