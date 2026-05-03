"""Async adapter around the existing BGE reranker."""

from __future__ import annotations

import asyncio

from globex_agent.domain.catalog.ports.retrieval_ports import Reranker
from globex_agent.infrastructure.recall.reranker import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RawTextPairReranker,
)


class BgeReranker(Reranker):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        local_files_only: bool = False,
        reranker: RawTextPairReranker | None = None,
    ) -> None:
        self._reranker = reranker or CrossEncoderReranker(
            model_name,
            device=device,
            batch_size=batch_size,
            local_files_only=local_files_only,
        )

    @property
    def reranker_id(self) -> str:
        return self._reranker.reranker_id

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = await asyncio.to_thread(
            self._reranker.score_texts,
            query,
            documents,
        )
        return [float(score) for score in scores]
