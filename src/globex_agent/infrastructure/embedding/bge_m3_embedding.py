"""Async adapter around the existing BGE-M3 SentenceTransformer encoder."""

from __future__ import annotations

import asyncio

from globex_agent.domain.catalog.ports.retrieval_ports import EmbeddingClient
from globex_agent.infrastructure.recall.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerTextEncoder,
    TextEncoder,
)


class BgeM3EmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        local_files_only: bool = False,
        encoder: TextEncoder | None = None,
    ) -> None:
        self._encoder = encoder or SentenceTransformerTextEncoder(
            model_name,
            device=device,
            batch_size=batch_size,
            local_files_only=local_files_only,
        )

    @property
    def encoder_id(self) -> str:
        return self._encoder.encoder_id

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = await asyncio.to_thread(self._encoder.encode_queries, texts)
        return [row.astype(float).tolist() for row in matrix]
