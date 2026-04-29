"""检索基础设施端口：EmbeddingClient / ItemVectorIndex / Reranker

Domain 不关心实现：Infrastructure 提供本地 embedding、向量索引、精排实现。
UseCase 通过这三个端口完成"embed → 向量召回 → rerank"二阶段召回，
任一环节不可用时由 UseCase 负责降级（关键词召回 / 跳过精排）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from globex_agent.domain.catalog.models import StandardItem


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class VectorHit:
    item_id: str
    score: float


class ItemVectorIndex(ABC):
    @abstractmethod
    async def ensure_ready(self, vector_dim: int) -> None:
        """确保 collection 存在（幂等）。"""

    @abstractmethod
    async def upsert_items(
        self, items: list[StandardItem], embeddings: list[list[float]]
    ) -> None:
        ...

    @abstractmethod
    async def search(self, embedding: list[float], top_n: int) -> list[VectorHit]:
        ...


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """返回与 documents 等长的精排分数；失败抛异常，由调用方降级。"""
