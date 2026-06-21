# -*- coding: utf-8 -*-
"""检索基础设施端口：EmbeddingClient / ProductVectorIndex / Reranker

Domain 不关心实现：Infrastructure 提供 embedding、Qdrant/OpenSearch 商品索引和 reranker。
检索端口同时接收 query 文本与 embedding：Qdrant 基线忽略文本继续纯向量召回，
OpenSearch 在一次请求内执行 ANN + BM25 + RRF。任一环节不可用时仍由 UseCase 降级。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain.catalog.product import Product


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class VectorHit:
    product_id: str
    score: float


class ProductVectorIndex(ABC):
    @abstractmethod
    async def ensure_ready(self, vector_dim: int) -> None:
        """确保 collection 存在（幂等）。"""

    @abstractmethod
    async def upsert_products(self, products: list[Product], embeddings: list[list[float]]) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        ...


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """返回与 documents 等长的精排分数；失败抛异常，由调用方降级。"""
