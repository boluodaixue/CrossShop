"""Exact and Faiss HNSW inner-product indexes for Query/Item recall."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from globex_agent.infrastructure.recall.base import RecallHit, RecallResult, SearchDocument
from globex_agent.infrastructure.recall.embedding import TextEncoder, embedding_document_text


class VectorIndex(Protocol):
    """Minimal vector-index contract used by the semantic SearchBackend."""

    @property
    def index_id(self) -> str: ...

    @property
    def document_ids(self) -> tuple[str, ...]: ...

    @property
    def dimension(self) -> int: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    def search(
        self,
        query_embedding: NDArray[np.floating],
        *,
        top_k: int,
    ) -> tuple[RecallHit, ...]: ...

    def save(self, path: Path) -> None: ...

    def subset(self, document_ids: Sequence[str]) -> VectorIndex: ...


class ExactVectorIndex:
    """Immutable normalized-vector baseline for correctness comparisons.

    Inner product on L2-normalized vectors is cosine similarity. This exact
    implementation stays available beside the course-mainline Faiss HNSW
    index so approximate-search behavior can be checked independently.
    """

    def __init__(
        self,
        document_ids: Sequence[str],
        embeddings: NDArray[np.floating],
    ) -> None:
        ids = tuple(str(document_id) for document_id in document_ids)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if matrix.shape[0] != len(ids):
            raise ValueError("document_ids and embeddings must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("document_id must be unique")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings must contain only finite values")

        self._document_ids = ids
        self._embeddings = _normalize_rows(matrix)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return self._document_ids

    @property
    def index_id(self) -> str:
        return "exact-cosine-inner-product-v1"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"metric": "inner_product", "normalized": True}

    @property
    def dimension(self) -> int:
        return self._embeddings.shape[1] if self._embeddings.ndim == 2 else 0

    def search(
        self,
        query_embedding: NDArray[np.floating],
        *,
        top_k: int,
    ) -> tuple[RecallHit, ...]:
        top_k = max(1, min(top_k, len(self._document_ids)))
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"query dimension {vector.shape[0]} does not match index {self.dimension}"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("query embedding must not be a zero vector")
        scores = self._embeddings @ (vector / norm)
        ranked = sorted(
            zip(scores.tolist(), self._document_ids, strict=True),
            key=lambda entry: (-entry[0], entry[1]),
        )
        return tuple(
            RecallHit(document_id=document_id, score=round(float(score), 8), rank=rank)
            for rank, (score, document_id) in enumerate(ranked[:top_k], start=1)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            document_ids=np.asarray(self._document_ids, dtype=np.str_),
            embeddings=self._embeddings,
        )

    @classmethod
    def load(cls, path: Path) -> ExactVectorIndex:
        with np.load(path, allow_pickle=False) as payload:
            document_ids = payload["document_ids"].astype(str).tolist()
            embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        return cls(document_ids, embeddings)

    def subset(self, document_ids: Sequence[str]) -> ExactVectorIndex:
        """Return a view-copy containing only the requested indexed documents."""

        positions = {document_id: index for index, document_id in enumerate(self._document_ids)}
        requested = tuple(sorted(str(document_id) for document_id in document_ids))
        if len(set(requested)) != len(requested):
            raise ValueError("subset document_id must be unique")
        missing = [document_id for document_id in requested if document_id not in positions]
        if missing:
            raise ValueError(f"subset document IDs are absent from the index: {missing[:3]}")
        matrix = np.asarray(
            [self._embeddings[positions[document_id]] for document_id in requested],
            dtype=np.float32,
        )
        if not requested:
            matrix = np.empty((0, self.dimension), dtype=np.float32)
        return ExactVectorIndex(requested, matrix)


class FaissHNSWIndex:
    """Faiss HNSW index over normalized vectors using inner product."""

    def __init__(
        self,
        document_ids: Sequence[str],
        embeddings: NDArray[np.floating],
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        if m < 2:
            raise ValueError("HNSW m must be at least 2")
        if ef_construction < 1 or ef_search < 1:
            raise ValueError("HNSW ef values must be positive")
        ids = tuple(str(document_id) for document_id in document_ids)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if matrix.shape[0] != len(ids):
            raise ValueError("document_ids and embeddings must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("document_id must be unique")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings must contain only finite values")
        if matrix.shape[1] < 1:
            raise ValueError("embedding dimension must be positive")

        faiss = _load_faiss()
        normalized = np.ascontiguousarray(_normalize_rows(matrix))
        index = faiss.IndexHNSWFlat(
            normalized.shape[1],
            m,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        if normalized.shape[0]:
            index.add(normalized)
        self._document_ids = ids
        self._index = index
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

    @property
    def index_id(self) -> str:
        return "faiss-hnsw-inner-product-v1"

    @property
    def document_ids(self) -> tuple[str, ...]:
        return self._document_ids

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "metric": "inner_product",
            "normalized": True,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
        }

    def search(
        self,
        query_embedding: NDArray[np.floating],
        *,
        top_k: int,
    ) -> tuple[RecallHit, ...]:
        if not self._document_ids:
            return ()
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension {vector.shape[1]} does not match index {self.dimension}"
            )
        normalized = np.ascontiguousarray(_normalize_rows(vector))
        limit = max(1, min(top_k, len(self._document_ids)))
        scores, positions = self._index.search(normalized, limit)
        rows = [
            (float(score), self._document_ids[int(position)])
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]
        rows.sort(key=lambda entry: (-entry[0], entry[1]))
        return tuple(
            RecallHit(document_id=document_id, score=round(score, 8), rank=rank)
            for rank, (score, document_id) in enumerate(rows, start=1)
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _load_faiss().write_index(self._index, str(path))

    @classmethod
    def load(
        cls,
        path: Path,
        document_ids: Sequence[str],
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> FaissHNSWIndex:
        faiss = _load_faiss()
        loaded = faiss.read_index(str(path))
        ids = tuple(str(document_id) for document_id in document_ids)
        if int(loaded.ntotal) != len(ids):
            raise ValueError("Faiss index count does not match document IDs")
        if int(loaded.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
            raise ValueError("Faiss index must use inner product")
        loaded.hnsw.efSearch = ef_search
        instance = cls.__new__(cls)
        instance._document_ids = ids
        instance._index = loaded
        instance._m = m
        instance._ef_construction = ef_construction
        instance._ef_search = ef_search
        return instance

    def subset(self, document_ids: Sequence[str]) -> FaissHNSWIndex:
        positions = {document_id: index for index, document_id in enumerate(self._document_ids)}
        requested = tuple(sorted(str(document_id) for document_id in document_ids))
        if len(set(requested)) != len(requested):
            raise ValueError("subset document_id must be unique")
        missing = [document_id for document_id in requested if document_id not in positions]
        if missing:
            raise ValueError(f"subset document IDs are absent from the index: {missing[:3]}")
        matrix = np.asarray(
            [self._index.reconstruct(positions[document_id]) for document_id in requested],
            dtype=np.float32,
        )
        if not requested:
            matrix = np.empty((0, self.dimension), dtype=np.float32)
        return FaissHNSWIndex(
            requested,
            matrix,
            m=self._m,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
        )


class EmbeddingSearchBackend:
    """SearchBackend backed by independent Query and Item encoders."""

    def __init__(
        self,
        documents: Sequence[SearchDocument],
        encoder: TextEncoder,
        *,
        index: VectorIndex | None = None,
    ) -> None:
        ordered = tuple(sorted(documents, key=lambda document: document.document_id))
        if len({document.document_id for document in ordered}) != len(ordered):
            raise ValueError("document_id must be unique")
        self._documents = ordered
        self._encoder = encoder

        if index is None:
            texts = [
                embedding_document_text(document.title, document.body)
                for document in ordered
            ]
            embeddings = encoder.encode_documents(texts)
            index = ExactVectorIndex(
                [document.document_id for document in ordered],
                embeddings,
            )
        expected_ids = tuple(document.document_id for document in ordered)
        if index.document_ids != expected_ids:
            raise ValueError("index document IDs do not match the sorted document collection")
        self._index = index

    @property
    def backend_id(self) -> str:
        return f"embedding:{self._encoder.encoder_id}:{self._index.index_id}"

    @property
    def dimension(self) -> int:
        return self._index.dimension

    @property
    def index_id(self) -> str:
        return self._index.index_id

    @property
    def index_parameters(self) -> dict[str, Any]:
        return self._index.parameters

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        top_k = max(1, min(top_k, len(self._documents)))
        if not self._documents:
            return RecallResult(
                backend_id=self.backend_id,
                hits=(),
                total_recall=0,
                truncated=False,
            )
        query_matrix = self._encoder.encode_queries([query])
        if query_matrix.shape[0] != 1:
            raise ValueError("query encoder must return exactly one vector")
        hits = self._index.search(query_matrix[0], top_k=top_k)
        return RecallResult(
            backend_id=self.backend_id,
            hits=hits,
            total_recall=len(self._documents),
            truncated=len(self._documents) > top_k,
        )

    def save_index(self, path: Path) -> None:
        self._index.save(path)

    def subset(self, documents: Sequence[SearchDocument]) -> EmbeddingSearchBackend:
        """Reuse the global Item embeddings for a closed judged candidate pool."""

        document_ids = [document.document_id for document in documents]
        return EmbeddingSearchBackend(
            documents,
            self._encoder,
            index=self._index.subset(document_ids),
        )

    @classmethod
    def from_index(
        cls,
        documents: Sequence[SearchDocument],
        encoder: TextEncoder,
        path: Path,
    ) -> EmbeddingSearchBackend:
        return cls(documents, encoder, index=ExactVectorIndex.load(path))

    @classmethod
    def with_faiss_hnsw(
        cls,
        documents: Sequence[SearchDocument],
        encoder: TextEncoder,
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> EmbeddingSearchBackend:
        ordered = tuple(sorted(documents, key=lambda document: document.document_id))
        texts = [
            embedding_document_text(document.title, document.body)
            for document in ordered
        ]
        embeddings = encoder.encode_documents(texts)
        index = FaissHNSWIndex(
            [document.document_id for document in ordered],
            embeddings,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        return cls(ordered, encoder, index=index)

    @classmethod
    def from_faiss_hnsw(
        cls,
        documents: Sequence[SearchDocument],
        encoder: TextEncoder,
        path: Path,
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> EmbeddingSearchBackend:
        ordered = tuple(sorted(documents, key=lambda document: document.document_id))
        index = FaissHNSWIndex.load(
            path,
            [document.document_id for document in ordered],
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        return cls(ordered, encoder, index=index)


def _normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    if matrix.shape[0] == 0:
        return matrix.copy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("document embeddings must not contain zero vectors")
    return np.asarray(matrix / norms, dtype=np.float32)


def _load_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency smoke tests cover this
        raise RuntimeError("faiss-cpu is required for the course ANN backend") from exc
    return faiss
