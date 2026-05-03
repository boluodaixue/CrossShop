"""Query/Item dual-tower encoding without a User tower."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MAX_SEQ_LENGTH = 512
ITEM_TEXT_FORMAT_VERSION = "item-text-v2"


class TextEncoder(Protocol):
    """Independently encode query text and item text into the same vector space."""

    @property
    def encoder_id(self) -> str:
        """Return a stable identifier for reports and persisted indexes."""

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode query-side text as a two-dimensional float32 matrix."""

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode item-side text as a two-dimensional float32 matrix."""


class SentenceTransformerTextEncoder:
    """Lazy Sentence Transformers adapter for Query/Item dual-tower models.

    BGE-M3 consumes the plain Query/Item text used by the project.  The adapter
    retains model-specific ``query:``/``passage:`` prefixes when an mE5 model is
    selected explicitly, so callers always pass plain text.  CPU remains the
    default because the laptop GPU is reserved for the separate reranker.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        max_seq_length: int = DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
        local_files_only: bool = False,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_seq_length < 8:
            raise ValueError("max_seq_length must be at least 8")

        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._local_files_only = local_files_only
        default_query_prefix, default_document_prefix = _model_prefixes(model_name)
        self._query_prefix = (
            default_query_prefix if query_prefix is None else query_prefix
        )
        self._document_prefix = (
            default_document_prefix if document_prefix is None else document_prefix
        )
        self._model: object | None = None

    @property
    def encoder_id(self) -> str:
        return self._model_name

    @property
    def max_seq_length(self) -> int:
        """Return the configured inference window for index manifests."""

        return self._max_seq_length

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode([f"{self._query_prefix}{text.strip()}" for text in texts])

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return self._encode([f"{self._document_prefix}{text.strip()}" for text in texts])

    def _encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._load_model()
        embeddings = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError("encoder returned an invalid embedding matrix")
        return matrix

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - dependency smoke test covers this
                raise RuntimeError(
                    "sentence-transformers is required for semantic recall"
                ) from exc

            model = SentenceTransformer(
                self._model_name,
                device=self._device,
                local_files_only=self._local_files_only,
            )
            model.max_seq_length = self._max_seq_length
            self._model = model
        return self._model


def clean_product_body(title: str, body: str) -> str:
    """Remove joined-dataset artifacts without inventing missing attributes.

    The ESCI mirror's ``product_text`` already starts with ``product_title``
    and serializes missing source fields as the literal token ``None``.  The
    title is supplied separately by ``SearchDocument``, so retaining either
    artifact wastes the encoder window and changes BM25 field weighting.
    """

    clean_title = " ".join(title.split())
    clean_body = " ".join(body.split())
    if clean_title and clean_body.startswith(clean_title):
        clean_body = clean_body[len(clean_title) :].lstrip(" :-|\t\n")
    return " ".join(token for token in clean_body.split() if token != "None")


def embedding_document_text(title: str, body: str) -> str:
    """Build a field-marked Item-tower input with the title first."""

    clean_title = " ".join(title.split())
    clean_body = clean_product_body(clean_title, body)
    parts = [f"Title: {clean_title}"] if clean_title else []
    if clean_body:
        parts.append(f"Details: {clean_body}")
    return "\n".join(parts)


def reranker_document_text(title: str, body: str) -> str:
    """Build the inference-time cross-encoder text in truncation priority order.

    ``SearchDocument`` currently has only title and body fields.  Keeping this
    builder separate allows future structured brand/category/attribute budgets
    without coupling the persisted Item-tower index to the Reranker format.
    """

    return embedding_document_text(title, body)


def document_text(title: str, body: str) -> str:
    """Backward-compatible alias for the Item-tower representation."""

    return embedding_document_text(title, body)


def _model_prefixes(model_name: str) -> tuple[str, str]:
    """Return model-specific asymmetric prefixes.

    Multilingual E5 requires ``query:``/``passage:``.  BGE-M3's official
    Sentence Transformers checkpoint accepts the raw query and document text.
    """

    normalized = model_name.casefold()
    if "e5" in normalized:
        return "query: ", "passage: "
    return "", ""
