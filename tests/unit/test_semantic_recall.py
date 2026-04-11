from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from numpy.typing import NDArray

from globex_agent.recall import (
    EmbeddingSearchBackend,
    ExactVectorIndex,
    FaissHNSWIndex,
    FusionWeights,
    RecallHit,
    RecallResult,
    RerankedSearchBackend,
    SearchDocument,
    SentenceTransformerTextEncoder,
    SubprocessCrossEncoderReranker,
    WeightedFusionSearchBackend,
    clean_product_body,
    embedding_document_text,
    reranker_document_text,
)


class FakeEncoder:
    encoder_id = "fake-encoder-v1"

    def encode_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray(
            [[1.0, 0.0] if "quiet" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    def encode_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray(
            [[1.0, 0.0] if "headphones" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


class CapturingSentenceModel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts: Sequence[str], **kwargs: object) -> NDArray[np.float32]:
        del kwargs
        self.texts = list(texts)
        return np.ones((len(texts), 2), dtype=np.float32)


class InjectedSentenceTransformerTextEncoder(SentenceTransformerTextEncoder):
    def __init__(self, model_name: str, model: CapturingSentenceModel) -> None:
        super().__init__(model_name)
        self._injected_model = model

    def _load_model(self) -> object:
        return self._injected_model


class StaticBackend:
    def __init__(self, backend_id: str, scores: dict[str, float]) -> None:
        self.backend_id = backend_id
        self._scores = scores

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        del query
        ranked = sorted(self._scores.items(), key=lambda entry: (-entry[1], entry[0]))
        return RecallResult(
            backend_id=self.backend_id,
            hits=tuple(
                RecallHit(document_id=document_id, score=score, rank=rank)
                for rank, (document_id, score) in enumerate(ranked[:top_k], start=1)
            ),
            total_recall=len(ranked),
            truncated=len(ranked) > top_k,
        )


class CapturingBackend(StaticBackend):
    def __init__(self, backend_id: str, scores: dict[str, float]) -> None:
        super().__init__(backend_id, scores)
        self.requested_top_ks: list[int] = []

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        self.requested_top_ks.append(top_k)
        return super().search(query, top_k)


class FailingBackend:
    backend_id = "failing"

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        del query, top_k
        raise TimeoutError("simulated outage")


class FakeReranker:
    reranker_id = "fake-reranker-v1"

    def score(
        self,
        query: str,
        documents: Sequence[SearchDocument],
    ) -> tuple[float, ...]:
        del query
        return tuple(1.0 if document.document_id == "b" else 0.0 for document in documents)


class FailingReranker:
    reranker_id = "failing-reranker"

    def score(
        self,
        query: str,
        documents: Sequence[SearchDocument],
    ) -> tuple[float, ...]:
        del query, documents
        raise TimeoutError("simulated outage")


def test_exact_index_is_deterministic_and_round_trips() -> None:
    index = ExactVectorIndex(
        ["b", "a", "c"],
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    assert [hit.document_id for hit in index.search(np.asarray([1.0, 0.0]), top_k=2)] == [
        "a",
        "b",
    ]

    with NamedTemporaryFile(suffix=".npz", dir=Path.cwd(), delete=False) as target:
        path = Path(target.name)
    try:
        index.save(path)
        loaded = ExactVectorIndex.load(path)
        assert loaded.document_ids == ("b", "a", "c")
        assert loaded.dimension == 2
    finally:
        path.unlink(missing_ok=True)


def test_internal_vector_index_supports_recall_depth_above_50() -> None:
    document_ids = [f"item-{index:03d}" for index in range(120)]
    embeddings = np.asarray(
        [[1.0, float(index) / 1000] for index in range(120)],
        dtype=np.float32,
    )
    index = ExactVectorIndex(document_ids, embeddings)

    assert len(index.search(np.asarray([1.0, 0.0]), top_k=100)) == 100
    subset = index.subset(document_ids[:80])
    assert len(subset.search(np.asarray([1.0, 0.0]), top_k=100)) == 80


def test_faiss_hnsw_ip_round_trips_and_subsets() -> None:
    pytest.importorskip("faiss")
    index = FaissHNSWIndex(
        ["a", "b", "c"],
        np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
        m=8,
        ef_construction=40,
        ef_search=20,
    )
    with NamedTemporaryFile(suffix=".faiss", dir=Path.cwd(), delete=False) as target:
        path = Path(target.name)
    try:
        index.save(path)
        loaded = FaissHNSWIndex.load(
            path,
            index.document_ids,
            m=8,
            ef_construction=40,
            ef_search=20,
        )
        assert loaded.search(np.asarray([1.0, 0.0]), top_k=2)[0].document_id == "a"
        assert loaded.subset(["a", "c"]).document_ids == ("a", "c")
    finally:
        path.unlink(missing_ok=True)


def test_embedding_backend_uses_independent_query_and_item_encoders() -> None:
    documents = [
        SearchDocument("item-b", "travel backpack"),
        SearchDocument("item-a", "quiet headphones"),
    ]
    backend = EmbeddingSearchBackend(documents, FakeEncoder())

    result = backend.search("quiet commute", top_k=1)

    assert result.backend_id.startswith("embedding:fake-encoder-v1")
    assert result.hits[0].document_id == "item-a"
    assert result.total_recall == 2


def test_bge_m3_uses_raw_text_while_e5_keeps_required_prefixes() -> None:
    bge_model = CapturingSentenceModel()
    bge = InjectedSentenceTransformerTextEncoder("BAAI/bge-m3", bge_model)
    bge.encode_queries(["quiet headphones"])
    assert bge_model.texts == ["quiet headphones"]
    bge.encode_documents(["over-ear headphones"])
    assert bge_model.texts == ["over-ear headphones"]

    e5_model = CapturingSentenceModel()
    e5 = InjectedSentenceTransformerTextEncoder(
        "intfloat/multilingual-e5-small",
        e5_model,
    )
    e5.encode_queries(["quiet headphones"])
    assert e5_model.texts == ["query: quiet headphones"]
    e5.encode_documents(["over-ear headphones"])
    assert e5_model.texts == ["passage: over-ear headphones"]


def test_fusion_minmax_weights_and_deduplicates() -> None:
    lexical = StaticBackend("lexical", {"a": 3.0, "b": 2.0, "c": 1.0})
    semantic = StaticBackend("semantic", {"c": 9.0, "b": 5.0, "d": 1.0})
    backend = WeightedFusionSearchBackend(
        lexical,
        semantic,
        weights=FusionWeights(semantic=0.7, lexical=0.3),
    )

    result = backend.search("query", top_k=4)

    assert [hit.document_id for hit in result.hits] == ["c", "b", "a", "d"]
    assert len({hit.document_id for hit in result.hits}) == 4
    assert result.fallback_reason is None


def test_fusion_falls_back_when_semantic_recall_fails() -> None:
    lexical = StaticBackend("lexical", {"a": 3.0, "b": 2.0})
    backend = WeightedFusionSearchBackend(lexical, FailingBackend())

    result = backend.search("query", top_k=1)

    assert [hit.document_id for hit in result.hits] == ["a"]
    assert result.fallback_reason == "semantic recall failed: TimeoutError"


def test_reranker_changes_only_the_coarse_candidate_order() -> None:
    documents = [SearchDocument("a", "A"), SearchDocument("b", "B")]
    coarse = StaticBackend("coarse", {"a": 2.0, "b": 1.0})
    backend = RerankedSearchBackend(coarse, documents, FakeReranker(), candidate_k=2)

    result = backend.search("query", top_k=1)

    assert result.hits[0].document_id == "b"
    assert result.total_recall == 2
    assert result.fallback_reason is None


def test_product_mainline_requests_vector_top_100_before_reranker_top_10() -> None:
    documents = [
        SearchDocument(document_id, document_id)
        for document_id in (f"item-{index:03d}" for index in range(120))
    ]
    vector_recall = CapturingBackend(
        "query-item-vector",
        {
            document.document_id: float(120 - index)
            for index, document in enumerate(documents)
        },
    )
    backend = RerankedSearchBackend(
        vector_recall,
        documents,
        FakeReranker(),
        candidate_k=100,
    )

    result = backend.search("query", top_k=10)

    assert vector_recall.requested_top_ks == [100]
    assert len(result.hits) == 10


def test_reranker_failure_returns_coarse_order() -> None:
    documents = [SearchDocument("a", "A"), SearchDocument("b", "B")]
    coarse = StaticBackend("coarse", {"a": 2.0, "b": 1.0})
    backend = RerankedSearchBackend(coarse, documents, FailingReranker(), candidate_k=2)

    result = backend.search("query", top_k=1)

    assert result.hits[0].document_id == "a"
    assert result.fallback_reason == "reranker failed: TimeoutError"


def test_subprocess_reranker_uses_persistent_jsonl_worker() -> None:
    worker = Path(__file__).parents[1] / "fixtures" / "fake_reranker_worker.py"
    reranker = SubprocessCrossEncoderReranker(
        Path(sys.executable),
        worker_path=worker,
        use_fp16=False,
    )
    documents = [SearchDocument("a", "A"), SearchDocument("b", "B")]
    try:
        assert reranker.score("query", documents) == (0.0, 1.0)
        assert reranker.score("query again", documents) == (0.0, 1.0)
    finally:
        reranker.close()


def test_vector_index_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        ExactVectorIndex(["a"], np.asarray([[0.0, 0.0]], dtype=np.float32))


def test_item_text_removes_duplicate_title_and_literal_none() -> None:
    title = "Acme Quiet Headphones"
    dirty_body = "Acme Quiet Headphones None Acme wireless over-ear None"

    assert clean_product_body(title, dirty_body) == "Acme wireless over-ear"
    assert embedding_document_text(title, dirty_body) == (
        "Title: Acme Quiet Headphones\nDetails: Acme wireless over-ear"
    )
    assert reranker_document_text(title, dirty_body) == (
        "Title: Acme Quiet Headphones\nDetails: Acme wireless over-ear"
    )
