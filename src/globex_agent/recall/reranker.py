"""Replaceable cross-encoder reranking over vector-recall candidates."""

from __future__ import annotations

import atexit
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from globex_agent.recall.base import (
    RecallHit,
    RecallResult,
    SearchBackend,
    SearchDocument,
)
from globex_agent.recall.embedding import reranker_document_text

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MAX_LENGTH = 256


class PairReranker(Protocol):
    """Score query-item pairs jointly after coarse recall."""

    @property
    def reranker_id(self) -> str:
        """Return a stable model identifier for evaluation reports."""

    def score(self, query: str, documents: Sequence[SearchDocument]) -> tuple[float, ...]:
        """Return one score for each document, preserving input order."""


class RawTextPairReranker(Protocol):
    """Cross-encoder boundary for already formatted knowledge-card summaries."""

    @property
    def reranker_id(self) -> str: ...

    def score_texts(self, query: str, texts: Sequence[str]) -> tuple[float, ...]: ...


class CrossEncoderReranker:
    """Lazy in-process cross-encoder fallback for CPU or single-env runs."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        local_files_only: bool = False,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._local_files_only = local_files_only
        self._model: object | None = None

    @property
    def reranker_id(self) -> str:
        return self._model_name

    @property
    def max_length(self) -> int:
        return self._max_length

    def score(self, query: str, documents: Sequence[SearchDocument]) -> tuple[float, ...]:
        return self.score_texts(
            query,
            [reranker_document_text(document.title, document.body) for document in documents],
        )

    def score_texts(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        if not texts:
            return ()
        model = self._load_model()
        pairs = [(query, text) for text in texts]
        predictions = model.predict(  # type: ignore[attr-defined]
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = tuple(float(value) for value in predictions.reshape(-1))
        if len(scores) != len(texts):
            raise ValueError("reranker returned the wrong number of scores")
        return scores

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - dependency smoke test covers this
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder reranking"
                ) from exc
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
                local_files_only=self._local_files_only,
            )
        return self._model


class SubprocessCrossEncoderReranker:
    """Score pairs in a persistent GPU worker owned by another Python env."""

    def __init__(
        self,
        python_executable: Path,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        worker_path: Path | None = None,
        device: str = "cuda:0",
        batch_size: int = 1,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        use_fp16: bool = True,
        local_files_only: bool = False,
    ) -> None:
        if not python_executable.is_file():
            raise ValueError(f"reranker Python does not exist: {python_executable}")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        self._python_executable = python_executable
        self._worker_path = worker_path or (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "reranker"
            / "bge_reranker_worker.py"
        )
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._use_fp16 = use_fp16
        self._local_files_only = local_files_only
        self._process: subprocess.Popen[str] | None = None
        self._request_id = 0
        atexit.register(self.close)

    @property
    def reranker_id(self) -> str:
        precision = "fp16" if self._use_fp16 else "fp32"
        return f"subprocess:{self._model_name}:{self._device}:{precision}"

    def score(
        self,
        query: str,
        documents: Sequence[SearchDocument],
    ) -> tuple[float, ...]:
        return self.score_texts(
            query,
            [reranker_document_text(document.title, document.body) for document in documents],
        )

    def score_texts(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        if not texts:
            return ()
        process = self._ensure_started()
        self._request_id += 1
        payload = {
            "id": self._request_id,
            "query": query,
            "documents": list(texts),
        }
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("reranker worker pipes are unavailable")
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            return_code = process.poll()
            raise RuntimeError(f"reranker worker stopped unexpectedly: {return_code}")
        response = json.loads(response_line)
        if response.get("id") != self._request_id:
            raise RuntimeError("reranker worker returned a mismatched request ID")
        if "error" in response:
            raise RuntimeError(f"reranker worker failed: {response['error']}")
        scores = tuple(float(value) for value in response.get("scores", ()))
        if len(scores) != len(texts):
            raise RuntimeError("reranker worker returned the wrong number of scores")
        return scores

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
            process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        command = [
            str(self._python_executable),
            str(self._worker_path),
            "--model",
            self._model_name,
            "--device",
            self._device,
            "--batch-size",
            str(self._batch_size),
            "--max-length",
            str(self._max_length),
        ]
        if self._use_fp16:
            command.append("--fp16")
        if self._local_files_only:
            command.append("--local-files-only")
        process = subprocess.Popen(  # noqa: S603 - executable is an explicit local path
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if process.stdout is None:
            process.terminate()
            raise RuntimeError("reranker worker stdout is unavailable")
        ready_line = process.stdout.readline()
        if not ready_line:
            return_code = process.poll()
            raise RuntimeError(f"reranker worker failed to start: {return_code}")
        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            process.terminate()
            raise RuntimeError(f"reranker worker did not become ready: {ready}")
        self._process = process
        return process


class RerankedSearchBackend:
    """Retrieve coarse candidates, then jointly score query-item pairs."""

    def __init__(
        self,
        base_backend: SearchBackend,
        documents: Sequence[SearchDocument],
        reranker: PairReranker,
        *,
        candidate_k: int = 100,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self._base = base_backend
        self._documents = {document.document_id: document for document in documents}
        if len(self._documents) != len(documents):
            raise ValueError("document_id must be unique")
        self._reranker = reranker
        self._candidate_k = candidate_k

    @property
    def backend_id(self) -> str:
        return f"rerank:{self._reranker.reranker_id}:candidate-k={self._candidate_k}"

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        top_k = max(1, top_k)
        recall_k = max(top_k, self._candidate_k)
        coarse = self._base.search(query, top_k=recall_k)
        candidates = [
            self._documents[hit.document_id]
            for hit in coarse.hits
            if hit.document_id in self._documents
        ]
        if len(candidates) <= top_k:
            return _coarse_result(
                coarse,
                backend_id=self.backend_id,
                top_k=top_k,
                reason="reranker skipped: candidate count is not greater than top_k",
            )

        try:
            scores = self._reranker.score(query, candidates)
        except Exception as exc:  # noqa: BLE001 - reranker failure must degrade safely
            return _coarse_result(
                coarse,
                backend_id=self.backend_id,
                top_k=top_k,
                reason=f"reranker failed: {type(exc).__name__}",
            )

        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda entry: (-entry[0], entry[1].document_id),
        )
        hits = tuple(
            RecallHit(
                document_id=document.document_id,
                score=round(float(score), 8),
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked[:top_k], start=1)
        )
        return RecallResult(
            backend_id=self.backend_id,
            hits=hits,
            total_recall=coarse.total_recall,
            truncated=coarse.total_recall > top_k,
            fallback_reason=coarse.fallback_reason,
        )


def _coarse_result(
    result: RecallResult,
    *,
    backend_id: str,
    top_k: int,
    reason: str,
) -> RecallResult:
    inherited = f"; {result.fallback_reason}" if result.fallback_reason else ""
    return RecallResult(
        backend_id=backend_id,
        hits=tuple(
            RecallHit(document_id=hit.document_id, score=hit.score, rank=rank)
            for rank, hit in enumerate(result.hits[:top_k], start=1)
        ),
        total_recall=result.total_recall,
        truncated=result.total_recall > top_k,
        fallback_reason=f"{reason}{inherited}",
    )
