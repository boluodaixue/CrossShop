"""Async adapter around the existing BGE-M3 SentenceTransformer encoder."""

from __future__ import annotations

import asyncio
import atexit
import json
import subprocess
import threading
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np

from globex_agent.domain.catalog.ports.retrieval_ports import EmbeddingClient
from globex_agent.infrastructure.recall.embedding import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerTextEncoder,
    TextEncoder,
)


class BgeM3EmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cuda:0",
        batch_size: int = 32,
        max_seq_length: int = DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
        local_files_only: bool = False,
        encoder: TextEncoder | None = None,
    ) -> None:
        self._encoder = encoder or SentenceTransformerTextEncoder(
            model_name,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            local_files_only=local_files_only,
        )

    @property
    def encoder_id(self) -> str:
        return self._encoder.encoder_id

    def ensure_ready(self) -> None:
        """Load the in-process model and verify the catalog vector contract."""

        matrix = np.asarray(self._encoder.encode_queries(["__startup__"]), dtype=np.float32)
        if matrix.shape != (1, 1024) or not np.isfinite(matrix).all():
            raise RuntimeError(f"BGE-M3 encoder returned an invalid startup vector: {matrix.shape}")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-3):
            raise RuntimeError("BGE-M3 encoder returned a non-normalized startup vector")

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = await asyncio.to_thread(self._encoder.encode_queries, texts)
        return [row.astype(float).tolist() for row in matrix]

    def close(self) -> None:
        """Match the persistent worker adapter's lifecycle API."""


class SubprocessBgeM3EmbeddingClient(EmbeddingClient):
    """Encode queries in one persistent CUDA worker process.

    The main project environment is intentionally CPU-only on some machines.
    This adapter keeps the BGE-M3 model resident in the explicitly configured
    CUDA environment instead of starting Python or loading the checkpoint for
    every query.
    """

    def __init__(
        self,
        python_executable: Path,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        worker_path: Path | None = None,
        device: str = "cuda:0",
        batch_size: int = 4,
        max_seq_length: int = DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
        local_files_only: bool = False,
    ) -> None:
        if not python_executable.is_file():
            raise ValueError(f"BGE-M3 CUDA Python does not exist: {python_executable}")
        if not device.startswith("cuda"):
            raise ValueError("SubprocessBgeM3EmbeddingClient requires a CUDA device")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_seq_length != DEFAULT_EMBEDDING_MAX_SEQ_LENGTH:
            raise ValueError("catalog BGE-M3 query max_seq_length must be 512")
        self._python_executable = python_executable
        self._worker_path = worker_path or (
            Path(__file__).resolve().parents[4]
            / "scripts"
            / "embedding"
            / "bge_m3_query_worker.py"
        )
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._local_files_only = local_files_only
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._process_state_lock = threading.Lock()
        self._request_id = 0
        self._dimension = 1024
        self._model_reuse_count = 0
        atexit.register(self.close)

    @property
    def encoder_id(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_reuse_count(self) -> int:
        return self._model_reuse_count

    def ensure_ready(self) -> None:
        """Start and contract-check the worker during application startup."""

        self._ensure_started()

    def warmup(self) -> None:
        """Run one small local query so the first request pays no model cost."""

        self._embed_sync(["__startup__"])

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        task = asyncio.create_task(asyncio.to_thread(self._embed_sync, texts))
        try:
            return await task
        except asyncio.CancelledError:
            self.abort()
            with suppress(BaseException):
                await task
            raise

    def close(self) -> None:
        with self._lock:
            process = self._get_process()
            self._set_process(None)
            if process is None or process.poll() is not None:
                return
            try:
                if process.stdin is not None:
                    process.stdin.write('{"command":"shutdown"}\n')
                    process.stdin.flush()
                process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._terminate_process(process)

    def abort(self) -> None:
        """Terminate an in-flight request without waiting for its I/O lock."""

        process = self._get_process()
        self._set_process(None)
        if process is not None:
            self._terminate_process(process)

    def _embed_sync(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            process = self._ensure_started_locked()
            try:
                self._request_id += 1
                request_id = self._request_id
                payload = {"id": request_id, "texts": list(texts)}
                if process.stdin is None or process.stdout is None:
                    raise RuntimeError("BGE-M3 query worker pipes are unavailable")
                process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                process.stdin.flush()
                response_line = process.stdout.readline()
                if not response_line:
                    raise RuntimeError(
                        f"BGE-M3 query worker stopped unexpectedly: {process.poll()}"
                    )
                response = json.loads(response_line)
                if response.get("id") != request_id:
                    raise RuntimeError("BGE-M3 query worker returned a mismatched request ID")
                if "error" in response:
                    raise RuntimeError(f"BGE-M3 query worker failed: {response['error']}")
                if response.get("device") != self._device:
                    raise RuntimeError("BGE-M3 query worker returned a wrong device")
                vectors = np.asarray(response.get("vectors", ()), dtype=np.float32)
                if vectors.shape != (len(texts), self._dimension):
                    raise RuntimeError(
                        "BGE-M3 query worker returned an invalid vector shape: "
                        f"{vectors.shape}"
                    )
                if not np.isfinite(vectors).all():
                    raise RuntimeError("BGE-M3 query worker returned non-finite vectors")
                norms = np.linalg.norm(vectors, axis=1)
                if not np.allclose(norms, 1.0, atol=2e-3):
                    raise RuntimeError("BGE-M3 query worker returned non-normalized vectors")
                self._model_reuse_count += 1
                return vectors.tolist()
            except Exception:
                self._reset_process_locked(process)
                raise

    def _ensure_started(self) -> subprocess.Popen[str]:
        with self._lock:
            return self._ensure_started_locked()

    def _ensure_started_locked(self) -> subprocess.Popen[str]:
        process = self._get_process()
        if process is not None and process.poll() is None:
            return process
        if not self._worker_path.is_file():
            raise RuntimeError(f"BGE-M3 query worker does not exist: {self._worker_path}")
        command = [
            str(self._python_executable),
            str(self._worker_path),
            "--model",
            self._model_name,
            "--device",
            self._device,
            "--batch-size",
            str(self._batch_size),
            "--max-seq-length",
            str(self._max_seq_length),
        ]
        if self._local_files_only:
            command.append("--local-files-only")
        process = subprocess.Popen(  # noqa: S603 - executable is explicit local path
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("BGE-M3 query worker stdout is unavailable")
            ready_line = process.stdout.readline()
            if not ready_line:
                raise RuntimeError(
                    "BGE-M3 query worker failed to start: "
                    f"returncode={process.poll()}"
                )
            ready = json.loads(ready_line)
            expected = {
                "status": "ready",
                "model": self._model_name,
                "device": self._device,
                "model_device": self._device,
                "cuda_available": True,
                "precision": "fp16",
                "pooling": "cls",
                "normalized": True,
                "dimension": self._dimension,
                "max_seq_length": self._max_seq_length,
            }
            if any(ready.get(key) != value for key, value in expected.items()):
                raise RuntimeError(
                    "BGE-M3 query worker contract mismatch: "
                    f"expected={expected}, got={ready}"
                )
        except Exception:
            self._terminate_process(process)
            raise
        self._set_process(process)
        return process

    def _reset_process_locked(self, process: subprocess.Popen[str]) -> None:
        if self._get_process() is process:
            self._set_process(None)
        self._terminate_process(process)

    def _get_process(self) -> subprocess.Popen[str] | None:
        with self._process_state_lock:
            return self._process

    def _set_process(self, process: subprocess.Popen[str] | None) -> None:
        with self._process_state_lock:
            self._process = process

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
