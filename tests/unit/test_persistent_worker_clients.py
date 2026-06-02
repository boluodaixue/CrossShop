from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import pytest

import globex_agent.infrastructure.embedding.bge_m3_embedding as embedding_module
import globex_agent.infrastructure.recall.reranker as reranker_module
from globex_agent.infrastructure.embedding.bge_m3_embedding import (
    SubprocessBgeM3EmbeddingClient,
)
from globex_agent.infrastructure.recall.reranker import SubprocessCrossEncoderReranker
from globex_agent.infrastructure.rerank.bge_reranker import BgeReranker


class _FakeStdout:
    def __init__(self, process: _FakeProcess, ready_line: str) -> None:
        self._process = process
        self._lines: Queue[str] = Queue()
        self._lines.put(ready_line)

    def push(self, line: str) -> None:
        self._lines.put(line)

    def readline(self) -> str:
        self._process.active_reads += 1
        self._process.max_active_reads = max(
            self._process.max_active_reads,
            self._process.active_reads,
        )
        try:
            time.sleep(0.01)
            try:
                return self._lines.get(timeout=2)
            except Empty:
                return ""
        finally:
            self._process.active_reads -= 1


class _FakeStdin:
    def __init__(self, process: _FakeProcess) -> None:
        self._process = process

    def write(self, line: str) -> int:
        self._process.request_lines.append(line)
        request = json.loads(line)
        if request.get("command") == "shutdown":
            self._process.shutdowns += 1
            self._process.returncode = 0
            return len(line)
        self._process.stdout.push(self._process.response_for(request))
        return len(line)

    def flush(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, kind: str, mode: str) -> None:
        self.kind = kind
        self.mode = mode
        self.returncode: int | None = None
        self.terminated = False
        self.shutdowns = 0
        self.active_reads = 0
        self.max_active_reads = 0
        self.request_lines: list[str] = []
        if kind == "embedding":
            ready = {
                "status": "ready",
                "model": "fake-model",
                "device": "cuda:0",
                "model_device": "cuda:0",
                "cuda_available": True,
                "precision": "fp16",
                "pooling": "cls",
                "normalized": True,
                "dimension": 1024,
                "max_seq_length": 512,
            }
        else:
            ready = {
                "status": "ready",
                "model": "fake-model",
                "device": "cuda:0",
                "model_device": "cuda:0",
                "precision": "fp32",
                "max_length": 256,
                "batch_size": 1,
            }
        self.stdout = _FakeStdout(self, json.dumps(ready))
        self.stdin = _FakeStdin(self)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9

    def response_for(self, request: dict[str, Any]) -> str:
        if self.mode == "malformed":
            return "{not-json\n"
        response_id = (
            int(request["id"]) + 100
            if self.mode == "mismatched"
            else request["id"]
        )
        if self.kind == "embedding":
            if self.mode == "invalid-shape":
                vectors: list[list[float]] = [[1.0, 0.0]]
            elif self.mode == "nonfinite":
                vectors = [[float("nan")] + [0.0] * 1023]
            else:
                vectors = [[1.0] + [0.0] * 1023 for _ in request["texts"]]
            return json.dumps(
                {"id": response_id, "device": "cuda:0", "vectors": vectors},
            )
        if self.mode == "wrong-count":
            scores = [0.1]
        elif self.mode == "nonfinite":
            scores = [float("inf") for _ in request["documents"]]
        else:
            scores = [float(index) for index, _ in enumerate(request["documents"])]
        return json.dumps({"id": response_id, "scores": scores})


class _ProcessFactory:
    def __init__(self, kind: str, modes: list[str]) -> None:
        self.kind = kind
        self.modes = deque(modes)
        self.processes: list[_FakeProcess] = []

    def __call__(self, command: list[str], **kwargs: Any) -> _FakeProcess:
        del command, kwargs
        mode = self.modes.popleft() if self.modes else "normal"
        process = _FakeProcess(self.kind, mode)
        self.processes.append(process)
        return process


def _worker_path() -> Path:
    return Path(__file__).parents[1] / "fixtures" / "fake_reranker_worker.py"


def _embedding_client() -> SubprocessBgeM3EmbeddingClient:
    return SubprocessBgeM3EmbeddingClient(
        Path(sys.executable),
        model_name="fake-model",
        worker_path=_worker_path(),
    )


def _reranker() -> SubprocessCrossEncoderReranker:
    return SubprocessCrossEncoderReranker(
        Path(sys.executable),
        model_name="fake-model",
        worker_path=_worker_path(),
        use_fp16=False,
    )


def test_embedding_concurrent_requests_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _ProcessFactory("embedding", ["normal"])
    monkeypatch.setattr(embedding_module.subprocess, "Popen", factory)
    client = _embedding_client()

    async def run() -> list[list[list[float]]]:
        return list(
            await asyncio.gather(
                client.embed_batch(["first"]),
                client.embed_batch(["second"]),
            )
        )

    try:
        results = asyncio.run(run())
        assert [len(result[0]) for result in results] == [1024, 1024]
        assert factory.processes[0].max_active_reads == 1
    finally:
        client.close()


@pytest.mark.parametrize("mode", ["malformed", "mismatched", "invalid-shape", "nonfinite"])
def test_embedding_protocol_error_poisoned_worker_is_restarted(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    factory = _ProcessFactory("embedding", [mode, "normal"])
    monkeypatch.setattr(embedding_module.subprocess, "Popen", factory)
    client = _embedding_client()
    try:
        with pytest.raises((RuntimeError, json.JSONDecodeError)):
            asyncio.run(client.embed_batch(["bad"]))
        assert factory.processes[0].terminated
        assert client._process is None
        result = asyncio.run(client.embed_batch(["recovered"]))
        assert len(result) == 1
        assert len(result[0]) == 1024
        assert len(factory.processes) == 2
    finally:
        client.close()


def test_embedding_close_sends_shutdown_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _ProcessFactory("embedding", ["normal"])
    monkeypatch.setattr(embedding_module.subprocess, "Popen", factory)
    client = _embedding_client()
    asyncio.run(client.embed_batch(["query"]))
    client.close()
    client.close()
    assert factory.processes[0].shutdowns == 1


def test_worker_requests_escape_non_ascii_for_cross_python_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_factory = _ProcessFactory("embedding", ["normal"])
    monkeypatch.setattr(embedding_module.subprocess, "Popen", embedding_factory)
    embedding_client = _embedding_client()
    try:
        asyncio.run(embedding_client.embed_batch(["中文查询"]))
        request_line = embedding_factory.processes[0].request_lines[0]
        assert all(ord(character) < 128 for character in request_line)
    finally:
        embedding_client.close()

    reranker_factory = _ProcessFactory("reranker", ["normal"])
    monkeypatch.setattr(reranker_module.subprocess, "Popen", reranker_factory)
    reranker = _reranker()
    try:
        reranker.score_texts("中文查询", ["中文商品描述"])
        request_line = reranker_factory.processes[0].request_lines[0]
        assert all(ord(character) < 128 for character in request_line)
    finally:
        reranker.close()


def test_reranker_concurrent_requests_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _ProcessFactory("reranker", ["normal"])
    monkeypatch.setattr(reranker_module.subprocess, "Popen", factory)
    reranker = _reranker()
    texts = ["first", "second"]
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda query: reranker.score_texts(query, texts),
                    ["one", "two"],
                )
            )
        assert results == [(0.0, 1.0), (0.0, 1.0)]
        assert factory.processes[0].max_active_reads == 1
    finally:
        reranker.close()


def test_reranker_ensure_ready_starts_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _ProcessFactory("reranker", ["normal"])
    monkeypatch.setattr(reranker_module.subprocess, "Popen", factory)
    reranker = _reranker()
    try:
        reranker.ensure_ready()
        assert len(factory.processes) == 1
        assert factory.processes[0].shutdowns == 0
    finally:
        reranker.close()


@pytest.mark.parametrize("mode", ["mismatched", "wrong-count", "nonfinite"])
def test_reranker_protocol_error_poisoned_worker_is_restarted(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    factory = _ProcessFactory("reranker", [mode, "normal"])
    monkeypatch.setattr(reranker_module.subprocess, "Popen", factory)
    reranker = _reranker()
    try:
        with pytest.raises(RuntimeError):
            reranker.score_texts("bad", ["one", "two"])
        assert factory.processes[0].terminated
        assert reranker._process is None
        assert reranker.score_texts("recovered", ["one", "two"]) == (0.0, 1.0)
        assert len(factory.processes) == 2
    finally:
        reranker.close()


def test_reranker_close_sends_shutdown_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _ProcessFactory("reranker", ["normal"])
    monkeypatch.setattr(reranker_module.subprocess, "Popen", factory)
    reranker = _reranker()
    assert reranker.score_texts("query", ["one"]) == (0.0,)
    reranker.close()
    reranker.close()
    assert factory.processes[0].shutdowns == 1


def test_async_rerank_cancellation_aborts_inflight_worker() -> None:
    started = threading.Event()
    released = threading.Event()

    class BlockingReranker:
        reranker_id = "blocking-worker"
        aborted = False

        def score_texts(self, query: str, texts: list[str]) -> tuple[float, ...]:
            del query, texts
            started.set()
            released.wait(timeout=2)
            return (0.5,)

        def abort(self) -> None:
            self.aborted = True
            released.set()

    worker = BlockingReranker()
    adapter = BgeReranker(reranker=worker)

    async def run() -> None:
        task = asyncio.create_task(adapter.rerank("query", ["document"]))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert worker.aborted
