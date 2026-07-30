from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.rerank.serve_bge_reranker import (
    SCORE_MODE,
    RerankerService,
    RerankRequest,
    ServiceConfig,
    create_app,
)


class _FakeBackend:
    def __init__(self, scores: list[float] | None = None) -> None:
        self.warmed_up = False
        self.calls: list[tuple[str, list[str]]] = []
        self._scores = scores

    def warmup(self) -> None:
        self.warmed_up = True

    def score_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self._scores is not None:
            return self._scores[: len(documents)]
        return [
            float(len(self.calls) * 10 + offset) for offset in range(len(documents))
        ]


class _WrongCountBackend(_FakeBackend):
    def score_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        return [1.0, 2.0]


def _config(**overrides: object) -> ServiceConfig:
    values: dict[str, object] = {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "model_path": Path("D:/models/bge-reranker-v2-m3"),
        "device": "cuda:0",
        "max_length": 256,
        "micro_batch_size": 2,
    }
    values.update(overrides)
    return ServiceConfig(**values)  # type: ignore[arg-type]


def test_http_protocol_warmup_health_and_original_indices() -> None:
    backend = _FakeBackend()
    app = create_app(_config(), backend_factory=lambda _: backend)

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/rerank",
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "轻便通勤包",
                "documents": ["商品 A", "商品 B", "商品 C"],
            },
        )

    assert backend.warmed_up is True
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": "BAAI/bge-reranker-v2-m3",
        "model_path": str(Path("D:/models/bge-reranker-v2-m3")),
        "device": "cuda:0",
        "dtype": "float16",
        "max_length": 256,
        "micro_batch_size": 2,
        "score_mode": "raw_logit",
    }
    assert response.status_code == 200
    assert response.json() == {
        "model": "BAAI/bge-reranker-v2-m3",
        "score_mode": SCORE_MODE,
        "results": [
            {"index": 0, "relevance_score": 10.0},
            {"index": 1, "relevance_score": 11.0},
            {"index": 2, "relevance_score": 20.0},
        ],
    }
    assert backend.calls == [
        ("轻便通勤包", ["商品 A", "商品 B"]),
        ("轻便通勤包", ["商品 C"]),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "BAAI/bge-reranker-v2-m3", "query": " ", "documents": ["doc"]},
        {"model": "BAAI/bge-reranker-v2-m3", "query": "query", "documents": []},
        {"model": "BAAI/bge-reranker-v2-m3", "query": "query", "documents": [""]},
        {
            "model": "BAAI/bge-reranker-v2-m3",
            "query": "query",
            "documents": ["doc"],
            "unexpected": True,
        },
    ],
)
def test_http_protocol_rejects_invalid_requests(payload: dict[str, object]) -> None:
    app = create_app(_config(), backend_factory=lambda _: _FakeBackend())
    with TestClient(app) as client:
        response = client.post("/rerank", json=payload)
    assert response.status_code == 422


def test_http_protocol_rejects_a_different_model() -> None:
    app = create_app(_config(), backend_factory=lambda _: _FakeBackend())
    with TestClient(app) as client:
        response = client.post(
            "/rerank",
            json={"model": "another/model", "query": "query", "documents": ["doc"]},
        )
    assert response.status_code == 400
    assert "unsupported model" in response.json()["detail"]


@pytest.mark.parametrize("backend", [_WrongCountBackend(), _FakeBackend([math.nan])])
def test_backend_contract_errors_are_reported_as_unavailable(
    backend: _FakeBackend,
) -> None:
    service = RerankerService(_config(micro_batch_size=1), backend)
    request = RerankRequest(
        model="BAAI/bge-reranker-v2-m3",
        query="query",
        documents=["doc"],
    )

    with pytest.raises(RuntimeError):
        service.rerank(request)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"device": "cpu"}, "explicit CUDA"),
        ({"max_length": 0}, "max_length"),
        ({"micro_batch_size": 0}, "micro_batch_size"),
    ],
)
def test_service_config_rejects_unsafe_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)
