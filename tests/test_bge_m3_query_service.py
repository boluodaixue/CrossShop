from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.embedding.serve_bge_m3_query import (
    MAX_LENGTH,
    MODEL_ID,
    VECTOR_DIMENSION,
    BgeM3CpuBackend,
    EmbeddingRequest,
    EmbeddingService,
    ServiceConfig,
    create_app,
)
from scripts.index.product_opensearch_common import BgeM3Encoder


def _unit_vector(axis: int = 0) -> list[float]:
    vector = [0.0] * VECTOR_DIMENSION
    vector[axis] = 1.0
    return vector


class _FakeBackend:
    def __init__(self) -> None:
        self.warmed_up = False
        self.calls: list[list[str]] = []

    def warmup(self) -> None:
        self.warmed_up = True

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [
            _unit_vector((len(self.calls) + index) % 8) for index, _ in enumerate(texts)
        ]

    def token_count(self, texts: Sequence[str]) -> list[int]:
        return [min(len(text) + 2, MAX_LENGTH) for text in texts]


class _BrokenBackend(_FakeBackend):
    def __init__(self, vectors: object) -> None:
        super().__init__()
        self._vectors = vectors

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self._vectors  # type: ignore[return-value]


def _config(**overrides: object) -> ServiceConfig:
    values: dict[str, object] = {
        "model_path": Path(r"D:\models\bge-m3"),
        "micro_batch_size": 2,
    }
    values.update(overrides)
    return ServiceConfig(**values)  # type: ignore[arg-type]


def test_openai_protocol_health_batching_and_original_indices() -> None:
    backend = _FakeBackend()
    app = create_app(_config(), backend_factory=lambda _: backend)

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/embeddings",
            json={
                "model": MODEL_ID,
                "input": ["降噪耳机", "轻便旅行背包", "旅行茶具"],
                "encoding_format": "float",
                "dimensions": 1024,
            },
        )

    assert backend.warmed_up is True
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model": MODEL_ID,
        "model_path": r"D:\models\bge-m3",
        "device": "cpu",
        "dtype": "float32",
        "dimension": 1024,
        "max_length": 512,
        "micro_batch_size": 2,
        "pooling": "cls",
        "normalization": "l2",
        "local_files_only": True,
    }
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == MODEL_ID
    assert [item["index"] for item in body["data"]] == [0, 1, 2]
    assert all(item["object"] == "embedding" for item in body["data"])
    assert all(len(item["embedding"]) == 1024 for item in body["data"])
    expected_tokens = sum(
        len(text) + 2 for text in ["降噪耳机", "轻便旅行背包", "旅行茶具"]
    )
    assert body["usage"] == {
        "prompt_tokens": expected_tokens,
        "total_tokens": expected_tokens,
    }
    assert backend.calls == [
        ["降噪耳机", "轻便旅行背包"],
        ["旅行茶具"],
    ]


def test_openai_protocol_accepts_one_string_input() -> None:
    backend = _FakeBackend()
    app = create_app(_config(), backend_factory=lambda _: backend)
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": MODEL_ID, "input": "noise cancelling headphones"},
        )
    assert response.status_code == 200
    body = response.json()
    assert [item["index"] for item in body["data"]] == [0]
    assert body["usage"] == {
        "prompt_tokens": len("noise cancelling headphones") + 2,
        "total_tokens": len("noise cancelling headphones") + 2,
    }
    shared_response = EmbeddingService(_config(), _FakeBackend()).embed(
        EmbeddingRequest(model=MODEL_ID, input="noise cancelling headphones")
    )
    assert body == shared_response.model_dump()


@pytest.mark.parametrize(
    "payload",
    [
        {"model": MODEL_ID, "input": " "},
        {"model": MODEL_ID, "input": []},
        {"model": MODEL_ID, "input": [""]},
        {"model": MODEL_ID, "input": ["ok"], "dimensions": 768},
        {"model": MODEL_ID, "input": ["ok"], "encoding_format": "base64"},
        {"model": MODEL_ID, "input": ["ok"], "unexpected": True},
    ],
)
def test_openai_protocol_rejects_invalid_requests(payload: dict[str, object]) -> None:
    app = create_app(_config(), backend_factory=lambda _: _FakeBackend())
    with TestClient(app) as client:
        response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422


def test_openai_protocol_rejects_a_different_model() -> None:
    app = create_app(_config(), backend_factory=lambda _: _FakeBackend())
    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "another/model", "input": ["query"]},
        )
    assert response.status_code == 400
    assert "unsupported model" in response.json()["detail"]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([], "vector count"),
        ([[1.0]], "1024 dimensions"),
        ([_unit_vector()[:-1] + [math.nan]], "non-finite"),
        ([[0.5] + [0.0] * (VECTOR_DIMENSION - 1)], "L2-normalized"),
    ],
)
def test_backend_contract_errors_are_reported_as_unavailable(
    vectors: object,
    message: str,
) -> None:
    service = EmbeddingService(_config(), _BrokenBackend(vectors))
    request = EmbeddingRequest(model=MODEL_ID, input=["query"])
    with pytest.raises(RuntimeError, match=message):
        service.embed(request)


@pytest.mark.parametrize("batch_size", [0, -1, 65])
def test_service_config_rejects_unsafe_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="micro_batch_size"):
        _config(micro_batch_size=batch_size)


def test_cpu_backend_reuses_frozen_h1_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class _FakeEncoder:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [_unit_vector() for _ in texts]

        def token_count(self, texts: list[str]) -> list[int]:
            return [len(text) + 2 for text in texts]

    monkeypatch.setattr(Path, "is_dir", lambda _: True)
    monkeypatch.setattr(
        "scripts.index.product_opensearch_common.BgeM3Encoder",
        _FakeEncoder,
    )
    backend = BgeM3CpuBackend(_config())

    assert backend.encode_batch(["query"]) == [_unit_vector()]
    assert backend.token_count(["query"]) == [7]
    assert calls == [
        {
            "model_name": r"D:\models\bge-m3",
            "device": "cpu",
            "max_seq_length": MAX_LENGTH,
            "local_files_only": True,
        }
    ]


def test_encoder_token_count_uses_matching_truncation_without_padding() -> None:
    calls: list[dict[str, object]] = []

    class _FakeTokenizer:
        def __call__(
            self, texts: list[str], **kwargs: object
        ) -> dict[str, list[list[int]]]:
            calls.append({"texts": texts, **kwargs})
            max_length = int(kwargs["max_length"])
            return {
                "attention_mask": [
                    [1] * min(len(text) + 2, max_length) for text in texts
                ]
            }

    encoder = object.__new__(BgeM3Encoder)
    encoder._tokenizer = _FakeTokenizer()  # type: ignore[attr-defined]
    encoder._max_seq_length = 5  # type: ignore[attr-defined]

    assert encoder.token_count(["a", "abcdefgh"]) == [3, 5]
    assert calls == [
        {
            "texts": ["a", "abcdefgh"],
            "padding": False,
            "truncation": True,
            "max_length": 5,
            "return_attention_mask": True,
        }
    ]
