"""Serve BAAI/bge-reranker-v2-m3 behind the existing ``/rerank`` protocol.

This is a local development and acceptance service, not an application-layer
adapter.  ``torch`` and ``transformers`` are intentionally imported lazily so
they remain dependencies of the dedicated model environment rather than the
main CrossShop application.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # The dedicated Python 3.9 model environment uses stdlib HTTP.
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[assignment,misc]

LOGGER = logging.getLogger("crossshop.bge_reranker")

DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_PATH = Path(r"D:\models\bge-reranker-v2-m3")
DEFAULT_DEVICE = "cuda:0"
DEFAULT_MAX_LENGTH = 256
DEFAULT_MICRO_BATCH_SIZE = 2
SCORE_MODE = "raw_logit"


@dataclass(frozen=True)
class ServiceConfig:
    model_id: str = DEFAULT_MODEL_ID
    model_path: Path = DEFAULT_MODEL_PATH
    device: str = DEFAULT_DEVICE
    max_length: int = DEFAULT_MAX_LENGTH
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not str(self.model_path).strip():
            raise ValueError("model_path must not be empty")
        if not self.device.startswith("cuda:"):
            raise ValueError("device must be an explicit CUDA device such as cuda:0")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1)

    @field_validator("model", "query")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("documents")
    @classmethod
    def reject_blank_documents(cls, value: list[str]) -> list[str]:
        if any(not document.strip() for document in value):
            raise ValueError("documents must contain only non-blank strings")
        return value


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    score_mode: str
    results: list[RerankResult]


class BatchScoreBackend(Protocol):
    def warmup(self) -> None: ...

    def score_batch(self, query: str, documents: Sequence[str]) -> list[float]: ...


class BgeSequenceClassificationBackend:
    """CUDA FP16 sequence-classification backend with raw-logit scores."""

    def __init__(self, config: ServiceConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The dedicated reranker environment requires torch and transformers"
            ) from exc

        if not config.model_path.is_dir():
            raise RuntimeError(
                f"reranker model directory does not exist: {config.model_path}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the reranker; CPU fallback is disabled"
            )

        try:
            device_index = int(config.device.partition(":")[2])
        except ValueError as exc:
            raise RuntimeError(f"invalid CUDA device: {config.device}") from exc
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {config.device} is unavailable; found {torch.cuda.device_count()} device(s)"
            )

        self._torch = torch
        self._device = torch.device(config.device)
        self._max_length = config.max_length
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                local_files_only=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                config.model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
            )
            self._model.to(self._device)
            self._model.eval()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory while loading bge-reranker-v2-m3; CPU fallback is disabled"
            ) from exc

    def warmup(self) -> None:
        scores = self.score_batch("warmup", ["warmup document"])
        if len(scores) != 1 or not math.isfinite(scores[0]):
            raise RuntimeError("reranker warmup returned an invalid score")

    def score_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [[query, document] for document in documents]
        try:
            encoded = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {
                name: tensor.to(self._device) for name, tensor in encoded.items()
            }
            with self._torch.inference_mode():
                logits = (
                    self._model(**encoded).logits.reshape(-1).float().cpu().tolist()
                )
        except self._torch.cuda.OutOfMemoryError as exc:
            self._torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA out of memory during reranking; reduce micro-batch size; CPU fallback is disabled"
            ) from exc
        return [float(score) for score in logits]


class RerankerService:
    def __init__(self, config: ServiceConfig, backend: BatchScoreBackend) -> None:
        self.config = config
        self._backend = backend
        self._inference_lock = threading.Lock()

    def warmup(self) -> None:
        self._backend.warmup()

    def rerank(self, request: RerankRequest) -> RerankResponse:
        if request.model != self.config.model_id:
            raise ValueError(
                f"unsupported model {request.model!r}; expected {self.config.model_id!r}"
            )

        scores: list[float] = []
        size = self.config.micro_batch_size
        # Serialize GPU inference on the 4 GiB local acceptance device.
        with self._inference_lock:
            for start in range(0, len(request.documents), size):
                batch = request.documents[start : start + size]
                batch_scores = self._backend.score_batch(request.query, batch)
                if len(batch_scores) != len(batch):
                    raise RuntimeError(
                        "reranker backend returned a score count different from the document count"
                    )
                if any(not math.isfinite(score) for score in batch_scores):
                    raise RuntimeError("reranker backend returned a non-finite score")
                scores.extend(batch_scores)

        return RerankResponse(
            model=self.config.model_id,
            score_mode=SCORE_MODE,
            results=[
                RerankResult(index=index, relevance_score=score)
                for index, score in enumerate(scores)
            ],
        )

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.config.model_id,
            "model_path": str(self.config.model_path),
            "device": self.config.device,
            "dtype": "float16",
            "max_length": self.config.max_length,
            "micro_batch_size": self.config.micro_batch_size,
            "score_mode": SCORE_MODE,
        }


BackendFactory = Callable[[ServiceConfig], BatchScoreBackend]


def create_app(
    config=None,
    *,
    backend_factory: BackendFactory = BgeSequenceClassificationBackend,
) -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed; use this module's standard-library CLI"
        )
    resolved_config = config or config_from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = RerankerService(resolved_config, backend_factory(resolved_config))
        service.warmup()
        app.state.reranker_service = service
        LOGGER.info("reranker ready: %s", service.health())
        yield

    application = FastAPI(title="CrossShop BGE Reranker", lifespan=lifespan)

    def get_service(request: Request) -> RerankerService:
        return request.app.state.reranker_service

    @application.get("/health")
    def health(request: Request) -> dict:
        return get_service(request).health()

    @application.post("/rerank", response_model=RerankResponse)
    def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
        service = get_service(request)
        try:
            return service.rerank(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


def config_from_environment() -> ServiceConfig:
    return ServiceConfig(
        model_id=os.getenv("BGE_RERANKER_MODEL_ID", DEFAULT_MODEL_ID),
        model_path=Path(os.getenv("BGE_RERANKER_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        device=os.getenv("BGE_RERANKER_DEVICE", DEFAULT_DEVICE),
        max_length=int(os.getenv("BGE_RERANKER_MAX_LENGTH", str(DEFAULT_MAX_LENGTH))),
        micro_batch_size=int(
            os.getenv("BGE_RERANKER_MICRO_BATCH_SIZE", str(DEFAULT_MICRO_BATCH_SIZE))
        ),
    )


app = create_app() if FastAPI is not None else None


def _serve_stdlib(config: ServiceConfig, port: int) -> None:
    service = RerankerService(config, BgeSequenceClassificationBackend(config))
    service.warmup()
    LOGGER.info("reranker ready: %s", service.health())

    class Handler(BaseHTTPRequestHandler):
        server_version = "CrossShopBgeReranker/1.0"

        def _write_json(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._write_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
                return
            self._write_json(HTTPStatus.OK, service.health())

        def do_POST(self) -> None:
            if self.path != "/rerank":
                self._write_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10 * 1024 * 1024:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(length))
                request = RerankRequest.model_validate(payload)
                response = service.rerank(request)
                self._write_json(HTTPStatus.OK, response.model_dump())
            except (ValidationError, json.JSONDecodeError) as exc:
                self._write_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": str(exc)})
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            except RuntimeError as exc:
                self._write_json(HTTPStatus.SERVICE_UNAVAILABLE, {"detail": str(exc)})

        def log_message(self, message: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), message % args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    LOGGER.info("listening on http://127.0.0.1:%d", port)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())
    # Deliberately do not expose a host option: this development service is loopback-only.
    _serve_stdlib(config_from_environment(), args.port)


if __name__ == "__main__":
    main()
