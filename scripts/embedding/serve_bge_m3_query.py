"""Serve the frozen H1 BGE-M3 Query encoder through ``/v1/embeddings``.

This local development service binds only to ``127.0.0.1``.  It directly
reuses H1's ``BgeM3Encoder`` on CPU, so Query vectors use the same CLS pooling,
float32 L2 normalization, 1024 dimensions, and max length 512 as Item vectors.
Torch and Transformers remain dependencies of the dedicated model environment.
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
from typing import Any, Literal, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:
    from fastapi import FastAPI, HTTPException, Request
except ImportError:  # The dedicated model environment may use stdlib HTTP only.
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment,misc]
    Request = Any  # type: ignore[assignment,misc]

LOGGER = logging.getLogger("globex.bge_m3_query")

MODEL_ID = "BAAI/bge-m3"
DEFAULT_MODEL_PATH = Path(r"D:\models\bge-m3")
DEVICE = "cpu"
DTYPE = "float32"
VECTOR_DIMENSION = 1024
MAX_LENGTH = 512
DEFAULT_MICRO_BATCH_SIZE = 4
MAX_REQUEST_INPUTS = 64
POOLING = "cls"
NORMALIZATION = "l2"


@dataclass(frozen=True)
class ServiceConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("model_path must not be empty")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.micro_batch_size > MAX_REQUEST_INPUTS:
            raise ValueError(f"micro_batch_size must not exceed {MAX_REQUEST_INPUTS}")


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    input: Union[str, list[str]]  # noqa: UP007 - dedicated env is Python 3.9
    encoding_format: Literal["float"] = "float"
    dimensions: Optional[Literal[VECTOR_DIMENSION]] = None  # noqa: UP045 - Python 3.9

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("input")
    @classmethod
    def validate_input(
        cls,
        value: Union[str, list[str]],  # noqa: UP007 - dedicated env is Python 3.9
    ) -> Union[str, list[str]]:  # noqa: UP007 - dedicated env is Python 3.9
        texts = [value] if isinstance(value, str) else value
        if not texts:
            raise ValueError("input must not be empty")
        if len(texts) > MAX_REQUEST_INPUTS:
            raise ValueError(f"input must contain at most {MAX_REQUEST_INPUTS} texts")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("input must contain only non-blank strings")
        return value

    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class BatchEmbeddingBackend(Protocol):
    def warmup(self) -> None: ...

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

    def token_count(self, texts: Sequence[str]) -> list[int]: ...


class BgeM3CpuBackend:
    """Thin CPU wrapper around the exact H1 ``BgeM3Encoder``."""

    def __init__(self, config: ServiceConfig) -> None:
        from scripts.index.product_opensearch_common import BgeM3Encoder

        if not config.model_path.is_dir():
            raise RuntimeError(
                f"BGE-M3 model directory does not exist: {config.model_path}"
            )
        self._encoder = BgeM3Encoder(
            model_name=str(config.model_path),
            device=DEVICE,
            max_seq_length=MAX_LENGTH,
            local_files_only=True,
        )

    def warmup(self) -> None:
        vectors = self.encode_batch(["warmup"])
        _validate_vectors(vectors, expected_count=1)

    def encode_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encoder.encode(list(texts))

    def token_count(self, texts: Sequence[str]) -> list[int]:
        return self._encoder.token_count(list(texts))


class EmbeddingService:
    def __init__(self, config: ServiceConfig, backend: BatchEmbeddingBackend) -> None:
        self.config = config
        self._backend = backend
        self._inference_lock = threading.Lock()

    def warmup(self) -> None:
        self._backend.warmup()

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.model != MODEL_ID:
            raise ValueError(
                f"unsupported model {request.model!r}; expected {MODEL_ID!r}"
            )

        texts = request.texts()
        vectors: list[list[float]] = []
        size = self.config.micro_batch_size
        # CPU inference is serialized to bound RAM use and prevent oversubscription.
        with self._inference_lock:
            token_counts = self._backend.token_count(texts)
            _validate_token_counts(token_counts, expected_count=len(texts))
            for start in range(0, len(texts), size):
                batch = texts[start : start + size]
                batch_vectors = self._backend.encode_batch(batch)
                _validate_vectors(batch_vectors, expected_count=len(batch))
                vectors.extend(batch_vectors)

        prompt_tokens = sum(token_counts)
        return EmbeddingResponse(
            data=[
                EmbeddingData(index=index, embedding=vector)
                for index, vector in enumerate(vectors)
            ],
            model=MODEL_ID,
            usage=EmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model": MODEL_ID,
            "model_path": str(self.config.model_path),
            "device": DEVICE,
            "dtype": DTYPE,
            "dimension": VECTOR_DIMENSION,
            "max_length": MAX_LENGTH,
            "micro_batch_size": self.config.micro_batch_size,
            "pooling": POOLING,
            "normalization": NORMALIZATION,
            "local_files_only": True,
        }


def _validate_vectors(vectors: object, *, expected_count: int) -> None:
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        actual = len(vectors) if isinstance(vectors, list) else type(vectors).__name__
        raise RuntimeError(
            "embedding backend returned an invalid vector count: "
            f"expected={expected_count}, actual={actual}"
        )
    for index, vector in enumerate(vectors):
        if not isinstance(vector, list) or len(vector) != VECTOR_DIMENSION:
            actual = len(vector) if isinstance(vector, list) else type(vector).__name__
            raise RuntimeError(
                f"embedding {index} must have {VECTOR_DIMENSION} dimensions, "
                f"actual={actual}"
            )
        if not all(
            type(value) in (int, float) and math.isfinite(value) for value in vector
        ):
            raise RuntimeError(f"embedding {index} contains a non-finite value")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError(
                f"embedding {index} must be L2-normalized, norm={norm:.6f}"
            )


def _validate_token_counts(counts: object, *, expected_count: int) -> None:
    if not isinstance(counts, list) or len(counts) != expected_count:
        actual = len(counts) if isinstance(counts, list) else type(counts).__name__
        raise RuntimeError(
            "embedding backend returned an invalid token count list: "
            f"expected={expected_count}, actual={actual}"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts
    ):
        raise RuntimeError("embedding backend returned an invalid token count")


BackendFactory = Callable[[ServiceConfig], BatchEmbeddingBackend]


def create_app(
    config=None,
    *,
    backend_factory: BackendFactory = BgeM3CpuBackend,
) -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed; use this module's standard-library CLI"
        )
    resolved_config = config or config_from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = EmbeddingService(
            resolved_config,
            backend_factory(resolved_config),
        )
        service.warmup()
        app.state.embedding_service = service
        LOGGER.info("BGE-M3 Query encoder ready: %s", service.health())
        yield

    application = FastAPI(title="Globex BGE-M3 Query Encoder", lifespan=lifespan)

    def get_service(request: Request) -> EmbeddingService:
        return request.app.state.embedding_service

    @application.get("/health")
    def health(request: Request) -> dict[str, object]:
        return get_service(request).health()

    @application.post("/v1/embeddings", response_model=EmbeddingResponse)
    def embeddings(
        payload: EmbeddingRequest,
        request: Request,
    ) -> EmbeddingResponse:
        service = get_service(request)
        try:
            return service.embed(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


def config_from_environment() -> ServiceConfig:
    return ServiceConfig(
        model_path=Path(os.getenv("BGE_M3_QUERY_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        micro_batch_size=int(
            os.getenv(
                "BGE_M3_QUERY_MICRO_BATCH_SIZE",
                str(DEFAULT_MICRO_BATCH_SIZE),
            )
        ),
    )


app = create_app() if FastAPI is not None else None


def _serve_stdlib(config: ServiceConfig, port: int) -> None:
    service = EmbeddingService(config, BgeM3CpuBackend(config))
    service.warmup()
    LOGGER.info("BGE-M3 Query encoder ready: %s", service.health())

    class Handler(BaseHTTPRequestHandler):
        server_version = "GlobexBgeM3Query/1.0"

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
            if self.path != "/v1/embeddings":
                self._write_json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 10 * 1024 * 1024:
                    raise ValueError("invalid request body length")
                payload = json.loads(self.rfile.read(length))
                request = EmbeddingRequest.model_validate(payload)
                response = service.embed(request)
                self._write_json(HTTPStatus.OK, response.model_dump())
            except (ValidationError, json.JSONDecodeError) as exc:
                self._write_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": str(exc)})
            except ValueError as exc:
                self._write_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
            except RuntimeError as exc:
                self._write_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"detail": str(exc)},
                )

        def log_message(self, message: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), message % args)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    LOGGER.info("listening on http://127.0.0.1:%d", port)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper())
    # Deliberately no host option: this development service is loopback-only.
    _serve_stdlib(config_from_environment(), args.port)


if __name__ == "__main__":
    main()
