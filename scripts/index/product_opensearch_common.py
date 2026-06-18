"""Shared H1-only OpenSearch and BGE-M3 helpers."""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional


class OpenSearchError(RuntimeError):
    pass


class OpenSearchClient:
    def __init__(self, endpoint: str, timeout: int = 120) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        body: Optional[dict[str, Any]] = None,  # noqa: UP045 - CUDA env is Python 3.9
        *,
        content_type: str = "application/json",
        raw_body: Optional[bytes] = None,  # noqa: UP045 - CUDA env is Python 3.9
    ) -> Any:
        data = raw_body
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint + path,
            data=data,
            method=method,
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenSearchError(
                f"{method} {path} failed ({exc.code}): {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenSearchError(f"{method} {path} failed: {exc.reason}") from exc
        return json.loads(payload) if payload else None

    def exists(self, path: str) -> bool:
        request = urllib.request.Request(self.endpoint + path, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise

    def bulk(self, index_name: str, actions: list[tuple[str, dict[str, Any]]]) -> None:
        lines: list[str] = []
        for document_id, document in actions:
            lines.append(
                json.dumps({"index": {"_id": document_id}}, separators=(",", ":"))
            )
            lines.append(
                json.dumps(document, ensure_ascii=False, separators=(",", ":"))
            )
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        result = self.request(
            "POST",
            f"/{index_name}/_bulk?filter_path=errors,items.*.error",
            content_type="application/x-ndjson",
            raw_body=payload,
        )
        if result.get("errors"):
            errors = [
                item for item in result.get("items", []) if item["index"].get("error")
            ]
            raise OpenSearchError(f"bulk indexing failed: {errors[:3]}")


class BgeM3Encoder:
    """The frozen shared Query/Item CLS-pooling encoder."""

    def __init__(
        self,
        model_name: str,
        device: str,
        max_seq_length: int,
        local_files_only: bool,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self._torch = torch
        self._device = device
        self._max_seq_length = max_seq_length
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(device)
        self._model.eval()
        hidden_size = int(self._model.config.hidden_size)
        if hidden_size != 1024:
            raise RuntimeError(
                f"BGE-M3 hidden dimension must be 1024, got {hidden_size}"
            )

    def encode(self, texts: list[str]) -> list[list[float]]:
        torch = self._torch
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            cls_embeddings = self._model(**encoded).last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(
                cls_embeddings.float(), p=2, dim=1
            )
        vectors = normalized.cpu().tolist()
        for vector in vectors:
            if len(vector) != 1024 or not all(math.isfinite(value) for value in vector):
                raise RuntimeError("BGE-M3 returned an invalid vector")
        return vectors


def search_path(
    index_name: str,
    pipeline: Optional[str] = None,  # noqa: UP045 - CUDA env is Python 3.9
) -> str:
    path = f"/{index_name}/_search"
    if pipeline:
        path += "?search_pipeline=" + urllib.parse.quote(pipeline, safe="")
    return path
