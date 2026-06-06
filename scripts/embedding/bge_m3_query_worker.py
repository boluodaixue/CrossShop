"""Persistent CUDA worker for BGE-M3 query-side encoding."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.max_seq_length != 512:
        raise SystemExit("catalog BGE-M3 query max_seq_length must be 512")

    torch, tokenizer, model, dimension = _load_model(
        args.model,
        device=args.device,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )
    _write(
        {
            "status": "ready",
            "model": args.model,
            "device": args.device,
            "model_device": str(next(model.parameters()).device),
            "cuda_available": bool(torch.cuda.is_available()),
            "precision": "fp16" if args.device.startswith("cuda") else "fp32",
            "pooling": "cls",
            "normalized": True,
            "dimension": dimension,
            "max_seq_length": args.max_seq_length,
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "shutdown":
            break
        request_id = request.get("id")
        try:
            vectors = _encode(
                torch,
                tokenizer,
                model,
                [str(text) for text in request["texts"]],
                device=args.device,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
            )
            _write(
                {
                    "id": request_id,
                    "device": args.device,
                    "vectors": vectors,
                }
            )
        except Exception as exc:  # noqa: BLE001 - return diagnostics to caller
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            _write({"id": request_id, "error": f"{type(exc).__name__}: {exc}"})


def _load_model(
    model_name: str,
    *,
    device: str,
    max_seq_length: int,
    local_files_only: bool,
) -> tuple[Any, Any, Any, int]:
    import torch
    from transformers import AutoModel, AutoTokenizer, logging

    logging.set_verbosity_error()
    if device != "cpu" and not device.startswith("cuda"):
        raise RuntimeError(f"BGE-M3 query worker device must be cpu or cuda, got {device}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the configured BGE-M3 worker environment"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    dimension = int(model.config.hidden_size)
    if dimension != 1024:
        raise RuntimeError(f"BGE-M3 hidden dimension must be 1024, got {dimension}")
    return torch, tokenizer, model, dimension


def _encode(
    torch: Any,
    tokenizer: Any,
    model: Any,
    texts: Sequence[str],
    *,
    device: str,
    batch_size: int,
    max_seq_length: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            cls_embeddings = model(**encoded).last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(
                cls_embeddings.float(),
                p=2,
                dim=1,
            )
            batch_vectors = normalized.cpu().tolist()
        vectors.extend(batch_vectors)
    return vectors


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
