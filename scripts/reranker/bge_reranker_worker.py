"""Persistent JSONL worker for BGE-Reranker-v2-m3 GPU inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    torch, tokenizer, model = _load_model(
        args.model,
        device=args.device,
        use_fp16=args.fp16,
        local_files_only=args.local_files_only,
    )
    _write(
        {
            "status": "ready",
            "model": args.model,
            "device": args.device,
            "model_device": str(next(model.parameters()).device),
            "precision": "fp16" if args.fp16 else "fp32",
            "max_length": args.max_length,
            "batch_size": args.batch_size,
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "shutdown":
            break
        request_id = request.get("id")
        try:
            scores = _score(
                torch,
                tokenizer,
                model,
                str(request["query"]),
                [str(document) for document in request["documents"]],
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            _write({"id": request_id, "scores": scores})
        except Exception as exc:  # noqa: BLE001 - preserve worker for caller fallback
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            _write({"id": request_id, "error": f"{type(exc).__name__}: {exc}"})


def _load_model(
    model_name: str,
    *,
    device: str,
    use_fp16: bool,
    local_files_only: bool,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        logging,
    )

    logging.set_verbosity_error()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the reranker environment")
    dtype = torch.float16 if use_fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return torch, tokenizer, model


def _score(
    torch: Any,
    tokenizer: Any,
    model: Any,
    query: str,
    documents: Sequence[str],
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        inputs = tokenizer(
            [query] * len(batch),
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits.reshape(-1).float().cpu().tolist()
        scores.extend(float(value) for value in logits)
    return scores


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
