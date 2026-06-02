"""Tiny JSONL worker used to test the cross-environment reranker client."""

from __future__ import annotations

import json
import sys

print(
    json.dumps(
        {
            "status": "ready",
            "model": "BAAI/bge-reranker-v2-m3",
            "device": "cuda:0",
            "model_device": "cuda:0",
            "precision": "fp32",
            "max_length": 256,
            "batch_size": 1,
        }
    ),
    flush=True,
)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("command") == "shutdown":
        break
    scores = [float(index) for index, _ in enumerate(request["documents"])]
    print(json.dumps({"id": request["id"], "scores": scores}), flush=True)
