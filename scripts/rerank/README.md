# Local BGE reranker

This development/acceptance service loads `BAAI/bge-reranker-v2-m3` from the
local `D:\models\bge-reranker-v2-m3` directory. It binds only to `127.0.0.1`,
uses CUDA FP16, warms the model before reporting healthy, and never falls back
to CPU. Its CLI uses Python's standard-library HTTP server so the dedicated
model environment does not need FastAPI or Uvicorn installed.

Start it from the repository root with the dedicated model environment:

```powershell
C:\Anaconda\envs\blog_04\python.exe -m scripts.rerank.serve_bge_reranker --port 8001
```

Then configure the existing client:

```dotenv
RERANKER_BASE_URL=http://127.0.0.1:8001
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

Runtime model settings can be overridden with `BGE_RERANKER_MODEL_PATH`,
`BGE_RERANKER_DEVICE`, `BGE_RERANKER_MAX_LENGTH`, and
`BGE_RERANKER_MICRO_BATCH_SIZE`. Defaults are `cuda:0`, max length 256, and
micro-batch size 2. `/health` reports the effective settings.

`/rerank` returns the model's raw sequence-classification logits as
`relevance_score`. These scores are finite relative-ranking values, not
probabilities. The existing `HttpReranker` client rejects malformed responses,
including duplicate or out-of-range indices and missing or non-finite scores.
