# CrossShop Agent

CrossShop Agent is a reference cross-border shopping assistant built with AgentScope 2.x. It searches a normalized product catalog, explains category knowledge, remembers conversation context, and supports a controlled order workflow. It does not execute real payments.

## Architecture

- **Product facts:** read-only normalized JSONL (`JsonlProductRepository`).
- **Product retrieval:** OpenSearch Product Index with ANN + BM25 hybrid retrieval and RRF, followed by optional reranking.
- **CategoryInsight:** an independent OpenSearch Knowledge Index for category guidance and source-aware explanations.
- **Application state:** SQLite stores conversations, events, orders, preferences, and the latest complete `AgentState` snapshot. At the end of each turn the snapshot for a session is overwritten; historical state rollback is not provided. With `DATABASE_URL=file`, session snapshots are written under `DATA_DIR/sessions/` instead.
- **Redis:** optional semantic/embedding cache, Stream queue, Pub/Sub event bridge, and resilience coordination.

Order tables are the source of truth for transaction state; an AgentState snapshot is only conversational context.

## Quick start

Requirements: Python 3.11–3.13, `uv`, and Node.js 18+ for the frontend. Copy `.env.example` to `.env` and provide an OpenAI-compatible model endpoint. Start OpenSearch (and optionally Redis/Qdrant), then run:

```powershell
uv sync
uv run uvicorn app.presentation.server:create_app --factory --reload
```

To build the frontend:

```powershell
cd frontend
npm ci
npm run build
```

Product and knowledge indexes are built offline. See `scripts/index/` and `scripts/data/`; no generated index or local database is committed.

## Configuration

All settings are documented in `.env.example`. Important variables include `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `PRODUCT_CATALOG_ROOT`, `OPENSEARCH_ENDPOINT`, `CATEGORY_KB_BACKEND`, `CATEGORY_KNOWLEDGE_INDEX`, `DATABASE_URL`, and optional Redis settings. Never commit `.env` or credentials.

## Tests and evaluation

The default suite is offline and uses fakes/in-memory repositories where possible:

```powershell
uv run ruff check .
uv run pytest
```

OpenSearch, embedding, reranker, and LLM checks are opt-in integration checks. Evaluation cases and contracts live under `eval/`; generated reports and private run artifacts are ignored.

## Data and licensing

Application code is licensed under Apache-2.0. See `NOTICE` and `THIRD_PARTY_DATA.md` for provenance and redistribution boundaries. External catalogs and generated indexes must be obtained or generated separately under their terms.

## Limitations and roadmap

This is a reference implementation, not a production marketplace: prices, delivery, tax, carrier rules, and availability are not live guarantees. Authentication, payment, and multi-region deployment are outside the current scope. Future work may add per-session serialization, multi-instance coordination, richer synthetic fixtures, and production observability.
