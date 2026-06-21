"""Run the H3 catalog flow against a real HTTP Cross-Encoder reranker.

The script reuses the recorded 1024-dimensional H2 BGE-M3 query vectors so
the Query encoder and reranker never need to share GPU memory.  Each platform
performs exactly one OpenSearch Hybrid request and sends only its eligible
Product documents to the production ``HttpReranker`` once.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.catalog.opensearch_product_h1 import INDEX_NAMES, RRF_PIPELINE_NAME
from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.infrastructure.persistence.jsonl_product_repository import (
    JsonlProductRepository,
)
from app.infrastructure.rerank.http_reranker import HttpReranker
from app.infrastructure.settings import Settings
from app.infrastructure.vector.opensearch_product_index import (
    OpenSearchProductIndex,
)
from scripts.index.smoke_catalog_search_h3 import (
    PLATFORM_SPECS,
    RecordedQueryEmbedder,
    _assert_result_semantics,
    _load_recorded_vector,
    _spec_dict,
)


@dataclass(frozen=True)
class _HttpRerankerSettings:
    reranker_base_url: str
    reranker_model: str


class RecordingOpenSearchProductIndex(OpenSearchProductIndex):
    """Count the real H2 adapter calls and retain the exact RRF hit order."""

    def __init__(self, endpoint: str, index_name: str, timeout_seconds: float) -> None:
        super().__init__(endpoint, index_name, timeout_seconds=timeout_seconds)
        self.search_calls = 0
        self.hits: list[VectorHit] = []

    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        self.search_calls += 1
        hits = await super().search(
            query=query,
            embedding=embedding,
            top_n=top_n,
        )
        self.hits = list(hits)
        return hits


class RecordingHttpReranker:
    """Record and validate one real ``HttpReranker`` call per platform."""

    def __init__(self, delegate: HttpReranker) -> None:
        self._delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        started = time.perf_counter()
        scores = await self._delegate.rerank(query, documents)
        elapsed_seconds = time.perf_counter() - started
        if len(scores) != len(documents):
            raise RuntimeError(
                "real reranker score count mismatch: "
                f"scores={len(scores)}, documents={len(documents)}"
            )
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("real reranker returned a non-finite score")
        self.calls.append(
            {
                "query": query,
                "documents": list(documents),
                "scores": list(scores),
                "elapsed_seconds": elapsed_seconds,
            }
        )
        return scores


def _http_reranker(
    *, base_url: str, model: str, timeout_seconds: float
) -> HttpReranker:
    settings = _HttpRerankerSettings(
        reranker_base_url=base_url,
        reranker_model=model,
    )
    return HttpReranker(
        cast(Settings, settings),
        timeout_seconds=timeout_seconds,
    )


async def _warm_reranker(
    reranker: HttpReranker,
    *,
    query: str,
) -> dict[str, Any]:
    document = "Globex reranker warmup product document"
    started = time.perf_counter()
    scores = await reranker.rerank(query, [document])
    elapsed_seconds = time.perf_counter() - started
    if len(scores) != 1 or not math.isfinite(scores[0]):
        raise RuntimeError(f"reranker warmup returned invalid scores: {scores!r}")
    return {
        "query": query,
        "document_count": 1,
        "score": scores[0],
        "elapsed_seconds": elapsed_seconds,
    }


async def _reranker_health(
    *, base_url: str, timeout_seconds: float, expected_model: str
) -> dict[str, Any]:
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        response = await client.get("/health")
        response.raise_for_status()
        health = response.json()
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise RuntimeError(f"reranker health response is not ready: {health!r}")
    if health.get("model") != expected_model:
        raise RuntimeError(
            "reranker health model mismatch: "
            f"expected={expected_model!r}, actual={health.get('model')!r}"
        )
    return health


async def _eligible_rrf_candidates(
    *,
    usecase: CatalogSearchUseCase,
    repository: JsonlProductRepository,
    hits: list[VectorHit],
    spec,
) -> list[tuple[float, Any]]:
    product_ids = [hit.product_id for hit in hits]
    products = await repository.find_by_ids(product_ids)
    if [product.product_id for product in products] != product_ids:
        raise RuntimeError("repository did not preserve the recorded RRF hit order")
    scored = [(hit.score, product) for hit, product in zip(hits, products, strict=True)]
    eligible, _ = usecase._apply_constraints(scored, spec)
    return eligible


def _write_json(path: Path, value: Any) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text(path: Path, value: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(value, encoding="utf-8")
    os.replace(temporary_path, path)


def _validated_output_root(path: Path) -> Path:
    output_root = path.resolve()
    if output_root == PROJECT_ROOT or PROJECT_ROOT not in output_root.parents:
        raise ValueError(
            f"--output-root must be a directory inside the project root: {PROJECT_ROOT}"
        )
    return output_root


def _render_readme(report: dict[str, Any], args: argparse.Namespace) -> str:
    reranker = report["reranker"]
    health = reranker["health"]
    repository = report["repository"]
    platform_rows = []
    for platform in PLATFORM_SPECS:
        result = report["platforms"][platform]
        changed = "是" if result["ranking_changed"] else "否"
        platform_rows.append(
            f"| {platform} | {result['hybrid_request_count']} | "
            f"{result['reranker_call_count']} | {result['total_candidates']} | "
            f"{result['hit_count']} | {result['warm_rerank_seconds']} 秒 | {changed} |"
        )
    rows = "\n".join(platform_rows)
    return f"""# H3 真实 Cross-Encoder Reranker 验收

本目录由 `scripts/index/smoke_catalog_search_h3_real_reranker.py` 生成真实验收证据。
验收复用 H2 保存的 1024 维 BGE-M3 Query 向量，避免 Query 模型和 Reranker 同时占用 GPU。

## 本次实测命令

```powershell
.venv\\Scripts\\python.exe scripts/index/smoke_catalog_search_h3_real_reranker.py `
  --reranker-url {args.reranker_url.rstrip("/")} `
  --reranker-model {args.reranker_model}
```

模型健康信息来自同一服务的 `/health`：

- 模型：`{health["model"]}`
- 本地目录：`{health["model_path"]}`
- 设备：`{health["device"]}`
- 精度：`{health["dtype"]}`
- `max_length={health["max_length"]}`
- `micro_batch_size={health["micro_batch_size"]}`
- 分数模式：`{health["score_mode"]}`

| 平台 | Hybrid 请求 | HttpReranker 请求 | 合格 Product | 最终 hits | warm 后端到端精排耗时 | 排序变化 |
|---|---:|---:|---:|---:|---:|---|
{rows}

模型额外 warmup 请求耗时为 {reranker["warmup"]["elapsed_seconds"]} 秒；全量 Repository 单次装载 {repository["product_count"]:,} Product / {repository["sku_count"]:,} SKU，耗时 {repository["load_seconds"]} 秒。以上均来自 `acceptance-report.json` 所记录的同一次运行。端到端精排耗时包含本地 HTTP 调用，不等同于纯 GPU 推理时间。

## 可复现文件

- `acceptance-report.json`：模型健康参数，以及三平台 Hybrid 次数、RRF 顺序、真实分数、精排顺序和 warm 后延迟。
- `<platform>-rerank-request.json`：现有 `HttpReranker` 实际使用的模型、query 和候选文本。
- `<platform>-rerank-response.json`：候选 Product 对应的真实 relevance score。
- `<platform>-usecase-result.json`：通过硬约束、精排和 top_k 后的最终 UseCase 结果。

opt-in 测试命令：

```powershell
$env:GLOBEX_RUN_H3_REAL_RERANKER_INTEGRATION = "1"
$env:RERANKER_BASE_URL = "{args.reranker_url.rstrip("/")}"
$env:RERANKER_MODEL = "{args.reranker_model}"
.venv\\Scripts\\python.exe -m pytest tests/test_h3_real_reranker_integration.py -q
```

本验收只证明真实 Cross-Encoder 已被正确调用且参与排序。没有标注集，因此不声明 NDCG、Recall 或语义质量提升。
"""


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _validated_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.reranker_url.strip():
        raise ValueError("--reranker-url is required")
    if not args.reranker_model.strip():
        raise ValueError("--reranker-model is required")

    async with httpx.AsyncClient(
        base_url=args.opensearch_endpoint.rstrip("/"),
        timeout=args.opensearch_timeout_seconds,
    ) as client:
        response = await client.get("/")
        response.raise_for_status()
        opensearch_version = response.json()["version"]["number"]
    if opensearch_version != "2.19.1":
        raise RuntimeError(f"OpenSearch 2.19.1 required, got {opensearch_version}")

    reranker_client = _http_reranker(
        base_url=args.reranker_url,
        model=args.reranker_model,
        timeout_seconds=args.reranker_timeout_seconds,
    )
    try:
        reranker_health = await _reranker_health(
            base_url=args.reranker_url,
            timeout_seconds=args.reranker_timeout_seconds,
            expected_model=args.reranker_model,
        )
        warmup = await _warm_reranker(
            reranker_client,
            query=args.warmup_query,
        )
    except Exception as error:
        raise RuntimeError(
            "real reranker warmup failed; verify --reranker-url, "
            "--reranker-model, and the local model service"
        ) from error

    load_started = time.perf_counter()
    repository = JsonlProductRepository(args.catalog_root)
    repository_load_seconds = time.perf_counter() - load_started
    if repository.product_count != 45_286 or repository.sku_count != 261_369:
        raise RuntimeError(
            "unexpected full catalog counts: "
            f"products={repository.product_count}, skus={repository.sku_count}"
        )

    report: dict[str, Any] = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "opensearch_version": opensearch_version,
        "pipeline": RRF_PIPELINE_NAME,
        "reranker": {
            "url": args.reranker_url.rstrip("/"),
            "path": "/rerank",
            "model": args.reranker_model,
            "timeout_seconds": args.reranker_timeout_seconds,
            "health": reranker_health,
            "warmup": warmup,
        },
        "repository": {
            "catalog_root": str(args.catalog_root.resolve()),
            "load_count": 1,
            "load_seconds": repository_load_seconds,
            "product_count": repository.product_count,
            "sku_count": repository.sku_count,
        },
        "platforms": {},
        "quality_claim": (
            "No ranking-quality improvement is claimed without a labelled dataset."
        ),
    }

    for platform, spec in PLATFORM_SPECS.items():
        vector_path = (
            args.h2_artifact_root / f"{platform}-hybrid-request.json"
        ).resolve()
        vector = _load_recorded_vector(vector_path, spec.normalized_query)
        vector_sha256 = hashlib.sha256(vector_path.read_bytes()).hexdigest()
        embedder = RecordedQueryEmbedder(spec.normalized_query, vector)
        reranker = RecordingHttpReranker(reranker_client)
        index = RecordingOpenSearchProductIndex(
            args.opensearch_endpoint,
            INDEX_NAMES[platform],
            args.opensearch_timeout_seconds,
        )
        usecase = CatalogSearchUseCase(
            repository,
            embedder=embedder,
            vector_index=index,
            reranker=reranker,
        )
        try:
            result = await usecase.execute(spec)
        finally:
            await index.close()

        if index.search_calls != 1:
            raise RuntimeError(
                f"{platform} must execute exactly one Hybrid search, "
                f"actual={index.search_calls}"
            )
        if embedder.calls != 1:
            raise RuntimeError(f"{platform} embedder calls={embedder.calls}")
        if len(reranker.calls) != 1:
            raise RuntimeError(
                f"{platform} must call HttpReranker once, actual={len(reranker.calls)}"
            )
        if result["rerank_applied"] is not True:
            raise RuntimeError(f"{platform} rerank_applied is not true")
        if result["recall_strategy"] != "embedding_rerank":
            raise RuntimeError(
                f"{platform} unexpected strategy: {result['recall_strategy']}"
            )

        eligible = await _eligible_rrf_candidates(
            usecase=usecase,
            repository=repository,
            hits=index.hits,
            spec=spec,
        )
        call = reranker.calls[0]
        expected_documents = [product.searchable_text() for _, product in eligible]
        if call["documents"] != expected_documents:
            raise RuntimeError(
                f"{platform} HttpReranker did not receive exactly eligible Products"
            )
        if len(call["scores"]) != len(eligible):
            raise RuntimeError(f"{platform} score count mismatch")

        eligible_rrf_order = [product.product_id for _, product in eligible]
        reranked = sorted(
            zip(call["scores"], eligible, strict=True),
            key=lambda item: item[0],
            reverse=True,
        )
        reranked_order = [product.product_id for _, (_, product) in reranked]
        result_order = [hit["product_id"] for hit in result["hits"]]
        if result_order != reranked_order[: spec.top_k]:
            raise RuntimeError(
                f"{platform} final hits do not match real reranker order"
            )
        semantics = await _assert_result_semantics(repository, spec, result)

        request_artifact = output_root / f"{platform}-rerank-request.json"
        response_artifact = output_root / f"{platform}-rerank-response.json"
        result_artifact = output_root / f"{platform}-usecase-result.json"
        _write_json(
            request_artifact,
            {
                "method": "POST",
                "path": "/rerank",
                "payload": {
                    "model": args.reranker_model,
                    "query": call["query"],
                    "documents": call["documents"],
                },
            },
        )
        _write_json(
            response_artifact,
            {
                "elapsed_seconds": call["elapsed_seconds"],
                "results": [
                    {
                        "index": index_value,
                        "product_id": product.product_id,
                        "relevance_score": score,
                    }
                    for index_value, (score, (_, product)) in enumerate(
                        zip(call["scores"], eligible, strict=True)
                    )
                ],
            },
        )
        _write_json(result_artifact, result)

        report["platforms"][platform] = {
            "index": INDEX_NAMES[platform],
            "spec": _spec_dict(spec),
            "hybrid_request_count": index.search_calls,
            "embedder_call_count": embedder.calls,
            "query_vector_dimension": len(vector),
            "query_vector_source": str(vector_path.relative_to(PROJECT_ROOT)),
            "query_vector_source_sha256": vector_sha256,
            "raw_rrf_order": [hit.product_id for hit in index.hits],
            "eligible_rrf_order": eligible_rrf_order,
            "reranker_call_count": len(reranker.calls),
            "reranked_document_count": len(call["documents"]),
            "reranker_scores": call["scores"],
            "reranked_order": reranked_order,
            "ranking_changed": eligible_rrf_order != reranked_order,
            "warm_rerank_seconds": call["elapsed_seconds"],
            "rerank_request_artifact": str(request_artifact.relative_to(PROJECT_ROOT)),
            "rerank_response_artifact": str(
                response_artifact.relative_to(PROJECT_ROOT)
            ),
            "result_artifact": str(result_artifact.relative_to(PROJECT_ROOT)),
            "total_candidates": result["total_candidates"],
            "rerank_applied": result["rerank_applied"],
            **semantics,
        }

    report_path = output_root / "acceptance-report.json"
    _write_json(report_path, report)
    _write_text(output_root / "README.md", _render_readme(report, args))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opensearch-endpoint",
        default=os.getenv("GLOBEX_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"),
    )
    parser.add_argument(
        "--reranker-url",
        default=os.getenv("RERANKER_BASE_URL", ""),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "catalogs-v2",
    )
    parser.add_argument(
        "--h2-artifact-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "h2-opensearch",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "h3-real-reranker",
    )
    parser.add_argument("--opensearch-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--reranker-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--warmup-query", default="warmup")
    return parser


def main() -> None:
    report = asyncio.run(run(build_parser().parse_args()))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
