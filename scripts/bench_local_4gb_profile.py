"""Reproducible local 4GB device split benchmark.

The benchmark deliberately keeps the production model, index, and Top-100
candidate depth unchanged.  It compares CPU/GPU query-worker overlap and
tests GPU reranker batch sizes 4/8/16 with three independent runs before
timing the selected combined query→ANN→rerank path three times.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from globex_agent.infrastructure.embedding.bge_m3_embedding import (
    SubprocessBgeM3EmbeddingClient,
)
from globex_agent.infrastructure.persistence.sqlite_item_repository import (
    SqliteItemRepository,
)
from globex_agent.infrastructure.recall.reranker import SubprocessCrossEncoderReranker
from globex_agent.infrastructure.recall.search_document import standard_item_search_text
from globex_agent.infrastructure.vector.faiss_product_index import (
    PartitionedFaissProductIndex,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "D:/models/bge-m3"
RERANKER_MODEL = "D:/models/bge-reranker-v2-m3"
CUDA_PYTHON = Path("C:/Anaconda/envs/blog_04/python.exe")
CPU_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
QUERIES = ("头戴式主动降噪耳机", "乳胶枕 护颈", "平板电脑支架")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "eval" / "local_4gb_profile.json",
    )
    parser.add_argument("--reranker-python", type=Path, default=CUDA_PYTHON)
    args = parser.parse_args()

    if not CPU_PYTHON.is_file() or not args.reranker_python.is_file():
        raise SystemExit("configured worker Python executable is missing")

    cpu_query = SubprocessBgeM3EmbeddingClient(
        CPU_PYTHON,
        model_name=MODEL,
        device="cpu",
        batch_size=4,
        max_seq_length=512,
        local_files_only=True,
    )
    gpu_query = SubprocessBgeM3EmbeddingClient(
        args.reranker_python,
        model_name=MODEL,
        device="cuda:0",
        batch_size=4,
        max_seq_length=512,
        local_files_only=True,
    )
    index = _load_index()
    try:
        cpu_query.ensure_ready()
        gpu_query.ensure_ready()
        cpu_query.warmup()
        gpu_query.warmup()
        overlap = await _compare_query_top100(cpu_query, gpu_query, index)
        repo = _repository()
        gpu_hits = await index.search(await gpu_query.embed(QUERIES[0]), top_n=100)
        items = await repo.find_by_ids([hit.item_id for hit in gpu_hits])
        by_id = {item.item_id: item for item in items}
        documents = [
            standard_item_search_text(by_id[hit.item_id])
            for hit in gpu_hits
            if hit.item_id in by_id
        ]
        if len(documents) < 100:
            raise RuntimeError(f"expected 100 hydrated documents, got {len(documents)}")
        batch_results = _benchmark_batches(args.reranker_python, documents[:100])
        usable = [row for row in batch_results if not row["oom"]]
        if not usable:
            raise RuntimeError("all reranker batch sizes failed")
        selected = min(usable, key=lambda row: row["median_ms"])
        chain = await _benchmark_combined(
            args.reranker_python,
            cpu_query,
            index,
            repo,
            selected["batch_size"],
        )
        result = {
            "profile": {
                "query_embedding": {
                    "device": "cpu",
                    "precision": "fp32",
                    "worker": str(CPU_PYTHON),
                },
                "reranker": {
                    "device": "cuda:0",
                    "precision": "fp16",
                    "worker": str(args.reranker_python),
                },
                "model": MODEL,
                "reranker_model": RERANKER_MODEL,
                "max_seq_length": 512,
                "reranker_max_length": 256,
                "candidate_k": 100,
                "top_k": 10,
            },
            "query_contract": {
                "cpu_dimension": cpu_query.dimension,
                "cpu_normalized": True,
                "gpu_dimension": gpu_query.dimension,
                "top100_overlap": overlap,
            },
            "reranker_independent": batch_results,
            "selected_batch_size": selected["batch_size"],
            "combined_chain": chain,
            "queries": list(QUERIES),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        cpu_query.close()
        gpu_query.close()
        await index.close()


async def _compare_query_top100(cpu, gpu, index) -> dict:
    rows = []
    for query in QUERIES:
        cpu_hits = await index.search(await cpu.embed(query), top_n=100)
        gpu_hits = await index.search(await gpu.embed(query), top_n=100)
        cpu_ids = {hit.item_id for hit in cpu_hits}
        gpu_ids = {hit.item_id for hit in gpu_hits}
        rows.append({
            "query": query,
            "cpu_count": len(cpu_ids),
            "gpu_count": len(gpu_ids),
            "intersection": len(cpu_ids & gpu_ids),
            "overlap": round(len(cpu_ids & gpu_ids) / 100, 4),
        })
    return {"per_query": rows, "mean": round(statistics.mean(row["overlap"] for row in rows), 4)}


def _benchmark_batches(python: Path, documents: list[str]) -> list[dict]:
    rows = []
    for batch_size in (4, 8, 16):
        reranker = SubprocessCrossEncoderReranker(
            python,
            model_name=RERANKER_MODEL,
            device="cuda:0",
            batch_size=batch_size,
            max_length=256,
            use_fp16=True,
            local_files_only=True,
        )
        times = []
        error = None
        try:
            reranker.ensure_ready()
            reranker.warmup()
            for _ in range(3):
                started = time.perf_counter()
                reranker.score_texts(QUERIES[0], documents)
                times.append(round((time.perf_counter() - started) * 1000, 3))
        except Exception as exc:  # noqa: BLE001 - benchmark must record OOM/failure
            error = f"{type(exc).__name__}: {exc}"
        finally:
            reranker.close()
        rows.append({
            "batch_size": batch_size,
            "runs_ms": times,
            "median_ms": round(statistics.median(times), 3) if times else None,
            "oom": bool(
                error
                and (
                    "out of memory" in error.casefold()
                    or "oom" in error.casefold()
                )
            ),
            "error": error,
        })
    return rows


async def _benchmark_combined(python, query_client, index, repo, batch_size: int) -> dict:
    reranker = SubprocessCrossEncoderReranker(
        python,
        model_name=RERANKER_MODEL,
        device="cuda:0",
        batch_size=batch_size,
        max_length=256,
        use_fp16=True,
        local_files_only=True,
    )
    times = []
    try:
        reranker.ensure_ready()
        reranker.warmup()
        for _ in range(3):
            started = time.perf_counter()
            hits = await index.search(await query_client.embed(QUERIES[0]), top_n=100)
            items = await repo.find_by_ids([hit.item_id for hit in hits])
            by_id = {item.item_id: item for item in items}
            docs = [
                standard_item_search_text(by_id[hit.item_id])
                for hit in hits
                if hit.item_id in by_id
            ]
            scores = reranker.score_texts(QUERIES[0], docs)
            if len(scores) != len(docs):
                raise RuntimeError("reranker score count mismatch")
            times.append(round((time.perf_counter() - started) * 1000, 3))
    finally:
        reranker.close()
    return {
        "batch_size": batch_size,
        "candidate_count": 100,
        "runs_ms": times,
        "median_ms": round(statistics.median(times), 3),
    }


def _load_index() -> PartitionedFaissProductIndex:
    from globex_agent.composition import _build_vector_index

    index = _build_vector_index()
    if not isinstance(index, PartitionedFaissProductIndex):
        raise RuntimeError("formal partitioned Faiss index is not available")
    return index


def _repository() -> SqliteItemRepository:
    paths = [
        PROJECT_ROOT / "data" / "processed" / "databases" / "taobao" / "catalog.sqlite3",
        PROJECT_ROOT / "data" / "processed" / "databases" / "amazon" / "catalog.sqlite3",
    ]
    return SqliteItemRepository([path for path in paths if path.exists()])


if __name__ == "__main__":
    asyncio.run(main())
