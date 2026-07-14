"""Opt-in H3 integration test using a real HTTP Cross-Encoder reranker."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from pathlib import Path

import pytest

from scripts.index.smoke_catalog_search_h3 import PLATFORM_SPECS
from scripts.index.smoke_catalog_search_h3_real_reranker import run


def test_real_h3_rejects_output_root_outside_project() -> None:
    outside_project = Path(__file__).resolve().parents[2] / "outside-h3-evidence"
    with pytest.raises(ValueError, match="--output-root must be a directory inside"):
        asyncio.run(run(argparse.Namespace(output_root=outside_project)))


def test_real_h3_rejects_project_root_as_output_root() -> None:
    project_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="--output-root must be a directory inside"):
        asyncio.run(run(argparse.Namespace(output_root=project_root)))


def test_real_h3_cross_encoder_reranker() -> None:
    if os.getenv("CROSSSHOP_RUN_H3_REAL_RERANKER_INTEGRATION") != "1":
        pytest.skip(
            "set CROSSSHOP_RUN_H3_REAL_RERANKER_INTEGRATION=1 and start the "
            "local reranker service to run"
        )
    reranker_url = os.getenv("RERANKER_BASE_URL", "")
    if not reranker_url:
        pytest.fail("RERANKER_BASE_URL is required for the opt-in real test")

    root = Path(__file__).resolve().parents[1]
    report = asyncio.run(
        run(
            argparse.Namespace(
                opensearch_endpoint=os.getenv(
                    "CROSSSHOP_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"
                ),
                reranker_url=reranker_url,
                reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                catalog_root=root / "data" / "processed" / "catalogs-v2",
                h2_artifact_root=root / "artifacts" / "h2-opensearch",
                output_root=root / "artifacts" / "h3-real-reranker",
                opensearch_timeout_seconds=30.0,
                reranker_timeout_seconds=120.0,
                warmup_query="warmup",
            )
        )
    )

    assert report["repository"]["load_count"] == 1
    assert report["repository"]["product_count"] == 45_286
    assert report["repository"]["sku_count"] == 261_369
    assert report["reranker"]["health"] == {
        "status": "ok",
        "model": report["reranker"]["model"],
        "model_path": r"D:\models\bge-reranker-v2-m3",
        "device": "cuda:0",
        "dtype": "float16",
        "max_length": 256,
        "micro_batch_size": 2,
        "score_mode": "raw_logit",
    }
    assert set(report["platforms"]) == set(PLATFORM_SPECS)
    for platform in report["platforms"].values():
        assert platform["hybrid_request_count"] == 1
        assert platform["embedder_call_count"] == 1
        assert platform["query_vector_dimension"] == 1024
        assert platform["reranker_call_count"] == 1
        assert platform["reranked_document_count"] == platform["total_candidates"]
        assert len(platform["reranker_scores"]) == platform["total_candidates"]
        assert all(math.isfinite(score) for score in platform["reranker_scores"])
        assert platform["rerank_applied"] is True
        assert 0 < platform["hit_count"] <= 5
        assert len(platform["product_ids"]) == len(set(platform["product_ids"]))
        assert platform["warm_rerank_seconds"] >= 0
