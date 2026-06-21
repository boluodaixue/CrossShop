"""Opt-in real OpenSearch integration test for the H3 manual composition."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pytest

from scripts.index.smoke_catalog_search_h3 import PLATFORM_SPECS, run


def test_real_h3_catalog_search_composition(tmp_path: Path) -> None:
    if os.getenv("GLOBEX_RUN_H3_OPENSEARCH_INTEGRATION") != "1":
        pytest.skip("set GLOBEX_RUN_H3_OPENSEARCH_INTEGRATION=1 to run")

    root = Path(__file__).resolve().parents[1]
    report = asyncio.run(
        run(
            argparse.Namespace(
                endpoint=os.getenv(
                    "GLOBEX_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"
                ),
                catalog_root=root / "data" / "processed" / "catalogs-v2",
                h2_artifact_root=root / "artifacts" / "h2-opensearch",
                output_root=tmp_path / "h3-integration",
                timeout_seconds=30.0,
            )
        )
    )

    assert report["repository"]["load_count"] == 1
    assert report["repository"]["product_count"] == 45_286
    assert report["repository"]["sku_count"] == 261_369
    assert set(report["platforms"]) == set(PLATFORM_SPECS)
    for platform in report["platforms"].values():
        assert platform["hybrid_request_count"] == 1
        assert platform["embedder_call_count"] == 1
        assert platform["query_vector_dimension"] == 1024
        assert platform["reranker_call_count"] == 1
        assert platform["reranked_document_count"] == platform["total_candidates"]
        assert 0 < platform["hit_count"] <= 5
        assert len(platform["product_ids"]) == len(set(platform["product_ids"]))
