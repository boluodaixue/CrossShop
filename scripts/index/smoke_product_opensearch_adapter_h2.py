"""Run one real H2 adapter Hybrid request against each frozen platform index."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog.opensearch_product_h1 import (
    INDEX_NAMES,
    RRF_PIPELINE_NAME,
    hybrid_query,
)
from app.infrastructure.vector.opensearch_product_index import OpenSearchProductIndex
from scripts.index.product_opensearch_common import BgeM3Encoder, OpenSearchClient

QUERIES = {
    "crossshop_reference": "降噪耳机",
    "reference_seed": "轻便旅行背包",
    "amazon": "noise cancelling headphones",
}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    version = OpenSearchClient(args.endpoint).request("GET", "/")["version"]["number"]
    if version != "2.19.1":
        raise RuntimeError(f"OpenSearch 2.19.1 required, got {version}")
    encoder = BgeM3Encoder(
        args.model,
        args.device,
        args.max_seq_length,
        args.local_files_only,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "opensearch_version": version,
        "pipeline": RRF_PIPELINE_NAME,
        "platforms": {},
    }

    for platform, query in QUERIES.items():
        vector = encoder.encode([query])[0]
        index_name = INDEX_NAMES[platform]
        request = {
            "method": "POST",
            "path": f"/{index_name}/_search?search_pipeline={RRF_PIPELINE_NAME}",
            "body": hybrid_query(query, vector, size=args.size),
        }
        adapter = OpenSearchProductIndex(args.endpoint, index_name)
        try:
            hits = await adapter.search(
                query=query,
                embedding=vector,
                top_n=args.size,
            )
        finally:
            await adapter.close()
        if not hits:
            raise RuntimeError(f"{platform} H2 adapter returned no Product hits")
        result = {
            "query": query,
            "index": index_name,
            "hybrid_request_count": 1,
            "hit_count": len(hits),
            "hits": [
                {"product_id": hit.product_id, "score": hit.score} for hit in hits
            ],
        }
        (args.output_root / f"{platform}-hybrid-request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_root / f"{platform}-hybrid-result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["platforms"][platform] = result

    (args.output_root / "adapter-smoke-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("CROSSSHOP_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"),
    )
    parser.add_argument(
        "--model", default=os.getenv("BGE_M3_MODEL", "D:/models/bge-m3")
    )
    parser.add_argument("--device", default=os.getenv("BGE_M3_DEVICE", "cuda"))
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "h2-opensearch",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
