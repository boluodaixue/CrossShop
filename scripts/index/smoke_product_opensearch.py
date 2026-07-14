"""Run exactly one ANN, BM25, and Hybrid+RRF smoke query per H1 platform."""

from __future__ import annotations

import argparse
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
    ann_query,
    bm25_query,
    hybrid_query,
)
from scripts.index.product_opensearch_common import (
    BgeM3Encoder,
    OpenSearchClient,
    search_path,
)

QUERIES = {
    "crossshop_reference": "降噪耳机",
    "reference_seed": "轻便旅行背包",
    "amazon": "noise cancelling headphones",
}


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    def compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
        source = hit["_source"]
        skus = source["skus"]
        return {
            "_id": hit["_id"],
            "_score": hit.get("_score"),
            "product_id": source["product_id"],
            "title": source["title"],
            "locale": source["locale"],
            "sku_count": len(skus),
            "in_stock_sku_count": sum(sku["stock"] > 0 for sku in skus),
        }

    return {
        "took": response.get("took"),
        "timed_out": response.get("timed_out"),
        "total": response["hits"]["total"],
        "hits": [compact_hit(hit) for hit in response["hits"]["hits"]],
    }


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
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "opensearch-products-v1"
        / "smoke",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    client = OpenSearchClient(args.endpoint)
    encoder = BgeM3Encoder(
        args.model, args.device, args.max_seq_length, args.local_files_only
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "opensearch_version": client.request("GET", "/")["version"]["number"],
        "pipeline": RRF_PIPELINE_NAME,
        "platforms": {},
    }

    for platform, query_text in QUERIES.items():
        vector = encoder.encode([query_text])[0]
        index_name = INDEX_NAMES[platform]
        requests = {
            "ann": {
                "method": "POST",
                "path": search_path(index_name),
                "body": ann_query(vector, size=args.size),
            },
            "bm25": {
                "method": "POST",
                "path": search_path(index_name),
                "body": bm25_query(query_text, size=args.size),
            },
            "hybrid_rrf": {
                "method": "POST",
                "path": search_path(index_name, RRF_PIPELINE_NAME),
                "body": hybrid_query(query_text, vector, size=args.size),
            },
        }
        platform_result: dict[str, Any] = {
            "query": query_text,
            "requests": {},
            "responses": {},
        }
        for mode in ("ann", "bm25", "hybrid_rrf"):
            request = requests[mode]
            response = client.request(
                request["method"], request["path"], request["body"]
            )
            compact = compact_response(response)
            if compact["timed_out"]:
                raise RuntimeError(f"{platform}/{mode} timed out")
            if any(hit["in_stock_sku_count"] == 0 for hit in compact["hits"]):
                raise RuntimeError(
                    f"{platform}/{mode} returned a Product without an in-stock SKU"
                )
            # Keep requests reproducible without duplicating 1024 floats in the summary.
            request_file = args.output_root / f"{platform}-{mode}-request.json"
            response_file = args.output_root / f"{platform}-{mode}-response.json"
            request_file.write_text(
                json.dumps(request, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            response_file.write_text(
                json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            platform_result["requests"][mode] = request_file.name
            platform_result["responses"][mode] = response_file.name
            platform_result.setdefault("hit_counts", {})[mode] = len(compact["hits"])
        report["platforms"][platform] = platform_result

    (args.output_root / "smoke-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
