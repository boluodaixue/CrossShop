"""Fully compare indexed H1 identity/facts with every existing H0 SearchUnit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog.opensearch_product_h1 import (
    EMBEDDING_VERSION,
    VECTOR_DIMENSION,
)
from scripts.index.build_product_opensearch import PARTITIONS
from scripts.index.product_opensearch_common import OpenSearchClient

AUDIT_FIELDS = (
    "product_id",
    "sku_id",
    "search_unit_id",
    "stock",
    "ships_to",
    "price_minor",
    "currency",
    "brand",
    "category",
    "locale",
    "platform",
    "embedding_version",
)


def audit_batch(
    client: OpenSearchClient,
    index_name: str,
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    ids = [row["search_unit_id"] for row in rows]
    response = client.request(
        "POST",
        f"/{index_name}/_mget",
        {
            "docs": [
                {"_id": document_id, "_source": list(AUDIT_FIELDS)}
                for document_id in ids
            ]
        },
    )
    documents = response["docs"]
    if len(documents) != len(rows):
        raise RuntimeError("_mget returned a different document count")
    checked = 0
    first_id: dict[str, Any] = {}
    for expected, actual in zip(rows, documents):
        document_id = expected["search_unit_id"]
        if not actual.get("found"):
            raise RuntimeError(f"missing indexed document: {document_id}")
        if actual["_id"] != document_id or actual["_source"]["search_unit_id"] != document_id:
            raise RuntimeError(f"identity mismatch: {document_id}")
        for field in AUDIT_FIELDS:
            expected_value = EMBEDDING_VERSION if field == "embedding_version" else expected[field]
            if actual["_source"].get(field) != expected_value:
                raise RuntimeError(
                    f"{document_id} field {field}: indexed={actual['_source'].get(field)!r}, source={expected_value!r}"
                )
        checked += 1
        if not first_id:
            first_id = {"index": index_name, "id": document_id}
    return checked, first_id


def audit_vector(client: OpenSearchClient, sample: dict[str, Any]) -> None:
    response = client.request(
        "GET",
        f"/{sample['index']}/_doc/"
        f"{urllib.parse.quote(sample['id'], safe='')}?_source=content_vector",
    )
    vector = response["_source"]["content_vector"]
    if len(vector) != VECTOR_DIMENSION:
        raise RuntimeError(f"wrong vector dimension for {sample['id']}: {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise RuntimeError(f"non-finite vector for {sample['id']}")
    norm = math.sqrt(sum(value * value for value in vector))
    if not 0.99 <= norm <= 1.01:
        raise RuntimeError(f"non-normalized vector for {sample['id']}: {norm}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "catalogs-v2")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "opensearch-products-v1" / "audit-report.json")
    parser.add_argument("--endpoint", default=os.getenv("GLOBEX_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    client = OpenSearchClient(args.endpoint)
    report: dict[str, Any] = {"executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "partitions": {}, "indexes": {}}
    samples: list[dict[str, Any]] = []

    for partition in PARTITIONS:
        checked = 0
        batch: list[dict[str, Any]] = []
        source_path = args.catalog_root / partition.relative_path
        with source_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                batch.append(json.loads(line))
                if len(batch) >= args.batch_size:
                    count, sample = audit_batch(client, partition.index_name, batch)
                    checked += count
                    if sample and not any(s["index"] == sample["index"] for s in samples):
                        samples.append(sample)
                    batch = []
            if batch:
                count, sample = audit_batch(client, partition.index_name, batch)
                checked += count
                if sample and not any(s["index"] == sample["index"] for s in samples):
                    samples.append(sample)
        report["partitions"][partition.partition_id] = {"source_records": checked, "matched_records": checked}
        print(f"{partition.partition_id}: {checked:,} records match", flush=True)

    for sample in samples:
        audit_vector(client, sample)
    for index_name in sorted({partition.index_name for partition in PARTITIONS}):
        report["indexes"][index_name] = {"document_count": int(client.request("GET", f"/{index_name}/_count")["count"])}
    report["vector_samples"] = samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
