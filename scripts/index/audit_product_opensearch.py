"""Fully compare indexed H1 Product/SKU facts with every H0 Product."""

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
    INDEX_FIELDS,
    RRF_PIPELINE_NAME,
    VECTOR_DIMENSION,
    product_index_body,
    validate_product,
)
from scripts.index.build_product_opensearch import PARTITIONS
from scripts.index.product_opensearch_common import OpenSearchClient

AUDIT_FIELDS = tuple(sorted(INDEX_FIELDS - {"content_vector"}))


def expected_document(product: dict[str, Any], locale: str | None) -> dict[str, Any]:
    return {**product, "locale": locale, "embedding_version": EMBEDDING_VERSION}


def audit_batch(
    client: OpenSearchClient,
    index_name: str,
    locale: str | None,
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    ids = [row["product_id"] for row in rows]
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
    first_id: dict[str, Any] = {}
    for product, actual in zip(rows, documents):
        product_id = product["product_id"]
        if not actual.get("found"):
            raise RuntimeError(f"missing indexed Product: {product_id}")
        if actual["_id"] != product_id:
            raise RuntimeError(f"identity mismatch: {product_id}")
        expected = expected_document(product, locale)
        if actual["_source"] != expected:
            differing = sorted(
                key for key in expected if actual["_source"].get(key) != expected[key]
            )
            raise RuntimeError(f"{product_id} indexed fact mismatch: {differing}")
        if not first_id:
            first_id = {"index": index_name, "id": product_id}
    return len(rows), first_id


def audit_vector(client: OpenSearchClient, sample: dict[str, Any]) -> dict[str, Any]:
    response = client.request(
        "GET",
        f"/{sample['index']}/_doc/{urllib.parse.quote(sample['id'], safe='')}?_source=content_vector",
    )
    vector = response["_source"]["content_vector"]
    if len(vector) != VECTOR_DIMENSION:
        raise RuntimeError(f"wrong vector dimension for {sample['id']}: {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise RuntimeError(f"non-finite vector for {sample['id']}")
    norm = math.sqrt(sum(value * value for value in vector))
    if not 0.99 <= norm <= 1.01:
        raise RuntimeError(f"non-normalized vector for {sample['id']}: {norm}")
    return {**sample, "dimension": len(vector), "l2_norm": norm}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "catalogs-v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "processed"
        / "opensearch-products-v1"
        / "audit-report.json",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("GLOBEX_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"),
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    client = OpenSearchClient(args.endpoint)
    report: dict[str, Any] = {
        "executed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "partitions": {},
        "indexes": {},
    }
    samples: list[dict[str, Any]] = []
    product_ids: set[str] = set()
    sku_ids: set[str] = set()
    expected_counts: dict[str, int] = {}
    sku_count = 0

    for partition in PARTITIONS:
        checked = 0
        batch: list[dict[str, Any]] = []
        source_path = args.catalog_root / partition.relative_path
        with source_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                product = json.loads(line)
                validate_product(product)
                product_id = product["product_id"]
                if product_id in product_ids:
                    raise RuntimeError(f"duplicate product_id: {product_id}")
                product_ids.add(product_id)
                for sku in product["skus"]:
                    sku_count += 1
                    if sku["sku_id"] in sku_ids:
                        raise RuntimeError(f"duplicate sku_id: {sku['sku_id']}")
                    sku_ids.add(sku["sku_id"])
                batch.append(product)
                if len(batch) >= args.batch_size:
                    count, sample = audit_batch(
                        client, partition.index_name, partition.locale, batch
                    )
                    checked += count
                    if sample and not any(
                        item["index"] == sample["index"] for item in samples
                    ):
                        samples.append(sample)
                    batch = []
            if batch:
                count, sample = audit_batch(
                    client, partition.index_name, partition.locale, batch
                )
                checked += count
                if sample and not any(
                    item["index"] == sample["index"] for item in samples
                ):
                    samples.append(sample)
        expected_counts[partition.index_name] = (
            expected_counts.get(partition.index_name, 0) + checked
        )
        report["partitions"][partition.partition_id] = {
            "locale": partition.locale,
            "source_products": checked,
            "matched_products": checked,
        }
        print(f"{partition.partition_id}: {checked:,} Products match", flush=True)

    expected_mapping = product_index_body()["mappings"]
    vector_samples = [audit_vector(client, sample) for sample in samples]
    for index_name, expected_count in sorted(expected_counts.items()):
        actual_count = int(client.request("GET", f"/{index_name}/_count")["count"])
        if actual_count != expected_count:
            raise RuntimeError(
                f"{index_name} count={actual_count}, expected={expected_count}"
            )
        actual_mapping = client.request("GET", f"/{index_name}/_mapping")[index_name][
            "mappings"
        ]
        if actual_mapping != expected_mapping:
            raise RuntimeError(f"{index_name} mapping differs from frozen mapping")
        missing_vector = int(
            client.request(
                "POST",
                f"/{index_name}/_count",
                {
                    "query": {
                        "bool": {"must_not": [{"exists": {"field": "content_vector"}}]}
                    }
                },
            )["count"]
        )
        if missing_vector:
            raise RuntimeError(
                f"{index_name} has {missing_vector} Products without vectors"
            )
        report["indexes"][index_name] = {
            "document_count": actual_count,
            "expected_count": expected_count,
            "missing_vectors": missing_vector,
            "mapping_vector_dimension": VECTOR_DIMENSION,
        }
    report["identity"] = {
        "unique_product_ids": len(product_ids),
        "unique_sku_ids": len(sku_ids),
        "sku_records": sku_count,
    }
    report["vector_samples"] = vector_samples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.artifact_root:
        args.artifact_root.mkdir(parents=True, exist_ok=True)
        snapshots = {
            "audit-report.json": report,
            "mapping-actual.json": client.request(
                "GET", "/globex-products-*-v1/_mapping"
            ),
            "rrf-pipeline-actual.json": client.request(
                "GET", f"/_search/pipeline/{RRF_PIPELINE_NAME}"
            ),
            "index-stats-actual.json": client.request(
                "GET",
                "/_cat/indices/globex-products-*-v1?format=json&bytes=b&h=health,index,docs.count,docs.deleted,store.size",
            ),
        }
        for filename, payload in snapshots.items():
            (args.artifact_root / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
