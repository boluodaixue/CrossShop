"""Encode existing H0 Products and build three indexes with resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog.opensearch_product_h1 import (
    EMBEDDING_VERSION,
    INDEX_NAMES,
    RRF_PIPELINE_NAME,
    index_document,
    product_index_body,
    product_searchable_text,
    rrf_pipeline_body,
)
from scripts.index.product_opensearch_common import BgeM3Encoder, OpenSearchClient


@dataclass(frozen=True)
class Partition:
    partition_id: str
    relative_path: str
    index_name: str
    locale: str | None


PARTITIONS = (
    Partition(
        "globex_reference",
        "globex_reference/products.jsonl",
        INDEX_NAMES["globex_reference"],
        None,
    ),
    Partition("taobao", "taobao/products.jsonl", INDEX_NAMES["taobao"], "cn"),
    Partition("amazon_us", "amazon/us/products.jsonl", INDEX_NAMES["amazon"], "us"),
    Partition("amazon_es", "amazon/es/products.jsonl", INDEX_NAMES["amazon"], "es"),
    Partition("amazon_jp", "amazon/jp/products.jsonl", INDEX_NAMES["amazon"], "jp"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_rows(path: Path, skip: int) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line_number <= skip or not line.strip():
                continue
            yield line_number, json.loads(line)


def checkpoint_contract(
    partition: Partition, source_path: Path, model_name: str, max_seq_length: int
) -> dict[str, Any]:
    return {
        "partition_id": partition.partition_id,
        "source_path": partition.relative_path,
        "source_sha256": sha256(source_path),
        "index_name": partition.index_name,
        "locale": partition.locale,
        "embedding_version": EMBEDDING_VERSION,
        "embedding_dimension": 1024,
        "model": model_name,
        "max_seq_length": max_seq_length,
    }


def load_checkpoint(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            **contract,
            "next_line": 0,
            "docs_written": 0,
            "last_product_id": None,
            "completed": False,
        }
    state = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: (state.get(key), expected)
        for key, expected in contract.items()
        if state.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"checkpoint contract mismatch for {path}: {mismatches}")
    return state


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare(client: OpenSearchClient, reset: bool) -> None:
    version = client.request("GET", "/")["version"]["number"]
    if version != "2.19.1":
        raise RuntimeError(f"OpenSearch 2.19.1 required, got {version}")
    client.request("PUT", f"/_search/pipeline/{RRF_PIPELINE_NAME}", rrf_pipeline_body())
    for index_name in INDEX_NAMES.values():
        if reset and client.exists(f"/{index_name}"):
            client.request("DELETE", f"/{index_name}")
        if not client.exists(f"/{index_name}"):
            client.request("PUT", f"/{index_name}", product_index_body())


def build_partition(
    client: OpenSearchClient,
    encoder: BgeM3Encoder,
    partition: Partition,
    catalog_root: Path,
    state_root: Path,
    model_name: str,
    max_seq_length: int,
    batch_size: int,
    bulk_size: int,
) -> dict[str, Any]:
    source_path = catalog_root / partition.relative_path
    contract = checkpoint_contract(partition, source_path, model_name, max_seq_length)
    checkpoint_path = state_root / "checkpoints" / f"{partition.partition_id}.json"
    state = load_checkpoint(checkpoint_path, contract)
    if state["completed"]:
        print(
            f"{partition.partition_id}: already completed ({state['docs_written']:,})",
            flush=True,
        )
        return state

    pending: list[tuple[str, dict[str, Any], int]] = []
    batch_rows: list[tuple[int, dict[str, Any]]] = []
    started = time.monotonic()

    def flush_bulk() -> None:
        nonlocal pending, state
        if not pending:
            return
        client.bulk(partition.index_name, [(doc_id, doc) for doc_id, doc, _ in pending])
        last_id, _, last_line = pending[-1]
        state.update(
            next_line=last_line,
            docs_written=state["docs_written"] + len(pending),
            last_product_id=last_id,
        )
        write_checkpoint(checkpoint_path, state)
        pending = []
        if state["docs_written"] % 1000 < bulk_size:
            elapsed = time.monotonic() - started
            print(
                f"{partition.partition_id}: {state['docs_written']:,} products, {elapsed:.1f}s",
                flush=True,
            )

    def flush_encoded() -> None:
        nonlocal pending, batch_rows
        if not batch_rows:
            return
        vectors = encoder.encode(
            [product_searchable_text(row) for _, row in batch_rows]
        )
        for (line_number, row), vector in zip(batch_rows, vectors):
            product_id = row["product_id"]
            pending.append(
                (product_id, index_document(row, partition.locale, vector), line_number)
            )
        batch_rows = []
        if len(pending) >= bulk_size:
            flush_bulk()

    for line_number, row in iter_rows(source_path, int(state["next_line"])):
        batch_rows.append((line_number, row))
        if len(batch_rows) >= batch_size:
            flush_encoded()
    flush_encoded()
    flush_bulk()
    state["completed"] = True
    write_checkpoint(checkpoint_path, state)
    print(
        f"{partition.partition_id}: completed ({state['docs_written']:,})", flush=True
    )
    return state


def runtime_stats(client: OpenSearchClient) -> dict[str, Any]:
    nodes = client.request(
        "GET",
        "/_nodes/stats/jvm,fs?filter_path=nodes.*.jvm.mem.heap_used_in_bytes,nodes.*.jvm.mem.heap_max_in_bytes,nodes.*.fs.total.available_in_bytes,nodes.*.fs.total.total_in_bytes",
    )["nodes"]
    raw_indexes = client.request(
        "GET",
        "/_cat/indices/globex-products-*-v1?format=json&bytes=b&h=index,docs.count,store.size",
    )
    indexes = [
        {
            "index": row["index"],
            "lucene_docs_including_nested": int(row["docs.count"]),
            "primary_store_bytes": int(row["store.size"]),
        }
        for row in raw_indexes
    ]
    host_disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "nodes": nodes,
        "indexes": indexes,
        "host_project_drive": {
            "total_bytes": host_disk.total,
            "used_bytes": host_disk.used,
            "free_bytes": host_disk.free,
        },
    }


def finalize(client: OpenSearchClient) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index_name in INDEX_NAMES.values():
        client.request(
            "PUT", f"/{index_name}/_settings", {"index": {"refresh_interval": "1s"}}
        )
        client.request("POST", f"/{index_name}/_refresh")
        counts[index_name] = int(
            client.request("GET", f"/{index_name}/_count")["count"]
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "catalogs-v2",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "opensearch-products-v1",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("GLOBEX_OPENSEARCH_ENDPOINT", "http://127.0.0.1:9200"),
    )
    parser.add_argument(
        "--model", default=os.getenv("BGE_M3_MODEL", "D:/models/bge-m3")
    )
    parser.add_argument("--device", default=os.getenv("BGE_M3_DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bulk-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--partition",
        action="append",
        choices=tuple(p.partition_id for p in PARTITIONS),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete only the three frozen H1 indexes and old H1 checkpoints.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.bulk_size < args.batch_size:
        parser.error("batch size must be >=1 and bulk size must be >= batch size")

    client = OpenSearchClient(args.endpoint)
    if args.finalize_only:
        print(json.dumps(finalize(client), indent=2, sort_keys=True))
        return
    if args.reset:
        checkpoint_root = (args.state_root / "checkpoints").resolve()
        if args.state_root.resolve() not in checkpoint_root.parents:
            raise RuntimeError(f"unsafe checkpoint path: {checkpoint_root}")
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
    prepare(client, args.reset)
    if args.prepare_only:
        return

    selected = set(
        args.partition or (partition.partition_id for partition in PARTITIONS)
    )
    encoder = BgeM3Encoder(
        args.model, args.device, args.max_seq_length, args.local_files_only
    )
    states = []
    for partition in PARTITIONS:
        if partition.partition_id in selected:
            states.append(
                build_partition(
                    client,
                    encoder,
                    partition,
                    args.catalog_root,
                    args.state_root,
                    args.model,
                    args.max_seq_length,
                    args.batch_size,
                    args.bulk_size,
                )
            )
    if all(state["completed"] for state in states):
        counts = finalize(client)
        report = {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "counts": counts,
            "partitions": states,
            "runtime": runtime_stats(client),
        }
        args.state_root.mkdir(parents=True, exist_ok=True)
        (args.state_root / "build-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
