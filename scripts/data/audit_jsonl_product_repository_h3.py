"""Load and audit the full H0 catalog through JsonlProductRepository."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.infrastructure.persistence.jsonl_product_repository import (
        JsonlProductRepository,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=ROOT / "data" / "processed" / "catalogs-v2",
    )
    parser.add_argument(
        "--h2-report",
        type=Path,
        default=ROOT / "artifacts" / "h2-opensearch" / "adapter-smoke-report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "h3-catalog-search" / "repository-audit.json",
    )
    args = parser.parse_args()

    tracemalloc.start()
    started = time.perf_counter()
    repository = JsonlProductRepository(args.catalog_root)
    load_seconds = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if repository.product_count != 45_286:
        raise RuntimeError(f"unexpected Product count: {repository.product_count}")
    if repository.sku_count != 261_369:
        raise RuntimeError(f"unexpected SKU count: {repository.sku_count}")

    h2_report = json.loads(args.h2_report.read_text(encoding="utf-8"))
    platform_hits = {
        platform: [hit["product_id"] for hit in evidence["hits"]]
        for platform, evidence in h2_report["platforms"].items()
    }
    flattened_ids = [
        product_id
        for product_ids in platform_hits.values()
        for product_id in product_ids
    ]
    restored = asyncio.run(repository.find_by_ids(flattened_ids))
    restored_ids = [product.product_id for product in restored]
    if restored_ids != flattened_ids:
        raise RuntimeError("saved H2 Product hits were not restored in input order")

    result = {
        "catalog_root": str(args.catalog_root.resolve()),
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "h2_saved_hits": {
            "all_restored_in_order": True,
            "count": len(restored_ids),
            "platforms": platform_hits,
        },
        "load": {
            "elapsed_seconds": load_seconds,
            "tracemalloc_current_bytes": current_bytes,
            "tracemalloc_peak_bytes": peak_bytes,
            "tracemalloc_scope": (
                "Python allocations traced during repository construction only; "
                "this is not process RSS and excludes untraced native allocations"
            ),
        },
        "product_count": repository.product_count,
        "sku_count": repository.sku_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
