"""Build a query-grouped US/ES/JP ESCI subset and three catalog partitions.

The default selection keeps 300 complete query groups per locale: 240 upstream
train groups become train, another 30 upstream train groups become dev, and 30
upstream test groups remain test. Selection is deterministic by SHA-256 of the
locale, source split, and query ID. Every selected query keeps all ESCI rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from globex_agent.catalog import (
    normalize_esci_judgment,
    normalize_esci_product,
    normalize_esci_query,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "amazon_esci"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
OFFICIAL_SOURCE_URL = "https://github.com/amazon-science/esci-data"
DATASET_FILES = {
    "examples": "shopping_queries_dataset_examples.parquet",
    "products": "shopping_queries_dataset_products.parquet",
}
DOWNLOAD_URLS = {
    name: (
        "https://media.githubusercontent.com/media/amazon-science/esci-data/"
        f"main/shopping_queries_dataset/{filename}"
    )
    for name, filename in DATASET_FILES.items()
}
LOCALES = ("us", "es", "jp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-per-locale", type=int, default=240)
    parser.add_argument("--dev-per-locale", type=int, default=30)
    parser.add_argument("--test-per-locale", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing official parquet files before building.",
    )
    args = parser.parse_args()
    if min(args.train_per_locale, args.dev_per_locale, args.test_per_locale) < 0:
        parser.error("query counts cannot be negative")
    if args.train_per_locale + args.dev_per_locale + args.test_per_locale == 0:
        parser.error("at least one query group must be selected")

    raw_paths = {
        name: args.raw_dir / filename for name, filename in DATASET_FILES.items()
    }
    if args.download:
        for name, path in raw_paths.items():
            _download_if_missing(DOWNLOAD_URLS[name], path)
    missing = [str(path) for path in raw_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing ESCI parquet files; pass --download or provide --raw-dir: "
            + ", ".join(missing)
        )

    parquet = _import_parquet()
    selected_splits = _select_query_groups(
        parquet,
        raw_paths["examples"],
        train_count=args.train_per_locale,
        dev_count=args.dev_per_locale,
        test_count=args.test_per_locale,
        batch_size=args.batch_size,
    )
    examples = _load_selected_examples(
        parquet,
        raw_paths["examples"],
        selected_splits,
        batch_size=args.batch_size,
    )
    product_keys = {
        (str(row["product_locale"]), str(row["product_id"])) for row in examples
    }
    ingested_at = datetime.now(timezone.utc)
    products = _load_selected_products(
        parquet,
        raw_paths["products"],
        product_keys,
        ingested_at=ingested_at,
        batch_size=args.batch_size,
    )
    _validate(selected_splits, examples, products, product_keys)
    manifest_path = _write_outputs(
        args.output_root,
        selected_splits,
        examples,
        products,
        raw_paths,
        ingested_at,
    )

    print(f"selected query groups: {len(selected_splits):,}")
    print(f"preserved judgments: {len(examples):,}")
    print(f"unique catalog items: {len(products):,}")
    print(f"items by locale: {dict(sorted(Counter(i.locale.value for i in products).items()))}")
    print(f"manifest: {manifest_path}")


def _import_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required; install the project dependencies before running"
        ) from exc
    return parquet


def _select_query_groups(
    parquet: Any,
    examples_path: Path,
    *,
    train_count: int,
    dev_count: int,
    test_count: int,
    batch_size: int,
) -> dict[tuple[str, str, str], str]:
    available: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    columns = ["query_id", "query", "product_locale", "small_version", "split"]
    for batch in parquet.ParquetFile(examples_path).iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        for row in batch.to_pylist():
            locale = str(row["product_locale"])
            source_split = str(row["split"])
            if locale not in LOCALES or source_split not in {"train", "test"}:
                continue
            if int(row.get("small_version") or 0) != 1:
                continue
            query_id = str(row["query_id"])
            query = " ".join(str(row.get("query") or "").split())
            prior = available[(locale, source_split)].setdefault(query_id, query)
            if prior != query:
                raise RuntimeError(f"query text drift for {locale}/{query_id}")

    selected: dict[tuple[str, str, str], str] = {}
    for locale in LOCALES:
        upstream_train = _stable_query_ids(available[(locale, "train")], locale, "train")
        upstream_test = _stable_query_ids(available[(locale, "test")], locale, "test")
        needed_train = train_count + dev_count
        if len(upstream_train) < needed_train or len(upstream_test) < test_count:
            raise RuntimeError(
                f"not enough query groups for {locale}: "
                f"train={len(upstream_train)}, test={len(upstream_test)}"
            )
        for query_id in upstream_train[:train_count]:
            selected[(locale, "train", query_id)] = "train"
        for query_id in upstream_train[train_count:needed_train]:
            selected[(locale, "train", query_id)] = "dev"
        for query_id in upstream_test[:test_count]:
            selected[(locale, "test", query_id)] = "test"
    return selected


def _stable_query_ids(values: dict[str, str], locale: str, split: str) -> list[str]:
    return sorted(
        values,
        key=lambda query_id: (
            hashlib.sha256(f"{locale}:{split}:{query_id}".encode()).hexdigest(),
            query_id,
        ),
    )


def _load_selected_examples(
    parquet: Any,
    examples_path: Path,
    selected_splits: dict[tuple[str, str, str], str],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    columns = [
        "query",
        "query_id",
        "product_id",
        "product_locale",
        "esci_label",
        "small_version",
        "split",
    ]
    rows: list[dict[str, Any]] = []
    for batch in parquet.ParquetFile(examples_path).iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        for row in batch.to_pylist():
            key = (
                str(row["product_locale"]),
                str(row["split"]),
                str(row["query_id"]),
            )
            if key in selected_splits:
                rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["product_locale"]),
            str(row["split"]),
            str(row["query_id"]),
            str(row["product_id"]),
        ),
    )


def _load_selected_products(
    parquet: Any,
    products_path: Path,
    product_keys: set[tuple[str, str]],
    *,
    ingested_at: datetime,
    batch_size: int,
) -> list[Any]:
    products = []
    seen: set[tuple[str, str]] = set()
    for batch in parquet.ParquetFile(products_path).iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            key = (str(row["product_locale"]), str(row["product_id"]))
            if key not in product_keys:
                continue
            if key in seen:
                raise RuntimeError(f"duplicate ESCI product row: {key}")
            seen.add(key)
            products.append(normalize_esci_product(row, ingested_at=ingested_at))
    return sorted(products, key=lambda item: item.item_id)


def _validate(
    selected_splits: dict[tuple[str, str, str], str],
    examples: list[dict[str, Any]],
    products: list[Any],
    product_keys: set[tuple[str, str]],
) -> None:
    found_groups = {
        (str(row["product_locale"]), str(row["split"]), str(row["query_id"]))
        for row in examples
    }
    if found_groups != set(selected_splits):
        raise RuntimeError("one or more selected query groups have no rows")
    if any(int(row.get("small_version") or 0) != 1 for row in examples):
        raise RuntimeError("selected examples contain a non-small-version row")
    normalized_keys = {(item.locale.value, item.item_id.rsplit(":", 1)[-1]) for item in products}
    if normalized_keys != product_keys:
        missing = sorted(product_keys - normalized_keys)[:10]
        raise RuntimeError(f"missing selected product metadata: {missing}")
    if len({item.item_id for item in products}) != len(products):
        raise RuntimeError("duplicate normalized Amazon item_id")


def _write_outputs(
    output_root: Path,
    selected_splits: dict[tuple[str, str, str], str],
    examples: list[dict[str, Any]],
    products: list[Any],
    raw_paths: dict[str, Path],
    ingested_at: datetime,
) -> Path:
    examples_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        key = (
            str(row["product_locale"]),
            str(row["split"]),
            str(row["query_id"]),
        )
        examples_by_group[key].append(row)

    outputs: dict[str, Path] = {}
    per_locale: dict[str, dict[str, Any]] = {}
    for locale in LOCALES:
        locale_products = [item for item in products if item.locale.value == locale]
        group_keys = sorted(key for key in selected_splits if key[0] == locale)
        queries = [
            normalize_esci_query(
                examples_by_group[key][0],
                split=selected_splits[key],
            )
            for key in group_keys
        ]
        qrels = [
            normalize_esci_judgment(row)
            for key in group_keys
            for row in examples_by_group[key]
        ]
        item_path = output_root / "catalogs" / "amazon" / locale / "items.jsonl"
        query_path = output_root / "eval" / "amazon_esci" / locale / "queries.jsonl"
        qrel_path = output_root / "eval" / "amazon_esci" / locale / "qrels.jsonl"
        _write_jsonl(item_path, (item.model_dump(mode="json") for item in locale_products))
        _write_jsonl(query_path, (query.model_dump(mode="json") for query in queries))
        _write_jsonl(qrel_path, (qrel.model_dump(mode="json") for qrel in qrels))
        outputs[f"{locale}_items"] = item_path
        outputs[f"{locale}_queries"] = query_path
        outputs[f"{locale}_qrels"] = qrel_path
        per_locale[locale] = {
            "item_count": len(locale_products),
            "query_count": len(queries),
            "qrel_count": len(qrels),
            "split_counts": dict(sorted(Counter(q.split for q in queries).items())),
            "label_counts": dict(sorted(Counter(q.label for q in qrels).items())),
            "source_query_ids": {
                split: [
                    key[2]
                    for key in group_keys
                    if selected_splits[key] == split
                ]
                for split in ("train", "dev", "test")
            },
        }

    manifest_path = output_root / "manifests" / "amazon_esci_multilingual.json"
    manifest = {
        "dataset_version": "amazon-esci-small-query-grouped-multilingual-v1",
        "official_source_url": OFFICIAL_SOURCE_URL,
        "license": "Apache-2.0",
        "ingested_at": ingested_at.isoformat(),
        "platform": "amazon",
        "locales": list(LOCALES),
        "languages": {"us": "en", "es": "es", "jp": "ja"},
        "task_version": "small_version=1",
        "selection_method": (
            "complete query groups sorted by SHA-256(locale:source_split:query_id)"
        ),
        "split_policy": (
            "240 upstream train -> train, next 30 upstream train -> dev, "
            "30 upstream test -> test, per locale by default"
        ),
        "query_translation_policy": (
            "none for semantic retrieval and multilingual reranking; any future "
            "locale translation may supplement lexical recall only"
        ),
        "price_policy": "unavailable because ESCI contains no product price field",
        "category_policy": "unknown because ESCI contains no product category field",
        "judgment_policy": "retain every ESCI row and label in each selected query group",
        "gain_mapping": {
            "Exact": 1.0,
            "Substitute": 0.1,
            "Complement": 0.01,
            "Irrelevant": 0.0,
        },
        "per_locale": per_locale,
        "total_item_count": len(products),
        "total_query_count": len(selected_splits),
        "total_qrel_count": len(examples),
        "source_files": {
            name: {
                "filename": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in raw_paths.items()
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        print(f"using existing source: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    print(f"downloading {url}")
    request = Request(url, headers={"User-Agent": "globex-agent-learning/0.1"})
    with (
        urlopen(request, timeout=300) as response,  # noqa: S310 - fixed HTTPS hosts
        temporary.open("wb") as target,
    ):
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(path)
    print(f"downloaded {path} ({path.stat().st_size:,} bytes)")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
