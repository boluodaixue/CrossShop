"""Build two physical SQLite catalogs with locale-specific FTS5 partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from globex_agent.domain import MarketLocale, Platform, StandardItem
from globex_agent.infrastructure.recall import standard_item_to_search_document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
PLATFORM_LOCALES = {
    Platform.AMAZON: (MarketLocale.US, MarketLocale.ES, MarketLocale.JP),
    Platform.TAOBAO: (MarketLocale.CN,),
}
CJK_LOCALES = frozenset({MarketLocale.CN, MarketLocale.JP})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--catalog-root", type=Path, default=None, help="validated schema-v2 JSONL root"
    )
    args = parser.parse_args()
    args.processed_root = args.processed_root.resolve()
    if args.catalog_root is not None:
        args.catalog_root = args.catalog_root.resolve()

    built_at = datetime.now(timezone.utc)
    databases: dict[str, dict[str, object]] = {}
    for platform, locales in PLATFORM_LOCALES.items():
        catalog_root = args.catalog_root or (args.processed_root / "catalogs")
        sources = {
            locale: (catalog_root / platform.value / locale.value / "items.jsonl")
            for locale in locales
        }
        missing = [str(path) for path in sources.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing normalized {platform.value} catalog files: {missing}")
        database_path = args.processed_root / "databases" / platform.value / "catalog.sqlite3"
        counts = _build_database(database_path, platform, sources, built_at)
        databases[platform.value] = {
            "path": _relative(database_path),
            "bytes": database_path.stat().st_size,
            "sha256": _sha256(database_path),
            "locale_counts": dict(sorted(counts.items())),
            "source_files": {
                locale.value: {
                    "path": _relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for locale, path in sources.items()
            },
        }
        print(
            f"built {platform.value}: {sum(counts.values()):,} items, "
            f"{database_path.stat().st_size / 1024 / 1024:.1f} MiB"
        )

    manifest_path = args.processed_root / "manifests" / "catalog_databases.json"
    manifest = {
        "schema_version": "platform-catalog-sqlite-fts5-v2",
        "catalog_schema": "catalog-schema-v2",
        "built_at": built_at.isoformat(),
        "physical_database_policy": "one SQLite database per platform",
        "retrieval_partition_policy": "one FTS5 table per platform/locale",
        "tokenizer_policy": {
            "us": "unicode61 remove_diacritics 2",
            "es": "unicode61 remove_diacritics 2",
            "jp": "trigram",
            "cn": "trigram",
        },
        "databases": databases,
    }
    _write_json(manifest_path, manifest)
    print(f"manifest: {manifest_path}")


def _build_database(
    path: Path,
    platform: Platform,
    sources: dict[MarketLocale, Path],
    built_at: datetime,
) -> Counter[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE items (
                item_id TEXT PRIMARY KEY,
                same_group_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                locale TEXT NOT NULL,
                language TEXT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                brand TEXT,
                category_json TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                materials_json TEXT NOT NULL,
                variants_json TEXT NOT NULL,
                price_cny TEXT,
                original_price_cny TEXT,
                currency_raw TEXT,
                price_source TEXT NOT NULL,
                availability TEXT NOT NULL,
                ships_to_json TEXT,
                record_json TEXT NOT NULL
            );
            CREATE INDEX idx_items_locale ON items(locale, item_id);
            CREATE INDEX idx_items_same_group ON items(same_group_id);
            """
        )
        for locale in sources:
            tokenizer = "trigram" if locale in CJK_LOCALES else "unicode61 remove_diacritics 2"
            connection.execute(
                f"""
                CREATE VIRTUAL TABLE items_fts_{locale.value} USING fts5(
                    item_id UNINDEXED,
                    title,
                    body,
                    tokenize='{tokenizer}'
                )
                """
            )

        counts: Counter[str] = Counter()
        with connection:
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", "platform-catalog-sqlite-fts5-v2"),
                    ("platform", platform.value),
                    ("built_at", built_at.isoformat()),
                ),
            )
            for locale, source_path in sources.items():
                with source_path.open(encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        try:
                            item = StandardItem.model_validate_json(line)
                        except Exception as exc:  # noqa: BLE001 - annotate source line
                            raise RuntimeError(
                                f"invalid catalog row {source_path}:{line_number}"
                            ) from exc
                        if item.platform is not platform or item.locale is not locale:
                            raise RuntimeError(
                                f"wrong partition at {source_path}:{line_number}: "
                                f"{item.platform.value}/{item.locale}"
                            )
                        document = standard_item_to_search_document(item)
                        record_json = item.model_dump_json()
                        connection.execute(
                            """
                            INSERT INTO items VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                            """,
                            (
                                item.item_id,
                                item.same_group_id,
                                item.platform.value,
                                item.locale.value,
                                item.language,
                                item.title,
                                document.body,
                                item.brand,
                                json.dumps(item.category_path, ensure_ascii=False),
                                json.dumps(
                                    [
                                        attribute.model_dump(mode="json")
                                        for attribute in item.attributes
                                    ],
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                json.dumps(
                                    [
                                        material.model_dump(mode="json")
                                        for material in item.materials
                                    ],
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                json.dumps(
                                    [variant.model_dump(mode="json") for variant in item.variants],
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                str(item.price_cny) if item.price_cny is not None else None,
                                str(item.original_price_cny)
                                if item.original_price_cny is not None
                                else None,
                                item.currency_raw.value if item.currency_raw else None,
                                item.price_source.value,
                                item.availability.value,
                                json.dumps(item.ships_to, ensure_ascii=False)
                                if item.ships_to is not None
                                else None,
                                record_json,
                            ),
                        )
                        connection.execute(
                            f"INSERT INTO items_fts_{locale.value} VALUES (?, ?, ?)",
                            (item.item_id, item.title, document.body),
                        )
                        counts[locale.value] += 1
                connection.execute(
                    f"INSERT INTO items_fts_{locale.value}(items_fts_{locale.value}) "
                    "VALUES ('optimize')"
                )
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()
    temporary.replace(path)
    return counts


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _write_json(path: Path, payload: dict[str, object]) -> None:
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
