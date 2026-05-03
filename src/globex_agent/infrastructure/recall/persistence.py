"""Persistence helpers shared by index-building and retrieval evaluation scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from globex_agent.domain import StandardItem
from globex_agent.infrastructure.recall.base import SearchDocument
from globex_agent.infrastructure.recall.embedding import clean_product_body


def load_search_documents_jsonl(path: Path) -> list[SearchDocument]:
    with path.open(encoding="utf-8") as source:
        records: list[dict[str, Any]] = [
            json.loads(line) for line in source if line.strip()
        ]
    return [
        SearchDocument(
            document_id=record["document_id"],
            title=record["title"],
            body=clean_product_body(record["title"], record["body"]),
        )
        for record in records
    ]


def load_standard_item_documents_jsonl(path: Path) -> list[SearchDocument]:
    """Load normalized catalog rows as the shared Item-tower document text."""

    documents: list[SearchDocument] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                documents.append(
                    standard_item_to_search_document(StandardItem.model_validate_json(line))
                )
    return documents


def standard_item_to_search_document(item: StandardItem) -> SearchDocument:
    body = " ".join(
        part
        for part in (
            item.brand or "",
            " ".join(item.category_path),
            _json_text(item.attributes),
            _json_text(item.variants),
            item.description,
        )
        if part
    )
    return SearchDocument(
        document_id=item.item_id,
        title=item.title,
        body=clean_product_body(item.title, body),
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_index_manifest(
    index_path: Path,
    *,
    items_path: Path,
    model_name: str,
    document_count: int,
    dimension: int,
    max_seq_length: int,
    text_format_version: str,
    index_type: str,
    index_parameters: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "architecture": "query-item-dual-tower",
        "user_tower": False,
        "encoder_model": model_name,
        "index_type": index_type,
        "index_parameters": index_parameters,
        "document_count": document_count,
        "dimension": dimension,
        "max_seq_length": max_seq_length,
        "text_format_version": text_format_version,
        "items_sha256": sha256(items_path),
        "index_sha256": sha256(index_path),
    }
    if extra_metadata:
        manifest.update(extra_metadata)
    manifest_path = index_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
