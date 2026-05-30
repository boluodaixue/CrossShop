"""Build one frozen BGE-M3 Faiss HNSW index per platform/locale partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from globex_agent.infrastructure.recall import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    ITEM_TEXT_FORMAT_VERSION,
    EmbeddingSearchBackend,
    FaissHNSWIndex,
    SentenceTransformerTextEncoder,
)
from globex_agent.infrastructure.recall.persistence import (
    load_standard_item_documents_jsonl,
    sha256,
    write_index_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = (
    ("amazon", "us"),
    ("amazon", "es"),
    ("amazon", "jp"),
    ("taobao", "cn"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=None,
        help="Validated catalog-schema-v2 JSONL root; defaults to staging when present.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "index" / "catalog",
    )
    parser.add_argument(
        "--precomputed-root",
        type=Path,
        help="Use bge-m3-items.npz files produced by encode_catalog_items_cuda.py.",
    )
    parser.add_argument(
        "--partition",
        action="append",
        choices=tuple(f"{platform}:{locale}" for platform, locale in PARTITIONS),
        help="Build only a selected platform:locale partition; may be repeated.",
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    catalog_root = args.catalog_root or (
        args.processed_root / "staging" / "catalog-schema-v2"
        if (args.processed_root / "staging" / "catalog-schema-v2").exists()
        else args.processed_root / "catalogs"
    )

    selected = set(
        args.partition
        or (f"{platform}:{locale}" for platform, locale in PARTITIONS)
    )
    encoder = SentenceTransformerTextEncoder(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )
    for platform, locale in PARTITIONS:
        partition_id = f"{platform}:{locale}"
        if partition_id not in selected:
            continue
        items_path = (
            catalog_root
            / platform
            / locale
            / "items.jsonl"
        )
        if not items_path.exists():
            raise FileNotFoundError(f"normalized catalog not found: {items_path}")
        documents = load_standard_item_documents_jsonl(items_path)
        embeddings_path = None
        precomputed_metadata: dict[str, object] = {}
        if args.precomputed_root is not None:
            embeddings_path = (
                args.precomputed_root / platform / locale / "bge-m3-items.npz"
            )
            if not embeddings_path.exists():
                raise FileNotFoundError(f"precomputed embeddings not found: {embeddings_path}")
            embeddings_manifest_path = embeddings_path.with_suffix(".manifest.json")
            if not embeddings_manifest_path.exists():
                raise FileNotFoundError(
                    f"precomputed manifest not found: {embeddings_manifest_path}"
                )
            embeddings_manifest = json.loads(
                embeddings_manifest_path.read_text(encoding="utf-8")
            )
            expected = {
                "encoder_model": args.model,
                "partition_id": partition_id,
                "document_count": len(documents),
                "max_seq_length": args.max_seq_length,
                "text_format_version": ITEM_TEXT_FORMAT_VERSION,
                "items_sha256": sha256(items_path),
            }
            mismatches = {
                key: (embeddings_manifest.get(key), value)
                for key, value in expected.items()
                if embeddings_manifest.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"precomputed embedding manifest mismatch for {partition_id}: "
                    f"{mismatches}"
                )
            with np.load(embeddings_path, allow_pickle=False) as payload:
                document_ids = payload["document_ids"].astype(str).tolist()
                embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            index = FaissHNSWIndex(
                document_ids,
                embeddings,
                m=args.hnsw_m,
                ef_construction=args.ef_construction,
                ef_search=args.ef_search,
            )
            backend = EmbeddingSearchBackend(documents, encoder, index=index)
            precomputed_metadata = {
                "item_embeddings_sha256": sha256(embeddings_path),
                "item_embeddings_manifest_sha256": sha256(embeddings_manifest_path),
                "item_encoding_inference_dtype": embeddings_manifest["inference_dtype"],
                "item_encoding_pooling": embeddings_manifest["pooling"],
            }
            print(
                f"indexing precomputed {partition_id}: {len(documents):,} items",
                flush=True,
            )
        else:
            print(f"encoding {partition_id}: {len(documents):,} items", flush=True)
            backend = EmbeddingSearchBackend.with_faiss_hnsw(
                documents,
                encoder,
                m=args.hnsw_m,
                ef_construction=args.ef_construction,
                ef_search=args.ef_search,
            )
        index_path = (
            args.output_root / platform / locale / "bge-m3-hnsw-ip.faiss"
        )
        backend.save_index(index_path)
        mapping_path = index_path.with_suffix(".mapping.json")
        mapping_path.write_text(
            json.dumps(
                {
                    "mapping_version": "position-to-item-id-v1",
                    "document_ids": sorted(document.document_id for document in documents),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path = write_index_manifest(
            index_path,
            items_path=items_path,
            model_name=args.model,
            document_count=len(documents),
            dimension=backend.dimension,
            max_seq_length=encoder.max_seq_length,
            text_format_version=ITEM_TEXT_FORMAT_VERSION,
            index_type=backend.index_id,
            index_parameters=backend.index_parameters,
            extra_metadata={
                "platform": platform,
                "locale": locale,
                "partition_id": partition_id,
                "position_mapping_path": str(mapping_path),
                "position_mapping_sha256": sha256(mapping_path),
                "embedding_fine_tuned": False,
                "semantic_query_policy": "encode original query without translation",
                "lexical_translation_policy": (
                    "optional locale translation belongs to the separate lexical branch"
                ),
                "item_encoding_runtime": (
                    "precomputed-transformers-cls"
                    if embeddings_path is not None
                    else "sentence-transformers"
                ),
                **precomputed_metadata,
            },
        )
        print(f"built {partition_id}: {index_path}", flush=True)
        print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
