"""Build the persisted Item-tower vector index for the checked-in ESCI subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from globex_agent.infrastructure.recall import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    ITEM_TEXT_FORMAT_VERSION,
    EmbeddingSearchBackend,
    SentenceTransformerTextEncoder,
)
from globex_agent.infrastructure.recall.persistence import (
    load_search_documents_jsonl,
    write_index_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "recall_items.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--index-backend",
        choices=("faiss-hnsw", "exact"),
        default="faiss-hnsw",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "index" / "esci-bge-m3-hnsw-ip.faiss",
    )
    args = parser.parse_args()
    if args.index_backend == "faiss-hnsw" and args.output.suffix != ".faiss":
        parser.error("--output must end in .faiss for --index-backend faiss-hnsw")
    if args.index_backend == "exact" and args.output.suffix != ".npz":
        parser.error("--output must end in .npz for --index-backend exact")

    documents = load_search_documents_jsonl(args.items)
    encoder = SentenceTransformerTextEncoder(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )
    if args.index_backend == "faiss-hnsw":
        backend = EmbeddingSearchBackend.with_faiss_hnsw(
            documents,
            encoder,
            m=args.hnsw_m,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
    else:
        backend = EmbeddingSearchBackend(documents, encoder)
    backend.save_index(args.output)
    manifest_path = write_index_manifest(
        args.output,
        items_path=args.items,
        model_name=args.model,
        document_count=len(documents),
        dimension=backend.dimension,
        max_seq_length=encoder.max_seq_length,
        text_format_version=ITEM_TEXT_FORMAT_VERSION,
        index_type=backend.index_id,
        index_parameters=backend.index_parameters,
    )

    print("architecture: Query tower + Item tower (no User tower)")
    print(f"model: {args.model}")
    print(f"documents: {len(documents)}")
    print(f"dimension: {backend.dimension}")
    print(f"max sequence length: {encoder.max_seq_length}")
    print(f"item text format: {ITEM_TEXT_FORMAT_VERSION}")
    print(f"index type: {backend.index_id}")
    print(f"index parameters: {backend.index_parameters}")
    print(f"index: {args.output}")
    print(f"manifest: {manifest_path}")
if __name__ == "__main__":
    main()
