"""Run the product vector Top-100 -> Reranker Top-10 retrieval mainline."""

from __future__ import annotations

import argparse
from pathlib import Path

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import Platform
from globex_agent.infrastructure.recall import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    EmbeddingSearchBackend,
    KeywordSearchBackend,
    RerankedSearchBackend,
    SentenceTransformerTextEncoder,
    SubprocessCrossEncoderReranker,
    standard_item_to_search_document,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="预算 1500 元的通勤降噪耳机，不要入耳式")
    parser.add_argument(
        "--platform",
        choices=[platform.value for platform in Platform],
        default=Platform.SHOPEE.value,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reranker-python", type=Path)
    parser.add_argument("--reranker-device", default="cuda:0")
    parser.add_argument("--reranker-fp32", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    platform = Platform(args.platform)
    catalog = LocalCatalog.from_jsonl(CATALOG_PATH, strict=True).catalog
    documents = [
        standard_item_to_search_document(item)
        for item in catalog.items
        if item.platform is platform and item.is_available
    ]
    title_by_id = {document.document_id: document.title for document in documents}

    bm25 = KeywordSearchBackend(documents)
    encoder = SentenceTransformerTextEncoder(
        args.embedding_model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    embedding = EmbeddingSearchBackend.with_faiss_hnsw(documents, encoder)
    if args.reranker_python is None:
        pair_reranker = CrossEncoderReranker(
            args.reranker_model,
            device=args.device,
            batch_size=1,
            local_files_only=args.local_files_only,
        )
    else:
        pair_reranker = SubprocessCrossEncoderReranker(
            args.reranker_python,
            args.reranker_model,
            device=args.reranker_device,
            batch_size=1,
            use_fp16=not args.reranker_fp32,
            local_files_only=args.local_files_only,
        )
    reranked = RerankedSearchBackend(
        embedding,
        documents,
        pair_reranker,
        candidate_k=100,
    )

    print("architecture: Query tower + Item tower (no User tower)")
    print(
        "course mainline: Query/Item vector recall Top-100 "
        f"-> Reranker Top-{args.top_k}"
    )
    print(f"query: {args.query}")
    for name, backend in (
        ("BM25 baseline", bm25),
        ("Query/Item vector recall", embedding),
        ("vector recall + Reranker (course mainline)", reranked),
    ):
        result = backend.search(args.query, top_k=args.top_k)
        ranking = ", ".join(
            f"{hit.rank}.{title_by_id[hit.document_id]}({hit.score:.4f})"
            for hit in result.hits
        )
        print(f"{name}: {ranking}")


if __name__ == "__main__":
    main()
