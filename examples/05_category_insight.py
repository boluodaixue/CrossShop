"""Run CategoryInsight quick/deep against the local OpenSearch knowledge base."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from globex_agent.category_insight import CategoryInsightService, CategoryTaxonomy
from globex_agent.recall.category_kb import (
    DEFAULT_CATEGORY_INDEX,
    OpenSearchCategoryKnowledgeBase,
    OpenSearchHttpClient,
)
from globex_agent.recall.embedding import SentenceTransformerTextEncoder
from globex_agent.recall.reranker import (
    CrossEncoderReranker,
    SubprocessCrossEncoderReranker,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("category", nargs="?", default="bathroom fan")
    parser.add_argument("--depth", choices=("quick", "deep"), default="quick")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9200")
    parser.add_argument("--index-name", default=DEFAULT_CATEGORY_INDEX)
    parser.add_argument("--reranker-python", type=Path)
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Explicitly opt in to the current experimental Reranker.",
    )
    parser.add_argument("--reranker-device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    encoder = SentenceTransformerTextEncoder(local_files_only=args.local_files_only)
    knowledge_base = OpenSearchCategoryKnowledgeBase(
        OpenSearchHttpClient(args.endpoint, timeout=30), index_name=args.index_name
    )
    reranker = None
    if args.reranker_python:
        reranker = SubprocessCrossEncoderReranker(
            args.reranker_python,
            device=args.reranker_device,
            batch_size=1,
            use_fp16=True,
            local_files_only=args.local_files_only,
        )
    elif args.enable_reranker:
        reranker = CrossEncoderReranker(
            device="cpu", batch_size=1, local_files_only=args.local_files_only
        )
    service = CategoryInsightService(
        CategoryTaxonomy.from_json(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "category_taxonomy.json"
        ),
        encoder,
        knowledge_base,
        reranker,
    )
    try:
        run = service.insight(args.category, depth=args.depth)
        print(run.output.model_dump_json(indent=2))
        print(json.dumps(asdict(run.diagnostics), ensure_ascii=False, indent=2))
    finally:
        close = getattr(reranker, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
