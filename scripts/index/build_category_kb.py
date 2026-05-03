"""Encode CategoryCards with BGE-M3 and bulk-index them into OpenSearch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from globex_agent.category_insight import CategoryCard, admit_card
from globex_agent.infrastructure.recall.category_kb import (
    DEFAULT_ANALYZER,
    OpenSearchHttpClient,
    setup_category_index,
)
from globex_agent.infrastructure.recall.embedding import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerTextEncoder,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3_INDEX = "globex_category_kb_v3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "category_insight" / "category_cards_v3.jsonl"
        ),
    )
    parser.add_argument(
        "--retrieval-texts",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "category_insight"
            / "category_retrieval_texts_v3.jsonl"
        ),
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:9200")
    parser.add_argument("--index-name", default=DEFAULT_V3_INDEX)
    parser.add_argument("--analyzer", default=DEFAULT_ANALYZER)
    parser.add_argument("--search-analyzer")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "output" / "category_kb" / "index_manifest_v3.json",
    )
    args = parser.parse_args()

    cards = _load_cards(args.cards)
    retrieval_texts = _load_retrieval_texts(args.retrieval_texts, cards)
    encoder = SentenceTransformerTextEncoder(
        args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        local_files_only=args.local_files_only,
    )
    embeddings = encoder.encode_documents(retrieval_texts["bilingual"])
    client = OpenSearchHttpClient(args.endpoint, timeout=30)
    setup_category_index(
        client,
        cards,
        embeddings,
        index_name=args.index_name,
        analyzer=args.analyzer,
        search_analyzer=args.search_analyzer,
        retrieval_texts=retrieval_texts["bilingual"],
        retrieval_texts_en=retrieval_texts["en"],
        retrieval_texts_zh=retrieval_texts["zh"],
        recreate=args.recreate,
    )
    count = client.request("GET", f"{args.index_name}/_count")["count"]
    if count != len(cards):
        raise RuntimeError(f"indexed card count {count} does not match source {len(cards)}")
    health = client.request(
        "GET",
        f"_cluster/health/{args.index_name}",
        params={"wait_for_status": "green", "timeout": "10s"},
    )

    manifest = {
        "index_name": args.index_name,
        "endpoint": args.endpoint,
        "opensearch_version": client.request("GET", "")["version"]["number"],
        "cluster_status": health["status"],
        "analyzer": args.analyzer,
        "search_analyzer": args.search_analyzer or args.analyzer,
        "embedding_model": encoder.encoder_id,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_max_seq_length": args.max_seq_length,
        "card_count": len(cards),
        "cards_sha256": hashlib.sha256(args.cards.read_bytes()).hexdigest(),
        "retrieval_texts_sha256": hashlib.sha256(
            args.retrieval_texts.read_bytes()
        ).hexdigest(),
        "embedding_text_policy": "versioned English + Chinese retrieval sidecar",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"indexed {count} cards into {args.index_name} with "
        f"{encoder.encoder_id} dim={embeddings.shape[1]}"
    )


def _load_cards(path: Path) -> list[CategoryCard]:
    cards: list[CategoryCard] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        admission = admit_card(raw)
        if not admission.accepted or admission.card is None:
            raise ValueError(f"card line {line_number} failed admission: {admission.reason}")
        cards.append(admission.card)
    if not cards:
        raise ValueError("cards file is empty")
    return cards


def _load_retrieval_texts(
    path: Path, cards: list[CategoryCard]
) -> dict[str, list[str]]:
    by_card_id: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        card_id = str(row["card_id"])
        if card_id in by_card_id:
            raise ValueError(f"duplicate retrieval text at line {line_number}: {card_id}")
        texts = {
            "en": " ".join(str(row["retrieval_text_en"]).split()),
            "zh": " ".join(str(row["retrieval_text_zh"]).split()),
            "bilingual": " ".join(
                str(row["retrieval_text_bilingual"]).split()
            ),
        }
        if any(not text for text in texts.values()):
            raise ValueError(f"blank retrieval text at line {line_number}: {card_id}")
        by_card_id[card_id] = texts
    card_ids = {card.card_id for card in cards}
    if set(by_card_id) != card_ids:
        missing = card_ids - by_card_id.keys()
        extra = by_card_id.keys() - card_ids
        raise ValueError(
            f"retrieval sidecar is not card-aligned; missing={sorted(missing)[:3]} "
            f"extra={sorted(extra)[:3]}"
        )
    return {
        language: [by_card_id[card.card_id][language] for card in cards]
        for language in ("en", "zh", "bilingual")
    }


if __name__ == "__main__":
    main()
