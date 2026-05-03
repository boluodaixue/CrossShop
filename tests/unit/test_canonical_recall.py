from globex_agent.infrastructure.recall import (
    CanonicalDedupSearchBackend,
    CanonicalListing,
    PreferredLanguageSearchBackend,
    RecallHit,
    RecallResult,
)


class RankedBackend:
    backend_id = "ranked-ann"

    def __init__(self, document_ids: list[str]) -> None:
        self._document_ids = document_ids
        self.requested_top_ks: list[int] = []

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        del query
        self.requested_top_ks.append(top_k)
        selected = self._document_ids[:top_k]
        return RecallResult(
            backend_id=self.backend_id,
            hits=tuple(
                RecallHit(document_id=document_id, score=1.0 / rank, rank=rank)
                for rank, document_id in enumerate(selected, start=1)
            ),
            total_recall=len(self._document_ids),
            truncated=len(self._document_ids) > top_k,
        )


def test_canonical_backend_expands_until_duplicates_do_not_consume_top_k() -> None:
    documents = [*(f"parallel-{index}" for index in range(7)), "two", "three"]
    listings = [
        CanonicalListing(document_id, "parallel")
        for document_id in documents[:7]
    ] + [
        CanonicalListing("two", "canonical-two"),
        CanonicalListing("three", "canonical-three"),
    ]
    ranked = RankedBackend(documents)
    backend = CanonicalDedupSearchBackend(ranked, listings)

    result = backend.search("query", top_k=3)

    assert [hit.document_id for hit in result.hits] == [
        "parallel-0",
        "two",
        "three",
    ]
    assert ranked.requested_top_ks == [6, 9]
    assert result.total_recall == 3
    assert result.truncated is False


def test_display_language_changes_only_listing_not_relevance() -> None:
    relevance = RankedBackend(["parallel-en", "en-only"])
    listings = [
        CanonicalListing("parallel-en", "parallel", "en"),
        CanonicalListing("parallel-zh", "parallel", "zh"),
        CanonicalListing("en-only", "en-only", "en"),
    ]
    backend = PreferredLanguageSearchBackend(
        relevance,
        listings,
        preferred_language="zh",
    )

    result = backend.search("中文查询", top_k=2)

    assert [hit.document_id for hit in result.hits] == ["parallel-zh", "en-only"]
    assert [hit.score for hit in result.hits] == [1.0, 0.5]
    assert [hit.rank for hit in result.hits] == [1, 2]
