from globex_agent.recall import KeywordSearchBackend, SearchDocument


def test_keyword_backend_ranks_title_match_and_is_deterministic() -> None:
    documents = [
        SearchDocument("item-b", "office keyboard", "quiet mechanical switches"),
        SearchDocument("item-a", "quiet office keyboard", "compact mechanical"),
        SearchDocument("item-c", "travel backpack", "water resistant"),
    ]
    backend = KeywordSearchBackend(documents)

    first = backend.search("quiet office keyboard", top_k=2)
    second = backend.search("quiet office keyboard", top_k=2)

    assert first == second
    assert first.backend_id == "keyword-bm25-v1"
    assert [hit.document_id for hit in first.hits] == ["item-a", "item-b"]
    assert [hit.rank for hit in first.hits] == [1, 2]
    assert first.total_recall == 2
    assert first.truncated is False


def test_keyword_backend_returns_empty_for_unmatched_query() -> None:
    backend = KeywordSearchBackend([SearchDocument("item-a", "headphones")])

    result = backend.search("backpack", top_k=10)

    assert result.hits == ()
    assert result.total_recall == 0
    assert result.truncated is False
