from globex_agent.category_insight import CategoryCard, rerank_category_hits
from globex_agent.infrastructure.recall.category_kb import CategoryCardHit


class FakeRawTextReranker:
    reranker_id = "fake"

    def __init__(self, scores: tuple[float, ...], *, fail: bool = False) -> None:
        self.scores = scores
        self.fail = fail
        self.texts: list[str] = []

    def score_texts(self, query: str, texts: list[str]) -> tuple[float, ...]:
        del query
        self.texts = list(texts)
        if self.fail:
            raise RuntimeError("boom")
        return self.scores


def _hits(count: int, *, top_score: float = 0.8) -> tuple[CategoryCardHit, ...]:
    return tuple(
        CategoryCardHit(
            card=CategoryCard(
                card_id=f"cc-{index:02d}",
                category="乳胶枕",
                card_type="attribute",
                summary=f"功能：护颈椎 {index + 1}.0%",
                raw_evidence=[f"淘宝商品-{index}"],
                last_updated="2026-08-18T00:00:00+08:00",
                confidence=0.6,
            ),
            score=top_score - index * 0.01,
            rank=index + 1,
        )
        for index in range(count)
    )


def test_category_reranker_scores_query_against_summary_only() -> None:
    hits = _hits(3)
    reranker = FakeRawTextReranker((0.1, 0.9, 0.3))
    result = rerank_category_hits(
        "护颈椎乳胶枕", hits, reranker, final_k=2, bypass_top_score=0.92
    )

    assert result.reranked is True
    assert [hit.card.card_id for hit in result.hits] == ["cc-01", "cc-02"]
    assert reranker.texts == [hit.card.summary for hit in hits]


def test_category_reranker_can_score_context_complete_retrieval_text() -> None:
    hits = tuple(
        CategoryCardHit(
            card=hit.card,
            score=hit.score,
            rank=hit.rank,
            retrieval_text=f"bilingual retrieval text {hit.rank} 双语检索文本",
            retrieval_text_en=f"English retrieval text {hit.rank}",
            retrieval_text_zh=f"中文检索文本 {hit.rank}",
        )
        for hit in _hits(3)
    )
    reranker = FakeRawTextReranker((0.1, 0.9, 0.3))
    result = rerank_category_hits(
        "测试查询",
        hits,
        reranker,
        final_k=2,
        bypass_top_score=0.92,
        document_mode="contextual",
    )

    assert result.reranked is True
    assert reranker.texts == [hit.retrieval_text_zh for hit in hits]
    assert [hit.retrieval_text for hit in result.hits] == [
        "bilingual retrieval text 2 双语检索文本",
        "bilingual retrieval text 3 双语检索文本",
    ]


def test_category_reranker_applies_both_course_bypasses() -> None:
    reranker = FakeRawTextReranker(())
    high_score = rerank_category_hits(
        "query", _hits(3, top_score=0.93), reranker, final_k=2
    )
    short_list = rerank_category_hits("query", _hits(2), reranker, final_k=2)

    assert high_score.reranked is False
    assert high_score.bypass_reason == "coarse top score >= 0.92"
    assert short_list.reranked is False
    assert short_list.bypass_reason == "candidate count is not greater than final_k"


def test_category_reranker_can_disable_the_top_score_bypass() -> None:
    reranker = FakeRawTextReranker((0.3, 0.9, 0.1))
    result = rerank_category_hits(
        "query",
        _hits(3, top_score=0.99),
        reranker,
        final_k=2,
        bypass_top_score=None,
    )

    assert result.reranked is True
    assert result.bypass_reason is None
    assert [hit.card.card_id for hit in result.hits] == ["cc-01", "cc-00"]


def test_category_reranker_falls_back_to_coarse_order() -> None:
    hits = _hits(3)
    reranker = FakeRawTextReranker((), fail=True)
    result = rerank_category_hits("query", hits, reranker, final_k=2)

    assert result.reranked is False
    assert result.bypass_reason == "reranker failed: RuntimeError"
    assert [hit.card.card_id for hit in result.hits] == ["cc-00", "cc-01"]
