from typing import Any

import numpy as np

from globex_agent.category_insight import CategoryCard
from globex_agent.category_insight.service import CategoryInsightService, CategoryTaxonomy
from globex_agent.infrastructure.recall.category_kb import (
    CategoryCardHit,
    CategoryQueryType,
    CategorySearchResult,
    HybridWeights,
)


class FakeEncoder:
    encoder_id = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if self.fail:
            raise RuntimeError("tower down")
        return np.ones((len(texts), 4), dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeKnowledgeBase:
    def __init__(self, hits: tuple[CategoryCardHit, ...], *, fail: bool = False) -> None:
        self.hits = hits
        self.fail = fail
        self.bm25_calls = 0

    def search(self, *args: Any, **kwargs: Any) -> CategorySearchResult:
        del args, kwargs
        if self.fail:
            raise RuntimeError("OpenSearch down")
        return CategorySearchResult(
            self.hits,
            CategoryQueryType.NOUN,
            HybridWeights(0.5, 0.5),
            True,
        )

    def search_bm25(self, *args: Any, **kwargs: Any) -> tuple[CategoryCardHit, ...]:
        del args, kwargs
        self.bm25_calls += 1
        if self.fail:
            raise RuntimeError("OpenSearch down")
        return self.hits


class FakeReranker:
    reranker_id = "fake"

    def score_texts(self, query: str, texts: list[str]) -> tuple[float, ...]:
        del query
        return tuple(float(index) for index in range(len(texts)))


def _taxonomy() -> CategoryTaxonomy:
    return CategoryTaxonomy(
        {
            "categories": [
                {
                    "category": "乳胶枕",
                    "category_kind": "ordinary",
                    "aliases": ["乳胶枕头"],
                }
            ]
        }
    )


def test_taxonomy_uses_top_level_kind_when_category_entry_missing() -> None:
    taxonomy = CategoryTaxonomy(
        {
            "category_kind": "bundle",
            "categories": [
                {"category": "礼盒", "aliases": ["礼盒套装"]}
            ],
        }
    )

    assert taxonomy.category_kind("礼盒") == "bundle"
    assert taxonomy.is_known("礼盒")


def _card(
    card_id: str,
    card_type: str,
    summary: str,
    evidence: list[str],
    confidence: float = 0.7,
) -> CategoryCard:
    return CategoryCard(
        card_id=card_id,
        category="乳胶枕",
        card_type=card_type,
        summary=summary,
        raw_evidence=evidence,
        last_updated="2026-08-15T00:00:00+08:00",
        confidence=confidence,
    )


def _hits() -> tuple[CategoryCardHit, ...]:
    cards = [
        _card(
            "b1",
            "bestseller",
            "乳胶枕：特拉雷乳胶枕 / 泰国天然乳胶枕 / 儿童乳胶枕",
            [
                "商品A | 399 | 同质商品数=12，属性覆盖=0.80",
                "商品B | 599 | 同质商品数=10，属性覆盖=0.70",
            ],
        ),
        _card(
            "b2",
            "bestseller",
            "乳胶枕：成人护颈乳胶枕 / 记忆棉乳胶枕 / 可水洗乳胶枕",
            ["商品C | 699 | 同质商品数=9，属性覆盖=0.70"],
        ),
        _card("a1", "attribute", "材质：泰国天然乳胶 75.0% / 记忆棉 25.0%", ["e1"]),
        _card(
            "a2",
            "attribute",
            "功能：护颈椎 60.0% / 助睡眠 40.0%",
            ["e2"],
        ),
        _card("a3", "attribute", "适用人群：成人 100.0%", ["e3"]),
        _card(
            "p1",
            "price_range",
            "便宜款 100-400 / 中档 400-800 / 高端 800-1500",
            ["观察价格样本 n=963"],
        ),
    ]
    return tuple(
        CategoryCardHit(card=card, score=0.8 - index * 0.05, rank=index + 1)
        for index, card in enumerate(cards)
    )


def test_quick_mode_returns_bestsellers_and_price_but_not_attributes() -> None:
    service = CategoryInsightService(
        _taxonomy(), FakeEncoder(), FakeKnowledgeBase(_hits()), FakeReranker()
    )
    run = service.insight("乳胶枕头", depth="quick")

    assert run.output.category == "乳胶枕"
    assert run.output.components == []
    assert [item.name for item in run.output.bestsellers] == ["商品A", "商品B", "商品C"]
    assert run.output.attributes == []
    assert [tier.tier for tier in run.output.price_tiers] == ["budget", "mid", "premium"]
    assert run.output.price_tiers[2].range_cny == (800.0, 1500.0)
    assert run.output.confidence == 0.7
    assert run.diagnostics.final_count == 6


def test_reranker_is_optional_and_hybrid_order_is_the_default() -> None:
    hits = _hits()
    service = CategoryInsightService(
        _taxonomy(),
        FakeEncoder(),
        FakeKnowledgeBase(hits),
    )

    run = service.insight("乳胶枕", depth="quick")

    assert run.diagnostics.recall_mode == "hybrid"
    assert run.diagnostics.reranked is False
    assert run.diagnostics.final_count == len(hits)
    assert [item.name for item in run.output.bestsellers] == [
        "商品A",
        "商品B",
        "商品C",
    ]


def test_deep_mode_extracts_attribute_distributions() -> None:
    service = CategoryInsightService(
        _taxonomy(), FakeEncoder(), FakeKnowledgeBase(_hits()), FakeReranker()
    )
    run = service.insight("乳胶枕", depth="deep")

    assert [attribute.name for attribute in run.output.attributes] == [
        "材质",
        "功能",
        "适用人群",
    ]
    assert run.output.attributes[0].distribution == {
        "泰国天然乳胶": 0.75,
        "记忆棉": 0.25,
    }


def test_vector_failure_degrades_to_bm25() -> None:
    knowledge_base = FakeKnowledgeBase(_hits())
    service = CategoryInsightService(
        _taxonomy(), FakeEncoder(fail=True), knowledge_base, FakeReranker()
    )
    run = service.insight("乳胶枕")

    assert knowledge_base.bm25_calls == 1
    assert run.diagnostics.recall_mode == "bm25_fallback"
    assert "vector recall failed" in (run.diagnostics.fallback_reason or "")
    assert run.output.confidence > 0


def test_opensearch_failure_returns_course_empty_fallback() -> None:
    service = CategoryInsightService(
        _taxonomy(), FakeEncoder(), FakeKnowledgeBase((), fail=True), FakeReranker()
    )
    run = service.insight("乳胶枕")

    assert run.output.category == "乳胶枕"
    assert run.output.bestsellers == []
    assert run.output.price_tiers == []
    assert run.output.confidence == 0
    assert run.diagnostics.recall_mode == "empty_fallback"
