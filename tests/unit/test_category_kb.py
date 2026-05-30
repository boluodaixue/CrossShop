import json
from typing import Any

import numpy as np

from globex_agent.category_insight import CategoryCard
from globex_agent.infrastructure.recall.category_kb import (
    WEIGHTS_BY_QUERY_TYPE,
    CategoryQueryType,
    HybridWeights,
    OpenSearchCategoryKnowledgeBase,
    OpenSearchRequestError,
    build_category_index_mapping,
    build_search_pipeline,
    category_document_text,
    category_retrieval_text,
    category_retrieval_text_zh,
    classify_category_query,
    pipeline_name,
    setup_category_index,
)


class FakeTransport:
    def __init__(self, *, index_exists: bool = False) -> None:
        self.index_exists = index_exists
        self.calls: list[dict[str, Any]] = []
        self.search_response: dict[str, Any] = {"hits": {"hits": []}}

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any | None = None,
        params: dict[str, str] | None = None,
        content_type: str = "application/json",
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "params": params,
                "content_type": content_type,
            }
        )
        if method == "GET" and path == "globex_category_kb":
            if not self.index_exists:
                raise OpenSearchRequestError(404, "missing")
            return {"globex_category_kb": {}}
        if method == "PUT" and path == "globex_category_kb":
            self.index_exists = True
            return {"acknowledged": True}
        if path == "_bulk":
            return {"errors": False, "items": []}
        if path.endswith("/_search"):
            return self.search_response
        return {"acknowledged": True}


def _card() -> CategoryCard:
    return CategoryCard(
        card_id="cc-test-bestseller-01",
        category="test products",
        card_type="bestseller",
        summary="test products：form one / form two / form three",
        raw_evidence=["Example item | 99 | generated orders=10"],
        last_updated="2026-08-15T00:00:00+08:00",
        confidence=0.6,
    )


def test_mapping_and_embedding_text_use_context_complete_english_v2_text() -> None:
    card = _card()
    mapping = build_category_index_mapping(
        4, analyzer="ik_max_word", search_analyzer="ik_smart"
    )
    properties = mapping["mappings"]["properties"]

    assert category_document_text(card) == (
        "Category: test products. Knowledge type: bestselling product forms. "
        "Summary: form one / form two / form three"
    )
    assert category_document_text(card) == category_retrieval_text(card)
    assert mapping["settings"]["index"]["knn"] is True
    assert mapping["mappings"]["dynamic"] == "strict"
    assert properties["category"]["analyzer"] == "ik_max_word"
    assert properties["category"]["search_analyzer"] == "ik_smart"
    assert properties["retrieval_text"]["analyzer"] == "ik_max_word"
    assert properties["retrieval_text"]["search_analyzer"] == "ik_smart"
    assert properties["content_vector"]["dimension"] == 4
    assert properties["content_vector"]["method"] == {
        "name": "hnsw",
        "engine": "faiss",
        "space_type": "cosinesimil",
        "parameters": {"m": 32, "ef_construction": 200},
    }


def test_pipelines_keep_knn_and_bm25_weight_order() -> None:
    assert WEIGHTS_BY_QUERY_TYPE[CategoryQueryType.NOUN] == HybridWeights(0.5, 0.5)
    assert WEIGHTS_BY_QUERY_TYPE[CategoryQueryType.ATTRIBUTE_CONSTRAINT] == (
        HybridWeights(0.7, 0.3)
    )
    assert WEIGHTS_BY_QUERY_TYPE[CategoryQueryType.STYLE] == HybridWeights(0.9, 0.1)
    assert WEIGHTS_BY_QUERY_TYPE[CategoryQueryType.COLLOQUIAL] == HybridWeights(1.0, 0.0)
    pipeline = build_search_pipeline(HybridWeights(0.7, 0.3))
    processor = pipeline["phase_results_processors"][0]["normalization-processor"]
    assert processor["normalization"] == {"technique": "min_max"}
    assert processor["combination"]["technique"] == "arithmetic_mean"
    assert processor["combination"]["parameters"]["weights"] == [0.7, 0.3]


def test_setup_creates_index_pipelines_and_bulk_documents() -> None:
    client = FakeTransport()
    card = _card()
    bilingual_text = "Category: test products. 品类：测试商品。"
    setup_category_index(
        client,
        [card],
        np.ones((1, 4), dtype=np.float32),
        retrieval_texts=[bilingual_text],
        retrieval_texts_en=["Category: test products."],
        retrieval_texts_zh=["品类：测试商品。"],
    )

    mapping_call = next(
        call
        for call in client.calls
        if call["path"] == "globex_category_kb" and call["method"] == "PUT"
    )
    assert mapping_call["body"]["mappings"]["properties"]["content_vector"][
        "dimension"
    ] == 4
    pipeline_calls = [
        call for call in client.calls if call["path"].startswith("_search/pipeline/")
    ]
    assert {call["path"] for call in pipeline_calls} == {
        f"_search/pipeline/{pipeline_name(CategoryQueryType.NOUN)}",
        f"_search/pipeline/{pipeline_name(CategoryQueryType.ATTRIBUTE_CONSTRAINT)}",
        f"_search/pipeline/{pipeline_name(CategoryQueryType.STYLE)}",
    }
    bulk_call = next(call for call in client.calls if call["path"] == "_bulk")
    lines = bulk_call["body"].splitlines()
    assert json.loads(lines[0]) == {
        "index": {"_index": "globex_category_kb", "_id": card.card_id}
    }
    indexed = json.loads(lines[1])
    assert indexed["card_id"] == card.card_id
    assert indexed["retrieval_text"] == bilingual_text
    assert indexed["retrieval_text_en"] == "Category: test products."
    assert indexed["retrieval_text_zh"] == "品类：测试商品。"
    assert indexed["content_vector"] == [1.0, 1.0, 1.0, 1.0]


def test_hybrid_search_uses_knn_first_and_matching_pipeline() -> None:
    client = FakeTransport(index_exists=True)
    card = _card()
    client.search_response = {
        "hits": {
            "hits": [
                {
                    "_score": 0.8,
                    "_source": {
                        **card.model_dump(mode="json"),
                        "retrieval_text": "Category: test products. 品类：测试商品。",
                        "retrieval_text_en": "Category: test products.",
                        "retrieval_text_zh": "品类：测试商品。",
                    },
                }
            ]
        }
    }
    kb = OpenSearchCategoryKnowledgeBase(client)
    result = kb.search(
        "test products with form one",
        [0.1, 0.2, 0.3, 0.4],
        top_k=30,
        query_type=CategoryQueryType.ATTRIBUTE_CONSTRAINT,
    )
    call = client.calls[-1]
    queries = call["body"]["query"]["hybrid"]["queries"]

    assert "knn" in queries[0]
    assert "multi_match" in queries[1]
    assert queries[1]["multi_match"]["fields"] == [
        "category^2",
        "retrieval_text",
    ]
    assert "analyzer" not in queries[1]["multi_match"]
    assert call["body"]["_source"]["excludes"] == ["content_vector"]
    assert call["params"] == {
        "search_pipeline": pipeline_name(CategoryQueryType.ATTRIBUTE_CONSTRAINT)
    }
    assert result.used_bm25 is True
    assert result.weights == HybridWeights(0.7, 0.3)
    assert result.hits[0].card == card
    assert result.hits[0].retrieval_text == (
        "Category: test products. 品类：测试商品。"
    )
    assert result.hits[0].retrieval_text_en == "Category: test products."
    assert result.hits[0].retrieval_text_zh == "品类：测试商品。"


def test_colloquial_search_removes_bm25_branch() -> None:
    client = FakeTransport(index_exists=True)
    kb = OpenSearchCategoryKnowledgeBase(client)
    result = kb.search(
        "my bathroom stays steamy, what kind of fan should I look for",
        [0.1, 0.2],
        query_type=CategoryQueryType.COLLOQUIAL,
    )
    call = client.calls[-1]

    assert "knn" in call["body"]["query"]
    assert "hybrid" not in call["body"]["query"]
    assert call["params"] is None
    assert result.used_bm25 is False


def test_fallback_query_classifier_covers_four_course_types() -> None:
    assert classify_category_query("label makers and printers") == CategoryQueryType.NOUN
    assert classify_category_query("black steel fence panels") == (
        CategoryQueryType.ATTRIBUTE_CONSTRAINT
    )
    assert classify_category_query("clean modern privacy boundary") == CategoryQueryType.STYLE
    assert classify_category_query("I need power farther from the outlet") == (
        CategoryQueryType.COLLOQUIAL
    )
    assert classify_category_query("机械键盘") == CategoryQueryType.NOUN
    assert classify_category_query("想要三模并且RGB的机械键盘") == (
        CategoryQueryType.ATTRIBUTE_CONSTRAINT
    )
    assert classify_category_query("适合办公、外观简洁的机械键盘") == (
        CategoryQueryType.STYLE
    )
    assert classify_category_query("预算不超过500元，哪个机械键盘比较好") == (
        CategoryQueryType.COLLOQUIAL
    )


def test_price_card_retrieval_text_explains_numeric_tiers_in_english() -> None:
    card = CategoryCard(
        card_id="cc-test-price-range-01",
        category="test products",
        card_type="price_range",
        summary="便宜款 100-420 / 中档 420-1040 / 高端 1040-1510",
        raw_evidence=["generated CNY prices"],
        last_updated="2026-08-15T00:00:00+08:00",
        confidence=0.55,
    )

    assert category_retrieval_text(card) == (
        "Category: test products. Knowledge type: price tiers. "
        "Summary: Budget tier CNY 100-420; mid-range tier CNY 420-1040; "
        "premium tier CNY 1040-1510"
    )


def test_chinese_retrieval_text_matches_taobao_course_format() -> None:
    card = CategoryCard(
        card_id="cc-latex-pillow-attribute-01",
        category="乳胶枕",
        card_type="attribute",
        summary="材质：泰国天然乳胶 96.2% / 记忆棉 3.8%",
        raw_evidence=["泰国天然乳胶: taobao:cn:747848614498 梦洁宝贝泰国乳胶枕头"],
        last_updated="2026-08-18T00:00:00+08:00",
        confidence=0.84,
    )

    assert category_retrieval_text_zh(card) == (
        "品类：乳胶枕。知识类型：目录样本属性出现率。"
        "摘要：材质：泰国天然乳胶 96.2% / 记忆棉 3.8%"
    )
