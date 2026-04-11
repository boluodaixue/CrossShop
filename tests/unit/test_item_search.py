from globex_agent.catalog import LocalCatalog
from globex_agent.domain import Platform, ResultStatus, SearchRequest
from globex_agent.recall import RecallHit, RecallResult, SearchDocument
from globex_agent.tools.item_search import search_items


class RecordingBackend:
    backend_id = "recording"

    def __init__(self) -> None:
        self.top_k: int | None = None

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        del query
        self.top_k = top_k
        return RecallResult(
            backend_id=self.backend_id,
            hits=(),
            total_recall=0,
            truncated=False,
        )


class RankedBackend:
    backend_id = "ranked-ann"

    def __init__(self, document_ids: list[str]) -> None:
        self._document_ids = document_ids
        self.requested_top_ks: list[int] = []

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        del query
        self.requested_top_ks.append(top_k)
        rows = self._document_ids[:top_k]
        return RecallResult(
            backend_id=self.backend_id,
            hits=tuple(
                RecallHit(document_id=document_id, score=1.0 / rank, rank=rank)
                for rank, document_id in enumerate(rows, start=1)
            ),
            total_recall=len(self._document_ids),
            truncated=len(self._document_ids) > top_k,
        )


class FailingReranker:
    reranker_id = "failing-reranker"

    def score(
        self,
        query: str,
        documents: list[SearchDocument],
    ) -> tuple[float, ...]:
        del query, documents
        raise TimeoutError("simulated outage")


def test_search_returns_one_platform_output(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
) -> None:
    request = demo_requests[0]
    result = search_items(
        demo_catalog,
        request.query,
        Platform.AMAZON,
        top_k=request.top_k,
        hard_constraints=request.hard_constraints,
    )

    assert result.status is ResultStatus.OK
    assert result.platform is Platform.AMAZON
    assert len(result.candidates) == 3
    assert result.total_recall == 7
    assert result.truncated is True
    assert all(candidate.platform is Platform.AMAZON for candidate in result.candidates)
    assert all("耳机" in candidate.category_path for candidate in result.candidates)


def test_search_honors_platform_and_top_k(demo_catalog: LocalCatalog) -> None:
    result = search_items(
        demo_catalog,
        query="头戴式降噪耳机",
        platform=Platform.SHOPEE,
        top_k=2,
    )

    assert len(result.candidates) == 2
    assert all(candidate.platform is Platform.SHOPEE for candidate in result.candidates)


def test_search_returns_structured_no_results(demo_catalog: LocalCatalog) -> None:
    result = search_items(
        demo_catalog,
        query="zzzzzzzz",
        platform=Platform.AMAZON,
        top_k=3,
    )

    assert result.status is ResultStatus.NO_RESULTS
    assert result.candidates == []
    assert result.issues[0].code == "no_search_match"


def test_item_search_caps_public_top_k_but_ann_can_overfetch(
    demo_catalog: LocalCatalog,
) -> None:
    backend = RecordingBackend()

    search_items(
        demo_catalog,
        query="headphones",
        platform=Platform.AMAZON,
        top_k=100,
        backend=backend,
    )

    assert backend.top_k == 100


def test_canonical_ann_fallback_deduplicates_and_selects_display_language(
    demo_catalog: LocalCatalog,
) -> None:
    source = demo_catalog.get("amazon:aqp-1001")
    assert source is not None
    parallel_en = source.model_copy(
        update={
            "item_id": "amazon:parallel-en",
            "same_group_id": "parallel-product",
            "language": "en",
            "title": "Quiet commuter headphones",
            "description": "Over-ear active noise cancelling headphones.",
        }
    )
    parallel_zh = source.model_copy(
        update={
            "item_id": "amazon:parallel-zh",
            "same_group_id": "parallel-product",
            "language": "zh",
            "title": "通勤头戴式降噪耳机",
            "description": "适合地铁通勤。",
        }
    )
    en_only = source.model_copy(
        update={
            "item_id": "amazon:en-only",
            "same_group_id": "en-only-product",
            "language": "en",
            "title": "English-only travel headphones",
        }
    )
    third = source.model_copy(
        update={
            "item_id": "amazon:third-zh",
            "same_group_id": "third-product",
            "language": "zh",
            "title": "第三款耳机",
        }
    )
    catalog = LocalCatalog([parallel_en, parallel_zh, en_only, third])
    backend = RankedBackend(
        [
            parallel_en.item_id,
            parallel_zh.item_id,
            en_only.item_id,
            third.item_id,
        ]
    )

    result = search_items(
        catalog,
        query="通勤降噪耳机",
        platform=Platform.AMAZON,
        top_k=2,
        backend=backend,
    )

    assert [candidate.item_id for candidate in result.candidates] == [
        parallel_zh.item_id,
        en_only.item_id,
    ]
    assert [candidate.same_group_id for candidate in result.candidates] == [
        "parallel-product",
        "en-only-product",
    ]
    assert backend.requested_top_ks == [4]


def test_reranker_failure_degrades_to_canonical_ann_order(
    demo_catalog: LocalCatalog,
) -> None:
    items = [
        item
        for item in demo_catalog.items
        if item.platform is Platform.AMAZON
    ][:3]
    catalog = LocalCatalog(items)
    backend = RankedBackend([item.item_id for item in items])

    result = search_items(
        catalog,
        query="通勤耳机",
        platform=Platform.AMAZON,
        top_k=2,
        backend=backend,
        reranker=FailingReranker(),
    )

    assert [candidate.item_id for candidate in result.candidates] == [
        item.item_id for item in items[:2]
    ]
