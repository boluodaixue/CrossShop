"""H3：Product 还原后的硬约束、SKU 投影与工具协议。"""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from app.application.tools.product_search_tool import build_product_search_tool
from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.money import Money
from app.domain.catalog.ports.retrieval_ports import VectorHit
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.sku import Sku
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.in_memory_repositories import (
    InMemoryProductRepository,
)


def _sku(sku_id: str, price: float, stock: int, currency: str = "CNY") -> Sku:
    return Sku(
        sku_id=sku_id,
        spec=sku_id,
        price=Money.from_major_units(price, currency),
        stock=stock,
    )


def _product(
    product_id: str, skus: list[Sku], *, ships_to: list[str] | None = None
) -> Product:
    return Product(
        product_id=product_id,
        title=f"测试商品 {product_id}",
        brand="Globex",
        category="测试品类",
        origin_country="CN",
        description="测试商品",
        ships_to=ships_to or ["CN", "US"],
        skus=skus,
    )


class _Embedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0]


class _VectorIndex:
    def __init__(self, product_ids: list[str]) -> None:
        self.product_ids = product_ids
        self.search_calls = 0

    async def search(
        self, *, query: str, embedding: list[float], top_n: int
    ) -> list[VectorHit]:
        self.search_calls += 1
        return [
            VectorHit(product_id=product_id, score=1.0 - index / 10)
            for index, product_id in enumerate(self.product_ids)
        ]


class _CapturingReranker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(documents)
        return [float(len(documents) - index) for index in range(len(documents))]


class _BrokenReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls += 1
        raise RuntimeError("reranker unavailable")


class _CountingRepository(InMemoryProductRepository):
    def __init__(self, products: list[Product]) -> None:
        super().__init__(products)
        self.list_all_calls = 0

    async def list_all(self) -> list[Product]:
        self.list_all_calls += 1
        return await super().list_all()


class _DuplicateRepository(_CountingRepository):
    async def find_by_ids(self, product_ids: list[str]) -> list[Product]:
        products = await super().find_by_ids(product_ids)
        return [*products, *products]


def test_product_search_spec_has_exact_v2_fields_and_top_k_range() -> None:
    assert [field.name for field in fields(ProductSearchSpec)] == [
        "normalized_query",
        "category",
        "ship_to",
        "locale",
        "top_k",
        "target_currency",
        "price_max_major",
    ]
    assert ProductSearchSpec(normalized_query="测试").top_k == 5
    for invalid in (0, 6, -1, True, 1.5):
        with pytest.raises(ValueError, match=r"1\.\.5"):
            ProductSearchSpec(normalized_query="测试", top_k=invalid)  # type: ignore[arg-type]


async def test_constraints_run_before_rerank_and_card_projects_eligible_skus() -> None:
    eligible = _product(
        "P-ELIGIBLE",
        [
            _sku("SKU-ZERO-CHEAP", 1, 0),
            _sku("SKU-B", 90, 2),
            _sku("SKU-A", 80, 3),
            _sku("SKU-TOO-HIGH", 100, 4),
        ],
    )
    over_cap = _product(
        "P-OVER",
        [_sku("SKU-OUT-CHEAP", 5, 0), _sku("SKU-IN-STOCK", 120, 2)],
    )
    out_of_stock = _product("P-OUT", [_sku("SKU-OUT", 2, 0)])
    repo = _CountingRepository([eligible, over_cap, out_of_stock])
    index = _VectorIndex(["P-OVER", "P-OUT", "P-ELIGIBLE"])
    reranker = _CapturingReranker()

    result = await CatalogSearchUseCase(
        repo,
        embedder=_Embedder(),
        vector_index=index,
        reranker=reranker,
    ).execute(
        ProductSearchSpec(normalized_query="测试商品", price_max_major=90),
    )

    assert index.search_calls == 1
    assert len(reranker.calls) == 1
    assert len(reranker.calls[0]) == 1
    assert "P-ELIGIBLE" in reranker.calls[0][0]
    assert result["recall_strategy"] == "embedding_rerank"
    assert result["total_candidates"] == 1

    card = result["hits"][0]
    assert card["product_id"] == "P-ELIGIBLE"
    assert card["price_major"] == 80
    assert card["currency"] == "CNY"
    assert [sku["sku_id"] for sku in card["skus"]] == ["SKU-A", "SKU-B"]
    assert all(sku["stock"] > 0 and sku["price_major"] <= 90 for sku in card["skus"])

    assert result["filtered_out"] == [
        {
            "product_id": "P-OVER",
            "title": "测试商品 P-OVER",
            "category": "测试品类",
            "price_major": 120,
            "currency": "CNY",
            "reason": "over_price_cap",
        },
    ]
    assert all(item["product_id"] != "P-OUT" for item in result["filtered_out"])


async def test_empty_eligible_set_does_not_rerank_or_issue_keyword_recall() -> None:
    over_cap = _product("P-OVER", [_sku("SKU-OVER", 120, 2)])
    repo = _CountingRepository([over_cap])
    index = _VectorIndex(["P-OVER"])
    reranker = _CapturingReranker()

    result = await CatalogSearchUseCase(
        repo,
        embedder=_Embedder(),
        vector_index=index,
        reranker=reranker,
    ).execute(
        ProductSearchSpec(normalized_query="测试商品", price_max_major=90),
    )

    assert index.search_calls == 1
    assert repo.list_all_calls == 0
    assert reranker.calls == []
    assert result["hits"] == []
    assert result["recall_strategy"] == "embedding_only"
    assert result["rerank_applied"] is False
    assert result["filtered_out"][0]["reason"] == "over_price_cap"


async def test_empty_hybrid_does_not_fallback_to_keyword() -> None:
    keyword_match = _product("P-KEYWORD", [_sku("SKU-KEYWORD", 10, 1)])
    repo = _CountingRepository([keyword_match])
    index = _VectorIndex([])

    result = await CatalogSearchUseCase(
        repo,
        embedder=_Embedder(),
        vector_index=index,
    ).execute(ProductSearchSpec(normalized_query="测试商品"))

    assert index.search_calls == 1
    assert repo.list_all_calls == 0
    assert result == {
        "hits": [],
        "total_candidates": 0,
        "recall_strategy": "embedding_only",
        "rerank_applied": False,
    }


async def test_missing_vector_hit_product_fails_instead_of_degrading() -> None:
    repo = _CountingRepository([])
    index = _VectorIndex(["P-MISSING"])

    with pytest.raises(RuntimeError, match="missing=.*P-MISSING"):
        await CatalogSearchUseCase(
            repo,
            embedder=_Embedder(),
            vector_index=index,
        ).execute(ProductSearchSpec(normalized_query="测试商品"))

    assert index.search_calls == 1
    assert repo.list_all_calls == 0


async def test_duplicate_repository_product_fails_instead_of_degrading() -> None:
    product = _product("P-DUPLICATE", [_sku("SKU-DUPLICATE", 10, 1)])
    repo = _DuplicateRepository([product])

    with pytest.raises(RuntimeError, match="重复 product_id"):
        await CatalogSearchUseCase(
            repo,
            embedder=_Embedder(),
            vector_index=_VectorIndex(["P-DUPLICATE"]),
        ).execute(ProductSearchSpec(normalized_query="测试商品"))

    assert repo.list_all_calls == 0


async def test_reranker_failure_keeps_vector_order_without_second_recall() -> None:
    first = _product("P-FIRST", [_sku("SKU-FIRST", 10, 1)])
    second = _product("P-SECOND", [_sku("SKU-SECOND", 20, 1)])
    repo = _CountingRepository([first, second])
    index = _VectorIndex(["P-FIRST", "P-SECOND"])
    reranker = _BrokenReranker()

    result = await CatalogSearchUseCase(
        repo,
        embedder=_Embedder(),
        vector_index=index,
        reranker=reranker,
    ).execute(ProductSearchSpec(normalized_query="测试商品"))

    assert reranker.calls == 1
    assert index.search_calls == 1
    assert repo.list_all_calls == 0
    assert result["recall_strategy"] == "embedding_only"
    assert result["rerank_applied"] is False
    assert [hit["product_id"] for hit in result["hits"]] == ["P-FIRST", "P-SECOND"]


async def test_mixed_currency_skus_are_filtered_and_sorted_in_target_currency() -> None:
    product = _product(
        "P-FX",
        [
            _sku("SKU-CNY", 100, 2, "CNY"),
            _sku("SKU-USD", 10, 3, "USD"),
            _sku("SKU-EUR", 9, 4, "EUR"),
            _sku("SKU-OUT", 1, 0, "USD"),
        ],
    )
    result = await CatalogSearchUseCase(
        InMemoryProductRepository([product]),
        embedder=_Embedder(),
        vector_index=_VectorIndex(["P-FX"]),
    ).execute(
        ProductSearchSpec(
            normalized_query="测试商品",
            target_currency="CNY",
            price_max_major=80,
        ),
    )

    card = result["hits"][0]
    assert card["currency"] == "CNY"
    assert card["price_major"] == pytest.approx(70.2)
    assert [sku["sku_id"] for sku in card["skus"]] == ["SKU-EUR", "SKU-USD"]
    assert [sku["price_major"] for sku in card["skus"]] == pytest.approx(
        [70.2, 71.0],
    )
    assert {sku["currency"] for sku in card["skus"]} == {"CNY"}


async def test_keyword_fallback_excludes_products_with_no_stock() -> None:
    in_stock = _product("P-IN", [_sku("SKU-IN", 10, 1)])
    out_of_stock = _product("P-OUT", [_sku("SKU-OUT", 1, 0)])
    result = await CatalogSearchUseCase(
        InMemoryProductRepository([out_of_stock, in_stock]),
    ).execute(ProductSearchSpec(normalized_query="测试商品"))

    assert result["recall_strategy"] == "keyword_2gram"
    assert [hit["product_id"] for hit in result["hits"]] == ["P-IN"]
    assert "filtered_out" not in result


async def test_top_k_is_applied_after_constraint_and_rerank() -> None:
    products = [
        _product(f"P-{index}", [_sku(f"SKU-{index}", 10, 1)]) for index in range(6)
    ]
    result = await CatalogSearchUseCase(
        InMemoryProductRepository(products),
        embedder=_Embedder(),
        vector_index=_VectorIndex([product.product_id for product in products]),
        reranker=_CapturingReranker(),
    ).execute(ProductSearchSpec(normalized_query="测试商品", top_k=5))

    assert result["total_candidates"] == 6
    assert len(result["hits"]) == 5


async def test_product_search_tool_passes_locale_to_exact_spec() -> None:
    class _CapturingUseCase:
        spec: ProductSearchSpec | None = None

        async def execute(self, spec: ProductSearchSpec) -> dict:
            self.spec = spec
            return {
                "hits": [],
                "total_candidates": 0,
                "recall_strategy": "keyword_2gram",
                "rerank_applied": False,
            }

    usecase = _CapturingUseCase()
    bus = TradeEventBus()
    queue = bus.subscribe("h3-locale")
    tool = build_product_search_tool(usecase, bus)  # type: ignore[arg-type]
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="h3-locale",
            buyer_id="buyer",
            locale="es-ES",
            currency="EUR",
        ),
    )
    try:
        response = await tool(normalized_query="maleta", locale="es-ES")
    finally:
        ShoppingContext.reset(token)

    assert json.loads(response.content[0].text)["hits"] == []
    assert usecase.spec is not None and usecase.spec.locale == "es-ES"
    invoke = queue.get_nowait()
    assert invoke.payload["args"]["locale"] == "es-ES"
