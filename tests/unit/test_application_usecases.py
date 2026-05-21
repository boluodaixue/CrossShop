"""Application use case tests for catalog search and order lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest

from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.application.usecases.order_usecases import (
    CancelOrderUseCase,
    OrderItemInput,
    PlaceOrderUseCase,
    QueryOrderUseCase,
)
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec
from globex_agent.domain.order.address import Address
from globex_agent.infrastructure.persistence.in_memory_repositories import (
    InMemoryItemRepository,
    InMemoryOrderRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTS_PATH = PROJECT_ROOT / "data" / "demo" / "products.jsonl"


@pytest.fixture()
def item_repo() -> InMemoryItemRepository:
    catalog = LocalCatalog.from_jsonl(PRODUCTS_PATH, strict=True).catalog
    return InMemoryItemRepository(list(catalog.items))


@pytest.fixture()
def order_repo() -> InMemoryOrderRepository:
    return InMemoryOrderRepository()


def _address() -> Address:
    return Address(
        recipient_name="张三",
        country="CN",
        state="浙江",
        city="杭州",
        address_line="西湖区某路 1 号",
        postal_code="310000",
        phone="13800000000",
    )


class TestCatalogSearch:
    async def test_keyword_recall_returns_item_cards(self, item_repo) -> None:
        usecase = CatalogSearchUseCase(item_repo)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="头戴式降噪耳机")
        )
        assert result["hits"]
        assert result["hits"][0]["item_id"].split(":", 1)[0] in {
            "amazon",
            "taobao",
            "shopee",
            "aliexpress",
            "ebay",
        }
        assert result["recall_strategy"] == "keyword_bm25"
        assert result["hits"][0]["price_source"] == "observed"

    async def test_ship_to_filter_and_top_k(self, item_repo) -> None:
        usecase = CatalogSearchUseCase(item_repo)
        result = await usecase.execute(
            ProductSearchSpec(normalized_query="耳机", ship_to="US", top_k=3)
        )
        assert len(result["hits"]) <= 3
        assert all("US" in hit["ships_to"] for hit in result["hits"])
        assert result.get("filtered_out")

    async def test_cn_ship_to_keeps_aliexpress_catalog(self, item_repo) -> None:
        usecase = CatalogSearchUseCase(item_repo)
        result = await usecase.execute(
            ProductSearchSpec(
                normalized_query="Budget Wave H1 头戴式降噪耳机",
                ship_to="CN",
                top_k=5,
            )
        )
        assert any(hit["item_id"].startswith("aliexpress:") for hit in result["hits"])

    async def test_no_hit_returns_empty(self, item_repo) -> None:
        result = await CatalogSearchUseCase(item_repo).execute(
            ProductSearchSpec(normalized_query="quantum flux capacitor")
        )
        assert result["hits"] == []


class TestOrderLifecycle:
    async def test_place_query_cancel_roundtrip(
        self,
        item_repo,
        order_repo,
    ) -> None:
        place = PlaceOrderUseCase(item_repo, order_repo)
        query = QueryOrderUseCase(order_repo)
        cancel = CancelOrderUseCase(order_repo)

        snapshot = await place.execute(
            buyer_id="buyer-1",
            items=[
                OrderItemInput(
                    item_id="amazon:aqp-1001",
                    variant_id="",
                    quantity=1,
                )
            ],
            shipping_address=_address(),
        )
        assert snapshot["status"] == "CONFIRMED"
        assert snapshot["lines"][0]["item_id"] == "amazon:aqp-1001"
        assert snapshot["lines"][0]["variant_id"].endswith(":default")

        queried = await query.execute(snapshot["order_id"])
        assert queried["order_id"] == snapshot["order_id"]

        cancelled = await cancel.execute(snapshot["order_id"], "买家改主意了")
        assert cancelled["status"] == "CANCELLED"

    async def test_unknown_order(self, order_repo) -> None:
        with pytest.raises(ValueError, match="订单不存在"):
            await QueryOrderUseCase(order_repo).execute("GBX-999999")
