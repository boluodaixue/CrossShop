from datetime import datetime, timezone

from globex_agent.catalog import (
    normalize_esci_judgment,
    normalize_esci_product,
    normalize_shopsimulator_product,
    normalize_shopsimulator_task,
)
from globex_agent.domain import MarketLocale, Platform, PriceSource


def test_esci_product_and_short_label_use_locale_partition() -> None:
    row = {
        "product_id": "B001",
        "product_locale": "jp",
        "product_title": "静かなヘッドホン",
        "product_description": "通勤用",
        "product_bullet_point": "ノイズキャンセリング",
        "product_brand": "Example",
        "product_color": "黒",
    }

    item = normalize_esci_product(
        row,
        ingested_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    qrel = normalize_esci_judgment(
        {
            "query_id": 7,
            "product_id": "B001",
            "product_locale": "jp",
            "esci_label": "S",
        }
    )

    assert item.item_id == "amazon:jp:B001"
    assert item.platform is Platform.AMAZON
    assert item.locale is MarketLocale.JP
    assert item.language == "ja"
    assert item.price_source is PriceSource.UNAVAILABLE
    assert item.price_cny is None
    assert qrel.label == "Substitute"
    assert qrel.gain == 0.1


def test_shopsimulator_multi_price_stays_unresolved_and_task_is_separate() -> None:
    row = {
        "asin": "123",
        "tag": "eval",
        "domain_zh": "家居家装",
        "title": "天然乳胶枕",
        "sub_title": "儿童款",
        "shop_name": "演示店",
        "category": "床上用品›枕头",
        "full_description": "护颈",
        "attribute": ["天然乳胶"],
        "customization_options": {
            "尺寸": [
                {
                    "value": "小号",
                    "asin": "123-a",
                    "price": 99,
                    "is_available": True,
                },
                {
                    "value": "大号",
                    "asin": "123-b",
                    "price": 129,
                    "is_available": True,
                },
            ]
        },
        "pricing": [99.0, 129.0],
        "instructions": [
            {
                "instruction": "给五岁孩子找一个天然乳胶枕",
                "instruction_simple": "儿童乳胶枕",
                "instruction_options": ["小号"],
                "options": ["小号"],
                "attributes": ["天然乳胶"],
            }
        ],
    }

    item = normalize_shopsimulator_product(
        row,
        ingested_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    task, query, qrel = normalize_shopsimulator_task(row)

    assert item.price_cny is None
    assert item.price_source is PriceSource.UNAVAILABLE
    assert item.variants and item.price_source is PriceSource.UNAVAILABLE
    assert len(item.variants) == 2
    assert task.target_item_id == item.item_id
    assert task.split == "test"
    assert query.query == task.query
    assert qrel.item_id == item.item_id
