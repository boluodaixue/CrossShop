import importlib.util
import json
import sys
from pathlib import Path


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "data" / "build_catalogs_v2.py"
    spec = importlib.util.spec_from_file_location("catalog_builder_v2", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def _row(platform: str, locale: str, item_id: str, **overrides) -> dict:
    value = {
        "item_id": f"{platform}:{locale}:{item_id}",
        "same_group_id": f"{platform}:{item_id}",
        "platform": platform,
        "locale": locale,
        "language": "zh" if platform == "taobao" else "en",
        "title": "测试商品",
        "description": "测试描述",
        "brand": None,
        "category_path": ["测试", "品类"],
        "rating": None,
        "review_count": 0,
        "attributes": [
            {"code": "shop_name", "name": "shop_name", "value": "测试店铺"},
            {"code": "feature", "name": "功能", "value": "轻便"},
        ],
        "materials": [],
        "variants": [],
        "ingested_at": "2026-08-26T00:00:00Z",
        "source_updated_at": None,
    }
    value.update(overrides)
    return value


def test_amazon_conversion_is_strict_and_deterministic():
    row = _row("amazon", "us", "A-1", brand=None)
    first = BUILDER._convert_amazon(row).product.to_dict()
    second = BUILDER._convert_amazon(row).product.to_dict()
    assert first == second
    assert first["brand"] == "未知"
    assert first["category"] == "未知"
    assert first["origin_country"] == "US"
    assert first["ships_to"] == ["US", "CN"]
    assert first["skus"][0]["sku_id"] == "amazon:us:A-1:default"
    assert 5000 <= first["skus"][0]["price"]["amount_in_minor_units"] <= 200000
    assert 0 <= first["skus"][0]["stock"] <= 100
    assert set(first) == BUILDER.PRODUCT_FIELDS
    assert set(first["skus"][0]) == BUILDER.SKU_FIELDS


def test_taobao_uses_shop_name_and_stable_delivery_and_stock():
    row = _row(
        "taobao",
        "cn",
        "T-1",
        variants=[
            {
                "variant_id": "taobao:variant:T-1",
                "options": [{"name": "颜色", "value": "黑色"}],
                "price_cny": "12.50",
                "price_source": "observed",
                "availability": "available",
            }
        ],
    )
    first = BUILDER._convert_taobao(row).product.to_dict()
    second = BUILDER._convert_taobao(row).product.to_dict()
    assert first == second
    assert first["brand"] == "测试店铺"
    assert first["origin_country"] == "CN"
    assert "CN" in first["ships_to"]
    assert 1 <= first["skus"][0]["stock"] <= 100


def test_published_catalog_has_only_strict_fields():
    root = Path(__file__).parents[1] / "data" / "processed" / "catalogs-v2"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["partitions"]) == {
        "globex_reference",
        "taobao",
        "amazon/us",
        "amazon/es",
        "amazon/jp",
    }
    for partition in manifest["partitions"].values():
        assert Path(root, partition["products"]["path"]).is_file()
        assert Path(root, partition["search_units"]["path"]).is_file()
        assert partition["products"]["records"] > 0
        assert partition["search_units"]["records"] > 0
    for path in root.glob("**/*.jsonl"):
        for line in path.open(encoding="utf-8"):
            value = json.loads(line)
            assert "availability" not in value
            assert "simulated" not in value
            assert "provenance" not in value
