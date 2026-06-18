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

EXPECTED_PRODUCTS = {
    "globex_reference": (
        60,
        "0d9cf6b39e2d0ef7a9ac182175118b5a8af1ce336311a816808042817744d2ca",
    ),
    "taobao": (
        23409,
        "775b35edeeafaba9d9745b9538683b06c9a0ab769191b7b41d053f396f843294",
    ),
    "amazon/us": (
        5638,
        "a930711136343c9a5d648e21408eb4d418515f6b2a28699bbe4ffac4459804d9",
    ),
    "amazon/es": (
        7856,
        "2efef0a4565ef8637273dd497b0eb981384f31b0ca17bd02c9de5bd5414cebf7",
    ),
    "amazon/jp": (
        8323,
        "b0ac282cd53d6d22e34236462ad2ec6d5f8cbc7ed9b89bed5cd5b0be7d611d10",
    ),
}


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
    assert (
        manifest["converter_version"] == "catalog-product-sku-builder-v3-product-only"
    )
    assert "index_schema_version" not in manifest
    assert manifest["counts"]["rejected_records"] == 12
    assert (
        sum(
            value
            for key, value in manifest["counts"].items()
            if key.endswith("_products")
        )
        == 45286
    )
    assert (
        sum(value for key, value in manifest["counts"].items() if key.endswith("_skus"))
        == 261369
    )
    assert not any("search_unit" in key for key in manifest["counts"])
    product_ids: set[str] = set()
    sku_ids: set[str] = set()
    for name, partition in manifest["partitions"].items():
        product_path = Path(root, partition["products"]["path"])
        assert product_path.is_file()
        assert "search_units" not in partition
        expected_records, expected_sha256 = EXPECTED_PRODUCTS[name]
        assert partition["products"]["records"] == expected_records
        assert partition["products"]["sha256"] == expected_sha256
        assert BUILDER._sha256(product_path) == expected_sha256
        products = list(BUILDER._read_jsonl(product_path))
        assert len(products) == expected_records
        for product in products:
            BUILDER._validate_product_row(product)
            assert product["product_id"] not in product_ids
            product_ids.add(product["product_id"])
            for sku in product["skus"]:
                assert sku["sku_id"] not in sku_ids
                sku_ids.add(sku["sku_id"])
        coverage = json.loads(
            Path(root, partition["coverage"]["path"]).read_text(encoding="utf-8")
        )
        assert set(coverage) == {"products"}
    assert len(product_ids) == 45286
    assert len(sku_ids) == 261369
    assert not list(root.glob("**/search_units.jsonl"))
    for path in root.glob("**/*.jsonl"):
        for line in path.open(encoding="utf-8"):
            value = json.loads(line)
            assert "availability" not in value
            assert "simulated" not in value
            assert "provenance" not in value
