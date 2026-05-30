"""Source adapters for ESCI and ShopSimulator product/task records."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from globex_agent.domain import (
    AvailabilityStatus,
    Currency,
    DataProvenance,
    MarketLocale,
    MaterialComponent,
    Platform,
    PriceSource,
    ProductAttribute,
    ProvenanceKind,
    RelevanceJudgment,
    RetrievalQuery,
    ShoppingTask,
    StandardItem,
    StandardItemVariant,
    VariantOption,
    canonical_options_serialization,
    canonical_variant_id,
)

ESCI_SOURCE_URL = "https://github.com/amazon-science/esci-data"
SHOPSIMULATOR_SOURCE_URL = "https://github.com/YYHDBL/shopping-grpo-longhorizon"

ESCI_LANGUAGE_BY_LOCALE = {
    MarketLocale.US: "en",
    MarketLocale.ES: "es",
    MarketLocale.JP: "ja",
}
ESCI_CURRENCY_BY_LOCALE = {
    MarketLocale.US: Currency.USD,
    MarketLocale.ES: Currency.EUR,
    MarketLocale.JP: Currency.JPY,
}
ESCI_GAIN_BY_LABEL = {
    "Exact": 1.0,
    "Substitute": 0.1,
    "Complement": 0.01,
    "Irrelevant": 0.0,
}
ESCI_CANONICAL_LABEL = {
    "E": "Exact",
    "S": "Substitute",
    "C": "Complement",
    "I": "Irrelevant",
    **{label: label for label in ESCI_GAIN_BY_LABEL},
}


def normalize_esci_product(row: dict[str, Any], *, ingested_at: datetime) -> StandardItem:
    """Map an official ESCI product row into the unified item contract."""

    locale = _market_locale(row.get("product_locale", row.get("locale")))
    if locale not in ESCI_LANGUAGE_BY_LOCALE:
        raise ValueError(f"unsupported ESCI locale: {locale.value}")
    product_id = _required_text(row.get("product_id"), "product_id")
    title = _required_text(row.get("product_title", row.get("title")), "product_title")
    description = _clean_text(row.get("product_description", row.get("description")))
    bullet = _clean_text(row.get("product_bullet_point", row.get("bullet")))
    brand = _clean_text(row.get("product_brand", row.get("brand"))) or None
    color = _clean_text(row.get("product_color", row.get("color")))
    attributes: list[ProductAttribute] = []
    if bullet:
        attributes.append(
            ProductAttribute(code="bullet_points", name="Bullet points", value=bullet)
        )
    if color:
        attributes.append(ProductAttribute(code="color", name="Color", value=color))

    return StandardItem(
        item_id=f"amazon:{locale.value}:{product_id}",
        same_group_id=f"amazon:{product_id}",
        platform=Platform.AMAZON,
        locale=locale,
        language=ESCI_LANGUAGE_BY_LOCALE[locale],
        title=title,
        description=description,
        brand=brand,
        category_path=["unknown"],
        price_cny=None,
        original_price_cny=None,
        currency_raw=ESCI_CURRENCY_BY_LOCALE[locale],
        price_source=PriceSource.UNAVAILABLE,
        attributes=attributes,
        ingested_at=ingested_at,
        provenance=DataProvenance(
            kind=ProvenanceKind.EXTERNAL_PUBLIC,
            source="Amazon ESCI Shopping Queries Dataset",
            source_record_id=f"{locale.value}:{product_id}",
            source_url=ESCI_SOURCE_URL,
            generated_at=ingested_at,
            notes="ESCI does not provide price, inventory, or product category fields.",
        ),
    )


def normalize_esci_query(
    row: dict[str, Any],
    *,
    split: str,
) -> RetrievalQuery:
    """Map one ESCI example row into the query contract."""

    locale = _market_locale(row.get("product_locale", row.get("locale")))
    return RetrievalQuery(
        query_id=f"esci:{locale.value}:{_required_text(row.get('query_id'), 'query_id')}",
        query=_required_text(row.get("query"), "query"),
        language=ESCI_LANGUAGE_BY_LOCALE[locale],
        platform=Platform.AMAZON,
        locale=locale,
        split=split,
        source="amazon-esci",
        source_split=_required_text(row.get("split"), "split"),
    )


def normalize_esci_judgment(row: dict[str, Any]) -> RelevanceJudgment:
    """Map one ESCI example row while preserving its graded label."""

    locale = _market_locale(row.get("product_locale", row.get("locale")))
    source_label = _required_text(row.get("esci_label"), "esci_label")
    try:
        label = ESCI_CANONICAL_LABEL[source_label]
        gain = ESCI_GAIN_BY_LABEL[label]
    except KeyError as exc:
        raise ValueError(f"unsupported ESCI label: {source_label}") from exc
    return RelevanceJudgment(
        query_id=f"esci:{locale.value}:{_required_text(row.get('query_id'), 'query_id')}",
        item_id=(f"amazon:{locale.value}:{_required_text(row.get('product_id'), 'product_id')}"),
        label=label,
        gain=gain,
    )


def normalize_shopsimulator_product(
    row: dict[str, Any],
    *,
    ingested_at: datetime,
) -> StandardItem:
    """Map one ShopSimulator item without collapsing unresolved variants."""

    asin = _required_text(row.get("asin"), "asin")
    pricing = _positive_prices(row.get("pricing"))
    distinct_prices = sorted(set(pricing))
    resolved_price = distinct_prices[0] if len(distinct_prices) == 1 else None
    category_path = _category_path(row)
    item_id = f"taobao:cn:{asin}"
    variants = _flatten_variants(row.get("customization_options"), Platform.TAOBAO, item_id)
    description = " ".join(
        part
        for part in (
            _clean_text(row.get("sub_title")),
            _clean_text(row.get("full_description")),
        )
        if part
    )
    attributes: list[ProductAttribute] = []
    shop_name = _clean_text(row.get("shop_name"))
    if shop_name:
        attributes.append(ProductAttribute(code="shop_name", name="店铺", value=shop_name))
    source_values = _string_list(row.get("attribute"))
    if source_values:
        attributes.append(
            ProductAttribute(
                code="source_attributes", name="来源属性", value=" / ".join(source_values)
            )
        )
    source_tag = _clean_text(row.get("tag"))
    if source_tag:
        attributes.append(ProductAttribute(code="source_tag", name="来源标签", value=source_tag))
    materials = _normalize_materials(row.get("material", row.get("材质", row.get("fabric"))))

    return StandardItem(
        item_id=item_id,
        same_group_id=f"taobao:{asin}",
        platform=Platform.TAOBAO,
        locale=MarketLocale.CN,
        language="zh",
        title=_required_text(row.get("title"), "title"),
        description=description,
        brand=None,
        category_path=category_path,
        price_cny=resolved_price if not variants else None,
        original_price_cny=None,
        currency_raw=Currency.CNY,
        price_source=(
            PriceSource.OBSERVED
            if resolved_price is not None and not variants
            else PriceSource.UNAVAILABLE
        ),
        attributes=attributes,
        materials=materials,
        variants=variants,
        availability=_availability(row.get("is_available", row.get("available")), variants),
        ingested_at=ingested_at,
        provenance=DataProvenance(
            kind=ProvenanceKind.EXTERNAL_PUBLIC,
            source="ShopSimulator fine_items_eval_train_all",
            source_record_id=asin,
            source_url=SHOPSIMULATOR_SOURCE_URL,
            generated_at=ingested_at,
            notes=(
                "Single observed price is exposed directly; a multi-price item stays "
                "price-unavailable until the task selects a concrete variant."
            ),
        ),
    )


def normalize_shopsimulator_task(
    row: dict[str, Any],
) -> tuple[ShoppingTask, RetrievalQuery, RelevanceJudgment]:
    """Derive the agent task and a single-target retrieval label from one product."""

    asin = _required_text(row.get("asin"), "asin")
    instructions = row.get("instructions")
    if not isinstance(instructions, list) or len(instructions) != 1:
        raise ValueError(f"ShopSimulator item {asin} must have exactly one instruction")
    instruction = instructions[0]
    if not isinstance(instruction, dict):
        raise ValueError(f"ShopSimulator item {asin} has an invalid instruction")
    source_split = _required_text(row.get("tag"), "tag")
    split = "test" if source_split == "eval" else "train"
    query_id = f"shopsim:{source_split}:{asin}"
    target_item_id = f"taobao:cn:{asin}"
    query = _required_text(instruction.get("instruction"), "instruction")
    task = ShoppingTask(
        task_id=query_id,
        query_id=query_id,
        query=query,
        simple_query=_clean_text(instruction.get("instruction_simple")),
        language="zh",
        platform=Platform.TAOBAO,
        locale=MarketLocale.CN,
        split=split,
        source_split=source_split,
        target_item_id=target_item_id,
        target_options=_string_list(
            instruction.get("instruction_options", instruction.get("options"))
        ),
        target_attributes=_string_list(instruction.get("attributes")),
        constraints={
            "options": _string_list(instruction.get("options")),
            "attributes": _string_list(instruction.get("attributes")),
        },
    )
    retrieval_query = RetrievalQuery(
        query_id=query_id,
        query=query,
        language="zh",
        platform=Platform.TAOBAO,
        locale=MarketLocale.CN,
        split=split,
        source="shopsimulator",
        source_split=source_split,
    )
    judgment = RelevanceJudgment(
        query_id=query_id,
        item_id=target_item_id,
        label="Exact",
        gain=1.0,
    )
    return task, retrieval_query, judgment


def _market_locale(value: object) -> MarketLocale:
    normalized = _required_text(value, "locale").lower()
    return MarketLocale(normalized)


def _required_text(value: object, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for entry in value if (text := _clean_text(entry))]


def _positive_prices(value: object) -> list[Decimal]:
    if not isinstance(value, list):
        return []
    prices: list[Decimal] = []
    for entry in value:
        try:
            price = Decimal(str(entry))
        except (InvalidOperation, TypeError, ValueError):
            continue
        if price.is_finite() and price > 0:
            prices.append(price.quantize(Decimal("0.01")))
    return prices


def _category_path(row: dict[str, Any]) -> list[str]:
    leaf_path = _clean_text(row.get("category"))
    parts = [part.strip() for part in leaf_path.replace(">", "›").split("›") if part.strip()]
    domain = _clean_text(row.get("domain_zh"))
    if domain and (not parts or parts[0] != domain):
        parts.insert(0, domain)
    return parts or ["未知类目"]


def _flatten_variants(
    value: object, platform: Platform, item_id: str
) -> list[StandardItemVariant]:
    if not isinstance(value, dict):
        return []
    records: list[dict[str, Any]] = []
    sequence = 0
    for option_name, option_values in sorted(value.items(), key=lambda entry: str(entry[0])):
        if not isinstance(option_values, list):
            continue
        for option in option_values:
            if not isinstance(option, dict):
                continue
            sequence += 1
            raw_price = _clean_text(option.get("price"))
            parsed_price = _positive_prices([raw_price])
            source_variant_id = _clean_text(option.get("asin")) or f"source-variant-{sequence}"
            typed_option = VariantOption(
                name=_clean_text(option_name),
                value=_clean_text(option.get("value")) or "未指定",
            )
            available = _availability(option.get("is_available"), [])
            records.append(
                {
                    "source_variant_id": source_variant_id,
                    "options": [typed_option],
                    "price_cny": parsed_price[0] if parsed_price else None,
                    "availability": available,
                }
            )
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(canonical_options_serialization(record["options"]), []).append(record)
    variants: list[StandardItemVariant] = []
    for fingerprint in sorted(groups):
        group = sorted(
            groups[fingerprint],
            key=lambda record: (
                record["source_variant_id"],
                str(record["price_cny"]),
                record["availability"].value,
            ),
        )
        chosen = group[0]
        conflict = len({record["price_cny"] for record in group}) > 1 or len(
            {record["availability"] for record in group}
        ) > 1
        variants.append(
            StandardItemVariant(
                variant_id=canonical_variant_id(platform, item_id, chosen["options"]),
                source_variant_id=chosen["source_variant_id"],
                options=chosen["options"],
                price_cny=None if conflict else chosen["price_cny"],
                price_source=(
                    PriceSource.UNAVAILABLE
                    if conflict or chosen["price_cny"] is None
                    else PriceSource.OBSERVED
                ),
                availability=AvailabilityStatus.UNKNOWN if conflict else chosen["availability"],
            )
        )
    return variants


def _availability(value: object, variants: list[StandardItemVariant]) -> AvailabilityStatus:
    if isinstance(value, bool):
        return AvailabilityStatus.AVAILABLE if value else AvailabilityStatus.UNAVAILABLE
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"available", "true", "1", "yes", "y", "在售"}:
            return AvailabilityStatus.AVAILABLE
        if normalized in {"unavailable", "false", "0", "no", "n", "下架", "缺货"}:
            return AvailabilityStatus.UNAVAILABLE
    if variants:
        statuses = {variant.availability for variant in variants}
        if AvailabilityStatus.AVAILABLE in statuses:
            return AvailabilityStatus.AVAILABLE
        if statuses == {AvailabilityStatus.UNAVAILABLE}:
            return AvailabilityStatus.UNAVAILABLE
    return AvailabilityStatus.UNKNOWN


def _normalize_materials(value: object) -> list[MaterialComponent]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, list) else [value]
    result: list[MaterialComponent] = []
    aliases = {
        "真皮": ("genuine_leather", "真皮"),
        "合成革": ("synthetic_leather", "合成革"),
        "蛋白皮": ("protein_leather", "蛋白皮"),
        "乳胶": ("latex", "乳胶"),
        "棉": ("cotton", "棉"),
    }
    for raw in values:
        name = _clean_text(raw)
        code, canonical = aliases.get(name, (None, name))
        if code is None:
            for alias, (alias_code, alias_name) in aliases.items():
                if alias in name:
                    code, canonical = alias_code, alias_name
                    break
        result.append(MaterialComponent(code=code, name=canonical))
    return result
