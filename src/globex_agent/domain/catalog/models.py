"""Course-aligned schemas for the deterministic Globex implementation.

The names follow chapters 09-1, 11, 12, and 14: platform-level products use
``item_id``; the same product across platforms is linked by ``same_group_id``;
tool outputs then flow through Candidate -> PricePoint -> LandedCost -> PickedItem.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveMoney = Annotated[
    Decimal,
    Field(gt=Decimal("0"), max_digits=12, decimal_places=2),
]
NonNegativeMoney = Annotated[
    Decimal,
    Field(ge=Decimal("0"), max_digits=12, decimal_places=2),
]
Rate = Annotated[
    Decimal,
    Field(ge=Decimal("0"), le=Decimal("1"), max_digits=5, decimal_places=4),
]


class StrictModel(BaseModel):
    """Reject undeclared fields so provider schema drift is visible."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Platform(str, Enum):
    AMAZON = "amazon"
    TAOBAO = "taobao"
    SHOPEE = "shopee"
    ALIEXPRESS = "aliexpress"
    EBAY = "ebay"


class MarketLocale(str, Enum):
    """Searchable market partition inside one logical platform catalog."""

    US = "us"
    ES = "es"
    JP = "jp"
    CN = "cn"


class Currency(str, Enum):
    CNY = "CNY"
    USD = "USD"
    SGD = "SGD"
    GBP = "GBP"
    EUR = "EUR"
    JPY = "JPY"


class ProvenanceKind(str, Enum):
    SYNTHETIC = "synthetic"
    DERIVED = "derived"
    EXTERNAL_PUBLIC = "external_public"
    LICENSED_PROVIDER = "licensed_provider"


class PriceSource(str, Enum):
    """Whether a price is observed, explicitly simulated, or unavailable."""

    OBSERVED = "observed"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"


class AvailabilityStatus(str, Enum):
    """Explicit three-state sale status; missing facts remain unknown."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


MATERIAL_TAXONOMY_VERSION = "material-taxonomy-v1"


class ProductAttribute(StrictModel):
    """A shared, non-transactional long-tail technical attribute."""

    code: NonEmptyText
    name: NonEmptyText
    value: str | int | float | bool
    unit: NonEmptyText | None = None


class MaterialComponent(StrictModel):
    """A normalized material fact; unresolved source text keeps code=None."""

    code: NonEmptyText | None = None
    name: NonEmptyText
    percentage: Annotated[Decimal, Field(ge=0, le=100, max_digits=5, decimal_places=2)] | None = (
        None
    )
    part: NonEmptyText | None = None
    taxonomy_version: NonEmptyText = MATERIAL_TAXONOMY_VERSION


class VariantOption(StrictModel):
    code: NonEmptyText | None = None
    name: NonEmptyText
    value: NonEmptyText


def normalize_variant_text(value: str) -> str:
    """Return the canonical comparison form for a user-selectable option."""

    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).casefold()


def canonical_options_serialization(options: list[VariantOption]) -> str:
    """Serialize options independently of source order, price, or stock."""

    values = sorted(
        (
            {
                "name": normalize_variant_text(option.name),
                "value": normalize_variant_text(option.value),
            }
            for option in options
        ),
        key=lambda value: (value["name"], value["value"]),
    )
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def canonical_variant_id(
    platform: Platform | str, item_id: str, options: list[VariantOption]
) -> str:
    """Build a stable internal SKU ID from platform, item, and normalized options."""

    platform_value = platform.value if isinstance(platform, Platform) else str(platform)
    payload = (
        f"{normalize_variant_text(platform_value)}|{normalize_variant_text(item_id)}|"
        f"{canonical_options_serialization(options)}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{platform_value}:variant:{digest}"


class StandardItemVariant(StrictModel):
    variant_id: NonEmptyText
    source_variant_id: NonEmptyText | None = None
    options: list[VariantOption] = Field(default_factory=list)
    price_cny: PositiveMoney | None = None
    original_price_cny: PositiveMoney | None = None
    price_source: PriceSource = PriceSource.UNAVAILABLE
    availability: AvailabilityStatus = AvailabilityStatus.UNKNOWN
    materials: list[MaterialComponent] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_variant(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "source_variant_id" not in payload and payload.get("variant_id"):
            payload["source_variant_id"] = payload["variant_id"]
        if "price_cny" not in payload and "price_major" in payload:
            payload["price_cny"] = payload.pop("price_major")
        if "price_cny" in payload and "price_source" not in payload:
            payload["price_source"] = PriceSource.OBSERVED
        payload.pop("currency", None)
        if "availability" not in payload and "is_available" in payload:
            raw = payload.pop("is_available")
            payload["availability"] = (
                AvailabilityStatus.AVAILABLE if raw else AvailabilityStatus.UNAVAILABLE
            )
        if "options" not in payload:
            name = payload.pop("option_name", None) or payload.pop("name", None)
            value_text = payload.pop("value", None) or payload.pop("spec", None)
            payload["options"] = [{"name": name or "规格", "value": value_text or "未指定"}]
        return payload

    @model_validator(mode="after")
    def validate_price(self) -> StandardItemVariant:
        if self.price_cny is None:
            if self.original_price_cny is not None:
                raise ValueError("variant original_price_cny requires price_cny")
            if self.price_source is not PriceSource.UNAVAILABLE:
                raise ValueError("variant missing price requires price_source=unavailable")
        elif self.price_source is PriceSource.UNAVAILABLE:
            raise ValueError("variant price cannot use price_source=unavailable")
        if self.original_price_cny is not None and self.original_price_cny < self.price_cny:
            raise ValueError("variant original_price_cny cannot be lower than price_cny")
        return self


class ResultStatus(str, Enum):
    OK = "ok"
    NO_RESULTS = "no_results"
    PARTIAL = "partial"
    INVALID_INPUT = "invalid_input"


class DataProvenance(StrictModel):
    """Where a record came from and how it may be interpreted."""

    kind: ProvenanceKind
    source: NonEmptyText
    source_record_id: NonEmptyText | None = None
    source_url: AnyHttpUrl | None = None
    generated_at: datetime
    notes: str = ""

    @field_validator("generated_at")
    @classmethod
    def generated_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value


class StandardItem(StrictModel):
    """One normalized platform product, following chapter 09-1."""

    item_id: NonEmptyText
    same_group_id: NonEmptyText
    platform: Platform
    locale: MarketLocale | None = None
    language: Literal["en", "es", "ja", "zh"] | None = None
    title: NonEmptyText
    description: str = ""
    brand: NonEmptyText | None = None
    category_path: Annotated[list[NonEmptyText], Field(min_length=1)]
    price_cny: PositiveMoney | None = None
    original_price_cny: PositiveMoney | None = None
    currency_raw: Currency | None = None
    price_source: PriceSource = PriceSource.UNAVAILABLE
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    review_count: Annotated[int, Field(ge=0)] = 0
    attributes: list[ProductAttribute] = Field(default_factory=list)
    materials: list[MaterialComponent] = Field(default_factory=list)
    variants: list[StandardItemVariant] = Field(default_factory=list)
    availability: AvailabilityStatus = AvailabilityStatus.UNKNOWN
    ships_to: list[NonEmptyText] | None = None
    source_updated_at: datetime | None = None
    ingested_at: datetime

    # Demo-only trace fields used by the controlled local dataset.
    url: AnyHttpUrl | None = None
    provenance: DataProvenance

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_item(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "availability" not in payload and "is_available" in payload:
            raw = payload.pop("is_available")
            payload["availability"] = (
                AvailabilityStatus.AVAILABLE if raw else AvailabilityStatus.UNAVAILABLE
            )
        if "price_cny" in payload and "price_source" not in payload:
            payload["price_source"] = PriceSource.OBSERVED
        attributes = payload.get("attributes", [])
        if isinstance(attributes, dict):
            typed: list[dict] = []
            materials: list[dict] = list(payload.get("materials", []))
            for key, raw in attributes.items():
                if str(key).casefold() in {"material", "materials", "材质", "fabric"}:
                    materials.append({"code": None, "name": str(raw)})
                    continue
                if str(key).casefold() in {"ships_to", "ship_to"}:
                    payload["ships_to"] = raw
                    continue
                scalar = raw if isinstance(raw, (str, int, float, bool)) else str(raw)
                typed.append({"code": str(key), "name": str(key), "value": scalar})
            payload["attributes"] = typed
            payload["materials"] = materials
        return payload

    @field_validator("source_updated_at", "ingested_at")
    @classmethod
    def timestamps_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def identifiers_and_prices_must_be_consistent(self) -> StandardItem:
        expected_prefix = f"{self.platform.value}:"
        if not self.item_id.startswith(expected_prefix):
            raise ValueError(f"item_id must start with {expected_prefix}")
        if self.variants and (self.price_cny is not None or self.original_price_cny is not None):
            raise ValueError("items with variants must keep prices on variants only")
        if self.variants and self.price_source is not PriceSource.UNAVAILABLE:
            raise ValueError("items with variants require price_source=unavailable")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("variant_id must be unique within item")
        option_fingerprints = [canonical_options_serialization(v.options) for v in self.variants]
        if len(option_fingerprints) != len(set(option_fingerprints)):
            raise ValueError("normalized options must be unique within item")
        if self.price_cny is None:
            if self.original_price_cny is not None:
                raise ValueError("original_price_cny requires price_cny")
            if self.price_source is not PriceSource.UNAVAILABLE:
                raise ValueError("missing price_cny requires price_source=unavailable")
        elif self.price_source is PriceSource.UNAVAILABLE:
            raise ValueError("available price_cny cannot use price_source=unavailable")
        if (
            self.original_price_cny is not None
            and self.price_cny is not None
            and self.original_price_cny < self.price_cny
        ):
            raise ValueError("original_price_cny cannot be lower than price_cny")
        return self

    @property
    def is_available(self) -> bool:
        """Read-only compatibility view; canonical storage is availability."""
        legacy_override = self.__dict__.get("is_available")
        if isinstance(legacy_override, bool):
            return legacy_override
        return self.availability is AvailabilityStatus.AVAILABLE

    def effective_materials(
        self, variant: StandardItemVariant | None = None
    ) -> list[MaterialComponent]:
        if variant is not None and variant.materials is not None:
            return list(variant.materials)
        return list(self.materials)


class UserProfile(StrictModel):
    user_id: NonEmptyText
    preferred_categories: list[NonEmptyText] = Field(default_factory=list)
    preferred_attributes: dict[str, str] = Field(default_factory=dict)
    blocked_materials: list[NonEmptyText] = Field(default_factory=list)
    blocked_brands: list[NonEmptyText] = Field(default_factory=list)
    positive_item_ids: list[NonEmptyText] = Field(default_factory=list)
    provenance: DataProvenance


class SearchRequest(StrictModel):
    query: NonEmptyText
    user_id: NonEmptyText | None = None
    budget: PositiveMoney | None = None
    currency: Currency = Currency.CNY
    platforms: set[Platform] = Field(default_factory=lambda: set(Platform))
    top_k: Annotated[int, Field(ge=1, le=50)] = 10
    hard_constraints: dict[str, Any] = Field(default_factory=dict)


class ToolIssue(StrictModel):
    """Structured warning or failure produced by a deterministic tool."""

    code: NonEmptyText
    message: NonEmptyText
    subject_id: NonEmptyText | None = None


class Candidate(StrictModel):
    """Single ItemSearch candidate, following chapter 11."""

    item_id: NonEmptyText
    platform: Platform
    locale: MarketLocale | None = None
    title: NonEmptyText
    price: PositiveMoney | None = None
    currency: Currency | None = None
    price_source: PriceSource = PriceSource.OBSERVED
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    sales: Annotated[int, Field(ge=0)] | None = None
    image_url: AnyHttpUrl | None = None
    attributes: dict[str, str] = Field(default_factory=dict)

    # Local-demo supplements. With no external item-detail service, the fixed
    # chain must carry these fields forward itself.
    same_group_id: NonEmptyText
    brand: NonEmptyText | None = None
    category_path: Annotated[list[NonEmptyText], Field(min_length=1)]


class ItemSearchOutput(StrictModel):
    platform: Platform
    locale: MarketLocale | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    total_recall: Annotated[int, Field(ge=0)] = 0
    truncated: bool = False

    # Deterministic-demo diagnostics, kept outside the course's four core fields.
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class PricePoint(StrictModel):
    """Currency-normalized listing, following chapter 12."""

    item_id: NonEmptyText
    platform: Platform
    title: NonEmptyText
    price_local: PositiveMoney
    currency_local: Currency
    price_cny: PositiveMoney
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    sales: Annotated[int, Field(ge=0)] | None = None
    note: str | None = None

    # Local-demo supplements used by the deterministic picker.
    same_group_id: NonEmptyText
    brand: NonEmptyText | None = None
    category_path: Annotated[list[NonEmptyText], Field(min_length=1)]
    attributes: dict[str, str] = Field(default_factory=dict)


class PriceCompareOutput(StrictModel):
    base_currency: Currency = Currency.CNY
    ranked: list[PricePoint] = Field(default_factory=list)
    cheapest_per_platform: dict[Platform, NonEmptyText] = Field(default_factory=dict)

    # The course catalog's same-product relation is retained for local inspection.
    cheapest_per_group: dict[str, NonEmptyText] = Field(default_factory=dict)
    fx_rate_version: NonEmptyText
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class LandedCost(StrictModel):
    """Estimated landed cost, following chapter 12."""

    item_id: NonEmptyText
    platform: Platform
    price_cny: PositiveMoney
    shipping_cny: NonNegativeMoney
    duty_cny: NonNegativeMoney
    landed_cny: PositiveMoney
    eta_days: Annotated[int, Field(ge=1)]
    duty_tier: Literal["免征", "标准", "高税"]

    # Local-demo supplements forwarded because no item-detail service exists.
    same_group_id: NonEmptyText
    title: NonEmptyText
    brand: NonEmptyText | None = None
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    category_path: Annotated[list[NonEmptyText], Field(min_length=1)]
    attributes: dict[str, str] = Field(default_factory=dict)
    currency: Currency = Currency.CNY
    duty_rate: Rate
    is_estimate: bool = True
    rule_version: NonEmptyText

    @model_validator(mode="after")
    def landed_price_must_equal_components(self) -> LandedCost:
        expected = self.price_cny + self.shipping_cny + self.duty_cny
        if self.landed_cny != expected:
            raise ValueError("landed_cny must equal price_cny + shipping_cny + duty_cny")
        return self


class ShippingCalcOutput(StrictModel):
    destination: NonEmptyText
    items: list[LandedCost] = Field(default_factory=list)

    # Deterministic-demo diagnostics.
    rule_version: NonEmptyText
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class PickedItem(StrictModel):
    """Selected item, following chapter 14."""

    item_id: NonEmptyText
    platform: Platform
    landed_cny: PositiveMoney
    score: Annotated[float, Field(ge=0, le=1)]
    reasons: Annotated[list[NonEmptyText], Field(max_length=3)] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    # Local-demo display fields.
    same_group_id: NonEmptyText
    title: NonEmptyText
    currency: Currency = Currency.CNY
    rating: Annotated[float, Field(ge=0, le=5)] | None = None


class ItemPickerOutput(StrictModel):
    picks: Annotated[list[PickedItem], Field(max_length=3)] = Field(default_factory=list)
    rejected_brief: Annotated[list[NonEmptyText], Field(max_length=8)] = Field(default_factory=list)

    # Deterministic-demo diagnostics.
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class ShoppingSummaryOutput(StrictModel):
    final_text: NonEmptyText
    picks: Annotated[list[PickedItem], Field(max_length=3)] = Field(default_factory=list)
    learned_preferences: list[str] = Field(default_factory=list)

    # Deterministic template supplements; the course version delegates prose to an LLM.
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    status: ResultStatus = ResultStatus.OK


class DeterministicPipelineResult(StrictModel):
    search: list[ItemSearchOutput]
    price_comparison: PriceCompareOutput
    shipping: ShippingCalcOutput
    selection: ItemPickerOutput
    summary: ShoppingSummaryOutput
