"""Core schemas for catalog, search, pricing, and recommendations.

Money is represented with ``Decimal`` so later price and shipping calculations do
not inherit binary floating-point rounding errors. All external data is rejected
when it contains undeclared fields; provider schema drift should be visible instead
of silently ignored.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

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
    """Base model that makes upstream schema changes fail loudly."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Platform(str, Enum):
    AMAZON = "amazon"
    SHOPEE = "shopee"
    ALIEXPRESS = "aliexpress"
    EBAY = "ebay"


class Currency(str, Enum):
    CNY = "CNY"
    USD = "USD"
    SGD = "SGD"
    EUR = "EUR"


class ProvenanceKind(str, Enum):
    SYNTHETIC = "synthetic"
    DERIVED = "derived"
    EXTERNAL_PUBLIC = "external_public"
    LICENSED_PROVIDER = "licensed_provider"


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


class Offer(StrictModel):
    """One platform listing for a canonical product."""

    offer_id: NonEmptyText
    platform: Platform
    listing_id: NonEmptyText
    price: PositiveMoney
    original_price: PositiveMoney | None = None
    currency: Currency
    shipping_fee: NonNegativeMoney = Decimal("0")
    in_stock: bool = True
    url: AnyHttpUrl
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    review_count: Annotated[int, Field(ge=0)] = 0
    source_updated_at: datetime
    provenance: DataProvenance

    @field_validator("source_updated_at")
    @classmethod
    def source_updated_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source_updated_at must include a timezone")
        return value

    @model_validator(mode="after")
    def original_price_must_not_be_lower(self) -> "Offer":
        if self.original_price is not None and self.original_price < self.price:
            raise ValueError("original_price cannot be lower than price")
        return self


class StandardItem(StrictModel):
    """A normalized product with one or more platform offers."""

    canonical_product_id: NonEmptyText
    title: NonEmptyText
    brand: NonEmptyText | None = None
    category_path: Annotated[list[NonEmptyText], Field(min_length=1)]
    description: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    offers: Annotated[list[Offer], Field(min_length=1)]
    provenance: DataProvenance

    @field_validator("attributes")
    @classmethod
    def attributes_must_have_non_empty_keys(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() for key in value):
            raise ValueError("attribute keys cannot be empty")
        return value

    @model_validator(mode="after")
    def offer_ids_must_be_unique(self) -> "StandardItem":
        offer_ids = [offer.offer_id for offer in self.offers]
        if len(offer_ids) != len(set(offer_ids)):
            raise ValueError("offer_id must be unique within a product")
        return self


class CatalogRecord(StrictModel):
    """One JSONL input record before listings are grouped into StandardItem."""

    canonical_product_id: NonEmptyText
    title: NonEmptyText
    brand: NonEmptyText | None = None
    category_path: Annotated[list[NonEmptyText], Field(min_length=1)]
    description: str = ""
    attributes: dict[str, str] = Field(default_factory=dict)
    offer: Offer
    provenance: DataProvenance

    @field_validator("attributes")
    @classmethod
    def attributes_must_have_non_empty_keys(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() for key in value):
            raise ValueError("attribute keys cannot be empty")
        return value


class UserProfile(StrictModel):
    user_id: NonEmptyText
    preferred_categories: list[NonEmptyText] = Field(default_factory=list)
    preferred_attributes: dict[str, str] = Field(default_factory=dict)
    blocked_materials: list[NonEmptyText] = Field(default_factory=list)
    blocked_brands: list[NonEmptyText] = Field(default_factory=list)
    positive_item_ids: list[NonEmptyText] = Field(default_factory=list)
    provenance: DataProvenance


class ToolIssue(StrictModel):
    """Structured warning or failure produced by a deterministic tool."""

    code: NonEmptyText
    message: NonEmptyText
    subject_id: NonEmptyText | None = None


class SearchRequest(StrictModel):
    query: NonEmptyText
    user_id: NonEmptyText | None = None
    budget: PositiveMoney | None = None
    currency: Currency = Currency.CNY
    platforms: set[Platform] = Field(default_factory=lambda: set(Platform))
    top_k: Annotated[int, Field(ge=1, le=50)] = 10
    hard_constraints: dict[str, Any] = Field(default_factory=dict)


class SearchCandidate(StrictModel):
    item: StandardItem
    matched_offer_ids: Annotated[list[NonEmptyText], Field(min_length=1)]
    score: Annotated[float, Field(ge=0, le=1)]
    reasons: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def matched_offers_must_exist(self) -> "SearchCandidate":
        available = {offer.offer_id for offer in self.item.offers}
        missing = set(self.matched_offer_ids) - available
        if missing:
            raise ValueError(f"matched offer ids do not belong to item: {sorted(missing)}")
        return self


class SearchResult(StrictModel):
    request: SearchRequest
    candidates: list[SearchCandidate] = Field(default_factory=list)
    total_catalog_items: Annotated[int, Field(ge=0)]
    matched_catalog_items: Annotated[int, Field(ge=0)] = 0
    truncated: bool = False
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class NormalizedOffer(StrictModel):
    canonical_product_id: NonEmptyText
    title: NonEmptyText
    offer: Offer
    normalized_price: PositiveMoney
    base_currency: Currency


class PriceComparison(StrictModel):
    canonical_product_id: NonEmptyText
    title: NonEmptyText
    offers: Annotated[list[NormalizedOffer], Field(min_length=1)]
    best_offer_id: NonEmptyText

    @model_validator(mode="after")
    def offers_must_be_consistent(self) -> "PriceComparison":
        offer_ids = {point.offer.offer_id for point in self.offers}
        if self.best_offer_id not in offer_ids:
            raise ValueError("best_offer_id must refer to one of the compared offers")
        if any(point.canonical_product_id != self.canonical_product_id for point in self.offers):
            raise ValueError("all compared offers must belong to canonical_product_id")
        minimum = min(point.normalized_price for point in self.offers)
        best = next(point for point in self.offers if point.offer.offer_id == self.best_offer_id)
        if best.normalized_price != minimum:
            raise ValueError("best_offer_id must refer to the lowest normalized price")
        return self


class PriceCompareResult(StrictModel):
    base_currency: Currency
    comparisons: list[PriceComparison] = Field(default_factory=list)
    fx_rate_version: NonEmptyText
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class ShippingQuote(StrictModel):
    canonical_product_id: NonEmptyText
    offer_id: NonEmptyText
    platform: Platform
    item_price: PositiveMoney
    shipping_fee: NonNegativeMoney
    tax_fee: NonNegativeMoney
    landed_price: PositiveMoney
    currency: Currency
    eta_days: Annotated[int, Field(ge=1)]
    tax_rate: Rate
    tax_tier: NonEmptyText
    is_estimate: bool = True
    rule_version: NonEmptyText

    @model_validator(mode="after")
    def landed_price_must_equal_components(self) -> "ShippingQuote":
        expected = self.item_price + self.shipping_fee + self.tax_fee
        if self.landed_price != expected:
            raise ValueError("landed_price must equal item_price + shipping_fee + tax_fee")
        return self


class ShippingResult(StrictModel):
    destination: NonEmptyText
    quotes: list[ShippingQuote] = Field(default_factory=list)
    rule_version: NonEmptyText
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class PickedItem(StrictModel):
    canonical_product_id: NonEmptyText
    title: NonEmptyText
    offer_id: NonEmptyText
    platform: Platform
    landed_price: PositiveMoney
    currency: Currency
    score: Annotated[float, Field(ge=0, le=1)]
    rating: Annotated[float, Field(ge=0, le=5)] | None = None
    reasons: Annotated[list[NonEmptyText], Field(min_length=1)]


class RejectedItem(StrictModel):
    canonical_product_id: NonEmptyText
    reason_code: NonEmptyText
    reason: NonEmptyText


class ItemPickerResult(StrictModel):
    picks: Annotated[list[PickedItem], Field(max_length=3)] = Field(default_factory=list)
    rejected: list[RejectedItem] = Field(default_factory=list)
    status: ResultStatus = ResultStatus.OK
    issues: list[ToolIssue] = Field(default_factory=list)


class ShoppingRecommendation(StrictModel):
    request: SearchRequest
    items: Annotated[list[PickedItem], Field(max_length=3)] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ShoppingSummaryResult(StrictModel):
    recommendation: ShoppingRecommendation
    final_text: NonEmptyText
    status: ResultStatus = ResultStatus.OK


class DeterministicPipelineResult(StrictModel):
    search: SearchResult
    price_comparison: PriceCompareResult
    shipping: ShippingResult
    selection: ItemPickerResult
    summary: ShoppingSummaryResult
