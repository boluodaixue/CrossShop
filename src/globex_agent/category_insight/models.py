"""Course-aligned schemas for the CategoryInsight knowledge base."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from globex_agent.domain.models import NonEmptyText, StrictModel

CategoryCardType = Literal["bestseller", "attribute", "price_range"]
CategoryInsightDepth = Literal["quick", "deep"]


class CategoryCard(StrictModel):
    """One compact category-level fact card from chapter 13.

    Provenance deliberately lives in a sidecar dataset. Keeping this model to
    the seven course fields prevents indexing metadata from drifting into the RAG
    document contract.
    """

    card_id: NonEmptyText
    category: NonEmptyText
    card_type: CategoryCardType
    summary: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]
    raw_evidence: Annotated[
        list[
            Annotated[
                str,
                StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
            ]
        ],
        Field(min_length=1, max_length=3),
    ]
    last_updated: NonEmptyText
    confidence: Annotated[float, Field(ge=0, le=1)]


class Bestseller(StrictModel):
    name: NonEmptyText
    typical_price_cny: Annotated[float, Field(gt=0)]
    why_popular: NonEmptyText


class AttributeDist(StrictModel):
    name: NonEmptyText
    distribution: dict[NonEmptyText, Annotated[float, Field(ge=0, le=1)]]


class PriceTier(StrictModel):
    tier: Literal["budget", "mid", "premium"]
    range_cny: tuple[Annotated[float, Field(ge=0)], Annotated[float, Field(gt=0)]]
    notes: NonEmptyText


class CategoryInsightOutput(StrictModel):
    category: NonEmptyText
    components: list[NonEmptyText] = Field(default_factory=list)
    bestsellers: list[Bestseller] = Field(default_factory=list)
    attributes: list[AttributeDist] = Field(default_factory=list)
    price_tiers: list[PriceTier] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0, le=1)] = 0.0
