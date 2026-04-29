"""Platform-aware query, judgment, and shopping-task contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from globex_agent.domain.catalog.models import (
    MarketLocale,
    NonEmptyText,
    Platform,
    StrictModel,
)


class RetrievalQuery(StrictModel):
    """One immutable query routed to a platform/locale catalog partition."""

    query_id: NonEmptyText
    query: NonEmptyText
    language: Literal["en", "es", "ja", "zh"]
    platform: Platform
    locale: MarketLocale
    split: Literal["train", "dev", "test"]
    source: NonEmptyText
    source_split: NonEmptyText


class RelevanceJudgment(StrictModel):
    """One query-item judgment used by retrieval metrics."""

    query_id: NonEmptyText
    item_id: NonEmptyText
    label: NonEmptyText
    gain: float = Field(ge=0)


class ShoppingTask(StrictModel):
    """Agentic task kept separate from the product catalog and retrieval qrels."""

    task_id: NonEmptyText
    query_id: NonEmptyText
    query: NonEmptyText
    simple_query: str = ""
    language: Literal["en", "es", "ja", "zh"]
    platform: Platform
    locale: MarketLocale
    split: Literal["train", "dev", "test"]
    source_split: NonEmptyText
    target_item_id: NonEmptyText
    target_options: list[NonEmptyText] = Field(default_factory=list)
    target_attributes: list[NonEmptyText] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def target_must_match_platform(self) -> ShoppingTask:
        if not self.target_item_id.startswith(f"{self.platform.value}:"):
            raise ValueError("target_item_id must match the task platform")
        return self
