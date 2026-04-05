"""Deterministic admission gate for generated CategoryCard rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from globex_agent.category_insight.models import CategoryCard

MIN_CONFIDENCE = 0.5
_PRICE_TIER_PATTERN = re.compile(
    r"^便宜款\s+\d+(?:\.\d+)?-\d+(?:\.\d+)?\s*/\s*"
    r"中档\s+\d+(?:\.\d+)?-\d+(?:\.\d+)?\s*/\s*"
    r"高端\s+\d+(?:\.\d+)?-\d+(?:\.\d+)?$"
)


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    reason: str
    card: CategoryCard | None = None


def admit_card(raw: dict[str, Any]) -> AdmissionResult:
    """Apply schema, confidence, evidence, and type-format gates in order."""

    try:
        card = CategoryCard.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        return AdmissionResult(False, f"schema: {location}: {first['msg']}")

    if card.confidence < MIN_CONFIDENCE:
        return AdmissionResult(False, f"confidence {card.confidence} < {MIN_CONFIDENCE}")

    if card.card_type == "bestseller":
        if "：" not in card.summary or "/" not in card.summary:
            return AdmissionResult(False, "bestseller summary must use category：form / form")
        for evidence in card.raw_evidence:
            parts = [part.strip() for part in evidence.split("|")]
            if len(parts) != 3:
                return AdmissionResult(False, "bestseller evidence must be name | price | reason")
            try:
                float(parts[1])
            except ValueError:
                return AdmissionResult(False, "bestseller evidence price must be numeric")
    elif card.card_type == "attribute":
        if "：" not in card.summary or "%" not in card.summary:
            return AdmissionResult(False, "attribute summary must contain ： and percentages")
        distribution = card.summary.split("：", 1)[1]
        for token in distribution.split("/"):
            parts = token.strip().rsplit(" ", 1)
            if len(parts) != 2 or not parts[1].endswith("%"):
                return AdmissionResult(False, "attribute values must use value number%")
            try:
                float(parts[1].rstrip("%"))
            except ValueError:
                return AdmissionResult(False, "attribute percentage must be numeric")
    elif not _PRICE_TIER_PATTERN.fullmatch(card.summary):
        return AdmissionResult(
            False,
            "price_range summary must contain three finite 便宜款/中档/高端 ranges",
        )

    return AdmissionResult(True, "ok", card)
