"""Shared deterministic interpretation of supported hard constraints."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol
from unicodedata import normalize

SUPPORTED_HARD_CONSTRAINTS = frozenset(
    {
        "capacity_l",
        "connectivity",
        "excluded_material",
        "form_factor",
        "layout",
        "max_weight_g",
        "noise_canceling",
        "switch",
        "usage",
        "water_resistant",
    }
)


class HasAttributes(Protocol):
    attributes: dict[str, str]


@dataclass(frozen=True)
class ConstraintCheck:
    supported: bool
    matched: bool
    reason: str


def check_constraint(item: HasAttributes, key: str, expected: Any) -> ConstraintCheck:
    """Evaluate one supported constraint without guessing unknown semantics."""

    if key not in SUPPORTED_HARD_CONSTRAINTS:
        return ConstraintCheck(False, False, f"不支持的硬约束：{key}")

    actual = item.attributes.get(key)
    if key == "excluded_material":
        material = _text(item.attributes.get("material", ""))
        excluded = _text(expected)
        matched = bool(excluded) and excluded not in material
        return ConstraintCheck(True, matched, f"材质不得包含 {expected}")

    if key == "max_weight_g":
        weight = _decimal(item.attributes.get("weight_g"))
        maximum = _decimal(expected)
        matched = weight is not None and maximum is not None and weight <= maximum
        return ConstraintCheck(True, matched, f"重量不得超过 {expected}g")

    if key == "capacity_l":
        capacity = _decimal(item.attributes.get("capacity_l"))
        wanted = _decimal(expected)
        matched = capacity is not None and wanted is not None and capacity == wanted
        return ConstraintCheck(True, matched, f"容量必须为 {expected}L")

    if key == "noise_canceling" and _text(expected) == "required":
        matched = _text(actual) in {"active", "hybrid", "yes", "true"}
        return ConstraintCheck(True, matched, "必须支持主动降噪")

    actual_text = _text(actual)
    expected_text = _text(expected)
    if key in {"connectivity", "switch", "usage"}:
        matched = bool(actual_text and expected_text) and (
            expected_text in actual_text or actual_text in expected_text
        )
    else:
        matched = bool(actual_text and expected_text) and actual_text == expected_text
    return ConstraintCheck(True, matched, f"{key} 必须匹配 {expected}")


def normalize_text(value: object) -> str:
    return _text(value)


def _text(value: object) -> str:
    return normalize("NFKC", str(value or "")).strip().casefold()


def _decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
