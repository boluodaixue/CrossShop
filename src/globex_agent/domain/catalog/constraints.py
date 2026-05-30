"""Shared, fact-only hard constraints for item and variant search."""

from __future__ import annotations

from dataclasses import dataclass, field

from globex_agent.domain.catalog.exchange_rate import ExchangeRateTable
from globex_agent.domain.catalog.models import AvailabilityStatus, StandardItem
from globex_agent.domain.catalog.money import Money
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec


@dataclass(frozen=True)
class ConstraintDecision:
    eligible: bool
    rejection_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    matching_variant_ids: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str | None:
        return self.rejection_reasons[0] if self.rejection_reasons else None


class ConstraintEvaluator:
    """One rule implementation reused by recall and final item picking."""

    def __init__(self, rates: ExchangeRateTable | None = None) -> None:
        self._rates = rates or ExchangeRateTable()

    def evaluate(self, item: StandardItem, spec: ProductSearchSpec) -> ConstraintDecision:
        status = item.availability
        legacy = item.__dict__.get("is_available")
        if isinstance(legacy, bool):
            status = AvailabilityStatus.AVAILABLE if legacy else AvailabilityStatus.UNAVAILABLE
        if status is not AvailabilityStatus.AVAILABLE:
            reason = (
                "item_unavailable"
                if status is AvailabilityStatus.UNAVAILABLE
                else "item_availability_unknown"
            )
            return ConstraintDecision(False, [reason])
        if spec.platform and item.platform.value != spec.platform.strip().casefold():
            return ConstraintDecision(False, ["platform_mismatch"])
        if spec.locale and (
            item.locale is None or item.locale.value != spec.locale.strip().casefold()
        ):
            return ConstraintDecision(False, ["locale_mismatch"])
        if spec.brand and (
            item.brand is None or item.brand.casefold() != spec.brand.strip().casefold()
        ):
            return ConstraintDecision(False, ["brand_mismatch"])
        if spec.category and not _matches_category(item, spec.category):
            return ConstraintDecision(False, ["category_mismatch"])
        if spec.ship_to:
            destinations = explicit_ships_to(item)
            if destinations is None:
                return ConstraintDecision(False, ["ship_to_unknown"])
            if spec.ship_to.strip().upper() not in destinations:
                return ConstraintDecision(False, ["ship_to_unavailable"])
        material_decision = _evaluate_materials(item, spec)
        if material_decision is not None:
            return material_decision
        if not item.variants:
            if spec.price_max_major is None:
                return ConstraintDecision(True)
            if item.price_cny is None:
                return ConstraintDecision(False, ["price_unknown"])
            converted = _convert(item.price_cny, spec, self._rates)
            return ConstraintDecision(
                converted <= spec.price_max_major,
                ["over_price_cap"] if converted > spec.price_max_major else [],
            )
        available = [
            variant
            for variant in item.variants
            if variant.availability is AvailabilityStatus.AVAILABLE
        ]
        if not available:
            return ConstraintDecision(False, ["variant_unavailable"])
        if spec.price_max_major is None:
            return ConstraintDecision(True, matching_variant_ids=[v.variant_id for v in available])
        priced = [v for v in available if v.price_cny is not None]
        matching = [
            v.variant_id
            for v in priced
            if _convert(v.price_cny, spec, self._rates) <= spec.price_max_major
        ]
        if matching:
            warnings = (
                ["some_available_variant_prices_unknown"]
                if len(priced) < len(available)
                else []
            )
            return ConstraintDecision(True, warnings=warnings, matching_variant_ids=matching)
        if len(priced) < len(available):
            return ConstraintDecision(
                False, ["price_unknown"], warnings=["some_available_variant_prices_unknown"]
            )
        return ConstraintDecision(False, ["over_price_cap"])


def _convert(price, spec: ProductSearchSpec, rates: ExchangeRateTable) -> float:
    money = Money.from_major_units(float(price), "CNY")
    return (
        money.to_major_units()
        if spec.target_currency == "CNY"
        else rates.convert(money, spec.target_currency).to_major_units()
    )


def _evaluate_materials(item: StandardItem, spec: ProductSearchSpec) -> ConstraintDecision | None:
    if not spec.material_include and not spec.material_exclude:
        return None
    components = list(item.materials)
    for variant in item.variants:
        if variant.materials is not None:
            components.extend(variant.materials)
    names = {
        value.casefold()
        for component in components
        for value in (component.code or "", component.name)
    }
    includes = {value.strip().casefold() for value in spec.material_include}
    excludes = {value.strip().casefold() for value in spec.material_exclude}
    if excludes & names:
        return ConstraintDecision(False, ["material_excluded"])
    if excludes and not components and spec.material_unknown_policy == "reject":
        return ConstraintDecision(False, ["material_unknown"])
    if includes and not (includes & names):
        if not components and spec.material_unknown_policy == "reject":
            return ConstraintDecision(False, ["material_unknown"])
        if not components:
            return ConstraintDecision(True, warnings=["material_unknown"])
        return ConstraintDecision(False, ["material_not_included"])
    return ConstraintDecision(True)


def explicit_ships_to(item: StandardItem) -> frozenset[str] | None:
    if item.ships_to is not None:
        return frozenset(value.strip().upper() for value in item.ships_to if value.strip())
    return None


def _matches_category(item: StandardItem, requested: str) -> bool:
    needle = requested.strip().casefold()
    return any(needle in segment.casefold() for segment in item.category_path)
