"""Deterministic duty and shipping estimate from chapter 12."""

from decimal import ROUND_HALF_UP, Decimal

from globex_agent.domain import (
    Currency,
    LandedCost,
    Platform,
    PricePoint,
    ResultStatus,
    ShippingCalcOutput,
    ToolIssue,
)

SHIPPING_RULE_VERSION = "demo-cn-landed-cost-v1"
_CENT = Decimal("0.01")

# The values intentionally mirror the simplified tables in chapter 12.
_DUTY_TABLE: dict[Platform, tuple[Decimal, str]] = {
    Platform.AMAZON: (Decimal("0.13"), "标准"),
    Platform.TAOBAO: (Decimal("0.00"), "免征"),
    Platform.SHOPEE: (Decimal("0.06"), "免征"),
    Platform.ALIEXPRESS: (Decimal("0.13"), "标准"),
    Platform.EBAY: (Decimal("0.20"), "高税"),
}
_SHIPPING_TABLE: dict[Platform, list[tuple[Decimal, Decimal, int]]] = {
    Platform.AMAZON: [
        (Decimal("0"), Decimal("85"), 12),
        (Decimal("0.5"), Decimal("130"), 10),
        (Decimal("2.0"), Decimal("240"), 8),
    ],
    Platform.TAOBAO: [
        (Decimal("0"), Decimal("8"), 4),
        (Decimal("0.5"), Decimal("12"), 4),
        (Decimal("2.0"), Decimal("25"), 5),
    ],
    Platform.SHOPEE: [
        (Decimal("0"), Decimal("35"), 9),
        (Decimal("0.5"), Decimal("60"), 9),
        (Decimal("2.0"), Decimal("120"), 7),
    ],
    Platform.ALIEXPRESS: [
        (Decimal("0"), Decimal("20"), 25),
        (Decimal("0.5"), Decimal("40"), 22),
        (Decimal("2.0"), Decimal("90"), 18),
    ],
    Platform.EBAY: [
        (Decimal("0"), Decimal("90"), 14),
        (Decimal("0.5"), Decimal("150"), 12),
        (Decimal("2.0"), Decimal("300"), 10),
    ],
}


def calculate_shipping(
    points: list[PricePoint],
    destination: str = "CN",
) -> ShippingCalcOutput:
    """Estimate landed cost from ``PriceCompare.ranked`` only."""

    destination = destination.strip().upper()
    if destination != "CN":
        issue = ToolIssue(
            code="unsupported_destination",
            message=f"演示规则仅支持目的地 CN，不支持 {destination or '(empty)'}",
        )
        return ShippingCalcOutput(
            destination=destination or "UNKNOWN",
            items=[],
            rule_version=SHIPPING_RULE_VERSION,
            status=ResultStatus.INVALID_INPUT,
            issues=[issue],
        )

    landed: list[LandedCost] = []
    for point in points[:30]:
        weight_kg = _guess_weight_kg(point)
        shipping_cny, eta_days = _estimate_shipping(weight_kg, point.platform)
        duty_rate, duty_tier = _DUTY_TABLE[point.platform]
        duty_cny = (point.price_cny * duty_rate).quantize(_CENT, rounding=ROUND_HALF_UP)
        total = point.price_cny + shipping_cny + duty_cny
        landed.append(
            LandedCost(
                item_id=point.item_id,
                platform=point.platform,
                price_cny=point.price_cny,
                shipping_cny=shipping_cny,
                duty_cny=duty_cny,
                landed_cny=total,
                eta_days=eta_days,
                duty_tier=duty_tier,
                same_group_id=point.same_group_id,
                title=point.title,
                brand=point.brand,
                rating=point.rating,
                category_path=point.category_path,
                attributes=point.attributes,
                currency=Currency.CNY,
                duty_rate=duty_rate,
                rule_version=SHIPPING_RULE_VERSION,
            )
        )

    landed.sort(key=lambda item: (item.landed_cny, item.platform.value, item.item_id))
    status = ResultStatus.OK if landed else ResultStatus.NO_RESULTS
    issues = (
        []
        if landed
        else [ToolIssue(code="no_shipping_input", message="PriceCompare 没有可估算的候选")]
    )
    return ShippingCalcOutput(
        destination=destination,
        items=landed,
        rule_version=SHIPPING_RULE_VERSION,
        status=status,
        issues=issues,
    )


def _guess_weight_kg(_point: PricePoint) -> Decimal:
    """Use the same 0.5 kg placeholder as the course's stage-two example."""

    return Decimal("0.5")


def _estimate_shipping(weight_kg: Decimal, platform: Platform) -> tuple[Decimal, int]:
    table = _SHIPPING_TABLE[platform]
    fee, eta_days = table[0][1], table[0][2]
    for minimum_weight, candidate_fee, candidate_days in table:
        if weight_kg >= minimum_weight:
            fee, eta_days = candidate_fee, candidate_days
    return fee, eta_days
