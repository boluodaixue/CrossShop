"""Deterministic shipping and tax estimates for the demo destination."""

from decimal import ROUND_HALF_UP, Decimal

from globex_agent.domain import (
    Platform,
    PriceCompareResult,
    ResultStatus,
    ShippingQuote,
    ShippingResult,
    ToolIssue,
)
from globex_agent.tools.price_compare import convert_money

SHIPPING_RULE_VERSION = "demo-cn-landed-cost-v1"
_CENT = Decimal("0.01")
_PLATFORM_RULES: dict[Platform, tuple[Decimal, str, int]] = {
    Platform.AMAZON: (Decimal("0.13"), "标准", 12),
    Platform.SHOPEE: (Decimal("0.06"), "优惠", 9),
    Platform.ALIEXPRESS: (Decimal("0.13"), "标准", 22),
    Platform.EBAY: (Decimal("0.20"), "高税", 14),
}


def calculate_shipping(
    price_result: PriceCompareResult,
    destination: str = "CN",
) -> ShippingResult:
    """Estimate landed cost from normalized price, listed shipping, and demo tax rules."""

    destination = destination.strip().upper()
    if destination != "CN":
        issue = ToolIssue(
            code="unsupported_destination",
            message=f"演示规则仅支持目的地 CN，不支持 {destination or '(empty)'}",
        )
        return ShippingResult(
            destination=destination or "UNKNOWN",
            quotes=[],
            rule_version=SHIPPING_RULE_VERSION,
            status=ResultStatus.INVALID_INPUT,
            issues=[issue],
        )

    quotes: list[ShippingQuote] = []
    for comparison in price_result.comparisons:
        for point in comparison.offers:
            tax_rate, tax_tier, eta_days = _PLATFORM_RULES[point.offer.platform]
            shipping_fee = convert_money(
                point.offer.shipping_fee,
                point.offer.currency,
                point.base_currency,
            )
            tax_fee = (point.normalized_price * tax_rate).quantize(_CENT, rounding=ROUND_HALF_UP)
            landed_price = point.normalized_price + shipping_fee + tax_fee
            quotes.append(
                ShippingQuote(
                    canonical_product_id=comparison.canonical_product_id,
                    offer_id=point.offer.offer_id,
                    platform=point.offer.platform,
                    item_price=point.normalized_price,
                    shipping_fee=shipping_fee,
                    tax_fee=tax_fee,
                    landed_price=landed_price,
                    currency=point.base_currency,
                    eta_days=eta_days,
                    tax_rate=tax_rate,
                    tax_tier=tax_tier,
                    rule_version=SHIPPING_RULE_VERSION,
                )
            )

    quotes.sort(
        key=lambda quote: (
            quote.landed_price,
            quote.canonical_product_id,
            quote.offer_id,
        )
    )
    status = ResultStatus.OK if quotes else ResultStatus.NO_RESULTS
    issues = (
        []
        if quotes
        else [ToolIssue(code="no_shipping_input", message="PriceCompare 没有可计算的报价")]
    )
    return ShippingResult(
        destination=destination,
        quotes=quotes,
        rule_version=SHIPPING_RULE_VERSION,
        status=status,
        issues=issues,
    )
