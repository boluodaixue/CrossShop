"""Deterministic cross-platform price comparison."""

from decimal import ROUND_HALF_UP, Decimal

from globex_agent.domain import (
    Currency,
    NormalizedOffer,
    PriceCompareResult,
    PriceComparison,
    ResultStatus,
    SearchResult,
    ToolIssue,
)

FX_RATE_VERSION = "demo-fx-2026-08-12-v1"
FX_RATES_TO_CNY: dict[Currency, Decimal] = {
    Currency.CNY: Decimal("1.00"),
    Currency.USD: Decimal("7.20"),
    Currency.SGD: Decimal("5.30"),
    Currency.EUR: Decimal("7.80"),
}
_CENT = Decimal("0.01")


def compare_prices(
    search_result: SearchResult,
    base_currency: Currency = Currency.CNY,
) -> PriceCompareResult:
    """Normalize matched offer prices and choose the sticker-price winner per item."""

    comparisons: list[PriceComparison] = []
    issues: list[ToolIssue] = []
    for candidate in search_result.candidates:
        matched_ids = set(candidate.matched_offer_ids)
        normalized: list[NormalizedOffer] = []
        for offer in candidate.item.offers:
            if offer.offer_id not in matched_ids:
                continue
            try:
                price = convert_money(offer.price, offer.currency, base_currency)
            except KeyError:
                issues.append(
                    ToolIssue(
                        code="unsupported_currency",
                        message=f"缺少 {offer.currency.value} 到 {base_currency.value} 的演示汇率",
                        subject_id=offer.offer_id,
                    )
                )
                continue
            normalized.append(
                NormalizedOffer(
                    canonical_product_id=candidate.item.canonical_product_id,
                    title=candidate.item.title,
                    offer=offer,
                    normalized_price=price,
                    base_currency=base_currency,
                )
            )

        normalized.sort(
            key=lambda point: (
                point.normalized_price,
                point.offer.platform.value,
                point.offer.offer_id,
            )
        )
        if not normalized:
            issues.append(
                ToolIssue(
                    code="no_comparable_offer",
                    message="候选商品没有可换算的有效报价",
                    subject_id=candidate.item.canonical_product_id,
                )
            )
            continue
        comparisons.append(
            PriceComparison(
                canonical_product_id=candidate.item.canonical_product_id,
                title=candidate.item.title,
                offers=normalized,
                best_offer_id=normalized[0].offer.offer_id,
            )
        )

    if not comparisons:
        status = ResultStatus.NO_RESULTS
    elif issues:
        status = ResultStatus.PARTIAL
    else:
        status = ResultStatus.OK
    return PriceCompareResult(
        base_currency=base_currency,
        comparisons=comparisons,
        fx_rate_version=FX_RATE_VERSION,
        status=status,
        issues=issues,
    )


def convert_money(amount: Decimal, source: Currency, target: Currency) -> Decimal:
    """Convert with the versioned illustrative table, rounded half-up to cents."""

    source_rate = FX_RATES_TO_CNY[source]
    target_rate = FX_RATES_TO_CNY[target]
    return (amount * source_rate / target_rate).quantize(_CENT, rounding=ROUND_HALF_UP)
