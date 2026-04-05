"""Deterministic cross-platform price comparison from chapter 12."""

from decimal import ROUND_HALF_UP, Decimal

from globex_agent.domain import (
    Candidate,
    Currency,
    PriceCompareOutput,
    PricePoint,
    ResultStatus,
    ToolIssue,
)

FX_RATE_VERSION = "demo-fx-2026-08-12-v1"
FX_RATES_TO_CNY: dict[Currency, Decimal] = {
    Currency.CNY: Decimal("1.00"),
    Currency.USD: Decimal("7.18"),
    Currency.SGD: Decimal("5.32"),
    Currency.GBP: Decimal("9.05"),
    Currency.EUR: Decimal("7.78"),
    Currency.JPY: Decimal("0.046"),
}
_CENT = Decimal("0.01")


def compare_prices(
    candidates: list[Candidate],
    base_currency: Currency = Currency.CNY,
    top_n: int = 12,
) -> PriceCompareOutput:
    """Normalize currencies and return the cheapest ``PricePoint`` rows.

    This follows the course boundary: PriceCompare compares item prices only; it
    does not calculate shipping or duty and it does not compute a quality score.
    """

    top_n = max(1, min(top_n, 30))
    issues: list[ToolIssue] = []
    points: list[PricePoint] = []
    for candidate in candidates[:100]:
        if candidate.price is None or candidate.currency is None:
            issues.append(
                ToolIssue(
                    code="price_unavailable",
                    message="公开商品数据没有可用于比价的观测价格",
                    subject_id=candidate.item_id,
                )
            )
            continue
        try:
            normalized_price = convert_money(candidate.price, candidate.currency, base_currency)
        except KeyError:
            issues.append(
                ToolIssue(
                    code="unsupported_currency",
                    message=(
                        f"缺少 {candidate.currency.value} 到 {base_currency.value} "
                        "的演示汇率"
                    ),
                    subject_id=candidate.item_id,
                )
            )
            continue
        points.append(
            PricePoint(
                item_id=candidate.item_id,
                platform=candidate.platform,
                title=candidate.title,
                price_local=candidate.price,
                currency_local=candidate.currency,
                price_cny=normalized_price,
                rating=candidate.rating,
                sales=candidate.sales,
                note=_pack_note(candidate),
                same_group_id=candidate.same_group_id,
                brand=candidate.brand,
                category_path=candidate.category_path,
                attributes=candidate.attributes,
            )
        )

    points.sort(key=lambda point: (point.price_cny, point.platform.value, point.item_id))
    ranked = points[:top_n]
    cheapest_per_platform: dict = {}
    cheapest_per_group: dict[str, str] = {}
    for point in points:
        cheapest_per_platform.setdefault(point.platform, point.item_id)
        cheapest_per_group.setdefault(point.same_group_id, point.item_id)

    if not ranked:
        status = ResultStatus.NO_RESULTS
        issues.append(ToolIssue(code="no_comparable_item", message="没有可换算的候选商品"))
    elif issues:
        status = ResultStatus.PARTIAL
    else:
        status = ResultStatus.OK
    return PriceCompareOutput(
        base_currency=base_currency,
        ranked=ranked,
        cheapest_per_platform=cheapest_per_platform,
        cheapest_per_group=cheapest_per_group,
        fx_rate_version=FX_RATE_VERSION,
        status=status,
        issues=issues,
    )


def convert_money(amount: Decimal, source: Currency, target: Currency) -> Decimal:
    """Convert with the versioned demo table, rounded half-up to cents."""

    source_rate = FX_RATES_TO_CNY[source]
    target_rate = FX_RATES_TO_CNY[target]
    return (amount * source_rate / target_rate).quantize(_CENT, rounding=ROUND_HALF_UP)


def _pack_note(candidate: Candidate) -> str | None:
    if candidate.price is None or candidate.currency is None:
        return None
    pack_size = candidate.attributes.get("pack_size")
    try:
        count = int(pack_size) if pack_size is not None else 1
    except (TypeError, ValueError):
        return None
    if count > 1:
        unit_price = (candidate.price / count).quantize(_CENT, rounding=ROUND_HALF_UP)
        return f"一套 {count} 件，等价单件 {unit_price} {candidate.currency.value}"
    return None
