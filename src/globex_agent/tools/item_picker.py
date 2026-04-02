"""Deterministic hard-constraint filtering and candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from globex_agent.domain import (
    ItemPickerResult,
    PickedItem,
    RejectedItem,
    ResultStatus,
    SearchCandidate,
    SearchResult,
    ShippingQuote,
    ShippingResult,
    StandardItem,
    ToolIssue,
    UserProfile,
)
from globex_agent.tools.constraints import (
    SUPPORTED_HARD_CONSTRAINTS,
    check_constraint,
    normalize_text,
)


@dataclass(frozen=True)
class _Eligible:
    candidate: SearchCandidate
    quote: ShippingQuote


def pick_items(
    search_result: SearchResult,
    shipping_result: ShippingResult,
    profile: UserProfile | None = None,
    top_n: int = 3,
) -> ItemPickerResult:
    """Apply hard constraints first, then rank surviving canonical products."""

    request = search_result.request
    unsupported = sorted(set(request.hard_constraints) - SUPPORTED_HARD_CONSTRAINTS)
    if unsupported:
        issues = [
            ToolIssue(
                code="unsupported_hard_constraint",
                message=f"无法验证硬约束 {key}，为避免违规推荐已停止精挑",
                subject_id=key,
            )
            for key in unsupported
        ]
        return ItemPickerResult(status=ResultStatus.INVALID_INPUT, issues=issues)

    best_quotes = _best_quote_per_product(shipping_result.quotes)
    eligible: list[_Eligible] = []
    rejected: list[RejectedItem] = []
    for candidate in search_result.candidates:
        product_id = candidate.item.canonical_product_id
        quote = best_quotes.get(product_id)
        if quote is None:
            rejected.append(
                RejectedItem(
                    canonical_product_id=product_id,
                    reason_code="missing_shipping_quote",
                    reason="没有可用的到手价估算",
                )
            )
            continue
        failure = _hard_failure(candidate.item, quote, search_result, profile)
        if failure is not None:
            rejected.append(
                RejectedItem(
                    canonical_product_id=product_id,
                    reason_code=failure[0],
                    reason=failure[1],
                )
            )
            continue
        eligible.append(_Eligible(candidate, quote))

    picks = _rank_eligible(eligible, profile)[: max(1, min(top_n, 3))]
    if picks:
        status = ResultStatus.OK
        issues: list[ToolIssue] = []
    else:
        status = ResultStatus.NO_RESULTS
        issues = [
            ToolIssue(code="all_candidates_rejected", message="所有候选均违反硬约束或缺少报价")
        ]
    return ItemPickerResult(
        picks=picks,
        rejected=rejected,
        status=status,
        issues=issues,
    )


def _hard_failure(
    item: StandardItem,
    quote: ShippingQuote,
    search_result: SearchResult,
    profile: UserProfile | None,
) -> tuple[str, str] | None:
    request = search_result.request
    if request.budget is not None and quote.landed_price > request.budget:
        return (
            "over_budget",
            f"到手价 {quote.currency.value} {quote.landed_price} 超过预算 {request.budget}",
        )

    if profile is not None:
        if item.brand and any(
            normalize_text(blocked) == normalize_text(item.brand)
            for blocked in profile.blocked_brands
        ):
            return "blocked_brand", f"品牌 {item.brand} 在用户黑名单中"
        material = normalize_text(item.attributes.get("material", ""))
        for blocked in profile.blocked_materials:
            if normalize_text(blocked) in material:
                return "blocked_material", f"材质 {blocked} 在用户黑名单中"

    for key, expected in request.hard_constraints.items():
        check = check_constraint(item, key, expected)
        if not check.matched:
            return "hard_constraint_mismatch", check.reason
    return None


def _best_quote_per_product(quotes: list[ShippingQuote]) -> dict[str, ShippingQuote]:
    best: dict[str, ShippingQuote] = {}
    for quote in quotes:
        current = best.get(quote.canonical_product_id)
        if current is None or (quote.landed_price, quote.offer_id) < (
            current.landed_price,
            current.offer_id,
        ):
            best[quote.canonical_product_id] = quote
    return best


def _rank_eligible(
    eligible: list[_Eligible],
    profile: UserProfile | None,
) -> list[PickedItem]:
    if not eligible:
        return []
    prices = [entry.quote.landed_price for entry in eligible]
    minimum = min(prices)
    maximum = max(prices)

    picks: list[PickedItem] = []
    for entry in eligible:
        candidate = entry.candidate
        quote = entry.quote
        rating = _offer_rating(candidate.item, quote.offer_id)
        price_score = _price_score(quote.landed_price, minimum, maximum)
        rating_score = Decimal(str(rating / 5 if rating is not None else 0.5))
        preference_score = Decimal(str(_preference_alignment(candidate.item, profile)))
        score = (
            Decimal("0.45") * Decimal(str(candidate.score))
            + Decimal("0.30") * price_score
            + Decimal("0.15") * rating_score
            + Decimal("0.10") * preference_score
        )
        reasons = [
            "满足全部硬约束",
            f"到手价 {quote.currency.value} {quote.landed_price}",
        ]
        if preference_score > 0:
            reasons.append("命中用户画像偏好")
        elif rating is not None:
            reasons.append(f"商品评分 {rating:.1f}/5")
        else:
            reasons.append(f"查询相关度 {candidate.score:.2f}")
        picks.append(
            PickedItem(
                canonical_product_id=candidate.item.canonical_product_id,
                title=candidate.item.title,
                offer_id=quote.offer_id,
                platform=quote.platform,
                landed_price=quote.landed_price,
                currency=quote.currency,
                score=round(float(score), 4),
                rating=rating,
                reasons=reasons,
            )
        )

    picks.sort(
        key=lambda pick: (
            -pick.score,
            pick.landed_price,
            pick.canonical_product_id,
        )
    )
    return picks


def _price_score(price: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
    if minimum == maximum:
        return Decimal("1")
    return Decimal("1") - (price - minimum) / (maximum - minimum)


def _offer_rating(item: StandardItem, offer_id: str) -> float | None:
    return next((offer.rating for offer in item.offers if offer.offer_id == offer_id), None)


def _preference_alignment(item: StandardItem, profile: UserProfile | None) -> float:
    if profile is None:
        return 0.0
    signals = 0
    matched = 0
    if profile.positive_item_ids:
        signals += 1
        matched += item.canonical_product_id in profile.positive_item_ids
    if profile.preferred_categories:
        signals += 1
        categories = " ".join(item.category_path)
        matched += any(category in categories for category in profile.preferred_categories)
    for key, expected in profile.preferred_attributes.items():
        signals += 1
        matched += item.attributes.get(key) == expected
    return matched / signals if signals else 0.0
