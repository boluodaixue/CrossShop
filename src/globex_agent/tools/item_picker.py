"""Deterministic hard-constraint filtering and ItemPicker ranking."""

from __future__ import annotations

from decimal import Decimal

from globex_agent.domain import (
    ItemPickerOutput,
    LandedCost,
    PickedItem,
    ResultStatus,
    SearchRequest,
    ToolIssue,
    UserProfile,
)
from globex_agent.tools.constraints import (
    SUPPORTED_HARD_CONSTRAINTS,
    check_constraint,
    normalize_text,
)


def pick_items(
    landed: list[LandedCost],
    request: SearchRequest,
    user_profile: UserProfile | None = None,
    top_n: int = 3,
) -> ItemPickerOutput:
    """Pick at most three items from ``ShippingCalc.items``.

    The course signature receives natural-language preferences and optional
    CategoryInsight. Stage two has neither an LLM nor CategoryInsight, so this
    local implementation injects the already parsed request and user profile.
    """

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
        return ItemPickerOutput(status=ResultStatus.INVALID_INPUT, issues=issues)

    eligible: list[LandedCost] = []
    failures_by_group: dict[str, list[str]] = {}
    for cost in landed:
        failure = _hard_failure(cost, request, user_profile)
        if failure is not None:
            failures_by_group.setdefault(cost.same_group_id, []).append(failure)
            continue
        eligible.append(cost)

    eligible_groups = {cost.same_group_id for cost in eligible}
    rejected = [
        f"{group_id}：{reasons[0]}"
        for group_id, reasons in failures_by_group.items()
        if group_id not in eligible_groups
    ]
    picks = _take_unique_groups(
        _rank_eligible(eligible, user_profile),
        max(1, min(top_n, 3)),
    )
    if picks:
        status = ResultStatus.OK
        issues: list[ToolIssue] = []
    else:
        status = ResultStatus.NO_RESULTS
        issues = [
            ToolIssue(code="all_candidates_rejected", message="所有候选均违反硬约束或缺少到手价")
        ]
    return ItemPickerOutput(
        picks=picks,
        rejected_brief=rejected[:8],
        status=status,
        issues=issues,
    )


def _hard_failure(
    cost: LandedCost,
    request: SearchRequest,
    profile: UserProfile | None,
) -> str | None:
    if request.budget is not None and cost.landed_cny > request.budget:
        return f"到手价 CNY {cost.landed_cny} 超过预算 {request.budget}"

    if profile is not None:
        if cost.brand and any(
            normalize_text(blocked) == normalize_text(cost.brand)
            for blocked in profile.blocked_brands
        ):
            return f"品牌 {cost.brand} 在用户黑名单中"
        material = normalize_text(cost.attributes.get("material", ""))
        for blocked in profile.blocked_materials:
            if normalize_text(blocked) in material:
                return f"材质 {blocked} 在用户黑名单中"

    for key, expected in request.hard_constraints.items():
        check = check_constraint(cost, key, expected)
        if not check.matched:
            return check.reason
    return None


def _take_unique_groups(ranked: list[PickedItem], top_n: int) -> list[PickedItem]:
    picks: list[PickedItem] = []
    seen_groups: set[str] = set()
    for item in ranked:
        if item.same_group_id in seen_groups:
            continue
        seen_groups.add(item.same_group_id)
        picks.append(item)
        if len(picks) == top_n:
            break
    return picks


def _rank_eligible(
    eligible: list[LandedCost],
    profile: UserProfile | None,
) -> list[PickedItem]:
    if not eligible:
        return []
    prices = [item.landed_cny for item in eligible]
    minimum = min(prices)
    maximum = max(prices)

    picks: list[PickedItem] = []
    for item in eligible:
        price_score = _price_score(item.landed_cny, minimum, maximum)
        rating_score = Decimal(str(item.rating / 5 if item.rating is not None else 0.5))
        preference_score = Decimal(str(_preference_alignment(item, profile)))
        eta_score = Decimal("1") if item.eta_days <= 12 else Decimal("0")
        duty_score = Decimal("1") if item.duty_tier == "免征" else Decimal("0")
        score = (
            Decimal("0.25") * price_score
            + Decimal("0.15") * rating_score
            + Decimal("0.20") * preference_score
            + Decimal("0.20") * eta_score
            + Decimal("0.20") * duty_score
        )
        reasons = [f"到手价 CNY {item.landed_cny}"]
        if item.eta_days <= 12:
            reasons.append(f"预计 {item.eta_days} 天到手")
        if item.duty_tier == "免征":
            reasons.append("演示规则中的免征档")
        elif preference_score > 0:
            reasons.append("命中用户画像偏好")
        elif item.rating is not None:
            reasons.append(f"商品评分 {item.rating:.1f}/5")

        picks.append(
            PickedItem(
                item_id=item.item_id,
                platform=item.platform,
                landed_cny=item.landed_cny,
                score=round(float(score), 4),
                reasons=reasons[:3],
                flags=[],
                same_group_id=item.same_group_id,
                title=item.title,
                currency=item.currency,
                rating=item.rating,
            )
        )

    picks.sort(key=lambda pick: (-pick.score, pick.landed_cny, pick.item_id))
    return picks


def _price_score(price: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
    if minimum == maximum:
        return Decimal("1")
    return Decimal("1") - (price - minimum) / (maximum - minimum)


def _preference_alignment(item: LandedCost, profile: UserProfile | None) -> float:
    if profile is None:
        return 0.0
    signals = 0
    matched = 0
    if profile.positive_item_ids:
        signals += 1
        matched += item.item_id in profile.positive_item_ids
    if profile.preferred_categories:
        signals += 1
        categories = " ".join(item.category_path)
        matched += any(category in categories for category in profile.preferred_categories)
    for key, expected in profile.preferred_attributes.items():
        signals += 1
        matched += item.attributes.get(key) == expected
    return matched / signals if signals else 0.0
