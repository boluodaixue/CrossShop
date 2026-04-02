"""Deterministic local ItemSearch implementation."""

from __future__ import annotations

import re
from unicodedata import normalize

from globex_agent.catalog import LocalCatalog
from globex_agent.domain import (
    ResultStatus,
    SearchCandidate,
    SearchRequest,
    SearchResult,
    StandardItem,
    ToolIssue,
    UserProfile,
)
from globex_agent.tools.constraints import SUPPORTED_HARD_CONSTRAINTS, check_constraint

_ASCII_TOKEN = re.compile(r"[a-z0-9%]+")
_CJK_SEGMENT = re.compile(r"[\u3400-\u9fff]+")


def search_items(
    catalog: LocalCatalog,
    request: SearchRequest,
    profile: UserProfile | None = None,
) -> SearchResult:
    """Recall local products with deterministic lexical and attribute scoring."""

    query_features = _features(request.query)
    issues = [
        ToolIssue(
            code="unsupported_constraint",
            message=f"ItemSearch 无法识别硬约束 {key}，交由 ItemPicker 拒绝处理",
            subject_id=key,
        )
        for key in sorted(request.hard_constraints)
        if key not in SUPPORTED_HARD_CONSTRAINTS
    ]

    ranked: list[SearchCandidate] = []
    for item in catalog.items:
        offers = [
            offer for offer in item.offers if offer.in_stock and offer.platform in request.platforms
        ]
        if not offers:
            continue

        filtered_item = item.model_copy(update={"offers": offers})
        score, reasons = _score_item(filtered_item, request, query_features, profile)
        if score <= 0:
            continue
        ranked.append(
            SearchCandidate(
                item=filtered_item,
                matched_offer_ids=[offer.offer_id for offer in offers],
                score=score,
                reasons=reasons,
            )
        )

    ranked.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.item.title.casefold(),
            candidate.item.canonical_product_id,
        )
    )
    matched_count = len(ranked)
    candidates = ranked[: request.top_k]
    if not candidates:
        status = ResultStatus.NO_RESULTS
        issues.append(ToolIssue(code="no_search_match", message="本地商品目录没有匹配该查询的候选"))
    elif issues:
        status = ResultStatus.PARTIAL
    else:
        status = ResultStatus.OK

    return SearchResult(
        request=request,
        candidates=candidates,
        total_catalog_items=len(catalog),
        matched_catalog_items=matched_count,
        truncated=matched_count > request.top_k,
        status=status,
        issues=issues,
    )


def _score_item(
    item: StandardItem,
    request: SearchRequest,
    query_features: set[str],
    profile: UserProfile | None,
) -> tuple[float, list[str]]:
    heading = " ".join([item.title, item.brand or "", *item.category_path])
    details = " ".join(
        [item.description, *(f"{key} {value}" for key, value in item.attributes.items())]
    )
    heading_overlap = _coverage(query_features, _features(heading))
    detail_overlap = _coverage(query_features, _features(details))

    checks = [
        check_constraint(item, key, expected) for key, expected in request.hard_constraints.items()
    ]
    supported = [check for check in checks if check.supported]
    constraint_score = (
        sum(check.matched for check in supported) / len(supported) if supported else 0.0
    )
    profile_score = _profile_alignment(item, profile)
    score = min(
        1.0,
        0.55 * heading_overlap
        + 0.30 * detail_overlap
        + 0.10 * constraint_score
        + 0.05 * profile_score,
    )
    score = round(score, 4)

    reasons: list[str] = []
    if heading_overlap > 0:
        reasons.append("标题或类目与查询匹配")
    if detail_overlap > 0:
        reasons.append("描述或属性与查询匹配")
    if supported:
        matched = sum(check.matched for check in supported)
        reasons.append(f"可识别硬约束匹配 {matched}/{len(supported)}")
    if profile_score > 0:
        reasons.append("命中用户画像偏好")
    return score, reasons or ["本地词项弱匹配"]


def _profile_alignment(item: StandardItem, profile: UserProfile | None) -> float:
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


def _coverage(query_features: set[str], document_features: set[str]) -> float:
    if not query_features:
        return 0.0
    return len(query_features & document_features) / len(query_features)


def _features(text: str) -> set[str]:
    normalized = normalize("NFKC", text).casefold()
    features = set(_ASCII_TOKEN.findall(normalized))
    for segment in _CJK_SEGMENT.findall(normalized):
        if len(segment) == 1:
            features.add(segment)
            continue
        features.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return features
