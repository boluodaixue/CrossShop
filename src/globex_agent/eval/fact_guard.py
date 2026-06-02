"""Deterministic source-boundary checks for Agent final replies."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True, slots=True)
class FactViolation:
    code: str
    message: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class _PriceClaim:
    amount: Decimal
    start: int
    end: int
    segment: str
    range_bounds: tuple[Decimal, Decimal] | None = None


_PRICE_PATTERNS = (
    re.compile(r"(?P<symbol>[¥￥$])\s*(?P<amount>\d+(?:\.\d+)?)"),
    re.compile(r"(?P<amount>\d+(?:\.\d+)?)\s*(?:元|人民币|CNY|USD|美元)"),
)
_RANGE_RE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*(?:-|–|—|~|～|至|到)\s*"
    r"(?P<high>\d+(?:\.\d+)?)"
)
_STORE_OR_STOCK_RE = re.compile(r"店铺|商家|店家|店|库存|现货|有货|缺货")
_NEGATED_STORE_OR_STOCK_RE = re.compile(
    r"(?:未|没有|无|无法|不能|不代表|不证明|不提供).{0,8}(?:店铺|商家|店家|店|库存|现货|有货|缺货)"
)
_STORE_VERIFICATION_RE = re.compile(
    r"(?:向|联系|咨询).{0,4}(?:店铺|商家|店家).{0,4}(?:确认|询问|核实)"
)
_NAMED_STORE_RE = re.compile(r"(?P<name>[\u4e00-\u9fffA-Za-z0-9_-]{2,})(?:店|商店|店铺)")
_CATEGORY_CONTEXT_RE = re.compile(
    r"品类|参考|档位|区间|便宜款|中档|高端|价位|aggregate|category",
    re.IGNORECASE,
)
_PRODUCT_CONTEXT_RE = re.compile(r"商品|这款|该款|SKU|货号|具体", re.IGNORECASE)
_VARIANT_SCOPE_RE = re.compile(
    r"颜色|配色|款式|规格|型号|接口|选项|变体|不同|各(?:种|个)?|按.{0,4}(?:颜色|款式|规格|型号)|同价",
    re.IGNORECASE,
)
_NON_PRODUCT_PRICE_RE = re.compile(
    r"预算|比如|例如|以内|上限|运费|关税|税费|到手价|合计|小计",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[，,。！？；;\n]")
_BRACKET_CODE_RE = re.compile(r"(?:\[|【)\s*[A-Za-z]{1,5}\d{1,5}\s*(?:\]|】)")
_QUANTITY_SUFFIX_RE = re.compile(
    r"(?:x|×)\d+$|\d+(?:个|件)$|(?:一个|一件|单条|套装)$",
    re.IGNORECASE,
)
_GENERIC_VARIANT_SUFFIX_RE = re.compile(r"(?:款式|规格|型号|配色|款|色|版|型)$")


def _decimal_values(value: Any) -> set[Decimal]:
    values: set[Decimal] = set()
    if isinstance(value, bool) or value is None:
        return values
    if isinstance(value, (int, float, Decimal, str)):
        try:
            values.add(Decimal(str(value)).quantize(Decimal("0.01")))
        except (InvalidOperation, ValueError):
            return values
    return values


def _decimal_value(value: Any) -> Decimal | None:
    values = _decimal_values(value)
    return next(iter(values), None)


def _available_variant_entries(fact: dict[str, Any]) -> list[tuple[int, dict[str, Any], Decimal]]:
    entries: list[tuple[int, dict[str, Any], Decimal]] = []
    for index, variant in enumerate(fact.get("variants") or []):
        if not isinstance(variant, dict):
            continue
        if variant.get("availability") in {"unavailable", "out_of_stock"}:
            continue
        price = _decimal_value(variant.get("price_major"))
        if price is not None:
            entries.append((index, variant, price))
    return entries


def _collect_product_prices(facts: Iterable[dict[str, Any]]) -> set[Decimal]:
    prices: set[Decimal] = set()
    for fact in facts:
        if fact.get("variants"):
            continue
        for key in ("price_major", "product_price_major"):
            prices.update(_decimal_values(fact.get(key)))
        landed = fact.get("landed_price")
        if isinstance(landed, dict):
            for key in ("subtotal_major", "shipping_major", "duty_major", "landed_major"):
                prices.update(_decimal_values(landed.get(key)))
    return prices


def _collect_category_prices(insights: Iterable[dict[str, Any]]) -> set[Decimal]:
    prices: set[Decimal] = set()
    for item in insights:
        payload = item.get("insights", item)
        if not isinstance(payload, dict):
            continue
        for tier in payload.get("price_tiers") or []:
            if not isinstance(tier, dict):
                continue
            bounds = tier.get("range_cny") or []
            if isinstance(bounds, (list, tuple)):
                for bound in bounds:
                    prices.update(_decimal_values(bound))
    return prices


def _near(text: str, start: int, end: int, radius: int = 32) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _claim_segment(text: str, start: int, end: int) -> str:
    left_boundary = max(
        (match.start() for match in _CLAUSE_BOUNDARY_RE.finditer(text, 0, start)),
        default=0,
    )
    right_match = _CLAUSE_BOUNDARY_RE.search(text, end)
    right_boundary = right_match.start() if right_match else len(text)
    return text[left_boundary:right_boundary]


def _canonical_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = _BRACKET_CODE_RE.sub("", text)
    text = _QUANTITY_SUFFIX_RE.sub("", text)
    return "".join(
        char for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _variant_aliases(value: Any) -> tuple[str, ...]:
    canonical = _canonical_text(value)
    if len(canonical) < 2:
        return ()
    aliases = {canonical}
    stem = _GENERIC_VARIANT_SUFFIX_RE.sub("", canonical)
    if len(stem) >= 2:
        aliases.add(stem)
    return tuple(sorted(aliases, key=len, reverse=True))


def _longest_common_substring(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    longest = 0
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, 1):
            value = previous[index - 1] + 1 if left_char == right_char else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest


def _alias_score(alias: str, segment: str) -> int:
    normalized_segment = _canonical_text(segment)
    if alias in normalized_segment:
        return len(alias) + 100
    if len(alias) >= 4:
        common = _longest_common_substring(alias, normalized_segment)
        return common if common >= 2 else 0
    return 0


def _variant_matches(
    fact: dict[str, Any], amount: Decimal, segment: str
) -> tuple[bool, bool]:
    """Return (supported, ambiguous) using only this fact's variants."""

    hits_by_alias: dict[str, set[int]] = {}
    prices_by_index: dict[int, Decimal] = {}
    for index, variant, price in _available_variant_entries(fact):
        prices_by_index[index] = price
        raw_values: list[Any] = [variant.get("display_name")]
        for option in variant.get("options") or []:
            if isinstance(option, dict):
                raw_values.append(option.get("value"))
        for raw_value in raw_values:
            for alias in _variant_aliases(raw_value):
                if _alias_score(alias, segment):
                    hits_by_alias.setdefault(alias, set()).add(index)

    if not hits_by_alias:
        return False, False
    if any(len(indices) > 1 for indices in hits_by_alias.values()):
        return False, True
    hits_by_alias = {
        alias: indices
        for alias, indices in hits_by_alias.items()
        if any(prices_by_index[index] == amount for index in indices)
    }
    if not hits_by_alias:
        return False, False
    matched_indices = {index for indices in hits_by_alias.values() for index in indices}
    return bool(matched_indices), False


def _variant_scope_context(text: str, segment: str) -> bool:
    return bool(_VARIANT_SCOPE_RE.search(segment) or _VARIANT_SCOPE_RE.search(text))


def _has_product_context(text: str) -> bool:
    cleaned = re.sub(r"(?:非|不是)具体商品(?:价|价格)?", "", text)
    return bool(_PRODUCT_CONTEXT_RE.search(cleaned))


def _fact_labels(fact: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _canonical_text(fact.get(key))
        for key in ("item_id", "title", "brand", "store", "store_name")
        if fact.get(key)
    )


def _contextual_facts(
    facts: list[dict[str, Any]], text: str, claim: _PriceClaim
) -> list[dict[str, Any]]:
    context = _canonical_text(_near(text, claim.start, claim.end, radius=160))
    return [
        fact
        for fact in facts
        if any(len(label) >= 2 and label in context for label in _fact_labels(fact))
    ]


def _relevant_facts(
    facts: list[dict[str, Any]], text: str, claim: _PriceClaim
) -> list[dict[str, Any]]:
    contextual = _contextual_facts(facts, text, claim)
    return contextual or facts


def _range_supported(
    facts: list[dict[str, Any]], text: str, claim: _PriceClaim
) -> bool:
    if claim.range_bounds is None or not _variant_scope_context(
        _near(text, claim.start, claim.end, radius=72), claim.segment
    ):
        return False
    low, high = claim.range_bounds
    candidates = []
    for fact in _relevant_facts(facts, text, claim):
        prices = {price for _, _, price in _available_variant_entries(fact)}
        if prices and (min(prices), max(prices)) == (low, high):
            candidates.append(fact)
    if len(candidates) == 1:
        return True
    return len(_contextual_facts(candidates, text, claim)) == 1


def _same_price_variant_scope_supported(
    facts: list[dict[str, Any]], text: str, claim: _PriceClaim, amount: Decimal
) -> bool:
    if not _variant_scope_context(_near(text, claim.start, claim.end, radius=72), claim.segment):
        return False
    candidates = []
    for fact in _relevant_facts(facts, text, claim):
        prices = {price for _, _, price in _available_variant_entries(fact)}
        if prices == {amount}:
            candidates.append(fact)
    if len(candidates) == 1:
        return True
    return len(_contextual_facts(candidates, text, claim)) == 1


def _item_price_supported(
    facts: list[dict[str, Any]], text: str, claim: _PriceClaim, amount: Decimal
) -> bool:
    candidates = [
        fact
        for fact in _relevant_facts(facts, text, claim)
        if not fact.get("variants")
        and amount
        in (
            _decimal_values(fact.get("price_major"))
            | _decimal_values(fact.get("product_price_major"))
            | _decimal_values(fact.get("price_min_major"))
            | _decimal_values(fact.get("price_max_major"))
        )
    ]
    if len(candidates) == 1:
        return True
    if len(candidates) > 1:
        return len(_contextual_facts(candidates, text, claim)) == 1
    return False


def _iter_price_claims(text: str) -> list[_PriceClaim]:
    ranges = tuple(
        match for match in _RANGE_RE.finditer(text) if _range_has_price_context(text, match)
    )
    raw: list[tuple[Decimal, int, int]] = []
    for pattern in _PRICE_PATTERNS:
        for match in pattern.finditer(text):
            raw.append(
                (
                    Decimal(match.group("amount")).quantize(Decimal("0.01")),
                    match.start("amount"),
                    match.end("amount"),
                )
            )
    for range_match in ranges:
        for group in ("low", "high"):
            raw.append(
                (
                    Decimal(range_match.group(group)).quantize(Decimal("0.01")),
                    range_match.start(group),
                    range_match.end(group),
                )
            )
    claims: list[_PriceClaim] = []
    seen_spans: set[tuple[Decimal, int, int]] = set()
    for amount, start, end in sorted(raw, key=lambda value: (value[1], value[2])):
        span_key = (amount, start, end)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        range_bounds = None
        for range_match in ranges:
            if range_match.start() <= start and end <= range_match.end():
                range_bounds = (
                    Decimal(range_match.group("low")).quantize(Decimal("0.01")),
                    Decimal(range_match.group("high")).quantize(Decimal("0.01")),
                )
                break
        claims.append(
            _PriceClaim(
                amount=amount,
                start=start,
                end=end,
                segment=_claim_segment(text, start, end),
                range_bounds=range_bounds,
            )
        )
    return claims


def _range_has_price_context(text: str, match: re.Match[str]) -> bool:
    before = text[max(0, match.start() - 8) : match.start()]
    after = text[match.end() : min(len(text), match.end() + 8)]
    return bool(
        re.search(r"[¥￥$]|价格|价位|区间|预算", before)
        or re.search(r"元|人民币|CNY|USD|美元", after, re.IGNORECASE)
    )


def _is_non_product_price_context(claim: _PriceClaim, text: str) -> bool:
    context = _near(text, claim.start, claim.end, radius=72)
    if not _NON_PRODUCT_PRICE_RE.search(context):
        return False
    return not (_PRODUCT_CONTEXT_RE.search(claim.segment) and "预算" not in claim.segment)


def _store_claim_supported(text: str, facts: list[dict[str, Any]]) -> bool:
    known = {
        str(fact.get(key)).casefold()
        for fact in facts
        for key in ("shop", "store", "store_name")
        if fact.get(key)
    }
    if not known:
        return False
    if any(value in text.casefold() for value in known):
        return True
    for match in _NAMED_STORE_RE.finditer(text):
        if match.group("name").casefold() not in {"这家", "店铺", "商家"}:
            return False
    return True


def validate_final_response(
    text: str,
    *,
    product_facts: Iterable[dict[str, Any]] = (),
    category_insights: Iterable[dict[str, Any]] = (),
) -> tuple[FactViolation, ...]:
    """Check final claims against exposed product/category evidence only."""

    product_facts = list(product_facts)
    category_insights = list(category_insights)
    category_prices = _collect_category_prices(category_insights)
    violations: list[FactViolation] = []
    seen_price_violations: list[tuple[str, Decimal, int, int]] = []

    for claim in _iter_price_claims(text):
        amount = claim.amount
        if _is_non_product_price_context(claim, text):
            continue
        context = _near(text, claim.start, claim.end)
        has_product_context = _has_product_context(claim.segment)
        has_category_context = bool(
            _CATEGORY_CONTEXT_RE.search(claim.segment)
            or (not has_product_context and _CATEGORY_CONTEXT_RE.search(context))
        )
        supported = _item_price_supported(product_facts, text, claim, amount)
        if not supported:
            variant_candidates: list[dict[str, Any]] = []
            for fact in _relevant_facts(product_facts, text, claim):
                if not fact.get("variants"):
                    continue
                matched, _ambiguous = _variant_matches(fact, amount, claim.segment)
                if matched:
                    variant_candidates.append(fact)
            if len(variant_candidates) == 1:
                supported = True
            elif len(variant_candidates) > 1:
                supported = len(_contextual_facts(variant_candidates, text, claim)) == 1
            supported = supported or _range_supported(product_facts, text, claim)
            supported = supported or _same_price_variant_scope_supported(
                product_facts, text, claim, amount
            )
        if supported:
            continue
        if amount in category_prices:
            if has_product_context:
                code = "category_price_as_product_fact"
                message = "品类参考价格被表述成了具体商品事实"
            elif not has_category_context:
                code = "unsourced_specific_price"
                message = "具体价格没有来自 product_search_tool，也未标明品类参考口径"
            else:
                continue
        elif has_category_context and not has_product_context:
            continue
        else:
            code = "unsourced_specific_price"
            message = "具体价格没有在可审计的工具结果中找到来源"
        duplicate = any(
            old_code == code
            and old_amount == amount
            and old_start < claim.end
            and claim.start < old_end
            for old_code, old_amount, old_start, old_end in seen_price_violations
        )
        if duplicate:
            continue
        seen_price_violations.append((code, amount, claim.start, claim.end))
        violations.append(FactViolation(code, message, context))

    store_or_stock_text = _STORE_VERIFICATION_RE.sub("", text)
    if (
        _STORE_OR_STOCK_RE.search(store_or_stock_text)
        and not _NEGATED_STORE_OR_STOCK_RE.search(text)
    ):
        has_explicit_store = any(
            bool(fact.get("shop") or fact.get("store") or fact.get("store_name"))
            for fact in product_facts
        )
        has_explicit_inventory = any(
            fact.get("availability") in {"available", "unavailable"}
            or fact.get("inventory", {}).get("status") in {"available", "unavailable"}
            for fact in product_facts
        )
        if (
            re.search(r"店铺|商家|店家|店", text)
            and (not has_explicit_store or not _store_claim_supported(text, product_facts))
        ) or (
            re.search(r"库存|现货|有货|缺货", text) and not has_explicit_inventory
        ):
            violations.append(
                FactViolation(
                    "unsupported_store_or_inventory",
                    "回复包含未被 product_search_tool 证明的店铺或库存事实",
                    _near(text, *_STORE_OR_STOCK_RE.search(text).span()),
                )
            )

    return tuple(violations)
