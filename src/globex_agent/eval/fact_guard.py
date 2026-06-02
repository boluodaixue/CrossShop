"""Deterministic source-boundary checks for Agent final replies."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True, slots=True)
class FactViolation:
    code: str
    message: str
    excerpt: str


_PRICE_PATTERNS = (
    re.compile(r"(?P<symbol>[¥￥$])\s*(?P<amount>\d+(?:\.\d+)?)"),
    re.compile(r"(?P<amount>\d+(?:\.\d+)?)\s*(?:元|人民币|CNY|USD|美元)"),
)
_STORE_OR_STOCK_RE = re.compile(r"店铺|商家|店家|库存|现货|有货|缺货")
_NEGATED_STORE_OR_STOCK_RE = re.compile(
    r"(?:未|没有|无|无法|不能|不代表|不证明|不提供).{0,8}(?:店铺|商家|店家|库存|现货|有货|缺货)"
)
_STORE_VERIFICATION_RE = re.compile(
    r"(?:向|联系|咨询).{0,4}(?:店铺|商家|店家).{0,4}(?:确认|询问|核实)"
)
_CATEGORY_CONTEXT_RE = re.compile(r"品类|参考|档位|区间|aggregate|category", re.IGNORECASE)
_PRODUCT_CONTEXT_RE = re.compile(r"商品|这款|该款|SKU|货号|具体", re.IGNORECASE)
_CLAUSE_BOUNDARY_RE = re.compile(r"[，,。！？；;\n]")


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


def _collect_product_prices(facts: Iterable[dict[str, Any]]) -> set[Decimal]:
    prices: set[Decimal] = set()
    for fact in facts:
        for key in ("price_major", "price_min_major", "price_max_major"):
            prices.update(_decimal_values(fact.get(key)))
        landed = fact.get("landed_price")
        if isinstance(landed, dict):
            for key in ("subtotal_major", "shipping_major", "duty_major", "landed_major"):
                prices.update(_decimal_values(landed.get(key)))
        for variant in fact.get("variants") or []:
            if isinstance(variant, dict):
                prices.update(_decimal_values(variant.get("price_major")))
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


def _near(text: str, start: int, end: int, radius: int = 24) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _claim_segment(text: str, start: int, end: int) -> str:
    left_boundary = max(
        (match.start() for match in _CLAUSE_BOUNDARY_RE.finditer(text, 0, start)),
        default=0,
    )
    right_match = _CLAUSE_BOUNDARY_RE.search(text, end)
    right_boundary = right_match.start() if right_match else len(text)
    return text[left_boundary:right_boundary]


def validate_final_response(
    text: str,
    *,
    product_facts: Iterable[dict[str, Any]] = (),
    category_insights: Iterable[dict[str, Any]] = (),
) -> tuple[FactViolation, ...]:
    """Check claims against the two distinct tool-source classes.

    This is deliberately a conservative P0 guard, not a natural-language
    truth classifier.  A judge can still inspect the full transcript, while
    these checks prevent an unsupported specific price or a category tier from
    silently becoming a product fact.
    """

    product_facts = list(product_facts)
    category_insights = list(category_insights)
    product_prices = _collect_product_prices(product_facts)
    category_prices = _collect_category_prices(category_insights)
    violations: list[FactViolation] = []

    for pattern in _PRICE_PATTERNS:
        for match in pattern.finditer(text):
            amount = Decimal(match.group("amount")).quantize(Decimal("0.01"))
            context = _near(text, match.start(), match.end())
            segment = _claim_segment(text, match.start(), match.end())
            has_product_context = bool(_PRODUCT_CONTEXT_RE.search(segment))
            has_category_context = bool(
                _CATEGORY_CONTEXT_RE.search(segment)
                or (
                    not has_product_context
                    and _CATEGORY_CONTEXT_RE.search(context)
                )
            )
            if amount in product_prices:
                continue
            if amount in category_prices:
                if has_product_context:
                    violations.append(
                        FactViolation(
                            "category_price_as_product_fact",
                            "品类参考价格被表述成了具体商品事实",
                            context,
                        )
                    )
                elif not has_category_context:
                    violations.append(
                        FactViolation(
                            "unsourced_specific_price",
                            "具体价格没有来自 product_search_tool，也未标明品类参考口径",
                            context,
                        )
                    )
                continue
            if has_category_context and not has_product_context:
                continue
            violations.append(
                FactViolation(
                    "unsourced_specific_price",
                    "具体价格没有在可审计的工具结果中找到来源",
                    context,
                )
            )

    store_or_stock_text = _STORE_VERIFICATION_RE.sub("", text)
    if (
        _STORE_OR_STOCK_RE.search(store_or_stock_text)
        and not _NEGATED_STORE_OR_STOCK_RE.search(text)
    ):
        has_explicit_store = any(
            fact.get("shop") or fact.get("store") or fact.get("store_name")
            for fact in product_facts
        )
        has_explicit_inventory = any(
            fact.get("availability") is not None for fact in product_facts
        )
        if (re.search(r"店铺|商家|店家", text) and not has_explicit_store) or (
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
