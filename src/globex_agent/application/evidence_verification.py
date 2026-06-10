"""Small, online-only verification protocol for recommendation answers.

The model is allowed to produce only a natural-language answer and selections.
This module validates mechanical boundaries, builds a deterministic semantic
projection for one whole-answer judge, and hydrates cards from the same frozen
Top-K. It deliberately has no claim ledger, field catalog, or per-claim API.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

VerificationStatus = Literal["supported", "unsupported", "unavailable"]
_MONEY_QUANTUM = Decimal("0.01")
_EXTERNAL_REDACT_KEY_PARTS = {
    "address",
    "address_line",
    "phone",
    "postal",
    "postal_code",
    "recipient",
    "recipient_name",
    "shipping_address",
    "shipping_summary",
    "confirmation_token",
    "token",
    "email",
    "email_address",
}
_EXTERNAL_DROP_KEYS = {"content_vector", "evidence_snapshots", "provenance"}
_PHONE_RE = re.compile(
    r"(?<!\d)1\d{10}(?!\d)|(?<!\w)\+\d[\d\s().-]{7,}\d(?!\w)|"
    r"(?<!\w)\(\d{2,4}\)[ -]?\d{3,4}[ -]?\d{3,4}(?!\w)|"
    r"(?<!\w)\d{3}[ -]\d{3}[ -]\d{4}(?!\w)"
)
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])",
    flags=re.IGNORECASE,
)


def _external_key_is_sensitive(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _EXTERNAL_REDACT_KEY_PARTS or any(
        normalized.endswith(f"_{part}")
        for part in _EXTERNAL_REDACT_KEY_PARTS
        if part not in {"token", "postal"}
    )


def sanitize_evidence_judge_payload(value: Any, *, key: str = "") -> Any:
    """Remove PII/tokens and audit-only fields at the HTTP send boundary.

    This intentionally avoids a generic numeric redaction so complete SKU and
    variant identifiers remain intact.  Only phone-shaped free text and email
    addresses are removed; structured sensitive fields are replaced wholesale.
    """

    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in _EXTERNAL_DROP_KEYS:
        return None
    if _external_key_is_sensitive(key):
        return "[redacted]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child in value.items():
            child_name = str(child_key)
            if child_name.casefold().replace("-", "_") in _EXTERNAL_DROP_KEYS:
                continue
            result[child_name] = sanitize_evidence_judge_payload(child, key=child_name)
        return result
    if isinstance(value, list):
        return [sanitize_evidence_judge_payload(child, key=key) for child in value]
    if isinstance(value, str):
        return _EMAIL_RE.sub("[redacted]", _PHONE_RE.sub("[redacted]", value))
    return value


class RecommendationSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    variant_id: str | None = None


class RecommendationDraft(BaseModel):
    """The only answer envelope accepted from the generation model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_text: str = Field(min_length=1)
    selections: list[RecommendationSelection] = Field(default_factory=list, max_length=5)


class UnsupportedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    reason: str = ""


class EvidenceJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: VerificationStatus
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def validate_fail_closed(self) -> EvidenceJudgeResult:
        """Reject a semantically contradictory judge response.

        The online gate must never treat a supported verdict with explicit
        unsupported claims as a pass.  When claims are present, each one must
        carry the original fragment and a reason so malformed judge JSON is
        unavailable rather than silently accepted.
        """

        if self.verdict == "supported" and self.unsupported_claims:
            raise ValueError("supported verdict cannot contain unsupported_claims")
        if any(
            not claim.text.strip() or not claim.reason.strip()
            for claim in self.unsupported_claims
        ):
            raise ValueError("unsupported_claims entries require non-empty text and reason")
        return self


def normalize_evidence_judge_result(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize the one tolerated legacy key before strict validation.

    Some OpenAI-compatible models still emit ``claim`` despite the current
    output contract using ``text``.  Accepting that spelling is deliberately
    narrow: conflicting keys, non-string values, and all other extra fields
    remain invalid and are rejected by the strict Pydantic model below.
    """

    normalized = dict(value)
    unsupported_claims = normalized.get("unsupported_claims")
    if not isinstance(unsupported_claims, list):
        return normalized

    normalized_claims: list[Any] = []
    for entry in unsupported_claims:
        if not isinstance(entry, dict):
            normalized_claims.append(entry)
            continue
        has_text = "text" in entry
        has_claim = "claim" in entry
        if has_text and has_claim:
            raise ValueError("unsupported_claims entry cannot contain both text and claim")
        for key in ("text", "claim", "reason"):
            if key in entry and not isinstance(entry[key], str):
                raise ValueError(f"unsupported_claims.{key} must be a string")
        if has_claim:
            entry = {**entry, "text": entry["claim"]}
            del entry["claim"]
        normalized_claims.append(entry)
    normalized["unsupported_claims"] = normalized_claims
    return normalized


class EvidenceJudgeError(RuntimeError):
    """Raised when the online semantic judge cannot return a valid result."""


class EvidenceJudge(Protocol):
    async def judge(
        self,
        *,
        query: str,
        draft: RecommendationDraft,
        semantic_evidence: dict[str, Any] | None = None,
        tool_outputs: list[dict[str, Any]] | None = None,
    ) -> EvidenceJudgeResult: ...


def build_semantic_evidence_bundle(
    *,
    query: str,
    draft: RecommendationDraft,
    top_k_cards: list[dict[str, Any]],
    category_insight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only facts needed to judge the submitted natural language.

    The complete frozen cards remain available to Fact Guard, hydration, order
    preparation, and audit. This projection is intentionally selection-scoped:
    a draft that discusses a product must select it, otherwise the judge has no
    semantic evidence for that product and must fail closed.
    """

    cards_by_id = {str(card.get("item_id", "")): card for card in top_k_cards}
    products: list[dict[str, Any]] = []
    for selection in draft.selections:
        card = cards_by_id.get(selection.item_id)
        if card is None:
            continue
        products.append(_project_selected_product(card, selection.variant_id))

    projected_category = None
    if isinstance(category_insight, dict):
        projected_category = {
            "insights": category_insight.get("insights", {}),
            "source_boundary": category_insight.get("source_boundary", {}),
        }
    return {
        "query": query,
        "selections": [selection.model_dump(mode="json") for selection in draft.selections],
        "products": products,
        "category_insight": projected_category,
    }


def _project_selected_product(card: dict[str, Any], variant_id: str | None) -> dict[str, Any]:
    """Keep speakable product facts, never backend metadata or raw snapshots."""

    fields = (
        "item_id",
        "title",
        "brand",
        "shop",
        "store",
        "platform",
        "category",
        "origin_country",
        "highlights",
        "availability",
        "price_major",
        "currency",
        "landed_price",
    )
    projected = {key: copy.deepcopy(card[key]) for key in fields if key in card}
    variants = card.get("variants")
    if variant_id is not None and isinstance(variants, list):
        selected = next(
            (
                variant
                for variant in variants
                if isinstance(variant, dict)
                and str(variant.get("variant_id")) == variant_id
            ),
            None,
        )
        if selected is not None:
            projected["selected_variant"] = copy.deepcopy(selected)
    return projected


@dataclass(frozen=True)
class FactGuardResult:
    status: VerificationStatus
    errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "supported" and not self.errors


@dataclass(frozen=True)
class SearchQuoteContext:
    """The request context belonging to one product-search tool call.

    This is deliberately kept outside ProductCard and outside any model-facing
    evidence payload.  The orchestrator uses it only to validate quotes from
    the exact search invocation that produced a frozen card.
    """

    ship_to: str | None = None
    target_currency: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> SearchQuoteContext:
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return cls()
        ship_to = value.get("ship_to")
        currency = value.get("target_currency") or value.get("currency")
        return cls(
            ship_to=str(ship_to).strip().upper() if ship_to else None,
            target_currency=str(currency).strip().upper() if currency else None,
        )


def parse_recommendation_draft(raw: Any) -> tuple[RecommendationDraft | None, str | None]:
    """Parse strict JSON while tolerating a single fenced JSON response."""

    if isinstance(raw, RecommendationDraft):
        return raw, None
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        return None, "model output is empty"
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None, "model output is not a JSON object"
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"invalid recommendation JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return None, "recommendation draft must be a JSON object"
    try:
        return RecommendationDraft.model_validate(payload), None
    except ValidationError as exc:
        return None, f"invalid recommendation draft: {exc.errors()[0].get('msg', 'schema error')}"


def fact_guard(
    draft: RecommendationDraft,
    *,
    top_k_cards: list[dict[str, Any]],
    expected_ship_to: str | None = None,
    expected_currency: str | None = None,
    card_contexts: dict[str, SearchQuoteContext | dict[str, Any]] | None = None,
) -> FactGuardResult:
    """Apply only deterministic structural, identity, and quote checks."""

    errors: list[str] = []
    if not draft.answer_text.strip():
        errors.append("answer_text is empty")
    if len(draft.selections) > min(5, len(top_k_cards)):
        errors.append("selection count exceeds frozen Top-K")

    cards_by_id = {str(card.get("item_id", "")): card for card in top_k_cards}
    normalized_contexts = {
        str(item_id): SearchQuoteContext.from_mapping(context)
        for item_id, context in (card_contexts or {}).items()
    }
    has_explicit_card_contexts = card_contexts is not None

    def context_for(card: dict[str, Any]) -> SearchQuoteContext | None:
        item_id = str(card.get("item_id", ""))
        if item_id in normalized_contexts:
            return normalized_contexts[item_id]
        if has_explicit_card_contexts:
            errors.append(f"missing search quote context for frozen card: {item_id}")
            return None
        if expected_ship_to or expected_currency:
            return SearchQuoteContext(
                ship_to=expected_ship_to.strip().upper() if expected_ship_to else None,
                target_currency=expected_currency.strip().upper() if expected_currency else None,
            )
        return None

    # Validate every frozen card once.  Selected-card checks below only handle
    # identity and availability, avoiding duplicate quote errors.
    for card in top_k_cards:
        _validate_card_quotes(card, errors, context=context_for(card))

    seen_items: set[str] = set()
    for selection in draft.selections:
        item_id = selection.item_id
        if item_id in seen_items:
            errors.append(f"duplicate selection item_id: {item_id}")
        seen_items.add(item_id)
        card = cards_by_id.get(item_id)
        if card is None:
            errors.append(f"selection is outside frozen Top-K: {item_id}")
            continue
        if str(card.get("availability", "")).casefold() not in _AVAILABLE_STATUSES:
            errors.append(f"selected item is not available: {item_id}")
        if selection.variant_id is not None:
            variants = card.get("variants")
            if not isinstance(variants, list):
                errors.append(f"selected item has no variants: {item_id}")
                continue
            variant = next(
                (
                    candidate
                    for candidate in variants
                    if str(candidate.get("variant_id")) == selection.variant_id
                ),
                None,
            )
            if variant is None:
                errors.append(f"variant does not belong to item: {item_id}/{selection.variant_id}")
            elif str(variant.get("availability", "")).casefold() not in {
                "available",
                "in_stock",
                "可售",
            }:
                errors.append(
                    f"selected variant is not available: {item_id}/{selection.variant_id}"
                )

    return FactGuardResult(
        "supported" if not errors else "unsupported",
        tuple(dict.fromkeys(errors)),
    )


def hydrate_recommended_cards(
    top_k_cards: list[dict[str, Any]],
    selections: list[RecommendationSelection],
) -> list[dict[str, Any]]:
    """Copy selected cards from frozen backend output; never model-hydrate facts."""

    by_id = {str(card.get("item_id", "")): card for card in top_k_cards}
    hydrated: list[dict[str, Any]] = []
    for selection in selections:
        card = by_id.get(selection.item_id)
        if card is None:
            continue
        result = copy.deepcopy(card)
        if selection.variant_id is not None:
            result["selected_variant_id"] = selection.variant_id
        hydrated.append(result)
    return hydrated


class UnavailableEvidenceJudge:
    async def judge(
        self,
        *,
        query: str,
        draft: RecommendationDraft,
        semantic_evidence: dict[str, Any] | None = None,
        tool_outputs: list[dict[str, Any]] | None = None,
    ) -> EvidenceJudgeResult:
        del query, draft, semantic_evidence, tool_outputs
        return EvidenceJudgeResult(
            verdict="unavailable",
            reason="online Evidence Judge is not configured",
        )


class OpenAIEvidenceJudge:
    """One whole-answer JSON judge; it is separate from the offline Final Judge."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_seconds = max(1.0, timeout_seconds)

    async def judge(
        self,
        *,
        query: str,
        draft: RecommendationDraft,
        semantic_evidence: dict[str, Any] | None = None,
        tool_outputs: list[dict[str, Any]] | None = None,
    ) -> EvidenceJudgeResult:
        if semantic_evidence is None:
            # Compatibility for direct callers from the previous protocol.
            semantic_evidence = {
                "query": query,
                "answer_text": draft.answer_text,
                "selections": [s.model_dump(mode="json") for s in draft.selections],
                "legacy_tool_outputs": tool_outputs or [],
            }
        system = (
            "你是 Globex 在线 Evidence Judge。一次性阅读整个 query、answer_text、selections "
            "和后端生成的 semantic_evidence，判断自然语言是否被事实整体支持。"
            "后端已经确定性检查了 JSON 结构、候选范围、variant 归属/可售、卡片同源和报价算术；"
            "你只判断自然语言语义：商品特点、价格/库存、SKU、到手价、跨商品比较和品类知识边界。"
            "answer_text 中具体推荐、描述或比较的每件商品都必须有对应 selection 和 evidence。"
            "不要逐 claim 调用，不要生成 evidence ID、rubric 或字段目录。"
            "只输出一个严格 JSON object，顶层只能有 verdict、unsupported_claims、reason；"
            "verdict 必须是 supported、unsupported 或 unavailable；"
            "unsupported_claims 必须是数组，数组每项只能有 text 和 reason 两个键，"
            "其中 text 是 answer_text 的原文片段，reason 是简短原因；"
            "必须使用精确键名 text，不要输出 claim、evidence_id 或其他字段。"
        )
        user = json.dumps(
            sanitize_evidence_judge_payload(
                {
                    "query": query,
                    "answer_text": draft.answer_text,
                    "selections": [
                        selection.model_dump(mode="json") for selection in draft.selections
                    ],
                    "semantic_evidence": semantic_evidence,
                },
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self._model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (
            httpx.HTTPError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise EvidenceJudgeError(f"Evidence Judge transport/JSON failure: {exc}") from exc
        parsed, error = parse_json_object(content)
        if error:
            raise EvidenceJudgeError(error)
        try:
            return EvidenceJudgeResult.model_validate(
                normalize_evidence_judge_result(parsed),
            )
        except (ValidationError, ValueError) as exc:
            raise EvidenceJudgeError(f"invalid Evidence Judge result: {exc}") from exc


class EvidenceVerificationService:
    """Mandatory orchestration boundary shared by recommendation and order text."""

    def __init__(
        self,
        judge: EvidenceJudge | None,
        *,
        judge_retries: int = 2,
        judge_timeout_seconds: float = 30.0,
        judge_total_timeout_seconds: float = 70.0,
    ) -> None:
        self._judge = judge or UnavailableEvidenceJudge()
        self._judge_retries = max(0, judge_retries)
        self._judge_timeout_seconds = max(1.0, judge_timeout_seconds)
        self._judge_total_timeout_seconds = max(
            self._judge_timeout_seconds,
            judge_total_timeout_seconds,
        )

    async def verify(
        self,
        *,
        query: str,
        draft: RecommendationDraft,
        top_k_cards: list[dict[str, Any]],
        tool_outputs: list[dict[str, Any]],
        expected_ship_to: str | None = None,
        expected_currency: str | None = None,
        card_contexts: dict[str, SearchQuoteContext | dict[str, Any]] | None = None,
        category_insight: dict[str, Any] | None = None,
    ) -> tuple[FactGuardResult, EvidenceJudgeResult]:
        guard = fact_guard(
            draft,
            top_k_cards=top_k_cards,
            expected_ship_to=expected_ship_to,
            expected_currency=expected_currency,
            card_contexts=card_contexts,
        )
        if not guard.passed:
            return guard, EvidenceJudgeResult(verdict="unsupported", reason="; ".join(guard.errors))
        last_error: Exception | None = None
        deadline = asyncio.get_running_loop().time() + self._judge_total_timeout_seconds
        semantic_evidence = build_semantic_evidence_bundle(
            query=query,
            draft=draft,
            top_k_cards=top_k_cards,
            category_insight=category_insight,
        )
        for _ in range(self._judge_retries + 1):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                last_error = TimeoutError("Evidence Judge total budget exceeded")
                break
            try:
                judge_result = await asyncio.wait_for(
                    self._judge.judge(
                        query=query,
                        draft=draft,
                        semantic_evidence=semantic_evidence,
                    ),
                    timeout=min(self._judge_timeout_seconds, remaining),
                )
                # Re-validate even when an adapter returns a pre-built model;
                # this keeps fail-closed semantics for custom integrations.
                if isinstance(judge_result, EvidenceJudgeResult):
                    judge_result = EvidenceJudgeResult.model_validate(
                        judge_result.model_dump(mode="python")
                    )
                else:
                    judge_result = EvidenceJudgeResult.model_validate(judge_result)
                return guard, judge_result
            except Exception as exc:  # noqa: BLE001 - retry only this external boundary
                last_error = exc
        return guard, EvidenceJudgeResult(
            verdict="unavailable",
            reason=(
                "Evidence Judge unavailable after retries: "
                f"{type(last_error).__name__}: {last_error}" if last_error else
                "Evidence Judge unavailable after retries: unknown error"
            ),
        )


def parse_json_object(content: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(content, dict):
        return content, None
    if not isinstance(content, str):
        return None, "Evidence Judge returned non-text JSON content"
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Evidence Judge returned invalid JSON: {exc.msg}"
    if not isinstance(value, dict):
        return None, "Evidence Judge result must be a JSON object"
    return value, None


def _validate_card_quotes(
    card: dict[str, Any],
    errors: list[str],
    *,
    context: SearchQuoteContext | None,
) -> None:
    item_id = str(card.get("item_id", ""))
    variants = card.get("variants") or []
    if not isinstance(variants, list):
        errors.append(f"variants is not a list: {item_id}")
        return
    has_context = context is not None
    has_ship_to = bool(context and context.ship_to)
    quote_context = context or SearchQuoteContext()

    if variants and card.get("landed_price") is not None:
        errors.append(f"variant-bearing card has top-level landed_price: {card.get('item_id')}")

    top_quote = card.get("landed_price")
    if top_quote is not None:
        if has_context and not has_ship_to:
            errors.append(f"quote present without ship_to context: {item_id}")
        elif not isinstance(top_quote, dict):
            errors.append(f"landed_price is not an object: {item_id}")
        elif not variants:
            _validate_quote(top_quote, errors, context=quote_context, require_quantity_one=True)

    if (
        not variants
        and has_ship_to
        and card.get("price_major") is not None
        and str(card.get("availability", "")).casefold() in _AVAILABLE_STATUSES
        and not isinstance(top_quote, dict)
    ):
        errors.append(f"priced available item is missing top-level quote: {item_id}")

    for variant in variants:
        if not isinstance(variant, dict):
            errors.append(f"variant is not an object: {card.get('item_id')}")
            continue
        quote = variant.get("landed_price")
        if quote is not None and has_context and not has_ship_to:
            errors.append(f"quote present without ship_to context: {item_id}")
        if quote is not None and not isinstance(quote, dict):
            errors.append(f"variant landed_price is not an object: {item_id}")
        if isinstance(quote, dict):
            _validate_quote(
                quote,
                errors,
                context=quote_context,
                require_quantity_one=True,
            )
        if (
            has_ship_to
            and variant.get("price_major") is not None
            and str(variant.get("availability", "")).casefold() in _AVAILABLE_STATUSES
            and not isinstance(quote, dict)
        ):
            errors.append(
                f"priced available variant is missing quote: "
                f"{item_id}/{variant.get('variant_id')}"
            )


def _validate_quote(
    quote: dict[str, Any],
    errors: list[str],
    *,
    context: SearchQuoteContext,
    require_quantity_one: bool,
) -> None:
    if require_quantity_one and quote.get("quantity") != 1:
        errors.append("search quote quantity must be 1")
    if context.ship_to and str(quote.get("ship_to", "")).upper() != context.ship_to.upper():
        errors.append("quote destination does not match request")
    if context.target_currency and (
        str(quote.get("currency", "")).upper() != context.target_currency.upper()
    ):
        errors.append("quote currency does not match request")
    if "unavailable_reason" in quote:
        return
    fields = ("subtotal_major", "freight_major", "tariff_major", "landed_total_major")
    if not all(field in quote for field in fields):
        errors.append("quote is missing landed-price arithmetic fields")
        return
    try:
        subtotal = _money(quote["subtotal_major"])
        freight = _money(quote["freight_major"])
        tariff = _money(quote["tariff_major"])
        landed = _money(quote["landed_total_major"])
    except (InvalidOperation, TypeError, ValueError):
        errors.append("quote arithmetic contains a non-numeric amount")
        return
    if (subtotal + freight + tariff).quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP) != landed:
        errors.append("subtotal + freight + tariff does not equal landed total")


def _money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


_AVAILABLE_STATUSES = {"available", "in_stock", "可售"}
