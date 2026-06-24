"""Privacy-safe OpenTelemetry tracing for the AgentScope runtime.

AgentScope's :class:`TracingMiddleware` remains the single source of agent,
model and tool spans. This module configures the exporter, removes sensitive
payloads before export, and adds the small business spans/events that
AgentScope cannot observe itself.

Tracing is optional and fail-open. With neither LangFuse nor a generic OTLP
endpoint configured, these helpers use OpenTelemetry's no-op implementation
and perform no network I/O.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit

from agentscope.middleware import ReplyBudgetControlMiddleware, TracingMiddleware

from app.infrastructure.settings import Settings

logger = logging.getLogger(__name__)

_initialized = False
_provider: Any | None = None
_flush_timeout_seconds = 2.0

_BUDGET_HINT = (
    "<system-reminder>本次会话已达到 Token 预算上限。请立即停止调用工具，"
    "基于当前已获得的信息给买家一个明确的收尾回复（如实说明信息可能不完整）。</system-reminder>"
)

_INPUT_CONTENT_ATTRIBUTES = frozenset(
    {"gen_ai.input.messages", "gen_ai.tool.call.arguments"},
)
_OUTPUT_CONTENT_ATTRIBUTES = frozenset(
    {"gen_ai.output.messages", "gen_ai.tool.call.result"},
)
_IDENTIFIER_ATTRIBUTES = frozenset(
    {
        "gen_ai.conversation.id",
        "session_id",
        "shopping_session_id",
        "buyer_id",
        "user_id",
    },
)
_REDACT_KEYS = frozenset(
    {
        "address",
        "address_line",
        "api_key",
        "authorization",
        "confirmation_token",
        "password",
        "phone",
        "postal_code",
        "recipient",
        "recipient_name",
        "secret",
        "secret_key",
        "shipping_address",
        "shipping_summary",
        "token",
    },
)
_SECRET_RE = re.compile(
    r"(?i)(bearer\s+[^\s,;]+|sk-[a-z0-9_-]{8,}|pk-lf-[a-z0-9_-]{8,}|sk-lf-[a-z0-9_-]{8,})",
)
_PHONE_RE = re.compile(r"(?<!\d)1\d{10}(?!\d)")

_BUSINESS_EVENT_FIELDS = frozenset(
    {
        "agent",
        "budget_tier",
        "context_messages",
        "elapsed_ms",
        "from",
        "hit_count",
        "platform",
        "recall_strategy",
        "rerank_applied",
        "saved",
        "site_locale",
        "similarity",
        "summary_length",
        "to",
        "tool",
    },
)
_BUSINESS_EVENT_TYPES = frozenset(
    {
        "agent.dispatch",
        "cache.hit",
        "context.compressed",
        "error",
        "final.result",
        "model.fallback",
        "task.queued",
        "task.started",
        "tool.invoke",
        "tool.result",
    },
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def text_digest(value: str) -> str:
    """Return a stable one-way identifier without exporting the source text."""

    return _digest(value)


def _sanitize_value(value: Any, *, key: str = "") -> Any:
    if key.casefold() in _REDACT_KEYS:
        return "[redacted]"
    if isinstance(value, Mapping):
        return {
            str(child_key): _sanitize_value(child, key=str(child_key))
            for child_key, child in value.items()
            if str(child_key) != "content_vector"
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(child, key=key) for child in value]
    if isinstance(value, str):
        cleaned = _SECRET_RE.sub("[redacted]", value)
        cleaned = _PHONE_RE.sub("[redacted]", cleaned)
        # Captured content is explicitly opt-in, but remains bounded so a
        # complete long conversation or product result cannot be exported.
        return cleaned[:2000]
    return value


def _sanitize_content(value: Any) -> str:
    serialized = value if isinstance(value, str) else str(value)
    summary: dict[str, Any] = {
        "digest": _digest(serialized),
        "characters": len(serialized),
        "kind": "text",
    }
    try:
        parsed = json.loads(serialized)
    except (TypeError, ValueError):
        pass
    else:
        summary["kind"] = type(parsed).__name__
        if isinstance(parsed, Mapping):
            summary["top_level_keys"] = sorted(str(key) for key in parsed)[:20]
    # Even explicit capture mode exports only a fingerprint and shape. It never
    # exports a complete query, response, product card, address or credential.
    return json.dumps(summary, ensure_ascii=False)


def _sanitize_attributes(
    attributes: Mapping[str, Any] | None,
    *,
    capture_input: bool,
    capture_output: bool,
) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for raw_key, value in (attributes or {}).items():
        key = str(raw_key)
        if key in _INPUT_CONTENT_ATTRIBUTES:
            if capture_input:
                safe[key] = _sanitize_content(value)
            continue
        if key in _OUTPUT_CONTENT_ATTRIBUTES:
            if capture_output:
                safe[key] = _sanitize_content(value)
            continue
        if key in _IDENTIFIER_ATTRIBUTES:
            safe[f"{key}.hash"] = _digest(str(value))
            continue
        safe[key] = _sanitize_value(value, key=key)
    return safe


def _clone_span(span: Any, attributes: Mapping[str, Any], events: Sequence[Any]) -> Any:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import Status

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=events,
        links=span.links,
        kind=span.kind,
        instrumentation_scope=getattr(span, "instrumentation_scope", None),
        # AgentScope sets ERROR descriptions from ``str(exception)``. Keep the
        # status code for diagnostics but never export that free-form text.
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
    )


def _sanitize_events(
    events: Iterable[Any],
    *,
    capture_input: bool,
    capture_output: bool,
) -> list[Any]:
    from opentelemetry.sdk.trace import Event

    safe_events: list[Any] = []
    for event in events:
        attributes = _sanitize_attributes(
            event.attributes,
            capture_input=capture_input,
            capture_output=capture_output,
        )
        if event.name == "exception":
            # Exception messages/stacktraces can embed an upstream response,
            # query, address or complete product card. Keep type/status only,
            # even when generic model/tool content capture is explicitly on.
            attributes.pop("exception.message", None)
            attributes.pop("exception.stacktrace", None)
        safe_events.append(Event(event.name, attributes, event.timestamp))
    return safe_events


class PrivacySafeSpanExporter:
    """Exporter decorator that removes sensitive attributes before transport."""

    def __init__(
        self, inner: Any, *, capture_input: bool, capture_output: bool
    ) -> None:
        self._inner = inner
        self._capture_input = capture_input
        self._capture_output = capture_output

    def export(self, spans: Sequence[Any]) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            safe_spans = [
                _clone_span(
                    span,
                    _sanitize_attributes(
                        span.attributes,
                        capture_input=self._capture_input,
                        capture_output=self._capture_output,
                    ),
                    _sanitize_events(
                        span.events,
                        capture_input=self._capture_input,
                        capture_output=self._capture_output,
                    ),
                )
                for span in spans
            ]
            return self._inner.export(safe_spans)
        except Exception as err:  # noqa: BLE001 - telemetry must be fail-open
            logger.warning("Trace export skipped: %s", type(err).__name__)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        try:
            self._inner.shutdown()
        except Exception as err:  # noqa: BLE001
            logger.warning("Trace exporter shutdown skipped: %s", type(err).__name__)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        try:
            return bool(self._inner.force_flush(timeout_millis))
        except Exception as err:  # noqa: BLE001
            logger.warning("Trace exporter flush skipped: %s", type(err).__name__)
            return False


def _trace_endpoint(settings: Settings) -> tuple[str, dict[str, str] | None]:
    if settings.langfuse_enabled:
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning(
                "LangFuse enabled but credentials are incomplete; tracing remains disabled",
            )
            return "", None
        endpoint = settings.otlp_endpoint.strip() or (
            f"{settings.langfuse_host.rstrip('/')}/api/public/otel"
        )
        token = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode(),
        ).decode()
        return endpoint, {"Authorization": f"Basic {token}"}
    return settings.otlp_endpoint.strip(), None


def _traces_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"


def _endpoint_label(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    host = parsed.hostname or "configured-endpoint"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme or 'http'}://{host}{port}"


def setup_tracing(settings: Settings, *, exporter: Any | None = None) -> None:
    """Initialize one process-wide OTel provider when tracing is configured.

    ``exporter`` is a deterministic local-test seam. Production uses OTLP/HTTP.
    Initialization errors are logged without credentials and never stop startup.
    """

    global _initialized, _provider, _flush_timeout_seconds
    if _initialized:
        return
    endpoint, headers = _trace_endpoint(settings)
    if exporter is None and not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

        transport = exporter or OTLPSpanExporter(
            endpoint=_traces_url(endpoint),
            headers=headers,
            timeout=settings.langfuse_flush_timeout_seconds,
        )
        safe_exporter = PrivacySafeSpanExporter(
            transport,
            capture_input=settings.langfuse_capture_input,
            capture_output=settings.langfuse_capture_output,
        )
        provider = TracerProvider(
            resource=Resource.create({"service.name": "globex-agent"}),
            sampler=ParentBased(TraceIdRatioBased(settings.langfuse_sample_rate)),
        )
        processor = (
            SimpleSpanProcessor(safe_exporter)
            if exporter is not None
            else BatchSpanProcessor(safe_exporter)
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _provider = provider
        _flush_timeout_seconds = settings.langfuse_flush_timeout_seconds
        _initialized = True
        logger.info(
            "OTel tracing enabled: %s",
            "local exporter" if exporter is not None else _endpoint_label(endpoint),
        )
    except Exception as err:  # noqa: BLE001 - optional observability is fail-open
        logger.warning("OTel tracing unavailable: %s", type(err).__name__)


async def shutdown_tracing() -> None:
    """Bounded flush and shutdown for workers and short acceptance runs."""

    global _provider
    provider = _provider
    if provider is None:
        return
    timeout_millis = max(50, int(_flush_timeout_seconds * 1000))

    def _close() -> None:
        try:
            provider.force_flush(timeout_millis=timeout_millis)
        except Exception as err:  # noqa: BLE001
            logger.warning("Trace provider flush skipped: %s", type(err).__name__)
        try:
            provider.shutdown()
        except Exception as err:  # noqa: BLE001
            logger.warning("Trace provider shutdown skipped: %s", type(err).__name__)

    await asyncio.to_thread(_close)
    _provider = None


def build_agent_middlewares(settings: Settings) -> list:
    """Return the shared AgentScope middleware list for every agent."""

    middlewares: list = [TracingMiddleware()]
    if settings.reply_token_budget > 0:
        middlewares.append(
            ReplyBudgetControlMiddleware(
                token_budget=settings.reply_token_budget,
                hint_message=_BUDGET_HINT,
            ),
        )
    return middlewares


@contextmanager
def trace_span(name: str, attributes: Mapping[str, Any] | None = None):
    """Create a small span with metadata-only attributes."""

    from opentelemetry import trace

    tracer = trace.get_tracer("globex-agent")
    with tracer.start_as_current_span(name, attributes=dict(attributes or {})) as span:
        yield span


@contextmanager
def trace_intent(*, session_id: str, buyer_id: str, locale: str, currency: str):
    """Create the root span for exactly one shopping intent."""

    with trace_span(
        "globex.shopping_intent",
        {
            "globex.request.id": f"req-{uuid.uuid4().hex}",
            "globex.session.hash": _digest(session_id),
            "globex.buyer.hash": _digest(buyer_id),
            "globex.locale": locale,
            "globex.currency": currency,
        },
    ) as span:
        yield span


def set_span_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    if span is None or not getattr(span, "is_recording", lambda: False)():
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(str(key), value)


def record_business_event(event_type: str, payload: Any) -> None:
    """Attach a safe summary of an EventBus event to the current span."""

    try:
        _record_business_event(event_type, payload)
    except Exception as exc:  # noqa: BLE001 - tracing must never break business flow
        logger.debug("business trace event skipped: %s", type(exc).__name__)


def _record_business_event(event_type: str, payload: Any) -> None:
    if event_type not in _BUSINESS_EVENT_TYPES:
        return
    from opentelemetry import trace

    span = trace.get_current_span()
    if not span.is_recording():
        return
    source = payload if isinstance(payload, Mapping) else {}
    attributes = {
        f"globex.{key}": source[key]
        for key in _BUSINESS_EVENT_FIELDS
        if key in source and isinstance(source[key], (str, bool, int, float))
    }
    if event_type == "error" or "error" in source:
        attributes["globex.error"] = True
    if event_type == "final.result":
        cards = source.get("recommended_cards")
        if isinstance(cards, list):
            attributes["globex.recommended_count"] = len(cards)
    span.add_event(f"globex.{event_type}", attributes=attributes)
