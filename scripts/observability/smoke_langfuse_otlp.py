"""Send one metadata-only trace to the configured LangFuse OTLP endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from opentelemetry.sdk.trace.export import SpanExportResult

from app.infrastructure.settings import load_settings
from app.infrastructure.tracing import (
    _endpoint_label,
    _trace_endpoint,
    _traces_url,
    set_span_attributes,
    setup_tracing,
    shutdown_tracing,
    trace_intent,
    trace_span,
)


class _ResultCapturingExporter:
    """Record transport outcome without printing credentials or payloads."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.result: SpanExportResult | None = None
        self.span_count = 0

    def export(self, spans) -> SpanExportResult:
        self.span_count += len(spans)
        self.result = self._inner.export(spans)
        return self.result

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return bool(self._inner.force_flush(timeout_millis))

    def shutdown(self) -> None:
        self._inner.shutdown()


async def run() -> dict[str, Any]:
    settings = load_settings()
    if not settings.langfuse_enabled:
        raise RuntimeError("LANGFUSE_ENABLED is not enabled")

    endpoint, headers = _trace_endpoint(settings)
    if not endpoint or headers is None:
        raise RuntimeError("LangFuse credentials are incomplete")

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    exporter = _ResultCapturingExporter(
        OTLPSpanExporter(
            endpoint=_traces_url(endpoint),
            headers=headers,
            timeout=settings.langfuse_flush_timeout_seconds,
        ),
    )
    setup_tracing(settings, exporter=exporter)

    trace_id = ""
    try:
        with trace_intent(
            session_id="langfuse-connectivity-smoke",
            buyer_id="langfuse-connectivity-smoke",
            locale="zh-CN",
            currency="CNY",
        ) as root:
            trace_id = f"{root.get_span_context().trace_id:032x}"
            set_span_attributes(
                root,
                {
                    "crossshop.acceptance.scenario": "langfuse_connectivity",
                    "crossshop.acceptance.contains_user_content": False,
                },
            )
            with trace_span(
                "crossshop.langfuse.connectivity",
                {
                    "crossshop.langfuse.protocol": "otlp_http",
                    "crossshop.langfuse.ingestion_version": 4,
                },
            ):
                pass
    finally:
        await shutdown_tracing()

    if exporter.result is not SpanExportResult.SUCCESS:
        raise RuntimeError("LangFuse OTLP export did not succeed")
    return {
        "passed": True,
        "endpoint": _endpoint_label(endpoint),
        "trace_id": trace_id,
        "exported_span_count": exporter.span_count,
        "capture_input": settings.langfuse_capture_input,
        "capture_output": settings.langfuse_capture_output,
    }


def main() -> None:
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
