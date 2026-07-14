from __future__ import annotations

from typing import Any

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.trace import Status, StatusCode

from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.settings import load_settings
from app.infrastructure.tracing import (
    PrivacySafeSpanExporter,
    _trace_endpoint,
    record_business_event,
    set_span_attributes,
    text_digest,
    trace_intent,
    trace_span,
)
from scripts.observability.agentscope_trace_harness import (
    run_agentscope_error_scenario,
    run_agentscope_scenarios,
)


class _CollectingExporter:
    def __init__(self) -> None:
        self.spans: list[Any] = []
        self.flush_count = 0
        self.shutdown_count = 0

    def export(self, spans) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, _timeout_millis: int = 30000) -> bool:
        self.flush_count += 1
        return True

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _BrokenExporter(_CollectingExporter):
    def export(self, spans) -> SpanExportResult:
        del spans
        raise TimeoutError("endpoint unavailable with secret sk-never-export-this")


def _provider(exporter: Any) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def test_default_export_drops_content_hashes_ids_and_redacts_events() -> None:
    collector = _CollectingExporter()
    exporter = PrivacySafeSpanExporter(
        collector,
        capture_input=False,
        capture_output=False,
    )
    provider = _provider(exporter)
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span(
        "invoke_agent MainAgent",
        attributes={
            "gen_ai.conversation.id": "session-private",
            "gen_ai.input.messages": "完整买家 query",
            "gen_ai.output.messages": "完整模型回复",
            "gen_ai.tool.call.arguments": '{"address":"北京市"}',
            "gen_ai.tool.call.result": '{"hits":[{"product_id":"p-1"}]}',
            "gen_ai.request.model": "deepseek-v4-flash",
        },
    ) as span:
        span.add_event(
            "exception",
            {
                "authorization": "Bearer sk-super-secret",
                "phone": "13800138000",
                "exception.type": "RuntimeError",
                "exception.message": "完整 query 与商品卡",
                "exception.stacktrace": "private stack",
            },
        )
    provider.shutdown()

    assert len(collector.spans) == 1
    exported = collector.spans[0]
    attrs = dict(exported.attributes)
    assert attrs["gen_ai.request.model"] == "deepseek-v4-flash"
    assert attrs["gen_ai.conversation.id.hash"] == text_digest("session-private")
    assert "gen_ai.conversation.id" not in attrs
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert "gen_ai.tool.call.arguments" not in attrs
    assert "gen_ai.tool.call.result" not in attrs
    event_attrs = dict(exported.events[0].attributes)
    assert event_attrs["authorization"] == "[redacted]"
    assert event_attrs["phone"] == "[redacted]"
    assert event_attrs["exception.type"] == "RuntimeError"
    assert "exception.message" not in event_attrs
    assert "exception.stacktrace" not in event_attrs


def test_opt_in_capture_exports_only_fingerprint_and_shape() -> None:
    collector = _CollectingExporter()
    provider = _provider(
        PrivacySafeSpanExporter(
            collector,
            capture_input=True,
            capture_output=True,
        ),
    )
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span(
        "execute_tool create_order_tool",
        attributes={
            "gen_ai.tool.call.arguments": (
                '{"address":"private","confirmation_token":"secret-token",'
                '"note":"Bearer sk-abcdefghijk"}'
            ),
            "gen_ai.tool.call.result": "x" * 5000,
        },
    ):
        pass
    provider.shutdown()

    attrs = dict(collector.spans[0].attributes)
    assert "private" not in attrs["gen_ai.tool.call.arguments"]
    assert "secret-token" not in attrs["gen_ai.tool.call.arguments"]
    assert "sk-abcdefghijk" not in attrs["gen_ai.tool.call.arguments"]
    arguments_summary = attrs["gen_ai.tool.call.arguments"]
    result_summary = attrs["gen_ai.tool.call.result"]
    assert '"digest"' in arguments_summary
    assert '"kind": "dict"' in arguments_summary
    assert '"characters": 5000' in result_summary
    assert "x" * 100 not in result_summary


def test_export_failure_is_fail_open() -> None:
    exporter = PrivacySafeSpanExporter(
        _BrokenExporter(),
        capture_input=False,
        capture_output=False,
    )
    provider = _provider(exporter)
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("safe-operation"):
        pass
    provider.shutdown()


def test_error_status_description_is_removed_before_export() -> None:
    collector = _CollectingExporter()
    provider = _provider(
        PrivacySafeSpanExporter(
            collector,
            capture_input=False,
            capture_output=False,
        ),
    )
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("failing-operation") as span:
        span.set_status(
            Status(
                StatusCode.ERROR,
                "private query with sk-status-never-export and address",
            ),
        )
    provider.shutdown()

    exported = collector.spans[0]
    assert exported.status.status_code is StatusCode.ERROR
    assert exported.status.description is None
    assert "sk-status-never-export" not in exported.to_json()


async def test_real_agentscope_error_spans_remove_exception_and_status_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _CollectingExporter()
    provider = _provider(
        PrivacySafeSpanExporter(
            collector,
            capture_input=False,
            capture_output=False,
        ),
    )
    monkeypatch.setattr(otel_trace, "get_tracer", provider.get_tracer)
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: provider)

    assert await run_agentscope_error_scenario() == 1
    provider.shutdown()

    names = {span.name for span in collector.spans}
    assert names == {
        "crossshop.shopping_intent",
        "invoke_agent commerce_concierge",
        "chat controlled-error",
    }
    assert all(span.status.status_code is StatusCode.ERROR for span in collector.spans)
    assert all(span.status.description is None for span in collector.spans)
    serialized = "\n".join(span.to_json() for span in collector.spans)
    assert "sk-local-never-export" not in serialized
    assert "controlled private status" not in serialized


async def test_real_agentscope_single_and_cross_platform_span_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _CollectingExporter()
    provider = _provider(
        PrivacySafeSpanExporter(
            collector,
            capture_input=False,
            capture_output=False,
        ),
    )
    monkeypatch.setattr(otel_trace, "get_tracer", provider.get_tracer)
    monkeypatch.setattr(otel_trace, "get_tracer_provider", lambda: provider)

    result = await run_agentscope_scenarios()
    provider.shutdown()

    assert result["cross_platforms"] == ["crossshop_reference", "reference_seed", "amazon"]
    roots = {
        span.context.trace_id: span
        for span in collector.spans
        if span.name == "crossshop.shopping_intent"
    }
    by_scenario: dict[str, list[Any]] = {}
    for span in collector.spans:
        scenario = roots[span.context.trace_id].attributes["crossshop.acceptance.scenario"]
        by_scenario.setdefault(scenario, []).append(span)

    single = by_scenario["agentscope_single_platform"]
    assert sum(span.name == "invoke_agent commerce_concierge" for span in single) == 1
    assert sum(span.name == "chat controlled-main-single" for span in single) == 2
    assert sum(span.name == "execute_tool product_search_tool" for span in single) == 1

    cross = by_scenario["agentscope_cross_platform"]
    by_id = {span.context.span_id: span for span in cross}
    dispatches = [span for span in cross if span.name == "execute_tool task_dispatch"]
    search_agents = [
        span for span in cross if span.name == "invoke_agent catalog_search_agent"
    ]
    product_tools = [
        span for span in cross if span.name == "execute_tool product_search_tool"
    ]
    assert len(dispatches) == len(search_agents) == len(product_tools) == 3
    assert all(
        span.parent is not None
        and by_id[span.parent.span_id].name == "execute_tool task_dispatch"
        for span in search_agents
    )
    assert all(
        span.parent is not None
        and by_id[span.parent.span_id].name == "invoke_agent catalog_search_agent"
        for span in product_tools
    )
    assert max(span.start_time for span in dispatches) < min(
        span.end_time for span in dispatches
    )
    serialized = "\n".join(span.to_json() for span in collector.spans)
    assert "controlled private single query" not in serialized
    assert "controlled private amazon demand" not in serialized
    assert "controlled private product card" not in serialized


def test_intent_business_events_and_product_stages_share_one_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = _CollectingExporter()
    provider = _provider(
        PrivacySafeSpanExporter(
            collector,
            capture_input=False,
            capture_output=False,
        ),
    )
    monkeypatch.setattr(otel_trace, "get_tracer", provider.get_tracer)
    bus = TradeEventBus()

    with trace_intent(
        session_id="session-private",
        buyer_id="buyer-private",
        locale="zh-CN",
        currency="CNY",
    ) as root:
        record_business_event(
            "agent.dispatch",
            {
                "agent": "search",
                "platform": "reference_seed",
                "demands": "完整私密需求不应进入 trace",
            },
        )
        with trace_span(
            "crossshop.embedding.request",
            {"crossshop.embedding.input_digest": text_digest("中性 query")},
        ):
            pass
        with trace_span("crossshop.product.opensearch.hybrid") as search_span:
            set_span_attributes(search_span, {"crossshop.opensearch.hit_count": 5})
        with trace_span("crossshop.product.constraints"):
            pass
        with trace_span("crossshop.product.rerank"):
            pass
        bus.publish(
            "session-private",
            "final.result",
            {"text": "完整回复不应进入 trace", "displayed_products": [{"rank": 1}]},
        )
        bus.publish(
            "session-private",
            "context.compressed",
            {
                "action": "l3_stage_summary",
                "before_tokens": 100,
                "after_tokens": 70,
                "source_segment_count": 2,
                "summary_hash": "a" * 64,
                "content": "摘要正文不应进入 trace",
            },
        )
        assert root.is_recording()
    provider.shutdown()

    by_name = {span.name: span for span in collector.spans}
    root = by_name["crossshop.shopping_intent"]
    root_trace_id = root.context.trace_id
    assert all(span.context.trace_id == root_trace_id for span in collector.spans)
    for child_name in (
        "crossshop.embedding.request",
        "crossshop.product.opensearch.hybrid",
        "crossshop.product.constraints",
        "crossshop.product.rerank",
    ):
        assert by_name[child_name].parent.span_id == root.context.span_id

    root_attrs = dict(root.attributes)
    assert root_attrs["crossshop.session.hash"] == text_digest("session-private")
    assert root_attrs["crossshop.buyer.hash"] == text_digest("buyer-private")
    events = {event.name: dict(event.attributes) for event in root.events}
    assert events["crossshop.agent.dispatch"]["crossshop.platform"] == "reference_seed"
    assert "demands" not in events["crossshop.agent.dispatch"]
    assert events["crossshop.final.result"]["crossshop.recommended_count"] == 1
    assert events["crossshop.context.compressed"] == {
        "crossshop.action": "l3_stage_summary",
        "crossshop.before_tokens": 100,
        "crossshop.after_tokens": 70,
        "crossshop.source_segment_count": 2,
    }
    serialized = "\n".join(span.to_json() for span in collector.spans)
    assert "完整私密需求" not in serialized
    assert "完整回复" not in serialized
    assert "摘要正文" not in serialized
    assert "a" * 64 not in serialized
    assert "session-private" not in serialized
    assert "buyer-private" not in serialized


def test_langfuse_settings_keep_old_opt_in_and_privacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "local-test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")
    monkeypatch.setenv("LANGFUSE_SAMPLE_RATE", "2")
    monkeypatch.delenv("LANGFUSE_CAPTURE_INPUT", raising=False)
    monkeypatch.delenv("LANGFUSE_CAPTURE_OUTPUT", raising=False)
    monkeypatch.delenv("LANGFUSE_FLUSH_TIMEOUT_SECONDS", raising=False)

    settings = load_settings()

    assert settings.langfuse_enabled is True
    assert settings.langfuse_public_key == "pk-test"
    assert settings.langfuse_secret_key == "sk-test"
    assert settings.langfuse_host == "https://langfuse.example"
    assert settings.langfuse_sample_rate == 1.0
    assert settings.langfuse_capture_input is False
    assert settings.langfuse_capture_output is False
    assert settings.langfuse_flush_timeout_seconds == 20.0

    endpoint, headers = _trace_endpoint(settings)

    assert endpoint == "https://langfuse.example/api/public/otel"
    assert headers is not None
    assert headers["Authorization"].startswith("Basic ")
    assert headers["x-langfuse-ingestion-version"] == "4"
