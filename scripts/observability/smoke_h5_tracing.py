"""Run a local, network-free H5 trace through the real HTTP client adapters."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
from opentelemetry.sdk.trace.export import SpanExportResult

from app.infrastructure.embedding.openai_embedding_client import OpenAIEmbeddingClient
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.rerank.http_reranker import HttpReranker
from app.infrastructure.settings import load_settings
from app.infrastructure.tracing import (
    set_span_attributes,
    setup_tracing,
    shutdown_tracing,
    text_digest,
    trace_intent,
    trace_span,
)
from app.infrastructure.vector.opensearch_product_index import OpenSearchProductIndex
from scripts.observability.agentscope_trace_harness import run_agentscope_scenarios

_SAFE_ATTRIBUTE_NAMES = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.agent.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
        "gen_ai.conversation.id.hash",
        "agentscope.is_external_execution",
    },
)


class CollectingExporter:
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


def _response(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0] * 1024}]},
        )
    if request.url.path.endswith("/_search"):
        return httpx.Response(
            200,
            json={
                "hits": {
                    "hits": [
                        {
                            "_id": "reference_seed:p-local",
                            "_score": 1.0,
                            "_source": {"product_id": "reference_seed:p-local"},
                        },
                    ],
                },
            },
        )
    if request.url.path.endswith("/rerank"):
        return httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.8}]},
        )
    raise AssertionError(f"unexpected local request path: {request.url.path}")


async def run(output: Path) -> dict[str, Any]:
    settings = replace(
        load_settings(),
        embedding_base_url="http://local.test/v1",
        embedding_api_key="local-only",
        embedding_model="BAAI/bge-m3",
        reranker_base_url="http://local.test",
        reranker_model="BAAI/bge-reranker-v2-m3",
        langfuse_enabled=False,
        otlp_endpoint="",
        langfuse_capture_input=False,
        langfuse_capture_output=False,
    )
    exporter = CollectingExporter()
    setup_tracing(settings, exporter=exporter)

    original_async_client = httpx.AsyncClient
    transport = httpx.MockTransport(_response)

    def local_client(*args, **kwargs):
        kwargs["transport"] = transport
        return original_async_client(*args, **kwargs)

    query = "neutral travel bag"
    bus = TradeEventBus()
    with patch("httpx.AsyncClient", side_effect=local_client):
        embedder = OpenAIEmbeddingClient(settings)
        index = OpenSearchProductIndex(
            "http://local.test",
            "crossshop-products-reference_seed-v1",
        )
        reranker = HttpReranker(settings)
        with trace_intent(
            session_id="local-session",
            buyer_id="local-buyer",
            locale="zh-CN",
            currency="CNY",
        ) as root:
            set_span_attributes(
                root,
                {"crossshop.acceptance.scenario": "retrieval_infrastructure"},
            )
            vector = await embedder.embed(query)
            hits = await index.search(query=query, embedding=vector, top_n=8)
            with trace_span(
                "crossshop.product.constraints",
                {"crossshop.constraints.candidate_count": len(hits)},
            ) as constraint_span:
                set_span_attributes(
                    constraint_span,
                    {
                        "crossshop.constraints.eligible_count": len(hits),
                        "crossshop.constraints.filtered_count": 0,
                    },
                )
            scores = await reranker.rerank(query, ["synthetic local document"])
            bus.publish(
                "local-session",
                "tool.result",
                {
                    "tool": "product_search_tool",
                    "platform": "reference_seed",
                    "hit_count": len(hits),
                    "recall_strategy": "embedding_rerank",
                    "hits": [{"product_id": "must-not-enter-trace"}],
                },
            )
        await index.close()

    agentscope_result = await run_agentscope_scenarios()
    await shutdown_tracing()
    spans = exporter.spans
    span_by_id = {span.context.span_id: span for span in spans}
    root_by_trace = {
        span.context.trace_id: span
        for span in spans
        if span.name == "crossshop.shopping_intent"
    }
    report_spans = [
        {
            "name": span.name,
            "span_id": f"{span.context.span_id:016x}",
            "parent_span_id": (f"{span.parent.span_id:016x}" if span.parent else None),
            "parent": (
                span_by_id[span.parent.span_id].name
                if span.parent and span.parent.span_id in span_by_id
                else None
            ),
            "scenario": root_by_trace[span.context.trace_id].attributes[
                "crossshop.acceptance.scenario"
            ],
            "start_offset_ms": round(
                (span.start_time - root_by_trace[span.context.trace_id].start_time)
                / 1_000_000,
                3,
            ),
            "duration_ms": round((span.end_time - span.start_time) / 1_000_000, 3),
            "status": {
                "code": span.status.status_code.name,
                "description": span.status.description,
            },
            "attributes": {
                key: value
                for key, value in span.attributes.items()
                if key in _SAFE_ATTRIBUTE_NAMES or key.startswith("crossshop.")
            },
            "events": [
                {"name": event.name, "attributes": dict(event.attributes)}
                for event in span.events
            ],
        }
        for span in spans
    ]
    serialized = json.dumps(report_spans, ensure_ascii=False)
    forbidden = [
        "neutral travel bag",
        "synthetic local document",
        "must-not-enter-trace",
        "local-session",
        "local-buyer",
        "local-only",
        "controlled private system prompt",
        "controlled private single-platform request",
        "controlled private cross-platform request",
        "controlled private single query",
        "controlled private crossshop_reference demand",
        "controlled private reference_seed demand",
        "controlled private amazon demand",
        "controlled private crossshop_reference query",
        "controlled private reference_seed query",
        "controlled private amazon query",
        "controlled private product card",
        "controlled private final reply",
        "controlled-root-single-session",
        "controlled-root-single-buyer",
        "controlled-main-single-session",
        "controlled-root-cross-session",
        "controlled-root-cross-buyer",
        "controlled-main-cross-session",
        "controlled-search-session-crossshop_reference",
        "controlled-search-session-reference_seed",
        "controlled-search-session-amazon",
        "controlled private error request",
        "controlled private status with sk-local-never-export and address",
        "controlled-root-error-session",
        "controlled-root-error-buyer",
        "controlled-main-error-session",
        "controlled-private-buyer",
        "controlled-private-product-",
        "local-model-secret",
    ]
    leaked = [value for value in forbidden if value in serialized]
    if leaked:
        raise RuntimeError(f"privacy acceptance failed: {leaked}")
    trace_ids = {span.context.trace_id for span in spans}
    infrastructure_required = {
        "crossshop.shopping_intent",
        "crossshop.embedding.request",
        "crossshop.product.opensearch.hybrid",
        "crossshop.product.constraints",
        "crossshop.product.rerank",
    }
    infrastructure_names = {
        span.name
        for span in spans
        if root_by_trace[span.context.trace_id].attributes["crossshop.acceptance.scenario"]
        == "retrieval_infrastructure"
    }
    missing = sorted(infrastructure_required - infrastructure_names)
    if missing or len(trace_ids) != 4:
        raise RuntimeError(
            f"trace structure failed: missing={missing}, trace_count={len(trace_ids)}",
        )
    topology = _validate_agentscope_topology(spans, root_by_trace)
    report = {
        "external_langfuse_verified": False,
        "transport": (
            "in-memory exporter + httpx MockTransport + controlled AgentScope model"
        ),
        "network_calls": 0,
        "query_digest": text_digest(query),
        "vector_dimension": len(vector),
        "opensearch_hit_count": len(hits),
        "reranker_score_count": len(scores),
        "trace_count": len(trace_ids),
        "span_count": len(spans),
        "privacy_forbidden_value_leaks": leaked,
        "agentscope_scenarios": agentscope_result,
        "agentscope_topology": topology,
        "spans": report_spans,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _validate_agentscope_topology(spans, root_by_trace) -> dict[str, Any]:
    by_scenario: dict[str, list[Any]] = {}
    for span in spans:
        scenario = root_by_trace[span.context.trace_id].attributes[
            "crossshop.acceptance.scenario"
        ]
        by_scenario.setdefault(scenario, []).append(span)

    single = by_scenario["agentscope_single_platform"]
    cross = by_scenario["agentscope_cross_platform"]
    error = by_scenario["agentscope_error_privacy"]
    single_root = next(span for span in single if span.name == "crossshop.shopping_intent")
    single_main = _one(single, "invoke_agent commerce_concierge")
    _require_parent(single_main, single_root)
    for span in single:
        if span.name in {
            "chat controlled-main-single",
            "execute_tool product_search_tool",
        }:
            _require_parent(span, single_main)
    if sum(span.name == "chat controlled-main-single" for span in single) != 2:
        raise RuntimeError("single-platform Main model span count is not 2")
    if sum(span.name == "execute_tool product_search_tool" for span in single) != 1:
        raise RuntimeError("single-platform product tool span count is not 1")

    cross_root = next(span for span in cross if span.name == "crossshop.shopping_intent")
    cross_main = _one(cross, "invoke_agent commerce_concierge")
    _require_parent(cross_main, cross_root)
    dispatches = [span for span in cross if span.name == "execute_tool task_dispatch"]
    search_agents = [
        span for span in cross if span.name == "invoke_agent catalog_search_agent"
    ]
    product_tools = [
        span for span in cross if span.name == "execute_tool product_search_tool"
    ]
    if len(dispatches) != 3 or len(search_agents) != 3 or len(product_tools) != 3:
        raise RuntimeError(
            "cross-platform AgentScope count mismatch: "
            f"dispatch={len(dispatches)}, search_agent={len(search_agents)}, "
            f"product_tool={len(product_tools)}",
        )
    for span in dispatches:
        _require_parent(span, cross_main)
    dispatch_ids = {span.context.span_id for span in dispatches}
    for span in search_agents:
        if span.parent is None or span.parent.span_id not in dispatch_ids:
            raise RuntimeError("SearchAgent is not parented by task_dispatch")
    search_ids = {span.context.span_id for span in search_agents}
    for span in product_tools:
        if span.parent is None or span.parent.span_id not in search_ids:
            raise RuntimeError("product_search_tool is not parented by SearchAgent")
    for platform in _PLATFORM_ORDER:
        model_name = f"chat controlled-search-{platform}"
        models = [span for span in cross if span.name == model_name]
        if len(models) != 2:
            raise RuntimeError(f"{model_name} span count is not 2")
        for span in models:
            if span.parent is None or span.parent.span_id not in search_ids:
                raise RuntimeError(f"{model_name} is not parented by SearchAgent")
    latest_start = max(span.start_time for span in dispatches)
    earliest_end = min(span.end_time for span in dispatches)
    overlap_ms = round((earliest_end - latest_start) / 1_000_000, 3)
    if overlap_ms <= 0:
        raise RuntimeError("three task_dispatch spans did not overlap")
    error_required = {
        "crossshop.shopping_intent",
        "invoke_agent commerce_concierge",
        "chat controlled-error",
    }
    error_names = {span.name for span in error}
    if missing_error := sorted(error_required - error_names):
        raise RuntimeError(f"AgentScope error spans missing: {missing_error}")
    for span in error:
        if span.status.status_code.name != "ERROR":
            raise RuntimeError(f"AgentScope error span is not ERROR: {span.name}")
        if span.status.description is not None:
            raise RuntimeError(f"AgentScope status description leaked: {span.name}")
    return {
        "single_platform": {
            "main_agent_spans": 1,
            "main_model_spans": 2,
            "product_tool_spans": 1,
            "parent_links_verified": True,
        },
        "cross_platform": {
            "main_agent_spans": 1,
            "main_model_spans": 2,
            "task_dispatch_spans": len(dispatches),
            "search_agent_spans": len(search_agents),
            "search_model_spans": sum(
                span.name.startswith("chat controlled-search-") for span in cross
            ),
            "product_tool_spans": len(product_tools),
            "dispatch_overlap_ms": overlap_ms,
            "parent_links_verified": True,
        },
        "error_privacy": {
            "error_spans": len(error),
            "status_descriptions_removed": True,
        },
    }


_PLATFORM_ORDER = ("crossshop_reference", "reference_seed", "amazon")


def _one(spans, name: str):
    matches = [span for span in spans if span.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name!r}, got {len(matches)}")
    return matches[0]


def _require_parent(child, parent) -> None:
    if child.parent is None or child.parent.span_id != parent.context.span_id:
        raise RuntimeError(f"{child.name!r} is not parented by {parent.name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/h5-observability/local-trace-report.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(run(args.output))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trace_count": report["trace_count"],
                "span_count": report["span_count"],
                "external_langfuse_verified": report["external_langfuse_verified"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
