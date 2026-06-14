"""Reproducible Phase-D baseline using the configured main model adapter.

The script never calls the model API.  It reports whether the adapter's local
counting path is provider-native or estimated and deliberately does not derive
enforcement thresholds from estimated counts.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from statistics import median
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from globex_agent.application.context import FrozenSegment, TokenCounter, measure_budget
from globex_agent.application.context.assembler import assemble_llm_input_messages
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.category_insight_tool import build_category_insight_tool
from globex_agent.application.tools.order_tools import (
    build_prepare_cancel_tool,
    build_prepare_order_tool,
    build_query_order_tool,
)
from globex_agent.application.tools.product_search_tool import build_product_search_tool
from globex_agent.application.tools.remember_preference_tool import build_remember_preference_tool
from globex_agent.application.tools.task_dispatch_tool import build_task_dispatch_tool
from globex_agent.application.tools.web_search_tool import build_web_search_tool
from globex_agent.application.usecases.catalog_search import CatalogSearchUseCase
from globex_agent.catalog import LocalCatalog
from globex_agent.domain.catalog.product_search_spec import ProductSearchSpec
from globex_agent.infrastructure.checkpoint import InMemoryDispatchResultStore
from globex_agent.infrastructure.eventbus import TradeEventBus
from globex_agent.infrastructure.llm import create_chat_model
from globex_agent.infrastructure.persistence.in_memory_repositories import InMemoryItemRepository
from globex_agent.infrastructure.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]


def _tool_schemas(settings) -> list[dict[str, Any]]:
    bus = TradeEventBus()
    tools = [
        build_product_search_tool(None, bus),
        build_category_insight_tool(None, bus),
        build_prepare_order_tool(None, bus),
        build_query_order_tool(None, bus),
        build_prepare_cancel_tool(None, bus),
        build_task_dispatch_tool(
            object(),
            object(),
            bus,
            result_store=InMemoryDispatchResultStore(),
        ),
        build_remember_preference_tool(None, bus),
    ]
    if settings.tavily_api_key:
        tools.append(build_web_search_tool(settings, bus))
    return [convert_to_openai_tool(tool) for tool in tools]


def _flow_cases() -> list[dict[str, Any]]:
    path = ROOT / "data" / "eval" / "flow_regression_v2.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


async def _representative_product_outputs() -> list[dict[str, Any]]:
    catalog = LocalCatalog.from_jsonl(ROOT / "data" / "demo" / "products.jsonl", strict=True)
    usecase = CatalogSearchUseCase(InMemoryItemRepository(list(catalog.catalog.items)))
    outputs = []
    for query in ("降噪耳机", "通勤背包", "机械键盘"):
        result = await usecase.execute(ProductSearchSpec(normalized_query=query, top_k=5))
        outputs.append({key: value for key, value in result.items() if key != "evidence_snapshots"})
    return outputs


def _distribution(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    p95_index = max(0, int((len(ordered) - 1) * 0.95))
    return {
        "min": ordered[0],
        "p50": int(median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _reply_proxy(case: dict[str, Any]) -> str:
    return str(
        case.get("expected_degradation")
        or json.dumps(case.get("quality_expectation") or {}, ensure_ascii=False)
    )


async def main() -> None:
    settings = load_settings()
    model = create_chat_model(settings)
    prompt = load_prompts()["main_agent"]["system_prompt"]
    schemas = _tool_schemas(settings)
    cases = _flow_cases()
    product_outputs = await _representative_product_outputs()
    counter = TokenCounter(model=model)
    frozen = [
        FrozenSegment(
            segment_id=f"baseline-{index}",
            content=json.dumps(
                {
                    "user": case["query"],
                    "assistant": _reply_proxy(case),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        for index, case in enumerate(cases[:10])
    ]
    long_history = assemble_llm_input_messages(
        messages=[],
        frozen_segments=frozen,
        freeze_cursor=0,
        session_context=None,
    )

    reports = []
    active_counts = []
    l4_counts = []
    tool_counts = []
    for index, case in enumerate(cases):
        query = case["query"]
        l4 = {
            "schema_version": "session-context-v1",
            "revision": index + 1,
            "session": {"locale": case["locale"], "currency": case["currency"]},
            "request": {
                "session_origin_raw_query": cases[0]["query"],
                "current_raw_query": query,
            },
        }
        tool_message = ToolMessage(
            content=json.dumps(
                product_outputs[index % len(product_outputs)],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            tool_call_id=f"baseline-{index}",
        )
        report = measure_budget(
            fixed_prompt=[SystemMessage(content=prompt)],
            tool_schemas=schemas,
            l4_candidate=[SystemMessage(content=json.dumps(l4, ensure_ascii=False))],
            history=long_history if index == len(cases) - 1 else [],
            current=[HumanMessage(content=query)],
            active=[],
            tool_results={tool_message.tool_call_id: [tool_message]},
            model_context_tokens=settings.context_size,
            reply_reserved_tokens=settings.reply_token_budget,
            safety_margin_tokens=settings.context_safety_margin_tokens,
            model=model,
        )
        reports.append(report)
        active_counts.append(report.layers["current"].tokens)
        l4_counts.append(report.layers["l4_candidate"].tokens)
        tool_counts.extend(item.tokens for item in report.tool_results.values())

    reply_proxies = [
        counter.count(_reply_proxy(case), name="reply_proxy").tokens
        for case in cases
    ]
    representative = reports[0]
    result = {
        "model": settings.llm_model,
        "sample_counts": {
            "flow_cases": len(cases),
            "product_search_outputs": len(product_outputs),
            "frozen_turns_in_long_sample": len(frozen),
        },
        "counting": {
            "estimated": any(report.estimated for report in reports),
            "methods": sorted(
                {
                    item.method
                    for report in reports
                    for item in [*report.layers.values(), *report.tool_results.values()]
                    if item.tokens
                }
            ),
            "provider_message_counter_available": representative.layers[
                "fixed_prompt"
            ].method
            == "model.get_num_tokens_from_messages",
        },
        "fixed": {
            "system_prompt": representative.layers["fixed_prompt"].tokens,
            "tool_schemas": representative.layers["tool_schemas"].tokens,
            "tool_schema_count": len(schemas),
        },
        "distributions": {
            "active_query": _distribution(active_counts),
            "l4_candidate": _distribution(l4_counts),
            "product_search_result": _distribution(tool_counts),
            "reply_proxy_not_actual_response": _distribution(reply_proxies),
        },
        "long_sample": {
            "frozen_history": reports[-1].layers["history"].tokens,
            "total_input": reports[-1].total_input_tokens,
        },
        "configured": {
            "context_size": settings.context_size,
            "reply_reserved_tokens": settings.reply_token_budget,
            "safety_margin_tokens": settings.context_safety_margin_tokens,
            "soft_limit_tokens": settings.context_soft_limit_tokens,
            "mode": settings.context_budget_mode,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
