"""H6 reproducible acceptance harness and frozen behavior checks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields
from types import SimpleNamespace

from agentscope.message import (
    AssistantMsg,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultState,
    UserMsg,
)
from agentscope.state import AgentState
from agentscope.tool import ToolResponse

from app.application.prompts.loader import load_prompts
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_middleware import (
    CONTEXT_NAMESPACE,
    ContextLifecycleMiddleware,
)
from scripts.eval.run_h6_acceptance import (
    _overall_passed,
    _recommendation_display_mentions,
    run_fault_contracts,
    run_real_empty_result_probes,
)


def test_h6_fault_contract_matrix_passes() -> None:
    report = asyncio.run(run_fault_contracts())
    assert report["passed"] is True
    assert {item["classification"] for item in report["observations"]} == {
        "true_empty",
        "constraint_filtered",
        "data_consistency_failure",
        "transient_dependency_failure",
        "non_transient_dependency_failure",
        "degraded_success",
        "timeout",
        "circuit_open",
    }
    by_name = {item["name"]: item for item in report["observations"]}
    assert (
        by_name["hybrid-empty-is-not-system-failure"]["result"][
            "keyword_fallback_calls"
        ]
        == 0
    )
    assert (
        by_name["reranker-error-keeps-vector-order"]["result"][
            "additional_recall_calls"
        ]
        == 0
    )
    assert by_name["tool-timeout-is-explicit"]["result"]["automatic_retry_calls"] == 0


def test_h6_real_empty_probe_is_explicitly_online_only() -> None:
    assert asyncio.iscoroutinefunction(run_real_empty_result_probes)


def test_h6_cannot_pass_without_online_evidence() -> None:
    assert (
        _overall_passed(
            online_requested=False,
            fault_contracts={"passed": True},
            real_empty_results=None,
            online=None,
        )
        is False
    )


def test_h6_does_not_expand_frozen_product_search_spec() -> None:
    assert [field.name for field in fields(ProductSearchSpec)] == [
        "normalized_query",
        "category",
        "ship_to",
        "locale",
        "top_k",
        "target_currency",
        "price_max_major",
    ]


def test_h6_prompt_covers_multi_turn_route_and_reference_boundaries() -> None:
    prompt = load_prompts()["main_agent"]["system_prompt"]
    required = (
        "普通聊天",
        "纯品类选购知识",
        "宽泛商品需求",
        "未指定平台的简单检索默认由 Main 直接查 globex_reference",
        "更多推荐",
        "无故重复",
        "刚才第 N 个",
        "类别切换",
        "最终最多推荐 5 件商品",
        "预算/收货国等约束只对提出它的那轮诉求生效",
    )
    for fragment in required:
        assert fragment in prompt


def test_h6_display_cap_does_not_count_generic_excluded_product_types() -> None:
    events = [
        {
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "hits": [
                    {
                        "product_id": "P1",
                        "title": "AeroHush 主动降噪蓝牙耳机 Pro",
                        "brand": "AeroHush",
                    },
                    {
                        "product_id": "P2",
                        "title": "AeroHush Lite 半入耳蓝牙耳机",
                        "brand": "AeroHush",
                    },
                    {
                        "product_id": "P3",
                        "title": "超长户外续航低音炮无线蓝牙音箱 RGB",
                        "brand": "NoiseBrand",
                    },
                ],
            },
        },
    ]
    final_text = """| 商品 | 价格 |
|---|---|
| AeroHush 主动降噪蓝牙耳机 Pro | 1554.9 |

另有一款 NoiseBrand 蓝牙音箱，属于检索噪声，不展开。
"""

    mentions = _recommendation_display_mentions(final_text, events)

    assert [item["product_id"] for item in mentions] == ["P1"]


def test_h6_l4_reacquires_namespace_after_pre_model_replacement(tmp_path) -> None:
    """Regression: on_model_call replaces the namespace before tool acting."""

    async def run() -> dict:
        middleware = ContextLifecycleMiddleware(
            artifact_root=tmp_path / "artifacts",
            model_context_tokens=8_000,
            tool_result_limit=16_000,
        )
        agent = SimpleNamespace(
            state=AgentState(session_id="h6-l4-replacement"),
            name="commerce_concierge",
        )
        call = ToolCallBlock(
            id="h6-product-call",
            name="product_search_tool",
            input=json.dumps(
                {
                    "normalized_query": "旅行背包",
                    "platform": "taobao",
                    "top_k": 5,
                },
            ),
            state=ToolCallState.ALLOWED,
        )

        async def reply_handler(**_kwargs):
            # This mirrors advance_context_state assigning a new mapping.
            current = agent.state.middle_context[CONTEXT_NAMESPACE]
            agent.state.middle_context[CONTEXT_NAMESPACE] = dict(current)

            async def acting_handler(**_acting_kwargs):
                yield ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=json.dumps(
                                {
                                    "hits": [
                                        {
                                            "product_id": "taobao:cn:H6",
                                            "title": "H6 旅行背包",
                                        },
                                    ],
                                    "recall_strategy": "embedding_rerank",
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    ],
                    state=ToolResultState.SUCCESS,
                    metadata={},
                )

            async for _ in middleware.on_acting(
                agent,
                {"tool_call": call},
                acting_handler,
            ):
                pass
            yield AssistantMsg("commerce_concierge", "推荐 H6 旅行背包")

        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id="h6-l4-replacement",
                buyer_id="h6-buyer",
                locale="zh-CN",
                currency="CNY",
            ),
        )
        try:
            async for _ in middleware.on_reply(
                agent,
                {"inputs": UserMsg("h6-buyer", "找旅行背包")},
                reply_handler,
            ):
                pass
        finally:
            ShoppingContext.reset(token)
        return agent.state.middle_context[CONTEXT_NAMESPACE]

    namespace = asyncio.run(run())
    assert "current_turn_events" not in namespace
    assert namespace["session_context"]["last_search"] == {
        "tool_call_id": "h6-product-call",
        "args": {
            "normalized_query": "旅行背包",
            "platform": "taobao",
            "top_k": 5,
        },
        "status": "succeeded",
        "returned_count": 1,
        "product_refs": ["taobao:cn:H6"],
    }
