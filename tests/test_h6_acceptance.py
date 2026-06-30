"""H6 reproducible acceptance harness and frozen behavior checks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields
from types import SimpleNamespace

import pytest
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
from app.application.agents.orchestrator import (
    MainAgentOrchestrator,
    SelectedProductRef,
    SubmitIntentInput,
    _resolve_displayed_products,
)
from app.infrastructure.cache.semantic_cache import SemanticHit
from app.infrastructure.eventbus import TradeEventBus
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.context_middleware import (
    CONTEXT_NAMESPACE,
    ContextLifecycleMiddleware,
)
from scripts.eval.run_h6_acceptance import (
    _evaluate_turn,
    _numbered_product_refs,
    _overall_passed,
    _recommendation_display_mentions,
    _strict_displayed_identity_failures,
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
        "同时写出工具返回的 product_id 和完整原始 title",
        "title 必须逐字复制",
        "N. product_id | 完整原始 title",
        "product_id 必须逐字符复制完整值",
        "身份行数量和顺序必须与 selected_products 完全一致",
        "最终清单与 selected_products 必须恰好 N 件",
    )
    for fragment in required:
        assert fragment in prompt

    search_prompt = load_prompts()["sub_agents"]["search"]["system_prompt"]
    assert "title 必须逐字透传" in search_prompt
    assert "禁止翻译、缩写、改名或拼接" in search_prompt
    assert "原始 hits 有 3、4 或 5 件" in search_prompt
    assert "连 category 的 None/非 None 状态也必须保持不变" in search_prompt


def test_category_switch_accepts_equivalent_normalized_query_whitespace() -> None:
    turn = {
        "expectation": "category_switch",
        "exception": "",
        "final_output": "已切换到新类别。",
        "displayed_products": [],
        "events": [
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": "product_search_tool",
                    "args": {
                        "normalized_query": "降噪 耳机",
                        "ship_to": None,
                        "price_max_major": None,
                        "top_k": 5,
                    },
                },
            },
        ],
    }

    assert _evaluate_turn(turn) == []


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


def test_structured_selection_uses_only_exact_successful_hits_in_model_order() -> None:
    events = [
        SimpleNamespace(
            type="tool.result",
            payload={
                "tool": "product_search_tool",
                "platform": "amazon",
                "site_locale": "jp",
                "hits": [
                    {"product_id": "P1", "title": "完整原标题一"},
                    {"product_id": "P2", "title": "完整原标题二"},
                ],
            },
        ),
    ]
    selected = (
        SelectedProductRef(platform="amazon", product_id="P2"),
        SelectedProductRef(platform="amazon", product_id="UNKNOWN"),
        SelectedProductRef(platform="amazon", product_id="P1"),
    )

    displayed = _resolve_displayed_products(selected, events)

    assert [item["card"]["product_id"] for item in displayed] == ["P2", "P1"]
    assert [item["rank"] for item in displayed] == [1, 2]
    assert displayed[0]["card"]["title"] == "完整原标题二"


def test_structured_selection_derives_unscoped_amazon_locale_from_tool() -> None:
    events = [
        SimpleNamespace(
            type="tool.result",
            payload={
                "tool": "product_search_tool",
                "platform": "amazon",
                "site_locale": None,
                "hits": [
                    {
                        "product_id": "amazon:jp:B08PHF2DPK",
                        "title": "完全一致的原标题",
                    },
                ],
            },
        ),
    ]
    selected = (
        SelectedProductRef(
            platform="amazon",
            product_id="amazon:jp:B08PHF2DPK",
        ),
    )

    displayed = _resolve_displayed_products(selected, events)

    assert displayed[0]["site_locale"] is None
    assert displayed[0]["card"]["title"] == "完全一致的原标题"


def test_structured_selection_schema_forbids_model_supplied_site_locale() -> None:
    with pytest.raises(ValueError):
        SelectedProductRef.model_validate(
            {
                "platform": "amazon",
                "site_locale": "jp",
                "product_id": "amazon:jp:B08PHF2DPK",
            },
        )


def test_product_recommendation_semantic_cache_is_bypassed_for_l4_continuity() -> None:
    displayed = (
        {
            "rank": 1,
            "platform": "taobao",
            "site_locale": None,
            "card": {"product_id": "P1", "title": "完整原标题"},
        },
    )

    class Cache:
        remember_calls = 0

        async def lookup(self, *_args, **_kwargs):
            return SemanticHit("推荐 P1", 1.0, "找商品", displayed)

        async def remember(self, *_args, **_kwargs):
            self.remember_calls += 1

    class Preferences:
        async def list_by_buyer(self, _buyer_id):
            return []

    cache = Cache()
    orchestrator = MainAgentOrchestrator(
        SimpleNamespace(),
        TradeEventBus(),
        Preferences(),
        semantic_cache=cache,
    )
    intent = SubmitIntentInput("s1", "b1", "zh-CN", "CNY", "找商品")

    assert asyncio.run(orchestrator._lookup_cache(intent, False)) is None  # noqa: SLF001
    asyncio.run(
        orchestrator._remember_cache(  # noqa: SLF001
            intent,
            "推荐 P1",
            False,
            displayed,
        ),
    )
    assert cache.remember_calls == 0


def test_numbered_refs_prefer_verified_structured_display_order() -> None:
    turn = {
        "final_output": "不依赖 Markdown 编号格式",
        "events": [],
        "displayed_products": [
            {"rank": 3, "card": {"product_id": "P3"}},
            {"rank": 5, "card": {"product_id": "P5"}},
        ],
    }

    assert _numbered_product_refs(turn) == {3: "P3", 5: "P5"}


def test_strict_identity_rejects_rewritten_title_even_with_real_product_id() -> None:
    turn = {
        "final_output": "推荐 P1：原标题一精简版",
        "displayed_products": [
            {
                "rank": 1,
                "platform": "taobao",
                "site_locale": None,
                "card": {"product_id": "P1", "title": "完整原标题一"},
            },
        ],
        "events": [
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "platform": "taobao",
                    "site_locale": None,
                    "hits": [{"product_id": "P1", "title": "完整原标题一"}],
                },
            },
        ],
    }

    failures = _strict_displayed_identity_failures(turn)

    assert failures == ["final text omitted exact title: P1"]


def test_strict_identity_rejects_cross_paired_ids_and_titles() -> None:
    turn = {
        "final_output": "1. P1：完整原标题二\n2. P2：完整原标题一",
        "displayed_products": [
            {
                "rank": 1,
                "platform": "taobao",
                "site_locale": None,
                "card": {"product_id": "P1", "title": "完整原标题一"},
            },
            {
                "rank": 2,
                "platform": "taobao",
                "site_locale": None,
                "card": {"product_id": "P2", "title": "完整原标题二"},
            },
        ],
        "events": [
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "platform": "taobao",
                    "site_locale": None,
                    "hits": [
                        {"product_id": "P1", "title": "完整原标题一"},
                        {"product_id": "P2", "title": "完整原标题二"},
                    ],
                },
            },
        ],
    }

    failures = _strict_displayed_identity_failures(turn)

    assert failures == [
        "final text did not pair exact id and title: P1",
        "final text did not pair exact id and title: P2",
    ]


def test_strict_identity_rejects_cross_paired_ids_on_one_line() -> None:
    turn = {
        "final_output": "P1：完整原标题二；P2：完整原标题一",
        "displayed_products": [
            {
                "rank": 1,
                "platform": "taobao",
                "site_locale": None,
                "card": {"product_id": "P1", "title": "完整原标题一"},
            },
            {
                "rank": 2,
                "platform": "taobao",
                "site_locale": None,
                "card": {"product_id": "P2", "title": "完整原标题二"},
            },
        ],
        "events": [
            {
                "type": "tool.result",
                "payload": {
                    "tool": "product_search_tool",
                    "platform": "taobao",
                    "site_locale": None,
                    "hits": [
                        {"product_id": "P1", "title": "完整原标题一"},
                        {"product_id": "P2", "title": "完整原标题二"},
                    ],
                },
            },
        ],
    }

    failures = _strict_displayed_identity_failures(turn)

    assert failures == [
        "final text did not pair exact id and title: P1",
        "final text did not pair exact id and title: P2",
    ]


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
