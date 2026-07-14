"""H4：单一商品工具的平台选择与 SearchAgent 固定派发协议。"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields

from agentscope.message import AssistantMsg
from agentscope.tool import FunctionTool

from app.application.agents.search_agent import SearchAgentFactory
from app.application.prompts.loader import load_prompts
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus


class _CapturingUseCase:
    def __init__(self, label: str) -> None:
        self.label = label
        self.specs: list[ProductSearchSpec] = []

    async def execute(self, spec: ProductSearchSpec) -> dict:
        self.specs.append(spec)
        return {
            "hits": [{"product_id": self.label}],
            "recall_strategy": "embedding_rerank",
        }


class _Worker:
    async def reply(self, inputs):
        del inputs
        return AssistantMsg("worker", '{"hits": []}')


class _RecordingFactory:
    def __init__(self) -> None:
        self.scopes: list[tuple[str | None, str | None]] = []

    def build(
        self,
        *,
        platform: str | None = None,
        site_locale: str | None = None,
    ) -> _Worker:
        self.scopes.append((platform, site_locale))
        return _Worker()


def _context():
    return ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="h4-routing",
            buyer_id="buyer",
            locale="zh-CN",
            currency="CNY",
        ),
    )


def _text(chunk) -> str:
    return chunk.content[0].text


async def test_one_product_search_tool_selects_platform_outside_spec() -> None:
    reference = _CapturingUseCase("reference")
    reference_seed = _CapturingUseCase("reference_seed")
    tool = build_product_search_tool(
        {"crossshop_reference": reference, "reference_seed": reference_seed},  # type: ignore[arg-type]
        TradeEventBus(),
    )

    token = _context()
    try:
        response = await tool(
            normalized_query="轻便旅行背包",
            platform="reference_seed",
            ship_to="CN",
        )
    finally:
        ShoppingContext.reset(token)

    assert json.loads(_text(response))["hits"] == [{"product_id": "reference_seed"}]
    assert reference.specs == []
    assert len(reference_seed.specs) == 1
    assert [field.name for field in fields(reference_seed.specs[0])] == [
        "normalized_query",
        "category",
        "ship_to",
        "locale",
        "top_k",
        "target_currency",
        "price_max_major",
    ]


def test_product_search_tool_schema_only_admits_top_k_one_through_five() -> None:
    tool = build_product_search_tool(
        _CapturingUseCase("reference"),  # type: ignore[arg-type]
        TradeEventBus(),
    )
    schema = FunctionTool(tool).input_schema["properties"]["top_k"]
    allowed = set(schema["enum"])
    assert allowed == {1, 2, 3, 4, 5, "1", "2", "3", "4", "5"}
    assert schema["default"] == 5
    assert 8 not in allowed and "8" not in allowed


async def test_amazon_site_selects_same_platform_site_binding_not_spec_locale() -> None:
    amazon = _CapturingUseCase("amazon-all")
    amazon_jp = _CapturingUseCase("amazon-jp")
    bus = TradeEventBus()
    queue = bus.subscribe("h4-routing")
    tool = build_product_search_tool(
        {"amazon": amazon, "amazon:jp": amazon_jp},  # type: ignore[arg-type]
        bus,
    )

    token = _context()
    try:
        response = await tool(
            normalized_query="旅行用品",
            platform="amazon",
            site_locale="jp",
            locale="zh-CN",
        )
    finally:
        ShoppingContext.reset(token)

    assert json.loads(_text(response))["hits"] == [{"product_id": "amazon-jp"}]
    assert amazon.specs == []
    assert amazon_jp.specs[0].locale == "zh-CN"
    invoke = queue.get_nowait()
    assert invoke.payload["args"]["platform"] == "amazon"
    assert invoke.payload["args"]["site_locale"] == "jp"


async def test_multi_platform_tool_rejects_missing_or_invalid_scope() -> None:
    tool = build_product_search_tool(
        {"amazon": _CapturingUseCase("amazon")},  # type: ignore[arg-type]
        TradeEventBus(),
    )

    assert "必须明确指定 platform" in _text(
        await tool(normalized_query="旅行用品"),
    )
    assert "site_locale 仅可用于 amazon" in _text(
        await tool(
            normalized_query="旅行用品",
            platform="reference_seed",
            site_locale="jp",
        ),
    )


async def test_bound_search_tool_cannot_switch_platform_or_site() -> None:
    tool = build_product_search_tool(
        {"amazon:jp": _CapturingUseCase("amazon-jp")},  # type: ignore[arg-type]
        TradeEventBus(),
        fixed_platform="amazon",
        fixed_site_locale="jp",
    )

    assert "不能切换为 reference_seed" in _text(
        await tool(normalized_query="旅行用品", platform="reference_seed"),
    )
    assert "不能切换为 us" in _text(
        await tool(normalized_query="旅行用品", site_locale="us"),
    )


async def test_dispatch_requires_and_forwards_one_fixed_search_scope() -> None:
    search = _RecordingFactory()
    trade = _RecordingFactory()
    tool = build_task_dispatch_tool(search, trade, TradeEventBus())  # type: ignore[arg-type]

    missing = await tool(subagent_type="search_agent", demands="找旅行用品")
    assert "必须明确指定 platform" in _text(missing)
    await tool(
        subagent_type="search_agent",
        demands="只搜索 Amazon 日本站旅行用品",
        platform="amazon",
        site_locale="jp",
    )
    assert search.scopes == [("amazon", "jp")]

    invalid_trade = await tool(
        subagent_type="trade_agent",
        demands="查询订单",
        platform="reference_seed",
    )
    assert "trade_agent 不接受" in _text(invalid_trade)
    assert trade.scopes == []


def test_prompts_freeze_single_tool_dispatch_and_one_rewrite() -> None:
    prompts = load_prompts()
    main = prompts["main_agent"]["system_prompt"]
    search = prompts["sub_agents"]["search"]["system_prompt"]

    assert "这是唯一的商品检索工具" in main
    assert "一次简单检索" in main and "不要派发" in main
    assert "同一轮并发调用" in main
    assert "最终最多推荐 5 件商品" in main
    assert "N 等于返回 JSON 顶层原始 hits 数组的元素个数" in main
    assert "N >= 3 时立即停止商品检索" in main
    assert "第二次调用后无条件停止商品检索" in main
    assert "不能用筛选后的保留数量代替 N" in main
    assert "top_k 只允许 1..5" in main
    assert "商品检索次数判定（最高优先级，必须机械执行）" in search
    assert "N = 返回 JSON 顶层原始 hits 数组的元素个数" in search
    assert "真正满意的不足 3 件" in search
    assert "第二次调用后：无条件停止商品检索" in search
    assert "所有其他参数必须完全一致" in search
    assert "不得改变原始 hits 数量门槛" in search
    assert "top_k 默认传 5且只允许 1..5" in search
    assert "product_search_crossshop_reference_tool" not in main
    assert "product_search_reference_seed_tool" not in main
    assert "product_search_amazon_tool" not in main


def test_fixed_platform_binding_repeats_raw_hits_retry_gate() -> None:
    source = inspect.getsource(SearchAgentFactory.build)
    assert "重查门槛只看第一次" in source
    assert "hits 数组长度" in source
    assert "原始 hits >= 3 必须立即停止" in source
    assert "想凑满 5 件而重搜" in source
