"""Thin LangChain adapters around the stage-two deterministic business tools."""

from langchain_core.tools import tool

from globex_agent.agent.runtime import get_agent_runtime
from globex_agent.domain import (
    Candidate,
    Currency,
    LandedCost,
    MarketLocale,
    PickedItem,
    Platform,
    PricePoint,
)
from globex_agent.tools.item_picker import pick_items
from globex_agent.tools.item_search import search_items
from globex_agent.tools.price_compare import compare_prices
from globex_agent.tools.shipping_calc import calculate_shipping
from globex_agent.tools.shopping_summary import build_shopping_summary


@tool("item_search")
async def item_search_tool(
    query: str,
    platform: Platform,
    locale: MarketLocale | None = None,
    top_k: int = 20,
) -> str:
    """在指定平台检索商品候选集。

    Args:
        query: Planner 拆解后的具体搜索词。
        platform: 目标平台，一次只能传一个平台。
        locale: 目标市场；Amazon 多地区检索时必须指定 us、es 或 jp。
        top_k: 返回候选数量，默认 20，最大 50。
    """

    runtime = get_agent_runtime()
    backend = runtime.product_search_backend
    if runtime.product_search_router is not None:
        backend = runtime.product_search_router.backend_for(platform, locale)
    output = search_items(
        runtime.catalog,
        query=query,
        platform=platform,
        locale=locale,
        top_k=top_k,
        hard_constraints=runtime.request.hard_constraints,
        backend=backend,
        reranker=runtime.product_reranker,
        listing_languages=runtime.listing_languages,
    )
    runtime.artifacts.searches.append(output)
    return output.model_dump_json()


@tool("price_compare")
async def price_compare_tool(
    candidates: list[Candidate],
    base_currency: Currency = Currency.CNY,
    top_n: int = 12,
) -> str:
    """跨平台候选商品比价，输出币种归一后的排序。

    Args:
        candidates: 多个平台 ItemSearch 合流后的候选集。
        base_currency: 归一目标币种，默认人民币。
        top_n: 返回排序后的前 N 件，默认 12，最大 30。
    """

    runtime = get_agent_runtime()
    output = compare_prices(candidates, base_currency=base_currency, top_n=top_n)
    runtime.artifacts.price_comparison = output
    return output.model_dump_json()


@tool("shipping_calc")
async def shipping_calc_tool(
    points: list[PricePoint],
    destination: str = "CN",
) -> str:
    """为 PriceCompare.ranked 候选估算到手价。

    Args:
        points: PriceCompare 输出的 PricePoint 子集。
        destination: 收货国家 ISO 码，当前演示仅支持 CN。
    """

    runtime = get_agent_runtime()
    output = calculate_shipping(points, destination=destination)
    runtime.artifacts.shipping = output
    return output.model_dump_json()


@tool("item_picker")
async def item_picker_tool(
    landed: list[LandedCost],
    user_preferences: list[str] | None = None,
    top_n: int = 3,
) -> str:
    """从到手价候选中按硬约束和用户偏好精挑最多三件商品。

    Args:
        landed: ShippingCalc.items，已按到手价升序。
        user_preferences: 用户长期偏好句子；当前由本地画像补充。
        top_n: 最多返回的精选数量，默认 3。
    """

    del user_preferences  # The stage-two picker reads the injected local profile.
    runtime = get_agent_runtime()
    output = pick_items(
        landed,
        runtime.request,
        runtime.profile,
        top_n=top_n,
    )
    runtime.artifacts.selection = output
    return output.model_dump_json()


@tool("shopping_summary")
async def shopping_summary_tool(
    picks: list[PickedItem],
    user_query: str,
    new_preferences: list[str] | None = None,
) -> str:
    """生成最终购物清单和理由；这是购物链路的终结性工具。

    Args:
        picks: ItemPicker 返回的精选商品。
        user_query: 用户最初输入的购物意图原文。
        new_preferences: 本轮识别到、以后可写入长期记忆的新偏好。
    """

    runtime = get_agent_runtime()
    selection = runtime.artifacts.selection
    output = build_shopping_summary(
        picks,
        user_query,
        new_preferences,
        issues=selection.issues if selection else (),
        rejected_brief=selection.rejected_brief if selection else (),
    )
    runtime.artifacts.summary = output
    return output.model_dump_json()
