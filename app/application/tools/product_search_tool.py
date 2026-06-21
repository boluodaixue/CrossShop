"""product_search_tool

商品检索工具：结构化检索入参 → CatalogSearchUseCase → 商品卡 JSON。
MainAgent 单干与 SearchAgent 派发两条路径共用同一工具实例。
工厂模式注入 UseCase 与 EventBus，模型看到的只是工具入参与返回值结构。

注意：本模块不能用 `from __future__ import annotations`——
AgentScope 用 pydantic 从函数签名动态生成 JSON schema，字符串化注解会解析失败。
"""

import json
from collections.abc import Mapping
from typing import Literal

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

from app.application.usecases.catalog_search import CatalogSearchUseCase
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext
from app.infrastructure.eventbus import TradeEventBus

_PLATFORMS = frozenset({"globex_reference", "taobao", "amazon"})
_AMAZON_SITE_LOCALES = frozenset({"us", "es", "jp"})


def build_product_search_tool(
    usecase: CatalogSearchUseCase | Mapping[str, CatalogSearchUseCase],
    bus: TradeEventBus,
    *,
    fixed_platform: str | None = None,
    fixed_site_locale: str | None = None,
):
    """构建唯一公开的商品检索工具。

    单平台旧装配仍可直接传 ``CatalogSearchUseCase``；H4 三平台装配传映射，键为
    ``globex_reference`` / ``taobao`` / ``amazon``。Amazon 明确站点可额外提供
    ``amazon:us`` / ``amazon:es`` / ``amazon:jp`` 对应的 UseCase。平台与站点只在
    工具层选择已装配的 UseCase，不写入 ProductSearchSpec 或商品领域对象。

    ``fixed_platform`` 仅供派发后的 SearchAgent 固定任务边界；工具公开名称仍然只有
    ``product_search_tool``。
    """
    if fixed_platform is not None and fixed_platform not in _PLATFORMS:
        raise ValueError(f"不支持的平台：{fixed_platform}")
    if fixed_site_locale is not None:
        if fixed_platform != "amazon":
            raise ValueError("site_locale 仅可用于 amazon 平台")
        if fixed_site_locale not in _AMAZON_SITE_LOCALES:
            raise ValueError(f"不支持的 Amazon 站点：{fixed_site_locale}")

    async def product_search_tool(
        normalized_query: str,
        platform: Literal["globex_reference", "taobao", "amazon"] | None = None,
        site_locale: Literal["us", "es", "jp"] | None = None,
        category: str | None = None,
        ship_to: str | None = None,
        locale: str = "zh-CN",
        top_k: Literal[1, 2, 3, 4, 5, "1", "2", "3", "4", "5"] = 5,
        price_max_major: float | str | None = None,
        target_currency: str = "CNY",
    ) -> ToolChunk:
        """检索跨境商品库（embedding+rerank 二阶段召回），返回 Top-K 商品卡 JSON。
        传入 ship_to 时商品卡自动内联 landed_price 到手价明细
        （小计+运费+关税，统一折算 target_currency），无需另行计算价格。

        Args:
            normalized_query (`str`):
                标准化检索词，保留品类词与关键属性词
                （如"旅行三件套 抗造 轻便 无塑料"）。
            platform (`str | None`):
                商品平台："globex_reference"、"taobao" 或 "amazon"。
                三平台装配时必须明确传入；它只选择固定平台索引，不进入
                ProductSearchSpec。
            site_locale (`str | None`):
                仅当买家明确指定 Amazon 站点时传 "us"、"es" 或 "jp"；
                普通 locale 不得填入这里，也不得作为商品硬约束。
            category (`str | None`):
                品类槽位，可选，如"旅行装备"、"数码配件"。
            ship_to (`str | None`):
                收货国家二位码，可选，如 "CN"、"US"；传入后过滤不可送达商品，
                并内联到手价。
            locale (`str`):
                买家语言与本地化区域，默认 "zh-CN"；不作为普通商品硬约束。
            top_k (`int`):
                返回候选数量，只允许 1..5，默认 5；每次调用不得超过 5。
            price_max_major (`float | None`):
                价格上限（target_currency 主单位），买家有预算硬约束时必传，
                由检索链路结构化过滤。
            target_currency (`str`):
                价格口径币种，默认 "CNY"。
        """
        selected_platform = fixed_platform or platform
        selected_site_locale = fixed_site_locale or site_locale
        if fixed_platform is not None and platform not in (None, fixed_platform):
            return _error_chunk(
                f"当前检索任务已固定平台 {fixed_platform}，不能切换为 {platform}",
            )
        if fixed_site_locale is not None and site_locale not in (
            None,
            fixed_site_locale,
        ):
            return _error_chunk(
                f"当前 Amazon 检索任务已固定站点 {fixed_site_locale}，"
                f"不能切换为 {site_locale}",
            )
        selection_error = _validate_selection(
            selected_platform,
            selected_site_locale,
            multi_platform=isinstance(usecase, Mapping),
        )
        if selection_error:
            return _error_chunk(selection_error)

        # 模型有时会把数字参数当字符串传（实测 qwen3-max 传 "300"），
        # schema 层放宽为接受数字字符串，这里统一强转后再进检索链路。
        if isinstance(top_k, str):
            try:
                top_k = int(top_k)
            except ValueError:
                return _error_chunk(f"top_k 非法：{top_k}")
        if isinstance(price_max_major, str):
            try:
                price_max_major = float(price_max_major)
            except ValueError:
                return _error_chunk(f"price_max_major 非法：{price_max_major}")
        session_id = ShoppingContext.current_session_id()
        args = {
            "normalized_query": normalized_query,
            "platform": selected_platform,
            "site_locale": selected_site_locale,
            "category": category,
            "ship_to": ship_to,
            "locale": locale,
            "top_k": top_k,
            "price_max_major": price_max_major,
            "target_currency": target_currency,
        }
        bus.publish(
            session_id, "tool.invoke", {"tool": "product_search_tool", "args": args}
        )
        try:
            spec = ProductSearchSpec(
                normalized_query=normalized_query,
                category=category,
                ship_to=ship_to,
                locale=locale,
                top_k=top_k,
                price_max_major=price_max_major,
                target_currency=target_currency,
            )
            selected_usecase = _select_usecase(
                usecase,
                platform=selected_platform,
                site_locale=selected_site_locale,
            )
            result = await selected_usecase.execute(spec)
        except ValueError as err:
            bus.publish(
                session_id,
                "tool.result",
                {"tool": "product_search_tool", "error": str(err)},
            )
            return ToolChunk(
                content=[TextBlock(type="text", text=f"[error] {err}")],
                state=ToolResultState.ERROR,
            )
        bus.publish(
            session_id,
            "tool.result",
            {
                "tool": "product_search_tool",
                "platform": selected_platform,
                "site_locale": selected_site_locale,
                "hit_count": len(result["hits"]),
                "recall_strategy": result["recall_strategy"],
                # 商品卡随事件下发，前端无需再调接口即可渲染（含 landed_price 到手价）
                "hits": result["hits"],
            },
        )
        return ToolChunk(
            content=[
                TextBlock(type="text", text=json.dumps(result, ensure_ascii=False))
            ],
            state=ToolResultState.SUCCESS,
        )

    return product_search_tool


def _validate_selection(
    platform: str | None,
    site_locale: str | None,
    *,
    multi_platform: bool,
) -> str | None:
    if platform is None:
        if multi_platform:
            return "三平台检索必须明确指定 platform"
        if site_locale is not None:
            return "site_locale 必须与 amazon 平台同时使用"
        return None
    if platform not in _PLATFORMS:
        return f"不支持的平台：{platform}"
    if site_locale is None:
        return None
    if platform != "amazon":
        return "site_locale 仅可用于 amazon 平台"
    if site_locale not in _AMAZON_SITE_LOCALES:
        return f"不支持的 Amazon 站点：{site_locale}"
    return None


def _select_usecase(
    usecase: CatalogSearchUseCase | Mapping[str, CatalogSearchUseCase],
    *,
    platform: str | None,
    site_locale: str | None,
) -> CatalogSearchUseCase:
    if not isinstance(usecase, Mapping):
        return usecase
    key = f"amazon:{site_locale}" if site_locale is not None else platform
    if key not in usecase:
        raise ValueError(f"平台检索尚未装配：{key}")
    return usecase[key]


def _error_chunk(message: str) -> ToolChunk:
    return ToolChunk(
        content=[TextBlock(type="text", text=f"[error] {message}")],
        state=ToolResultState.ERROR,
    )
