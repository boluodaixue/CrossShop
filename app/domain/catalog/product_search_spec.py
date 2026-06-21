# -*- coding: utf-8 -*-
"""ProductSearchSpec 值对象

SearchAgent 把买家自然语言 query 改写为标准化检索规格：
normalized_query 用于召回，结构化槽位由 CatalogSearchUseCase 按既定语义消费。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProductSearchSpec:
    normalized_query: str
    category: Optional[str] = None
    ship_to: Optional[str] = None
    locale: str = "zh-CN"
    top_k: int = 5
    # 到手价目标币种：命中 ship_to 时商品卡内联 landed_price（小计+运费+关税）
    target_currency: str = "CNY"
    # 价格硬约束（目标币种主单位）：由检索链路结构化过滤，
    # 不交给 embedding/reranker。
    price_max_major: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.normalized_query or not self.normalized_query.strip():
            raise ValueError("ProductSearchSpec.normalized_query required")
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or not 1 <= self.top_k <= 5
        ):
            raise ValueError("ProductSearchSpec.top_k 必须为 1..5 的整数")
