# -*- coding: utf-8 -*-
"""presentation DTO

REST 请求 / 响应模型。shopping_session_id 缺省时由服务端生成。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SubmitIntentRequest(BaseModel):
    shopping_session_id: Optional[str] = Field(
        default=None, description="会话 ID，缺省则新建会话"
    )
    buyer_id: str = Field(min_length=1, description="买家 ID")
    locale: str = Field(default="zh-CN")
    currency: str = Field(default="CNY")
    raw_query: str = Field(min_length=1, description="买家自然语言购物意图")


class DisplayedProduct(BaseModel):
    rank: int = Field(ge=1, le=5)
    platform: Literal["globex_reference", "taobao", "amazon"]
    site_locale: Literal["us", "es", "jp"] | None = None
    card: dict[str, Any]


class SubmitIntentResponse(BaseModel):
    shopping_session_id: str
    final_text: str
    displayed_products: list[DisplayedProduct] = Field(default_factory=list)


class CancelOrderRequest(BaseModel):
    reason: str = Field(min_length=1, description="取消原因")
