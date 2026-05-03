"""REST request and response DTOs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubmitIntentRequest(BaseModel):
    shopping_session_id: str | None = Field(
        default=None,
        description="会话 ID，缺省则新建会话",
    )
    buyer_id: str = Field(min_length=1, description="买家 ID")
    locale: str = "zh-CN"
    currency: str = "CNY"
    raw_query: str = Field(min_length=1, description="买家自然语言购物意图")


class SubmitIntentResponse(BaseModel):
    shopping_session_id: str
    final_text: str


class CancelOrderRequest(BaseModel):
    reason: str = Field(min_length=1, description="取消原因")
