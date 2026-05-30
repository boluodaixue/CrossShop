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
    shopping_session_id: str = Field(min_length=1, description="确认会话 ID")
    buyer_id: str = Field(min_length=1, description="买家 ID")
    idempotency_key: str | None = Field(default=None, min_length=1)


class OrderItemRequest(BaseModel):
    item_id: str = Field(min_length=1)
    variant_id: str | None = None
    quantity: int = Field(default=1, ge=1)


class PrepareOrderRequest(BaseModel):
    shopping_session_id: str = Field(min_length=1)
    buyer_id: str = Field(min_length=1)
    items: list[OrderItemRequest] = Field(min_length=1)
    shipping_address: dict = Field(min_length=1)
    idempotency_key: str | None = Field(default=None, min_length=1)


class ConfirmOrderRequest(BaseModel):
    shopping_session_id: str = Field(min_length=1)
    confirmation_id: str = Field(min_length=1)
    confirmation_token: str = Field(min_length=1)


class ConfirmCancelRequest(BaseModel):
    shopping_session_id: str = Field(min_length=1)
    confirmation_id: str = Field(min_length=1)
    confirmation_token: str = Field(min_length=1)
