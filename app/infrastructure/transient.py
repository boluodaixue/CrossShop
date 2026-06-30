# -*- coding: utf-8 -*-
"""transient

上游瞬时故障判定。模型层与编排层共用同一份判据，避免两处标记表各自漂移。

为什么按 message 特征匹配而不是按异常类型：OpenAI 兼容网关把限流写在 SSE 流中间时，
抛出的是笼统的 openai.APIError，类型上无法与真实业务错误区分，只能看文案；
同一套判据还要覆盖 httpx 超时与网关 5xx。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from app.infrastructure.tracing import set_span_attributes

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_RETRY_DELAY_SECONDS = 0.2

# 全小写匹配。"throttling" 来自实测：网关限流返回 code=Throttling.Concurrency
_TRANSIENT_ERROR_MARKERS = (
    "too many concurrent",
    "rate limit",
    "request rate",
    "too many requests",
    "throttling",
    "429",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "internal server error",
    "bad gateway",
    "connection reset",
    "connection error",
)


def is_transient_error(error: BaseException) -> bool:
    """判断异常是否属于可重试的上游瞬时故障。"""
    message = str(error).lower()
    return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def is_retryable_dependency_error(error: BaseException) -> bool:
    """仅识别获批的 Embedding/OpenSearch 传输层瞬时故障。"""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPStatusError):
            if current.response.status_code in _RETRYABLE_STATUS_CODES:
                return True
        elif isinstance(
            current,
            (
                httpx.TimeoutException,
                httpx.ConnectError,
            ),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


async def retry_dependency_once(
    operation: Callable[[], Awaitable[_T]],
    *,
    dependency: str,
    span: object | None = None,
) -> _T:
    """执行只读依赖请求；仅在批准的瞬时故障下以原参数重试一次。"""

    for attempt in (1, 2):
        try:
            result = await operation()
        except Exception as error:
            if attempt == 2 or not is_retryable_dependency_error(error):
                set_span_attributes(
                    span,
                    {
                        "globex.dependency.attempts": attempt,
                        "globex.dependency.retried": attempt > 1,
                    },
                )
                raise
            logger.warning(
                "%s 瞬时故障，同参数重试一次（dependency_attempt=2）：%s",
                dependency,
                type(error).__name__,
            )
            set_span_attributes(
                span,
                {
                    "globex.dependency.attempts": attempt,
                    "globex.dependency.retry_reason": type(error).__name__,
                },
            )
            await asyncio.sleep(_RETRY_DELAY_SECONDS)
            continue
        set_span_attributes(
            span,
            {
                "globex.dependency.attempts": attempt,
                "globex.dependency.retried": attempt > 1,
            },
        )
        return result
    raise AssertionError("dependency retry loop exhausted")
