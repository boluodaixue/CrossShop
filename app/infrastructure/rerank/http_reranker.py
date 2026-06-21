"""HttpReranker

HTTP 精排客户端（默认对接 BAAI/bge-reranker-v2-m3 的 /rerank 服务）。
RERANKER_BASE_URL 未配置时组装根不会实例化本类；调用失败抛异常，
由 CatalogSearchUseCase 降级为按向量分排序并标注 rerank_applied=false。
"""

from __future__ import annotations

import math

import httpx

from app.domain.catalog.ports.retrieval_ports import Reranker
from app.infrastructure.settings import Settings


class HttpReranker(Reranker):
    def __init__(self, settings: Settings, timeout_seconds: float = 3.0) -> None:
        self._base_url = settings.reranker_base_url.rstrip("/")
        self._model = settings.reranker_model
        self._timeout = timeout_seconds

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/rerank",
                json={"model": self._model, "query": query, "documents": documents},
            )
            response.raise_for_status()
            body = response.json()
        return self._parse_scores(body, expected_count=len(documents))

    @staticmethod
    def _parse_scores(body: object, *, expected_count: int) -> list[float]:
        # 兼容 {results:[{index, relevance_score}]} 协议；BGE 本地服务返回原始 logit。
        if not isinstance(body, dict):
            raise HttpReranker._response_error("body 不是对象", body)
        results = body.get("results")
        if not isinstance(results, list) or len(results) != expected_count:
            raise HttpReranker._response_error("results 数量错误", body)

        scores = [0.0] * expected_count
        seen_indices: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise HttpReranker._response_error("result 不是对象", body)

            index = item.get("index")
            if type(index) is not int or not 0 <= index < expected_count:
                raise HttpReranker._response_error("index 非法", body)
            if index in seen_indices:
                raise HttpReranker._response_error("index 重复", body)

            if "relevance_score" in item:
                raw_score = item["relevance_score"]
            elif "score" in item:
                raw_score = item["score"]
            else:
                raise HttpReranker._response_error("缺少分数", body)
            try:
                score = float(raw_score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise HttpReranker._response_error("分数不是数值", body) from exc
            if not math.isfinite(score):
                raise HttpReranker._response_error("分数非有限值", body)

            seen_indices.add(index)
            scores[index] = score

        if seen_indices != set(range(expected_count)):
            raise HttpReranker._response_error("index 不完整", body)
        return scores

    @staticmethod
    def _response_error(reason: str, body: object) -> RuntimeError:
        return RuntimeError(f"rerank 响应异常（{reason}）：{str(body)[:200]}")
