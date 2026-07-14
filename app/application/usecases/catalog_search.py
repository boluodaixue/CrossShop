# -*- coding: utf-8 -*-
"""CatalogSearchUseCase

商品检索核心 UseCase，对齐参考实现五步流程：
    1. EmbeddingClient 把 normalized_query 向量化
    2. ProductVectorIndex.search(query, embedding, top_n) 拿候选 product_id
    3. ProductRepository.find_by_ids 还原 Product 聚合
    4. 先执行 ship_to / SKU 库存与预算硬约束，只把合格 Product 送入 Reranker
    5. Reranker 精排取 top_k；失败/未配置降级按向量分排序（rerank_applied=false）
    6. 组装商品卡 JSON；只投影本轮合格 SKU，并以内含最低价 SKU 展示起价

独立教学装配默认保留原三级降级链：
    embedding_rerank → embedding_only → keyword_2gram

H4 在线三平台装配显式关闭 keyword fallback。此时 Query Embedding 或 OpenSearch 异常
属于商品检索基础设施故障，必须明确报错；不得扫描共享全量 Repository 做本地关键词
二查，否则平台/站点任务会越界返回其他目录的商品。

计价收敛设计：到手价在检索链路内联计算（TariffSchedule 规则内核），
不给 Agent 单独暴露比价/运费工具，减少不必要的工具调用轮次。

过滤可观测：被 ship_to / price_max_major 硬约束挡掉的候选以 filtered_out 摘要回传，
让模型能区分"库里没有这个商品"与"有但不满足约束"，不致于给出误导性结论。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.domain.catalog.exchange_rate import ExchangeRateTable
from app.domain.catalog.money import Money
from app.domain.catalog.ports.product_repository import ProductRepository
from app.domain.catalog.ports.retrieval_ports import (
    EmbeddingClient,
    ProductVectorIndex,
    Reranker,
)
from app.domain.catalog.product import Product
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.domain.catalog.sku import Sku
from app.domain.shipping.tariff_schedule import TariffSchedule
from app.infrastructure.tracing import set_span_attributes, trace_span

logger = logging.getLogger(__name__)

# 一阶段召回候选数（> top_k，给精排留空间）
_RECALL_TOP_N = 8

# 被硬约束挡掉的候选回传条数上限（只回摘要，避免上下文膨胀）
_FILTERED_OUT_LIMIT = 3


class _ProductHydrationError(RuntimeError):
    """向量命中与严格 Product Repository 不一致，禁止降级掩盖。"""


class CatalogRetrievalError(RuntimeError):
    """Query Embedding/OpenSearch 商品召回基础设施异常。"""


@dataclass(frozen=True)
class ProductCard:
    product_id: str
    title: str
    brand: str
    category: str
    origin_country: str
    price_major: float
    currency: str
    highlights: list[str]
    skus: list[dict]
    score: float
    landed_price: Optional[dict]  # ship_to 命中时的到手价明细，未命中为 None

    def to_dict(self) -> dict:
        card = {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "category": self.category,
            "origin_country": self.origin_country,
            "price_major": self.price_major,
            "currency": self.currency,
            "highlights": self.highlights,
            "skus": self.skus,
            "score": round(self.score, 4),
        }
        if self.landed_price is not None:
            card["landed_price"] = self.landed_price
        return card


def tokenize(text: str) -> set[str]:
    """极简分词：空格切词 + 中文连续段落的 2-gram（教学关键词兜底用）。"""
    terms: set[str] = set()
    for chunk in text.lower().split():
        terms.add(chunk)
        if any("\u4e00" <= ch <= "\u9fff" for ch in chunk) and len(chunk) >= 2:
            terms.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return terms


class CatalogSearchUseCase:
    def __init__(
        self,
        product_repo: ProductRepository,
        embedder: Optional[EmbeddingClient] = None,
        vector_index: Optional[ProductVectorIndex] = None,
        reranker: Optional[Reranker] = None,
        tariff_schedule: Optional[TariffSchedule] = None,
        *,
        allow_keyword_fallback: bool = True,
    ) -> None:
        self._product_repo = product_repo
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._tariff = tariff_schedule or TariffSchedule(rates=ExchangeRateTable())
        self._allow_keyword_fallback = allow_keyword_fallback

    async def execute(self, spec: ProductSearchSpec) -> dict:
        rerank_applied = False
        if self._embedder is None or self._vector_index is None:
            if not self._allow_keyword_fallback:
                raise CatalogRetrievalError("商品 Hybrid 检索基础设施未配置")
            scored = await self._keyword_recall(spec)
            recall_strategy = "keyword_2gram"
        else:
            try:
                scored = await self._vector_recall(spec)
                recall_strategy = "embedding_only"
            except _ProductHydrationError:
                # 全量 Repository 与索引不一致属于数据契约损坏，不能重试或降级掩盖。
                raise
            except Exception as err:
                if not self._allow_keyword_fallback:
                    raise CatalogRetrievalError("商品 Hybrid 检索不可用") from err
                logger.warning("向量召回不可用，降级关键词召回：%s", err)
                scored = await self._keyword_recall(spec)
                recall_strategy = "keyword_2gram"

        # 库存 / ship_to / 价格硬约束必须先于精排。全 SKU 缺货的 Product 静默排除，
        # 保持 OpenSearch stock>0 前置过滤语义，不虚构第三种 filtered_out 原因。
        with trace_span(
            "crossshop.product.constraints",
            {
                "crossshop.constraints.candidate_count": len(scored),
                "crossshop.constraints.has_ship_to": bool(spec.ship_to),
                "crossshop.constraints.has_price_cap": spec.price_max_major is not None,
            },
        ) as constraint_span:
            eligible, filtered_out = self._apply_constraints(scored, spec)
            set_span_attributes(
                constraint_span,
                {
                    "crossshop.constraints.eligible_count": len(eligible),
                    "crossshop.constraints.filtered_count": len(filtered_out),
                },
            )

        if recall_strategy == "embedding_only" and eligible:
            # 只精排合格候选；失败时沿用原向量分和原顺序，不发第二次召回。
            try:
                eligible = await self._rerank(spec, eligible)
                recall_strategy = "embedding_rerank"
                rerank_applied = True
            except Exception as err:  # noqa: BLE001
                logger.warning("rerank 不可用，按向量分排序：%s", err)

        hits = [
            self._to_card(score, product, spec)
            for score, product in eligible[: spec.top_k]
        ]
        result = {
            "hits": [card.to_dict() for card in hits],
            "total_candidates": len(eligible),
            "recall_strategy": recall_strategy,
            "rerank_applied": rerank_applied,
        }
        if filtered_out:
            # 如实告知"召回到了但被硬约束挡掉"，否则模型分不清"库里没有"与"被过滤"，
            # 会把超预算商品答成"没有这个商品"
            result["filtered_out"] = filtered_out
        return result

    def _apply_constraints(
        self,
        scored: list[tuple[float, Product]],
        spec: ProductSearchSpec,
    ) -> tuple[list[tuple[float, Product]], list[dict]]:
        eligible: list[tuple[float, Product]] = []
        filtered_out: list[dict] = []
        for score, product in scored:
            if not self._in_stock_skus(product):
                continue
            reason = self._reject_reason(product, spec)
            if reason is None:
                eligible.append((score, product))
            elif len(filtered_out) < _FILTERED_OUT_LIMIT:
                filtered_out.append(self._to_rejected(product, spec, reason))
        return eligible, filtered_out

    def _reject_reason(
        self, product: Product, spec: ProductSearchSpec
    ) -> Optional[str]:
        """返回硬约束拒绝原因，None 表示通过。"""
        if spec.ship_to and spec.ship_to not in product.ships_to:
            return "ship_to_unavailable"
        if not self._within_price_cap(product, spec):
            return "over_price_cap"
        return None

    def _to_rejected(
        self, product: Product, spec: ProductSearchSpec, reason: str
    ) -> dict:
        lowest_in_stock = min(
            self._converted_in_stock_skus(product, spec),
            key=lambda pair: (pair[1].amount_in_minor_units, pair[0].sku_id),
        )[1]
        return {
            "product_id": product.product_id,
            "title": product.title,
            "category": product.category,
            "price_major": round(lowest_in_stock.to_major_units(), 2),
            "currency": spec.target_currency,
            "reason": reason,
        }

    def _within_price_cap(self, product: Product, spec: ProductSearchSpec) -> bool:
        if spec.price_max_major is None:
            return True
        return any(
            price.to_major_units() <= spec.price_max_major
            for _, price in self._converted_in_stock_skus(product, spec)
        )

    @staticmethod
    def _in_stock_skus(product: Product) -> list[Sku]:
        return [sku for sku in product.skus if sku.stock > 0]

    def _converted_in_stock_skus(
        self,
        product: Product,
        spec: ProductSearchSpec,
    ) -> list[tuple[Sku, Money]]:
        return [
            (sku, self._tariff.rates.convert(sku.price, spec.target_currency))
            for sku in self._in_stock_skus(product)
        ]

    def _eligible_skus(
        self,
        product: Product,
        spec: ProductSearchSpec,
    ) -> list[tuple[Sku, Money]]:
        candidates = self._converted_in_stock_skus(product, spec)
        if spec.price_max_major is not None:
            candidates = [
                pair
                for pair in candidates
                if pair[1].to_major_units() <= spec.price_max_major
            ]
        candidates.sort(
            key=lambda pair: (pair[1].amount_in_minor_units, pair[0].sku_id)
        )
        return candidates

    # ---- 一阶段：向量召回 ----

    async def _vector_recall(
        self, spec: ProductSearchSpec
    ) -> list[tuple[float, Product]]:
        embedding = await self._embedder.embed(spec.normalized_query)
        vector_hits = await self._vector_index.search(
            query=spec.normalized_query,
            embedding=embedding,
            top_n=_RECALL_TOP_N,
        )
        hit_ids = [hit.product_id for hit in vector_hits]
        if len(hit_ids) != len(set(hit_ids)):
            raise _ProductHydrationError("向量召回返回重复 product_id")
        if not hit_ids:
            return []

        products = await self._product_repo.find_by_ids(hit_ids)
        product_ids = [product.product_id for product in products]
        if len(product_ids) != len(set(product_ids)):
            raise _ProductHydrationError("ProductRepository 返回重复 product_id")

        missing_ids = sorted(set(hit_ids) - set(product_ids))
        unexpected_ids = sorted(set(product_ids) - set(hit_ids))
        if missing_ids or unexpected_ids:
            raise _ProductHydrationError(
                "ProductRepository 无法精确还原向量候选："
                f"missing={missing_ids}, unexpected={unexpected_ids}",
            )

        by_id = {product.product_id: product for product in products}
        return [(hit.score, by_id[hit.product_id]) for hit in vector_hits]

    # ---- 二阶段：精排 ----

    async def _rerank(
        self,
        spec: ProductSearchSpec,
        scored: list[tuple[float, Product]],
    ) -> list[tuple[float, Product]]:
        if self._reranker is None:
            raise RuntimeError("Reranker 未配置")
        documents = [product.searchable_text() for _, product in scored]
        rerank_scores = await self._reranker.rerank(spec.normalized_query, documents)
        reranked = [
            (rerank_scores[i], product) for i, (_, product) in enumerate(scored)
        ]
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        return reranked

    # ---- 独立教学装配兜底（H4 在线平台路由显式关闭）----

    async def _keyword_recall(
        self, spec: ProductSearchSpec
    ) -> list[tuple[float, Product]]:
        query_terms = tokenize(spec.normalized_query)
        candidates: list[tuple[float, Product]] = []
        for product in await self._product_repo.list_all():
            if not self._in_stock_skus(product):
                continue
            score = self._keyword_score(query_terms, product, spec)
            if score > 0:
                candidates.append((score, product))
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates

    @staticmethod
    def _keyword_score(
        query_terms: set[str], product: Product, spec: ProductSearchSpec
    ) -> float:
        doc_terms = tokenize(product.searchable_text())
        matched = query_terms & doc_terms
        if not matched:
            return 0.0
        score = float(len(matched))
        if spec.category and spec.category in product.category:
            score += 3.0
        return score

    # ---- 商品卡组装（含到手价内联）----

    def _to_card(
        self, score: float, product: Product, spec: ProductSearchSpec
    ) -> ProductCard:
        eligible_skus = self._eligible_skus(product, spec)
        if not eligible_skus:
            raise ValueError(f"Product 没有本轮合格 SKU：{product.product_id}")
        primary, primary_in_target = eligible_skus[0]
        landed_price: Optional[dict] = None
        if spec.ship_to:
            try:
                quote = self._tariff.quote(
                    subtotal=primary.price,
                    category=product.category,
                    ship_to=spec.ship_to,
                    quantity=1,
                    target_currency=spec.target_currency,
                )
                landed_price = quote.to_dict()
            except ValueError as err:
                # 目的国不在规则表内：如实标注，不编造数字
                landed_price = {"unavailable_reason": str(err)}
        return ProductCard(
            product_id=product.product_id,
            title=product.title,
            brand=product.brand,
            category=product.category,
            origin_country=product.origin_country,
            price_major=primary_in_target.to_major_units(),
            currency=primary_in_target.currency,
            highlights=[
                f"{h.label}：{h.detail}" if h.detail else h.label
                for h in product.highlights
            ],
            skus=[
                {
                    "sku_id": sku.sku_id,
                    "spec": sku.spec,
                    "price_major": price.to_major_units(),
                    "currency": price.currency,
                    "stock": sku.stock,
                }
                for sku, price in eligible_skus
            ],
            score=score,
            landed_price=landed_price,
        )
