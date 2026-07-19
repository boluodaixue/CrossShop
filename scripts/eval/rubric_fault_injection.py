"""Scoped deterministic dependency faults for Rubric end-to-end cases.

The profile mutates only the already-built evaluation container and restores
every field when the case finishes.  Production composition and retry/query
semantics remain unchanged.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from app.composition import Container
from app.domain.catalog.ports.retrieval_ports import VectorHit


class _StaticEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        del text
        self.calls += 1
        return [0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class _TimeoutEmbedder(_StaticEmbedder):
    async def embed(self, text: str) -> list[float]:
        del text
        self.calls += 1
        raise TimeoutError("injected product embedding timeout")


class _StaticIndex:
    def __init__(self, product_ids: list[str], *, fail: bool = False) -> None:
        self._product_ids = product_ids
        self._fail = fail
        self.calls = 0

    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ) -> list[VectorHit]:
        del query, embedding, top_n
        self.calls += 1
        if self._fail:
            raise RuntimeError("injected OpenSearch outage")
        return [
            VectorHit(product_id=product_id, score=1.0 - index * 0.1)
            for index, product_id in enumerate(self._product_ids)
        ]


class _MissingProductRepository:
    def __init__(self, delegate: Any, missing_id: str) -> None:
        self._delegate = delegate
        self._missing_id = missing_id
        self.calls = 0

    async def find_by_id(self, product_id: str) -> Any:
        if product_id == self._missing_id:
            return None
        return await self._delegate.find_by_id(product_id)

    async def find_by_ids(self, product_ids: list[str]) -> list[Any]:
        self.calls += 1
        restored = await self._delegate.find_by_ids(product_ids)
        return [item for item in restored if item.product_id != self._missing_id]

    async def list_all(self) -> list[Any]:
        return await self._delegate.list_all()


class _BrokenReranker:
    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        del query, documents
        self.calls += 1
        raise RuntimeError("injected reranker outage")


class _FailingOrderRepository:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.calls = 0

    async def save(self, order: Any) -> None:
        del order
        self.calls += 1
        raise RuntimeError("injected order repository write failure")

    async def find_by_id(self, order_id: str) -> Any:
        return await self._delegate.find_by_id(order_id)

    async def next_order_id(self) -> str:
        return await self._delegate.next_order_id()


class _InventoryTrackingProductRepository:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._observed: dict[tuple[str, str], tuple[Any, int]] = {}

    async def find_by_id(self, product_id: str) -> Any:
        product = await self._delegate.find_by_id(product_id)
        if product is not None:
            for sku in product.skus:
                self._observed.setdefault(
                    (product.product_id, sku.sku_id),
                    (sku, sku.stock),
                )
        return product

    async def find_by_ids(self, product_ids: list[str]) -> list[Any]:
        return await self._delegate.find_by_ids(product_ids)

    async def list_all(self) -> list[Any]:
        return await self._delegate.list_all()

    def details(self) -> dict[str, Any]:
        deltas = [sku.stock - original for sku, original in self._observed.values()]
        return {
            "inventory_observed_skus": len(deltas),
            "inventory_net_delta": sum(deltas),
            "inventory_restored": bool(deltas) and all(delta == 0 for delta in deltas),
        }


class RubricFaultInjection(AbstractContextManager["RubricFaultInjection"]):
    """Apply one authored fault profile and restore it after the case."""

    def __init__(self, container: Container, profile: str) -> None:
        self._container = container
        self.profile = profile
        self._originals: list[tuple[Any, str, Any]] = []
        self._probes: dict[str, Any] = {}

    def _set(self, target: Any, field: str, value: Any) -> None:
        self._originals.append((target, field, getattr(target, field)))
        setattr(target, field, value)

    def __enter__(self) -> RubricFaultInjection:
        usecases: list[Any] = []
        seen_ids: set[int] = set()
        for usecase in self._container.catalog_searches.values():
            if id(usecase) in seen_ids:
                continue
            seen_ids.add(id(usecase))
            usecases.append(usecase)
        if self.profile == "fault-opensearch-unavailable":
            embedder = _StaticEmbedder()
            index = _StaticIndex([], fail=True)
            for usecase in usecases:
                self._set(usecase, "_embedder", embedder)
                self._set(usecase, "_vector_index", index)
            self._probes = {"embedder": embedder, "index": index}
        elif self.profile == "fault-repository-inconsistency":
            embedder = _StaticEmbedder()
            index = _StaticIndex(["P1008"])
            repository = _MissingProductRepository(
                self._container.product_repo,
                "P1008",
            )
            for usecase in usecases:
                self._set(usecase, "_embedder", embedder)
                self._set(usecase, "_vector_index", index)
                self._set(usecase, "_product_repo", repository)
            self._probes = {
                "embedder": embedder,
                "index": index,
                "repository": repository,
            }
        elif self.profile == "fault-embedding-timeout":
            embedder = _TimeoutEmbedder()
            for usecase in usecases:
                self._set(usecase, "_embedder", embedder)
            self._probes = {"embedder": embedder}
        elif self.profile == "fault-reranker-fallback":
            embedder = _StaticEmbedder()
            index = _StaticIndex(["P1008", "P1010"])
            reranker = _BrokenReranker()
            for usecase in usecases:
                self._set(usecase, "_embedder", embedder)
                self._set(usecase, "_vector_index", index)
                self._set(usecase, "_reranker", reranker)
            self._probes = {
                "embedder": embedder,
                "index": index,
                "reranker": reranker,
            }
        elif self.profile == "fault-order-write-failure":
            sessions = self._container.orchestrator._sessions  # noqa: SLF001
            main_factory = sessions._main_factory  # noqa: SLF001
            trade_factory = main_factory._trade_factory  # noqa: SLF001
            place_order = trade_factory._place_order  # noqa: SLF001
            repository = _FailingOrderRepository(place_order._order_repo)  # noqa: SLF001
            products = _InventoryTrackingProductRepository(  # noqa: SLF001
                place_order._product_repo,
            )
            self._set(place_order, "_order_repo", repository)
            self._set(place_order, "_product_repo", products)
            self._probes = {
                "order_repository": repository,
                "inventory": products,
            }
        else:
            raise ValueError(f"unsupported Rubric fault profile: {self.profile}")
        return self

    def details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "profile": self.profile,
            "injection_scope": "evaluation-container-only",
            "production_configuration_changed": False,
        }
        for name, probe in self._probes.items():
            if hasattr(probe, "calls"):
                details[f"{name}_calls"] = int(probe.calls)
            if hasattr(probe, "details"):
                details.update(probe.details())
        if self.profile == "fault-reranker-fallback":
            details.update(
                {
                    "expected_recall_strategy": "embedding_only",
                    "additional_recall_calls": 0,
                },
            )
        elif self.profile == "fault-repository-inconsistency":
            details["missing_product_id"] = "P1008"
        return details

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        for target, field, value in reversed(self._originals):
            setattr(target, field, value)
        self._originals.clear()


FAULT_PROFILES = frozenset(
    {
        "fault-opensearch-unavailable",
        "fault-repository-inconsistency",
        "fault-embedding-timeout",
        "fault-reranker-fallback",
        "fault-order-write-failure",
    },
)
