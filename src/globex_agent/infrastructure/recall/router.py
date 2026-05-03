"""Platform/locale routing for independently built retrieval partitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from globex_agent.domain import MarketLocale, Platform
from globex_agent.infrastructure.recall.base import SearchBackend


@dataclass(frozen=True, slots=True)
class CatalogPartition:
    platform: Platform
    locale: MarketLocale

    @property
    def partition_id(self) -> str:
        return f"{self.platform.value}:{self.locale.value}"


class PartitionedSearchBackendRouter:
    """Resolve one backend without ever mixing platform catalog partitions."""

    def __init__(self, backends: Mapping[CatalogPartition, SearchBackend]) -> None:
        if not backends:
            raise ValueError("at least one search backend partition is required")
        self._backends = dict(backends)

    @property
    def partitions(self) -> tuple[CatalogPartition, ...]:
        return tuple(
            sorted(
                self._backends,
                key=lambda partition: (partition.platform.value, partition.locale.value),
            )
        )

    def backend_for(
        self,
        platform: Platform,
        locale: MarketLocale | None,
    ) -> SearchBackend:
        if locale is not None:
            partition = CatalogPartition(platform, locale)
            try:
                return self._backends[partition]
            except KeyError as exc:
                raise KeyError(
                    f"search partition is not configured: {partition.partition_id}"
                ) from exc

        matches = [
            backend
            for partition, backend in self._backends.items()
            if partition.platform is platform
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"search platform is not configured: {platform.value}")
        raise ValueError(f"locale is required for multi-locale platform: {platform.value}")
