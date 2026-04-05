"""Canonical-aware product recall and post-recall listing presentation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from globex_agent.recall.base import RecallHit, RecallResult, SearchBackend

ListingLanguage = Literal["en", "es", "ja", "zh"]


@dataclass(frozen=True, slots=True)
class CanonicalListing:
    """Non-search metadata linking one listing to a canonical product."""

    document_id: str
    canonical_id: str
    language: ListingLanguage | None = None


class CanonicalDedupSearchBackend:
    """Overfetch a ranked backend and keep one relevance hit per canonical ID."""

    def __init__(
        self,
        base_backend: SearchBackend,
        listings: Sequence[CanonicalListing],
        *,
        overfetch_factor: int = 2,
    ) -> None:
        if overfetch_factor < 1:
            raise ValueError("overfetch_factor must be at least 1")
        listing_by_document = {listing.document_id: listing for listing in listings}
        if len(listing_by_document) != len(listings):
            raise ValueError("canonical listing document_id must be unique")
        if any(not listing.canonical_id.strip() for listing in listings):
            raise ValueError("canonical_id must not be empty")
        self._base = base_backend
        self._listing_by_document = listing_by_document
        self._canonical_count = len(
            {listing.canonical_id for listing in listing_by_document.values()}
        )
        self._overfetch_factor = overfetch_factor

    @property
    def backend_id(self) -> str:
        return (
            f"canonical-dedup:overfetch={self._overfetch_factor}:"
            f"{self._base.backend_id}"
        )

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        top_k = max(1, top_k)
        requested = max(top_k, top_k * self._overfetch_factor)
        previous_hit_count = -1

        while True:
            coarse = self._base.search(query, top_k=requested)
            canonical_hits = self._deduplicate(coarse.hits)
            if len(canonical_hits) >= top_k or not coarse.truncated:
                break
            if len(coarse.hits) <= previous_hit_count:
                break
            previous_hit_count = len(coarse.hits)
            expanded = min(
                coarse.total_recall,
                max(requested + 1, requested * self._overfetch_factor),
            )
            if expanded <= requested:
                break
            requested = expanded

        selected = canonical_hits[:top_k]
        hits = tuple(
            RecallHit(
                document_id=hit.document_id,
                score=hit.score,
                rank=rank,
            )
            for rank, hit in enumerate(selected, start=1)
        )
        total_recall = (
            min(self._canonical_count, coarse.total_recall)
            if coarse.total_recall
            else 0
        )
        return RecallResult(
            backend_id=self.backend_id,
            hits=hits,
            total_recall=total_recall,
            truncated=total_recall > len(hits),
            fallback_reason=coarse.fallback_reason,
        )

    def _deduplicate(self, hits: Sequence[RecallHit]) -> tuple[RecallHit, ...]:
        seen: set[str] = set()
        unique: list[RecallHit] = []
        for hit in hits:
            listing = self._listing_by_document.get(hit.document_id)
            if listing is None or listing.canonical_id in seen:
                continue
            seen.add(listing.canonical_id)
            unique.append(hit)
        return tuple(unique)


class PreferredLanguageSearchBackend:
    """Choose a display listing after relevance ordering is complete."""

    def __init__(
        self,
        base_backend: SearchBackend,
        listings: Sequence[CanonicalListing],
        *,
        preferred_language: ListingLanguage | None,
    ) -> None:
        listing_by_document = {listing.document_id: listing for listing in listings}
        if len(listing_by_document) != len(listings):
            raise ValueError("canonical listing document_id must be unique")
        documents_by_canonical: dict[str, list[CanonicalListing]] = defaultdict(list)
        for listing in listings:
            documents_by_canonical[listing.canonical_id].append(listing)
        self._base = base_backend
        self._listing_by_document = listing_by_document
        self._documents_by_canonical = {
            canonical_id: tuple(
                sorted(group, key=lambda listing: listing.document_id)
            )
            for canonical_id, group in documents_by_canonical.items()
        }
        self._preferred_language = preferred_language

    @property
    def backend_id(self) -> str:
        language = self._preferred_language or "none"
        return f"display-language={language}:{self._base.backend_id}"

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        relevance = self._base.search(query, top_k=top_k)
        hits = tuple(
            RecallHit(
                document_id=self._display_document_id(hit.document_id),
                score=hit.score,
                rank=hit.rank,
            )
            for hit in relevance.hits
        )
        return RecallResult(
            backend_id=self.backend_id,
            hits=hits,
            total_recall=relevance.total_recall,
            truncated=relevance.truncated,
            fallback_reason=relevance.fallback_reason,
        )

    def _display_document_id(self, representative_id: str) -> str:
        if self._preferred_language is None:
            return representative_id
        representative = self._listing_by_document.get(representative_id)
        if representative is None:
            return representative_id
        if representative.language == self._preferred_language:
            return representative_id
        alternatives = self._documents_by_canonical.get(
            representative.canonical_id,
            (),
        )
        preferred = next(
            (
                listing
                for listing in alternatives
                if listing.language == self._preferred_language
            ),
            None,
        )
        return preferred.document_id if preferred is not None else representative_id
