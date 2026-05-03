"""Stable contracts shared by lexical and future dual-tower retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SearchDocument:
    """Minimal product representation needed by a retrieval backend."""

    document_id: str
    title: str
    body: str = ""


@dataclass(frozen=True, slots=True)
class RecallHit:
    """One ranked document returned by a backend."""

    document_id: str
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Backend result with enough metadata for ItemSearch and evaluation."""

    backend_id: str
    hits: tuple[RecallHit, ...]
    total_recall: int
    truncated: bool
    fallback_reason: str | None = None


class SearchBackend(Protocol):
    """Interface that keyword and Query/Item dual-tower backends both implement."""

    @property
    def backend_id(self) -> str:
        """Return the versioned backend identifier written to evaluation reports."""

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        """Return a deterministic ranking for one query."""
