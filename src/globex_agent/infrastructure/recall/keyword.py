"""Small dependency-free BM25 baseline for Chinese and ASCII product text."""

from __future__ import annotations

import math
import re
from collections import Counter
from unicodedata import normalize

from globex_agent.infrastructure.recall.base import RecallHit, RecallResult, SearchDocument

_ASCII_TOKEN = re.compile(r"[a-z0-9%]+")
_CJK_SEGMENT = re.compile(r"[\u3400-\u9fff]+")


class KeywordSearchBackend:
    """Rank an immutable document collection with a reproducible BM25 variant.

    Title terms receive a weight of three. The implementation intentionally has
    no model, network, tokenizer package, or user-profile input, which makes it a
    stable baseline for the Query/Item dual-tower work in the next stage.
    """

    backend_id = "keyword-bm25-v1"

    def __init__(
        self,
        documents: list[SearchDocument] | tuple[SearchDocument, ...],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: int = 3,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if title_weight < 1:
            raise ValueError("title_weight must be at least 1")

        ordered = tuple(sorted(documents, key=lambda document: document.document_id))
        if len({document.document_id for document in ordered}) != len(ordered):
            raise ValueError("document_id must be unique")

        self._documents = ordered
        self._k1 = k1
        self._b = b
        self._term_frequencies = tuple(
            _weighted_term_frequency(document, title_weight) for document in ordered
        )
        self._document_lengths = tuple(
            sum(term_frequency.values()) for term_frequency in self._term_frequencies
        )
        self._average_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequency = _document_frequency(self._term_frequencies)

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        top_k = max(1, min(top_k, len(self._documents)))
        query_terms = tokenize(query)
        scored = [
            (self._score(query_terms, index), document.document_id)
            for index, document in enumerate(self._documents)
        ]
        matched = [(score, document_id) for score, document_id in scored if score > 0]
        matched.sort(key=lambda entry: (-entry[0], entry[1]))
        hits = tuple(
            RecallHit(document_id=document_id, score=round(score, 8), rank=rank)
            for rank, (score, document_id) in enumerate(matched[:top_k], start=1)
        )
        return RecallResult(
            backend_id=self.backend_id,
            hits=hits,
            total_recall=len(matched),
            truncated=len(matched) > top_k,
        )

    def _score(self, query_terms: tuple[str, ...], document_index: int) -> float:
        if not query_terms or self._average_length == 0:
            return 0.0

        term_frequency = self._term_frequencies[document_index]
        document_length = self._document_lengths[document_index]
        score = 0.0
        for term in set(query_terms):
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequency[term]
            inverse_document_frequency = math.log(
                1
                + (len(self._documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + self._k1 * (
                1 - self._b + self._b * document_length / self._average_length
            )
            score += inverse_document_frequency * frequency * (self._k1 + 1) / denominator
        return score


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize ASCII words and Chinese character bigrams deterministically."""

    normalized = normalize("NFKC", text).casefold()
    terms = list(_ASCII_TOKEN.findall(normalized))
    for segment in _CJK_SEGMENT.findall(normalized):
        if len(segment) == 1:
            terms.append(segment)
        else:
            terms.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tuple(terms)


def _weighted_term_frequency(
    document: SearchDocument,
    title_weight: int,
) -> Counter[str]:
    frequencies = Counter(tokenize(document.body))
    title_frequencies = Counter(tokenize(document.title))
    for term, frequency in title_frequencies.items():
        frequencies[term] += frequency * title_weight
    return frequencies


def _document_frequency(
    term_frequencies: tuple[Counter[str], ...],
) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for term_frequency in term_frequencies:
        frequencies.update(term_frequency.keys())
    return frequencies
