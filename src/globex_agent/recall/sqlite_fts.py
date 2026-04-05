"""Read-only SQLite FTS5 lexical backend for one catalog locale."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from globex_agent.domain import MarketLocale, Platform
from globex_agent.recall.base import RecallHit, RecallResult

_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_CJK_LOCALES = frozenset({MarketLocale.CN, MarketLocale.JP})


class SQLiteFtsSearchBackend:
    """Search one locale-specific FTS table in a platform catalog database."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        platform: Platform,
        locale: MarketLocale,
    ) -> None:
        self._path = Path(database_path)
        self._platform = platform
        self._locale = locale
        if not self._path.exists():
            raise FileNotFoundError(f"catalog database not found: {self._path}")

    @property
    def backend_id(self) -> str:
        return f"sqlite-fts5-v1:{self._platform.value}:{self._locale.value}"

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        top_k = max(1, top_k)
        expression = _fts_expression(query, trigram=self._locale in _CJK_LOCALES)
        if not expression:
            return RecallResult(self.backend_id, (), 0, False)
        table = f"items_fts_{self._locale.value}"
        with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                f"""
                SELECT item_id, -bm25({table}, 0.0, 6.0, 1.0) AS score
                FROM {table}
                WHERE {table} MATCH ?
                ORDER BY score DESC, item_id ASC
                LIMIT ?
                """,
                (expression, top_k),
            ).fetchall()
            total = int(
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE {table} MATCH ?",
                    (expression,),
                ).fetchone()[0]
            )
        hits = tuple(
            RecallHit(document_id=str(item_id), score=float(score), rank=rank)
            for rank, (item_id, score) in enumerate(rows, start=1)
        )
        return RecallResult(
            backend_id=self.backend_id,
            hits=hits,
            total_recall=total,
            truncated=total > top_k,
        )


def _fts_expression(query: str, *, trigram: bool) -> str:
    clean = " ".join(query.split())
    if trigram:
        compact = re.sub(r"\s+", "", clean)
        terms = [compact[index : index + 3] for index in range(max(0, len(compact) - 2))]
        terms = list(dict.fromkeys(terms))[:48]
    else:
        terms = _WORD_PATTERN.findall(clean.casefold())[:32]
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
