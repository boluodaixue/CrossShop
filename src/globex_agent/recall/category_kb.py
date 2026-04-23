"""OpenSearch BM25/KNN Hybrid retrieval for CategoryInsight cards."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from globex_agent.category_insight.models import CategoryCard

DEFAULT_CATEGORY_INDEX = "globex_category_kb"
DEFAULT_VECTOR_DIMENSION = 1024
DEFAULT_COARSE_K = 30
DEFAULT_ANALYZER = "ik_max_word"
DEFAULT_SEARCH_ANALYZER = "ik_smart"
_SAFE_INDEX_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


class CategoryQueryType(str, Enum):
    NOUN = "noun"
    ATTRIBUTE_CONSTRAINT = "attribute_constraint"
    STYLE = "style"
    COLLOQUIAL = "colloquial"


@dataclass(frozen=True, slots=True)
class HybridWeights:
    semantic: float
    lexical: float


WEIGHTS_BY_QUERY_TYPE = {
    CategoryQueryType.NOUN: HybridWeights(semantic=0.5, lexical=0.5),
    CategoryQueryType.ATTRIBUTE_CONSTRAINT: HybridWeights(
        semantic=0.7, lexical=0.3
    ),
    CategoryQueryType.STYLE: HybridWeights(semantic=0.9, lexical=0.1),
    CategoryQueryType.COLLOQUIAL: HybridWeights(semantic=1.0, lexical=0.0),
}


@dataclass(frozen=True, slots=True)
class CategoryCardHit:
    card: CategoryCard
    score: float
    rank: int
    retrieval_text: str | None = None
    retrieval_text_en: str | None = None
    retrieval_text_zh: str | None = None


@dataclass(frozen=True, slots=True)
class CategorySearchResult:
    hits: tuple[CategoryCardHit, ...]
    query_type: CategoryQueryType
    weights: HybridWeights
    used_bm25: bool


class OpenSearchTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any | None = None,
        params: dict[str, str] | None = None,
        content_type: str = "application/json",
    ) -> Any: ...


class OpenSearchRequestError(RuntimeError):
    def __init__(self, status: int | None, message: str) -> None:
        self.status = status
        super().__init__(message)


class OpenSearchHttpClient:
    """Small dependency-free JSON transport for the local teaching cluster."""

    def __init__(self, endpoint: str = "http://127.0.0.1:9200", *, timeout: float = 10) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any | None = None,
        params: dict[str, str] | None = None,
        content_type: str = "application/json",
    ) -> Any:
        suffix = f"?{urlencode(params)}" if params else ""
        url = f"{self._endpoint}/{path.lstrip('/')}{suffix}"
        if body is None:
            data = None
        elif content_type == "application/x-ndjson":
            data = str(body).encode("utf-8")
        else:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=data,
            method=method.upper(),
            headers={"Content-Type": content_type, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                payload = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenSearchRequestError(exc.code, detail) from exc
        except (URLError, TimeoutError) as exc:
            raise OpenSearchRequestError(None, str(exc)) from exc
        if not payload:
            return {}
        return json.loads(payload.decode("utf-8"))


_CARD_TYPE_LABELS = {
    "bestseller": "bestselling product forms",
    "attribute": "product attribute distribution",
    "price_range": "price tiers",
}
_CARD_TYPE_LABELS_ZH = {
    "bestseller": "热门商品形态",
    "attribute": "商品属性分布",
    "price_range": "价格档位",
}
_PRICE_SUMMARY_PATTERN = re.compile(
    r"便宜款\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*/\s*"
    r"中档\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*/\s*"
    r"高端\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)"
)


def category_retrieval_text(card: CategoryCard) -> str:
    """Build one English, context-complete text for recall and contextual reranking."""

    summary = _english_summary(card)
    knowledge_type = _CARD_TYPE_LABELS[card.card_type]
    return (
        f"Category: {card.category}. Knowledge type: {knowledge_type}. "
        f"Summary: {summary}"
    )


def category_retrieval_text_zh(card: CategoryCard) -> str:
    """Build a Chinese, context-complete retrieval text for course-compliant cards."""

    return (
        f"品类：{card.category}。"
        f"知识类型：{_CARD_TYPE_LABELS_ZH[card.card_type]}。"
        f"摘要：{card.summary}"
    )


def category_document_text(card: CategoryCard) -> str:
    """Backward-compatible name for the v2 embedding/BM25 retrieval text."""

    return category_retrieval_text(card)


def _english_summary(card: CategoryCard) -> str:
    clean = " ".join(card.summary.split()).replace("：", ":")
    if card.card_type == "bestseller":
        prefix = f"{card.category}:"
        if clean.casefold().startswith(prefix.casefold()):
            clean = clean[len(prefix) :].strip()
        return clean
    if card.card_type == "price_range":
        match = _PRICE_SUMMARY_PATTERN.fullmatch(card.summary.strip())
        if match:
            low_min, low_max, mid_min, mid_max, high_min, high_max = match.groups()
            return (
                f"Budget tier CNY {low_min}-{low_max}; "
                f"mid-range tier CNY {mid_min}-{mid_max}; "
                f"premium tier CNY {high_min}-{high_max}"
            )
    return clean


def build_category_index_mapping(
    dimension: int = DEFAULT_VECTOR_DIMENSION,
    *,
    analyzer: str = DEFAULT_ANALYZER,
    search_analyzer: str | None = DEFAULT_SEARCH_ANALYZER,
) -> dict[str, Any]:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    resolved_search_analyzer = search_analyzer or analyzer
    analyzed_text = {
        "type": "text",
        "analyzer": analyzer,
        "search_analyzer": resolved_search_analyzer,
    }
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "card_id": {"type": "keyword"},
                "category": analyzed_text,
                "card_type": {"type": "keyword"},
                "summary": analyzed_text,
                "retrieval_text": analyzed_text,
                "retrieval_text_en": analyzed_text,
                "retrieval_text_zh": analyzed_text,
                "raw_evidence": analyzed_text,
                "last_updated": {"type": "date"},
                "confidence": {"type": "float"},
                "content_vector": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "cosinesimil",
                        "parameters": {"m": 32, "ef_construction": 200},
                    },
                },
            },
        },
    }


def pipeline_name(query_type: CategoryQueryType) -> str:
    return f"globex-category-hybrid-{query_type.value}-v2"


def build_search_pipeline(weights: HybridWeights) -> dict[str, Any]:
    if weights.semantic < 0 or weights.lexical < 0:
        raise ValueError("weights must be non-negative")
    if weights.semantic + weights.lexical <= 0:
        raise ValueError("at least one weight must be positive")
    return {
        "description": "Category card KNN + BM25 min-max weighted fusion",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {
                            "weights": [weights.semantic, weights.lexical]
                        },
                    },
                }
            }
        ],
    }


def setup_category_index(
    client: OpenSearchTransport,
    cards: Sequence[CategoryCard],
    embeddings: NDArray[np.floating],
    *,
    index_name: str = DEFAULT_CATEGORY_INDEX,
    analyzer: str = DEFAULT_ANALYZER,
    search_analyzer: str | None = DEFAULT_SEARCH_ANALYZER,
    retrieval_texts: Sequence[str] | None = None,
    retrieval_texts_en: Sequence[str] | None = None,
    retrieval_texts_zh: Sequence[str] | None = None,
    recreate: bool = False,
) -> None:
    _validate_index_name(index_name)
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(cards):
        raise ValueError("embeddings must be a card-aligned two-dimensional matrix")
    if matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain finite positive-dimensional vectors")
    if len({card.card_id for card in cards}) != len(cards):
        raise ValueError("card_id must be unique")
    for name, texts in (
        ("retrieval_texts", retrieval_texts),
        ("retrieval_texts_en", retrieval_texts_en),
        ("retrieval_texts_zh", retrieval_texts_zh),
    ):
        if texts is not None and len(texts) != len(cards):
            raise ValueError(f"{name} must be card-aligned")
    resolved_texts = (
        [category_retrieval_text(card) for card in cards]
        if retrieval_texts is None
        else [" ".join(text.split()) for text in retrieval_texts]
    )
    if any(not text for text in resolved_texts):
        raise ValueError("retrieval_texts must not contain blank values")
    resolved_texts_en = _optional_clean_texts(
        retrieval_texts_en, len(cards), "retrieval_texts_en"
    )
    resolved_texts_zh = _optional_clean_texts(
        retrieval_texts_zh, len(cards), "retrieval_texts_zh"
    )

    exists = _index_exists(client, index_name)
    if exists and recreate:
        client.request("DELETE", index_name)
        exists = False
    if not exists:
        client.request(
            "PUT",
            index_name,
            body=build_category_index_mapping(
                matrix.shape[1],
                analyzer=analyzer,
                search_analyzer=search_analyzer,
            ),
        )

    for query_type, weights in WEIGHTS_BY_QUERY_TYPE.items():
        if weights.lexical == 0:
            continue
        client.request(
            "PUT",
            f"_search/pipeline/{pipeline_name(query_type)}",
            body=build_search_pipeline(weights),
        )

    lines: list[str] = []
    for position, (card, vector, retrieval_text) in enumerate(
        zip(cards, matrix, resolved_texts, strict=True)
    ):
        lines.append(
            json.dumps(
                {"index": {"_index": index_name, "_id": card.card_id}},
                separators=(",", ":"),
            )
        )
        source = card.model_dump(mode="json")
        source["retrieval_text"] = retrieval_text
        if resolved_texts_en is not None:
            source["retrieval_text_en"] = resolved_texts_en[position]
        if resolved_texts_zh is not None:
            source["retrieval_text_zh"] = resolved_texts_zh[position]
        source["content_vector"] = vector.astype(float).tolist()
        lines.append(json.dumps(source, ensure_ascii=False, separators=(",", ":")))
    response = client.request(
        "POST",
        "_bulk",
        params={"refresh": "true"},
        body="\n".join(lines) + "\n",
        content_type="application/x-ndjson",
    )
    if response.get("errors"):
        failures = [item for item in response.get("items", []) if item["index"].get("error")]
        raise OpenSearchRequestError(None, f"bulk indexing failed: {failures[:2]}")


class OpenSearchCategoryKnowledgeBase:
    def __init__(
        self,
        client: OpenSearchTransport,
        *,
        index_name: str = DEFAULT_CATEGORY_INDEX,
    ) -> None:
        _validate_index_name(index_name)
        self._client = client
        self._index_name = index_name

    def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        *,
        top_k: int = DEFAULT_COARSE_K,
        query_type: CategoryQueryType | str | None = None,
    ) -> CategorySearchResult:
        clean_query = " ".join(query.split())
        if not clean_query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        resolved_type = (
            classify_category_query(clean_query)
            if query_type is None
            else CategoryQueryType(query_type)
        )
        weights = WEIGHTS_BY_QUERY_TYPE[resolved_type]
        vector_query = {
            "knn": {
                "content_vector": {
                    "vector": [float(value) for value in query_embedding],
                    "k": top_k,
                }
            }
        }
        params: dict[str, str] | None = None
        if weights.lexical == 0:
            query_body: dict[str, Any] = vector_query
            used_bm25 = False
        else:
            query_body = {
                "hybrid": {
                    "queries": [
                        vector_query,
                        {
                            "multi_match": {
                                "query": clean_query,
                                "fields": ["category^2", "retrieval_text"],
                            }
                        },
                    ]
                }
            }
            params = {"search_pipeline": pipeline_name(resolved_type)}
            used_bm25 = True
        response = self._client.request(
            "POST",
            f"{self._index_name}/_search",
            params=params,
            body={
                "size": top_k,
                "query": query_body,
                "_source": {"excludes": ["content_vector"]},
            },
        )
        hits = tuple(
            _category_hit(hit, rank)
            for rank, hit in enumerate(response.get("hits", {}).get("hits", []), start=1)
        )
        return CategorySearchResult(hits, resolved_type, weights, used_bm25)

    def search_bm25(
        self, query: str, *, top_k: int = DEFAULT_COARSE_K
    ) -> tuple[CategoryCardHit, ...]:
        clean_query = " ".join(query.split())
        response = self._client.request(
            "POST",
            f"{self._index_name}/_search",
            body={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": clean_query,
                        "fields": ["category^2", "retrieval_text"],
                    }
                },
                "_source": {"excludes": ["content_vector"]},
            },
        )
        return tuple(
            _category_hit(hit, rank)
            for rank, hit in enumerate(response.get("hits", {}).get("hits", []), start=1)
        )


def _category_hit(hit: dict[str, Any], rank: int) -> CategoryCardHit:
    source = dict(hit["_source"])
    retrieval_text = source.pop("retrieval_text", None)
    retrieval_text_en = source.pop("retrieval_text_en", None)
    retrieval_text_zh = source.pop("retrieval_text_zh", None)
    return CategoryCardHit(
        card=CategoryCard.model_validate(source),
        score=float(hit.get("_score") or 0.0),
        rank=rank,
        retrieval_text=str(retrieval_text) if retrieval_text is not None else None,
        retrieval_text_en=(
            str(retrieval_text_en) if retrieval_text_en is not None else None
        ),
        retrieval_text_zh=(
            str(retrieval_text_zh) if retrieval_text_zh is not None else None
        ),
    )


def _optional_clean_texts(
    texts: Sequence[str] | None, expected_count: int, name: str
) -> list[str] | None:
    if texts is None:
        return None
    if len(texts) != expected_count:
        raise ValueError(f"{name} must be card-aligned")
    cleaned = [" ".join(text.split()) for text in texts]
    if any(not text for text in cleaned):
        raise ValueError(f"{name} must not contain blank values")
    return cleaned


def classify_category_query(query: str) -> CategoryQueryType:
    """Rule fallback when Planner has not supplied the course query type."""

    clean = " ".join(query.casefold().split())
    tokens = set(re.findall(r"[a-z0-9]+", clean))
    colloquial_starts = (
        "i ",
        "i'm ",
        "i am ",
        "help me ",
        "my ",
        "what ",
        "how ",
        "which ",
        "the kids ",
        "people ",
        "我",
        "就想",
        "帮我",
        "哪个",
        "哪款",
        "预算",
    )
    if clean.startswith(colloquial_starts) or len(tokens) >= 13:
        return CategoryQueryType.COLLOQUIAL
    style_tokens = {
        "aesthetic",
        "classic",
        "clean",
        "compact",
        "low-maintenance",
        "modern",
        "neat",
        "quiet",
        "reliable",
        "rugged",
        "secure",
        "simple",
        "slim",
        "stable",
        "subtle",
        "tidy",
        "witty",
        "实用",
        "简洁",
        "有质感",
        "适合",
        "美观",
        "复古",
        "现代",
    }
    if any(token in clean for token in style_tokens):
        return CategoryQueryType.STYLE
    attribute_tokens = {
        "adjustable",
        "black",
        "cordless",
        "electric",
        "flat-free",
        "leather",
        "manual",
        "metal",
        "outdoor",
        "plastic",
        "portable",
        "qwerty",
        "rfid",
        "steel",
        "without",
        "with",
        "带",
        "支持",
        "防水",
        "蓝牙",
        "有线",
        "无线",
        "材质",
        "续航",
        "接口",
        "想要",
    }
    if any(token in clean for token in attribute_tokens) or any(
        character.isdigit() for character in clean
    ):
        return CategoryQueryType.ATTRIBUTE_CONSTRAINT
    return CategoryQueryType.NOUN


def _index_exists(client: OpenSearchTransport, index_name: str) -> bool:
    try:
        client.request("GET", index_name)
    except OpenSearchRequestError as exc:
        if exc.status == 404:
            return False
        raise
    return True


def _validate_index_name(index_name: str) -> None:
    if not _SAFE_INDEX_NAME.fullmatch(index_name):
        raise ValueError("index_name must match ^[a-z][a-z0-9_-]*$")
