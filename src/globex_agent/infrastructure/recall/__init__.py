"""Replaceable retrieval backends used by ItemSearch and offline evaluation."""

from globex_agent.infrastructure.recall.base import (
    RecallHit,
    RecallResult,
    SearchBackend,
    SearchDocument,
)
from globex_agent.infrastructure.recall.canonical import (
    CanonicalDedupSearchBackend,
    CanonicalListing,
    ListingLanguage,
    PreferredLanguageSearchBackend,
)
from globex_agent.infrastructure.recall.category_kb import (
    DEFAULT_CATEGORY_INDEX,
    DEFAULT_COARSE_K,
    WEIGHTS_BY_QUERY_TYPE,
    CategoryCardHit,
    CategoryQueryType,
    CategorySearchResult,
    HybridWeights,
    OpenSearchCategoryKnowledgeBase,
    OpenSearchHttpClient,
    category_document_text,
    category_retrieval_text,
    category_retrieval_text_zh,
    classify_category_query,
    setup_category_index,
)
from globex_agent.infrastructure.recall.embedding import (
    DEFAULT_EMBEDDING_MAX_SEQ_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerTextEncoder,
    TextEncoder,
    clean_product_body,
    document_text,
    embedding_document_text,
    reranker_document_text,
)
from globex_agent.infrastructure.recall.fusion import (
    FusionWeights,
    LocalizedFusionSearchBackend,
    WeightedFusionSearchBackend,
)
from globex_agent.infrastructure.recall.index import (
    EmbeddingSearchBackend,
    ExactVectorIndex,
    FaissHNSWIndex,
    VectorIndex,
)
from globex_agent.infrastructure.recall.keyword import KeywordSearchBackend
from globex_agent.infrastructure.recall.persistence import index_manifest_compatible
from globex_agent.infrastructure.recall.reranker import (
    DEFAULT_RERANKER_MAX_LENGTH,
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    PairReranker,
    RawTextPairReranker,
    RerankedSearchBackend,
    SubprocessCrossEncoderReranker,
)
from globex_agent.infrastructure.recall.router import (
    CatalogPartition,
    PartitionedSearchBackendRouter,
)
from globex_agent.infrastructure.recall.search_document import (
    ITEM_TEXT_FORMAT_VERSION,
    standard_item_search_text,
    standard_item_to_search_document,
)
from globex_agent.infrastructure.recall.sqlite_fts import SQLiteFtsSearchBackend

__all__ = [
    "DEFAULT_EMBEDDING_MAX_SEQ_LENGTH",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_CATEGORY_INDEX",
    "DEFAULT_COARSE_K",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RERANKER_MAX_LENGTH",
    "CrossEncoderReranker",
    "CanonicalDedupSearchBackend",
    "CanonicalListing",
    "CatalogPartition",
    "CategoryCardHit",
    "CategoryQueryType",
    "CategorySearchResult",
    "EmbeddingSearchBackend",
    "ExactVectorIndex",
    "FaissHNSWIndex",
    "FusionWeights",
    "ITEM_TEXT_FORMAT_VERSION",
    "KeywordSearchBackend",
    "ListingLanguage",
    "LocalizedFusionSearchBackend",
    "HybridWeights",
    "OpenSearchCategoryKnowledgeBase",
    "OpenSearchHttpClient",
    "PairReranker",
    "PartitionedSearchBackendRouter",
    "PreferredLanguageSearchBackend",
    "RawTextPairReranker",
    "RecallHit",
    "RecallResult",
    "RerankedSearchBackend",
    "SearchBackend",
    "SearchDocument",
    "SentenceTransformerTextEncoder",
    "SQLiteFtsSearchBackend",
    "SubprocessCrossEncoderReranker",
    "TextEncoder",
    "VectorIndex",
    "WeightedFusionSearchBackend",
    "WEIGHTS_BY_QUERY_TYPE",
    "category_document_text",
    "category_retrieval_text",
    "category_retrieval_text_zh",
    "classify_category_query",
    "clean_product_body",
    "document_text",
    "embedding_document_text",
    "reranker_document_text",
    "setup_category_index",
    "standard_item_to_search_document",
    "standard_item_search_text",
    "index_manifest_compatible",
]
