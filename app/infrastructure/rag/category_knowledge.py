# -*- coding: utf-8 -*-
"""category_knowledge

默认使用独立 OpenSearch 知识索引（ANN/BM25/RRF + Reranker），在线只读。
qdrant 配置保留 AgentScope KnowledgeBase v1，用于旧资料基线复现。
OpenSearch 建库和发布入口：scripts.index.build_knowledge_opensearch。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

from agentscope.credential import OpenAICredential
from agentscope.embedding import OpenAIEmbeddingModel
from agentscope.rag import ApproxTokenChunker, KnowledgeBase, QdrantStore, TextParser

from app.infrastructure.settings import PROJECT_ROOT, Settings
from app.infrastructure.embedding.openai_embedding_client import OpenAIEmbeddingClient
from app.infrastructure.rerank.http_reranker import HttpReranker
from app.infrastructure.rag.opensearch_knowledge import OpenSearchKnowledgeBase

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge" / "category-insight-v1"

_KB_DESCRIPTION = (
    "CrossShop 品类洞察知识库 v1：历史目录款型、属性分布、人民币价格区间，"
    "以及经批准的选购维度和避坑提示。"
)


def build_category_knowledge_base(settings: Settings) -> KnowledgeBase | OpenSearchKnowledgeBase:
    """构建品类知识库对象（不建库，建库见 bootstrap_category_knowledge）。"""
    if settings.category_kb_backend == "opensearch":
        embedding_settings = replace(
            settings, embedding_base_url=settings.product_embedding_base_url,
            embedding_api_key=settings.product_embedding_api_key,
            embedding_model=settings.product_embedding_model,
            embedding_dim=settings.product_embedding_dim,
        )
        if settings.product_embedding_dim != 1024 or settings.product_embedding_model != "BAAI/bge-m3":
            raise ValueError("Knowledge v2 requires the frozen BGE-M3 1024d contract")
        return OpenSearchKnowledgeBase(
            settings.opensearch_endpoint, OpenAIEmbeddingClient(embedding_settings),
            HttpReranker(settings, timeout_seconds=10.0) if settings.reranker_base_url else None,
            index=settings.category_knowledge_index, candidate_k=settings.category_candidate_k,
            timeout=settings.opensearch_timeout_seconds,
        )
    if settings.category_kb_backend != "qdrant":
        raise ValueError("CATEGORY_KB_BACKEND must be opensearch or qdrant")
    # CategoryInsight v1 was released and evaluated with the local BGE-M3
    # encoder (1024d). Querying that frozen Qdrant collection with the generic
    # EMBEDDING_* gateway can either fail or, worse, mix embedding spaces.
    # Reuse the configured BGE-M3 endpoint/model contract; the collection is
    # still category-only and remains entirely separate from product indexes.
    credential = OpenAICredential(
        api_key=settings.product_embedding_api_key,
        base_url=settings.product_embedding_base_url,
    )
    embedding_model = OpenAIEmbeddingModel(
        credential=credential,
        model=settings.product_embedding_model,
        dimensions=settings.product_embedding_dim,
        pass_dimensions=False,  # 兼容不接受 dimensions 入参的网关，维度仅用于建 collection
    )
    if settings.qdrant_url:
        vector_store = QdrantStore(url=settings.qdrant_url)
    else:
        local_path = settings.data_dir / "qdrant_kb"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store = QdrantStore(path=str(local_path))
    return KnowledgeBase(
        name="category_insight",
        description=_KB_DESCRIPTION,
        embedding_model=embedding_model,
        vector_store=vector_store,
        collection=settings.category_kb_collection,
    )


async def bootstrap_category_knowledge(
    knowledge_base: KnowledgeBase | OpenSearchKnowledgeBase,
    knowledge_dir: Optional[Path] = None,
) -> int:
    """把版本目录中的 Markdown 灌入知识库（幂等），返回新增文档数。"""
    if isinstance(knowledge_base, OpenSearchKnowledgeBase):
        try:
            await knowledge_base.ensure_ready()
        except Exception as err:  # read-only readiness; other Agent capabilities stay available
            logger.warning("知识库未就绪，请运行知识索引构建命令：%s", type(err).__name__)
        return 0
    directory = knowledge_dir or KNOWLEDGE_DIR
    try:
        await knowledge_base.ensure_collection()
        existing = {doc.document_id for doc in await knowledge_base.list_documents()}
        parser, chunker = TextParser(), ApproxTokenChunker(chunk_size=512, overlap=50)
        inserted = 0
        for md_file in sorted(directory.glob("*.md")):
            document_id = md_file.stem
            if document_id in existing:
                continue
            sections = await parser.parse(str(md_file), filename=md_file.name)
            chunks = await chunker.chunk(sections)
            await knowledge_base.insert_document(
                chunks=chunks,
                document_id=document_id,
                document_metadata={"source": md_file.name},
            )
            inserted += 1
        logger.info(
            "品类知识库就绪：新增 %d 篇，累计 %d 篇",
            inserted,
            len(existing) + inserted,
        )
        return inserted
    except Exception as err:  # noqa: BLE001 —— 知识库不可用不阻塞启动
        logger.warning("品类知识库建库失败，category_insight 将不可用：%s", err)
        return 0
