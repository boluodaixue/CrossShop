"""Versioned knowledge-only ANN/BM25/RRF retrieval with explicit rerank status."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

from app.domain.catalog.ports.retrieval_ports import EmbeddingClient, Reranker
from app.infrastructure.tracing import trace_span

DIMENSION = 1024
PIPELINE = "crossshop-knowledge-rrf-v2"
INDEX_PREFIX = "crossshop-knowledge-v2-"
ALIAS = "crossshop-knowledge-current"
CONTRACT = "bge-m3-cls-l2-1024-max512-knowledge-char360-v2"


def load_documents(root: Path) -> list[dict[str, Any]]:
    """Load a corpus root with historical Markdown and optional official JSON."""
    legacy = root / "historical"
    if not legacy.exists():
        legacy = root / "knowledge" / "category-insight-v1"
    files = sorted(legacy.glob("*.md"))
    if len(files) not in {8, 64}:
        raise ValueError("Expected the frozen 64-document CategoryInsight v1 corpus")
    documents = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        documents.append({
            "document_id": path.stem, "title": text.splitlines()[0].lstrip("# "),
            "content": text, "source": path.name, "url": "",
            "publisher": "CrossShop public demo" if len(files) == 8 else "CrossShop CategoryInsight v1",
            "checked_on": "2026-08-28", "scope": "历史发布资料；具体使用边界见正文。",
            "section": "full card", "source_kind": "legacy_release",
        })
    source_path = root / "official" / "sources.json"
    if not source_path.exists():
        source_path = root / "shopping-guide-v2" / "sources.json"
    if not source_path.exists():
        return documents
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source = source.get("documents", source) if isinstance(source, dict) else source
    for item in source["documents"]:
        required = {"document_id", "title", "content", "url", "publisher",
                    "checked_on", "scope", "section"}
        if not required.issubset(item) or any(not item[key] for key in required):
            raise ValueError("Official knowledge requires text and source metadata")
        if not item["url"].startswith("https://"):
            raise ValueError("Official source must use HTTPS")
        documents.append({**item, "source": item["url"], "source_kind": "official_source_summary"})
    ids = [item["document_id"] for item in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate knowledge document IDs")
    return documents


def make_chunks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bounded text chunks; title/scope/source repeated rather than detached."""
    chunks = []
    for doc in documents:
        text = doc["content"]
        # Short cards stay intact. Long material is deterministically windowed.
        for start in range(0, len(text), 320):
            content = text[start:start + 360]
            if start and len(text) - start <= 40:
                break
            chunk_id = f"{doc['document_id']}:{start // 320}"
            chunks.append({**doc, "content": content, "chunk_id": chunk_id})
    return chunks


def corpus_hash(chunks: list[dict[str, Any]]) -> str:
    payload = json.dumps({"contract": CONTRACT, "chunks": chunks},
                         ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def searchable_text(doc: dict[str, Any]) -> str:
    return f"{doc['title']}\n{doc['content']}"


def checked_vector(vector: list[float]) -> list[float]:
    if len(vector) != DIMENSION or not all(math.isfinite(x) for x in vector):
        raise ValueError("Knowledge embedding must be finite and 1024 dimensional")
    norm = math.sqrt(sum(x * x for x in vector))
    if not 0.99 <= norm <= 1.01:
        raise ValueError("Knowledge embedding must be L2 normalized")
    return vector


def index_body(release: str, count: int) -> dict:
    keywords = ("document_id", "chunk_id", "source", "url", "publisher", "scope",
                "section", "source_kind")
    properties = {key: {"type": "keyword", "index": False} for key in keywords}
    properties["document_id"] = {"type": "keyword"}
    properties.update({
        "title": {"type": "text", "analyzer": "cjk"},
        "content": {"type": "text", "analyzer": "cjk"},
        "checked_on": {"type": "date", "format": "strict_date"},
        "content_vector": {"type": "knn_vector", "dimension": DIMENSION,
                           "method": {"name": "hnsw", "engine": "lucene",
                                      "space_type": "cosinesimil",
                                      "parameters": {"ef_construction": 100, "m": 16}}},
    })
    return {"settings": {"index": {"knn": True, "number_of_shards": 1,
                                    "number_of_replicas": 0}},
            "mappings": {"dynamic": "strict", "_meta": {"contract": CONTRACT,
                         "release": release, "chunk_count": count}, "properties": properties}}


def pipeline_body() -> dict:
    return {"phase_results_processors": [{"score-ranker-processor": {
        "combination": {"technique": "rrf", "rank_constant": 60}}}]}


def search_body(question: str, vector: list[float], count: int) -> dict:
    return {"size": count, "_source": {"excludes": ["content_vector"]},
            "query": {"hybrid": {"queries": [
                {"knn": {"content_vector": {"vector": checked_vector(vector), "k": count}}},
                {"multi_match": {"query": question, "fields": ["title^2", "content"]}},
            ]}}}


class OpenSearchKnowledgeBase:
    """Same tool-facing search shape as KnowledgeBase; no automatic index writes."""

    def __init__(self, endpoint: str, embedder: EmbeddingClient,
                 reranker: Reranker | None, *, index: str = ALIAS,
                 candidate_k: int = 20, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None):
        if not re.fullmatch(r"crossshop-knowledge-[a-z0-9-]+", index):
            raise ValueError("Knowledge index must use the crossshop-knowledge namespace")
        if not 5 <= candidate_k <= 100:
            raise ValueError("Knowledge candidate_k must be 5..100")
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.client = httpx.AsyncClient(base_url=endpoint.rstrip("/"),
                                       timeout=timeout, transport=transport)

    async def request(self, method: str, path: str, **kwargs) -> Any:
        response = await self.client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    async def ensure_ready(self) -> None:
        mapping = await self.request("GET", f"/{self.index}/_mapping")
        if len(mapping) != 1:
            raise RuntimeError("Knowledge alias must resolve to exactly one release")
        metadata = next(iter(mapping.values()))["mappings"]
        if metadata.get("_meta", {}).get("contract") != CONTRACT:
            raise RuntimeError("Knowledge encoding contract mismatch")
        if metadata["properties"]["content_vector"]["dimension"] != DIMENSION:
            raise RuntimeError("Knowledge vector dimension mismatch")

    async def search(self, *, queries: list[str], top_k: int = 3) -> list:
        if len(queries) != 1 or not queries[0].strip() or not 1 <= top_k <= 5:
            raise ValueError("Knowledge search requires one question and top_k=1..5")
        question = queries[0].strip()
        with trace_span("crossshop.knowledge.hybrid", {"crossshop.knowledge.index": self.index}):
            await self.ensure_ready()
            vector = await self.embedder.embed(question)
            response = await self.request("POST", f"/{self.index}/_search",
                                          params={"search_pipeline": PIPELINE},
                                          json=search_body(question, vector, self.candidate_k))
            hits = response["hits"]["hits"]
            if not hits:
                return []
            ranked = [(float(hit["_score"]), hit) for hit in hits]
            status = "unconfigured"
            if self.reranker is not None:
                try:
                    scores = await self.reranker.rerank(
                        question, [searchable_text(hit["_source"]) for hit in hits],
                    )
                    if len(scores) != len(hits) or not all(math.isfinite(x) for x in scores):
                        raise ValueError("Invalid knowledge reranker scores")
                    ranked = sorted(zip(scores, hits, strict=True), key=lambda row: -row[0])
                    status = "applied"
                except Exception:  # fail visibly to Hybrid ordering, never fake rerank
                    status = "unavailable"
            results = []
            seen = set()
            for score, hit in ranked:
                doc = hit["_source"]
                if doc["document_id"] in seen:
                    continue
                seen.add(doc["document_id"])
                metadata = {key: value for key, value in doc.items()
                            if key not in {"content", "content_vector"}}
                metadata.update({"rerank_status": status, "index": hit.get("_index", self.index),
                                 "score_kind": "reranker" if status == "applied" else "rrf"})
                results.append(SimpleNamespace(
                    document_id=doc["document_id"], score=score,
                    chunk=SimpleNamespace(content=doc["content"], metadata=metadata),
                ))
                if len(results) == top_k:
                    break
            return results

    async def close(self) -> None:
        await self.client.aclose()
