"""Build a content-addressed knowledge release; optionally switch only its alias."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.infrastructure.rag.category_knowledge import build_category_knowledge_base
from app.infrastructure.rag.opensearch_knowledge import (
    ALIAS,
    CONTRACT,
    INDEX_PREFIX,
    PIPELINE,
    OpenSearchKnowledgeBase,
    checked_vector,
    corpus_hash,
    index_body,
    load_documents,
    make_chunks,
    pipeline_body,
    searchable_text,
)
from app.infrastructure.settings import PROJECT_ROOT, load_settings


async def build(kb: OpenSearchKnowledgeBase, root: Path, *, publish: bool = False) -> dict:
    documents = load_documents(root)
    chunks = make_chunks(documents)
    release = corpus_hash(chunks)
    index = INDEX_PREFIX + release[:16]
    exists = await kb.client.head(f"/{index}")
    if exists.status_code == 404:
        await kb.request("PUT", f"/{index}", json=index_body(release, len(chunks)))
    else:
        exists.raise_for_status()
        mapping = await kb.request("GET", f"/{index}/_mapping")
        if mapping[index]["mappings"]["_meta"] != {
            "contract": CONTRACT, "release": release, "chunk_count": len(chunks),
        }:
            raise RuntimeError("Refusing to overwrite a different knowledge release")
    await kb.request("PUT", f"/_search/pipeline/{PIPELINE}", json=pipeline_body())
    # Deterministic IDs allow an interrupted build to be resumed, with no index deletion.
    for start in range(0, len(chunks), 8):
        batch = chunks[start:start + 8]
        vectors = await kb.embedder.embed_batch([searchable_text(doc) for doc in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding batch count mismatch")
        lines = []
        for doc, vector in zip(batch, vectors, strict=True):
            lines.extend([
                json.dumps({"index": {"_index": index, "_id": doc["chunk_id"]}}),
                json.dumps({**doc, "content_vector": checked_vector(vector)}, ensure_ascii=False),
            ])
        response = await kb.request("POST", "/_bulk",
                                    content=("\n".join(lines) + "\n").encode("utf-8"),
                                    headers={"Content-Type": "application/x-ndjson"})
        if response.get("errors"):
            raise RuntimeError("Knowledge bulk write failed; alias unchanged")
    await kb.request("POST", f"/{index}/_refresh")
    count = (await kb.request("GET", f"/{index}/_count"))["count"]
    if count != len(chunks):
        raise RuntimeError("Knowledge chunk count mismatch; alias unchanged")
    kb.index = index
    await kb.ensure_ready()
    probes = []
    for question, expected in (
        ("Apple 35W 双口同时充 Mac 和 iPhone 各有多少功率？", "guide-apple-35w"),
        ("DHL 清关时商品描述要写材质和用途吗？", "guide-dhl-customs-description"),
        ("FedEx 锂电池单独寄和装在设备内寄是一样的吗？", "guide-fedex-battery-form"),
    ):
        results = await kb.search(queries=[question], top_k=3)
        probes.append({"question": question, "expected": expected,
                       "document_ids": [item.document_id for item in results],
                       "passed": expected in {item.document_id for item in results},
                       "rerank_applied": bool(results) and all(
                           item.chunk.metadata["rerank_status"] == "applied" for item in results),
                       "sources_present": bool(results) and all(
                           item.chunk.metadata.get("source") for item in results)})
    qualified = all(row["passed"] and row["rerank_applied"] for row in probes)
    previous = []
    if publish:
        if not qualified:
            raise RuntimeError(f"Knowledge smoke not qualified; alias unchanged: {probes}")
        response = await kb.client.get(f"/_alias/{ALIAS}")
        if response.status_code != 404:
            response.raise_for_status()
            previous = list(response.json())
        if any(not name.startswith(INDEX_PREFIX) for name in previous):
            raise RuntimeError("Knowledge alias targets unexpected index; refusing switch")
        actions = [{"remove": {"index": name, "alias": ALIAS}} for name in previous]
        actions.append({"add": {"index": index, "alias": ALIAS}})
        await kb.request("POST", "/_aliases", json={"actions": actions})
    return {"schema_version": "knowledge-opensearch-release-v2", "release": release,
            "index": index, "alias": ALIAS, "published": publish, "previous_indexes": previous,
            "document_count": len(documents), "chunk_count": count, "contract": CONTRACT,
            "smoke_probes": probes, "smoke_qualified": qualified,
            "benchmark": False}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "artifacts/knowledge-v2/build-report.json")
    args = parser.parse_args()
    if args.validate_only:
        documents = load_documents(PROJECT_ROOT)
        chunks = make_chunks(documents)
        report = {"validated": True, "document_count": len(documents),
                  "chunk_count": len(chunks), "release": corpus_hash(chunks),
                  "online_executed": False}
    else:
        kb = build_category_knowledge_base(load_settings())
        if not isinstance(kb, OpenSearchKnowledgeBase):
            raise ValueError("Set CATEGORY_KB_BACKEND=opensearch for this builder")
        try:
            report = await build(kb, PROJECT_ROOT, publish=args.publish)
        finally:
            await kb.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
