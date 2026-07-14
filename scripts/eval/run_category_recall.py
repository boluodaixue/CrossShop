# -*- coding: utf-8 -*-
"""品类知识库（CategoryInsight）召回评测 —— 见教程 13-1 §5。

与商品检索评测共用 `scripts/eval/metrics.py` 的三个指标，区别只在**标注单位**：

    商品检索      标注单位 = product_id
    品类知识库    标注单位 = 知识文档名（如 kc-travel-gear-selection-guide-01.md）

标注单位为什么是文档名而不是 chunk id：`bootstrap_category_knowledge` 用
`ApproxTokenChunker(chunk_size=512, overlap=50)` 切片，chunk 边界会随 chunk_size
或文档内容变动而漂移，拿 chunk id 当标注会导致「改了一句话就要重标一遍」。
文档名稳定，且「该问题该由哪篇文档回答」本身就是运营能稳定判断的粒度。

用法（项目根目录执行，需 embedding 凭据 + Qdrant）：

    uv run python scripts/eval/run_category_recall.py
    uv run python scripts/eval/run_category_recall.py --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.infrastructure.rag.category_knowledge import (  # noqa: E402
    bootstrap_category_knowledge,
    build_category_knowledge_base,
)
from app.infrastructure.settings import PROJECT_ROOT, load_settings  # noqa: E402
from scripts.eval.metrics import (  # noqa: E402
    Aggregate,
    QueryResult,
    Thresholds,
    evaluate,
    gate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

_DATASET = (
    PROJECT_ROOT
    / "data"
    / "category_insight"
    / "releases"
    / "category-insight-v1"
    / "retrieval_eval_cases.jsonl"
)
_RELEASE_DIR = (
    PROJECT_ROOT / "data" / "category_insight" / "releases" / "category-insight-v1"
)


@dataclass(frozen=True)
class RecallRunMetadata:
    dataset_path: str
    embedding_model: str
    collection: str
    release_manifest_sha256: str
    approved_cards_sha256: str
    expected_document_count: int
    actual_document_count: int
    inserted_document_count: int


def load_dataset(path: Path) -> list[dict]:
    return [
        json.loads(raw)
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_document_sources(cases: list[dict]) -> set[str]:
    expected: set[str] = set()
    for case in cases:
        relevant = case.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError("each dataset case must contain non-empty relevant list")
        if any(not isinstance(source, str) or not source for source in relevant):
            raise ValueError("dataset relevant values must be non-empty strings")
        expected.update(relevant)
    if not expected:
        raise ValueError("dataset relevant document set is empty")
    return expected


def validate_document_collection(
    expected_sources: set[str],
    documents: list[object],
) -> set[str]:
    actual_sources = {str(getattr(document, "source", "")) for document in documents}
    if "" in actual_sources or len(actual_sources) != len(documents):
        raise RuntimeError(
            "knowledge collection has blank or duplicate document sources"
        )
    if actual_sources != expected_sources:
        missing = sorted(expected_sources - actual_sources)
        unexpected = sorted(actual_sources - expected_sources)
        raise RuntimeError(
            "knowledge collection does not exactly match dataset relevant documents: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return actual_sources


def metadata_console_line(metadata: RecallRunMetadata) -> str:
    return (
        "验收元数据："
        f"dataset={metadata.dataset_path} "
        f"embedding_model={metadata.embedding_model} "
        f"collection={metadata.collection} "
        f"manifest_sha256={metadata.release_manifest_sha256} "
        f"approved_cards_sha256={metadata.approved_cards_sha256} "
        f"expected_documents={metadata.expected_document_count} "
        f"actual_documents={metadata.actual_document_count} "
        f"inserted={metadata.inserted_document_count}"
    )


def source_of(item) -> str:
    """取一条检索结果所属的知识文档名。

    与 `category_insight_tool` 保持同一口径：metadata.source 缺失时退回 document_id，
    否则评测口径和线上口径会不一致。
    """
    metadata = getattr(item.chunk, "metadata", None)
    if metadata:
        return metadata.get("source", item.document_id)
    return item.document_id


async def run_dataset(knowledge_base, cases: list[dict], top_k: int) -> Aggregate:
    results: list[QueryResult] = []
    for case in cases:
        hits = await knowledge_base.search(queries=[case["query"]], top_k=top_k)
        # 同一篇文档可能命中多个 chunk：按首次出现保序去重，落到文档粒度
        retrieved: list[str] = []
        for item in hits:
            src = source_of(item)
            if src not in retrieved:
                retrieved.append(src)

        relevant = case["relevant"]
        results.append(
            QueryResult(
                query=case["query"],
                retrieved=retrieved,
                relevant=relevant,
                recall=recall_at_k(retrieved, relevant, top_k),
                mrr=mrr(retrieved, relevant),
                ndcg=ndcg_at_k(retrieved, relevant, top_k),
                kind=case.get("kind", "knowledge"),
            ),
        )
    return evaluate(results, k=top_k)


def render_report(
    agg: Aggregate,
    thresholds: Thresholds,
    metadata: RecallRunMetadata,
) -> str:
    verdict, reasons = gate(agg, thresholds)
    lines = [
        f"# 品类知识库召回评测报告（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）",
        "",
        f"标注集 `{metadata.dataset_path}`，K={agg.k}，共 {agg.count} 条。标注单位为知识文档名。",
        "",
        "| 运行元数据 | 值 |",
        "|---|---|",
        f"| Embedding model | `{metadata.embedding_model}` |",
        f"| Qdrant collection | `{metadata.collection}` |",
        f"| Release manifest SHA-256 | `{metadata.release_manifest_sha256}` |",
        f"| Approved cards SHA-256 | `{metadata.approved_cards_sha256}` |",
        f"| Expected documents | {metadata.expected_document_count} |",
        f"| Actual documents | {metadata.actual_document_count} |",
        f"| Inserted this run | {metadata.inserted_document_count} |",
        "",
        f"| 指标 | 值 | 阈值 |",
        "|---|---|---|",
        f"| Recall@{agg.k} | {agg.recall:.3f} | ≥ {thresholds.recall}（阻断） |",
        f"| MRR | {agg.mrr:.3f} | ≥ {thresholds.mrr}（阻断） |",
        f"| NDCG@{agg.k} | {agg.ndcg:.3f} | ≥ {thresholds.ndcg}（告警） |",
        "",
        f"门禁结论：**{verdict}**",
        "",
    ]
    if reasons:
        lines += ["未达标项：", *[f"- {r}" for r in reasons], ""]
    lines += [
        "| query | Recall | MRR | NDCG | 召回文档 | 标注文档 |",
        "|---|---|---|---|---|---|",
    ]
    for r in agg.per_query:
        lines.append(
            f"| {r.query} | {r.recall:.2f} | {r.mrr:.2f} | {r.ndcg:.2f} | "
            f"{','.join(r.retrieved) or '（空）'} | {','.join(r.relevant)} |",
        )
    return "\n".join(lines) + "\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description="品类知识库召回评测")
    parser.add_argument("--dataset", default=str(_DATASET))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.75)
    parser.add_argument("--min-mrr", type=float, default=0.65)
    parser.add_argument("--min-ndcg", type=float, default=0.70)
    parser.add_argument("--report-dir", default="eval")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    cases = load_dataset(dataset_path)
    expected_sources = expected_document_sources(cases)
    print(f"标注集 {dataset_path}：{len(cases)} 条，K={args.top_k}")

    # This frozen 64-card evaluator remains the Qdrant v1 baseline, not v2 quality.
    settings = replace(load_settings(), category_kb_backend="qdrant")
    knowledge_base = build_category_knowledge_base(settings)
    inserted = await bootstrap_category_knowledge(knowledge_base)
    documents = await knowledge_base.list_documents()
    metadata = RecallRunMetadata(
        dataset_path=str(dataset_path),
        embedding_model=settings.embedding_model,
        collection=settings.category_kb_collection,
        release_manifest_sha256=_sha256(_RELEASE_DIR / "manifest.json"),
        approved_cards_sha256=_sha256(_RELEASE_DIR / "approved_cards.jsonl"),
        expected_document_count=len(expected_sources),
        actual_document_count=len(documents),
        inserted_document_count=inserted,
    )
    print(metadata_console_line(metadata))
    validate_document_collection(expected_sources, documents)
    print("知识库文档集合与标注集完全一致")

    agg = await run_dataset(knowledge_base, cases, args.top_k)
    thresholds = Thresholds(
        recall=args.min_recall, mrr=args.min_mrr, ndcg=args.min_ndcg
    )
    print(
        f"  Recall@{agg.k}={agg.recall:.3f}  MRR={agg.mrr:.3f}  NDCG@{agg.k}={agg.ndcg:.3f}",
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        report_dir
        / f"category-recall-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    report_path.write_text(render_report(agg, thresholds, metadata), encoding="utf-8")
    print(f"报告已写入 {report_path}")

    verdict, reasons = gate(agg, thresholds)
    print(f"门禁：{verdict}" + (f"（{'；'.join(reasons)}）" if reasons else ""))
    if verdict == "BLOCK":
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
