from types import SimpleNamespace

import pytest

from scripts.eval.metrics import Aggregate, QueryResult, Thresholds
from scripts.eval.run_category_recall import (
    RecallRunMetadata,
    expected_document_sources,
    metadata_console_line,
    render_report,
    validate_document_collection,
)


def test_document_collection_must_exactly_match_dataset_relevant_sources() -> None:
    cases = [
        {"query": "q1", "relevant": ["a.md", "b.md"]},
        {"query": "q2", "relevant": ["b.md", "c.md"]},
    ]
    expected = expected_document_sources(cases)
    documents = [
        SimpleNamespace(source="a.md"),
        SimpleNamespace(source="b.md"),
        SimpleNamespace(source="c.md"),
    ]
    assert expected == {"a.md", "b.md", "c.md"}
    assert validate_document_collection(expected, documents) == expected

    with pytest.raises(
        RuntimeError, match=r"missing=\['c.md'\].*unexpected=\['x.md'\]"
    ):
        validate_document_collection(
            expected,
            [
                SimpleNamespace(source="a.md"),
                SimpleNamespace(source="b.md"),
                SimpleNamespace(source="x.md"),
            ],
        )
    with pytest.raises(RuntimeError, match="blank or duplicate"):
        validate_document_collection(
            expected,
            [
                SimpleNamespace(source="a.md"),
                SimpleNamespace(source="a.md"),
                SimpleNamespace(source="c.md"),
            ],
        )


def test_render_report_uses_actual_run_metadata() -> None:
    result = QueryResult(
        query="自定义问题",
        retrieved=["a.md"],
        relevant=["a.md"],
        recall=1.0,
        mrr=1.0,
        ndcg=1.0,
        kind="selection_guide",
    )
    aggregate = Aggregate(
        k=3,
        count=1,
        recall=1.0,
        mrr=1.0,
        ndcg=1.0,
        per_query=[result],
    )
    metadata = RecallRunMetadata(
        dataset_path=r"D:\custom\recall.jsonl",
        embedding_model="BAAI/bge-m3",
        collection="globex_category_kb_v1",
        release_manifest_sha256="a" * 64,
        approved_cards_sha256="b" * 64,
        expected_document_count=64,
        actual_document_count=64,
        inserted_document_count=0,
    )

    report = render_report(aggregate, Thresholds(), metadata)

    assert r"D:\custom\recall.jsonl" in report
    assert "BAAI/bge-m3" in report
    assert "globex_category_kb_v1" in report
    assert "a" * 64 in report
    assert "b" * 64 in report
    assert "| Expected documents | 64 |" in report
    assert "| Actual documents | 64 |" in report
    assert "| Inserted this run | 0 |" in report
    console = metadata_console_line(metadata)
    assert r"dataset=D:\custom\recall.jsonl" in console
    assert "embedding_model=BAAI/bge-m3" in console
    assert "collection=globex_category_kb_v1" in console
    assert "expected_documents=64 actual_documents=64 inserted=0" in console
