import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from globex_agent.domain import (
    DataProvenance,
    MarketLocale,
    Platform,
    PriceSource,
    ProvenanceKind,
    ShoppingTask,
    StandardItem,
)
from globex_agent.eval.shopsimulator_retrieval import (
    ItemRelevanceAnnotation,
    QueryRelevanceAnnotation,
    RetrievalPoolCandidate,
    RetrievalPoolCase,
    annotation_prompt,
    pool_fingerprint,
    query_needs_review,
    select_retrieval_tasks,
    validate_annotation_for_case,
)


def _item(item_id: str, category: str, attributes: list[str]) -> StandardItem:
    return StandardItem(
        item_id=item_id,
        same_group_id=item_id,
        platform=Platform.TAOBAO,
        locale=MarketLocale.CN,
        language="zh",
        title=f"{category}{item_id}",
        category_path=["根", category],
        attributes={"source_attributes": attributes},
        price_source=PriceSource.UNAVAILABLE,
        is_available=True,
        ingested_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        provenance=DataProvenance(
            kind=ProvenanceKind.EXTERNAL_PUBLIC,
            source="test",
            source_record_id=item_id,
            generated_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
        ),
    )


def _task(item_id: str, index: int) -> ShoppingTask:
    return ShoppingTask(
        task_id=f"task:{index}",
        query_id=f"source-query:{index}",
        query=f"找商品 {index}",
        language="zh",
        platform=Platform.TAOBAO,
        locale=MarketLocale.CN,
        split="test",
        source_split="eval",
        target_item_id=item_id,
    )


def test_selection_is_category_diverse_and_uses_new_query_ids() -> None:
    items = {
        item.item_id: item
        for item in (
            _item("taobao:cn:1", "枕头", ["天然", "儿童"]),
            _item("taobao:cn:2", "枕头", ["天然"]),
            _item("taobao:cn:3", "灯具", ["防水", "户外"]),
            _item("taobao:cn:4", "灯具", ["防水"]),
            _item("taobao:cn:5", "杯子", ["陶瓷"]),
            _item("taobao:cn:6", "杯子", ["陶瓷", "带盖"]),
        )
    }
    tasks = [_task(item_id, index) for index, item_id in enumerate(items, start=1)]

    selected = select_retrieval_tasks(
        tasks,
        items,
        dev_count=1,
        test_count=2,
        min_two_attribute_peers=0,
    )

    assert len(selected) == 3
    assert {tuple(row.category_path) for row in selected} == {
        ("根", "枕头"),
        ("根", "灯具"),
        ("根", "杯子"),
    }
    assert all(row.query_id.startswith("shopsim-retrieval:v1:") for row in selected)


def test_prompt_hides_source_target_and_pool_sources() -> None:
    case = RetrievalPoolCase(
        query_id="q1",
        source_task_id="task1",
        source_target_item_id="taobao:cn:1",
        query="天然儿童枕",
        split="test",
        category_path=["家居", "枕头"],
        candidates=[
            RetrievalPoolCandidate(
                item_id=f"taobao:cn:{index}",
                title=f"乳胶枕 {index}",
                category_path=["家居", "枕头"],
                pool_sources=["source_target" if index == 1 else "semantic"],
            )
            for index in (1, 2)
        ],
    )

    prompt = annotation_prompt(case)

    assert "source_target" not in prompt
    assert "semantic" not in prompt
    assert "taobao:cn:1" in prompt
    assert '"grades"' in prompt


def test_annotation_gate_requires_two_positives_without_target_bias() -> None:
    case = RetrievalPoolCase(
        query_id="q1",
        source_task_id="task1",
        source_target_item_id="taobao:cn:1",
        query="天然儿童枕",
        split="test",
        category_path=["家居", "枕头"],
        candidates=[
            RetrievalPoolCandidate(
                item_id=f"taobao:cn:{index}",
                title=f"乳胶枕 {index}",
                category_path=["家居", "枕头"],
                pool_sources=["semantic"],
            )
            for index in (1, 2)
        ],
    )
    annotation = QueryRelevanceAnnotation(
        query_id="q1",
        pool_fingerprint=pool_fingerprint(case),
        annotator_model="cheap-model",
        annotation_policy="llm_silver_v1",
        review_status="machine_initial",
        items=[
            ItemRelevanceAnnotation(
                item_id="taobao:cn:1", label="Exact", gain=3, reason="满足条件"
            ),
            ItemRelevanceAnnotation(
                item_id="taobao:cn:2", label="Irrelevant", gain=0, reason="不满足"
            ),
        ],
    )

    validate_annotation_for_case(annotation, case)
    assert query_needs_review(annotation, case)

    valid_without_positive_target = annotation.model_copy(
        update={
            "items": [
                ItemRelevanceAnnotation(
                    item_id="taobao:cn:1",
                    label="Partial",
                    gain=1,
                ),
                ItemRelevanceAnnotation(
                    item_id="taobao:cn:2",
                    label="Exact",
                    gain=3,
                ),
                ItemRelevanceAnnotation(
                    item_id="taobao:cn:3",
                    label="Substitute",
                    gain=2,
                ),
            ]
        }
    )
    extended_case = case.model_copy(
        update={
            "candidates": [
                *case.candidates,
                RetrievalPoolCandidate(
                    item_id="taobao:cn:3",
                    title="乳胶枕 3",
                    category_path=["家居", "枕头"],
                    pool_sources=["fts"],
                ),
            ]
        }
    )
    valid_without_positive_target = valid_without_positive_target.model_copy(
        update={"pool_fingerprint": pool_fingerprint(extended_case)}
    )
    validate_annotation_for_case(valid_without_positive_target, extended_case)
    assert not query_needs_review(valid_without_positive_target, extended_case)

    invalid = annotation.model_copy(update={"items": annotation.items[:1]})
    with pytest.raises(ValueError, match="annotation candidates differ"):
        validate_annotation_for_case(invalid, case)


def test_pool_builder_uses_the_same_sorted_document_order_as_faiss() -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_path = (
        project_root
        / "scripts"
        / "data"
        / "prepare_shopsimulator_retrieval_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("shopsim_retrieval_builder", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    documents = module._load_index_ordered_documents(
        project_root / "data" / "demo" / "products.jsonl"
    )
    document_ids = [document.document_id for document in documents]

    assert document_ids == sorted(document_ids)
