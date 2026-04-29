"""Contracts and deterministic helpers for a pooled Taobao retrieval benchmark."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import Field, model_validator

from globex_agent.domain import ShoppingTask, StandardItem
from globex_agent.domain.catalog.models import NonEmptyText, StrictModel

DATASET_VERSION = "shopsimulator-retrieval-zh-v1"
LABEL_GAINS = {
    "Exact": 3.0,
    "Substitute": 2.0,
    "Partial": 1.0,
    "Irrelevant": 0.0,
}
POSITIVE_LABELS = frozenset({"Exact", "Substitute"})


class SelectedRetrievalTask(StrictModel):
    """One source task selected for the independent retrieval benchmark."""

    query_id: NonEmptyText
    source_task_id: NonEmptyText
    source_query_id: NonEmptyText
    source_target_item_id: NonEmptyText
    query: NonEmptyText
    query_source: Literal["simple_query", "full_query_fallback"]
    split: Literal["dev", "test"]
    category_path: list[NonEmptyText] = Field(min_length=1)
    peer_support_one_attribute: int = Field(ge=0)
    peer_support_two_attributes: int = Field(ge=0)


class RetrievalPoolCandidate(StrictModel):
    """Compact product evidence shown to a relevance annotator."""

    item_id: NonEmptyText
    title: NonEmptyText
    category_path: list[NonEmptyText] = Field(min_length=1)
    description: str = ""
    source_attributes: list[str] = Field(default_factory=list)
    variant_values: list[str] = Field(default_factory=list)
    price_summary: str | None = None
    pool_sources: list[NonEmptyText] = Field(min_length=1)
    source_ranks: dict[str, int] = Field(default_factory=dict)


class RetrievalPoolCase(StrictModel):
    """One query and its multi-system judgment pool."""

    dataset_version: Literal[DATASET_VERSION] = DATASET_VERSION
    query_id: NonEmptyText
    source_task_id: NonEmptyText
    source_target_item_id: NonEmptyText
    query: NonEmptyText
    split: Literal["dev", "test"]
    category_path: list[NonEmptyText] = Field(min_length=1)
    candidates: list[RetrievalPoolCandidate] = Field(min_length=2)

    @model_validator(mode="after")
    def candidate_ids_must_be_unique_and_include_source_target(
        self,
    ) -> RetrievalPoolCase:
        item_ids = [candidate.item_id for candidate in self.candidates]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("candidate item IDs must be unique")
        if self.source_target_item_id not in item_ids:
            raise ValueError("candidate pool must contain the source target")
        return self


class ItemRelevanceAnnotation(StrictModel):
    """One model or human relevance judgment with short visible evidence."""

    item_id: NonEmptyText
    label: Literal["Exact", "Substitute", "Partial", "Irrelevant"]
    gain: float = Field(ge=0, le=3)
    reason: str = ""

    @model_validator(mode="after")
    def gain_must_match_label(self) -> ItemRelevanceAnnotation:
        expected = LABEL_GAINS[self.label]
        if self.gain != expected:
            raise ValueError(f"gain for {self.label} must be {expected:g}")
        return self


class QueryRelevanceAnnotation(StrictModel):
    """All judgments for a query, kept separate from finalized qrels."""

    dataset_version: Literal[DATASET_VERSION] = DATASET_VERSION
    query_id: NonEmptyText
    pool_fingerprint: NonEmptyText
    annotator_model: NonEmptyText
    annotation_policy: Literal["llm_silver_v1", "human_reviewed_v1"]
    review_status: Literal["machine_initial", "needs_review", "human_reviewed"]
    items: list[ItemRelevanceAnnotation] = Field(min_length=2)

    @model_validator(mode="after")
    def item_ids_must_be_unique(self) -> QueryRelevanceAnnotation:
        item_ids = [item.item_id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("annotation item IDs must be unique")
        return self


def select_retrieval_tasks(
    tasks: Sequence[ShoppingTask],
    items_by_id: dict[str, StandardItem],
    *,
    dev_count: int = 30,
    test_count: int = 200,
    min_two_attribute_peers: int = 2,
) -> list[SelectedRetrievalTask]:
    """Select category-diverse tasks with evidence that alternatives exist.

    One representative is selected from as many eligible leaf-categories as the
    requested size permits. A stable category hash assigns dev categories; the
    remaining representatives and then additional high-support tasks fill test.
    """

    if dev_count < 1 or test_count < 1:
        raise ValueError("dev_count and test_count must both be positive")
    if min_two_attribute_peers < 0:
        raise ValueError("min_two_attribute_peers must be non-negative")
    eval_tasks = [task for task in tasks if task.split == "test"]
    if len(eval_tasks) < dev_count + test_count:
        raise ValueError("not enough test tasks for the requested benchmark")

    items_by_category: dict[tuple[str, ...], list[StandardItem]] = defaultdict(list)
    tasks_by_category: dict[tuple[str, ...], list[ShoppingTask]] = defaultdict(list)
    for item in items_by_id.values():
        items_by_category[tuple(item.category_path)].append(item)
    for task in eval_tasks:
        target = items_by_id.get(task.target_item_id)
        if target is None:
            raise ValueError(f"task references unknown target: {task.target_item_id}")
        tasks_by_category[tuple(target.category_path)].append(task)

    if len(tasks_by_category) < dev_count:
        raise ValueError("not enough represented categories for the requested dev split")

    support: dict[str, tuple[int, int]] = {}
    representatives: dict[tuple[str, ...], ShoppingTask] = {}
    for category, category_tasks in tasks_by_category.items():
        peers = items_by_category[category]
        for task in category_tasks:
            target = items_by_id[task.target_item_id]
            support[task.task_id] = _attribute_peer_support(target, peers)
        eligible_tasks = [
            task
            for task in category_tasks
            if support[task.task_id][1] >= min_two_attribute_peers
        ]
        if not eligible_tasks:
            continue
        representatives[category] = min(
            eligible_tasks,
            key=lambda task: (
                not bool(task.simple_query.strip()),
                -support[task.task_id][1],
                -support[task.task_id][0],
                _stable_key(task.task_id),
            ),
        )

    ordered_categories = sorted(
        representatives,
        key=lambda value: _stable_key("/".join(value)),
    )[: dev_count + test_count]
    dev_categories = set(ordered_categories[:dev_count])
    selected: list[tuple[ShoppingTask, Literal["dev", "test"]]] = []
    for category in ordered_categories:
        split: Literal["dev", "test"] = "dev" if category in dev_categories else "test"
        selected.append((representatives[category], split))

    current_test = sum(split == "test" for _, split in selected)
    if current_test > test_count:
        raise ValueError(
            "category coverage alone exceeds test_count; increase the requested test size"
        )
    selected_ids = {task.task_id for task, _ in selected}
    extra_test = sorted(
        (
            task
            for category, category_tasks in tasks_by_category.items()
            if category in representatives
            if category not in dev_categories
            for task in category_tasks
            if task.task_id not in selected_ids
            and support[task.task_id][1] >= min_two_attribute_peers
        ),
        key=lambda task: (
            not bool(task.simple_query.strip()),
            -support[task.task_id][1],
            -support[task.task_id][0],
            _stable_key(task.task_id),
        ),
    )
    selected.extend(
        (task, "test") for task in extra_test[: test_count - current_test]
    )
    if sum(split == "test" for _, split in selected) != test_count:
        raise ValueError("could not fill the requested test split")

    result = []
    for task, split in selected:
        target = items_by_id[task.target_item_id]
        one_attribute, two_attributes = support[task.task_id]
        result.append(
            SelectedRetrievalTask(
                query_id=f"shopsim-retrieval:v1:{task.target_item_id.rsplit(':', 1)[-1]}",
                source_task_id=task.task_id,
                source_query_id=task.query_id,
                source_target_item_id=task.target_item_id,
                query=_clean_query(task.simple_query or task.query),
                query_source=(
                    "simple_query" if task.simple_query.strip() else "full_query_fallback"
                ),
                split=split,
                category_path=target.category_path,
                peer_support_one_attribute=one_attribute,
                peer_support_two_attributes=two_attributes,
            )
        )
    return sorted(result, key=lambda row: (row.split, _stable_key(row.query_id)))


def annotation_prompt(case: RetrievalPoolCase) -> str:
    """Build the annotator prompt without revealing pool systems or source target."""

    candidates = []
    for candidate in case.candidates:
        candidates.append(
            {
                "item_id": candidate.item_id,
                "title": candidate.title,
                "category_path": candidate.category_path,
                "description": candidate.description,
                "attributes": candidate.source_attributes,
                "variants": candidate.variant_values,
                "price": candidate.price_summary,
            }
        )
    import json

    return (
        "你是中文电商普通商品检索的相关性标注员。只能依据给出的 Query 和商品可见字段判断，"
        "不得猜测缺失属性。逐件标注：Exact=3（类目正确且明确满足所有硬约束，软偏好基本满足）；"
        "Substitute=2（核心购买意图和硬约束可接受，只有软偏好或非关键规格差异）；"
        "Partial=1（同类或相关，但关键条件缺失、无法确认或有一项冲突）；"
        "Irrelevant=0（品类/用途错误或明确违反核心条件）。价格有多个规格时，只能在给出的规格"
        "价格能够支持 Query 时判定满足。候选已经固定顺序，返回严格 JSON："
        '{"grades":[3,0,2,...]}。grades 必须与候选顺序逐一对应，元素只能是 0/1/2/3，'
        f"数量必须恰好为 {len(candidates)}。不要返回 item_id、理由或 Markdown。\n\n"
        f"Query：{case.query}\n"
        f"候选商品：{json.dumps(candidates, ensure_ascii=False, separators=(',', ':'))}"
    )


def validate_annotation_for_case(
    annotation: QueryRelevanceAnnotation,
    case: RetrievalPoolCase,
) -> None:
    """Reject partial/mismatched model output before it can become qrels."""

    if annotation.query_id != case.query_id:
        raise ValueError("annotation query_id does not match pool case")
    if annotation.pool_fingerprint != pool_fingerprint(case):
        raise ValueError("annotation pool fingerprint does not match pool case")
    expected = {candidate.item_id for candidate in case.candidates}
    actual = {item.item_id for item in annotation.items}
    if actual != expected:
        missing = sorted(expected - actual)[:3]
        extra = sorted(actual - expected)[:3]
        raise ValueError(f"annotation candidates differ; missing={missing}, extra={extra}")


def query_needs_review(
    annotation: QueryRelevanceAnnotation,
    case: RetrievalPoolCase,
) -> bool:
    """Gate machine labels that are unsafe to publish as a retrieval case."""

    positive_count = sum(item.label in POSITIVE_LABELS for item in annotation.items)
    return positive_count < 2


def stable_audit_query_ids(query_ids: Iterable[str], ratio: float = 0.1) -> set[str]:
    """Select a deterministic audit sample without depending on file order."""

    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be between zero and one")
    ordered = sorted(set(query_ids), key=_stable_key)
    count = round(len(ordered) * ratio)
    return set(ordered[:count])


def pool_fingerprint(case: RetrievalPoolCase) -> str:
    """Fingerprint the visible annotation input so stale labels cannot be reused."""

    return hashlib.sha256(annotation_prompt(case).encode()).hexdigest()


def _attribute_peer_support(
    target: StandardItem,
    peers: Sequence[StandardItem],
) -> tuple[int, int]:
    target_attributes = _normalized_attributes(target)
    overlaps = [
        len(target_attributes & _normalized_attributes(peer))
        for peer in peers
        if peer.item_id != target.item_id
    ]
    return sum(value >= 1 for value in overlaps), sum(value >= 2 for value in overlaps)


def _normalized_attributes(item: StandardItem) -> set[str]:
    values = item.attributes.get("source_attributes", [])
    return {" ".join(str(value).casefold().split()) for value in values if str(value).strip()}


def _stable_key(value: str) -> str:
    return hashlib.sha256(f"{DATASET_VERSION}:{value}".encode()).hexdigest()


def _clean_query(value: str) -> str:
    return " ".join(value.replace("_x000D_", " ").split())
