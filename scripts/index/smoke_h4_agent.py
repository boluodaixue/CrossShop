"""Reproducible H4 acceptance for one search tool and platform-scoped agents.

The real-infrastructure section builds the production composition root and invokes
the single ``product_search_tool`` against Reference, ReferenceSeed, Amazon, and Amazon JP.
Recording wrappers observe the already-wired BGE-M3 query encoder, OpenSearch index,
and HTTP reranker without substituting any of them.

The deterministic section uses fake UseCases and SearchAgents only to prove routing
semantics that must not depend on an LLM making the same choice on every run.  It is
reported separately and is never presented as model or retrieval evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import httpx
from agentscope.message import AssistantMsg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.application.agents.orchestrator import SubmitIntentInput
from app.application.tools.product_search_tool import build_product_search_tool
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.composition import Container, build_container
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus


@dataclass(frozen=True)
class SearchCase:
    name: str
    platform: str
    normalized_query: str
    locale: str
    target_currency: str
    ship_to: str | None = None
    price_max_major: float | None = None
    site_locale: str | None = None


REAL_CASES = (
    SearchCase(
        name="crossshop_reference",
        platform="crossshop_reference",
        normalized_query="降噪耳机",
        locale="zh-CN",
        ship_to="CN",
        price_max_major=30.0,
        target_currency="USD",
    ),
    SearchCase(
        name="reference_seed",
        platform="reference_seed",
        normalized_query="轻便旅行背包",
        locale="zh-CN",
        ship_to="CN",
        price_max_major=12.0,
        target_currency="USD",
    ),
    SearchCase(
        name="amazon",
        platform="amazon",
        normalized_query="noise cancelling headphones",
        locale="en-US",
        ship_to="CN",
        price_max_major=35.0,
        target_currency="USD",
    ),
    SearchCase(
        name="amazon-jp",
        platform="amazon",
        site_locale="jp",
        normalized_query="ノイズキャンセリング ヘッドホン",
        locale="ja-JP",
        target_currency="JPY",
    ),
)


def _chunk_json(chunk: Any) -> dict[str, Any]:
    text = chunk.content[0].text
    if text.startswith("[error]"):
        raise RuntimeError(text)
    body = json.loads(text)
    if not isinstance(body, dict):
        raise TypeError("product_search_tool did not return a JSON object")
    return body


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _validated_output_root(path: Path) -> Path:
    output_root = path.resolve()
    if output_root == PROJECT_ROOT or PROJECT_ROOT not in output_root.parents:
        raise ValueError(
            f"--output-root must be a directory inside the project root: {PROJECT_ROOT}"
        )
    return output_root


async def _health_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(url)
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise TypeError(f"health endpoint did not return an object: {url}")
    return body


class _RecordingEmbedder:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def embed(self, text: str) -> list[float]:
        started = time.perf_counter()
        vector = await self.delegate.embed(text)
        self.calls.append(
            {
                "text": text,
                "dimension": len(vector),
                "finite": all(math.isfinite(value) for value in vector),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return vector


class _RecordingIndex:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def search(
        self,
        *,
        query: str,
        embedding: list[float],
        top_n: int,
    ) -> list[Any]:
        started = time.perf_counter()
        hits = await self.delegate.search(
            query=query,
            embedding=embedding,
            top_n=top_n,
        )
        self.calls.append(
            {
                "query": query,
                "embedding_dimension": len(embedding),
                "top_n": top_n,
                "hit_product_ids": [hit.product_id for hit in hits],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return hits


class _RecordingReranker:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        started = time.perf_counter()
        scores = await self.delegate.rerank(query, documents)
        self.calls.append(
            {
                "query": query,
                "document_count": len(documents),
                "scores": scores,
                "finite": all(math.isfinite(score) for score in scores),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return scores


def _drain_events(queue: asyncio.Queue) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while not queue.empty():
        events.append(queue.get_nowait().to_dict())
    return events


async def _validate_cards(
    container: Container,
    case: SearchCase,
    result: dict[str, Any],
) -> dict[str, Any]:
    hits = result["hits"]
    product_ids = [hit["product_id"] for hit in hits]
    products = await container.product_repo.find_by_ids(product_ids)
    if [product.product_id for product in products] != product_ids:
        raise RuntimeError(f"{case.name}: Repository could not restore hit order")
    if case.site_locale == "jp" and any(
        not product_id.startswith("amazon:jp:") for product_id in product_ids
    ):
        raise RuntimeError("amazon-jp returned a Product outside locale=jp")

    sku_count = 0
    for card, product in zip(hits, products, strict=True):
        if case.ship_to and case.ship_to not in product.ships_to:
            raise RuntimeError(
                f"{case.name}: ineligible ship_to Product leaked: {product.product_id}"
            )
        source_skus = {sku.sku_id: sku for sku in product.skus}
        for card_sku in card["skus"]:
            sku_count += 1
            source = source_skus.get(card_sku["sku_id"])
            if source is None:
                raise RuntimeError(f"{case.name}: unknown SKU {card_sku['sku_id']}")
            if source.stock <= 0 or card_sku["stock"] <= 0:
                raise RuntimeError(f"{case.name}: out-of-stock SKU leaked")
            if (
                case.price_max_major is not None
                and float(card_sku["price_major"]) > case.price_max_major
            ):
                raise RuntimeError(f"{case.name}: over-budget SKU leaked")
    return {
        "restored_product_count": len(products),
        "validated_sku_count": sku_count,
        "product_ids": product_ids,
    }


async def run_real_infrastructure(
    container: Container,
    *,
    timeout_seconds: float,
    output_root: Path | None = None,
) -> dict[str, Any]:
    settings = container.settings
    opensearch = await _health_json(
        settings.opensearch_endpoint,
        timeout_seconds=timeout_seconds,
    )
    query_health = await _health_json(
        f"{settings.product_embedding_base_url.rstrip('/').removesuffix('/v1')}/health",
        timeout_seconds=timeout_seconds,
    )
    reranker_health = await _health_json(
        f"{settings.reranker_base_url.rstrip('/')}/health",
        timeout_seconds=timeout_seconds,
    )
    if opensearch.get("version", {}).get("number") != "2.19.1":
        raise RuntimeError("H4 acceptance requires OpenSearch 2.19.1")
    if query_health.get("dimension") != 1024 or query_health.get("device") != "cpu":
        raise RuntimeError(f"unexpected BGE-M3 Query health: {query_health!r}")
    if (
        reranker_health.get("model") != "BAAI/bge-reranker-v2-m3"
        or reranker_health.get("device") != "cuda:0"
    ):
        raise RuntimeError(f"unexpected reranker health: {reranker_health!r}")
    if container.product_repo.product_count != 45_286:
        raise RuntimeError("composition did not load the full JsonlProductRepository")
    if container.product_repo.sku_count != 261_369:
        raise RuntimeError("composition SKU count mismatch")

    usecases = container.catalog_searches
    if any(
        usecase._product_repo is not container.product_repo
        for usecase in usecases.values()
    ):
        raise RuntimeError(
            "platform UseCases do not share composition ProductRepository"
        )

    tool = build_product_search_tool(usecases, container.bus)
    report: dict[str, Any] = {
        "opensearch": opensearch,
        "query_encoder": query_health,
        "reranker": reranker_health,
        "repository": {
            "implementation": type(container.product_repo).__name__,
            "shared_by_all_usecases": True,
            "product_count": container.product_repo.product_count,
            "sku_count": container.product_repo.sku_count,
        },
        "cases": {},
    }
    for case in REAL_CASES:
        usecase_key = (
            f"amazon:{case.site_locale}" if case.site_locale else case.platform
        )
        usecase = usecases[usecase_key]
        original_embedder = usecase._embedder
        original_index = usecase._vector_index
        original_reranker = usecase._reranker
        embedder = _RecordingEmbedder(original_embedder)
        index = _RecordingIndex(original_index)
        reranker = _RecordingReranker(original_reranker)
        usecase._embedder = embedder
        usecase._vector_index = index
        usecase._reranker = reranker

        session_id = f"h4-real-{case.name}"
        event_queue = container.bus.subscribe(session_id)
        token = ShoppingContext.set(
            ShoppingContextSnapshot(
                shopping_session_id=session_id,
                buyer_id="h4-acceptance",
                locale=case.locale,
                currency=case.target_currency,
            )
        )
        started = time.perf_counter()
        try:
            chunk = await tool(
                normalized_query=case.normalized_query,
                platform=case.platform,
                site_locale=case.site_locale,
                ship_to=case.ship_to,
                locale=case.locale,
                top_k=5,
                price_max_major=case.price_max_major,
                target_currency=case.target_currency,
            )
            result = _chunk_json(chunk)
        finally:
            ShoppingContext.reset(token)
            usecase._embedder = original_embedder
            usecase._vector_index = original_index
            usecase._reranker = original_reranker
        elapsed_seconds = time.perf_counter() - started
        events = _drain_events(event_queue)
        container.bus.unsubscribe(session_id, event_queue)

        if len(embedder.calls) != 1 or embedder.calls[0]["dimension"] != 1024:
            raise RuntimeError(f"{case.name}: expected one 1024d Query encoding")
        if len(index.calls) != 1:
            raise RuntimeError(f"{case.name}: expected exactly one Hybrid request")
        if len(reranker.calls) != 1 or not reranker.calls[0]["finite"]:
            raise RuntimeError(f"{case.name}: expected one real finite rerank response")
        if result.get("rerank_applied") is not True:
            raise RuntimeError(f"{case.name}: rerank_applied is not true")
        card_validation = await _validate_cards(container, case, result)
        invokes = [event for event in events if event["type"] == "tool.invoke"]
        if len(invokes) != 1:
            raise RuntimeError(f"{case.name}: expected one tool invocation event")
        invoke_args = invokes[0]["payload"]["args"]
        if invoke_args["platform"] != case.platform:
            raise RuntimeError(f"{case.name}: platform event mismatch")
        if invoke_args["site_locale"] != case.site_locale:
            raise RuntimeError(f"{case.name}: site locale event mismatch")

        filtered_out = result.get("filtered_out", [])
        case_report = {
            "input": asdict(case),
            "usecase_key": usecase_key,
            "index_name": original_index.index_name,
            "index_site_locale": original_index.site_locale,
            "tool_invoke_count": len(invokes),
            "embedding": embedder.calls[0],
            "hybrid": index.calls[0],
            "reranker_call": reranker.calls[0],
            "rerank_applied": result["rerank_applied"],
            "hit_count": len(result["hits"]),
            "filtered_out_count": len(filtered_out),
            "filtered_out_reasons": [item["reason"] for item in filtered_out],
            "elapsed_seconds": elapsed_seconds,
            **card_validation,
        }
        report["cases"][case.name] = case_report
        if output_root is not None:
            _atomic_json(output_root / f"real-{case.name}-result.json", result)
            _atomic_json(output_root / f"real-{case.name}-evidence.json", case_report)
    return report


class _FakeUseCase:
    def __init__(self, label: str, hit_counts: list[int] | None = None) -> None:
        self.label = label
        self.hit_counts = list(hit_counts or [3])
        self.specs: list[ProductSearchSpec] = []

    async def execute(self, spec: ProductSearchSpec) -> dict[str, Any]:
        self.specs.append(spec)
        index = min(len(self.specs) - 1, len(self.hit_counts) - 1)
        count = self.hit_counts[index]
        return {
            "hits": [
                {"product_id": f"{self.label}-{position}"} for position in range(count)
            ],
            "filtered_out": [],
            "recall_strategy": "deterministic_fake",
        }


class _FakeWorker:
    def __init__(self, scope: tuple[str | None, str | None], delay: float) -> None:
        self.scope = scope
        self.delay = delay
        self.started = 0.0
        self.finished = 0.0

    async def reply(self, inputs: Any) -> AssistantMsg:
        del inputs
        self.started = time.perf_counter()
        await asyncio.sleep(self.delay)
        self.finished = time.perf_counter()
        return AssistantMsg("fake-search", '{"hits": []}')


class _FakeSearchFactory:
    def __init__(self, delay: float = 0.15) -> None:
        self.delay = delay
        self.scopes: list[tuple[str | None, str | None]] = []
        self.workers: list[_FakeWorker] = []

    def build(
        self,
        *,
        platform: str | None = None,
        site_locale: str | None = None,
    ) -> _FakeWorker:
        scope = (platform, site_locale)
        worker = _FakeWorker(scope, self.delay)
        self.scopes.append(scope)
        self.workers.append(worker)
        return worker


async def run_deterministic_protocol() -> dict[str, Any]:
    # A simple single-platform request is a direct call to the same public tool.
    bus = TradeEventBus()
    direct_queue = bus.subscribe("h4-direct-main")
    direct_usecase = _FakeUseCase("reference_seed")
    direct_tool = build_product_search_tool({"reference_seed": direct_usecase}, bus)  # type: ignore[arg-type]
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="h4-direct-main",
            buyer_id="acceptance",
            locale="zh-CN",
            currency="CNY",
        )
    )
    try:
        direct_result = _chunk_json(
            await direct_tool(normalized_query="旅行背包", platform="reference_seed")
        )
    finally:
        ShoppingContext.reset(token)
    direct_events = _drain_events(direct_queue)
    if any(event["type"] == "agent.dispatch" for event in direct_events):
        raise RuntimeError("simple direct search unexpectedly dispatched an Agent")
    if len(direct_usecase.specs) != 1 or len(direct_result["hits"]) != 3:
        raise RuntimeError("simple direct search did not execute exactly once")

    # Three platform agents are fixed to their scopes and their reply intervals overlap.
    dispatch_bus = TradeEventBus()
    search_factory = _FakeSearchFactory()
    trade_factory = _FakeSearchFactory()
    dispatch_tool = build_task_dispatch_tool(
        search_factory,  # type: ignore[arg-type]
        trade_factory,  # type: ignore[arg-type]
        dispatch_bus,
    )
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="h4-concurrent-dispatch",
            buyer_id="acceptance",
            locale="zh-CN",
            currency="CNY",
        )
    )
    try:
        await asyncio.gather(
            dispatch_tool(
                subagent_type="search_agent",
                demands="reference search",
                platform="crossshop_reference",
            ),
            dispatch_tool(
                subagent_type="search_agent",
                demands="reference_seed search",
                platform="reference_seed",
            ),
            dispatch_tool(
                subagent_type="search_agent",
                demands="amazon search",
                platform="amazon",
            ),
        )
    finally:
        ShoppingContext.reset(token)
    expected_scopes = {
        ("crossshop_reference", None),
        ("reference_seed", None),
        ("amazon", None),
    }
    if set(search_factory.scopes) != expected_scopes:
        raise RuntimeError(f"unexpected dispatched scopes: {search_factory.scopes}")
    overlap_start = max(worker.started for worker in search_factory.workers)
    overlap_end = min(worker.finished for worker in search_factory.workers)
    overlap_seconds = overlap_end - overlap_start
    if overlap_seconds <= 0:
        raise RuntimeError("three platform SearchAgent intervals did not overlap")

    # The tool itself performs no hidden retry. The current caller explicitly issues
    # one rewritten call after a low-hit result, preserving every other argument.
    rewrite_bus = TradeEventBus()
    rewrite_queue = rewrite_bus.subscribe("h4-one-rewrite")
    rewrite_usecase = _FakeUseCase("amazon", hit_counts=[1, 3])
    original_args: dict[str, Any] = {
        "normalized_query": "travel bag light",
        "platform": "amazon",
        "site_locale": "jp",
        "category": "bags",
        "ship_to": "JP",
        "locale": "ja-JP",
        "top_k": 5,
        "price_max_major": 12000.0,
        "target_currency": "JPY",
    }
    # Add a site-bound fake UseCase because platform selection occurs outside Spec.
    rewrite_tool = build_product_search_tool(
        {"amazon:jp": rewrite_usecase},  # type: ignore[arg-type]
        rewrite_bus,
    )
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="h4-one-rewrite",
            buyer_id="acceptance",
            locale="ja-JP",
            currency="JPY",
        )
    )
    try:
        first = _chunk_json(await rewrite_tool(**original_args))
        if len(first["hits"]) >= 3:
            raise RuntimeError("rewrite fixture must produce fewer than three hits")
        rewritten_args = {**original_args, "normalized_query": "軽量 旅行 バッグ"}
        second = _chunk_json(await rewrite_tool(**rewritten_args))
    finally:
        ShoppingContext.reset(token)
    if len(second["hits"]) < 3 or len(rewrite_usecase.specs) != 2:
        raise RuntimeError("one-rewrite policy did not stop after the second call")
    rewrite_invokes = [
        event["payload"]["args"]
        for event in _drain_events(rewrite_queue)
        if event["type"] == "tool.invoke"
    ]
    if len(rewrite_invokes) != 2:
        raise RuntimeError("expected exactly two explicit tool calls")
    differing_keys = {
        key
        for key in rewrite_invokes[0]
        if rewrite_invokes[0][key] != rewrite_invokes[1][key]
    }
    if differing_keys != {"normalized_query"}:
        raise RuntimeError(f"rewrite changed frozen constraints: {differing_keys}")

    return {
        "evidence_type": "deterministic_protocol_only_not_llm_behavior",
        "simple_single_platform": {
            "tool": "product_search_tool",
            "platform": "reference_seed",
            "product_search_calls": len(direct_usecase.specs),
            "agent_dispatch_calls": 0,
        },
        "concurrent_dispatch": {
            "scopes": [list(scope) for scope in search_factory.scopes],
            "intervals": [
                {
                    "scope": list(worker.scope),
                    "started": worker.started,
                    "finished": worker.finished,
                }
                for worker in search_factory.workers
            ],
            "overlap_seconds": overlap_seconds,
        },
        "one_rewrite": {
            "explicit_product_search_calls": len(rewrite_invokes),
            "hidden_retry_calls": 0,
            "changed_keys": sorted(differing_keys),
            "first_hit_count": len(first["hits"]),
            "second_hit_count": len(second["hits"]),
            "constraints_preserved": True,
        },
    }


def _is_natural_language_reply(final_text: str) -> bool:
    stripped = final_text.strip()
    if not stripped or stripped.startswith("[error]"):
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return True
    return not isinstance(parsed, (dict, list))


def _mentioned_recommendations(
    final_text: str,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match recommendations using unique anchors derived only from returned cards."""
    cards: dict[str, dict[str, Any]] = {}
    for event in events:
        payload = event.get("payload", {})
        if event.get("type") != "tool.result" or payload.get("tool") != (
            "product_search_tool"
        ):
            continue
        for hit in payload.get("hits", []):
            product_id = hit.get("product_id")
            title = hit.get("title")
            if isinstance(product_id, str) and isinstance(title, str):
                cards[product_id] = {
                    "product_id": product_id,
                    "title": title,
                    "brand": hit.get("brand"),
                }

    normalized_final = _normalized_name(final_text)
    normalized_titles = {
        product_id: _normalized_name(card["title"])
        for product_id, card in cards.items()
    }
    mentioned: list[dict[str, Any]] = []
    for product_id, card in cards.items():
        evidence = _card_mention_evidence(
            card,
            normalized_final=normalized_final,
            normalized_titles=normalized_titles,
        )
        if evidence is not None:
            mentioned.append(
                {
                    "product_id": product_id,
                    "title": card["title"],
                    "evidence": evidence,
                }
            )
    return mentioned


def _normalized_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _clean_brand(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value
    for suffix in ("旗舰店", "专卖店", "官方店", "官方", "店铺"):
        cleaned = cleaned.replace(suffix, "")
    return _normalized_name(cleaned)


def _card_mention_evidence(
    card: dict[str, Any],
    *,
    normalized_final: str,
    normalized_titles: dict[str, str],
) -> dict[str, str] | None:
    product_id = card["product_id"]
    if _normalized_name(product_id) in normalized_final:
        return {"kind": "product_id", "value": product_id}

    normalized_title = normalized_titles[product_id]
    if normalized_title and normalized_title in normalized_final:
        return {"kind": "exact_title", "value": card["title"]}

    brand = _clean_brand(card.get("brand"))
    if len(brand) >= 2 and brand in normalized_final:
        return {"kind": "event_brand", "value": str(card["brand"])}

    # Marketplace SEO titles are often shortened by MainAgent. Accept only a
    # fragment derived from the real card title that is unique among returned cards.
    for length in range(min(12, len(normalized_title)), 3, -1):
        for start in range(len(normalized_title) - length + 1):
            fragment = normalized_title[start : start + length]
            if fragment not in normalized_final:
                continue
            if any(
                fragment in other_title
                for other_id, other_title in normalized_titles.items()
                if other_id != product_id
            ):
                continue
            return {"kind": "unique_event_title_fragment", "value": fragment}
    return None


def _invoke_platform(event: dict[str, Any]) -> str | None:
    payload = event.get("payload", {})
    args = payload.get("args", {})
    platform = args.get("platform") if isinstance(args, dict) else None
    return platform if isinstance(platform, str) else None


def _ordered_routing_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按事件时间稳定排序；旧测试证据没有时间时保留原始顺序。"""
    return [
        event
        for _, event in sorted(
            enumerate(events),
            key=lambda pair: (str(pair[1].get("occurred_at") or ""), pair[0]),
        )
    ]


def _product_search_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把并发交错的 invoke/result 按平台和时间还原为独立调用。"""
    calls: list[dict[str, Any]] = []
    for event in _ordered_routing_events(events):
        payload = event.get("payload", {})
        if event.get("type") == "tool.invoke" and payload.get("tool") == (
            "product_search_tool"
        ):
            args = payload.get("args")
            calls.append(
                {
                    "occurred_at": event.get("occurred_at"),
                    "args": dict(args) if isinstance(args, dict) else {},
                    "result": None,
                }
            )
            continue
        if event.get("type") != "tool.result" or payload.get("tool") != (
            "product_search_tool"
        ):
            continue

        result_platform = payload.get("platform")
        pending = [call for call in calls if call["result"] is None]
        if isinstance(result_platform, str):
            matching = [
                call
                for call in pending
                if call["args"].get("platform") == result_platform
            ]
            target = matching[0] if matching else None
        else:
            # 参数校验错误在工具返回前尚未写入 platform；它紧跟对应 invoke，
            # 因而与最近一个未完成调用配对。
            target = pending[-1] if pending else None
        if target is not None:
            target["result"] = dict(payload)
    return calls


def _evaluate_product_search_protocol(
    events: list[dict[str, Any]],
    *,
    expected_platforms: set[str],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    calls = _product_search_calls(events)

    if any(
        event.get("type") == "tool.result"
        and event.get("payload", {}).get("harness") == "loop_detected"
        for event in events
    ):
        failures.append("harness reported loop_detected during a legal H4 route")

    grouped: dict[str, list[dict[str, Any]]] = {
        platform: [] for platform in expected_platforms
    }
    for position, call in enumerate(calls, start=1):
        args = call["args"]
        platform = args.get("platform")
        if platform not in expected_platforms:
            failures.append(
                f"product search call {position} used unexpected platform {platform!r}"
            )
            continue
        grouped[platform].append(call)

        top_k = args.get("top_k")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
            failures.append(
                f"{platform} product search call {position} used illegal top_k={top_k!r}"
            )
        result = call["result"]
        if result is None:
            failures.append(f"{platform} product search call {position} has no result")
        elif "error" in result:
            failures.append(
                f"{platform} product search call {position} returned error: "
                f"{result['error']}"
            )

    summaries: dict[str, Any] = {}
    for platform in sorted(expected_platforms):
        platform_calls = grouped[platform]
        if not platform_calls:
            failures.append(f"{platform} SearchAgent made no product search call")
            summaries[platform] = []
            continue
        if len(platform_calls) > 2:
            failures.append(
                f"{platform} made {len(platform_calls)} product searches; maximum is 2"
            )
        if len(platform_calls) >= 2:
            first_result = platform_calls[0]["result"] or {}
            first_hit_count = first_result.get("hit_count")
            if (
                not isinstance(first_hit_count, int)
                or isinstance(first_hit_count, bool)
                or first_hit_count >= 3
            ):
                failures.append(
                    f"{platform} issued a second search although the first successful "
                    f"hit_count was not below 3 (observed {first_hit_count!r})"
                )

            first_args = platform_calls[0]["args"]
            second_args = platform_calls[1]["args"]
            all_keys = set(first_args) | set(second_args)
            missing = object()
            changed_keys = sorted(
                key
                for key in all_keys
                if first_args.get(key, missing) != second_args.get(key, missing)
            )
            if changed_keys != ["normalized_query"]:
                failures.append(
                    f"{platform} rewrite changed invalid keys: {changed_keys}"
                )

        summaries[platform] = [
            {
                "occurred_at": call["occurred_at"],
                "args": call["args"],
                "hit_count": (
                    call["result"].get("hit_count")
                    if isinstance(call["result"], dict)
                    else None
                ),
                "error": (
                    call["result"].get("error")
                    if isinstance(call["result"], dict)
                    else None
                ),
            }
            for call in platform_calls
        ]
    return failures, {"calls_by_platform": summaries}


def _dispatch_overlap(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    dispatches: dict[str, datetime] = {}
    finishes: dict[str, datetime] = {}
    for event in events:
        payload = event.get("payload", {})
        platform = payload.get("platform")
        if not isinstance(platform, str):
            continue
        if event.get("type") == "agent.dispatch" and payload.get("agent") == (
            "search_agent"
        ):
            started_at = payload.get("started_at")
            if isinstance(started_at, str):
                dispatches[platform] = datetime.fromisoformat(started_at)
        if (
            event.get("type") == "tool.result"
            and payload.get("tool") == "task_dispatch"
            and payload.get("agent") == "search_agent"
        ):
            finished_at = payload.get("finished_at")
            if isinstance(finished_at, str):
                finishes[platform] = datetime.fromisoformat(finished_at)
    complete_platforms = sorted(dispatches.keys() & finishes.keys())
    overlap_seconds = None
    if complete_platforms:
        overlap_seconds = (
            min(finishes[platform] for platform in complete_platforms)
            - max(dispatches[platform] for platform in complete_platforms)
        ).total_seconds()
    return {
        "complete_platforms": complete_platforms,
        "intervals": {
            platform: {
                "started_at": dispatches[platform].isoformat(),
                "finished_at": finishes[platform].isoformat(),
            }
            for platform in complete_platforms
        },
        "overlap_seconds": overlap_seconds,
        "overlapped": overlap_seconds is not None and overlap_seconds > 0,
    }


def evaluate_real_main_case(
    name: str,
    final_text: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate observed MainAgent routing; absence of evidence is a failure."""
    failures: list[str] = []
    natural_language = _is_natural_language_reply(final_text)
    if not natural_language:
        failures.append("final reply is empty, an error, or structured JSON")

    mentions = _mentioned_recommendations(final_text, events)
    if not 1 <= len(mentions) <= 5:
        failures.append(
            "final reply must mention between 1 and 5 cards returned by the tool"
        )

    product_invokes = [
        event
        for event in events
        if event.get("type") == "tool.invoke"
        and event.get("payload", {}).get("tool") == "product_search_tool"
    ]
    search_dispatches = [
        event
        for event in events
        if event.get("type") == "agent.dispatch"
        and event.get("payload", {}).get("agent") == "search_agent"
    ]
    overlap = _dispatch_overlap(events)

    if name == "single-reference_seed":
        protocol_failures, search_protocol = _evaluate_product_search_protocol(
            events,
            expected_platforms={"reference_seed"},
        )
        failures.extend(protocol_failures)
        if search_dispatches:
            failures.append("single ReferenceSeed search dispatched a SearchAgent")
        if not 1 <= len(product_invokes) <= 2:
            failures.append(
                "single ReferenceSeed search must directly invoke the tool once or twice"
            )
        if any(_invoke_platform(event) != "reference_seed" for event in product_invokes):
            failures.append("single ReferenceSeed search invoked a non-ReferenceSeed platform")
        route = {
            "expected": "direct product_search_tool platform=reference_seed",
            "product_search_invoke_count": len(product_invokes),
            "product_search_platforms": [
                _invoke_platform(event) for event in product_invokes
            ],
            "search_agent_dispatch_count": len(search_dispatches),
            "search_protocol": search_protocol,
        }
    elif name == "cross-platform":
        protocol_failures, search_protocol = _evaluate_product_search_protocol(
            events,
            expected_platforms={"crossshop_reference", "reference_seed", "amazon"},
        )
        failures.extend(protocol_failures)
        platforms = [event["payload"].get("platform") for event in search_dispatches]
        expected = {"crossshop_reference", "reference_seed", "amazon"}
        if len(search_dispatches) != 3 or set(platforms) != expected:
            failures.append(
                "cross-platform search did not dispatch exactly three platforms"
            )
        if set(overlap["complete_platforms"]) != expected:
            failures.append("dispatch start/finish timestamps are incomplete")
        elif not overlap["overlapped"]:
            failures.append("three SearchAgent execution intervals did not overlap")
        route = {
            "expected": "three concurrent platform-fixed SearchAgents",
            "search_agent_dispatch_count": len(search_dispatches),
            "search_agent_platforms": platforms,
            "concurrency": overlap,
            "search_protocol": search_protocol,
        }
    else:
        raise ValueError(f"unsupported real MainAgent case: {name}")

    return {
        "passed": not failures,
        "failures": failures,
        "natural_language_reply": natural_language,
        "mentioned_recommendation_count": len(mentions),
        "mentioned_recommendations": mentions,
        "route": route,
    }


def _routing_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for event in events:
        payload = event.get("payload", {})
        event_type = event.get("type")
        if (
            event_type == "agent.dispatch"
            and payload.get("agent") == "search_agent"
            or event_type == "tool.invoke"
            and payload.get("tool") == "product_search_tool"
            or event_type == "tool.result"
            and payload.get("tool") in {"product_search_tool", "task_dispatch"}
        ):
            retained.append(event)
    return retained


def _load_saved_conversation_events(
    conversation_root: Path,
    *,
    query: str,
    final_text: str,
) -> tuple[list[dict[str, Any]], Path]:
    for path in sorted(conversation_root.glob("h4-main-*.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        buyer_turns = {
            row.get("content")
            for row in rows
            if row.get("kind") == "turn" and row.get("role") == "buyer"
        }
        agent_turns = {
            row.get("content")
            for row in rows
            if row.get("kind") == "turn" and row.get("role") == "agent"
        }
        if query not in buyer_turns or final_text not in agent_turns:
            continue
        events = [
            {
                "type": row["type"],
                "payload": row.get("payload", {}),
                "occurred_at": row.get("occurred_at", ""),
            }
            for row in rows
            if row.get("kind") == "event"
        ]
        return events, path
    raise RuntimeError(
        "no saved conversation contains the exact query and final_text; "
        "offline reevaluation cannot infer missing tool evidence"
    )


def reevaluate_existing_report(
    report_path: Path,
    *,
    conversation_root: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    main = report.get("real_main_agent")
    if not isinstance(main, dict) or not isinstance(main.get("cases"), dict):
        raise TypeError("acceptance report has no real_main_agent cases")

    sources: dict[str, str] = {}
    for name, case in main["cases"].items():
        events = case.get("routing_events")
        if not isinstance(events, list) or not events:
            events, source = _load_saved_conversation_events(
                conversation_root,
                query=case["query"],
                final_text=case["final_text"],
            )
            sources[name] = str(source.resolve())
        retained = _routing_events(events)
        evaluation = evaluate_real_main_case(name, case["final_text"], retained)
        invokes = [event for event in retained if event["type"] == "tool.invoke"]
        dispatches = [event for event in retained if event["type"] == "agent.dispatch"]
        case["route_evaluation"] = evaluation
        case["success"] = evaluation["passed"]
        case["failure"] = (
            None if evaluation["passed"] else "; ".join(evaluation["failures"])
        )
        case["routing_events"] = retained
        case["tool_invocations"] = [event["payload"] for event in invokes]
        case["agent_dispatches"] = [event["payload"] for event in dispatches]
        case["observed_platforms"] = sorted(
            {
                platform
                for platform in [
                    *(_invoke_platform(event) for event in invokes),
                    *(event["payload"].get("platform") for event in dispatches),
                ]
                if isinstance(platform, str)
            }
        )
    report["offline_reevaluation"] = {
        "evaluated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "external_requests": 0,
        "saved_conversation_sources": sources,
    }
    _atomic_json(report_path, report)
    return report


def _print_json_terminal_safe(value: Any, *, stream: TextIO = sys.stdout) -> None:
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    encoding = stream.encoding or "utf-8"
    safe = rendered.encode(encoding, errors="backslashreplace").decode(encoding)
    print(safe, file=stream)


async def run_real_main_agent(
    container: Container,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Attempt two real MainAgent conversations and report behavior, never infer it."""
    cases = (
        (
            "single-reference_seed",
            "只在示例平台帮我找轻便旅行背包，预算 100 元以内，收货到中国，最多推荐 5 件。",
        ),
        (
            "cross-platform",
            "请同时比较 CrossShop、示例平台和 Amazon 的降噪耳机，预算 300 元以内，收货到中国，最多推荐 5 件。",
        ),
    )
    report: dict[str, Any] = {"attempted": True, "cases": {}}
    for name, query in cases:
        session_id = f"h4-main-{name}-{int(time.time())}"
        queue = container.bus.subscribe(session_id)
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                container.orchestrator.handle_intent(
                    SubmitIntentInput(
                        shopping_session_id=session_id,
                        buyer_id="h4-acceptance",
                        locale="zh-CN",
                        currency="CNY",
                        raw_query=query,
                    )
                ),
                timeout=timeout_seconds,
            )
            error = None
            final_text = response.final_text
        except Exception as exc:  # noqa: BLE001 - evidence must retain real failure
            error = f"{type(exc).__name__}: {exc}"
            final_text = ""
        elapsed_seconds = time.perf_counter() - started
        events = _drain_events(queue)
        container.bus.unsubscribe(session_id, queue)
        routing_events = _routing_events(events)
        invokes = [event for event in routing_events if event["type"] == "tool.invoke"]
        dispatches = [
            event for event in routing_events if event["type"] == "agent.dispatch"
        ]
        route_evaluation = evaluate_real_main_case(name, final_text, routing_events)
        observed_failure = error
        if observed_failure is None and final_text.startswith("[error]"):
            observed_failure = final_text
        if observed_failure is None and not route_evaluation["passed"]:
            observed_failure = "; ".join(route_evaluation["failures"])
        report["cases"][name] = {
            "query": query,
            "elapsed_seconds": elapsed_seconds,
            "success": observed_failure is None,
            "failure": observed_failure,
            "final_text": final_text,
            "route_evaluation": route_evaluation,
            "routing_events": routing_events,
            "tool_invocations": [event["payload"] for event in invokes],
            "agent_dispatches": [event["payload"] for event in dispatches],
            "observed_platforms": sorted(
                {
                    platform
                    for platform in [
                        *(_invoke_platform(event) for event in invokes),
                        *(event["payload"].get("platform") for event in dispatches),
                    ]
                    if isinstance(platform, str)
                }
            ),
        }
    return report


def _render_readme(report: dict[str, Any], args: argparse.Namespace) -> str:
    real_rows = []
    for name, case in report["real_infrastructure"]["cases"].items():
        real_rows.append(
            f"| {name} | {case['index_name']} | {case['embedding']['dimension']} | "
            f"1 | 1 | {case['hit_count']} | {case['filtered_out_count']} | "
            f"{case['elapsed_seconds']:.3f} 秒 |"
        )
    main = report.get("real_main_agent")
    if main is None:
        main_summary = "未请求真实 MainAgent 对话验收。"
    else:
        lines = []
        for name, case in main["cases"].items():
            state = "通过" if case["success"] else f"未通过（{case['failure']}）"
            lines.append(
                f"- `{name}`：{state}；派发 {len(case['agent_dispatches'])} 次；"
                f"工具调用 {len(case['tool_invocations'])} 次。"
            )
        main_summary = "\n".join(lines)
    if args.reevaluate_existing:
        command_suffix = " --reevaluate-existing"
    elif args.run_main_agent:
        command_suffix = " --run-main-agent"
    else:
        command_suffix = ""
    return f"""# H4 Agent 与三平台在线检索验收

本目录由 `scripts/index/smoke_h4_agent.py` 原子生成。真实基础设施部分通过生产
`build_container()` 装配，并调用唯一的 `product_search_tool`；确定性协议部分只验证
派发、并发和一次改写边界，不代表 LLM 行为或检索质量。

## 实测命令

```powershell
.venv\\Scripts\\python.exe scripts/index/smoke_h4_agent.py{command_suffix}
```

| 用例 | 固定索引 | Query 维度 | Hybrid | 真实 Reranker | hits | filtered_out | 总耗时 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(real_rows)}

服务健康信息和每次真实请求的原始观察值保存在 `acceptance-report.json`。每个平台一次
工具调用对应一次 BGE-M3 Query、一次 OpenSearch Hybrid，以及硬约束后的单次真实
`bge-reranker-v2-m3` 调用；Amazon JP 使用同一个 Amazon 索引和 `site_locale=jp`。

确定性协议验收：简单示例平台查询直接调用同一工具且派发数为 0；三个固定平台 SearchAgent
区间实际重叠 {report["deterministic_protocol"]["concurrent_dispatch"]["overlap_seconds"]:.3f} 秒；
候选不足由当前调用者显式改写一次，只有 `normalized_query` 改变，工具内部没有隐藏重试。

## 真实 MainAgent 尝试

{main_summary}

真实 MainAgent 的结果按事件原样保存。模型未遵循提示词或外部网关失败时，不会被确定性
协议测试掩盖，也不会伪造为通过。

本验收不迁移品类 RAG，不实现 H5/H6，也不在无标注数据的情况下声明排序质量提升。
"""


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _validated_output_root(args.output_root)
    deterministic = await run_deterministic_protocol()
    container = await build_container()
    try:
        real = await run_real_infrastructure(
            container,
            timeout_seconds=args.timeout_seconds,
            output_root=output_root,
        )
        main = (
            await run_real_main_agent(
                container,
                timeout_seconds=args.main_timeout_seconds,
            )
            if args.run_main_agent
            else None
        )
    finally:
        await container.shutdown()
    report = {
        "executed_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "deterministic_protocol": deterministic,
        "real_infrastructure": real,
        "real_main_agent": main,
        "scope": {
            "h4": True,
            "h5_executed": False,
            "h6_executed": False,
            "category_rag_migrated": False,
        },
    }
    _atomic_json(output_root / "acceptance-report.json", report)
    _atomic_text(output_root / "README.md", _render_readme(report, args))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "h4-agent",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--main-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--run-main-agent", action="store_true")
    parser.add_argument("--reevaluate-existing", action="store_true")
    parser.add_argument(
        "--conversation-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "conversations",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.reevaluate_existing:
        output_root = _validated_output_root(args.output_root)
        report = reevaluate_existing_report(
            output_root / "acceptance-report.json",
            conversation_root=args.conversation_root.resolve(),
        )
        _atomic_text(output_root / "README.md", _render_readme(report, args))
    else:
        report = asyncio.run(run(args))
    _print_json_terminal_safe(report)


if __name__ == "__main__":
    main()
