"""Targeted tests for the reproducible H4 acceptance harness."""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path

import pytest

from app.composition import build_container
from scripts.index.smoke_h4_agent import (
    PROJECT_ROOT,
    _print_json_terminal_safe,
    _product_search_calls,
    _validated_output_root,
    evaluate_real_main_case,
    reevaluate_existing_report,
    run_deterministic_protocol,
    run_real_infrastructure,
)


def test_h4_acceptance_rejects_project_root_output() -> None:
    with pytest.raises(ValueError, match="must be a directory inside"):
        _validated_output_root(PROJECT_ROOT)


def test_h4_acceptance_rejects_outside_project_output() -> None:
    with pytest.raises(ValueError, match="must be a directory inside"):
        _validated_output_root(PROJECT_ROOT.parent / "outside-h4-evidence")


def test_h4_deterministic_protocol_acceptance() -> None:
    report = asyncio.run(run_deterministic_protocol())
    assert report["evidence_type"] == "deterministic_protocol_only_not_llm_behavior"
    assert report["simple_single_platform"] == {
        "tool": "product_search_tool",
        "platform": "reference-seed",
        "product_search_calls": 1,
        "agent_dispatch_calls": 0,
    }
    assert report["concurrent_dispatch"]["overlap_seconds"] > 0
    assert {tuple(scope) for scope in report["concurrent_dispatch"]["scopes"]} == {
        ("crossshop_reference", None),
        ("reference-seed", None),
        ("amazon", None),
    }
    assert report["one_rewrite"]["explicit_product_search_calls"] == 2
    assert report["one_rewrite"]["hidden_retry_calls"] == 0
    assert report["one_rewrite"]["changed_keys"] == ["normalized_query"]
    assert report["one_rewrite"]["constraints_preserved"] is True


def _event(event_type: str, payload: dict) -> dict:
    return {"type": event_type, "payload": payload}


def _product_invoke(
    platform: str,
    *,
    normalized_query: str = "测试商品",
    top_k: int = 5,
    price_max_major: int = 300,
) -> dict:
    return _event(
        "tool.invoke",
        {
            "tool": "product_search_tool",
            "args": {
                "normalized_query": normalized_query,
                "platform": platform,
                "site_locale": None,
                "category": "测试品类",
                "ship_to": "CN",
                "locale": "zh-CN",
                "top_k": top_k,
                "price_max_major": price_max_major,
                "target_currency": "CNY",
            },
        },
    )


def _product_result(
    product_id: str,
    title: str,
    brand: str = "",
    *,
    platform: str | None = None,
) -> dict:
    platform = platform or product_id.split(":", 1)[0]
    return _event(
        "tool.result",
        {
            "tool": "product_search_tool",
            "platform": platform,
            "hit_count": 1,
            "hits": [{"product_id": product_id, "title": title, "brand": brand}],
        },
    )


def test_real_main_single_reference-seed_requires_direct_tool_route() -> None:
    events = [
        _product_invoke("reference-seed"),
        _product_result("reference-seed:cn:1", "轻便旅行背包"),
    ]
    accepted = evaluate_real_main_case(
        "single-reference-seed",
        "推荐轻便旅行背包（reference-seed:cn:1），价格以商品卡为准。",
        events,
    )
    assert accepted["passed"] is True
    assert accepted["route"]["search_agent_dispatch_count"] == 0

    dispatched = evaluate_real_main_case(
        "single-reference-seed",
        "推荐轻便旅行背包（reference-seed:cn:1）。",
        [
            *events,
            _event(
                "agent.dispatch",
                {"agent": "search_agent", "platform": "reference-seed"},
            ),
        ],
    )
    assert dispatched["passed"] is False
    assert "dispatched a SearchAgent" in " ".join(dispatched["failures"])


def test_real_main_cross_platform_requires_timestamp_overlap() -> None:
    starts = {
        "crossshop_reference": "2026-08-27T12:00:00+00:00",
        "reference-seed": "2026-08-27T12:00:00.010000+00:00",
        "amazon": "2026-08-27T12:00:00.020000+00:00",
    }
    finishes = {
        "crossshop_reference": "2026-08-27T12:00:00.200000+00:00",
        "reference-seed": "2026-08-27T12:00:00.210000+00:00",
        "amazon": "2026-08-27T12:00:00.220000+00:00",
    }
    events = []
    for platform, started_at in starts.items():
        events.append(
            _event(
                "agent.dispatch",
                {
                    "agent": "search_agent",
                    "platform": platform,
                    "started_at": started_at,
                },
            )
        )
        events.append(_product_invoke(platform))
        events.append(
            _event(
                "tool.result",
                {
                    "tool": "task_dispatch",
                    "agent": "search_agent",
                    "platform": platform,
                    "finished_at": finishes[platform],
                },
            )
        )
        events.append(
            _product_result(
                f"{platform}:1",
                f"{platform} 降噪耳机",
                platform=platform,
            )
        )
    accepted = evaluate_real_main_case(
        "cross-platform",
        "推荐 crossshop_reference 降噪耳机、reference-seed 降噪耳机和 amazon 降噪耳机。",
        events,
    )
    assert accepted["passed"] is True
    assert accepted["route"]["concurrency"]["overlapped"] is True
    assert accepted["mentioned_recommendation_count"] == 3

    sequential_events = []
    for position, platform in enumerate(starts):
        sequential_events.extend(
            [
                _event(
                    "agent.dispatch",
                    {
                        "agent": "search_agent",
                        "platform": platform,
                        "started_at": f"2026-08-27T12:00:0{position}+00:00",
                    },
                ),
                _event(
                    "tool.result",
                    {
                        "tool": "task_dispatch",
                        "agent": "search_agent",
                        "platform": platform,
                        "finished_at": f"2026-08-27T12:00:0{position}.500000+00:00",
                    },
                ),
                _product_invoke(platform),
                _product_result(
                    f"{platform}:1",
                    f"{platform} 降噪耳机",
                    platform=platform,
                ),
            ]
        )
    rejected = evaluate_real_main_case(
        "cross-platform",
        "推荐 crossshop_reference 降噪耳机、reference-seed 降噪耳机和 amazon 降噪耳机。",
        sequential_events,
    )
    assert rejected["passed"] is False
    assert "did not overlap" in " ".join(rejected["failures"])


def test_real_main_reply_must_be_natural_language_and_at_most_five_cards() -> None:
    events = [
        _product_invoke("reference-seed"),
        _event(
            "tool.result",
            {
                "tool": "product_search_tool",
                "hits": [
                    {"product_id": f"reference-seed:cn:{index}", "title": f"背包{index}"}
                    for index in range(6)
                ],
                "platform": "reference-seed",
                "hit_count": 6,
            },
        ),
    ]
    structured = evaluate_real_main_case(
        "single-reference-seed",
        '{"hits": ["reference-seed:cn:0"]}',
        events,
    )
    assert structured["passed"] is False
    assert structured["natural_language_reply"] is False

    six_cards = evaluate_real_main_case(
        "single-reference-seed",
        "、".join(f"推荐背包{index}" for index in range(6)),
        events,
    )
    assert six_cards["passed"] is False
    assert six_cards["mentioned_recommendation_count"] == 6


def test_real_main_matches_shortened_titles_only_to_real_returned_cards() -> None:
    cards = [
        {
            "product_id": "reference-seed:cn:1",
            "title": "WRELS 儿童背包户外轻便双肩包折叠亲子旅游休闲书包",
            "brand": "WRELS旗舰店",
        },
        {
            "product_id": "reference-seed:cn:2",
            "title": "可折叠单肩包轻便大容量包包简约时尚休闲包学生书包",
            "brand": "天天热购",
        },
        {
            "product_id": "reference-seed:cn:3",
            "title": "韩版胸包女斜挎小背包男士单肩包男休闲轻便出行",
            "brand": "时尚女包便当包手机包箱包",
        },
        {
            "product_id": "reference-seed:cn:4",
            "title": "蕉下短途旅行包袋女LC168轻便大容量登机包托特包35L",
            "brand": "优品客礼品店",
        },
        {
            "product_id": "reference-seed:cn:5",
            "title": "2025新款户外运动单肩包斜挎包便携旅行休闲登山胸包",
            "brand": "沛迪旗舰店",
        },
    ]
    events = [
        _product_invoke("reference-seed"),
        _event(
            "tool.result",
            {
                "tool": "product_search_tool",
                "platform": "reference-seed",
                "hit_count": len(cards),
                "hits": cards,
            },
        ),
    ]
    final_text = (
        "推荐 5 件：WRELS 折叠双肩背包、天天热购可折叠单肩包、"
        "韩版胸包（斜挎小背包）、蕉下短途旅行托特包35L、沛迪户外胸包。"
    )
    evaluation = evaluate_real_main_case("single-reference-seed", final_text, events)
    assert evaluation["passed"] is True
    assert evaluation["mentioned_recommendation_count"] == 5
    assert all(
        mention["evidence"]["kind"] in {"event_brand", "unique_event_title_fragment"}
        for mention in evaluation["mentioned_recommendations"]
    )


def test_offline_reevaluation_uses_saved_conversation_events(tmp_path: Path) -> None:
    query = "只在示例平台找背包"
    final_text = "推荐 WRELS 折叠双肩背包。"
    report_path = tmp_path / "acceptance-report.json"
    report_path.write_text(
        json.dumps(
            {
                "real_main_agent": {
                    "cases": {
                        "single-reference-seed": {
                            "query": query,
                            "final_text": final_text,
                            "success": False,
                            "failure": "old evaluator failure",
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    conversation_root = tmp_path / "conversations"
    conversation_root.mkdir()
    rows = [
        {"kind": "turn", "role": "buyer", "content": query},
        {"kind": "turn", "role": "agent", "content": final_text},
        {
            "kind": "event",
            "type": "tool.invoke",
            "payload": {
                "tool": "product_search_tool",
                "args": _product_invoke("reference-seed")["payload"]["args"],
            },
        },
        {
            "kind": "event",
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "platform": "reference-seed",
                "hit_count": 1,
                "hits": [
                    {
                        "product_id": "reference-seed:cn:1",
                        "title": "WRELS 儿童背包户外轻便双肩包折叠",
                        "brand": "WRELS旗舰店",
                    }
                ],
            },
        },
    ]
    (conversation_root / "h4-main-single.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    updated = reevaluate_existing_report(
        report_path,
        conversation_root=conversation_root,
    )
    case = updated["real_main_agent"]["cases"]["single-reference-seed"]
    assert case["success"] is True
    assert case["route_evaluation"]["route"]["product_search_platforms"] == ["reference-seed"]
    assert case["route_evaluation"]["mentioned_recommendation_count"] == 1
    assert case["routing_events"]
    assert updated["offline_reevaluation"]["external_requests"] == 0


def test_real_main_rewrite_requires_previous_hit_count_below_three() -> None:
    first = _product_invoke("reference-seed", normalized_query="降噪耳机")
    first_result = _product_result("reference-seed:cn:1", "降噪耳机", platform="reference-seed")
    first_result["payload"]["hit_count"] = 5
    second = _product_invoke("reference-seed", normalized_query="无线降噪耳机")
    second_result = _product_result(
        "reference-seed:cn:2",
        "无线降噪耳机",
        platform="reference-seed",
    )
    evaluation = evaluate_real_main_case(
        "single-reference-seed",
        "推荐降噪耳机 reference-seed:cn:1。",
        [first, first_result, second, second_result],
    )
    assert evaluation["passed"] is False
    assert "was not below 3" in " ".join(evaluation["failures"])


def test_product_search_results_are_associated_by_time_and_platform() -> None:
    reference_invoke = _product_invoke("crossshop_reference")
    reference_invoke["occurred_at"] = "2026-08-28T00:00:00.100000+08:00"
    reference-seed_invoke = _product_invoke("reference-seed")
    reference-seed_invoke["occurred_at"] = "2026-08-28T00:00:00.110000+08:00"
    reference-seed_result = _product_result(
        "reference-seed:cn:1",
        "示例平台商品",
        platform="reference-seed",
    )
    reference-seed_result["occurred_at"] = "2026-08-28T00:00:00.200000+08:00"
    reference_result = _product_result(
        "P1001",
        "CrossShop 商品",
        platform="crossshop_reference",
    )
    reference_result["occurred_at"] = "2026-08-28T00:00:00.300000+08:00"

    calls = _product_search_calls(
        [reference_invoke, reference-seed_invoke, reference-seed_result, reference_result]
    )
    assert calls[0]["args"]["platform"] == "crossshop_reference"
    assert calls[0]["result"]["platform"] == "crossshop_reference"
    assert calls[1]["args"]["platform"] == "reference-seed"
    assert calls[1]["result"]["platform"] == "reference-seed"


def test_real_main_rewrite_preserves_every_constraint() -> None:
    first = _product_invoke("reference-seed", normalized_query="降噪耳机")
    first_result = _product_result("reference-seed:cn:1", "降噪耳机", platform="reference-seed")
    second = _product_invoke("reference-seed", normalized_query="无线降噪耳机")
    second_result = _product_result(
        "reference-seed:cn:2",
        "无线降噪耳机",
        platform="reference-seed",
    )
    accepted = evaluate_real_main_case(
        "single-reference-seed",
        "推荐无线降噪耳机 reference-seed:cn:2。",
        [first, first_result, second, second_result],
    )
    assert accepted["passed"] is True

    second["payload"]["args"]["price_max_major"] = 500
    rejected = evaluate_real_main_case(
        "single-reference-seed",
        "推荐无线降噪耳机 reference-seed:cn:2。",
        [first, first_result, second, second_result],
    )
    assert rejected["passed"] is False
    assert "price_max_major" in " ".join(rejected["failures"])


def test_real_main_rejects_illegal_top_k_even_if_model_corrects_it() -> None:
    illegal = _product_invoke("reference-seed", top_k=8)
    error = _event(
        "tool.result",
        {
            "tool": "product_search_tool",
            "error": "ProductSearchSpec.top_k 必须为 1..5 的整数",
        },
    )
    corrected = _product_invoke("reference-seed", normalized_query="旅行背包")
    result = _product_result("reference-seed:cn:1", "旅行背包", platform="reference-seed")
    evaluation = evaluate_real_main_case(
        "single-reference-seed",
        "推荐旅行背包 reference-seed:cn:1。",
        [illegal, error, corrected, result],
    )
    assert evaluation["passed"] is False
    failures = " ".join(evaluation["failures"])
    assert "illegal top_k=8" in failures
    assert "returned error" in failures


def test_real_main_rejects_loop_detected_event() -> None:
    evaluation = evaluate_real_main_case(
        "single-reference-seed",
        "推荐旅行背包 reference-seed:cn:1。",
        [
            _event(
                "tool.result",
                {"tool": "task_dispatch", "harness": "loop_detected"},
            ),
            _product_invoke("reference-seed"),
            _product_result("reference-seed:cn:1", "旅行背包", platform="reference-seed"),
        ],
    )
    assert evaluation["passed"] is False
    assert "loop_detected" in " ".join(evaluation["failures"])


def test_terminal_json_output_is_safe_on_windows_gbk() -> None:
    binary = io.BytesIO()
    stream = io.TextIOWrapper(binary, encoding="gbk", errors="strict")
    _print_json_terminal_safe({"message": "旅行背包💼"}, stream=stream)
    stream.flush()
    rendered = binary.getvalue().decode("gbk")
    assert "旅行背包" in rendered
    assert "\\U0001f4bc" in rendered


@pytest.mark.skipif(
    os.getenv("CROSSSHOP_RUN_H4_REAL_INFRA") != "1",
    reason="set CROSSSHOP_RUN_H4_REAL_INFRA=1 with local model services",
)
def test_h4_real_composition_three_platforms_and_amazon_jp() -> None:
    async def _run() -> dict:
        container = await build_container()
        try:
            return await run_real_infrastructure(
                container,
                timeout_seconds=120.0,
            )
        finally:
            await container.shutdown()

    report = asyncio.run(_run())
    assert report["repository"]["implementation"] == "JsonlProductRepository"
    assert report["repository"]["shared_by_all_usecases"] is True
    assert set(report["cases"]) == {
        "crossshop_reference",
        "reference-seed",
        "amazon",
        "amazon-jp",
    }
    for case in report["cases"].values():
        assert case["tool_invoke_count"] == 1
        assert case["embedding"]["dimension"] == 1024
        assert case["hybrid"]["embedding_dimension"] == 1024
        assert case["reranker_call"]["finite"] is True
        assert case["rerank_applied"] is True
        assert case["restored_product_count"] == case["hit_count"]
    amazon_jp = report["cases"]["amazon-jp"]
    assert amazon_jp["index_name"] == "crossshop-products-amazon-v1"
    assert amazon_jp["index_site_locale"] == "jp"
    assert all(
        product_id.startswith("amazon:jp:") for product_id in amazon_jp["product_ids"]
    )
