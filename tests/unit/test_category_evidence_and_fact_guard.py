from __future__ import annotations

import json

import pytest

from globex_agent.application.agents.orchestrator import _sanitize_audit_payload
from globex_agent.application.prompts.loader import load_prompts
from globex_agent.application.tools.category_insight_tool import (
    build_category_insight_tool,
)
from globex_agent.category_insight.models import CategoryEvidenceRef
from globex_agent.eval.fact_guard import validate_final_response
from globex_agent.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from globex_agent.infrastructure.eventbus import TradeEventBus
from scripts.eval_flow_rubric import (
    JudgeSchemaError,
    _exposed_category_result,
    build_judge_evidence,
    normalize_judge_result,
    normalize_rubric,
    parse_conversation,
    render_evidence,
    render_facts,
    render_report,
    score_case,
)


class _Run:
    class _Output:
        def model_dump(self, mode: str) -> dict:
            assert mode == "json"
            return {
                "category": "乳胶枕",
                "components": [],
                "bestsellers": [],
                "attributes": [],
                "price_tiers": [
                    {"tier": "budget", "range_cny": [100.0, 300.0], "notes": "参考"}
                ],
                "confidence": 0.8,
            }

    output = _Output()
    diagnostics = type(
        "Diagnostics",
        (),
        {
            "evidence_refs": (
                CategoryEvidenceRef(
                    card_id="card-1",
                    category="乳胶枕",
                    card_type="price_range",
                    last_updated="2026-08-20T00:00:00+08:00",
                    confidence=0.8,
                ),
            )
        },
    )()


class _Service:
    def insight(self, category: str, *, depth: str):
        assert category == "乳胶枕"
        assert depth == "quick"
        return _Run()


async def test_category_tool_returns_compact_refs_and_source_boundary() -> None:
    bus = TradeEventBus()
    queue = bus.subscribe("evidence-session")
    tool = build_category_insight_tool(_Service(), bus)
    token = ShoppingContext.set(
        ShoppingContextSnapshot(
            shopping_session_id="evidence-session",
            buyer_id="buyer-1",
            locale="zh-CN",
            currency="CNY",
        )
    )
    try:
        result = json.loads(await tool.ainvoke({"category": "乳胶枕", "depth": "quick"}))
    finally:
        ShoppingContext.reset(token)

    assert result["evidence_refs"] == [
        {
            "card_id": "card-1",
            "category": "乳胶枕",
            "card_type": "price_range",
            "last_updated": "2026-08-20T00:00:00+08:00",
            "confidence": 0.8,
        }
    ]
    assert result["source_boundary"]["scope"] == "category_aggregate_reference"
    assert "具体商品实时价格" in result["source_boundary"]["exclusions"]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    result_event = next(event for event in events if event.type == "tool.result")
    assert result_event.payload["evidence_refs"] == result["evidence_refs"]
    assert "raw_evidence" not in result_event.payload


def test_fact_guard_distinguishes_product_and_category_prices() -> None:
    product = [{"item_id": "taobao:1", "price_major": 199, "currency": "CNY"}]
    category = [{"insights": {"price_tiers": [{"range_cny": [100, 300]}]}}]

    assert (
        validate_final_response(
            "商品价格是199元。",
            product_facts=product,
            category_insights=category,
        )
        == ()
    )
    codes = {
        violation.code
        for violation in validate_final_response(
            "这款商品价格是300元。",
            product_facts=product,
            category_insights=category,
        )
    }
    assert codes == {"category_price_as_product_fact"}

    assert (
        validate_final_response(
            "乳胶枕品类参考价约200元。",
            product_facts=product,
            category_insights=category,
        )
        == ()
    )
    assert {
        violation.code
        for violation in validate_final_response(
            "这款商品约200元。",
            product_facts=product,
            category_insights=category,
        )
    } == {"unsourced_specific_price"}


def test_fact_guard_detects_unsupported_price_store_and_inventory() -> None:
    violations = validate_final_response(
        "店铺现货，库存充足，价格是499元。",
        product_facts=[{"item_id": "taobao:1", "availability": None}],
        category_insights=[],
    )
    assert {violation.code for violation in violations} == {
        "unsourced_specific_price",
        "unsupported_store_or_inventory",
    }


def test_fact_guard_allows_store_verification_request_but_not_store_claim() -> None:
    facts = [{"item_id": "taobao:1", "availability": None}]

    assert (
        validate_final_response(
            "材质未确认，建议向店铺确认。",
            product_facts=facts,
        )
        == ()
    )
    violations = validate_final_response(
        "店铺现货，建议向店铺确认。",
        product_facts=facts,
    )
    assert {
        violation.code for violation in violations
    } == {"unsupported_store_or_inventory"}


def test_flow_parser_preserves_product_price_bounds_for_fact_guard(tmp_path) -> None:
    path = tmp_path / "conversation.jsonl"
    rows = [
        {"kind": "turn", "role": "buyer", "content": "找一个商品"},
        {"kind": "turn", "role": "agent", "content": "价格是149元。"},
        {
            "kind": "event",
            "type": "tool.result",
            "payload": {
                "tool": "product_search_tool",
                "hits": [
                    {
                        "item_id": "item-1",
                        "title": "商品",
                        "category": "品类",
                        "price_major": None,
                        "price_min_major": 149.0,
                        "price_max_major": 219.0,
                        "currency": "CNY",
                    }
                ],
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )

    parsed = parse_conversation(path)

    assert parsed["facts"][0]["price_min_major"] == 149.0
    assert parsed["facts"][0]["price_max_major"] == 219.0
    assert parsed["fact_violations"] == []

    rows[1]["content"] = "价格是220元。"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    assert {
        violation["code"] for violation in parse_conversation(path)["fact_violations"]
    } == {"unsourced_specific_price"}


def test_product_fact_boundaries_are_explicit_in_agent_prompts() -> None:
    prompts = load_prompts()
    main_prompt = prompts["main_agent"]["system_prompt"]
    search_prompt = prompts["sub_agents"]["search"]["system_prompt"]

    for prompt in (main_prompt, search_prompt):
        assert "shop/store/store_name" in prompt
        assert "区间内部价" in prompt


def test_audit_payload_redacts_pii_and_omits_raw_evidence() -> None:
    payload = _sanitize_audit_payload(
        {
            "phone": "13800000000",
            "raw_evidence": ["full source text"],
            "evidence_refs": [{"card_id": "card-1"}],
        }
    )
    assert payload["phone"] == "[redacted]"
    assert payload["raw_evidence"] == "[omitted: use evidence_refs]"
    assert payload["evidence_refs"] == [{"card_id": "card-1"}]


def test_category_insight_audit_whitelist_preserves_tiers_and_bounds_payload() -> None:
    payload = _sanitize_audit_payload(
        {
            "tool": "category_insight_tool",
            "category": "羽毛球包",
            "insights": {
                "category": "羽毛球包",
                "confidence": 0.9,
                "price_tiers": [
                    {
                        "tier": "budget",
                        "range_cny": [5, 90],
                        "source_scope": "category_reference",
                        "hidden": "must not persist",
                    }
                ],
                "components": ["x" * 2000 for _ in range(20)],
                "raw_evidence": ["full card text"],
            },
            "evidence_refs": [{"card_id": "card-1", "category": "羽毛球包"}],
            "source_boundary": {"scope": "category_aggregate_reference"},
        }
    )
    encoded = json.dumps(payload, ensure_ascii=False)

    assert payload["insights"]["price_tiers"] == [
        {"tier": "budget", "range_cny": [5, 90], "source_scope": "category_reference"}
    ]
    assert "hidden" not in encoded
    assert "raw_evidence" not in encoded
    assert "[omitted" not in encoded
    assert len(encoded.encode("utf-8")) < 20_000


def test_fact_guard_allows_category_tiers_from_exposed_audit_payload() -> None:
    category = [
        {
            "category": "羽毛球包",
            "insights": {
                "category": "羽毛球包",
                "price_tiers": [
                    {"tier": "budget", "range_cny": [5, 90], "source_scope": "category"},
                    {"tier": "mid", "range_cny": [90, 240], "source_scope": "category"},
                    {"tier": "premium", "range_cny": [240, 1180], "source_scope": "category"},
                ],
            },
        }
    ]
    assert validate_final_response(
        "品类参考区间：便宜款 5–90 元，中档 90–240 元，高端 240–1180 元。",
        category_insights=category,
    ) == ()


def test_fact_guard_excludes_budget_examples_but_not_product_price() -> None:
    assert validate_final_response("预算上限比如 20 元以内 / 50 元以内。") == ()
    assert validate_final_response(
        "这款商品价格是20元。", product_facts=[{"item_id": "item-1", "price_major": 20}]
    ) == ()


def test_fact_guard_requires_exact_variant_range_and_rejects_cross_item_price() -> None:
    facts = [
        {
            "item_id": "item-a",
            "title": "灯 A",
            "variants": [
                {
                    "options": [{"name": "颜色", "value": "[C4]黑色"}],
                    "display_name": "颜色: [C4]黑色",
                    "price_major": 18,
                },
                {
                    "options": [{"name": "颜色", "value": "[C4]白色"}],
                    "display_name": "颜色: [C4]白色",
                    "price_major": 21,
                },
            ],
        },
        {
            "item_id": "item-b",
            "title": "灯 B",
            "variants": [
                {
                    "options": [{"name": "颜色", "value": "[C6]黑色"}],
                    "display_name": "颜色: [C6]黑色",
                    "price_major": 329,
                }
            ],
        },
    ]
    assert validate_final_response("灯 A 的黑色款价格329元。", product_facts=facts)
    assert validate_final_response("按颜色变化价格18–21元。", product_facts=facts) == ()
    assert validate_final_response("按颜色变化价格18–22元。", product_facts=facts)
    assert validate_final_response("价格18–21元。", product_facts=facts)


def test_fact_guard_rejects_ambiguous_variant_alias_and_wrong_price() -> None:
    facts = [
        {
            "item_id": "item-1",
            "variants": [
                {
                    "options": [{"name": "接口", "value": "USB"}],
                    "display_name": "接口 USB",
                    "price_major": 18,
                },
                {
                    "options": [{"name": "接口", "value": "USB"}],
                    "display_name": "接口 USB",
                    "price_major": 21,
                },
            ],
        }
    ]
    assert validate_final_response("USB款价格18元。", product_facts=facts)
    assert validate_final_response("USB款价格22元。", product_facts=facts)


def test_judge_evidence_keeps_variant_prices_when_major_price_is_null() -> None:
    variants = [
        {"variant_id": "v-145", "display_name": "标准 145", "price_major": 145},
        {"variant_id": "v-11", "display_name": "单个 11", "price_major": 11},
        {"variant_id": "v-18", "display_name": "18 元", "price_major": 18},
        {"variant_id": "v-21", "display_name": "21 元", "price_major": 21},
        {"variant_id": "v-5", "display_name": "5 元", "price_major": 5},
        {"variant_id": "v-30", "display_name": "30 元", "price_major": 30},
        {"variant_id": "v-149", "display_name": "149 元", "price_major": 149},
        {"variant_id": "v-219", "display_name": "219 元", "price_major": 219},
    ]
    item = {
        "facts": [
            {
                "item_id": "item-variants",
                "title": "多变体商品",
                "category": "测试品类",
                "evidence_id": "ev-1",
                "content_hash": "hash-1",
                "schema_version": "schema-v2",
                "price_major": None,
                "price_source": "variant",
                "variants": variants,
            }
        ],
        "category_results": [],
    }

    evidence = build_judge_evidence(item)
    rendered = render_evidence(evidence)
    assert evidence["products"][0]["price_major"] is None
    assert [row["price_major"] for row in evidence["products"][0]["variants"]] == [
        145,
        11,
        18,
        21,
        5,
        30,
        149,
        219,
    ]
    assert "不可用" not in rendered
    assert "145" in rendered and "149" in rendered and "219" in rendered
    assert "item-variants" in render_facts(item["facts"])
    variant_prices = {
        entry["value"]
        for entry in evidence["evidence_catalog"]
        if entry["fact_type"] == "variant.price_major"
    }
    assert {5, 11, 18, 21, 30, 145, 149, 219} <= variant_prices


def test_judge_category_evidence_keeps_exposed_scope_and_fields() -> None:
    category = _exposed_category_result(
        {
            "category": "羽毛球包",
            "insights": {
                "category": "羽毛球包",
                "confidence": 0.8,
                "price_tiers": [
                    {"tier": "budget", "range_cny": [5, 30], "source_scope": "category"}
                ],
                "attributes": [{"name": "容量", "distribution": ["20L"]}],
                "bestsellers": [{"name": "示例款", "typical_price_cny": 149}],
                "hidden": "never expose",
            },
            "source_boundary": {"scope": "category_aggregate_reference"},
        }
    )
    evidence = build_judge_evidence({"facts": [], "category_results": [category]})
    rendered = render_evidence(evidence)

    assert evidence["category_insights"][0]["scope"] == "category_aggregate_reference"
    assert evidence["category_insights"][0]["insights"]["price_tiers"][0]["range_cny"] == [5, 30]
    assert "attributes" in rendered and "bestsellers" in rendered
    assert "never expose" not in rendered


def test_judge_evidence_is_bounded_and_drops_hidden_product_fields() -> None:
    evidence = build_judge_evidence(
        {
            "facts": [
                {
                    "item_id": "item-1",
                    "title": "商品",
                    "category": "品类",
                    "internal_raw_text": "秘密" * 10000,
                    "variants": [
                        {"variant_id": str(i), "display_name": "款" + str(i), "price_major": i}
                        for i in range(120)
                    ],
                }
            ],
            "category_results": [
                {
                    "category": "品类",
                    "insights": {
                        "attributes": [
                            {"name": "属性", "distribution": ["x" * 5000] * 100}
                            for _ in range(50)
                        ],
                        "bestsellers": [
                            {"name": "商品", "why_popular": "y" * 5000}
                            for _ in range(50)
                        ],
                    },
                }
            ],
        }
    )
    encoded = json.dumps(evidence, ensure_ascii=False)
    assert "internal_raw_text" not in encoded
    assert len(encoded.encode("utf-8")) <= 120_000
    assert evidence["products"][0]["completeness"] == {"variants": "bounded_truncated"}
    assert len(evidence["category_insights"][0]["insights"]["attributes"]) <= 24


def test_judge_rubric_rejects_paths_and_unknown_ids() -> None:
    evidence = build_judge_evidence(
        {
            "facts": [
                {
                    "item_id": "item-1",
                    "evidence_id": "snapshot-1",
                    "variants": [
                        {"variant_id": "v-1", "display_name": "145元", "price_major": 145}
                    ],
                }
            ],
            "category_results": [],
        }
    )
    rubric, issues = normalize_rubric(
        {
            "p0": [
                {
                    "id": "p0-price",
                    "criterion": "不得声称具体价格",
                    "source_scope": "product",
                    "evidence_refs": ["product:item-1:variant:v-1:price"],
                    "applies_if": "回答包含具体价格",
                },
                {
                    "id": "p0-bad-path",
                    "criterion": "路径不得作为证据引用",
                    "source_scope": "product",
                    "evidence_refs": ["product_facts.variants.price_major"],
                    "applies_if": "always",
                },
            ]
        },
        evidence,
    )
    assert rubric["p0"][0]["id"] == "p0-price"
    assert "unknown evidence_refs" in ";".join(issues)


def test_evidence_catalog_ids_are_stable_unique_and_cover_system_facts() -> None:
    item = {
        "query": "预算30元以内",
        "transcript": "[buyer] 预算30元以内",
        "facts": [
            {
                "item_id": "item-1",
                "evidence_id": "snapshot-1",
                "title": "商品",
                "category": "品类",
                "price_major": None,
                "price_min_major": 5,
                "price_max_major": 30,
                "store": "店铺",
                "availability": "in_stock",
                "variants": [
                    {
                        "variant_id": "v-145",
                        "display_name": "145元",
                        "price_major": 145,
                        "availability": "in_stock",
                    },
                    {
                        "variant_id": "v-11",
                        "display_name": "11元",
                        "price_major": 11,
                        "availability": "out_of_stock",
                    },
                    {
                        "variant_id": "v-18-21",
                        "display_name": "18–21元",
                        "price_major": 18,
                        "availability": "in_stock",
                    },
                ],
            }
        ],
        "category_results": [
            {
                "category": "羽毛球包",
                "insights": {
                    "price_tiers": [{"tier": "budget", "range_cny": [5, 30]}],
                    "attributes": [{"name": "容量", "distribution": ["大"]}],
                    "bestsellers": [{"name": "示例款", "typical_price_cny": 149}],
                },
            }
        ],
    }
    first = build_judge_evidence(item)
    second = build_judge_evidence(item)
    first_ids = [entry["evidence_id"] for entry in first["evidence_catalog"]]
    second_ids = [entry["evidence_id"] for entry in second["evidence_catalog"]]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert "product:item-1:variant:v-145:price" in first_ids
    assert "product:item-1:variant:v-11:price" in first_ids
    assert "system:shipping:base-fee" in first_ids
    assert any(entry["fact_type"] == "price_tier" for entry in first["evidence_catalog"])
    assert any(entry["fact_type"] == "attribute" for entry in first["evidence_catalog"])
    assert any(entry["fact_type"] == "bestseller" for entry in first["evidence_catalog"])
    assert "145" in json.dumps(first["evidence_catalog"], ensure_ascii=False)


def test_evidence_catalog_contains_stable_ids_for_exposed_product_highlights() -> None:
    item = {
        "facts": [
            {
                "item_id": "item-highlights",
                "evidence_id": "snapshot-highlights",
                "title": "商品",
                "highlights": ["材质: 乳胶", "尺寸: 90cm"],
            }
        ],
        "category_results": [],
    }

    first = build_judge_evidence(item)
    second = build_judge_evidence(item)
    first_highlights = [
        entry for entry in first["evidence_catalog"] if entry["fact_type"] == "highlight"
    ]
    second_highlights = [
        entry for entry in second["evidence_catalog"] if entry["fact_type"] == "highlight"
    ]

    assert [entry["evidence_id"] for entry in first_highlights] == [
        entry["evidence_id"] for entry in second_highlights
    ]
    assert {entry["value"] for entry in first_highlights} == {"材质: 乳胶", "尺寸: 90cm"}
    assert all(
        entry["evidence_id"].startswith("product:item-highlights:highlight:")
        for entry in first_highlights
    )


def test_judge_criterion_and_result_id_sets_must_match_exactly() -> None:
    evidence = build_judge_evidence({"facts": [{"item_id": "item-1"}]})
    valid_id = "product:item-1:title"
    rubric = {
        "p0": [
            {
                "id": "p0-1",
                "criterion": "商品标题",
                "source_scope": "product",
                "evidence_refs": [valid_id],
                "applies_if": "always",
            }
        ],
        "p1": [],
        "p2": [],
    }
    normalized, issues = normalize_rubric(rubric, evidence)
    assert issues == []
    assert score_case(
        {
            "p0": [
                {
                    "criterion_id": "p0-1",
                    "status": "pass",
                    "reason": "有证据",
                    "evidence_refs": [valid_id],
                }
            ],
            "p1": [],
            "p2": [],
        },
        rubric=normalized,
        evidence=evidence,
    ) == (1.0, True)
    with pytest.raises(JudgeSchemaError, match="missing criterion_ids"):
        normalize_judge_result(
            {"p0": [], "p1": [], "p2": []},
            rubric=normalized,
            evidence=evidence,
        )
    with pytest.raises(JudgeSchemaError, match="duplicate criterion_ids"):
        normalize_judge_result(
            {
                "p0": [
                    {"criterion_id": "p0-1", "status": "pass", "evidence_refs": [valid_id]},
                    {"criterion_id": "p0-1", "status": "pass", "evidence_refs": [valid_id]},
                ],
                "p1": [],
                "p2": [],
            },
            rubric=normalized,
            evidence=evidence,
        )
    with pytest.raises(JudgeSchemaError, match="extra criterion_ids"):
        normalize_judge_result(
            {
                "p0": [
                    {"criterion_id": "p0-1", "status": "pass", "evidence_refs": [valid_id]},
                    {"criterion_id": "p0-2", "status": "pass", "evidence_refs": [valid_id]},
                ],
                "p1": [],
                "p2": [],
            },
            rubric=normalized,
            evidence=evidence,
        )


def test_empty_rubric_or_judge_never_scores_as_one() -> None:
    evidence = build_judge_evidence({"facts": [{"item_id": "item-1"}]})
    _, rubric_issues = normalize_rubric({"p0": [], "p1": [], "p2": []}, evidence)
    assert "rubric has no criteria" in ";".join(rubric_issues)
    with pytest.raises(JudgeSchemaError, match="no criteria"):
        normalize_judge_result({"p0": [], "p1": [], "p2": []})


def test_judge_status_scoring_and_legacy_format_are_compatible() -> None:
    legacy = normalize_judge_result(
        {"p0": [{"pass": True, "criterion": "事实"}], "p1": [], "p2": []}
    )
    assert score_case(legacy, []) == (1.0, True)
    inconclusive = normalize_judge_result(
        {
            "p0": [
                {
                    "status": "not_evaluable",
                    "reason": "证据缺失",
                    "evidence_refs": [],
                }
            ],
            "p1": [],
            "p2": [],
        }
    )
    assert score_case(inconclusive, []) == (None, False)
    with pytest.raises(JudgeSchemaError):
        normalize_judge_result({"p0": [{"status": "unknown"}]})


def test_score_case_marks_no_applicable_p0_as_na_not_pass() -> None:
    evidence = build_judge_evidence({"facts": [{"item_id": "item-1"}]})
    rubric, issues = normalize_rubric(
        {
            "p0": [],
            "p1": [
                {
                    "id": "p1-1",
                    "criterion": "回答满足查询",
                    "source_scope": "conversation",
                    "evidence_refs": ["conversation:query"],
                    "applies_if": "always",
                }
            ],
            "p2": [],
        },
        evidence,
    )
    assert issues == []
    assert score_case(
        {
            "p0": [],
            "p1": [
                {
                    "criterion_id": "p1-1",
                    "status": "pass",
                    "evidence_refs": ["conversation:query"],
                }
            ],
            "p2": [],
        },
        rubric=rubric,
        evidence=evidence,
    ) == (1.0, None)


def test_deterministic_report_counts_fact_guard_passes() -> None:
    report = render_report(
        [
            {
                "source": "flow-1",
                "query": "商品",
                "transcript": "[buyer] 商品\n\n[agent] 回答",
                "facts": [],
                "category_results": [],
                "fact_violations": [],
                "rubric": {},
                "judged": {},
                "score": 1.0,
                "p0_pass": True,
            }
        ]
    )
    assert "总览：1/1 PASS" in report
