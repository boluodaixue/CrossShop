"""Unit tests for model-generated rubric contracts and user-visible payloads."""

import asyncio
import json

import pytest

from scripts import eval_flow_rubric


def _p2() -> list[dict[str, object]]:
    return [
        {
            "id": f"P2-{index}",
            "dimension": dimension,
            "requirements": [dimension],
            "expectation": f"回答体现{dimension}",
            "evidence_sources": ["query", "final_text", "final_cards"],
        }
        for index, dimension in enumerate(("需求覆盖度", "场景洞察力", "决策建议价值"), 1)
    ]


def test_model_generated_p1_is_preserved_and_p0_p2_are_strictly_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = {
        "query": "推荐羽毛球包",
        "tool_expectation": {
            "required": ["product_search_tool"],
            "optional": [],
            "tool_call_limits": {"product_search_tool": 2},
        },
    }

    async def fake_model(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "p0": [],
            "p1": [
                {
                    "id": "P1-model",
                    "criterion": "必须调用工具 product_search_tool",
                    "evidence_sources": [
                        "execution.tool_sequence",
                        "execution.tool_counts",
                    ],
                }
            ],
            "p2": _p2(),
        }

    monkeypatch.setattr(eval_flow_rubric, "_call_json_model", fake_model)
    result = asyncio.run(eval_flow_rubric.call_rubric_generator(None, item))
    assert result["evaluation_status"] == "completed"
    rubric = result["rubric"]
    assert rubric["p1"][0]["id"] == "P1-model"
    assert rubric["p1"][0]["criterion"] == "必须调用工具 product_search_tool"

    async def invalid_model(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "p0": [],
            "p1": [
                {
                    "id": "P1-model",
                    "criterion": "必须调用工具 product_search_tool",
                    "evidence_sources": [
                        "execution.tool_sequence",
                        "execution.tool_counts",
                    ],
                }
            ],
            "p2": [],
        }

    monkeypatch.setattr(eval_flow_rubric, "_call_json_model", invalid_model)
    with pytest.raises(ValueError, match="exactly once"):
        asyncio.run(eval_flow_rubric.call_rubric_generator(None, item))


def test_prompt_is_unbranded_and_defines_p1_and_evidence_boundaries() -> None:
    rubric_prompt = eval_flow_rubric.RUBRIC_GENERATOR_SYSTEM
    judge_prompt = eval_flow_rubric.FINAL_JUDGE_SYSTEM
    assert "Globex" not in rubric_prompt + judge_prompt
    for text in (rubric_prompt,):
        assert "P1" in text and "允许示例" in text and "禁止示例" in text
        assert "预算" in text and "防水" in text and "颜色" in text and "材质" in text
        assert "supported" in text and "true" in text
        assert "evidence-bounded" in text
        assert "不得预设最终证据包含任何外部品类知识" in text
        assert "每条 expectation 必须是条件式" in text
        assert "若 final_text/final_cards 提供相关事实" in text
        assert "若证据缺失" in text
        assert "不得因缺失未提供的知识本身扣分" in text
        assert "requirements 应改写为可评价行为" in text
        assert "明确无合格结果" in text
        assert "决策建议价值不等于必须提供下一步" in text
        assert "没有合格商品证据时" in text
        assert "场景洞察力同理" in text
    assert "不得因回答没有" in judge_prompt
    assert "只有 query 明确需要且证据支持时才评价这些" in judge_prompt
    assert "zero-hit" in judge_prompt


def test_rubric_input_shape_is_unchanged() -> None:
    payload = eval_flow_rubric.build_rubric_input({"query": "推荐商品"})
    assert set(payload) == {"query", "flow_type", "flow_expectations", "available_evidence"}


def test_invalid_p1_is_rejected_without_backend_override() -> None:
    rubric = {
        "p0": [],
        "p1": [
            {
                "id": "P1-invalid",
                "criterion": "预算必须满足",
                "evidence_sources": ["query"],
            }
        ],
        "p2": _p2(),
    }
    with pytest.raises(ValueError, match="P1 may use only execution"):
        eval_flow_rubric.validate_rubric(rubric)


def test_zero_hit_p2_uses_non_empty_evidence_bounded_requirements() -> None:
    rubric = {
        "p0": [],
        "p1": [],
        "p2": [
            {
                "id": "P2-coverage",
                "dimension": "需求覆盖度",
                "requirements": ["明确无合格商品结果，不以不合格商品替代"],
                "expectation": "若没有合格商品证据，诚实说明零命中并避免臆测。",
                "evidence_sources": ["query", "final_text", "final_cards"],
            },
            {
                "id": "P2-context",
                "dimension": "场景洞察力",
                "requirements": ["说明缺少品类知识时的证据限制"],
                "expectation": "不因未提供的防水等级或照射范围而编造事实。",
                "evidence_sources": ["query", "final_text", "final_cards"],
            },
            {
                "id": "P2-decision",
                "dimension": "决策建议价值",
                "requirements": ["在零命中时给出诚实可执行的结论"],
                "expectation": "只有 query 明确需要且证据支持时才评价对比或下一步。",
                "evidence_sources": ["query", "final_text", "final_cards"],
            },
        ],
    }
    validated = eval_flow_rubric.validate_rubric(rubric)
    assert all(entry["requirements"] for entry in validated["p2"])


def test_compact_card_filters_internal_highlights_but_keeps_user_visible_content() -> None:
    card = eval_flow_rubric._compact_card(
        {
            "title": "台灯",
            "category": "灯",
            "price_major": 20,
            "highlights": [
                " shop_name: 好店 ",
                "SOURCE_ATTRIBUTES: {'color': 'black'}",
                " source_tag: train ",
                "可调亮度",
            ],
        },
        "推荐台灯",
    )
    assert card is not None
    assert card["highlights"] == [" shop_name: 好店 ", "可调亮度"]

    payload = eval_flow_rubric.build_final_judge_input(
        {
            "query": "推荐台灯",
            "final_result": {
                "text": "推荐",
                "recommended_cards": [
                    {
                        "title": "台灯",
                        "highlights": [
                            "shop_name: 好店",
                            "source_attributes: hidden",
                            "source_tag: train",
                        ],
                    }
                ],
                "verification_status": "supported",
            },
            "events": [],
        },
        {"p0": [], "p1": [], "p2": _p2()},
    )
    payload_text = json.dumps(payload, ensure_ascii=False).casefold()
    assert "source_attributes:" not in payload_text
    assert "source_tag:" not in payload_text
    assert "shop_name: 好店" in payload_text
