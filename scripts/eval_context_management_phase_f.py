"""Deterministic 20-turn context-management evaluation for Phase F.

This harness makes no network or model API calls.  It uses an exact character
counter as the deliberately calibrated model for this scenario, then exercises
the production reducer, lifecycle, L2/L3 compaction, assembler and L1 budget
decision code.  The resulting JSON contains every measured model-call layer;
it is evidence for this fixture only, never a Qwen production threshold.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    messages_to_dict,
)

from globex_agent.application.context import (
    ContextBudgetPolicy,
    FrozenSegment,
    context_pre_model_hook,
    group_interaction_units,
    reduce_l4,
)

TURN_COUNT = 20
FIRST_QUERY = "预算 500 元，寄到美国；找防泼水、非真皮的轻量通勤双肩包。"
QUERIES = (
    FIRST_QUERY,
    "先比较容量和重量。",
    "继续看适合 15 英寸电脑的。",
    "搜索失败的话请重试，不要放宽原约束。",
    "比较肩带和背板舒适度。",
    "同时找简约款和户外款。",
    "排除明显真皮材质。",
    "继续保留寄往美国的条件。",
    "优先防泼水和轻量。",
    "给我一个前三名比较。",
    "第二款的容量如何？",
    "检查第一款是否适合通勤。",
    "继续按 500 元预算筛选。",
    "再确认币种和目的国。",
    "推荐一个最终候选。",
    "为 item-a 准备订单并查询状态。",
    "订单里的数量是多少？",
    "回到商品，确认没有放宽材质约束。",
    "总结当前推荐理由。",
    "最后确认预算、国家、材质和订单状态。",
)


class ExactCharacterModel:
    """A reproducible exact counter for the evaluation's synthetic model."""

    token_count_is_estimated = False

    def get_num_tokens(self, text: str) -> int:
        return len(text)

    def get_num_tokens_from_messages(self, messages: Sequence[BaseMessage]) -> int:
        return sum(len(str(message.content)) + 1 for message in messages)


def _canonical_messages(messages: Sequence[BaseMessage]) -> str:
    payload = messages_to_dict(list(messages))
    # LangGraph's add_messages reducer assigns transport IDs on first
    # checkpoint write. They are not conversation semantics; tool_call IDs are
    # nested separately and remain part of this digest.
    for item in payload:
        data = item.get("data")
        if isinstance(data, dict):
            data.pop("id", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_messages(messages: Sequence[BaseMessage]) -> str:
    return hashlib.sha256(_canonical_messages(messages).encode("utf-8")).hexdigest()


def _search_output(item_id: str) -> dict[str, Any]:
    return {
        "hits": [
            {
                "item_id": item_id,
                "title": "轻量防泼水通勤双肩包",
                "price_major": 399,
                "currency": "CNY",
                "ship_to": "US",
            }
        ],
        "returned_count": 1,
    }


def _tool_call(name: str, call_id: str, args: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "id": call_id, "args": dict(args), "type": "tool_call"}


def _search_args(query: str) -> dict[str, Any]:
    return {
        "normalized_query": query,
        "ship_to": "US",
        "price_max_major": 500,
        "target_currency": "CNY",
        "material_exclude": ["leather"],
        "top_k": 5,
    }


def _search_turn(turn: int, query: str) -> tuple[list[BaseMessage], list[dict[str, Any]]]:
    args = _search_args("轻量防泼水通勤双肩包")
    events: list[dict[str, Any]] = []
    messages: list[BaseMessage] = []
    if turn in {1, 6}:
        call_a = f"search-{turn}-a"
        call_b = f"search-{turn}-b"
        args_b = {**args, "normalized_query": "户外防泼水双肩包"}
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("product_search_tool", call_a, args),
                    _tool_call("product_search_tool", call_b, args_b),
                ],
            )
        )
        # Results deliberately complete in reverse order to verify call-id pairing.
        for call_id, call_args, item_id in (
            (call_b, args_b, "item-b"),
            (call_a, args, "item-a"),
        ):
            output = _search_output(item_id)
            messages.append(
                ToolMessage(
                    content=json.dumps(output, ensure_ascii=False),
                    tool_call_id=call_id,
                    name="product_search_tool",
                )
            )
            events.extend(
                [
                    {
                        "type": "tool.invoke",
                        "payload": {
                            "tool": "product_search_tool",
                            "tool_call_id": call_id,
                            "args": call_args,
                        },
                    },
                    {
                        "type": "tool.result",
                        "payload": {
                            "tool": "product_search_tool",
                            "tool_call_id": call_id,
                            "hit_count": 1,
                            "model_output": output,
                        },
                    },
                ]
            )
    elif turn == 4:
        failed_id = "search-4-failed"
        retry_id = "search-4-retry"
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("product_search_tool", failed_id, args)],
                ),
                ToolMessage(
                    content='{"error":"timeout"}',
                    tool_call_id=failed_id,
                    name="product_search_tool",
                    status="error",
                ),
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("product_search_tool", retry_id, args)],
                ),
                ToolMessage(
                    content=json.dumps(_search_output("item-a"), ensure_ascii=False),
                    tool_call_id=retry_id,
                    name="product_search_tool",
                ),
            ]
        )
        events.extend(
            [
                {
                    "type": "tool.invoke",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": failed_id,
                        "args": args,
                    },
                },
                {
                    "type": "tool.result",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": failed_id,
                        "error": "timeout",
                    },
                },
                {
                    "type": "tool.invoke",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": retry_id,
                        "args": args,
                    },
                },
                {
                    "type": "tool.result",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": retry_id,
                        "hit_count": 1,
                        "model_output": _search_output("item-a"),
                    },
                },
            ]
        )
    else:
        call_id = f"search-{turn}"
        output = _search_output("item-a")
        messages.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("product_search_tool", call_id, args)],
                ),
                ToolMessage(
                    content=json.dumps(output, ensure_ascii=False),
                    tool_call_id=call_id,
                    name="product_search_tool",
                ),
            ]
        )
        events.extend(
            [
                {
                    "type": "tool.invoke",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": call_id,
                        "args": args,
                    },
                },
                {
                    "type": "tool.result",
                    "payload": {
                        "tool": "product_search_tool",
                        "tool_call_id": call_id,
                        "hit_count": 1,
                        "model_output": output,
                    },
                },
            ]
        )
    final_text = (
        "保持预算、目的国、防泼水和非真皮约束；item-a 是当前推荐。"
        "选择依据为实际搜索结果，未扩大原始条件。" * 12
    )
    messages.append(AIMessage(content=final_text))
    events.append(
        {
            "type": "final.result",
            "payload": {
                "verification_status": "supported",
                "recommended_cards": [
                    {
                        "item_id": "item-a",
                        "title": "轻量防泼水通勤双肩包",
                        "price_major": 399,
                        "currency": "CNY",
                        "ship_to": "US",
                    }
                ],
            },
        }
    )
    return messages, events


def _order_turn(turn: int) -> tuple[list[BaseMessage], list[dict[str, Any]]]:
    prepare_id = f"prepare-{turn}"
    query_id = f"order-status-{turn}"
    prepare_args = {
        "items": [{"item_id": "item-a", "quantity": 2, "unit_price_major": 399}],
        "shipping_address": {"country": "US"},
    }
    prepared = {
        "confirmation_required": True,
        "confirmation_id": "confirm-20",
        "items": [{"item_id": "item-a", "quantity": 2, "unit_price_major": 399}],
    }
    status = {
        "order_id": "order-20",
        "status": "CONFIRMED",
        "items": [{"item_id": "item-a", "quantity": 2, "unit_price_major": 399}],
    }
    messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[_tool_call("prepare_order_tool", prepare_id, prepare_args)],
        ),
        ToolMessage(
            content=json.dumps(prepared, ensure_ascii=False),
            tool_call_id=prepare_id,
            name="prepare_order_tool",
        ),
        AIMessage(
            content="",
            tool_calls=[
                _tool_call("query_order_tool", query_id, {"order_id": "order-20"})
            ],
        ),
        ToolMessage(
            content=json.dumps(status, ensure_ascii=False),
            tool_call_id=query_id,
            name="query_order_tool",
        ),
        AIMessage(content="订单 order-20 已确认，数量 2，目的国仍为美国。"),
    ]
    events = [
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "prepare_order_tool",
                "tool_call_id": prepare_id,
                "args": prepare_args,
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "prepare_order_tool",
                "tool_call_id": prepare_id,
                "model_output": prepared,
            },
        },
        {
            "type": "tool.invoke",
            "payload": {
                "tool": "query_order_tool",
                "tool_call_id": query_id,
                "args": {"order_id": "order-20"},
            },
        },
        {
            "type": "tool.result",
            "payload": {
                "tool": "query_order_tool",
                "tool_call_id": query_id,
                "model_output": status,
            },
        },
        {
            "type": "final.result",
            "payload": {
                "verification_status": "supported",
                "recommended_cards": [{"item_id": "item-a", "price_major": 399}],
            },
        },
    ]
    return messages, events


def _intent(query: str) -> dict[str, Any]:
    return {
        "shopping_session_id": "phase-f-20-turn",
        "buyer_id": "phase-f-buyer",
        "locale": "zh-CN",
        "currency": "CNY",
        "raw_query": query,
    }


def _record_call(
    *,
    calls: list[dict[str, Any]],
    turn: int,
    stage: str,
    update: Mapping[str, Any],
    frozen_before: int,
) -> None:
    report = update["budget_report"]
    calls.append(
        {
            "turn": turn,
            "stage": stage,
            "layers": {
                key: value["tokens"] for key, value in report["layers"].items()
            },
            "tool_results": {
                key: value["tokens"] for key, value in report["tool_results"].items()
            },
            "total_input_tokens": report["total_input_tokens"],
            "available_input_tokens": report["available_input_tokens"],
            "estimated": report["estimated"],
            "counter_method": report["counter_method"],
            "decision": update["budget_decision"],
            "freeze_cursor": update["freeze_cursor"],
            "frozen_segments": len(update["frozen_segments"]),
            "l2_appended": len(update["frozen_segments"]) - frozen_before,
            "l3_action": update.get("context_compression"),
        }
    )


def _apply_hook(
    state: dict[str, Any],
    *,
    policy: ContextBudgetPolicy,
    turn: int,
    stage: str,
    calls: list[dict[str, Any]],
) -> None:
    cursor = int(state.get("freeze_cursor", 0))
    active_before = _digest_messages(state["messages"][cursor:])
    l4_before = json.dumps(
        state["session_context"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    frozen_before = len(state.get("frozen_segments") or [])
    update = context_pre_model_hook(
        state,
        model=ExactCharacterModel(),
        fixed_prompt="Globex Phase F exact evaluation prompt",
        tool_schemas=[
            {"name": "product_search_tool"},
            {"name": "prepare_order_tool"},
            {"name": "query_order_tool"},
        ],
        budget_policy=policy,
    )
    assert active_before == _digest_messages(state["messages"][cursor:])
    assert l4_before == json.dumps(
        state["session_context"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    _record_call(
        calls=calls,
        turn=turn,
        stage=stage,
        update=update,
        frozen_before=frozen_before,
    )
    state.update(update)


def _simulate(*, mode: str, soft_limit_tokens: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "messages": [],
        "frozen_segments": [],
        "freeze_cursor": 0,
    }
    calls: list[dict[str, Any]] = []
    policy = ContextBudgetPolicy(
        model_context_tokens=1_000_000,
        soft_limit_tokens=soft_limit_tokens,
        mode=mode,
        l3_keep_recent_frozen_segments=1,
    )
    for turn, query in enumerate(QUERIES, 1):
        intent = _intent(query)
        state["session_context"] = reduce_l4(
            intent,
            (),
            state.get("session_context"),
            now=f"2026-08-24T00:{turn:02d}:00Z",
        ).to_dict()
        state["messages"].append(HumanMessage(content=query))
        _apply_hook(
            state,
            policy=policy,
            turn=turn,
            stage="request_start",
            calls=calls,
        )
        turn_messages, events = (
            _order_turn(turn) if turn == 16 else _search_turn(turn, query)
        )
        # This is the second real model boundary of the turn: tool results are
        # present and must be consumed, while the final assistant answer has
        # not been generated yet.
        state["messages"].extend(turn_messages[:-1])
        _apply_hook(
            state,
            policy=policy,
            turn=turn,
            stage="after_tool_results",
            calls=calls,
        )
        state["messages"].append(turn_messages[-1])
        state["session_context"] = reduce_l4(
            intent,
            events,
            state["session_context"],
            now=f"2026-08-24T00:{turn:02d}:59Z",
        ).to_dict()
    state.pop("llm_input_messages", None)
    return {"state": state, "model_calls": calls}


def run_evaluation() -> dict[str, Any]:
    """Run observation calibration and the exact enforced 20-turn scenario."""

    observation = _simulate(mode="observe_only", soft_limit_tokens=0)
    calibration_calls = [
        item for item in observation["model_calls"] if item["turn"] <= 8
    ]
    soft_limit = max(item["total_input_tokens"] for item in calibration_calls)
    evaluated = _simulate(mode="enforce", soft_limit_tokens=soft_limit)
    state = evaluated["state"]
    calls = evaluated["model_calls"]
    messages = state["messages"]
    units = group_interaction_units(messages)
    frozen = [FrozenSegment.from_dict(item) for item in state["frozen_segments"]]
    frozen_text = "\n".join(segment.content for segment in frozen)
    l3_actions = [item["l3_action"] for item in calls if item.get("l3_action")]
    # A persisted action remains in state; model-call records only count actions
    # returned by that hook. Deduplicate identical persisted summaries.
    unique_actions = {
        item["summary_hash"]: item for item in l3_actions if item is not None
    }
    summary = state.get("stage_summary") or {}
    active_messages = messages[int(state["freeze_cursor"]) :]
    l4 = state["session_context"]
    checks = {
        "twenty_turns": len(units) == TURN_COUNT,
        "protocol_complete_without_orphan_tool_results": (
            len(units) == TURN_COUNT and all(unit.complete for unit in units)
        ),
        "parallel_search_preserved": (
            "search-1-a" in frozen_text and "search-1-b" in frozen_text
        ),
        "failed_search_and_retry_preserved": (
            '"status":"error"' in frozen_text and "search-4-retry" in frozen_text
        ),
        "l2_froze_only_settled_prefix": (
            len(frozen) == TURN_COUNT - 1
            and len(active_messages) > 0
            and isinstance(active_messages[0], HumanMessage)
        ),
        "l3_triggered_with_exact_counter": bool(unique_actions),
        "l3_reduced_tokens": all(
            action["after_tokens"] < action["before_tokens"]
            for action in unique_actions.values()
        ),
        # Every one of the 40 pre-model hook calls asserted active and L4
        # digests before returning, including the final active interaction.
        "active_messages_unchanged": len(calls) == TURN_COUNT * 2,
        "l4_origin_budget_country_preserved": (
            l4["request"]["session_origin_raw_query"] == FIRST_QUERY
            and l4["last_search"]["args"]["ship_to"] == "US"
            and l4["last_search"]["args"]["price_max_major"] == 500
        ),
        "l4_current_query_preserved": l4["request"]["current_raw_query"] == QUERIES[-1],
        "recommendation_preserved": l4["last_recommendation"]["items"][0]["item_id"]
        == "item-a",
        "order_preserved": (
            l4["order"]["order_id"] == "order-20"
            and l4["order"]["status"] == "CONFIRMED"
            and l4["order"]["items"][0]["quantity"] == 2
        ),
        "raw_messages_and_frozen_sources_auditable": (
            len(messages) > len(frozen)
            and len(summary.get("source_segment_ids") or []) < len(frozen) + 1
            and all(segment.content for segment in frozen)
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Phase F evaluation checks failed: {failed}")
    layer_names = sorted(
        {name for item in calls for name in item["layers"]}
    )
    layer_ranges = {
        name: {
            "min": min(item["layers"].get(name, 0) for item in calls),
            "max": max(item["layers"].get(name, 0) for item in calls),
        }
        for name in layer_names
    }
    return {
        "scenario": "phase-f-deterministic-20-turn-v1",
        "turns": TURN_COUNT,
        "model_calls": calls,
        "counter": {
            "method": "ExactCharacterModel.get_num_tokens[_from_messages]",
            "estimated": False,
            "scope": "synthetic evaluation model only; not qwen3-max",
            "model_context_tokens": 1_000_000,
            "calibration": (
                "soft limit equals the maximum exact no-L3 model-call input observed "
                "through turn 8"
            ),
            "soft_limit_tokens": soft_limit,
        },
        "token_summary": {
            "total_min": min(item["total_input_tokens"] for item in calls),
            "total_max": max(item["total_input_tokens"] for item in calls),
            "tool_results_total_min": min(
                sum(item["tool_results"].values()) for item in calls
            ),
            "tool_results_total_max": max(
                sum(item["tool_results"].values()) for item in calls
            ),
            "layers": layer_ranges,
        },
        "lifecycle": {
            "raw_message_count": len(messages),
            "interaction_units": len(units),
            "freeze_cursor": state["freeze_cursor"],
            "frozen_segment_count": len(frozen),
            "stage_summary_source_count": len(summary.get("source_segment_ids") or []),
            "active_message_count": len(active_messages),
        },
        "l4": {
            "revision": l4["revision"],
            "origin_raw_query": l4["request"]["session_origin_raw_query"],
            "current_raw_query": l4["request"]["current_raw_query"],
            "last_search": l4["last_search"],
            "last_recommendation": l4["last_recommendation"],
            "order": l4["order"],
        },
        "l3_actions": list(unique_actions.values()),
        "checks": checks,
        "checkpoint_state": state,
        "active_digest": _digest_messages(active_messages),
    }


def main() -> None:
    report = run_evaluation()
    # Checkpoint state contains LangChain messages and is intended for Python
    # recovery tests, not the portable JSON report.
    report.pop("checkpoint_state", None)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
