from __future__ import annotations

from langchain_core.messages import HumanMessage

from globex_agent.application.context import (
    CACHE_BREAKPOINT_TEXT,
    FrozenSegment,
    assemble_llm_input_messages,
)
from globex_agent.infrastructure.llm import provider_cache_usage_capability
from scripts.eval_context_management_phase_f import run_evaluation


def test_provider_cache_usage_stays_unverified_without_real_response_evidence() -> None:
    capability = provider_cache_usage_capability("qwen3-max")

    assert capability == {
        "model": "qwen3-max",
        "status": "not_verified",
        "provider_response_evidence": False,
        "cache_usage_statistics_enabled": False,
    }
    assert "cached_tokens" not in capability


def test_phase_f_exact_twenty_turn_evaluation_preserves_context_and_audit() -> None:
    report = run_evaluation()
    state = report["checkpoint_state"]

    assert report["turns"] == 20
    assert len(report["model_calls"]) == 40
    assert report["counter"]["estimated"] is False
    assert report["counter"]["soft_limit_tokens"] > 0
    assert all(report["checks"].values())
    assert report["lifecycle"] == {
        "raw_message_count": 86,
        "interaction_units": 20,
        "freeze_cursor": 82,
        "frozen_segment_count": 19,
        "stage_summary_source_count": 18,
        "active_message_count": 4,
    }
    assert report["l4"]["revision"] == 40
    assert report["l4"]["last_search"]["args"]["ship_to"] == "US"
    assert report["l4"]["last_search"]["args"]["price_max_major"] == 500
    assert report["l4"]["order"]["status"] == "CONFIRMED"
    assert len(report["l3_actions"]) == 4
    assert all(
        action["after_tokens"] < action["before_tokens"]
        for action in report["l3_actions"]
    )
    assert len(state["messages"]) == 86
    assert len(state["frozen_segments"]) == 19
    assert len(state["stage_summary"]["source_segment_ids"]) == 18
    assert isinstance(state["messages"][state["freeze_cursor"]], HumanMessage)
    visible_messages = assemble_llm_input_messages(
        messages=state["messages"],
        frozen_segments=[
            FrozenSegment.from_dict(item) for item in state["frozen_segments"]
        ],
        freeze_cursor=state["freeze_cursor"],
        session_context=state["session_context"],
        stage_summary=state["stage_summary"],
    )
    visible = "\n".join(str(message.content) for message in visible_messages)
    assert CACHE_BREAKPOINT_TEXT in visible
    assert visible.count("<stage-summary>") == 1
    assert 'id="frozen-78-82"' in visible
    assert 'id="frozen-0-5"' not in visible
