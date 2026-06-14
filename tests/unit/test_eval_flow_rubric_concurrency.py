"""Tests for bounded, per-case concurrency in the Final Judge evaluator."""

import asyncio
import sys

import pytest

from scripts import eval_flow_rubric


def _items(*case_ids: str) -> list[dict[str, str]]:
    return [{"case_id": case_id, "query": case_id} for case_id in case_ids]


def _completed_judge() -> dict:
    return {
        "evaluation_status": "completed",
        "quality_pass": True,
        "final_score": 1.0,
    }


@pytest.mark.asyncio
async def test_evaluate_items_is_bounded_concurrent_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(eval_flow_rubric, "_flow_summary", lambda item: {})
    active = 0
    maximum_active = 0
    events: list[tuple[str, str]] = []

    async def fake_rubric(client, item):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        events.append(("rubric_start", item["case_id"]))
        await asyncio.sleep(0.03 if item["case_id"] == "case-1" else 0.01)
        events.append(("rubric_end", item["case_id"]))
        return {"evaluation_status": "completed", "rubric": {"case": item["case_id"]}}

    async def fake_final(client, item, rubric):
        nonlocal active
        assert events[-1] == ("rubric_end", item["case_id"])
        await asyncio.sleep(0.01)
        active -= 1
        return _completed_judge()

    monkeypatch.setattr(eval_flow_rubric, "call_rubric_generator", fake_rubric)
    monkeypatch.setattr(eval_flow_rubric, "call_final_judge", fake_final)

    results = await eval_flow_rubric.evaluate_items(
        _items("case-1", "case-2", "case-3", "case-4")
    )

    assert 1 < maximum_active <= 2
    assert [row["case_id"] for row in results] == [
        "case-1",
        "case-2",
        "case-3",
        "case-4",
    ]
    progress = capsys.readouterr().out.splitlines()
    assert progress[0].startswith("[2/4] case-2 ")
    for index, case_id in enumerate(("case-1", "case-2", "case-3", "case-4"), 1):
        assert any(f"[{index}/4] {case_id} " in line for line in progress)


@pytest.mark.asyncio
async def test_evaluate_items_keeps_each_item_isolated_when_a_stage_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_flow_rubric, "_flow_summary", lambda item: {})

    async def fake_rubric(client, item):
        if item["case_id"] == "case-2":
            raise ValueError("rubric failed")
        return {"evaluation_status": "completed", "rubric": {"case": item["case_id"]}}

    async def fake_final(client, item, rubric):
        return _completed_judge()

    monkeypatch.setattr(eval_flow_rubric, "call_rubric_generator", fake_rubric)
    monkeypatch.setattr(eval_flow_rubric, "call_final_judge", fake_final)

    results = await eval_flow_rubric.evaluate_items(
        _items("case-1", "case-2", "case-3"), concurrency=3
    )

    assert [row["case_id"] for row in results] == ["case-1", "case-2", "case-3"]
    assert results[0]["evaluation_status"] == "completed"
    assert results[1]["evaluation_status"] == "inconclusive"
    assert results[1]["final_judge"]["failure_stage"] == "rubric_generator"
    assert results[2]["evaluation_status"] == "completed"


@pytest.mark.asyncio
async def test_concurrency_one_remains_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_flow_rubric, "_flow_summary", lambda item: {})
    active = 0
    maximum_active = 0

    async def fake_rubric(client, item):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        return {"evaluation_status": "completed", "rubric": {"case": item["case_id"]}}

    async def fake_final(client, item, rubric):
        nonlocal active
        await asyncio.sleep(0)
        active -= 1
        return _completed_judge()

    monkeypatch.setattr(eval_flow_rubric, "call_rubric_generator", fake_rubric)
    monkeypatch.setattr(eval_flow_rubric, "call_final_judge", fake_final)

    results = await eval_flow_rubric.evaluate_items(
        _items("case-1", "case-2"), concurrency=1
    )

    assert maximum_active == 1
    assert [row["case_id"] for row in results] == ["case-1", "case-2"]


def test_concurrency_cli_is_positive_and_passed_to_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_main_async(args):
        captured.update(vars(args))
        return None

    monkeypatch.setattr(eval_flow_rubric, "main_async", fake_main_async)
    monkeypatch.setattr(eval_flow_rubric.asyncio, "run", lambda coroutine: None)
    monkeypatch.setattr(sys, "argv", ["eval_flow_rubric.py"])

    eval_flow_rubric.main()

    assert captured["concurrency"] == 2


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_concurrency_cli_rejects_non_positive_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_flow_rubric.py", "--concurrency", value])

    with pytest.raises(SystemExit) as error:
        eval_flow_rubric.main()

    assert error.value.code == 2


@pytest.mark.asyncio
async def test_evaluate_items_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        await eval_flow_rubric.evaluate_items([], concurrency=0)
