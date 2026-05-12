"""Batch task dispatch tool that runs LangGraph subagents in parallel."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from uuid import uuid4

from langchain_core.tools import tool

from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def build_task_dispatch_tool(
    search_factory,
    trade_factory,
    bus: TradeEventBus,
):
    @tool
    async def task_dispatch(dispatches: list[dict]) -> str:
        """批量调度专家子代理并行执行独立子任务。

        Args:
            dispatches: 数组，每项含 subagent_type（search_agent/trade_agent）与 demands。
        """
        if not dispatches:
            return "[error] dispatches 不能为空"
        session_id = ShoppingContext.current_session_id()
        bus.publish(
            session_id,
            "plan.update",
            {
                "tasks": [
                    {
                        "subject": str(dispatch.get("subagent_type", "unknown")),
                        "state": "queued",
                        "demands": str(dispatch.get("demands", "")),
                    }
                    for dispatch in dispatches
                ]
            },
        )

        async def run_one(dispatch: dict) -> dict:
            subagent_type = str(dispatch.get("subagent_type", ""))
            demands = str(dispatch.get("demands", ""))
            thread_id = f"dispatch-{uuid4().hex[:12]}"
            started_at = datetime.now(timezone.utc).isoformat()
            started_monotonic = time.monotonic()
            bus.publish(
                session_id,
                "agent.dispatch",
                {
                    "agent": subagent_type,
                    "demands": demands,
                    "thread_id": thread_id,
                    "started_at": started_at,
                },
            )
            try:
                if subagent_type == "search_agent":
                    worker = search_factory.build()
                elif subagent_type == "trade_agent":
                    worker = trade_factory.build()
                else:
                    raise ValueError(f"未知 subagent_type：{subagent_type}")
                output = await worker.reply(
                    demands,
                    thread_id=thread_id,
                    event_sink=_publish_subagent_event,
                )
            except Exception as err:  # noqa: BLE001 - subagent failure must not kill others
                output = f"[error] {err}"
            finished_at = datetime.now(timezone.utc).isoformat()
            elapsed_ms = round((time.monotonic() - started_monotonic) * 1000)
            bus.publish(
                session_id,
                "tool.result",
                {
                    "tool": "task_dispatch",
                    "agent": subagent_type,
                    "thread_id": thread_id,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_ms": elapsed_ms,
                },
            )
            return {
                "subagent_type": subagent_type,
                "thread_id": thread_id,
                "output": output,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_ms": elapsed_ms,
            }

        async def _publish_subagent_event(event_type: str, payload: dict) -> None:
            bus.publish(session_id, event_type, payload)

        results = await asyncio.gather(
            *(run_one(dispatch) for dispatch in dispatches),
            return_exceptions=True,
        )
        payload = [
            (
                {"subagent_type": "unknown", "error": str(result)}
                if isinstance(result, BaseException)
                else result
            )
            for result in results
        ]
        return json.dumps({"dispatches": payload}, ensure_ascii=False)

    return task_dispatch
