"""Batch task dispatch tool that runs LangGraph subagents in parallel."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Annotated

from langchain_core.tools import InjectedToolCallId, tool

from globex_agent.application.agents.identity import ThreadIdentity
from globex_agent.infrastructure.checkpoint import (
    DispatchResultStore,
)
from globex_agent.infrastructure.context import ShoppingContext
from globex_agent.infrastructure.eventbus import TradeEventBus


def build_task_dispatch_tool(
    search_factory,
    trade_factory,
    bus: TradeEventBus,
    *,
    identity: ThreadIdentity | None = None,
    result_store: DispatchResultStore | None = None,
):
    identity = identity or ThreadIdentity()
    if result_store is None:
        raise RuntimeError(
            "task_dispatch requires an explicit completion marker store; "
            "production must not use InMemoryDispatchResultStore"
        )

    @tool
    async def task_dispatch(
        dispatches: list[dict],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """批量调度专家子代理并行执行独立子任务。

        Args:
            dispatches: 数组，每项含 subagent_type（search_agent/trade_agent）与 demands。
        """
        if not dispatches:
            return "[error] dispatches 不能为空"
        session_id = ShoppingContext.current_session_id()
        parent_thread_id = (
            ShoppingContext.current_main_thread_id()
            or identity.main_thread_id(session_id)
        )
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

        async def run_one(index: int, dispatch: dict) -> dict:
            subagent_type = str(dispatch.get("subagent_type", ""))
            demands = str(dispatch.get("demands", ""))
            dispatch_id = identity.dispatch_id(
                parent_thread_id,
                tool_call_id,
                index,
                subagent_type,
                demands,
            )
            thread_id = identity.child_thread_id(
                parent_thread_id,
                subagent_type,
                dispatch_id,
            )
            cached = await result_store.get(dispatch_id)
            if cached is not None:
                cached_result = dict(cached)
                cached_result["cached"] = True
                bus.publish(
                    session_id,
                    "tool.result",
                    {
                        "tool": "task_dispatch",
                        "agent": subagent_type,
                        "dispatch_id": dispatch_id,
                        "thread_id": thread_id,
                        "cached": True,
                    },
                )
                return cached_result
            started_at = datetime.now(timezone.utc).isoformat()
            started_monotonic = time.monotonic()
            bus.publish(
                session_id,
                "agent.dispatch",
                {
                    "agent": subagent_type,
                    "dispatch_id": dispatch_id,
                    "demands": demands,
                    "thread_id": thread_id,
                    "started_at": started_at,
                },
            )
            succeeded = False
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
                succeeded = True
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
                    "dispatch_id": dispatch_id,
                    "thread_id": thread_id,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "elapsed_ms": elapsed_ms,
                },
            )
            result = {
                "subagent_type": subagent_type,
                "dispatch_id": dispatch_id,
                "thread_id": thread_id,
                "output": output,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_ms": elapsed_ms,
            }
            if succeeded:
                await result_store.put(dispatch_id, result)
            return result

        async def _publish_subagent_event(event_type: str, payload: dict) -> None:
            bus.publish(session_id, event_type, payload)

        results = await asyncio.gather(
            *(run_one(index, dispatch) for index, dispatch in enumerate(dispatches)),
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
