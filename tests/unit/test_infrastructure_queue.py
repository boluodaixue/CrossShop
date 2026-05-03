"""Redis Stream queue and event backplane behavior tests."""

from __future__ import annotations

import asyncio
import json

from globex_agent.domain.queue.ports.task_queue import IntentTask, TaskStatus
from globex_agent.infrastructure.eventbus import TradeEvent, TradeEventBus
from globex_agent.infrastructure.queue.redis_stream_queue import (
    RedisEventBackplane,
    RedisStreamTaskQueue,
)


class FakeStreamClient:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict]] = []
        self.pending: dict[str, int] = {}
        self.acked: list[str] = []
        self.dead: list[dict] = []
        self.kv: dict[str, str] = {}
        self.published: list[tuple[str, str]] = []
        self._seq = 0

    async def xgroup_create(self, *_args, **_kwargs):
        return True

    async def xadd(self, stream: str, fields: dict):
        self._seq += 1
        message_id = f"{self._seq}-0"
        if stream.endswith(":dead"):
            self.dead.append(fields)
        else:
            self.entries.append((message_id, fields))
        return message_id

    async def xack(self, _stream, _group, message_id):
        self.acked.append(message_id)
        self.pending.pop(message_id, None)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return False
        self.kv[key] = value
        return True

    async def get(self, key):
        return self.kv.get(key)

    async def publish(self, channel, data):
        self.published.append((channel, data))


def _task() -> IntentTask:
    return IntentTask(
        task_id="task-1",
        shopping_session_id="s1",
        buyer_id="b1",
        locale="zh-CN",
        currency="CNY",
        raw_query="露营灯推荐",
    )


class TestQueue:
    async def test_enqueue_and_status_roundtrip(self) -> None:
        client = FakeStreamClient()
        queue = RedisStreamTaskQueue(client)
        await queue.enqueue(_task())
        assert len(client.entries) == 1
        await queue.set_status(TaskStatus(task_id="task-1", state="done", final_text="ok"))
        status = await queue.get_status("task-1")
        assert status is not None and status.final_text == "ok"

    async def test_unparsable_payload_goes_to_dead_letter(self) -> None:
        client = FakeStreamClient()
        queue = RedisStreamTaskQueue(client)

        async def handler(_task: IntentTask) -> None:
            raise AssertionError("handler must not run")

        await queue._handle_one(
            "9-0",
            {"payload": "{not-json"},
            handler,
            max_deliveries=3,
        )
        assert len(client.dead) == 1
        assert "9-0" in client.acked


class TestBackplane:
    async def test_publish_broadcasts_with_origin(self) -> None:
        client = FakeStreamClient()
        bus = TradeEventBus()
        bus.attach_backplane(RedisEventBackplane(client))
        bus.publish("s1", "tool.invoke", {"tool": "product_search_tool"})
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert client.published
        channel, data = client.published[0]
        assert channel == "globex:events:s1"
        envelope = json.loads(data)
        assert envelope["event"]["type"] == "tool.invoke"
        assert envelope["origin"]

    async def test_deliver_local_does_not_rebroadcast(self) -> None:
        client = FakeStreamClient()
        bus = TradeEventBus()
        bus.attach_backplane(RedisEventBackplane(client))
        queue = bus.subscribe("s1")
        bus.deliver_local(
            TradeEvent(
                shopping_session_id="s1",
                type="token.delta",
                payload={"token": "露"},
                occurred_at="2026-01-01T00:00:00Z",
            )
        )
        await asyncio.sleep(0)
        assert queue.get_nowait().payload == {"token": "露"}
        assert client.published == []
