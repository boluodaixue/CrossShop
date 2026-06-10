"""FastAPI server and WebSocket event delivery tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import globex_agent.presentation.server as server_module
from globex_agent.domain.queue.ports.task_queue import TaskStatus
from globex_agent.infrastructure.eventbus import TradeEventBus


@pytest.fixture
def test_container(monkeypatch):
    bus = TradeEventBus()
    container = SimpleNamespace(
        bus=bus,
        settings=SimpleNamespace(llm_model="test"),
        db_engine=None,
        cache=SimpleNamespace(enabled=False),
        semantic_cache=SimpleNamespace(enabled=False),
        task_queue=None,
        backplane=None,
    )

    async def startup() -> None:
        return None

    async def shutdown() -> None:
        return None

    container.startup = startup
    container.shutdown = shutdown

    async def build_test_container():
        return container

    monkeypatch.setattr(server_module, "build_container", build_test_container)
    return container


class TestServer:
    def test_health(self, test_container) -> None:
        with TestClient(server_module.app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert response.json()["queue"] == "disabled"

    def test_websocket_receives_session_events(self, test_container) -> None:
        with TestClient(server_module.app) as client, client.websocket_connect(
            "/commerce/events"
        ) as websocket:
            websocket.send_json({"shopping_session_id": "ws-test-session"})
            server_module.app.state.container.bus.publish(
                "ws-test-session",
                "tool.invoke",
                {"tool": "product_search_tool"},
            )
            event = websocket.receive_json()
            assert event["type"] == "tool.invoke"
            assert event["payload"]["tool"] == "product_search_tool"
            assert event["occurred_at"]

    def test_intent_response_has_text_and_compatibility_final_text(self, test_container) -> None:
        async def handle_intent(_intent):
            return SimpleNamespace(
                shopping_session_id="s1",
                final_text="已验证推荐",
                recommended_cards=[{"item_id": "item-1"}],
                verification_status="supported",
            )

        test_container.orchestrator = SimpleNamespace(handle_intent=handle_intent)
        with TestClient(server_module.app) as client:
            response = client.post(
                "/commerce/intents",
                json={"buyer_id": "b1", "raw_query": "推荐商品"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "已验证推荐"
        assert body["final_text"] == body["text"]
        assert body["recommended_cards"] == [{"item_id": "item-1"}]
        assert body["verification_status"] == "supported"

    def test_async_task_query_has_text(self, test_container) -> None:
        class Queue:
            async def get_status(self, _task_id):
                return TaskStatus(
                    task_id="task-1",
                    state="done",
                    text="异步已验证",
                    recommended_cards=[],
                    verification_status="supported",
                )

        test_container.task_queue = Queue()
        with TestClient(server_module.app) as client:
            response = client.get("/commerce/tasks/task-1")
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "异步已验证"
        assert body["final_text"] == "异步已验证"
