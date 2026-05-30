"""FastAPI server and WebSocket event delivery tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import globex_agent.presentation.server as server_module
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
