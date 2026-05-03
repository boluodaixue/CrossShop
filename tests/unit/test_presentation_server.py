"""FastAPI server and WebSocket event delivery tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from globex_agent.presentation.server import app


class TestServer:
    def test_health(self) -> None:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert response.json()["queue"] == "disabled"

    def test_websocket_receives_session_events(self) -> None:
        with TestClient(app) as client, client.websocket_connect(
            "/commerce/events"
        ) as websocket:
            websocket.send_json({"shopping_session_id": "ws-test-session"})
            app.state.container.bus.publish(
                "ws-test-session",
                "tool.invoke",
                {"tool": "product_search_tool"},
            )
            event = websocket.receive_json()
            assert event["type"] == "tool.invoke"
            assert event["payload"]["tool"] == "product_search_tool"
            assert event["occurred_at"]
