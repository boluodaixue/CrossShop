from __future__ import annotations

import asyncio
import json

import pytest

import scripts.eval_flow_queries as flow_queries


class _Connection:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def send(self, value: str) -> None:
        self.sent.append(value)

    async def recv(self) -> str:
        return json.dumps({"type": "final.result", "payload": {"text": "完成"}})


@pytest.mark.asyncio
async def test_capture_uses_async_websocket_handshake_before_events(monkeypatch) -> None:
    connection = _Connection()

    def fake_connect(*_args, **_kwargs):
        return connection

    monkeypatch.setattr(flow_queries, "websocket_connect", fake_connect)
    ready = asyncio.Event()
    done = asyncio.Event()
    stop = asyncio.Event()
    errors: list[str] = []
    events, error = await flow_queries._capture_events(
        "http://127.0.0.1:8000", "session-1", ready, done, errors, stop
    )
    assert ready.is_set()
    assert done.is_set()
    assert error is None
    assert errors == []
    assert json.loads(connection.sent[0]) == {"shopping_session_id": "session-1"}
    assert events[0]["type"] == "final.result"


@pytest.mark.asyncio
async def test_capture_failure_is_explicit_and_not_empty_success(monkeypatch) -> None:
    def fake_connect(*_args, **_kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(flow_queries, "websocket_connect", fake_connect)
    ready = asyncio.Event()
    done = asyncio.Event()
    stop = asyncio.Event()
    errors: list[str] = []
    events, error = await flow_queries._capture_events(
        "http://127.0.0.1:8000", "session-1", ready, done, errors, stop
    )
    assert events == []
    assert not ready.is_set()
    assert done.is_set()
    assert error == "OSError: connection refused"
    assert errors == [error]
