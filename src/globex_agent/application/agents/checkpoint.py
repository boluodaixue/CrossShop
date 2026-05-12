"""JSON-friendly checkpoint persistence for LangGraph's in-memory saver.

The application currently uses ``InMemorySaver`` because the lightweight local
setup should not require a service database.  Restart recovery therefore needs
an explicit bridge: serialize the latest checkpoint into JSON, write it through
``SessionStore``, and restore it into a fresh ``InMemorySaver`` on startup.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


def _encode_typed(value: tuple[str, bytes]) -> dict[str, str]:
    type_name, payload = value
    return {
        "type": type_name,
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _decode_typed(value: dict[str, str]) -> tuple[str, bytes]:
    return (
        value["type"],
        base64.b64decode(value["data"].encode("ascii")),
    )


def serialize_checkpoint(saver: InMemorySaver, thread_id: str) -> str:
    """Serialize the latest checkpoint for one thread to a JSON string."""

    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    latest = saver.get_tuple(config)
    if latest is None:
        return "{}"

    checkpoint = latest.checkpoint
    checkpoint_without_values = {
        key: value
        for key, value in checkpoint.items()
        if key != "channel_values"
    }
    channel_values = checkpoint.get("channel_values", {})
    payload = {
        "checkpoint": _encode_typed(
            saver.serde.dumps_typed(checkpoint_without_values)
        ),
        "channel_values": {
            key: _encode_typed(saver.serde.dumps_typed(value))
            for key, value in channel_values.items()
        },
        "metadata": _encode_typed(saver.serde.dumps_typed(latest.metadata)),
        "parent_config": latest.parent_config,
        "pending_writes": [
            {
                "id": write_id,
                "channel": channel,
                "value": _encode_typed(saver.serde.dumps_typed(value)),
            }
            for write_id, channel, value in latest.pending_writes
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def restore_checkpoint(saver: InMemorySaver, state_json: str, thread_id: str) -> None:
    """Restore a serialized checkpoint into a fresh in-memory saver."""

    payload = json.loads(state_json)
    if not payload:
        return

    checkpoint_without_values = saver.serde.loads_typed(
        _decode_typed(payload["checkpoint"])
    )
    channel_values = {
        key: saver.serde.loads_typed(_decode_typed(value))
        for key, value in payload["channel_values"].items()
    }
    checkpoint = {
        **checkpoint_without_values,
        "channel_values": channel_values,
    }
    metadata = saver.serde.loads_typed(_decode_typed(payload["metadata"]))
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    restored_config = saver.put(
        config,
        checkpoint,
        metadata,
        checkpoint.get("channel_versions", {}),
    )
    for write in payload.get("pending_writes", []):
        value = saver.serde.loads_typed(_decode_typed(write["value"]))
        saver.put_writes(
            restored_config,
            [(write["channel"], value)],
            write["id"],
        )


def compress_checkpoint(
    saver: InMemorySaver,
    thread_id: str,
    max_messages: int,
) -> dict[str, int] | None:
    """Trim the oldest non-system messages while keeping recent context."""

    if max_messages < 2:
        raise ValueError("max_messages must be at least 2")
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    latest = saver.get_tuple(config)
    if latest is None:
        return None

    messages = latest.checkpoint.get("channel_values", {}).get("messages")
    if not isinstance(messages, list) or len(messages) <= max_messages:
        return None

    keep_head: list[Any] = []
    if getattr(messages[0], "type", None) == "system":
        keep_head = [messages[0]]
    tail_size = max(1, max_messages - len(keep_head))
    kept = [*keep_head, *messages[-tail_size:]]
    checkpoint = latest.checkpoint.copy()
    checkpoint["channel_values"] = {
        **latest.checkpoint.get("channel_values", {}),
        "messages": kept,
    }
    saver.put(
        config,
        checkpoint,
        latest.metadata,
        checkpoint.get("channel_versions", {}),
    )
    return {
        "removed": len(messages) - len(kept),
        "kept": len(kept),
    }
