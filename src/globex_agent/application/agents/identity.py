"""Stable, privacy-preserving LangGraph thread identities."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass

IDENTITY_VERSION = "v1"
ROOT_CHECKPOINT_NS = ""


def _safe_component(value: str, *, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]", "-", value).strip("-")
    return value or fallback


@dataclass(frozen=True)
class ThreadIdentity:
    """Create stable IDs without placing session or query data in Redis keys."""

    environment: str = "local"
    hmac_key: str = "globex-local-thread-id-v1"
    version: str = IDENTITY_VERSION

    @property
    def _environment(self) -> str:
        return _safe_component(self.environment, fallback="local").lower()

    def digest(self, value: str) -> str:
        key = self.hmac_key.encode("utf-8")
        return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:32]

    def main_thread_id(self, shopping_session_id: str) -> str:
        return (
            f"globex:{self.version}:{self._environment}:main:"
            f"{self.digest('session:' + shopping_session_id)}"
        )

    def dispatch_id(
        self,
        parent_thread_id: str,
        tool_call_id: str | None,
        index: int,
        subagent_type: str,
        demands: str,
    ) -> str:
        """Derive one stable ID for a dispatch item.

        A LangGraph tool call ID is preferred because it is part of the parent
        checkpoint. Direct tool invocations without one use the canonical item
        payload and index, which is deterministic for the same parent replay.
        """

        canonical = json.dumps(
            {
                "parent": parent_thread_id,
                "tool_call_id": tool_call_id or "runtime",
                "index": index,
                "subagent_type": subagent_type,
                "demands": demands,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"dispatch-{self.digest('dispatch:' + canonical)}"

    def child_thread_id(self, parent_thread_id: str, role: str, dispatch_id: str) -> str:
        role_component = _safe_component(role, fallback="agent").lower()
        parent_digest = self.digest("parent:" + parent_thread_id)
        dispatch_digest = self.digest("dispatch-id:" + dispatch_id)
        return (
            f"globex:{self.version}:{self._environment}:{role_component}:"
            f"parent-{parent_digest}:dispatch-{dispatch_digest}"
        )


def graph_config(thread_id: str) -> dict:
    """Return the application-owned root graph configuration."""

    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": ROOT_CHECKPOINT_NS,
        },
        "recursion_limit": 30,
    }
