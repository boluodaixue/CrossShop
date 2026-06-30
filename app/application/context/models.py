"""Serializable L0-L4 context-management records.

The mutable portions deliberately stay JSON-compatible so they can be kept in
``AgentState.middle_context`` and survive AgentScope state persistence without
introducing a second checkpoint or orchestration protocol.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "session-context-v1"
STAGE_SUMMARY_SCHEMA_VERSION = "stage-summary-v1"


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True)
class L4Context:
    """Bounded, traceable projection of session/request/search/order facts.

    ``last_recommendation`` remains optional and is populated only from the
    authoritative structured selection after its product references are
    validated against real hits. Its current display and bounded verified
    display history share the same record; natural-language prose is never
    parsed.
    """

    schema_version: str = SCHEMA_VERSION
    revision: int = 0
    session: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    last_search: dict[str, Any] | None = None
    last_recommendation: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(
            {
                "schema_version": self.schema_version,
                "revision": self.revision,
                "session": self.session,
                "request": self.request,
                "last_search": self.last_search,
                "last_recommendation": self.last_recommendation,
                "order": self.order,
                "updated_at": self.updated_at,
            },
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> L4Context:
        if not raw:
            return cls()
        return cls(
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
            revision=int(raw.get("revision", 0)),
            session=dict(raw.get("session") or {}),
            request=dict(raw.get("request") or {}),
            last_search=dict(raw["last_search"])
            if isinstance(raw.get("last_search"), Mapping)
            else None,
            last_recommendation=dict(raw["last_recommendation"])
            if isinstance(raw.get("last_recommendation"), Mapping)
            else None,
            order=dict(raw["order"]) if isinstance(raw.get("order"), Mapping) else None,
            updated_at=str(raw.get("updated_at", "")),
        )

    def semantic_dict(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("revision", None)
        value.pop("updated_at", None)
        return value


SessionContext = L4Context
SessionContextSnapshot = L4Context


@dataclass(frozen=True)
class FrozenSegment:
    """Immutable deterministic representation of one settled interaction."""

    segment_id: str
    content: str
    source_message_start: int | None = None
    source_message_end: int | None = None
    context_revision: int = 0
    content_hash: str = ""
    segment_type: str = "interaction"
    created_at: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_hash and self.content_hash != digest:
            raise ValueError("frozen segment content hash mismatch")
        object.__setattr__(self, "content_hash", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "content": self.content,
            "source_message_start": self.source_message_start,
            "source_message_end": self.source_message_end,
            "context_revision": self.context_revision,
            "content_hash": self.content_hash,
            "segment_type": self.segment_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FrozenSegment:
        return cls(
            segment_id=str(raw["segment_id"]),
            content=str(raw.get("content", "")),
            source_message_start=raw.get("source_message_start"),
            source_message_end=raw.get("source_message_end"),
            context_revision=int(raw.get("context_revision", 0)),
            content_hash=str(raw.get("content_hash", "")),
            segment_type=str(raw.get("segment_type", "interaction")),
            created_at=str(raw.get("created_at", "")) or now_iso(),
        )


@dataclass(frozen=True)
class StageSummary:
    """Deterministic L3 view with a verifiable source-segment manifest."""

    source_segments: tuple[dict[str, str], ...]
    user_requests: tuple[str, ...]
    tool_facts: tuple[dict[str, Any], ...]
    assistant_outcomes: tuple[str, ...]
    source_message_start: int | None = None
    source_message_end: int | None = None
    context_revision: int = 0
    schema_version: str = STAGE_SUMMARY_SCHEMA_VERSION
    combined_source_hash: str = ""
    summary_hash: str = ""

    def __post_init__(self) -> None:
        combined = hashlib.sha256(
            canonical_json(self.source_segments).encode()
        ).hexdigest()
        if self.combined_source_hash and self.combined_source_hash != combined:
            raise ValueError("stage summary combined source hash mismatch")
        object.__setattr__(self, "combined_source_hash", combined)
        body = {
            "schema_version": self.schema_version,
            "source_segments": self.source_segments,
            "source_message_start": self.source_message_start,
            "source_message_end": self.source_message_end,
            "context_revision": self.context_revision,
            "combined_source_hash": combined,
            "content": {
                "user_requests": self.user_requests,
                "tool_facts": self.tool_facts,
                "assistant_outcomes": self.assistant_outcomes,
            },
        }
        summary_hash = hashlib.sha256(canonical_json(body).encode()).hexdigest()
        if self.summary_hash and self.summary_hash != summary_hash:
            raise ValueError("stage summary content hash mismatch")
        object.__setattr__(self, "summary_hash", summary_hash)

    @property
    def source_segment_ids(self) -> tuple[str, ...]:
        return tuple(item["segment_id"] for item in self.source_segments)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(
            {
                "schema_version": self.schema_version,
                "source_segment_ids": self.source_segment_ids,
                "source_segments": self.source_segments,
                "source_message_start": self.source_message_start,
                "source_message_end": self.source_message_end,
                "context_revision": self.context_revision,
                "combined_source_hash": self.combined_source_hash,
                "content": {
                    "user_requests": self.user_requests,
                    "tool_facts": self.tool_facts,
                    "assistant_outcomes": self.assistant_outcomes,
                },
                "summary_hash": self.summary_hash,
            },
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> StageSummary:
        if raw.get("schema_version") != STAGE_SUMMARY_SCHEMA_VERSION:
            raise ValueError("unsupported stage summary schema")
        content = raw.get("content")
        sources = raw.get("source_segments")
        if not isinstance(content, Mapping) or not isinstance(sources, (list, tuple)):
            raise TypeError("stage summary source and content are required")
        return cls(
            source_segments=tuple(
                {
                    "segment_id": str(item["segment_id"]),
                    "content_hash": str(item["content_hash"]),
                }
                for item in sources
                if isinstance(item, Mapping)
            ),
            user_requests=tuple(
                str(item) for item in content.get("user_requests") or ()
            ),
            tool_facts=tuple(
                dict(item)
                for item in content.get("tool_facts") or ()
                if isinstance(item, Mapping)
            ),
            assistant_outcomes=tuple(
                str(item) for item in content.get("assistant_outcomes") or ()
            ),
            source_message_start=raw.get("source_message_start"),
            source_message_end=raw.get("source_message_end"),
            context_revision=int(raw.get("context_revision", 0)),
            combined_source_hash=str(raw.get("combined_source_hash", "")),
            summary_hash=str(raw.get("summary_hash", "")),
        )


@dataclass(frozen=True)
class LayerTokenCount:
    name: str
    tokens: int
    method: str
    estimated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tokens": self.tokens,
            "method": self.method,
            "estimated": self.estimated,
        }


@dataclass(frozen=True)
class BudgetReport:
    model_context_tokens: int
    reply_reserved_tokens: int
    safety_margin_tokens: int
    layers: dict[str, LayerTokenCount]
    tool_results: dict[str, LayerTokenCount]
    total_input_tokens: int
    available_input_tokens: int
    overflow_tokens: int
    estimated: bool
    counter_method: str
    soft_limit_tokens: int = 0
    decision: str = "ok"
    enforcement_mode: str = "observe_only"

    @property
    def within_budget(self) -> bool:
        return self.overflow_tokens == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_context_tokens": self.model_context_tokens,
            "reply_reserved_tokens": self.reply_reserved_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "layers": {key: value.to_dict() for key, value in self.layers.items()},
            "tool_results": {
                key: value.to_dict() for key, value in self.tool_results.items()
            },
            "total_input_tokens": self.total_input_tokens,
            "available_input_tokens": self.available_input_tokens,
            "overflow_tokens": self.overflow_tokens,
            "estimated": self.estimated,
            "counter_method": self.counter_method,
            "soft_limit_tokens": self.soft_limit_tokens,
            "decision": self.decision,
            "enforcement_mode": self.enforcement_mode,
            "within_budget": self.within_budget,
        }
