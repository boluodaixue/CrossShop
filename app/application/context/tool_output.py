"""Explicit L0 model-view contracts and content-addressed raw artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.infrastructure.security.content_filter import sanitize_tool_output


@dataclass(frozen=True)
class ToolOutputContract:
    allowed_fields: tuple[str, ...]
    list_limits: Mapping[str, int] = field(default_factory=dict)
    max_string_chars: int = 512
    max_total_chars: int = 12_000
    default_list_limit: int = 12
    max_mapping_fields: int = 32


TOOL_OUTPUT_CONTRACTS: dict[str, ToolOutputContract] = {
    "product_search_tool": ToolOutputContract(
        allowed_fields=(
            "hits",
            "filtered_out",
            "recall_strategy",
            "rerank_applied",
            "eligible_count",
            "filtered_count",
            "returned_count",
            "filtered_count_by_reason",
            "error",
        ),
        list_limits={"hits": 5, "hits.*.skus": 12, "filtered_out": 5},
        max_total_chars=16_000,
    ),
    "category_insight_tool": ToolOutputContract(
        allowed_fields=("insights", "provenance", "source_boundary", "error"),
    ),
    "web_search_tool": ToolOutputContract(
        allowed_fields=("results", "error"),
        list_limits={"results": 5},
        max_total_chars=8_000,
    ),
    "create_order_tool": ToolOutputContract(
        allowed_fields=(
            "order_id",
            "buyer_id",
            "items",
            "shipping_address",
            "status",
            "created_at",
            "updated_at",
            "total_amount_major",
            "currency",
            "error",
        ),
        list_limits={"items": 10},
        max_total_chars=8_000,
    ),
    "query_order_tool": ToolOutputContract(
        allowed_fields=(
            "order_id",
            "buyer_id",
            "items",
            "shipping_address",
            "status",
            "created_at",
            "updated_at",
            "cancel_reason",
            "total_amount_major",
            "currency",
            "error",
        ),
        list_limits={"items": 10},
        max_total_chars=8_000,
    ),
    "cancel_order_tool": ToolOutputContract(
        allowed_fields=("order_id", "status", "cancel_reason", "updated_at", "error"),
        max_total_chars=4_000,
    ),
    "task_dispatch": ToolOutputContract(
        # SearchAgent dispatch returns the exact structured product-tool result
        # beside its human-facing conclusion. Main and L4 never parse prose.
        allowed_fields=(
            "agent",
            "platform",
            "site_locale",
            "search_result_available",
            "search_args",
            "hits",
            "filtered_out",
            "recall_strategy",
            "agent_output",
            "error",
        ),
        list_limits={"hits": 5, "hits.*.skus": 12, "filtered_out": 5},
        max_string_chars=2_000,
        max_total_chars=16_000,
    ),
    "remember_preference_tool": ToolOutputContract(
        allowed_fields=("saved", "kind", "error"),
        max_total_chars=1_000,
    ),
    "forget_preference_tool": ToolOutputContract(
        allowed_fields=("deleted", "error"),
        max_total_chars=1_000,
    ),
    "TaskCreate": ToolOutputContract(
        allowed_fields=("task", "error"), max_total_chars=4_000
    ),
    "TaskUpdate": ToolOutputContract(
        allowed_fields=("task", "error"), max_total_chars=4_000
    ),
    "TaskList": ToolOutputContract(
        allowed_fields=("tasks", "error"), list_limits={"tasks": 20}
    ),
    "TaskGet": ToolOutputContract(
        allowed_fields=("task", "error"), max_total_chars=4_000
    ),
}


class ToolArtifactStore:
    """Local immutable storage for the exact pre-projection tool response."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, content: str) -> dict[str, Any]:
        raw = content.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        path = self._path(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(raw)
            try:
                temporary.replace(path)
            except FileExistsError:  # pragma: no cover - concurrent identical output
                temporary.unlink(missing_ok=True)
        return {
            "scheme": "globex-artifact-sha256-v1",
            "sha256": digest,
            "bytes": len(raw),
            "media_type": _media_type(content),
        }

    def read(self, reference: Mapping[str, Any]) -> str:
        if reference.get("scheme") != "globex-artifact-sha256-v1":
            raise ValueError("unsupported artifact reference")
        digest = str(reference.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid artifact digest")
        raw = self._path(digest).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("artifact digest mismatch")
        return raw.decode("utf-8")

    def _path(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / f"{digest}.blob"


def _media_type(content: str) -> str:
    try:
        json.loads(content)
    except (TypeError, ValueError):
        return "text/plain; charset=utf-8"
    return "application/json"


def _path_key(path: tuple[str, ...]) -> str:
    return ".".join("*" if item.isdigit() else item for item in path)


def _bounded(
    value: Any, contract: ToolOutputContract, path: tuple[str, ...] = ()
) -> Any:
    if isinstance(value, str):
        return value[: contract.max_string_chars]
    if isinstance(value, Mapping):
        return {
            str(key): _bounded(child, contract, (*path, str(key)))
            for key, child in list(value.items())[: contract.max_mapping_fields]
        }
    if isinstance(value, (list, tuple)):
        limit = contract.list_limits.get(_path_key(path), contract.default_list_limit)
        return [
            _bounded(child, contract, (*path, str(index)))
            for index, child in enumerate(value[:limit])
        ]
    return value


def format_tool_output(
    tool_name: str,
    raw_output: Any,
    *,
    artifact_store: ToolArtifactStore,
    configured_limit: int,
) -> str:
    """Create a bounded model copy while retaining the exact source artifact."""

    if tool_name not in TOOL_OUTPUT_CONTRACTS:
        raise ValueError(f"missing L0 output contract for tool: {tool_name}")
    contract = TOOL_OUTPUT_CONTRACTS[tool_name]
    raw = (
        raw_output
        if isinstance(raw_output, str)
        else json.dumps(
            raw_output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )
    reference = artifact_store.put(raw)
    limit = max(256, min(max(256, configured_limit), contract.max_total_chars))
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        if len(raw) <= limit:
            safe, sanitized = sanitize_tool_output(raw)
            del safe
            return sanitized
        projected: Any = {
            "status": "truncated",
            "preview": raw[:160],
            "artifact_ref": reference,
        }
    else:
        projected = (
            _bounded(
                {key: parsed[key] for key in contract.allowed_fields if key in parsed},
                contract,
            )
            if isinstance(parsed, Mapping)
            else _bounded(parsed, contract)
        )
        if projected != parsed:
            projected = (
                {**projected, "artifact_ref": reference}
                if isinstance(projected, Mapping)
                else {"result": projected, "artifact_ref": reference}
            )
    encoded = json.dumps(
        projected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if len(encoded) > limit:
        encoded = json.dumps(
            {"status": "truncated", "artifact_ref": reference},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    safe, sanitized = sanitize_tool_output(encoded)
    del safe
    return sanitized
