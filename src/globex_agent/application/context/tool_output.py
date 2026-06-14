"""L0 contracts for model-visible tool results and cold artifact storage.

The authoritative tool events are emitted by the underlying tools before this
adapter runs.  This module only controls the copy placed in ``ToolMessage``.
When that copy must omit or bound data, the exact original return value is
stored content-addressably under the configured local data directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool


@dataclass(frozen=True)
class ToolOutputContract:
    """Explicit JSON boundary for one model-authorized tool."""

    allowed_fields: tuple[str, ...]
    list_limits: Mapping[str, int] = field(default_factory=dict)
    max_string_chars: int = 512
    max_total_chars: int = 12_000
    default_list_limit: int = 12
    max_mapping_fields: int = 32


TOOL_OUTPUT_CONTRACTS: dict[str, ToolOutputContract] = {
    "product_search_tool": ToolOutputContract(
        allowed_fields=(
            "tool_call_id",
            "hits",
            "total_candidates",
            "recall_strategy",
            "rerank_applied",
            "requested_top_k",
            "ann_recall_depth",
            "ann_candidate_count",
            "eligible_count",
            "filtered_count",
            "returned_count",
            "is_partial",
            "filtered_count_by_reason",
            "exhaustion_reason",
            "filtered_out",
            "error",
        ),
        list_limits={
            "hits": 5,
            "hits.*.category_path": 8,
            "hits.*.ships_to": 16,
            "hits.*.highlights": 8,
            "hits.*.variants": 12,
            "hits.*.matching_variant_ids": 12,
            "hits.*.warnings": 8,
            "filtered_out": 3,
        },
        max_string_chars=512,
        max_total_chars=16_000,
    ),
    "category_insight_tool": ToolOutputContract(
        allowed_fields=("insights", "provenance", "source_boundary", "error"),
        list_limits={
            "insights.components": 8,
            "insights.bestsellers": 8,
            "insights.attributes": 12,
            "insights.price_tiers": 8,
            "provenance": 8,
        },
        max_total_chars=12_000,
    ),
    "web_search_tool": ToolOutputContract(
        allowed_fields=("results", "error"),
        list_limits={"results": 5},
        max_string_chars=600,
        max_total_chars=8_000,
    ),
    "prepare_order_tool": ToolOutputContract(
        allowed_fields=(
            "confirmation_required",
            "confirmation_id",
            "action",
            "expires_at",
            "items",
            "shipping_summary",
            "merchandise_subtotal_major",
            "freight_major",
            "tariff_major",
            "landed_total_major",
            "total_amount_major",
            "currency",
            "pricing",
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
            "pricing",
            "merchandise_subtotal_major",
            "freight_major",
            "tariff_major",
            "landed_total_major",
            "error",
        ),
        list_limits={"items": 10},
        max_total_chars=8_000,
    ),
    "prepare_cancel_order_tool": ToolOutputContract(
        allowed_fields=(
            "confirmation_required",
            "confirmation_id",
            "action",
            "expires_at",
            "order_id",
            "reason",
            "status",
            "total_amount_major",
            "currency",
            "error",
        ),
        max_total_chars=4_000,
    ),
    "task_dispatch": ToolOutputContract(
        allowed_fields=("dispatches", "error"),
        list_limits={"dispatches": 8},
        max_string_chars=2_000,
        max_total_chars=12_000,
    ),
    "remember_preference_tool": ToolOutputContract(
        allowed_fields=("saved", "kind", "error"),
        max_string_chars=500,
        max_total_chars=1_000,
    ),
}


class ToolArtifactStore:
    """Small local content-addressed store for exact, non-Prompt tool output."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, content: str) -> dict[str, Any]:
        raw = content.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        media_type = _media_type(content)
        path = self._path(digest)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(raw)
            try:
                temporary.replace(path)
            except FileExistsError:  # pragma: no cover - concurrent identical write
                temporary.unlink(missing_ok=True)
        return {
            "scheme": "globex-artifact-sha256-v1",
            "sha256": digest,
            "bytes": len(raw),
            "media_type": media_type,
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
    return ".".join("*" if part.isdigit() else part for part in path)


def _bounded(value: Any, contract: ToolOutputContract, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, str):
        return value[: contract.max_string_chars]
    if isinstance(value, Mapping):
        items = list(value.items())[: contract.max_mapping_fields]
        return {
            str(key): _bounded(child, contract, (*path, str(key)))
            for key, child in items
        }
    if isinstance(value, (list, tuple)):
        limit = contract.list_limits.get(_path_key(path), contract.default_list_limit)
        return [
            _bounded(child, contract, (*path, str(index)))
            for index, child in enumerate(value[:limit])
        ]
    return value


def _project(parsed: Any, contract: ToolOutputContract) -> Any:
    if not isinstance(parsed, Mapping):
        return _bounded(parsed, contract)
    return _bounded(
        {key: parsed[key] for key in contract.allowed_fields if key in parsed},
        contract,
    )


def _minimal_summary(tool_name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"status": "truncated", "preview": str(value)[:240]}
    if tool_name == "product_search_tool":
        hits = value.get("hits") if isinstance(value.get("hits"), list) else []
        return {
            "status": "truncated",
            "returned_count": value.get("returned_count", len(hits)),
            "items": [
                {
                    key: hit.get(key)
                    for key in (
                        "item_id",
                        "title",
                        "price_major",
                        "price_min_major",
                        "price_max_major",
                        "currency",
                        "availability",
                    )
                    if isinstance(hit, Mapping) and key in hit
                }
                for hit in hits[:5]
            ],
        }
    if tool_name == "task_dispatch":
        dispatches = value.get("dispatches") if isinstance(value.get("dispatches"), list) else []
        return {
            "status": "truncated",
            "dispatches": [
                {
                    key: item.get(key)
                    for key in ("subagent_type", "dispatch_id", "thread_id", "error")
                    if isinstance(item, Mapping) and key in item
                }
                for item in dispatches[:8]
            ],
        }
    scalars = {
        str(key): child
        for key, child in value.items()
        if child is None or isinstance(child, (bool, int, float, str))
    }
    return {"status": "truncated", **dict(list(scalars.items())[:12])}


def format_tool_output(
    tool_name: str,
    raw_output: Any,
    *,
    artifact_store: ToolArtifactStore,
    configured_limit: int,
) -> str:
    """Return valid bounded JSON/text and archive exact omitted content."""

    raw = raw_output if isinstance(raw_output, str) else json.dumps(
        raw_output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    contract = TOOL_OUTPUT_CONTRACTS[tool_name]
    limit = max(256, min(max(256, configured_limit), contract.max_total_chars))
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        if len(raw) <= limit:
            return raw
        reference = artifact_store.put(raw)
        return json.dumps(
            {"status": "truncated", "preview": raw[:160], "artifact_ref": reference},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    projected = _project(parsed, contract)
    changed = projected != parsed
    encoded = json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if changed or len(encoded) > limit:
        reference = artifact_store.put(raw)
        if isinstance(projected, Mapping):
            projected = {**projected, "artifact_ref": reference}
        else:
            projected = {"result": projected, "artifact_ref": reference}
        encoded = json.dumps(
            projected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if len(encoded) > limit:
        projected = {**_minimal_summary(tool_name, projected), "artifact_ref": reference}
        encoded = json.dumps(
            projected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if len(encoded) > limit:  # The reference itself remains bounded and recoverable.
        encoded = json.dumps(
            {"status": "truncated", "artifact_ref": reference},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return encoded


def apply_tool_output_contract(
    tool: StructuredTool,
    *,
    artifact_store: ToolArtifactStore,
    configured_limit: int,
) -> StructuredTool:
    """Wrap a tool without changing its input schema or authoritative events."""

    if tool.name not in TOOL_OUTPUT_CONTRACTS:
        raise ValueError(f"missing L0 output contract for tool: {tool.name}")

    async def _contracted_call(**kwargs: Any) -> str:
        if tool.coroutine is None:
            raise TypeError(f"L0 requires an async tool: {tool.name}")
        # The outer ToolNode already validated and injected hidden arguments
        # (notably InjectedToolCallId). Calling the coroutine preserves them;
        # invoking the inner StructuredTool would require the original ToolCall
        # envelope a second time and lose that injection.
        output = await tool.coroutine(**kwargs)
        return format_tool_output(
            tool.name,
            output,
            artifact_store=artifact_store,
            configured_limit=configured_limit,
        )

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_contracted_call,
        return_direct=tool.return_direct,
    )
