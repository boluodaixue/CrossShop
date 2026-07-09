"""AgentScope-native custom context lifecycle middleware.

This is the only runtime bridge for the approved L0-L4 extension. It stores
JSON-compatible state in ``AgentState.middle_context`` and never introduces a
second Agent, orchestrator, checkpoint, or tool protocol.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentscope.message import Msg, TextBlock
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase
from agentscope.tool import ToolResponse

from app.application.context import (
    TOOL_OUTPUT_CONTRACTS,
    ContextBudgetExceeded,
    ContextBudgetPolicy,
    L4Context,
    ToolArtifactStore,
    advance_context_state,
    format_tool_output,
    reduce_l4,
)
from app.infrastructure.context import ShoppingContext
from app.infrastructure.tracing import trace_span

CONTEXT_NAMESPACE = "globex_context_v1"
_MAX_ARTIFACT_REFERENCES = 100


def _prompt_breakdown_attributes(
    report: Mapping[str, Any],
    *,
    agent_name: str,
) -> dict[str, Any]:
    """Return privacy-safe, estimated prompt-layer counters for one model call."""

    layers = report.get("layers")
    layers = layers if isinstance(layers, Mapping) else {}

    def _tokens(name: str) -> int:
        layer = layers.get(name)
        if not isinstance(layer, Mapping):
            return 0
        try:
            return max(0, int(layer.get("tokens") or 0))
        except (TypeError, ValueError):
            return 0

    tool_results = report.get("tool_results")
    tool_results = tool_results if isinstance(tool_results, Mapping) else {}
    tool_result_tokens = 0
    for value in tool_results.values():
        if not isinstance(value, Mapping):
            continue
        try:
            tool_result_tokens += max(0, int(value.get("tokens") or 0))
        except (TypeError, ValueError):
            continue

    return {
        "globex.prompt.agent": agent_name,
        "globex.prompt.system_tokens": _tokens("fixed_prompt"),
        "globex.prompt.tool_schema_tokens": _tokens("tool_schemas"),
        "globex.prompt.recent_history_tokens": (
            _tokens("l4_candidate") + _tokens("active") + _tokens("current")
        ),
        "globex.prompt.frozen_context_tokens": _tokens("history"),
        "globex.prompt.tool_result_tokens": tool_result_tokens,
        "globex.prompt.total_estimated_tokens": max(
            0,
            int(report.get("total_input_tokens") or 0),
        ),
        "globex.prompt.stable_prefix_estimated_tokens": max(
            0,
            int(report.get("stable_prefix_estimated_tokens") or 0),
        ),
        "globex.prompt.stable_prefix_sha256": str(
            report.get("stable_prefix_sha256") or "",
        ),
        "globex.prompt.estimated": bool(report.get("estimated", True)),
        "globex.prompt.counter_method": str(report.get("counter_method") or "unknown"),
        "globex.prompt.budget_decision": str(report.get("decision") or "unknown"),
    }


class ContextOwnershipError(RuntimeError):
    """Raised before model/tool work when a session is reused by another buyer."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _namespace(agent: Any) -> dict[str, Any]:
    value = agent.state.middle_context.get(CONTEXT_NAMESPACE)
    if not isinstance(value, dict):
        value = {}
        agent.state.middle_context[CONTEXT_NAMESPACE] = value
    return value


def _raw_tool_response(response: ToolResponse) -> str:
    if len(response.content) == 1 and isinstance(response.content[0], TextBlock):
        return response.content[0].text
    return json.dumps(
        [block.model_dump(mode="json") for block in response.content],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _intent_from_inputs(inputs: Any) -> dict[str, Any] | None:
    messages = (
        inputs
        if isinstance(inputs, list)
        else [inputs]
        if isinstance(inputs, Msg)
        else []
    )
    user_messages = [
        message
        for message in messages
        if isinstance(message, Msg) and message.role == "user"
    ]
    if not user_messages:
        return None
    snapshot = ShoppingContext.current()
    latest = user_messages[-1]
    return {
        "shopping_session_id": snapshot.shopping_session_id if snapshot else "",
        "buyer_id": snapshot.buyer_id if snapshot else latest.name,
        "locale": snapshot.locale if snapshot else "",
        "currency": snapshot.currency if snapshot else "",
        "raw_query": latest.get_text_content() or "",
    }


class ContextLifecycleMiddleware(MiddlewareBase):
    """L0-L4 lifecycle bound to AgentScope's native middleware hooks."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        model_context_tokens: int,
        tool_result_limit: int,
        reply_reserved_tokens: int = 2_048,
        safety_margin_tokens: int = 1_024,
        soft_limit_ratio: float = 0.70,
        budget_mode: str = "observe_only",
    ) -> None:
        if not 0 < soft_limit_ratio <= 1:
            raise ValueError("soft_limit_ratio must be within (0, 1]")
        self._artifacts = ToolArtifactStore(artifact_root)
        self._tool_result_limit = tool_result_limit
        self._policy = ContextBudgetPolicy(
            model_context_tokens=model_context_tokens,
            reply_reserved_tokens=reply_reserved_tokens,
            safety_margin_tokens=safety_margin_tokens,
            soft_limit_tokens=int(model_context_tokens * soft_limit_ratio),
            mode=budget_mode,
        )

    async def get_middleware_key(self) -> str:
        return CONTEXT_NAMESPACE

    async def on_reply(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Any,
    ) -> AsyncGenerator[Any, None]:
        namespace = _namespace(agent)
        intent = _intent_from_inputs(input_kwargs.get("inputs"))
        if intent is not None:
            previous = L4Context.from_dict(namespace.get("session_context"))
            previous_buyer = previous.session.get("buyer_id")
            incoming_buyer = intent.get("buyer_id")
            if previous_buyer and incoming_buyer and previous_buyer != incoming_buyer:
                raise ContextOwnershipError(
                    "shopping_session_id already belongs to another buyer",
                )
            namespace["current_intent"] = intent
            namespace["current_turn_events"] = []
            # Request-start facts survive even if the model call later fails.
            namespace["session_context"] = reduce_l4(
                intent,
                previous=previous,
            ).to_dict()

        final_msg: Msg | None = None
        try:
            async for item in next_handler(**input_kwargs):
                if isinstance(item, Msg):
                    final_msg = item
                yield item
        finally:
            # Completed tool events are reduced even when a later model call
            # fails; error/denied/interrupted responses are ignored by L4.
            # on_model_call replaces the namespace mapping with the assembled
            # context view. Re-acquire it here so tool events collected after
            # that replacement are reduced into the live AgentState rather
            # than an orphaned pre-model dictionary.
            namespace = _namespace(agent)
            active_intent = namespace.get("current_intent") or {}
            events = list(namespace.pop("current_turn_events", []))
            if final_msg is not None:
                events.append(
                    {
                        "type": "final.result",
                        "payload": {
                            "text": final_msg.get_text_content() or "",
                            "structured_output": final_msg.structured_output,
                        },
                        "occurred_at": _now(),
                    },
                )
            namespace["session_context"] = reduce_l4(
                active_intent,
                events,
                namespace.get("session_context"),
            ).to_dict()

    async def on_acting(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Any,
    ) -> AsyncGenerator[Any, None]:
        namespace = _namespace(agent)
        call = input_kwargs["tool_call"]
        try:
            args: Any = json.loads(call.input)
        except (TypeError, json.JSONDecodeError):
            args = call.input
        namespace.setdefault("current_turn_events", []).append(
            {
                "type": "tool.invoke",
                "payload": {
                    "tool": call.name,
                    "tool_call_id": call.id,
                    "args": args,
                },
                "occurred_at": _now(),
            },
        )

        async for chunk in next_handler(**input_kwargs):
            if not isinstance(chunk, ToolResponse):
                yield chunk
                continue
            raw = _raw_tool_response(chunk)
            reference = self._artifacts.put(raw)
            record = {
                "tool": call.name,
                "tool_call_id": call.id,
                "state": str(getattr(chunk.state, "value", chunk.state)),
                "artifact_ref": reference,
                "occurred_at": _now(),
            }
            artifacts = namespace.setdefault("tool_artifacts", [])
            artifacts.append(record)
            if len(artifacts) > _MAX_ARTIFACT_REFERENCES:
                del artifacts[:-_MAX_ARTIFACT_REFERENCES]

            is_error = str(getattr(chunk.state, "value", chunk.state)) != "success"
            namespace.setdefault("current_turn_events", []).append(
                {
                    "type": "tool.result",
                    "payload": {
                        "tool": call.name,
                        "tool_call_id": call.id,
                        "raw_output": raw,
                        "artifact_ref": reference,
                        "error": is_error,
                    },
                    "occurred_at": _now(),
                },
            )
            if call.name in TOOL_OUTPUT_CONTRACTS:
                bounded = format_tool_output(
                    call.name,
                    raw,
                    artifact_store=self._artifacts,
                    configured_limit=self._tool_result_limit,
                )
                metadata = {**chunk.metadata, "globex_artifact_ref": reference}
                chunk = chunk.model_copy(
                    update={
                        "content": [TextBlock(type="text", text=bounded)],
                        "metadata": metadata,
                    },
                )
            yield chunk

    async def on_model_call(
        self, agent: Any, input_kwargs: dict, next_handler: Any
    ) -> Any:
        namespace = _namespace(agent)
        messages = list(input_kwargs.get("messages") or [])
        raw_context = list(agent.state.context)
        fixed_count = max(0, len(messages) - len(raw_context))
        fixed_messages = messages[:fixed_count]
        unfinished = bool(agent.state.get_unfinished_tool_calls(agent.name))
        awaiting = bool(agent.state.get_awaiting_tool_calls(agent.name))
        updated, model_view = advance_context_state(
            context_messages=raw_context,
            namespace=namespace,
            fixed_messages=fixed_messages,
            tool_schemas=input_kwargs.get("tools") or [],
            policy=self._policy,
            has_pending=unfinished,
            has_interrupt=awaiting,
        )
        agent.state.middle_context[CONTEXT_NAMESPACE] = updated
        model = input_kwargs.get("current_model")
        if model is not None:
            provider_count = await model.count_tokens(
                model_view, input_kwargs.get("tools") or []
            )
            provider_counter_estimated = (
                type(model).count_tokens is ChatModelBase.count_tokens
            )
            report = dict(updated.get("budget_report") or {})
            report.update(
                {
                    "provider_input_tokens": provider_count,
                    # AgentScope's default counter is explicitly a byte estimate.
                    "provider_counter_method": "agentscope.model.count_tokens",
                    "provider_counter_estimated": provider_counter_estimated,
                },
            )
            updated["budget_report"] = report
            available = (
                self._policy.model_context_tokens
                - self._policy.reply_reserved_tokens
                - self._policy.safety_margin_tokens
            )
            if (
                self._policy.mode == "enforce"
                and not report["provider_counter_estimated"]
                and provider_count > available
            ):
                raise ContextBudgetExceeded(
                    "provider_hard_overflow", _report_from_mapping(report)
                )
        breakdown = _prompt_breakdown_attributes(
            updated.get("budget_report") or {},
            agent_name=str(getattr(agent, "name", "unknown")),
        )
        with trace_span("globex.prompt.breakdown", breakdown):
            return await next_handler(**{**input_kwargs, "messages": model_view})

    async def on_compress_context(
        self, agent: Any, input_kwargs: dict, next_handler: Any
    ) -> None:
        """Keep raw AgentScope context immutable; L2/L3 govern the model view."""

        del input_kwargs, next_handler
        namespace = _namespace(agent)
        namespace["builtin_compression_skipped"] = (
            int(namespace.get("builtin_compression_skipped", 0)) + 1
        )


def _report_from_mapping(raw: Mapping[str, Any]):
    """This path is reserved for future exact provider counters."""

    from app.application.context.models import BudgetReport, LayerTokenCount

    return BudgetReport(
        model_context_tokens=int(raw.get("model_context_tokens", 0)),
        reply_reserved_tokens=int(raw.get("reply_reserved_tokens", 0)),
        safety_margin_tokens=int(raw.get("safety_margin_tokens", 0)),
        layers={
            key: LayerTokenCount(**value)
            for key, value in (raw.get("layers") or {}).items()
        },
        tool_results={
            key: LayerTokenCount(**value)
            for key, value in (raw.get("tool_results") or {}).items()
        },
        total_input_tokens=int(raw.get("total_input_tokens", 0)),
        available_input_tokens=int(raw.get("available_input_tokens", 0)),
        overflow_tokens=int(raw.get("overflow_tokens", 0)),
        estimated=bool(raw.get("estimated", True)),
        counter_method=str(raw.get("counter_method", "unknown")),
        soft_limit_tokens=int(raw.get("soft_limit_tokens", 0)),
        decision=str(raw.get("decision", "provider_hard_overflow")),
        enforcement_mode=str(raw.get("enforcement_mode", "enforce")),
    )
