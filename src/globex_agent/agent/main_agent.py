"""Stage-three single AgentLoop assembled with LangGraph."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from globex_agent.agent.llm import get_llm
from globex_agent.agent.models import AgentMessageTrace, SingleAgentResult
from globex_agent.agent.prompts import get_system_prompt
from globex_agent.agent.runtime import AgentRuntime, agent_runtime_scope
from globex_agent.agent.tool_registry import FULL_TOOL_SET, TERMINAL_TOOLS
from globex_agent.catalog import LocalCatalog
from globex_agent.domain import SearchRequest, UserProfile
from globex_agent.infrastructure.recall import (
    ListingLanguage,
    PairReranker,
    PartitionedSearchBackendRouter,
    SearchBackend,
)

MAIN_AGENT_MAX_ITERATIONS = 30
MAIN_AGENT_TIMEOUT_SEC = 300
CHECKPOINTER = InMemorySaver()


def build_main_agent(model: BaseChatModel, prompt: str):
    """Build the chapter-02 ReAct AgentLoop with stage-three tools."""

    return create_react_agent(
        model=model,
        tools=FULL_TOOL_SET,
        prompt=prompt,
        checkpointer=CHECKPOINTER,
    )


async def run_agent(
    query: str,
    catalog: LocalCatalog,
    request: SearchRequest | None = None,
    profile: UserProfile | None = None,
    *,
    thread_id: str | None = None,
    model: BaseChatModel | None = None,
    parse_request_with_llm: bool = False,
    product_search_backend: SearchBackend | None = None,
    product_search_router: PartitionedSearchBackendRouter | None = None,
    product_reranker: PairReranker | None = None,
    listing_languages: Mapping[str, ListingLanguage] | None = None,
) -> SingleAgentResult:
    """Run one AgentLoop until ShoppingSummary/ChatFallback or a final answer."""

    actual_request = request or SearchRequest(query=query)
    if actual_request.query != query:
        actual_request = actual_request.model_copy(update={"query": query})
    actual_thread_id = thread_id or f"main-{uuid4().hex[:8]}"
    actual_model = model or get_llm()
    runtime = AgentRuntime(
        catalog=catalog,
        request=actual_request,
        profile=profile,
        planner_model=actual_model if parse_request_with_llm else None,
        product_search_backend=product_search_backend,
        product_search_router=product_search_router,
        product_reranker=product_reranker,
        listing_languages=listing_languages,
    )
    preferences = _format_preferences(profile)
    agent = build_main_agent(actual_model, get_system_prompt(preferences))

    try:
        with agent_runtime_scope(runtime):
            state = await asyncio.wait_for(
                agent.ainvoke(
                    {"messages": [("user", query)]},
                    config={
                        "configurable": {"thread_id": actual_thread_id},
                        "recursion_limit": MAIN_AGENT_MAX_ITERATIONS,
                    },
                ),
                timeout=MAIN_AGENT_TIMEOUT_SEC,
            )
    except asyncio.TimeoutError:
        return SingleAgentResult(
            status="timeout",
            thread_id=actual_thread_id,
            final_text=f"主任务超过 {MAIN_AGENT_TIMEOUT_SEC} 秒，已停止。",
        )
    except GraphRecursionError:
        return SingleAgentResult(
            status="max_iterations",
            thread_id=actual_thread_id,
            final_text=f"主任务达到 {MAIN_AGENT_MAX_ITERATIONS} 步上限，已停止。",
        )

    messages: list[BaseMessage] = state["messages"]
    final_text = _message_text(messages[-1]) if messages else ""
    return SingleAgentResult(
        status="ok",
        thread_id=actual_thread_id,
        final_text=final_text,
        terminal_tool=_terminal_tool(messages),
        messages=[_to_trace(message) for message in messages],
        summary=runtime.artifacts.summary,
    )


def _format_preferences(profile: UserProfile | None) -> str:
    if profile is None:
        return "（暂无沉淀偏好）"
    values = [f"偏好类目：{value}" for value in profile.preferred_categories]
    values.extend(
        f"偏好属性：{key}={value}" for key, value in profile.preferred_attributes.items()
    )
    values.extend(f"排除材质：{value}" for value in profile.blocked_materials)
    values.extend(f"排除品牌：{value}" for value in profile.blocked_brands)
    return "\n".join(f"- {value}" for value in values) or "（暂无沉淀偏好）"


def _terminal_tool(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name in TERMINAL_TOOLS:
            return message.name
    return None


def _to_trace(message: BaseMessage) -> AgentMessageTrace:
    tool_calls = []
    if isinstance(message, AIMessage):
        tool_calls = [call["name"] for call in message.tool_calls]
    return AgentMessageTrace(
        message_type=message.type,
        content=_message_text(message),
        tool_name=message.name if isinstance(message, ToolMessage) else None,
        tool_calls=tool_calls,
    )


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return json.dumps(message.content, ensure_ascii=False)
