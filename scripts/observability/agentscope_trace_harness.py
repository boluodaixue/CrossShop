"""Controlled, network-free AgentScope tracing scenarios for H5 acceptance.

The harness uses the real AgentScope ``Agent`` and ``TracingMiddleware`` with
deterministic local ``ChatModelBase`` implementations. It proves middleware
span topology without calling an external model or observability backend.
"""

import asyncio
import json
from typing import Literal

from agentscope.agent import Agent, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import TextBlock, ToolCallBlock, ToolCallState, UserMsg
from agentscope.middleware import TracingMiddleware
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from pydantic import BaseModel, SecretStr

from app.application.agents.permissions import allow_business_tools
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.infrastructure.context import ShoppingContext, ShoppingContextSnapshot
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.tracing import set_span_attributes, trace_intent

_PLATFORMS = ("crossshop_reference", "reference_seed", "amazon")


async def product_search_tool(
    normalized_query: str,
    platform: Literal["crossshop_reference", "reference_seed", "amazon"] | None = None,
) -> ToolChunk:
    """Return one synthetic product after a small deterministic async delay."""

    await asyncio.sleep(0.03)
    payload = {
        "hits": [
            {
                "product_id": f"controlled-private-product-{platform or 'reference_seed'}",
                "title": "controlled private product card",
            },
        ],
        "filtered_out": [],
        "normalized_query": normalized_query,
    }
    return ToolChunk(
        content=[TextBlock(type="text", text=json.dumps(payload))],
    )


class _ControlledTraceModel(ChatModelBase):
    class Parameters(BaseModel):
        pass

    def __init__(self, *, role: str, platform: str | None = None) -> None:
        super().__init__(
            credential=OpenAICredential(api_key=SecretStr("local-model-secret")),
            model=f"controlled-{role}{f'-{platform}' if platform else ''}",
            parameters=self.Parameters(),
            stream=False,
        )
        self.formatter = OpenAIChatFormatter()
        self._role = role
        self._platform = platform
        self._call_count = 0

    async def _call_api(
        self,
        model_name: str,
        messages,
        tools=None,
        tool_choice=None,
        **_kwargs,
    ) -> ChatResponse:
        del model_name, tools, tool_choice
        del messages
        self._call_count += 1
        if self._role == "error":
            raise RuntimeError(
                "controlled private status with sk-local-never-export and address",
            )
        if self._call_count > 1:
            return ChatResponse(
                content=[TextBlock(type="text", text="controlled private final reply")],
                is_last=True,
                finished_reason=FinishedReason.COMPLETED,
            )

        if self._role == "main-single":
            calls = [
                ToolCallBlock(
                    id="single-product-search",
                    name="product_search_tool",
                    input=json.dumps(
                        {
                            "normalized_query": "controlled private single query",
                            "platform": "reference_seed",
                        },
                    ),
                    state=ToolCallState.PENDING,
                ),
            ]
        elif self._role == "main-cross":
            calls = [
                ToolCallBlock(
                    id=f"dispatch-{platform}",
                    name="task_dispatch",
                    input=json.dumps(
                        {
                            "subagent_type": "search_agent",
                            "demands": f"controlled private {platform} demand",
                            "platform": platform,
                        },
                    ),
                    state=ToolCallState.PENDING,
                )
                for platform in _PLATFORMS
            ]
        elif self._role == "search" and self._platform in _PLATFORMS:
            calls = [
                ToolCallBlock(
                    id=f"product-search-{self._platform}",
                    name="product_search_tool",
                    input=json.dumps(
                        {
                            "normalized_query": (
                                f"controlled private {self._platform} query"
                            ),
                            "platform": self._platform,
                        },
                    ),
                    state=ToolCallState.PENDING,
                ),
            ]
        else:
            raise AssertionError(f"unsupported controlled role: {self._role}")
        return ChatResponse(
            content=calls,
            is_last=True,
            finished_reason=FinishedReason.COMPLETED,
        )


def _agent(
    *,
    name: str,
    model: _ControlledTraceModel,
    tools: list[FunctionTool],
    session_id: str,
) -> Agent:
    return allow_business_tools(
        Agent(
            name=name,
            system_prompt="controlled private system prompt",
            model=model,
            toolkit=Toolkit(tools=tools),
            middlewares=[TracingMiddleware()],
            state=AgentState(session_id=session_id),
            react_config=ReActConfig(max_iters=4),
        ),
    )


class _ControlledSearchFactory:
    def __init__(self) -> None:
        self.created_platforms: list[str] = []

    def build(self, *, platform: str, site_locale=None) -> Agent:
        del site_locale
        self.created_platforms.append(platform)
        return _agent(
            name="catalog_search_agent",
            model=_ControlledTraceModel(role="search", platform=platform),
            tools=[FunctionTool(product_search_tool, is_read_only=True)],
            session_id=f"controlled-search-session-{platform}",
        )


class _UnusedTradeFactory:
    def build(self):
        raise AssertionError("trade agent is outside this H5 trace acceptance")


async def _run_agent(
    agent: Agent,
    *,
    scenario: str,
    session_id: str,
    buyer_id: str,
    request: str,
) -> None:
    token = ShoppingContext.set(
        ShoppingContextSnapshot(session_id, buyer_id, "zh-CN", "CNY"),
    )
    try:
        with trace_intent(
            session_id=session_id,
            buyer_id=buyer_id,
            locale="zh-CN",
            currency="CNY",
        ) as root:
            set_span_attributes(root, {"crossshop.acceptance.scenario": scenario})
            await agent.reply(UserMsg("controlled-private-buyer", request))
    finally:
        ShoppingContext.reset(token)


async def run_agentscope_scenarios() -> dict[str, object]:
    """Run single-platform and three-platform real middleware scenarios."""

    single_model = _ControlledTraceModel(role="main-single")
    single = _agent(
        name="commerce_concierge",
        model=single_model,
        tools=[FunctionTool(product_search_tool, is_read_only=True)],
        session_id="controlled-main-single-session",
    )
    await _run_agent(
        single,
        scenario="agentscope_single_platform",
        session_id="controlled-root-single-session",
        buyer_id="controlled-root-single-buyer",
        request="controlled private single-platform request",
    )

    bus = TradeEventBus()
    search_factory = _ControlledSearchFactory()
    dispatch = build_task_dispatch_tool(
        search_factory,  # type: ignore[arg-type]
        _UnusedTradeFactory(),  # type: ignore[arg-type]
        bus,
    )
    cross_model = _ControlledTraceModel(role="main-cross")
    cross = _agent(
        name="commerce_concierge",
        model=cross_model,
        tools=[FunctionTool(dispatch, is_concurrency_safe=True)],
        session_id="controlled-main-cross-session",
    )
    await _run_agent(
        cross,
        scenario="agentscope_cross_platform",
        session_id="controlled-root-cross-session",
        buyer_id="controlled-root-cross-buyer",
        request="controlled private cross-platform request",
    )
    error_calls = await run_agentscope_error_scenario()
    return {
        "single_platform": "reference_seed",
        "cross_platforms": search_factory.created_platforms,
        "single_main_model_calls": single_model._call_count,
        "cross_main_model_calls": cross_model._call_count,
        "error_model_calls": error_calls,
    }


async def run_agentscope_error_scenario() -> int:
    """Exercise real AgentScope ERROR spans containing private exception text."""

    error_model = _ControlledTraceModel(role="error")
    failing = _agent(
        name="commerce_concierge",
        model=error_model,
        tools=[],
        session_id="controlled-main-error-session",
    )
    try:
        await _run_agent(
            failing,
            scenario="agentscope_error_privacy",
            session_id="controlled-root-error-session",
            buyer_id="controlled-root-error-buyer",
            request="controlled private error request",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("controlled AgentScope error scenario did not fail")
    return error_model._call_count
