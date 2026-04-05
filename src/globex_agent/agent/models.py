"""Stable result models for the stage-three single AgentLoop."""

from typing import Literal

from pydantic import BaseModel, Field

from globex_agent.domain import ShoppingSummaryOutput


class AgentMessageTrace(BaseModel):
    message_type: str
    content: str = ""
    tool_name: str | None = None
    tool_calls: list[str] = Field(default_factory=list)


class SingleAgentResult(BaseModel):
    status: Literal["ok", "timeout", "max_iterations"]
    thread_id: str
    final_text: str
    terminal_tool: str | None = None
    messages: list[AgentMessageTrace] = Field(default_factory=list)
    summary: ShoppingSummaryOutput | None = None
