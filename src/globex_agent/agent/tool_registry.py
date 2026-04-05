"""Tools exposed to the stage-three single AgentLoop."""

from langchain_core.tools import BaseTool

from globex_agent.tools.agent_tools import (
    item_picker_tool,
    item_search_tool,
    price_compare_tool,
    shipping_calc_tool,
    shopping_summary_tool,
)
from globex_agent.tools.category_insight import category_insight_tool
from globex_agent.tools.chat_fallback import chat_fallback
from globex_agent.tools.planner import planner

# Chapter 14 calls the completed nine-tool collection FULL_TOOL_SET. Stage six
# adds CategoryInsight; WebSearch and dispatch_tool remain intentionally absent.
FULL_TOOL_SET: list[BaseTool] = [
    planner,
    chat_fallback,
    category_insight_tool,
    item_search_tool,
    price_compare_tool,
    shipping_calc_tool,
    item_picker_tool,
    shopping_summary_tool,
]

TERMINAL_TOOLS = {"shopping_summary", "chat_fallback"}
