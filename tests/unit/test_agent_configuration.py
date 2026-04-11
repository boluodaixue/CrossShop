from globex_agent.agent.prompts import get_system_prompt
from globex_agent.agent.tool_registry import FULL_TOOL_SET, TERMINAL_TOOLS


def test_stage_six_registers_only_learned_tools() -> None:
    names = [tool.name for tool in FULL_TOOL_SET]

    assert names == [
        "planner",
        "chat_fallback",
        "category_insight",
        "item_search",
        "price_compare",
        "shipping_calc",
        "item_picker",
        "shopping_summary",
    ]
    assert {"shopping_summary", "chat_fallback"} == TERMINAL_TOOLS
    assert "dispatch_tool" not in names

    category_insight = next(
        tool for tool in FULL_TOOL_SET if tool.name == "category_insight"
    )
    properties = category_insight.args_schema.model_json_schema()["properties"]
    assert set(properties) == {"category", "depth"}


def test_item_search_tool_schema_does_not_expose_local_dependencies() -> None:
    item_search = next(tool for tool in FULL_TOOL_SET if tool.name == "item_search")
    properties = item_search.args_schema.model_json_schema()["properties"]

    assert set(properties) == {"query", "platform", "locale", "top_k"}
    assert "catalog" not in properties
    assert "profile" not in properties


def test_system_prompt_contains_loop_and_terminal_rules() -> None:
    prompt = get_system_prompt("- 偏好类目：头戴式耳机")

    assert "Think → Act → Observe → Reflect" in prompt
    assert "先调用 category_insight" in prompt
    assert "不得表述为实时价格或真实销量榜" in prompt
    assert "shopping_summary 和 chat_fallback 是终结性工具" in prompt
    assert "不准调用 dispatch_tool" in prompt
    assert "偏好类目：头戴式耳机" in prompt
