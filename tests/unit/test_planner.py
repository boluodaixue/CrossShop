import asyncio
from typing import cast
from unittest.mock import AsyncMock, Mock

from langchain_core.language_models.chat_models import BaseChatModel

from globex_agent.agent.runtime import AgentRuntime, agent_runtime_scope
from globex_agent.catalog import LocalCatalog
from globex_agent.domain import Currency, Platform, SearchRequest
from globex_agent.tools.planner import PlannerOutput, planner


def test_real_planner_parses_free_query_and_updates_runtime_request(
    demo_catalog: LocalCatalog,
) -> None:
    query = "500元以内防泼水、不要真皮的通勤背包"
    parsed = PlannerOutput(
        budget="500.00",
        currency=Currency.CNY,
        category="通勤背包",
        material_pref="不要真皮",
        hard_constraints={
            "water_resistant": "yes",
            "excluded_material": "真皮",
            "usage": "通勤",
        },
        platforms=[Platform.AMAZON, Platform.SHOPEE, Platform.ALIEXPRESS],
        top_k=3,
    )
    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(return_value=parsed.model_dump(mode="json"))
    model = Mock()
    model.with_structured_output.return_value = structured_model
    runtime = AgentRuntime(
        catalog=demo_catalog,
        request=SearchRequest(
            query=query,
            platforms=set(demo_catalog.platforms),
            top_k=3,
        ),
        planner_model=cast(BaseChatModel, model),
    )

    async def invoke_planner() -> str:
        with agent_runtime_scope(runtime):
            return await planner.ainvoke({"user_input": query})

    output = PlannerOutput.model_validate_json(asyncio.run(invoke_planner()))

    assert output == parsed
    assert runtime.request.budget == parsed.budget
    assert runtime.request.hard_constraints == parsed.hard_constraints
    assert runtime.request.platforms == set(parsed.platforms)
    model.with_structured_output.assert_called_once_with(method="json_mode")
    messages = structured_model.ainvoke.await_args.args[0]
    assert messages[-1].content == query
    assert "只提取用户明确表达的条件" in messages[0].content


def test_real_planner_defaults_to_catalog_platforms_when_model_omits_them(
    demo_catalog: LocalCatalog,
) -> None:
    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(
        return_value=PlannerOutput(category="机械键盘", platforms=[], top_k=3)
    )
    model = Mock()
    model.with_structured_output.return_value = structured_model
    runtime = AgentRuntime(
        catalog=demo_catalog,
        request=SearchRequest(query="推荐机械键盘"),
        planner_model=cast(BaseChatModel, model),
    )

    async def invoke_planner() -> str:
        with agent_runtime_scope(runtime):
            return await planner.ainvoke({"user_input": "推荐机械键盘"})

    output = PlannerOutput.model_validate_json(asyncio.run(invoke_planner()))

    assert output.platforms == list(demo_catalog.platforms)
    assert runtime.request.platforms == set(demo_catalog.platforms)


def test_real_planner_normalizes_only_empty_container_drift(
    demo_catalog: LocalCatalog,
) -> None:
    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(
        return_value={
            "budget": 500,
            "currency": "CNY",
            "category": "通勤背包",
            "material_pref": [],
            "style_pref": [],
            "hard_constraints": {"water_resistant": "yes"},
            "soft_preferences": {},
            "platforms": ["amazon"],
            "top_k": 3,
            "user_id": None,
        }
    )
    model = Mock()
    model.with_structured_output.return_value = structured_model
    runtime = AgentRuntime(
        catalog=demo_catalog,
        request=SearchRequest(query="500元防泼水通勤背包"),
        planner_model=cast(BaseChatModel, model),
    )

    async def invoke_planner() -> str:
        with agent_runtime_scope(runtime):
            return await planner.ainvoke({"user_input": "500元防泼水通勤背包"})

    output = PlannerOutput.model_validate_json(asyncio.run(invoke_planner()))

    assert output.material_pref is None
    assert output.style_pref is None
    assert output.soft_preferences == []
