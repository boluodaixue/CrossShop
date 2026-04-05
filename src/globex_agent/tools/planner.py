"""Stage-three Planner tool from chapters 02 and 10."""

from decimal import Decimal
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from globex_agent.agent.prompts import get_planner_prompt
from globex_agent.agent.runtime import get_agent_runtime
from globex_agent.domain import Currency, Platform, SearchRequest, UserProfile


class PlannerOutput(BaseModel):
    """Structured shopping intent consumed by later AgentLoop turns."""

    budget: Decimal | None = Field(default=None, gt=0)
    currency: Currency = Currency.CNY
    category: str | None = None
    material_pref: str | None = None
    style_pref: str | None = None
    hard_constraints: dict[str, Any] = Field(default_factory=dict)
    soft_preferences: list[str] = Field(default_factory=list)
    platforms: list[Platform] = Field(default_factory=list)
    top_k: int = Field(default=20, ge=1, le=50)
    user_id: str | None = None


@tool
async def planner(user_input: str) -> str:
    """拆解购物需求，返回预算、品类、偏好、硬约束和目标平台。

    Args:
        user_input: 用户原始购物意图。

    Returns:
        PlannerOutput 的 JSON 字符串。
    """

    runtime = get_agent_runtime()
    if runtime.planner_model is None:
        output = _from_preparsed_request(user_input, runtime.request, runtime.profile)
    else:
        output = await parse_shopping_intent(
            runtime.planner_model,
            user_input,
            available_platforms=list(runtime.catalog.platforms),
        )
        output = output.model_copy(
            update={
                "soft_preferences": _merge_preferences(
                    output.soft_preferences,
                    _soft_preferences(runtime.profile),
                ),
                "user_id": runtime.request.user_id,
            }
        )
        runtime.request = _to_search_request(output, user_input, runtime.request.user_id)
    return output.model_dump_json()


async def parse_shopping_intent(
    model: BaseChatModel,
    user_input: str,
    *,
    available_platforms: list[Platform],
) -> PlannerOutput:
    """Use the chapter-10 planner prompt to parse a free-form shopping query."""

    platform_text = ", ".join(platform.value for platform in available_platforms)
    prompt = get_planner_prompt().replace("{available_platforms}", platform_text)
    structured_model = model.with_structured_output(method="json_mode")
    result = await structured_model.ainvoke(
        [SystemMessage(content=prompt), HumanMessage(content=user_input)]
    )
    output = PlannerOutput.model_validate(_normalize_empty_values(result))
    if not output.platforms:
        output = output.model_copy(update={"platforms": available_platforms})
    return output


def _normalize_empty_values(result: Any) -> Any:
    """Normalize only common empty-container drift before strict validation."""

    if not isinstance(result, dict):
        return result
    normalized = dict(result)
    for field_name in ("budget", "category", "material_pref", "style_pref", "user_id"):
        if normalized.get(field_name) in ([], {}, ""):
            normalized[field_name] = None
    for field_name in ("soft_preferences", "platforms"):
        if normalized.get(field_name) in ({}, None):
            normalized[field_name] = []
    if normalized.get("hard_constraints") in ([], None):
        normalized["hard_constraints"] = {}
    return normalized


def _from_preparsed_request(
    user_input: str,
    request: SearchRequest,
    profile: UserProfile | None,
) -> PlannerOutput:
    return PlannerOutput(
        budget=request.budget,
        currency=request.currency,
        category=_infer_category(user_input),
        material_pref=_material_preference(request.hard_constraints),
        style_pref="小众" if "小众" in user_input else None,
        hard_constraints=request.hard_constraints,
        soft_preferences=_soft_preferences(profile),
        platforms=sorted(request.platforms, key=lambda value: value.value),
        top_k=request.top_k,
        user_id=request.user_id,
    )


def _to_search_request(
    output: PlannerOutput,
    user_input: str,
    user_id: str | None,
) -> SearchRequest:
    return SearchRequest(
        query=user_input,
        user_id=user_id,
        budget=output.budget,
        currency=output.currency,
        platforms=set(output.platforms),
        top_k=output.top_k,
        hard_constraints=output.hard_constraints,
    )


def _infer_category(user_input: str) -> str | None:
    for keyword, category in (
        ("耳机", "头戴式耳机"),
        ("背包", "背包"),
        ("键盘", "机械键盘"),
    ):
        if keyword in user_input:
            return category
    return None


def _material_preference(hard_constraints: dict[str, Any]) -> str | None:
    excluded = hard_constraints.get("excluded_material")
    return f"不要{excluded}" if excluded else None


def _soft_preferences(profile: UserProfile | None) -> list[str]:
    if profile is None:
        return []
    values = [*profile.preferred_categories]
    values.extend(f"{key}={value}" for key, value in profile.preferred_attributes.items())
    return values


def _merge_preferences(primary: list[str], secondary: list[str]) -> list[str]:
    return list(dict.fromkeys([*primary, *secondary]))
