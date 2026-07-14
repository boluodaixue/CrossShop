"""H5：历史偏好只供 Main 检索后挑选，SearchAgent 保持中性检索。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import fields

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import (
    AssistantMsg,
    Msg,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultBlock,
    UserMsg,
)
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from pydantic import BaseModel, SecretStr

from app.application.agents.orchestrator import MainAgentOrchestrator, SubmitIntentInput
from app.application.context import (
    ContextBudgetPolicy,
    L4Context,
    advance_context_state,
)
from app.application.memory.preference_selector import PreferenceSelector
from app.application.prompts.loader import load_prompts
from app.application.tools.task_dispatch_tool import build_task_dispatch_tool
from app.domain.buyer.preference import BuyerPreference
from app.domain.catalog.ports.retrieval_ports import EmbeddingClient
from app.domain.catalog.product_search_spec import ProductSearchSpec
from app.infrastructure.eventbus import TradeEventBus
from app.infrastructure.persistence.json_file_stores import JsonFilePreferenceStore
from app.infrastructure.settings import Settings


class _AxisEmbedding(EmbeddingClient):
    def __init__(self) -> None:
        self.calls: list[str | tuple[str, ...]] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float("咖啡" in text), float("旅行" in text)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return [await self.embed(text) for text in texts]


class _Worker:
    def __init__(self) -> None:
        self.inputs: Msg | list[Msg] | None = None

    async def reply(self, inputs):
        self.inputs = inputs
        return AssistantMsg("worker", '{"hits": []}')


class _Factory:
    def __init__(self) -> None:
        self.worker = _Worker()

    def build(self, **_kwargs) -> _Worker:
        return self.worker


class _ControlledPreferenceModel(ChatModelBase):
    """Deterministic AgentScope model used to prove the H5 wiring boundary.

    The model deliberately emits the same search call regardless of whether a
    preference hint is present, then uses the hint only after the tool result
    exists.  This is a controlled protocol acceptance, not evidence that an
    external stochastic LLM will always obey the prompt.
    """

    class Parameters(BaseModel):
        pass

    def __init__(self) -> None:
        super().__init__(
            credential=OpenAICredential(api_key=SecretStr("local-test")),
            model="controlled-h5-preference",
            parameters=self.Parameters(),
            stream=False,
        )
        self.formatter = OpenAIChatFormatter()
        self.tool_inputs: list[dict] = []

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice=None,
        **_kwargs,
    ) -> ChatResponse:
        del model_name, tools, tool_choice
        has_result = any(
            isinstance(block, ToolResultBlock)
            for message in messages
            for block in message.get_content_blocks()
        )
        if not has_result:
            args = {
                "normalized_query": "咖啡杯",
                "platform": "reference_seed",
                "category": "咖啡杯",
                "ship_to": "CN",
                "locale": "zh-CN",
                "top_k": 5,
                "price_max_major": 100.0,
                "target_currency": "CNY",
            }
            self.tool_inputs.append(args)
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="controlled-search",
                        name="product_search_tool",
                        input=json.dumps(args, ensure_ascii=False),
                        state=ToolCallState.PENDING,
                    ),
                ],
                is_last=True,
                finished_reason=FinishedReason.COMPLETED,
            )

        has_dislike = any(
            "不要塑料" in (message.get_text_content() or "") for message in messages
        )
        selected = "陶瓷咖啡杯 P-CERAMIC" if has_dislike else "塑料咖啡杯 P-PLASTIC"
        return ChatResponse(
            content=[TextBlock(type="text", text=f"最终推荐：{selected}")],
            is_last=True,
            finished_reason=FinishedReason.COMPLETED,
        )


def _messages(value: Msg | list[Msg] | None) -> list[Msg]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


async def test_main_request_hint_keeps_all_dislikes_and_related_like_only(
    tmp_path,
) -> None:
    store = JsonFilePreferenceStore(tmp_path)
    preferences = [
        BuyerPreference("buyer-1", "dislike", "不要塑料"),
        BuyerPreference("buyer-1", "dislike", "不要动物皮革"),
        BuyerPreference("buyer-1", "like", "喜欢旅行箱"),
        BuyerPreference("buyer-1", "like", "喜欢手冲咖啡杯"),
    ]
    for preference in preferences:
        await store.append(preference)
    generic_embedder = _AxisEmbedding()
    orchestrator = MainAgentOrchestrator(
        sessions=object(),  # type: ignore[arg-type]
        bus=TradeEventBus(),
        preference_store=store,
        preference_selector=PreferenceSelector(generic_embedder),
        preference_top_k=1,
    )
    intent = SubmitIntentInput("session-1", "buyer-1", "zh-CN", "CNY", "推荐咖啡杯")

    inputs = await orchestrator._build_inputs(intent)
    text = "\n".join(message.get_text_content() or "" for message in inputs)

    assert len(inputs) == 2
    assert "不要塑料" in text
    assert "不要动物皮革" in text
    assert "喜欢手冲咖啡杯" in text
    assert "喜欢旅行箱" not in text
    assert "推荐咖啡杯" in text
    assert generic_embedder.calls, "偏好相关性必须使用独立注入的通用 embedder"


async def test_two_buyers_and_sessions_build_isolated_hints(tmp_path) -> None:
    store = JsonFilePreferenceStore(tmp_path)
    await store.append(BuyerPreference("buyer-a", "dislike", "A 不要塑料"))
    await store.append(BuyerPreference("buyer-b", "dislike", "B 不要皮革"))
    orchestrator = MainAgentOrchestrator(
        sessions=object(),  # type: ignore[arg-type]
        bus=TradeEventBus(),
        preference_store=store,
        preference_selector=PreferenceSelector(_AxisEmbedding()),
    )
    intent_a = SubmitIntentInput("session-a", "buyer-a", "zh-CN", "CNY", "咖啡杯")
    intent_b = SubmitIntentInput("session-b", "buyer-b", "zh-CN", "CNY", "旅行箱")

    inputs_a, inputs_b = await asyncio.gather(
        orchestrator._build_inputs(intent_a),
        orchestrator._build_inputs(intent_b),
    )
    text_a = "\n".join(message.get_text_content() or "" for message in inputs_a)
    text_b = "\n".join(message.get_text_content() or "" for message in inputs_b)
    assert "A 不要塑料" in text_a and "B 不要皮革" not in text_a
    assert "B 不要皮革" in text_b and "A 不要塑料" not in text_b


async def test_every_turn_reselects_and_reinjects_preferences_after_l2_l3(
    tmp_path,
) -> None:
    """偏好 hint 是每轮请求输入，不能依赖上轮未被压缩的历史消息。"""
    store = JsonFilePreferenceStore(tmp_path)
    for preference in (
        BuyerPreference("buyer-1", "dislike", "不要塑料"),
        BuyerPreference("buyer-1", "like", "喜欢手冲咖啡杯"),
        BuyerPreference("buyer-1", "like", "喜欢轻便旅行箱"),
    ):
        await store.append(preference)
    orchestrator = MainAgentOrchestrator(
        sessions=object(),  # type: ignore[arg-type]
        bus=TradeEventBus(),
        preference_store=store,
        preference_selector=PreferenceSelector(_AxisEmbedding()),
        preference_top_k=1,
    )

    first = await orchestrator._build_inputs(
        SubmitIntentInput("session-1", "buyer-1", "zh-CN", "CNY", "推荐咖啡杯"),
    )
    assert [message.name for message in first] == ["memory_hint", "buyer-1"]
    first_hint = first[0].get_text_content() or ""
    assert "喜欢手冲咖啡杯" in first_hint
    assert "喜欢轻便旅行箱" not in first_hint

    # 偏好在两轮之间变化时，下一轮应立即读到新值。
    await store.append(BuyerPreference("buyer-1", "dislike", "不要动物皮革"))
    second = await orchestrator._build_inputs(
        SubmitIntentInput("session-1", "buyer-1", "zh-CN", "CNY", "推荐旅行箱"),
    )
    assert [message.name for message in second] == ["memory_hint", "buyer-1"]
    second_hint = second[0].get_text_content() or ""
    assert "不要塑料" in second_hint
    assert "不要动物皮革" in second_hint
    assert "喜欢轻便旅行箱" in second_hint
    assert "喜欢手冲咖啡杯" not in second_hint

    # 用真实 L2/L3 assembler 压缩第一轮。当前轮新注入的
    # memory_hint 仍必须在模型视图中紧贴当前买家请求。
    raw_context = [
        *first,
        AssistantMsg("commerce_concierge", "第一轮结果" * 1_000),
        *second,
    ]
    state, model_view = advance_context_state(
        context_messages=raw_context,
        namespace={"session_context": L4Context(revision=2).to_dict()},
        fixed_messages=[],
        tool_schemas=[],
        policy=ContextBudgetPolicy(
            model_context_tokens=8_000,
            soft_limit_tokens=1,
            l3_keep_recent_frozen_segments=0,
        ),
        has_pending=False,
        has_interrupt=False,
    )
    assert state["freeze_cursor"] == 3
    assert state["stage_summary"] is not None
    assert [message.name for message in model_view[-2:]] == [
        "memory_hint",
        "buyer-1",
    ]
    assert "喜欢轻便旅行箱" in (model_view[-2].get_text_content() or "")


async def test_no_preferences_adds_no_memory_hint_each_turn(tmp_path) -> None:
    orchestrator = MainAgentOrchestrator(
        sessions=object(),  # type: ignore[arg-type]
        bus=TradeEventBus(),
        preference_store=JsonFilePreferenceStore(tmp_path),
        preference_selector=PreferenceSelector(_AxisEmbedding()),
    )
    for raw_query in ("推荐咖啡杯", "再看看旅行箱"):
        inputs = await orchestrator._build_inputs(
            SubmitIntentInput("session-1", "buyer-1", "zh-CN", "CNY", raw_query),
        )
        assert len(inputs) == 1
        assert inputs[0].name == "buyer-1"
        assert "<buyer-preferences>" not in (inputs[0].get_text_content() or "")


async def test_task_dispatch_never_injects_buyer_preferences() -> None:
    search = _Factory()
    trade = _Factory()
    tool = build_task_dispatch_tool(search, trade, TradeEventBus())  # type: ignore[arg-type]

    await tool(
        subagent_type="search_agent",
        demands="示例平台找咖啡杯，预算 100 元",
        platform="reference_seed",
    )

    messages = _messages(search.worker.inputs)
    assert len(messages) == 1
    text = messages[0].get_text_content() or ""
    assert text == "示例平台找咖啡杯，预算 100 元"
    assert "<buyer-preferences>" not in text


async def test_controlled_main_search_args_stable_and_preferences_final_only() -> None:
    recorded_specs: list[dict] = []

    async def product_search_tool(
        normalized_query: str,
        platform: str,
        category: str | None = None,
        ship_to: str | None = None,
        locale: str = "zh-CN",
        top_k: int = 5,
        price_max_major: float | None = None,
        target_currency: str = "CNY",
    ) -> ToolChunk:
        recorded_specs.append(
            {
                "normalized_query": normalized_query,
                "platform": platform,
                "category": category,
                "ship_to": ship_to,
                "locale": locale,
                "top_k": top_k,
                "price_max_major": price_max_major,
                "target_currency": target_currency,
            },
        )
        return ToolChunk(
            content=[
                TextBlock(
                    type="text",
                    text=json.dumps(
                        {
                            "hits": [
                                {
                                    "product_id": "P-PLASTIC",
                                    "title": "塑料咖啡杯",
                                },
                                {
                                    "product_id": "P-CERAMIC",
                                    "title": "陶瓷咖啡杯",
                                },
                            ],
                            "filtered_out": [],
                        },
                        ensure_ascii=False,
                    ),
                ),
            ],
        )

    async def run(inputs: list[Msg]) -> tuple[dict, str]:
        model = _ControlledPreferenceModel()
        agent = Agent(
            name="commerce_concierge",
            system_prompt=load_prompts()["main_agent"]["system_prompt"],
            model=model,
            toolkit=Toolkit(
                tools=[FunctionTool(product_search_tool, is_read_only=True)],
            ),
        )
        reply = await agent.reply(inputs)
        assert len(model.tool_inputs) == 1
        return model.tool_inputs[0], reply.get_text_content() or ""

    current_request = UserMsg(
        "buyer-1",
        "只在示例平台推荐咖啡杯，100 元以内，送到中国，最多 5 件",
    )
    plain_args, plain_text = await run([current_request])
    preferred_args, preferred_text = await run(
        [
            UserMsg(
                "memory_hint",
                "<buyer-preferences>\n- [dislike] 不要塑料\n</buyer-preferences>",
            ),
            current_request,
        ],
    )

    expected_keys = {
        "normalized_query",
        "platform",
        "category",
        "ship_to",
        "locale",
        "top_k",
        "price_max_major",
        "target_currency",
    }
    assert plain_args.keys() == preferred_args.keys() == expected_keys
    assert plain_args == preferred_args
    assert recorded_specs == [plain_args, preferred_args], (plain_text, preferred_text)
    assert "P-PLASTIC" in plain_text
    assert "P-CERAMIC" in preferred_text
    assert "P-PLASTIC" not in preferred_text


async def test_json_preference_paths_do_not_collide_after_name_sanitizing(
    tmp_path,
) -> None:
    store = JsonFilePreferenceStore(tmp_path)
    await store.append(BuyerPreference("buyer/a", "like", "A 喜欢咖啡"))
    await store.append(BuyerPreference("buyera", "like", "B 喜欢旅行"))

    first = await store.list_by_buyer("buyer/a")
    second = await store.list_by_buyer("buyera")
    assert [preference.statement for preference in first] == ["A 喜欢咖啡"]
    assert [preference.statement for preference in second] == ["B 喜欢旅行"]
    assert store._path("buyer/a") != store._path("buyera")


def test_h5_prompt_freezes_neutral_search_and_post_result_preference_use() -> None:
    prompts = load_prompts()
    main = prompts["main_agent"]["system_prompt"]
    search = prompts["sub_agents"]["search"]["system_prompt"]

    assert "历史偏好的使用边界（最高优先级）" in main
    assert "hits / filtered_out 之后" in main
    assert "不得把历史 like/dislike 写入 ProductSearchSpec" in main
    assert "严禁复制、改写或暗示历史偏好" in main
    assert "不得据此再次检索或改写 query" in main
    assert "不会收到买家历史偏好" in search
    assert "不使用历史偏好筛选、删减或重排" in search


def test_h5_defaults_enable_relevant_likes_without_changing_search_spec() -> None:
    setting_fields = {field.name: field for field in fields(Settings)}
    assert setting_fields["preference_relevance_enabled"].default is True
    assert setting_fields["preference_top_k"].default == 5
    assert {field.name for field in fields(ProductSearchSpec)} == {
        "normalized_query",
        "category",
        "ship_to",
        "locale",
        "top_k",
        "target_currency",
        "price_max_major",
    }
