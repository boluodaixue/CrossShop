import asyncio
import json

from globex_agent.agent import ScriptedShoppingModel, run_agent
from globex_agent.agent.runtime import AgentRuntime, agent_runtime_scope
from globex_agent.catalog import LocalCatalog
from globex_agent.domain import SearchRequest, UserProfile
from globex_agent.recall import RecallHit, RecallResult
from globex_agent.tools.agent_tools import item_search_tool


class RecordingAnnBackend:
    backend_id = "recording-ann"

    def __init__(self, document_ids: list[str]) -> None:
        self._document_ids = document_ids
        self.requested_top_ks: list[int] = []

    def search(self, query: str, top_k: int = 20) -> RecallResult:
        del query
        self.requested_top_ks.append(top_k)
        selected = self._document_ids[:top_k]
        return RecallResult(
            backend_id=self.backend_id,
            hits=tuple(
                RecallHit(document_id=document_id, score=1.0 / rank, rank=rank)
                for rank, document_id in enumerate(selected, start=1)
            ),
            total_recall=len(self._document_ids),
            truncated=len(self._document_ids) > top_k,
        )


def test_scripted_single_agent_runs_full_tool_loop(
    demo_catalog: LocalCatalog,
    demo_requests: list[SearchRequest],
    demo_profiles: dict[str, UserProfile],
) -> None:
    request = demo_requests[0]
    result = asyncio.run(
        run_agent(
            request.query,
            demo_catalog,
            request,
            demo_profiles["demo-user-001"],
            thread_id="test-single-agent-shopping",
            model=ScriptedShoppingModel(),
        )
    )
    tool_calls = [
        name for message in result.messages for name in message.tool_calls
    ]

    assert result.status == "ok"
    assert tool_calls == [
        "planner",
        "item_search",
        "item_search",
        "item_search",
        "price_compare",
        "shipping_calc",
        "item_picker",
        "shopping_summary",
    ]
    assert result.terminal_tool == "shopping_summary"
    assert result.summary is not None
    assert [pick.same_group_id for pick in result.summary.picks] == [
        "hp-nimbus-lite",
        "hp-sonic-commute",
        "hp-aurora-quietpro",
    ]
    assert "找到 3 个合规推荐" in result.final_text


def test_non_shopping_request_uses_terminal_fallback(demo_catalog: LocalCatalog) -> None:
    query = "你好，给我讲个笑话"
    result = asyncio.run(
        run_agent(
            query,
            demo_catalog,
            thread_id="test-single-agent-fallback",
            model=ScriptedShoppingModel(),
        )
    )
    tool_calls = [
        name for message in result.messages for name in message.tool_calls
    ]

    assert result.status == "ok"
    assert tool_calls == ["chat_fallback"]
    assert result.terminal_tool == "chat_fallback"
    assert "当前阶段只处理商品搜索" in result.final_text


def test_item_search_tool_uses_hidden_runtime_ann_dependency(
    demo_catalog: LocalCatalog,
) -> None:
    amazon_ids = [
        item.item_id
        for item in demo_catalog.items
        if item.platform.value == "amazon"
    ]
    backend = RecordingAnnBackend(amazon_ids)
    runtime = AgentRuntime(
        catalog=demo_catalog,
        request=SearchRequest(query="通勤耳机"),
        product_search_backend=backend,
    )

    with agent_runtime_scope(runtime):
        raw = asyncio.run(
            item_search_tool.ainvoke(
                {"query": "通勤耳机", "platform": "amazon", "top_k": 2}
            )
        )

    output = json.loads(raw)
    assert backend.requested_top_ks == [4]
    assert len(output["candidates"]) == 2
    assert all(candidate["platform"] == "amazon" for candidate in output["candidates"])


def test_same_thread_can_run_a_second_turn(demo_catalog: LocalCatalog) -> None:
    thread_id = "test-single-agent-follow-up"
    asyncio.run(
        run_agent(
            "推荐一个通勤背包",
            demo_catalog,
            thread_id=thread_id,
            model=ScriptedShoppingModel(),
        )
    )
    second = asyncio.run(
        run_agent(
            "谢谢",
            demo_catalog,
            thread_id=thread_id,
            model=ScriptedShoppingModel(),
        )
    )

    assert second.terminal_tool == "chat_fallback"
    assert "你刚才说的是：“谢谢”" in second.final_text
