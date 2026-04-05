"""Terminal fallback tool for non-shopping conversation."""

from langchain_core.tools import tool

from globex_agent.agent.runtime import get_agent_runtime


@tool
async def chat_fallback(user_input: str) -> str:
    """处理不需要商品检索的闲聊或越界请求，并结束当前循环。

    Args:
        user_input: 用户原始输入。
    """

    message = (
        "我是 Globex 购物助手，当前阶段只处理商品搜索、比价、运费估算和精挑。"
        f"你刚才说的是：“{user_input}”。如果要购物，请告诉我品类、预算和硬约束。"
    )
    get_agent_runtime().artifacts.fallback_text = message
    return message
