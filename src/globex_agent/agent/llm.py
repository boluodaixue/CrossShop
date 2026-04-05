"""Unified model initialization for the stage-three AgentLoop."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Create the real OpenAI-compatible model configured in local ``.env``."""

    load_dotenv()
    required = ["LLM_MAIN", "OPENAI_API_KEY", "OPENAI_BASE_URL"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"真实模型模式缺少环境变量：{names}")
    return init_chat_model(
        os.environ["LLM_MAIN"],
        model_provider="openai",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=0.3,
    )
