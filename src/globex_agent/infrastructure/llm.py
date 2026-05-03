"""OpenAI-compatible chat model factory for all agents."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from globex_agent.infrastructure.settings import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Create the primary OpenAI-compatible Chat Completions model."""

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key or "not-configured",
        base_url=settings.llm_base_url,
        temperature=0.2,
        max_retries=settings.llm_max_retries,
    )
