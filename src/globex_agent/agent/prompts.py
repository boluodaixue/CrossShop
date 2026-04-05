"""Load the versioned prompt configuration from YAML, following chapter 10."""

from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt" / "prompts.yml"


@lru_cache(maxsize=1)
def _load_prompts() -> dict[str, str]:
    with PROMPT_PATH.open("r", encoding="utf-8") as source:
        prompts = yaml.safe_load(source)
    if not isinstance(prompts, dict):
        raise ValueError("prompts.yml must contain a mapping")
    return prompts


def get_system_prompt(long_term_preferences: str = "") -> str:
    template = _load_prompts()["system_prompt"]
    return template.format(
        long_term_preferences=long_term_preferences or "（暂无沉淀偏好）"
    )


def get_planner_prompt() -> str:
    return _load_prompts()["planner_prompt"]


def get_shopping_summary_prompt() -> str:
    return _load_prompts()["shopping_summary_prompt"]
