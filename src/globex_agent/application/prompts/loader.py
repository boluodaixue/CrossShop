"""Load and cache the versioned Globex prompt set."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent / "globex.yml"


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    with PROMPTS_PATH.open("r", encoding="utf-8") as source:
        loaded = yaml.safe_load(source)
    if not isinstance(loaded, dict):
        raise ValueError("globex.yml must contain a mapping")
    return loaded
