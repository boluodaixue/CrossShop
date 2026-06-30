"""Load the versioned end-to-end Rubric case suite."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.eval.rubric_contract import RubricCaseSuite


def load_case_suite(path: Path) -> RubricCaseSuite:
    """Parse YAML through the strict Pydantic contract."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("rubric case file must contain a YAML object")
    return RubricCaseSuite.model_validate(raw)
