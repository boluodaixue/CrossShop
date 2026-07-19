"""Explicit evaluation-only Harness controls.

The production composition root never derives these values from environment
variables.  Callers must pass an instance directly to ``build_container``;
the default instance preserves every online treatment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvaluationHarnessControls:
    loop_feedback_enabled: bool = True
    timeout_enforced: bool = True
    tool_output_reduction_enabled: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


FULL_HARNESS_CONTROLS = EvaluationHarnessControls()
