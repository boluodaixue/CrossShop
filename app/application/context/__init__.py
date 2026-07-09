"""Custom L0-L4 context lifecycle primitives for the AgentScope V2 path."""

from .assembler import (
    CACHE_BREAKPOINT_TEXT,
    advance_context_state,
    assemble_llm_input_messages,
    cache_prefix_sha256,
)
from .budget import (
    ContextBudgetExceeded,
    ContextBudgetPolicy,
    count_tokens,
    decide_budget,
    measure_budget,
)
from .compactor import (
    compact_interaction,
    freeze_settled_units,
    summarize_frozen_segments,
)
from .lifecycle import (
    InteractionUnit,
    ToolInteraction,
    group_interaction_units,
    settled_prefix_units,
)
from .models import (
    BudgetReport,
    FrozenSegment,
    L4Context,
    LayerTokenCount,
    SessionContext,
    SessionContextSnapshot,
    StageSummary,
)
from .reducer import L4Reducer, reduce_l4
from .tool_output import (
    TOOL_OUTPUT_CONTRACTS,
    ToolArtifactStore,
    ToolOutputContract,
    format_tool_output,
)

__all__ = [
    "CACHE_BREAKPOINT_TEXT",
    "TOOL_OUTPUT_CONTRACTS",
    "BudgetReport",
    "ContextBudgetExceeded",
    "ContextBudgetPolicy",
    "FrozenSegment",
    "InteractionUnit",
    "L4Context",
    "L4Reducer",
    "LayerTokenCount",
    "SessionContext",
    "SessionContextSnapshot",
    "StageSummary",
    "ToolArtifactStore",
    "ToolInteraction",
    "ToolOutputContract",
    "advance_context_state",
    "assemble_llm_input_messages",
    "cache_prefix_sha256",
    "compact_interaction",
    "count_tokens",
    "decide_budget",
    "format_tool_output",
    "freeze_settled_units",
    "group_interaction_units",
    "measure_budget",
    "reduce_l4",
    "settled_prefix_units",
    "summarize_frozen_segments",
]
