"""Context lifecycle, state, compaction, assembly and budget primitives."""

from .assembler import (
    CACHE_BREAKPOINT_TEXT,
    assemble_llm_input_messages,
    build_context_pre_model_hook,
    context_pre_model_hook,
)
from .budget import (
    ContextBudgetExceeded,
    ContextBudgetPolicy,
    TokenCounter,
    count_tokens,
    decide_budget,
    measure_budget,
)
from .compactor import compact_interaction, freeze_settled_units, summarize_frozen_segments
from .lifecycle import (
    InteractionUnit,
    ToolInteraction,
    group_interaction_units,
    settled_prefix_units,
)
from .models import (
    BudgetLayer,
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
    apply_tool_output_contract,
    format_tool_output,
)

__all__ = [
    "BudgetLayer",
    "BudgetReport",
    "CACHE_BREAKPOINT_TEXT",
    "FrozenSegment",
    "L4Context",
    "L4Reducer",
    "LayerTokenCount",
    "SessionContext",
    "SessionContextSnapshot",
    "StageSummary",
    "TokenCounter",
    "ContextBudgetExceeded",
    "ContextBudgetPolicy",
    "TOOL_OUTPUT_CONTRACTS",
    "ToolArtifactStore",
    "ToolOutputContract",
    "ToolInteraction",
    "InteractionUnit",
    "assemble_llm_input_messages",
    "apply_tool_output_contract",
    "build_context_pre_model_hook",
    "compact_interaction",
    "context_pre_model_hook",
    "count_tokens",
    "decide_budget",
    "format_tool_output",
    "measure_budget",
    "freeze_settled_units",
    "group_interaction_units",
    "reduce_l4",
    "settled_prefix_units",
    "summarize_frozen_segments",
]
