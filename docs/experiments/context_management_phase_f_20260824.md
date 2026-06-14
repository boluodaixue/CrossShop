# Context management Phase F evaluation — 2026-08-24

Status: deterministic offline implementation evidence. This record is not a Qwen
production token benchmark and does not claim provider prompt-cache hits.

## Reproduction

Run from the repository root with the existing Python environment. None of these
commands makes an external model request:

```powershell
$env:PYTHONPATH='src'
python scripts/eval_context_management_phase_f.py
python scripts/eval_flow_rubric.py --input data/eval/flow_regression_v2.jsonl --limit 14 --no-judge --out .pytest_tmp/phase-f-flow-no-judge.md
pytest -q tests/unit/test_context_phase_f.py
pytest -q tests/integration/test_redis_checkpoint_real.py::test_real_redis_phase_f_long_session_rebuild
```

The long-session command prints all 40 per-model-call layer reports as JSON. It
does not write conversation or model data to the repository.

## Provider cache capability audit

The current worktree has no `.env`, and neither `LLM_API_KEY` nor
`OPENAI_API_KEY` is present in the process environment. No key was read from
another worktree and no online request was attempted.

Local adapter evidence is limited to the following:

- `FallbackChatModel` wraps `ChatOpenAI` and passes successful messages through;
  it does not receive a separate Qwen cache-usage contract.
- Installed `langchain-openai==1.4.3` has generic OpenAI response normalization:
  when a provider really returns `usage.prompt_tokens_details.cached_tokens` or
  `cache_write_tokens`, it can map them to `usage_metadata.input_token_details`
  keys `cache_read` and `cache_creation`.
- No checked-in response, event, trace, JSON/JSONL report, or current local run
  contains those fields for the configured `qwen3-max` OpenAI-compatible
  endpoint.

Conclusion: capability is `not_verified`; `provider_response_evidence=false` and
`cache_usage_statistics_enabled=false`. The logical cache breakpoint remains in
the assembled prompt. No supplier adapter, hit counter, hit rate, `cache_control`,
or cache-hit claim was added.

## Deterministic 20-turn result

The fixture covers initial budget/country/material constraints, two parallel
search rounds, a failed search followed by a successful retry, repeated
recommendations, order preparation/status, L2 freezing and exact-config L3.
The exact counter defines one synthetic character as one token and reports
`estimated=false` for this synthetic model only. It must not be transferred to
`qwen3-max`.

The evaluation-only soft limit is 8,762 tokens. It is reproducibly calibrated as
the maximum exact no-L3 model-call input observed through turn 8; it is an
activation point for this test, not a production threshold. The hard synthetic
context size is 1,000,000, so this scenario tests scheduling rather than hard
overflow.

Actual 40-call measurements:

| Layer | Minimum | Maximum |
| --- | ---: | ---: |
| Fixed prompt | 39 | 39 |
| Tool schemas | 90 | 90 |
| Frozen/stage history + breakpoint | 41 | 7,752 |
| L4 candidate | 423 | 969 |
| Active before latest user | 0 | 0 |
| Current interaction | 10 | 33 |
| Tool results, aggregate per call | 0 | 268 |
| Total input | 625 | 8,762 |

L3 actions measured by the exact counter:

| Action | Covered source segments | Before | After | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1 | 7 | 9,609 | 4,708 | 4,901 |
| 2 | 11 | 8,801 | 5,461 | 3,340 |
| 3 | 15 | 8,966 | 5,626 | 3,340 |
| 4 | 18 | 8,796 | 7,169 | 1,627 |

Final lifecycle state: 86 raw protocol messages, 20 complete interaction units,
`freeze_cursor=82`, 19 immutable frozen segments, 18 stage-summary sources and
4 messages in the current active interaction. All source messages and frozen
segments remain present. L4 reaches revision 40 and retains the initial 500 CNY
budget/US destination, current query, last successful paired search,
recommendation, and confirmed order with quantity 2. All 40 hook calls assert
that active messages and L4 are unchanged by L2/L3. All 20 protocol units are
complete and contain no orphan `ToolMessage`.

## Redis recovery

The real Redis test writes the complete 20-turn `MainAgentState`, closes the
saver, rebuilds a new saver/graph, and verifies equality of L4 revision,
`freeze_cursor`, all frozen segments, stage summary, budget report/decision and
the persisted L3 action. The four-message current active interaction has the
same semantic digest after reconstruction; all 20 reconstructed protocol units
remain complete. Result: `1 passed` (two Redis client deprecation warnings).

## Existing Flow and online boundary

The offline `--no-judge` path parsed all 14 checked-in Flow rows and rendered
both Markdown and JSON. All 14 are explicitly `inconclusive: judge not run`, as
the query dataset contains neither a live execution nor online evidence; this
is a code/data/protocol smoke result, not a quality score.

The local API port `127.0.0.1:8000` is not listening, and no LLM key is
available. Therefore the real online Flow/evidence run and real provider cache
usage observation are environment-blocked. Existing online evidence-gate tests
remain the only runnable local verification; no success result was fabricated.

## Validation results

- Phase F unit + real Redis long-session focus: `3 passed`, 2 Redis warnings.
- Offline online-evidence/protocol gates: `25 passed`.
- Full integration suite: `8 passed`, 12 Redis warnings.
- Full unit suite: `307 passed`, 3 known environment failures, 1 warning. The
  failures are the two missing
  `data/processed/databases/amazon/catalog.sqlite3` cases and the existing FAISS
  write failure through the non-ASCII Windows worktree path; Phase F does not
  alter either subsystem.
- Python compile check passed; full `ruff check src tests scripts` passed.
