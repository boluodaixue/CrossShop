# Context budget baseline — 2026-08-24

## Reproduction

Run from the repository root with the existing Python 3.10 environment:

```powershell
$env:PYTHONPATH='src'
python scripts/measure_context_budget_baseline.py
```

The script performs no model API call. It loads the configured main model adapter, the
current main system prompt, and the exact OpenAI-format schemas of the seven tools exposed
in this environment. `web_search_tool` is absent because `TAVILY_API_KEY` is unset. It uses
14 checked-in Flow regression queries, three real local `CatalogSearchUseCase` Top-5 outputs,
and one deterministic ten-turn frozen-history sample.

## Counting capability

- Configured model: `qwen3-max`.
- `ChatOpenAI.get_num_tokens_from_messages` raises `NotImplementedError` for this model.
- Its text counter resolves `qwen3-max` to `cl100k_base`, not to a Qwen tokenizer.
- The application therefore records every reported number below as `estimated=true`, with
  method `model.get_num_tokens`; it does not present them as provider-accurate token counts.

## Observed layer distribution

All values are estimated tokens from the reproducible run above.

| Layer/sample | min | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| Current user query | 27 | 36 | 71 | 80 |
| L4 candidate | 96 | 105 | 140 | 149 |
| ProductSearch model-visible Top-5 result | 933 | 980 | 1002 | 1002 |
| Reply proxy (not an actual response) | 1 | 1 | 61 | 75 |

Fixed-layer estimates:

- Main system prompt: 1,077.
- Seven bound tool schemas: 1,181.
- Ten frozen interaction segments plus logical breakpoint: 767.
- Largest sampled combined input: 4,167.

The dataset does not contain a representative distribution of actual model replies. The
`expected_degradation`/quality text counted by the script is only a fixture proxy; its values
must not be used as a reply reserve.

## Configuration decision

The configured `CONTEXT_SIZE=128000` is retained as model metadata, but the current adapter
cannot verify real message-token usage for `qwen3-max`. Consequently no measured reply
reserve, safety margin, or soft L3 threshold can be justified yet:

```text
CONTEXT_BUDGET_MODE=observe_only
REPLY_TOKEN_BUDGET=0
CONTEXT_SAFETY_MARGIN_TOKENS=0
CONTEXT_SOFT_LIMIT_TOKENS=0
```

Zero means “not calibrated”, not “zero tokens are sufficient”. L1 persists estimated layer
reports on every main-model call but does not enforce estimated overflow decisions. Enforced
thresholds require either a provider-native message counter/usage result or the exact Qwen
tokenizer plus a real reply distribution, followed by rerunning this baseline.
