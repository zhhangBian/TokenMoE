## Why

The updated research notes identify a gap between the current replay prototype
and the intended TokenMoE claim: scheduler replay still uses true routed
experts as an upper-bound proxy instead of making decisions from pre-router
metadata. The real vLLM/Qwen experiment also used a dense Qwen2 model, so the
model preparation path needs explicit MoE targets before routed expert capture
can be evaluated end to end.

## What Changes

- Add a prediction-facing API that turns RouteSig and baseline predictors into
  per-request, per-layer expert working-set estimates before router execution.
- Replace the scheduler replay's true-active-expert decision proxy with a
  prediction-based DAG replay that uses workload dependencies, ready times,
  waiting penalties, and predicted per-layer top-M expert sets.
- Report temporal and spatial locality metrics needed by the new notes:
  reuse distance, phase-transition overlap, temporal Jaccard, batch fanout,
  per-expert token count, and predictor-vs-oracle replay comparisons.
- Extend model preparation so `/home/youwei/bzh/model/download.py` can download
  MoE target models instead of only dense Qwen models, using the `moe-target`
  profile, `https://hf-mirror.com`, and the local script's hardcoded token for
  this experiment.
- Preserve correctness boundaries: predictions may influence replay decisions
  and future system-side actions, but real router output remains the source of
  truth and low-confidence or missing metadata falls back to baseline.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `tokenmoe-route-signature`: expose prediction outputs and optional temporal
  expert features suitable for replay decisions.
- `tokenmoe-system-optimization`: require prediction-based DAG replay rather
  than using true routed experts as the scheduler decision signal.
- `tokenmoe-evaluation-resources`: require configurable MoE model download
  profiles, including target MoE models and dry-run resource checks.

## Impact

- Affected code: `tokenmoe/routesig.py`, `tokenmoe/metrics.py`,
  `tokenmoe/simulators.py`, `tokenmoe/reporting.py`,
  `scripts/analyze_traces.py`, tests, and documentation.
- External file to change: `/home/youwei/bzh/model/download.py`.
- Existing reports and figures may gain new prediction-vs-oracle and temporal
  locality sections.
- No model-output behavior should change. Real vLLM runtime integration remains
  gated behind later implementation and MoE model availability.

Confirmed implementation decisions:

- Use `moe-target` rather than smoke-only downloads. Initial target candidates
  are `Qwen/Qwen3-30B-A3B`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, and
  `deepseek-ai/DeepSeek-V2-Lite-Chat`.
- Use `https://hf-mirror.com` as the download endpoint.
- Keep a hardcoded Hugging Face token in `/home/youwei/bzh/model/download.py`
  for this local experiment, while avoiding token logging.
- Use vLLM as the first real MoE execution path.
- Use statistical RouteSig for replay decisions and compute temporal metrics as
  reported analysis features.
