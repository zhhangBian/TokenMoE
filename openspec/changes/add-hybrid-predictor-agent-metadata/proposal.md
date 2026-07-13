# Add Hybrid Predictor, Enriched Agent Metadata, and vLLM Router-Score Capture

## Why

The previous change (`add-prediction-replay-moe-download`) validated a real
vLLM prompt-routing prediction and offline replay pipeline. RouteSig beats
global/temporal baselines on chat/domain datasets, but on the only real agent
dataset (SWE-agent trajectories) it does not beat temporal locality:

- Qwen3 SWE-agent 2x hit rate: global 0.4416, temporal 0.4434, RouteSig 0.4399
- DeepSeek SWE-agent 2x hit rate: global 0.3690, temporal 0.3738, RouteSig 0.3451

The central research claim of TokenMoE — agent metadata provides pre-router
expert-demand signal beyond temporal locality — is therefore not yet proven on
real agent workloads. The target venue is a systems conference paper, so the
evidence must also become statistically rigorous (larger scale, confidence
intervals, ablations) before any runtime integration work is justified.

Three current weaknesses block that claim:

1. SWE-agent metadata is too coarse. RouteSig keys collapse to a few values,
   so the predictor cannot exploit agent structure even if it exists.
2. Predictors are pure statistics without hybridization. Temporal locality and
   metadata signal are treated as competitors instead of complements, and
   layers with no metadata signal are not gated back to baselines.
3. vLLM routed-experts capture returns only expert IDs. Router scores are not
   captured, so weighted coverage, probability-calibrated RouteSig statistics,
   and miss-cost modeling are all marked unavailable.

## What Changes

- Modify the local vLLM fork (`vllm/`, commit base
  `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`) to capture router top-k scores
  alongside routed expert IDs, mirroring the existing
  `RoutedExpertsCapturer` device-buffer -> D2H -> scheduler slot-buffer ->
  output plumbing. Score capture is opt-in and MUST NOT change model outputs.
- Introduce trace schema `tokenmoe.trace.v3` that adds per-token, per-layer,
  per-slot router scores aligned with selected expert IDs. v2 traces remain
  readable; mixed-version analysis is still rejected.
- Enrich the SWE-agent adapter with typed agent metadata: trajectory phase,
  tool type, event outcome, DAG depth, group-local step index, and finer
  prompt segment roles. Increase source-group count beyond 32.
- Add hybrid predictors: confidence-weighted RouteSig + temporal fusion,
  per-layer gating (use RouteSig only where its delta over global is positive
  and confident), sparse-key smoothing, and confidence calibration by block
  type and layer.
- Add a lightweight learned ranker (logistic regression or GBDT) over
  pre-router features only, as an upper predictor tier and ablation point.
- Scale required validation to 1k prompt records per dataset/model pair for
  the existing 7 pairs, with optional non-blocking 5k runs for Qwen3 on
  ShareGPT and SWE-agent.
- Add statistical rigor: bootstrap confidence intervals and at least two
  group-preserving time splits per dataset/model pair.
- Add a metadata ablation matrix (remove role/phase/tool type/block
  type/positions; length-only; temporal-only; RouteSig-only; hybrid) to
  separate agent-structure signal from length/order/dataset leakage.
- Define an explicit agent-metadata increment gate that decides whether the
  next stage pursues the agent-DAG claim or pivots to prompt/block-conditioned
  prefetch only.
- Optionally add a second agent dataset (orchestrator-generated trajectories
  with native role/phase metadata, or another public trajectory corpus)
  without blocking completion.

## Non-Goals

- No online expert prefetch, vLLM scheduler modification, EPLB replica
  placement, or any runtime acceleration claim. These remain next-stage.
- No change to router behavior or model outputs. Score capture is read-only.
- No revival of the Transformers router-logit backend as evidence.
- No changes to `/home/youwei/bzh/dataset/download_dataset.py` beyond
  `DATASET_LIST` (unchanged in this change unless a second agent dataset is
  adopted).

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `tokenmoe-tracing`: router score capture becomes a real vLLM capability with
  alignment and correctness-boundary guarantees; trace schema v3.
- `tokenmoe-route-signature`: hybrid predictors, per-layer gating, calibration,
  learned ranker, ablation evaluation, statistical significance reporting, and
  the agent-metadata increment gate.
- `tokenmoe-evaluation-resources`: enriched SWE-agent metadata fields, larger
  group counts, 1k-scale validation standard, multi-split evaluation
  resources, and the optional second agent dataset.

## Impact

- `vllm/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` and
  the routed-experts plumbing (`gpu_model_runner.py`, `outputs.py`,
  `scheduler.py`, `sampling_params.py`, engine outputs) gain an optional score
  path.
- `tokenmoe/schema.py`, `tokenmoe/trace.py`: v3 schema and validation.
- `scripts/collect_traces.py`: score capture flag, v3 output, preflight.
- `dataset_adapters/swe_agent.py`: enriched metadata and more groups.
- `tokenmoe/prediction.py`, `tokenmoe/routesig.py`, `tokenmoe/metrics.py`:
  hybrid predictors, gating, calibration, ranker, ablations, CIs.
- `scripts/analyze_traces.py`, `tokenmoe/reporting.py`: new reports.
- Existing 256-record artifacts remain as regression baselines; new artifacts
  are written alongside them at 1k scale.
