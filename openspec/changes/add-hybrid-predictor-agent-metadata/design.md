# Design: Hybrid Predictor, Enriched Agent Metadata, vLLM Router-Score Capture

## Context

Target output is a systems conference paper. The chosen roadmap is:

- Stage A (this change): prove or refute that agent/prompt metadata adds
  predictive signal beyond temporal locality on real agent workloads, with
  paper-grade rigor.
- Stage B (next change): edge expert prefetch (RQ2) — replay with real miss
  cost, then vLLM offload-runtime integration and end-to-end TTFT/stall
  numbers.
- Stage C: runtime integration of scheduling/EPLB only if evidence supports.

This change is the gate for the agent-DAG narrative. All decisions below are
made to keep the gate honest and the evidence reusable regardless of outcome.

## Goals / Non-Goals

Goals:

- Real router scores from vLLM without changing outputs.
- SWE-agent metadata rich enough that RouteSig keys have real cardinality.
- A hybrid predictor family that treats temporal locality as the baseline to
  beat, not a competitor to ignore.
- 1k-scale, CI-backed, ablation-supported evidence.

Non-goals: runtime acceleration, scheduler/EPLB modification, router changes.

## Decision 1: vLLM router-score capture design

Mirror the existing routed-experts path end to end:

- `RoutedExpertsCapturer` gains a parallel `score_device_buffer` of shape
  `(max_num_batched_tokens, num_layers, top_k)`, dtype `float16` (scores are
  post-softmax in `[0,1]`; fp16 resolution is sufficient and halves buffer
  size). `capture(layer_id, topk_ids, topk_weights)` writes both tensors with
  identical DP/SP slicing logic so IDs and scores can never misalign.
- D2H copy, `RoutedExpertsLists`, `ModelRunnerOutput`, scheduler-side
  `RoutedExpertsManager` slot buffer, and request outputs each gain an
  optional score twin, gated by a single config flag
  (`enable_return_routed_expert_scores`) that implies
  `enable_return_routed_experts`.
- Capture hook: the score tensor is taken at the same point as `topk_ids`
  (after top-k selection, after any renormalization the model applies, before
  expert dispatch). The captured value is the weight actually used to combine
  expert outputs, recorded per model family.
- Correctness boundary: the score path only reads tensors already produced by
  the router. If score capture fails preflight for a model family, collection
  proceeds with IDs only and the trace marks scores unavailable, exactly like
  today.
- Memory: scheduler-side slot buffer doubles in width class (fp16 vs
  uint8/uint16). For Qwen3-30B (48 layers, top-k 8) this adds
  `slots * 48 * 8 * 2` bytes; acceptable at current max_model_len settings,
  and logged at init like the existing buffer.

Alternative considered: hook router logits via forward hooks outside the fused
path. Rejected: fused kernels don't expose logits reliably, and the existing
capturer already solves batching/DP/SP alignment.

## Decision 2: Trace schema v3

- `tokenmoe.trace.v3` adds `router_scores[token][layer_slot][k]` aligned with
  `selected_experts`, plus `router_score_semantics` (e.g.
  `post_softmax_topk_renormalized`) recorded per model.
- v2 remains readable for regression comparison; analysis rejects mixed
  versions in one run, as before.
- `weighted_coverage` switches from `unavailable_router_scores_not_captured_by_vllm`
  to computed-when-present.
- RouteSig statistics may accumulate score mass instead of selection counts;
  both modes are kept and compared (score-weighted vs count-based) since this
  is itself an ablation.

## Decision 3: Enriched SWE-agent metadata

New per-record/per-block fields emitted by `dataset_adapters/swe_agent.py`:

- `trajectory_phase`: one of `issue_understanding`, `edit`, `test`, `debug`,
  `finalize`, derived from action-sequence heuristics; heuristic version
  recorded in the manifest.
- `tool_type`: `read_file`, `search`, `edit`, `run_test`, `inspect_error`,
  `other`.
- `event_outcome`: `test_failed`, `patch_applied`, `error_observed`, `none`.
- `dag_depth`, `group_local_step_index`, `on_critical_path`.
- Segment roles refined to `system`, `code_context`, `trajectory_event`,
  `tool_result`.
- Group count target: >= 64 source groups at 1k records (vs 32 at 256).

Fields that cannot be derived from the source data are marked unavailable in
the manifest rather than silently defaulted — same policy as v2 adapters.

RouteSig fallback order is extended to include `tool_type` and
`trajectory_phase` between the existing levels; the exact order is fixed in
the spec delta so results are reproducible.

## Decision 4: Predictor family and gating

Three tiers, each a paper ablation row:

1. `hybrid_routesig_temporal`: per-layer convex combination of RouteSig and
   temporal-window distributions. Weight = calibrated RouteSig confidence for
   the resolved fallback key at that layer.
2. Calibration/smoothing: additive smoothing for sparse keys; confidence
   calibrated per (block_type, layer) bucket against held-in data only.
3. `learned_ranker`: logistic regression or GBDT over pre-router features
   (metadata one-hots, temporal statistics, segment position/length, layer),
   producing per-layer expert rankings. Trained on the train split only;
   online evaluation stays predict-before-update.

Per-layer gating: a layer uses RouteSig/hybrid only if its train-split
`delta_vs_global` is positive with sufficient support; otherwise that layer
falls back to temporal/global. Gate decisions are persisted so reports can
show which layers carried the signal (this feeds Stage B layer targeting).

All predictor inputs are strictly pre-router: no hidden states, no current
request's router outputs.

## Decision 5: Statistical rigor

- Scale: 1k records per existing dataset/model pair (7 pairs). Optional
  non-blocking: 5k for Qwen3 ShareGPT and Qwen3 SWE-agent.
- Bootstrap CIs (>= 1000 resamples over evaluation groups) for hit-rate
  deltas between predictors; report per-pair and per-layer.
- At least two group-preserving time splits (e.g. 70/30 and 60/40, or two
  disjoint temporal folds); a claim must hold on both.
- Ablation matrix on agent datasets: full hybrid; minus role; minus phase;
  minus tool_type; minus block_type; minus positions; length-only;
  temporal-only; RouteSig-only.

## Decision 6: The gate

Written criterion, evaluated on SWE-agent (both models) at the 2x budget:

- PASS: hybrid predictor beats temporal-only with non-overlapping bootstrap
  CIs on both splits, AND at least one metadata ablation shows a significant
  drop (proving the increment comes from metadata, not fusion mechanics).
- FAIL: otherwise. The decision document then redirects Stage B to
  prompt/block-conditioned prefetch on chat/domain datasets (where signal is
  already established) and demotes agent-DAG to a characterization finding.

Either outcome is a legitimate paper input; the gate result is a required
deliverable (`docs/` decision record), not a success condition.

## Decision 7: Optional second agent dataset

The SWE-agent-only agent evidence is thin for a paper. Optional, non-blocking
in this change: generate trajectories from a real orchestrator (e.g. a
planner-coder-tester pipeline instrumented to emit native role/phase/tool
metadata) or adopt a second public trajectory corpus. Decision and rationale
are recorded in the manifest/docs; absence does not block archive.

## Risks / Trade-offs

- Score capture across fused kernels may differ per model family (Qwen3 vs
  DeepSeek renormalization). Mitigation: record score semantics per model;
  preflight asserts scores are finite, in [0,1] after documented transform,
  and aligned with IDs on a smoke batch.
- 1k/5k runs cost GPU time. Mitigation: existing batching + the v8 analysis
  optimizations; 5k marked optional.
- Learned ranker can overfit small agent data. Mitigation: group-preserving
  splits, two folds, simple model classes only.
- Heuristic phase labeling may inject noise. Mitigation: manifest-recorded
  heuristic version + ablation isolates its contribution.

## Migration

- v2 traces/analyses stay valid as regression baselines; no rewrite.
- New artifacts land under the same artifact root with `_1k`/`_5k` suffixed
  names to avoid clobbering 256-record baselines.
