# Tasks: add-hybrid-predictor-agent-metadata

## 1. Scope Confirmation

- [ ] 1.1 Confirm current-stage success standard: 1k-scale hybrid-predictor evidence with CIs and ablations on the existing 7 dataset/model pairs, plus router-score-enabled v3 traces
- [ ] 1.2 Confirm next-stage-only items: online prefetch, vLLM scheduler modification, EPLB placement, end-to-end acceleration claims
- [ ] 1.3 Confirm the agent-metadata increment gate criterion and that either PASS or FAIL is an acceptable completion outcome
- [ ] 1.4 Confirm 5k runs and the second agent dataset are optional and non-blocking

## 2. vLLM Router-Score Capture

- [ ] 2.1 Add score device buffer and extend `RoutedExpertsCapturer.capture` to record `topk_weights` with identical DP/SP slicing as `topk_ids`
- [ ] 2.2 Plumb scores through D2H copy, `RoutedExpertsLists`, `ModelRunnerOutput`, scheduler `RoutedExpertsManager`, and request outputs behind `enable_return_routed_expert_scores`
- [ ] 2.3 Bind score capture at the MoE layer for Qwen3-MoE and DeepSeek-V2 model classes; record per-model score semantics
- [ ] 2.4 Add preflight smoke check: scores finite, in documented range, aligned with expert IDs, outputs unchanged vs capture-off run (token IDs identical)
- [ ] 2.5 Fail closed: if score capture is unsupported for a config, continue ID-only capture and mark scores unavailable
- [ ] 2.6 Log added CPU slot-buffer memory at init

## 3. Trace Schema v3

- [ ] 3.1 Add `tokenmoe.trace.v3` with aligned `router_scores` and `router_score_semantics`; keep v2 readers
- [ ] 3.2 Reject mixed v2/v3 inputs in one analysis run
- [ ] 3.3 Enable `weighted_coverage` when scores are present; keep explicit unavailable marker otherwise
- [ ] 3.4 Update `scripts/collect_traces.py` with a score-capture flag, v3 output, and updated preflight assertions
- [ ] 3.5 Add fast schema tests: v3 roundtrip, score/ID alignment invariants, mixed-version rejection

## 4. Enriched SWE-agent Metadata

- [ ] 4.1 Implement `trajectory_phase`, `tool_type`, `event_outcome` heuristics in `dataset_adapters/swe_agent.py`; record heuristic version in manifest
- [ ] 4.2 Emit `dag_depth`, `group_local_step_index`, `on_critical_path`
- [ ] 4.3 Refine segment roles to system / code_context / trajectory_event / tool_result
- [ ] 4.4 Scale SWE workload to 1k records with >= 64 source groups; keep dependency edges and ready times valid
- [ ] 4.5 Mark underivable fields unavailable in the manifest instead of defaulting
- [ ] 4.6 Add adapter unit tests for new fields and group/dependency invariants

## 5. Predictor Upgrades

- [ ] 5.1 Implement `hybrid_routesig_temporal` with per-layer confidence-weighted fusion
- [ ] 5.2 Implement sparse-key smoothing and per-(block_type, layer) confidence calibration using train split only
- [ ] 5.3 Implement per-layer gating on train-split `delta_vs_global`; persist gate decisions per layer
- [ ] 5.4 Extend RouteSig fallback order with `tool_type` and `trajectory_phase` levels; fix the order in code and spec
- [ ] 5.5 Implement `learned_ranker` (logistic regression or GBDT) over pre-router features with predict-before-update online evaluation
- [ ] 5.6 Add score-weighted RouteSig statistics mode alongside count-based mode
- [ ] 5.7 Fast tests: fusion math, gating behavior, calibration boundaries, ranker feature construction uses no post-router inputs

## 6. Statistical Evaluation

- [ ] 6.1 Add bootstrap CIs (>= 1000 group-level resamples) for predictor hit-rate deltas
- [ ] 6.2 Add a second group-preserving time split; require reporting on both splits
- [ ] 6.3 Implement the metadata ablation matrix (full; -role; -phase; -tool_type; -block_type; -positions; length-only; temporal-only; RouteSig-only)
- [ ] 6.4 Report per-layer signal maps (which layers pass gating) for Stage B targeting
- [ ] 6.5 Keep analysis runtime tractable at 1k/5k (extend v8 optimizations as needed)

## 7. Workload Conversion and Trace Collection (Real Runs)

- [ ] 7.1 Convert all five datasets at 1k records (SWE-agent per task 4.4)
- [ ] 7.2 Collect Qwen3-30B-A3B v3 traces with router scores: 1k records for ShareGPT, LMSYS, SWE-agent, OpenCode, OpenMath
- [ ] 7.3 Collect DeepSeek-V2-Lite-Chat v3 traces with router scores: 1k records for ShareGPT and SWE-agent
- [ ] 7.4 Optional non-blocking: 5k Qwen3 runs for ShareGPT and SWE-agent
- [ ] 7.5 Store new artifacts with `_1k`/`_5k` suffixes; do not overwrite 256-record baselines
- [ ] 7.6 Record env reports and experiment logs per run, as in the previous change

## 8. Analysis, Reporting, and Gate

- [ ] 8.1 Run full analysis for all 7 pairs at 1k: prediction (all predictors and budgets), ablations, CIs, both splits, per-layer maps, weighted coverage
- [ ] 8.2 Regression-compare against 256-record v2 baselines for existing predictors
- [ ] 8.3 Evaluate the agent-metadata increment gate on SWE-agent (both models) and write the gate decision record under `docs/`
- [ ] 8.4 Keep claim-scope separation: agent evidence vs chat/prompt-only vs domain-instruction
- [ ] 8.5 Update README/docs for v3 schema, score capture, new predictors, and reproduction commands

## 9. Optional Second Agent Dataset

- [ ] 9.1 Decide orchestrator-generated trajectories vs second public corpus; record decision and rationale
- [ ] 9.2 If adopted: implement adapter/generator with native role/phase/tool metadata, dependency edges, ready times
- [ ] 9.3 If adopted: collect Qwen3 traces and include in gate evaluation as secondary evidence

## 10. Validation

- [ ] 10.1 `PYTHONPATH=. pytest -q tests` passes with new tests
- [ ] 10.2 `openspec validate add-hybrid-predictor-agent-metadata --strict` passes
- [ ] 10.3 vLLM output-equivalence smoke check (capture on vs off) recorded in logs
- [ ] 10.4 Real artifact validation snippet extended to v3/1k and passing for all 7 pairs
- [ ] 10.5 `git diff --check` clean
