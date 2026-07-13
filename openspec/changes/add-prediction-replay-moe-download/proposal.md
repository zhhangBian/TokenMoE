## Why

The current TokenMoE prototype shows that agent metadata can correlate with MoE
router choices, but the existing evidence is still built around a small
Transformers Tiny Mixtral trace path and scheduler replay that uses true routed
experts as an optimistic decision proxy. That is not enough to support the
intended research claim: TokenMoE should validate whether pre-router agent and
prompt-block metadata can predict real vLLM MoE expert working sets and improve
offline scheduling replay without changing model outputs.

The next pass must therefore move the experiment to real vLLM routed-experts
capture on downloaded MoE targets, normalize external prompt datasets into the
existing workload schema, and make RouteSig operate at prompt-block/token-segment
granularity. Online expert prefetch, vLLM scheduler modification, and proactive
EPLB replica placement are important follow-up goals, but they belong to the
next stage after this prediction/replay evidence is established.

## What Changes

- Use vLLM routed-experts capture as the only real trace collection path for
  this change. Remove the Transformers router-logit collection path from the
  active experiment pipeline.
- Collect prompt-only routed expert traces. vLLM may generate the minimum token
  needed to return outputs, but analysis keeps only prompt token routing.
- Add external dataset adapters in a dedicated folder, separate from core
  `tokenmoe/` logic. Each adapter converts one dataset into a standalone
  normalized workload JSONL with prompt block character spans, source
  provenance, and claim-scope metadata.
- Update `/home/youwei/bzh/dataset/download_dataset.py` only by changing
  `DATASET_LIST`; no other download logic may be edited.
- Add block/token-segment aware trace records and RouteSig predictions:
  predictions are per request, prompt segment, layer, and Top-M budget, and
  every predictor emits normalized expert weights.
- Persist model MoE-layer IDs and filter or mark non-MoE layers unavailable so
  zero-filled dense-layer slots from vLLM routed-experts buffers cannot become
  fake expert-0 demand.
- Use simple prediction baselines: global frequency, request LRU, sequence
  history, temporal window frequency, RouteSig, and oracle future demand as an
  upper bound only.
- Replace scheduler replay's true-active-expert decision proxy with
  prediction-based decisions. True vLLM routed experts are used only for
  post-decision scoring.
- Report prediction quality, fallback behavior, block-level locality, and
  scheduler replay metrics for baseline, RouteSig prediction, and oracle modes.
- Separate real-agent metadata evidence from chat/prompt-only and
  domain-instruction evidence. Non-agent datasets may support prompt/block
  locality but not the central agent-DAG metadata claim.
- Move prefetch replay and EPLB replay out of this change. They are documented
  as next-stage work, not implementation or validation requirements here. Legacy
  prefetch/EPLB simulator/report outputs must be removed from the main analysis
  path or labeled archived/excluded.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `tokenmoe-route-signature`: predict per prompt block/token segment and layer,
  expose confidence, fallback key, segment metadata, and budgeted expert sets.
- `tokenmoe-system-optimization`: require prediction-based scheduler replay
  driven by segment-aggregated predicted expert demand.
- `tokenmoe-evaluation-resources`: require real vLLM MoE prompt trace
  collection, external dataset conversion, and constrained dataset download
  updates.

## Impact

- Affected code: dataset adapter scripts in a dedicated folder,
  `tokenmoe/schema.py`, `tokenmoe/trace.py`, `tokenmoe/routesig.py`,
  `tokenmoe/metrics.py`, `tokenmoe/simulators.py`, `tokenmoe/reporting.py`,
  `scripts/collect_traces.py`, `scripts/analyze_traces.py`, tests, reports, and
  documentation.
- External file to change: `/home/youwei/bzh/dataset/download_dataset.py`, only
  its `DATASET_LIST`.
- External model paths used for validation:
  `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B` as the primary model,
  `/home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat` as cross-model
  validation, and `/home/youwei/bzh/model/mistralai/Mixtral-8x7B-Instruct-v0.1`
  as optional compatibility validation.
- External dataset targets:
  `anon8231489123/ShareGPT_Vicuna_unfiltered`, `lmsys/lmsys-chat-1m`,
  `nebius/SWE-agent-trajectories`, `nvidia/OpenCodeInstruct`, and
  `nvidia/OpenMathInstruct-2`.
- External normalized workloads and prompt-bearing traces should be written
  under a local artifact root such as
  `/home/youwei/bzh/dataset/tokenmoe_artifacts/`, not under tracked repository
  `data/` paths. Manifests record license, access, and redaction status.
- Unit tests may remain for pure data structures and formatting, but completion
  requires real vLLM MoE integration experiments and reports. Mock or synthetic
  traces are not sufficient completion evidence.

Confirmed implementation decisions:

- Main success standard: prove prompt-block agent metadata predicts real vLLM
  MoE prompt routing and improves prediction-based scheduler replay.
- Prompt trace scope: prompt tokens only.
- Dataset download constraint: edit only `DATASET_LIST` in
  `/home/youwei/bzh/dataset/download_dataset.py`.
- Dataset conversion output: one normalized workload JSONL per dataset, no mixed
  workload required in this change.
- Default workload size: 256 prompt records per dataset.
- Claim scope: SWE-agent trajectories are real-agent metadata evidence; ShareGPT
  and LMSYS are chat/prompt-only evidence; OpenCodeInstruct and
  OpenMathInstruct-2 are domain-instruction evidence.
- DAG replay evidence: only datasets with emitted dependency edges and ready
  times, currently SWE-agent-derived workloads, can support agent-DAG scheduler
  claims.
- Model coverage: Qwen3-30B-A3B over all five datasets; DeepSeek-V2-Lite-Chat
  over ShareGPT and SWE-agent-trajectories; Mixtral optional.
- Prediction granularity: per prompt block/token segment/layer, with token-span
  alignment from adapter-produced character spans and vLLM prompt token ID
  verification.
- RouteSig fallback order:
  `(agent_id, role, phase, block_type, segment_position)` ->
  `(role, phase, block_type, segment_position)` ->
  `(role, phase, block_type)` -> `(role, block_type)` -> `(block_type)` ->
  `(role, phase)` -> `(role)` -> `global`.
- Top-M budget: primary result uses `2 * router_top_k`; report budget curves at
  `1x`, `1.5x`, `2x`, and `3x` router top-k.
- Probability contract: every predictor returns normalized `expert_weights`
  aligned with predicted expert IDs.
- Demand unit: scheduler replay aggregates predicted demand as expert-token
  labels, `segment_token_count * router_top_k * expert_weight`.
- Baselines: global frequency, request LRU, sequence history, temporal window
  frequency, RouteSig, and oracle future demand.
- Evaluation protocol: per `(dataset, model)` group-preserving time-ordered
  70/30 split with predict-before-update online evaluation.
- Validation hard checks: reports must reject non-vLLM backends, fallback traces,
  mixed v1/v2 schemas, prompt/decode mixed traces, missing required coverage, and
  non-MoE layers treated as active experts.
