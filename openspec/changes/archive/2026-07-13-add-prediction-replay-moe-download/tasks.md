## 1. Scope and Artifact Cleanup

- [x] 1.1 Confirm current-stage success standard: real vLLM prompt-only MoE traces plus block-level prediction and scheduler replay
- [x] 1.2 Confirm next-stage-only items: online expert prefetch, vLLM scheduler modification, and EPLB replica placement
- [x] 1.3 Confirm model priority: Qwen3-30B-A3B primary, DeepSeek-V2-Lite cross-model validation, Mixtral optional compatibility
- [x] 1.4 Confirm dataset download constraint: only edit `DATASET_LIST` in `/home/youwei/bzh/dataset/download_dataset.py`
- [x] 1.5 Confirm prompt trace scope: prompt tokens only, with minimal vLLM generation only to trigger outputs
- [x] 1.6 Confirm prefetch replay and EPLB replay are moved out of this change
- [x] 1.7 Confirm scheduler replay agent-DAG claims require dependency edges and ready times

## 2. Dataset Download and Adapter Inputs

- [x] 2.1 Update only `DATASET_LIST` in `/home/youwei/bzh/dataset/download_dataset.py` with ShareGPT, LMSYS-Chat-1M, SWE-agent-trajectories, OpenCodeInstruct, and OpenMathInstruct-2
- [x] 2.2 Run the existing dataset download script and record any gated dataset failures without changing script logic
- [x] 2.3 Add a dedicated dataset adapter folder outside `tokenmoe/` for raw-to-workload conversion
- [x] 2.4 Implement one converter per dataset: ShareGPT, LMSYS, SWE-agent trajectories, OpenCodeInstruct, and OpenMathInstruct-2
- [x] 2.5 Emit one normalized prompt workload JSONL per dataset under `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/`
- [x] 2.6 Emit a manifest recording source paths, output paths, sample counts, field mapping version, conversion command, unavailable fields, license/access status, and redaction status
- [x] 2.7 Preserve `source_index`, group ID, and timestamp fields where available for group-preserving time splits
- [x] 2.8 Mark each adapter output with claim scope: real-agent metadata, chat/prompt-only, or domain-instruction
- [x] 2.9 For SWE-agent-derived workloads, emit dependency edges and ready times when event order allows DAG reconstruction

## 3. Prompt Block and Token Span Schema

- [x] 3.1 Introduce explicit workload and trace v2 schema versions while retaining v1 readers for compatibility
- [x] 3.2 Extend workload metadata with prompt block spans including segment ID, block type, segment position, character start/end, source group ID, source index, timestamp when available, claim scope, and alignment status fields
- [x] 3.3 Update adapters so every normalized prompt is built from typed blocks and records character spans
- [x] 3.4 Extend trace schema to store prompt token spans per segment after tokenizer alignment
- [x] 3.5 Extend trace schema with model `moe_layer_ids`, prompt-only routing flags, backend, fallback status, and segment-unavailable reasons
- [x] 3.6 Mark records with failed or ambiguous span alignment explicitly rather than silently treating them as normal block-level traces
- [x] 3.7 Add analysis-time rejection of mixed v1/v2 inputs
- [x] 3.8 Add fast schema tests for workload JSONL roundtrip, span invariants, v2 version checks, and manifest shape

## 4. vLLM Prompt-Only Routed Expert Collection

- [x] 4.1 Remove the active Transformers router-logit backend from the trace collection CLI and docs
- [x] 4.2 Keep trace collection focused on vLLM `enable_return_routed_experts=True`
- [x] 4.3 Configure collection with minimal generation and `routed_experts_prompt_start=0`
- [x] 4.4 Truncate routed experts to `prompt_token_count` when vLLM returns prompt plus decode routing
- [x] 4.5 Verify tokenizer-aligned token IDs against vLLM `prompt_token_ids`; mark the whole record segment-unavailable on mismatch
- [x] 4.6 Derive and persist `moe_layer_ids`, then filter or mark non-MoE layers unavailable before trace writing
- [x] 4.7 Add vLLM preflight assertions for MoE model, routed-experts enabled, no pipeline parallelism, no context parallelism, no KV transfer/connectors, explicit TP/EP, dtype, max model length, and GPU memory settings
- [x] 4.8 Validate segment token spans against routed-expert token dimensions and MoE-layer IDs
- [x] 4.9 Collect Qwen3-30B-A3B prompt traces for 256 records from each of the five external workload files
- [x] 4.10 Collect DeepSeek-V2-Lite-Chat prompt traces for 256 records from ShareGPT and 256 records from SWE-agent trajectories
- [x] 4.11 Optional Mixtral-8x7B-Instruct compatibility validation skipped for this stage; primary completion does not depend on it

## 5. Segment-Aware Prediction Interface

- [x] 5.1 Add a shared prediction result structure for request ID, segment ID, layer ID, expert IDs, normalized expert weights, confidence, source, fallback key, scores, block type, segment position, and token span
- [x] 5.2 Add segment-aware RouteSig update and lookup keyed by role, phase, block type, segment position, and fallback levels
- [x] 5.3 Add global frequency, request LRU, sequence history, temporal window frequency, and oracle adapters using the same prediction result shape
- [x] 5.4 Define expert-weight conversion rules: frequency predictors normalize counts, Top-M-only predictors use explicit uniform weights, and oracle weights are used only in oracle mode
- [x] 5.5 Implement Top-M budgets as `1x`, `1.5x`, `2x`, and `3x` router top-k, with `2x` as the main report setting
- [x] 5.6 Add confidence, fallback-key, unavailable-span, unavailable-layer, and claim-scope reporting to prediction outputs
- [x] 5.7 Add fast logic tests for prediction shape, expert-weight normalization, fallback order, segment aggregation, and budget selection

## 6. Prediction Evaluation

- [x] 6.1 Implement per `(dataset, model)` group-preserving time-ordered 70/30 train/eval split by conversation, trajectory, workflow, or source item before request/segment expansion
- [x] 6.2 During eval, enforce predict-before-update online scoring for every predictor
- [x] 6.3 Report expert-label hit rate, exact-token hit rate, weighted coverage or explicit unavailability, and confidence calibration
- [x] 6.4 Report per-block-type, per-layer, segment-length bucket, and fallback-key metrics
- [x] 6.5 Report real-agent metadata, chat/prompt-only, and domain-instruction claim scopes separately
- [x] 6.6 Report baseline, RouteSig, and oracle results separately without replacing prediction metrics with oracle values
- [x] 6.7 Reject evaluation inputs that include non-vLLM backends, fallback traces, mixed v1/v2 schemas, non-prompt routing, or non-MoE layers treated as active experts

## 7. Prediction-Based Scheduler Replay

- [x] 7.1 Refactor replay so legal ready-node construction remains dependency-safe and reports dependency violations
- [x] 7.2 Aggregate segment-level predictions into request and batch expert-token-label demand using `segment_token_count * router_top_k * normalized_expert_weight`
- [x] 7.3 Score candidate batches using predicted fanout cost, predicted token-density cost, waiting cost, confidence gating, and starvation guard
- [x] 7.4 Use true vLLM routed experts only after batch selection for actual fanout, token-density, hit-rate, and oracle-gap scoring
- [x] 7.5 Compare FIFO, temporal baseline, RouteSig prediction, and oracle future demand modes
- [x] 7.6 Report MoE execution proxy, scheduling cost, policy comparison, low-confidence gated fraction, missing-span fraction, unavailable-layer fraction, and unavailable metric fraction
- [x] 7.7 Ensure non-agent datasets are not used to support agent-DAG scheduling claims; report them only as prompt/domain locality replay where applicable
- [x] 7.8 Mark scheduler replay unavailable for datasets that lack dependency edges or ready times instead of reporting arbitrary independent-request batching as agent scheduling
- [x] 7.9 Remove prefetch and EPLB simulator/report outputs from the main analysis path for this change, or label legacy outputs as archived and excluded

## 8. Documentation and Validation

- [x] 8.1 Update README and dataset/model docs with dataset download, adapter conversion, vLLM trace collection, prediction evaluation, and scheduler replay commands
- [x] 8.2 Update reports to state clearly that results are prompt-only vLLM MoE traces and offline prediction replay, not online runtime acceleration
- [x] 8.3 Document next-stage work for online expert prefetch, vLLM scheduler modification, and proactive EPLB replica placement
- [x] 8.4 Document concrete vLLM launch/preflight profiles for Qwen3-30B-A3B and DeepSeek-V2-Lite-Chat
- [x] 8.5 Run fast tests for schema, adapter, prediction, and replay logic
- [x] 8.6 Run required real-model validation: Qwen3-30B-A3B over five datasets and DeepSeek-V2-Lite-Chat over ShareGPT plus SWE-agent trajectories
- [x] 8.7 Inspect generated reports for prediction-vs-baseline-vs-oracle separation, no Transformers-derived main experiment results, no fallback traces, prompt-only routing, required dataset/model coverage, correct claim-scope separation, and no active prefetch/EPLB result sections
