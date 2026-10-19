## Why

`#idea.md` requires agent roles, tool outputs, and tool return times to be linked to the next request's new-prefill and decode expert demand. Local Qwen3-30B-A3B weights and collection infrastructure already exist, but historical smoke runs have missing reasoning alignment and the scripts lack a reproducible small-scale Qwen collection workflow; this change provides real runtime data with fault repair and resumable collection.

## What Changes

- Add a local Qwen3-30B-A3B collection profile using mini-swe-agent 2.4.6 and SWE-bench Verified on one host: a 3-task pilot followed by 20 fixed tasks at N=1 and N=4, producing 40 primary small-scale sessions, with pilot data stored separately.
- Prefer `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B` and `/home/youwei/bzh/dataset/princeton-nlp/SWE-bench_Verified`. Check indexed weight shards, tokenizer files, dataset files, and task IDs; use existing download scripts only to restore missing assets.
- Add Qwen support to configuration and download entry points. Probe actual GPUs, free VRAM, virtual environments, and the container runtime, and save device mappings, effective parameters, versions, and task lists without inheriting remote DSW or historical GPU-count assumptions.
- Collect real multi-turn generation, per-token/per-layer expert IDs, engine steps, prefix-cache and preemption information, request lifecycles, tool commands and raw output streams, host load, and prompt provenance. Retain existing formats without router scores or inferred routing.
- Treat submission rate, SWE-bench correctness, and tool-command success as statistics. Retain valid traces from failing tests, incorrect solutions, format errors, timeouts, and exhausted step budgets. The executing agent diagnoses, repairs, verifies, and recollects affected tasks after collector, container, inference, alignment, or storage faults.
- Add a collection plan, recovery provenance, and runtime summaries that distinguish task outcomes, data validity, and task coverage. Summaries include token/routing coverage by phase, tool duration, chunk coverage, caching, queueing and generation time, and supported G1 scope.
- Preserve actual routing, structural validation, and capture-equivalence requirements. Measure local recorder overhead honestly; record and optimize values above the formal 5% target without discarding structurally valid local data solely for that reason.
- Deliver OpenSpec documents in this phase. During implementation, the executing agent performs local preparation, necessary downloads, GPU runs, repairs, and data delivery. The user's current authorization takes precedence over the historical instruction reserving GPU runs and downloads for the user.

## Capabilities

### New Capabilities

- `local-runtime-collection`: Local asset verification and downloads as needed, the small-scale Qwen collection matrix, staged execution, runtime-parameter optimization, and the final delivery manifest.

### Modified Capabilities

- `agent-session-runner`: Local execution profiles, classification of task outcomes and collection faults, and append-only recovery with provenance.
- `trace-dataset`: Local acceptance and runtime reporting, preservation of real failure traces, capture equivalence and data integrity, phase coverage, and G1 dataset boundaries.

## Impact

- Implementation scope: `collection/configs/`, `collection/scripts/{download_assets,prepare_configs,run_collection,summarize_runs}.py`, `collection/tokenmoe_collect/`, related tests, and operational documentation. Extend existing entry points rather than introduce another agent framework.
- Authoritative specification updates: `collection/data_collect.md` §1 (local scope and collection principles), §3.1 (local profile and version provenance), §4.1–4.3 (recovery provenance and evidenced pre-engine context rejections), §4.4 (reasoning alignment and evidence of template omissions), §5 (derived statistics and label availability), §6 (plans, recovery files, and reports), §7–7.1 (validation and local acceptance), §8.2 (3-task pilot and 20×2 matrix), §9 (local Qwen parameters), and §10 (local G1 boundaries). Existing routing and raw tool formats remain compatible; document new fields alongside their implementation.
- Address known issues such as reasoning alignment. Modify relevant Python code in `vllm/` only when real execution exposes a fork defect, and record its version; retain the inference backend.
- Store new experiments in separate directories under `/home/youwei/bzh/dataset/tokenmoe_artifacts/`, preserving historical smoke runs, remote primary experiments, and all raw failure data. Model and dataset scripts obtain authentication from the environment or credential storage without recording credentials in collection artifacts.
- Update stale local hardware and execution-ownership descriptions in `collection/README.md`, `HANDOFF.md`, `COLLECTION_STATUS.md`, and `openspec/config.yaml`. Retain the existing scope and acceptance requirements for formal Hopper collection.
- Predictors, tool-progress models, expert offload/cache scheduling, G2/G3, and performance-benefit experiments are outside this change; the local data supports subsequent small-scale research.
