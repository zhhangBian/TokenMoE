## Context

Local collection follows the record formats in `collection/data_collect.md`, using the repository's vLLM fork and mini-swe-agent 2.4.6. The data supports examination of how roles, tool categories, and outcomes affect subsequent expert-demand distributions in `#idea.md`, while preserving inputs for tool-return-time research.

Verified local state from read-only inspection on 2026-10-19:

| Object | Verified state |
|---|---|
| Model | The index in `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B` references 16 weight shards, all present and nonempty, totaling approximately 61.07 GB. Weight loading and content integrity have not yet been verified |
| Architecture | Local `config.json` specifies 48 layers, 128 routed experts, top-k=8, no shared experts, and an original maximum position of 40960 |
| Dataset | `/home/youwei/bzh/dataset/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet` is readable and contains 500 tasks. All 20 IDs in the existing pilot list are present |
| Devices | Four 80 GB A100 GPUs were visible. GPUs 1 and 2 were mostly idle; GPUs 0 and 3 each used approximately 40 GB. Occupancy must be checked again before execution |
| Environments | `tokenmoe-harness`, `tokenmoe-vllm`, and `tokenmoe-vllm030` virtual environments exist. Their presence does not establish dependency or fork-source compatibility |
| Code | Fork HEAD was `7665341a1eeb561a964531867341f01b22d1b09f` with a clean worktree. Download and configuration scripts do not yet include Qwen in `MODEL_NAMES` |
| Historical data | The `qwen3-feasibility-20260930` report records 12 requests, 5 tool calls, actual routing across 48 layers, passing structural rules, a 90.57% message-field location rate with missing reasoning alignment, and no formal equivalence record. These are historical reported results, not rerun results |

Assumptions requiring a real smoke run or pilot: the selected virtual environment can load the local weights; selected devices can run TP=2; the container entry point can enforce 2 CPU/4 GiB limits and share the host clock; current Qwen reasoning handling, tool parsing, and prompt alignment satisfy collection requirements; and N=4 produces actual overlapping work. Historical data cannot replace these checks.

## Goals / Non-Goals

**Goals:**

- Deliver a reproducible small-scale Qwen runtime dataset linking roles, requests, tool output streams, engine steps, and actual expert demand.
- Preserve task-failure traces, repair collection infrastructure, and identify the tasks and concurrency settings with valid collected data.
- Support adaptation to local resources, append-only recovery after failures, and runtime statistics recomputable from raw records.

**Non-Goals:**

- Implementing expert predictors, progress-estimation models, residency simulators, offload, or request scheduling.
- Using a single-role bash agent dataset as evidence for G2/G3, cross-model transfer, or actual transfer acceleration.
- Gating data validity on SWE-bench scores or submission rates, or requiring the scoring harness.

## Decisions

### 1. Reuse existing entry points with a separate local profile

Extend `download_assets.py`, `prepare_configs.py`, and `run_collection.py` while preserving existing remote-model workflows. Add the `local-qwen3-runtime-v1` profile and configuration templates so that the 500-task full template cannot be selected accidentally for local collection.

Use a separate local model configuration, preserving the historical 131072/YaRN configuration in `qwen3-30b-a3b.yaml`. Start with BF16, TP=2, `max_model_len=32768`, `max_num_seqs=4`, `max_num_batched_tokens=4096`, and `gpu_memory_utilization=0.8`, without YaRN overrides. Retain temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, and base_seed=42. These are initial settings requiring execution checks; a configuration file alone does not demonstrate feasibility.

Enable prefix caching, real decode, and routed-experts capture. Disable async scheduling and speculative decoding, with DP=1. Start with an existing eager/default execution path that passes actual-routing and equivalence checks. Give every CUDA Graph, batch/context, or memory-setting adjustment a new configuration fingerprint and experiment directory.

A separate profile preserves historical protocols and comparable records. A new Transformers collector would lose the existing engine timeline, while single-token prompt tests cannot provide decode or tool-loop data.

### 2. Prefer local assets and download only what is missing

Preflight records hashes of the model configuration, tokenizer, and weight index; shard names and sizes; dataset hashes and revision (explicitly unknown when unavailable); task lists; main-repository/fork commits and uncommitted patch hashes; dependency versions; actual virtual-environment/import paths; GPU UUIDs/models and free memory; and measured container-preflight results. Verify weights through the index, safetensors headers, and actual loading.

Reuse the download mechanisms in `/home/youwei/bzh/model/download.py` and `/home/youwei/bzh/dataset/download_dataset.py` through the parameterized `collection/scripts/download_assets.py` entry point. Add Qwen selection and checks for missing assets. Download only missing or confirmed damaged target assets, without invoking legacy top-level loops that traverse unrelated resources. Reuse an existing benchmark and process only images for this run's tasks. Read authentication from the environment or existing credential storage, excluding credentials from download logs and source manifests.

Prefer two idle 80 GB A100 GPUs for TP=2. Recheck device identity and occupancy before execution, leaving other tasks' processes intact. GPU indices are current candidates rather than assumptions for future runs. Use an available docker/podman runtime without depending on the remote DSW keeper. If resources are insufficient, reduce runtime settings or select other available devices and record the changes. Record missing devices, credentials, or permissions when independent recovery is impossible.

### 3. Fix the task matrix and separate pilot data

| Stage | Tasks and concurrency | Session budget and purpose |
|---|---|---|
| smoke | One real generation request and the existing 3 local tool tasks | Check 48 bound layers, valid expert IDs, decode, tool streams, and finalize; classify as diagnostic data |
| equiv | Existing fixed 50 SWE-bench first-step prompts, off/off/on/on | Greedy, max_tokens=256; retain comparisons from 200 real requests |
| pilot | The 3 Verified tasks in `local-debug.yaml`, N=1 | 10 steps and 600 seconds per session; verify benchmark containers, reasoning, routing, and output provenance |
| local collection | The 20 fixed tasks in `pilot.yaml`, at N=1 and N=4 | 20 steps and 900 seconds per session; 40 target sessions using real execution at both settings |

Write expanded task lists, limits, and sampling settings for every stage to `collection_plan.json` before launch. Both primary concurrency settings use identical tasks, role templates, base seeds, and serving parameters, with separate server lifetimes and recorded engine instances. Use tool timeout=60 seconds, HTTP timeout=600 seconds, request_retries=2, and 2 CPU/4 GiB containers. mini-swe-agent checks its time budget at turn boundaries; an in-flight model call or tool can extend wall time beyond that value. Report actual elapsed time.

N=4 means at most 4 concurrent sessions, independently of TP. Report actual request/session overlap and batch composition; do not synthesize N4 data by shifting N1 timelines. When infrastructure fails, complete N1 where possible and repair N4 before collecting its remaining units. An N2 fallback must record the plan change and outstanding N4 targets without claiming completion of the original matrix.

### 4. Preserve existing engine hooks and routing semantics

Engine hooks use the corresponding files at pinned upstream baseline `v0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`), with the actual fork commit/patch recorded in the manifest:

| Hook | Baseline file and current fork integration |
|---|---|
| Startup validation, trace-directory export, and step dispatch/completion timing | EngineCore initialization/step paths in `vllm/vllm/v1/engine/core.py`, calling the fork-added `vllm/vllm/tokenmoe_trace.py` |
| Request lifecycle, scheduled ranges, preemption, first token, completion, and routing-slice attribution | schedule, update_from_output, preempt, and finish paths in `vllm/vllm/v1/core/sched/scheduler.py` |
| Original Qwen MoE router results and bound-layer map | `vllm/vllm/model_executor/models/qwen3_moe.py` and `vllm/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` |
| Current-step routing device-to-host data and slot mapping | Capturer initialization and model-output paths in `vllm/vllm/v1/worker/gpu_model_runner.py` |
| Reasoning-field adaptation for the chat template | Assistant-message parsing in `vllm/vllm/entrypoints/chat_utils.py`, checked against the local tokenizer template and actual engine prompt |

`tokenmoe_trace.py` is a fork addition absent from the baseline tag. Reuse these hooks and repair them only when execution evidence identifies a defect. All fork edits remain Python-only, preserving router arithmetic and actual routing without a Transformers fallback.

Retain routing arrays of shape `[R,48,8]` with uint8 IDs and 8 distinct IDs in `[0,128)` per row, checking the layer axis against the actual `layer_map.json`. `token_positions` identifies first computed positions; prefix hits produce no fabricated rows, and recompute is excluded from duplicate demand counts. The final sampled token of a complete generation has no routing row. Validate abnormal requests against their actual computed interval. Engine-step times are wall-clock intervals, not per-layer CUDA kernel latency.

### 5. Evaluate task outcomes, data validity, and coverage separately

A planned unit has key `(benchmark_item_id, concurrency, base_seed, protocol_id)`. `collection_manifest.json` records `pending/running/valid/needs_recollection/blocked` status, artifact paths, session IDs, task terminal states, and reasons. A unit is covered when it has at least one valid execution; recovery duplicates do not increase target counts.

Preserve submissions, incorrect solutions, failing pytest runs, nonzero tool exits, format errors, and budget exhaustion. A terminated session with real requests and valid computation records can count toward coverage. Startup failures without real inference, lost engine records, corrupt files, and incomplete finalization require repair/recollection. Report missing tool-consumption relationships or decode demand in otherwise valid sessions without inventing transition samples.

Recovery follows this sequence: preserve original inputs/outputs/logs and configuration, identify the failing component, repair it, run a bounded check with the original failing input, and recollect affected units. HTTP retries retain existing step/attempt semantics. Server restarts and task reruns use new application/session/request IDs and directories, linked through `recovery_events.jsonl`. Each event records time, fault category, old/new IDs and paths, command/parameter summaries, repair version, and verification result. Failures elsewhere do not trigger recollection of already valid units.

### 6. Align reasoning against the actual engine prompt

Preserve original assistant reasoning, actual HTTP messages, chat-template/tokenizer fingerprints, and engine token IDs. Investigate field adaptation, template wrappers, newline transformations, and message order. Repair serialization/alignment defects without changing model prompts or disabling thinking to improve alignment metrics.

If the effective template omits historical reasoning in the corresponding context, prove that behavior with fixed-input template evidence and actual engine prompt token IDs. Record a `template_omitted` diagnosis with null prompt/token ranges and a non-null `alignment_error`, retaining the original reasoning in harness records. Unproven omissions remain alignment failures. Preserve the existing `location_rate` definition in `alignment.json` and add omission counts, unexpected failures, and `rendered_location_rate = located / (message_fields - template_omitted)`. Only proven template omissions can be excluded from the latter denominator. Lost tool payloads do not qualify for this exception.

Local acceptance requires at least 99% rendered-field alignment, with original location rates and all omission evidence disclosed. This local exception does not change the formal Hopper protocol's existing alignment threshold. The two rates coincide when there are no template omissions.

### 7. Derive reports from raw records and state research limits

Add collection-operation records and `reports/` alongside the existing `static/raw/runtime` layout without rearranging raw formats:

```text
<experiment>/
  collection_plan.json
  collection_manifest.json
  source_manifest.json
  recovery_events.jsonl
  smoke/  equiv/  pilot/  collection/N1/  collection/N4/
  reports/runtime_summary.json
  reports/runtime_summary.md
  reports/artifact_checksums.json
```

Each independent run directory retains `static/`, `raw/`, and `runtime/` and can contain multiple sessions. Document concrete CLI arguments in README/HANDOFF during implementation, including a dry-run that expands Qwen tasks and paths. Proposed arguments are not presented as existing interfaces.

Report task coverage, outcomes and fault categories, request/tool/step/chunk counts, prompt/output/cached/computed/recompute tokens, and expert lookup counts by layer and new_prefill/decode phase for each concurrency setting. Recompute lookup histograms from first-computation routing rows without claiming prediction hit rates or cache acceleration. Separately report coverage linking tool-output payloads to actual new prefill, retaining cache-hit labels for cached payloads.

Time reports include tool duration, time to first output, longest silence, output bytes/chunks, tool completion to consuming-request intervals, queueing time, time to first token, and generation intervals, with p50/p95. Report null and a reason for missing endpoints or undefined denominators. Generation intervals are request wall-clock metrics and cannot establish pure GPU compute time or directly observed per-token latency. Retain 1 Hz host-load sampling; preserve null GPU metrics and report missing telemetry.

Write versioned derived labels under `derived/labels_v<N>/`: tool categories from bash commands, outcome categories supported by exit status/output evidence, and first-step/tool-switch markers. Use unknown when classification is uncertain. Final exit codes, final outcomes, and future logs are supervision labels unavailable at tool issuance; preserve `available_at` and labeling-rule versions. State that the dataset has one generalist role. G1 groups by task ID, placing N1/N4, pilot, and recovery copies of the same task in one group; G2/G3 are not covered. Creating training sets or fitting predictors is unnecessary in this change.

### 8. Preserve equivalence requirements and measure overhead

Valid local data must pass structural rules 1–9, 11, and 12, while 10, 13, and 14 remain not_applicable. Retain the 50-prompt off/off/on/on capture-equivalence protocol, rerunning it after configuration, weight, or fork-behavior changes. Retain failed checks as diagnostic data and repair numerical/capture defects without weakening thresholds or fabricating passing records.

The local summarizer defaults to a submission-rate threshold of 0, preserving explicit thresholds for formal modes and reporting the selected profile and thresholds. Measure overhead with bounded capture-off/on comparisons using identical hardware, parameters, and fixed requests. Record measurement scope, repetitions, and queueing effects. Record, investigate, and optimize overhead above 5% through bounded comparisons while retaining structurally valid local data for delivery. The formal Hopper requirement of less than 5% remains in force.

## Risks / Trade-offs

- [40 sessions and one role cannot support every research claim] → Report task/tool/outcome and transition coverage with G1 limits; make no G2/G3 or unseen-tool generalization claims.
- [Some model-failure sessions are short] → Preserve their traces and disclose usable transition counts; repair transport/format handling defects without selecting successful tasks to improve submission rates.
- [A100 occupancy or long contexts cause OOM] → Recheck free memory, reduce context/batch settings in a separate configuration, and rerun its equivalence check while preserving other tasks' processes.
- [Tool logs are buffered] → Retain observed chunk-read times and raw bytes, distinguish silence from lack of progress, and avoid inventing tool-internal print times.
- [Abrupt termination prevents writer flush] → Preserve partial/raw records, clean up this run's processes/containers, inspect missing data, and recollect affected units rather than accepting corrupted fragments as complete terminal traces.
- [Parameter optimization changes the data distribution] → Retain protocol/configuration fingerprints and store different budgets or thinking settings separately.

## Migration Plan

1. Update the collection specification and historical execution-ownership guidance; extend existing scripts/local configurations and run targeted CPU checks for recovery and reporting behavior.
2. Execute local preflight, restore necessary assets, and run smoke, equiv, and pilot stages, retaining diagnostic results and repairing collection defects.
3. Fix a working configuration, complete the 20-task N1/N4 matrix, resume missing units from the manifest, and finalize, validate, report, and checksum the data.
4. Deliver valid directories, outstanding units, and recovery provenance. To roll back, stop this run's processes and restore the prior configuration while retaining all data and preserving historical/remote results.

## Open Questions

There are no blocking decisions for the user during specification authoring. Select devices and the container entry point, diagnose reasoning alignment, and determine stable parameters during real execution, recording findings in the manifest. Use the defaults above for task counts and budgets; later adjustments require a new protocol record.
