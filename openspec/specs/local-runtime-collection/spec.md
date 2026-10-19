# Local Runtime Collection

## Purpose
Define local-first Qwen3 runtime trace collection, resource and asset preparation, reproducible execution stages, and coverage-based delivery for TokenMoE.

## Requirements

### Requirement: Local-first asset preparation

The local collection entry point SHALL default to `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B` and `/home/youwei/bzh/dataset/princeton-nlp/SWE-bench_Verified`, with explicit path overrides supported. It SHALL check the model configuration, tokenizer, weight index, and all referenced shards, read the benchmark, and verify selected task IDs. Existing usable assets SHALL be reused. Missing or confirmed damaged target assets SHALL be restored through the existing parameterized download scripts. Authentication SHALL come from the environment or credential storage and SHALL NOT appear in plans, reports, or logs.

#### Scenario: Existing local assets
- **WHEN** all model shards are present, dataset files are readable, and all selected task IDs exist
- **THEN** the entry point reuses the existing paths, records verification results and version provenance, and does not download the model or benchmark again

#### Scenario: Missing Qwen shard
- **WHEN** a shard referenced by the weight index is missing
- **THEN** the entry point records the specific missing asset, restores it with a download job restricted to `Qwen/Qwen3-30B-A3B`, and verifies it again without downloading other models

#### Scenario: Missing benchmark
- **WHEN** no readable local SWE-bench Verified dataset file exists
- **THEN** the entry point uses the existing dataset download mechanism at the configured path and proceeds to collection only after successfully reading the task IDs

### Requirement: Machine and runtime preflight

Before launch, the executing agent SHALL record visible GPU UUIDs, models, free memory, and selected devices; verify that the TP device count matches the configuration; and check actual fork/virtual-environment/import paths, dependency versions, available data-disk space, and measured container-preflight results. Containers SHALL share the harness/engine host clock domain and enforce the configured limit of 2 CPU and 4 GiB per task. The entry point SHALL prefer devices with sufficient free resources, SHALL NOT terminate other tasks' processes, and SHALL NOT assume a fixed GPU count or an available remote namespace entry point.

#### Scenario: Historical GPU selection is occupied
- **WHEN** devices referenced by a historical configuration are occupied by other tasks while two other A100 GPUs meet the memory requirements
- **THEN** the executing agent selects the available devices, records their mapping, and leaves the occupying processes intact

#### Scenario: Runtime cannot enforce configured limits
- **WHEN** a container startup or cgroup probe fails to establish the configured resource limits
- **THEN** the result is retained as a preflight fault and the executing agent repairs or selects a working runtime without marking the stage as completed real-task collection

### Requirement: Explicit local Qwen protocol

The system SHALL provide a `local-qwen3-runtime-v1` profile using mini-swe-agent 2.4.6, SWE-bench Verified, and Qwen3-30B-A3B, defaulting to BF16, TP=2, context=32768, max_num_seqs=4, max_num_batched_tokens=4096, and gpu_memory_utilization=0.8, without YaRN overrides. It SHALL enable prefix caching, real decode, and routed-experts capture, disable async scheduling and speculative decoding, and use DP=1. Default sampling SHALL use temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, and base_seed=42. Thinking and tool-parsing settings SHALL be recorded explicitly and SHALL NOT change silently to improve submission rates.

Configuration and download scripts SHALL support explicit Qwen selection while preserving existing remote-model entry points. A dry run SHALL expand local paths, model/task lists, GPU/concurrency settings, limits, stages, and output directories without starting services, downloading assets, pulling images, or writing collection data.

#### Scenario: Local dry run
- **WHEN** the local Qwen profile is run in dry-run mode
- **THEN** the plan contains only the selected Qwen model and local task matrix, without expanding to three remote models or a 500-task full run

### Requirement: Bounded reproducible collection matrix

The system SHALL fix stages, task IDs, protocols, and parameters in `collection_plan.json`. The default pilot SHALL use the 3 Verified tasks in `local-debug.yaml` at N=1 with 10 steps/600 seconds per session. Primary small-scale collection SHALL use the 20 tasks in the existing `pilot.yaml` at N=1 and N=4 with 20 steps/900 seconds per session, producing 40 target sessions. Both primary settings SHALL use identical tasks, roles, base seeds, and serving configurations, with tool timeout=60 seconds, HTTP timeout=600 seconds, and request_retries=2. Pilot, smoke, equiv, and primary collection data SHALL be stored separately.

N=4 SHALL represent real execution with at most 4 concurrent sessions, independently of TP. Reports SHALL disclose actual overlap and SHALL NOT describe sequential execution or offline timeline shifts as measured concurrency. Time limits SHALL be documented with their actual semantics and measured wall time; checks at turn boundaries SHALL NOT be described as hard deadlines.

#### Scenario: Complete default matrix
- **WHEN** the default primary small-scale collection finishes
- **THEN** the manifest contains 40 valid planned units covering 20 tasks at N1 and N4, excluding pilot and recovery duplicates from target counts

#### Scenario: Concurrency reduced during debugging
- **WHEN** N4 encounters resource problems and the executing agent temporarily uses N2 to verify a repair
- **THEN** N2 results are recorded as a separate diagnostic configuration and the original N4 units remain outstanding

### Requirement: Staged real execution and configuration optimization

The executing agent SHALL complete resource preflight, necessary asset/image preparation, real smoke runs, capture equivalence, pilot runs, and primary small-scale collection, repairing collection faults before continuing. Smoke checks SHALL verify Qwen's 48 layers, 8-of-128 uint8 expert routing, real decode, tool output, and offline joins. Equivalence SHALL retain the existing 50-prompt off/off/on/on protocol. Parameter optimization SHALL respond to specific execution faults or measurable overhead, recording each configuration change, its rationale, and comparison results. Changes to effective serving parameters, weights, or capture behavior SHALL require equivalence verification for the new configuration.

#### Scenario: Real startup runs out of memory
- **WHEN** the server fails to start because of confirmed context/batch memory pressure
- **THEN** the executing agent preserves failure logs, adjusts local parameters, verifies the repair in a new directory, and generates an equivalence record matching the actual collection configuration

#### Scenario: SWE tests fail during pilot
- **WHEN** the pilot model and tools execute normally but tests run by the agent fail
- **THEN** the executing agent preserves original commands, outputs, exit codes, and subsequent requests and continues the remaining stages without treating the test result as a collection gate

### Requirement: Collection manifest and deliverable

Each experiment SHALL use a separate directory under `/home/youwei/bzh/dataset/tokenmoe_artifacts/` and preserve `collection_plan.json`, `collection_manifest.json`, `source_manifest.json`, `recovery_events.jsonl`, stage logs, raw and finalized data, validation/alignment results, runtime reports, and artifact checksums. Coverage units SHALL be identified by `(benchmark_item_id, concurrency, base_seed, protocol_id)` and record `pending/running/valid/needs_recollection/blocked` status, valid artifacts, session IDs, and fault provenance.

Completion SHALL require both planned coverage and data validity. Missing devices, permissions, or credentials that prevent independent recovery SHALL be recorded explicitly. Partial delivery SHALL list outstanding units; counts of successful directories SHALL NOT replace task-matrix verification.

#### Scenario: Interrupted run resumes
- **WHEN** N1 has valid traces for all targets but some N4 tasks remain incomplete after server failures
- **THEN** recovery processes only the missing or invalid N4 units, preserves every historical directory in the manifest, and checks both N1 and N4 coverage in the final report
