## Why

The completed local Qwen traces make it possible to test whether agent information improves expert residency beyond strong history-based caching, before investing in a large-model implementation. The project needs a reproducible path from demand analysis to real vLLM offloading experiments that measures latency, throughput, and memory together and reports small or negative incremental gains.

## What Changes

- Add task-grouped, causal demand analysis with separate new-prefill/decode targets, request-stage priors, temporal baselines, tool-conditioned predictors, and explicitly labeled offline oracles. Reproduce the exploratory findings before treating them as experiment results.
- Add a shared residency controller driven by observed agent events, predicted demand, arrival-time uncertainty, measured transfer costs, and the actual scheduler state. Use a historical-duration predictor initially; evaluate streamed-log features and a small language-model reader only after collecting suitable workloads.
- Add an opt-in expert cache to the existing vLLM fork: CPU weight storage, bounded GPU slots, exact demand recovery, asynchronous prefetch, tensor-parallel coordination, and cache telemetry. Start with Qwen3-30B-A3B BF16, TP=2, modular MoE execution, and an eager correctness configuration.
- Add calibrated replay and controlled live experiments. Separate cache savings from agent-feature gains, measure fixed-memory performance and memory required at fixed latency targets, and expose throughput/memory/latency tradeoffs.
- Add separate expansion protocols for genuine long-running tools, multiple roles, G2/G3, and a resource-selected larger model. The current one-role short-tool dataset does not establish these results.
- Preserve historical captures and collection entry points. Keep raw routing authoritative, make control events optional, and keep predictor feedback independent of full trace recording.

## Capabilities

### New Capabilities

- `expert-demand-analysis`: Reproducible causal features, grouped evaluation, strong baselines, phase-specific demand predictions, and limits on conclusions from trace statistics.
- `agent-aware-residency`: Session event consumption, arrival-time distributions, aggregate demand, bounded prefetch planning, and reactive fallback.
- `expert-cache-runtime`: Exact expert execution using a bounded CPU/GPU storage backend in the existing vLLM fork, with loading, synchronization, overflow handling, and compatibility checks.
- `optimization-evaluation`: Replay calibration, correctness and measurement gates, joint performance/memory experiments, workload expansion, and evidence-based delivery.

### Modified Capabilities

- `agent-session-runner`: Optional live control events with causal timestamps and unchanged agent-visible tool observations.
- `engine-trace-recorder`: Clarify independent opt-in cache enablement and add versioned cache/transfer and token-readiness measurements without requiring full routing capture for runtime operation.
- `trace-dataset`: Add optional optimization-experiment artifacts, feature availability, and cache-event validation while retaining the historical routing layout.

## Impact

- Keep collection in `collection/`; implement analysis, predictors, replay, controller logic, and experiment commands in a separate `optimization/` package when implementation begins. Its CPU components must not import vLLM. GPU execution remains inside the `vllm/` submodule. Do not create empty implementation directories during proposal authoring.
- Engine changes target the pinned v0.30.0 fork, particularly MoE weight loading, the modular runner, worker lifecycle, and configuration. Initial changes remain Python-only and use existing PyTorch/Triton operations; compatibility with `VLLM_USE_PRECOMPILED=1` must be verified.
- Extend optional harness event emission in `collection/tokenmoe_collect/`. Maintain `collection/data_collect.md` alongside implementation: §1 (optimization scope), §3.1–3.2 (protocol/backend provenance and measured cost profiles), §4.3/4.5/4.6 (event/feature availability), §4.10–4.11 (cache and graph events), §5 (labels and split rules), §6 (derived artifact layout), §7–7.1 (validation and separate gates), §8.2 (expansion matrices), §10 (G1/G2/G3), and §11 (historical local-data boundaries).
- Update `openspec/config.yaml` to describe completed collection and the authorized optimization scope. Existing collection schemas remain authoritative until their implementation changes are documented; do not rewrite archived data or acceptance records.
- Store experiment outputs outside the repository, defaulting to `/home/youwei/bzh/dataset/tokenmoe_artifacts/`. Record selected devices and actual free resources for every live run. Proposal authoring starts no GPU jobs, downloads, or service processes.
- NVMe execution, EP replica placement, changes to router selection, approximate expert substitution, and a new KV eviction policy are excluded from the first backend. Interfaces and reports retain model/backend provenance so later backends can be evaluated separately.
