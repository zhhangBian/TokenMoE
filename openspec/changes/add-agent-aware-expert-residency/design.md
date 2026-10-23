## Context

The user selected local Qwen validation followed by larger-model work and requested joint optimization of latency, throughput, and GPU memory. This change defines the implementation and experiment contracts; authoring these artifacts does not start implementation or experiments.

### Verified inputs

- Primary dataset: `/home/youwei/bzh/dataset/tokenmoe_artifacts/qwen3-runtime-20261019-063626/`, especially `reports/runtime_summary.md`, `reports/data_notes.md`, and `collection/{N1,N4}/attempt-01/{N1,N4}/`.
- There are 20 task identities, 40 primary sessions, 684 admitted requests, and 691,847 first-computed routing positions. One evidenced pre-engine rejection has no routing. Qwen uses 48 MoE layers, 128 routed experts per layer, and top-k=8. Only one generalist template and bash tool interface are represented.
- Tool observations contain 1,021 calls: 404 have zero output chunks, 511 have one, and 106 have at least two. Only 16 have first output at least 0.5 seconds before completion; seven run longer than five seconds. Existing data cannot establish general log-based progress prediction or G2/G3.
- The source baseline is vLLM v0.30.0, `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; the inspected trace fork is `7665341a1eeb561a964531867341f01b22d1b09f`. Its modular runner exposes routing before expert execution; its offloaders operate on parameters or layer groups. The trace recorder requires synchronous scheduling.

### Exploratory evidence, pending packaged reproduction

Read-only analysis in the design discussion held out each task together with its N1/N4 copies. For each request phase it selected 64 experts per layer using training-only counts or preceding same-session requests. Mixtures used normalized histograms with fixed 0.5 weights; tool keys used the sorted set of existing command-category labels, with fewer than eight training requests falling back to the global prior. A missing-tool key was not equivalent to the first-request flag. Ties used stable ascending expert-ID sorting and selected the final budget entries.

| Fixed-set lookup coverage | New prefill | Decode |
|---|---:|---:|
| Global prior | 86.4014% | 87.5611% |
| Previous request | 85.6853% | 89.2687% |
| Previous + global | 87.7591% | 90.6597% |
| Previous + tool categories | 89.0288% | 91.1638% |
| Previous + first/subsequent-stage prior | 89.7669% | 90.9441% |
| Per-request, per-phase hindsight fixed-set oracle | 94.749% | 94.439% |

On noninitial requests, previous+tool improves over previous+global by approximately 0.10 percentage points for prefill and 0.23 for decode. These are lookup-weighted screening statistics, not cache-hit, stall, or speedup measurements. The oracle chooses one set per request phase and is not a dynamic cache oracle. These results are not a locked confirmation test because the tasks have already informed design; later confirmatory experiments require untouched task groups.

## Goals / Non-Goals

**Goals:**

- Reproduce and challenge the observed opportunity using strong causal baselines.
- Implement exact expert offload and speculative prefetch within the existing fork, with explicit memory and transfer budgets.
- Isolate gains attributable to agent metadata from generic caching, request-stage priors, and temporal smoothing.
- Measure fixed-memory latency/throughput and minimum memory under predeclared latency constraints, including task-level completion behavior.
- Establish expansion protocols for long tools, multiple roles, unseen tasks/workflows/templates, and a larger model after local verification.

**Non-Goals:**

- Changing router top-k, routing weights, model weights, prompts, or tool observations to improve cache statistics.
- Approximate substitution, dropping cold experts, training the language model, or treating failed solutions as invalid traces.
- Implementing NVMe offload, EP replica placement, a new request scheduler, or KV eviction in the first backend.
- Promising that prefetch never regresses or that outputs are automatically bit-identical under changed batching.

## Decisions

### 1. Separate research logic from the GPU backend

Implement a small `optimization/tokenmoe_opt/` package with analysis, predictor, replay, controller, and experiment modules. Keep the existing harness in `collection/`; add an optional event sink there. GPU storage and execution live in the fork, with compact versioned messages between processes. CPU analysis and controller tests must not import vLLM. Start with command-line entry points rather than a new service platform.

```text
collection events -> session state -> demand + arrival predictors
                                         |
actual scheduler state ----------------> shared residency planner
                                         |
                                  per-worker plan mailboxes
                                         |
CPU expert store -> copy stream -> fixed GPU slots -> actual MoE execution
                                         ^                 |
                                  real router demands      |
                                         +--- feedback ----+
```

Alternative: put all analysis in `collection/` or modify the model class directly. This would couple research policies to capture and make baseline swaps difficult. Qwen's model class should need little or no policy-specific logic.

### 2. Use causal, versioned features and model-specific predictor state

Each feature and event carries its source time, controller receipt time, clock domain, version, and provenance. Availability is the later of source observation and receipt in a common clock domain. Historical data without receipt timestamps uses a documented conservative proxy and cannot establish transport latency. The old finalized `issued_at` is not proof that a live controller already knew the parsed command at that instant.

Training/evaluation groups are benchmark tasks, including all N1/N4, pilot, recovery, and duplicate executions. Prior state is keyed by model/weight revision, tokenizer/template version, layer, phase, and feature version. Fit smoothing, backoff, category rules, and mixture weights only inside training folds. Report both frozen unseen-task evaluation and explicitly labeled causal online adaptation; neither can read future requests in the test session. Shared test-session adaptation, if tested, is a separate chronological protocol.

Start with global, first/subsequent-request stage, previous-request, request-EWMA, previous+global, previous+stage, and previous+tool predictors. Include a diagnostic recent-token window, but do not assume it is stronger. Separate new prefill, full decode, and first-32-computed-decode-position targets; the latter is not TTFT. Preserve no-tools, first-request, multiple-tools, and unknown-category states independently.

Raw first-computed positions drive frequency learning. Operational execution demand additionally includes real recomputation; frozen replay must reject an unsupported recompute case rather than pretend it has no cache cost. Router scores are unnecessary and remain absent from capture.

Alternative: begin with a learned semantic model. Current incremental gains and label coverage justify interpretable tables first. Outcome-conditioned hindsight is a diagnostic; deployable outcome features require observations available before the decision.

### 3. Predict arrival distributions and represent dependencies

For one outstanding tool, start with an empirical duration/survival distribution conditioned on the available category and elapsed runtime. Completion and timeout observations update the state; censored timeout durations are not ordinary completed-duration samples. For several tools, use the observed execution/dependency structure: a parallel join waits for every required output, while a sequential queue includes pending tools. Tool completion does not guarantee immediate GPU service; queued requests use actual scheduler state. Terminal or cancelled sessions contribute no future demand.

Use broad distributions and a no-predictive-prefetch option for unsupported histories. A later stream-feature predictor consumes only bytes already observed. A small log-reading language model is an optional experimental predictor whose CPU/GPU cost, interference, coverage, and timing calibration must be measured separately. Synthetic sleeps are useful for mechanism tests but do not count as real long-tool evidence.

### 4. Plan shared residency using expected avoided stalls

At a plan epoch, aggregate model-specific per-layer demand from active and waiting sessions using expected token volume, phase, service availability, and arrival uncertainty. Active sessions are not assigned identical demand merely because they are active. Histogram expectations propose candidates; known scheduled batches use their actual unique expert demands, and replay accounts for batch-level deduplication and reuse.

The planner ranks candidate changes by estimated critical-path wait avoided minus exposed transfer cost and displaced reuse. Cost profiles are measured per device/TP group and include copy setup, bytes, concurrency, and compute contention. Existing residency, transfer reservations, demand pins, and a minimum-residence/hysteresis rule limit churn. Store the policy version, cost-model version, prediction confidence, candidate bytes, and reason with each plan.

Use one resource ledger per engine/TP group, with per-rank memory limits and shared-link constraints. Demand loads take priority over speculative loads; an already issued DMA is not assumed instantly cancellable. Stop admitting speculative work when confidence, bandwidth, or resource limits are insufficient. Expire stale plans and fall back to the selected reactive policy. This bounds speculative work; it does not establish a no-regression theorem.

Do not create a KV policy. If a selected deployment has KV transfers, include their observable reservations in the bandwidth ledger or explicitly mark joint arbitration unavailable and run isolated contention experiments.

### 5. Use an explicit expert memory budget and bounded loading

Initial support is Qwen3-30B-A3B BF16 with unquantized modular Triton MoE, TP=1 or 2, EP=1, DP=1, no speculative decoding, and synchronous scheduling. TP=2 is the default local comparison. Unsupported quantization, monolithic kernels, EP/EPLB, or overlapping weight-offload configurations fail before loading. Shared-expert/quantized adapters are later model-specific work.

Expose a default-off `expert_cache` configuration with per-rank total expert bytes, per-layer slot allocation, any demand/staging reserve, reactive policy, prefetch byte/inflight limits, and telemetry level. The budget includes resident and in-flight weight storage, GPU staging, required mapping state, and expert-execution scratch. Report other activation/workspace memory separately and include it in total GPU peak accounting. Fixed per-layer allocations are sufficient initially; an unequal allocation is another explicit policy under the same total bytes.

Load checkpoint expert shards into CPU storage without first creating a complete GPU expert bank. Preserve TP shard layout and gate/up/down ordering. Use bounded pinned staging when full pinned host storage is unavailable; record host pageable and pinned usage. Backend-specific weight transforms must be performed with bounded staging before an expert becomes usable. Dense weights remain resident.

For this Qwen configuration, an unsharded BF16 routed expert contains `3 * 2048 * 768 * 2 = 9 MiB`. All routed experts total 54 GiB; 64 per layer total 27 GiB, or 13.5 GiB per rank at TP=2 before reserves and nonexpert memory. These are shape-derived estimates, not measured process peaks. Merely lowering `gpu_memory_utilization` does not impose an expert budget.

Alternative: load everything on GPU then move cold experts to CPU. That would invalidate the claim that the same design can start a model whose weights do not fit and would hide initialization peak memory.

### 6. Recover actual demand without skipping computation

After the real router produces top-k IDs and weights, deduplicate the required logical experts, join existing loads, wait for every required shard, and invoke the existing modular computation. Logical-to-slot mappings are separate from EP placement. A missing expert is never represented as an EP `-1` entry passed through to a kernel that writes zero.

Require at least K usable slots in every layer for the initial token-microbatch executor. When a scheduled batch needs more distinct experts than fit, partition its tokens into deterministic, disjoint microbatches whose expert union fits. Preserve original top-k order and weights, scatter outputs to original token positions, and execute each token exactly once. Account for repeated loads between microbatches. Reject budgets that cannot execute one token; do not silently allocate an oversized full-layer bank. This is an exact-computation design, subject to the numerical checks below.

Slots follow `FREE -> LOADING -> READY -> IN_USE -> READY`; only unpinned READY slots can be evicted. Copy completion events gate publication, kernel completion events gate reuse, and generation/epoch tags prevent stale copies from installing into reused slots. Prefetch runs on a dedicated worker-owned copy stream, with bounded mailboxes and immutable command payloads. It must run while no `execute_model` call is active. Background copy threads must not issue unsynchronized TP collectives.

TP ranks install a common logical plan version and execute matching token partitions. Each rank waits for its own shard readiness before the agreed compute/collective sequence. Missing-shard or peer failure aborts the affected execution with evidence; it does not return partial output. Shutdown stops new plans and drains or records outstanding copies before freeing buffers.

The initial live 64-slot pilot exposes repeated prefill loads as a separate
execution cost. An optional expert-major prefill profile groups only the actual
router-selected token/expert pairs already available in the current batch. It
reuses the existing unquantized Triton expert operation and `moe_sum`, stores each
contribution at its original token/top-k position, and reduces in the original
order. Expert groups retain pins across bounded pair chunks; current-batch
frequency orders groups so hotter experts remain resident at the boundary.
This mode is default-off, reserves per-route output scratch explicitly, leaves
small batches/decode on the token executor, and requires fresh numerical, TP
failure and cost qualification. The one-layer algebra prototype is not backend
qualification. Graph support remains unqualified.

### 7. Integrate at verified fork seams

All paths below are repository-relative. Except the explicitly fork-added recorder, these hook files exist at baseline commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; implementation must record the actual fork revision and patches.

| Hook | Pinned source path / inspected behavior | Intended change and verification |
|---|---|---|
| Configuration | `vllm/vllm/config/offload.py`, `vllm/vllm/config/vllm.py` | Add independent default-off cache settings, compatibility validation, and compilation/config hashing; reject unsupported combinations before allocation. |
| Weight allocation and loading | `vllm/vllm/model_executor/layers/fused_moe/routed_experts.py`, `vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | CPU expert allocation, bounded transforms, physical slot parameters; smoke-test actual layout and startup peak. |
| Demand interception | `vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py`, `_apply_quant_method` | Insert readiness/microbatch handling between `router.select_experts` and `forward_modular`; keep routing arithmetic intact. |
| Logical routing feedback | `vllm/vllm/model_executor/layers/fused_moe/router/base_router.py`, `_select_experts` | Collect compact counts from actual selections with request attribution; avoid requiring full routing-file output. |
| Mapping and kernel behavior | `vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py`, `vllm/vllm/model_executor/layers/fused_moe/fused_moe.py` | Preserve EP meanings; validate slot indexing and prohibit passing a cache miss as an absent EP expert. |
| Worker lifecycle and idle prefetch | `vllm/vllm/v1/worker/gpu_model_runner.py`, `vllm/vllm/v1/worker/gpu_worker.py` | Initialize worker-owned stores/streams/mailboxes before memory profiling, accept plans while idle, and drain them on shutdown. The mailbox transport is new and requires a live smoke test. |
| Worker transport | `vllm/vllm/v1/executor/multiproc_executor.py` | Use bounded local control transport outside serialized model-execution RPC; verify TP ordering, idle delivery, and failure propagation. |
| Schedule attribution | `vllm/vllm/v1/core/sched/scheduler.py`, `vllm/vllm/v1/engine/core.py` | Associate request/session demand with actual batches and lifecycle; retain the existing scheduling policy. |
| Existing offload facilities | `vllm/vllm/model_executor/offloader/{base,uva,prefetch}.py` | Reuse suitable stream/lifecycle patterns; leave existing backends selectable as comparison baselines. |
| Trace extension | `vllm/vllm/tokenmoe_trace.py` (added at fork commit `7665341a1eeb561a964531867341f01b22d1b09f`) | Optional cache event export and measured intervals; retain the full recorder's stricter startup gates. |

Eager execution is the initial correctness profile. Performance comparisons must hold graph mode constant; also report the native full-resident graph-enabled reference so an eager-only comparison does not conceal the cost of graph loss. Piecewise graphs are an explicitly gated optimization with persistent addresses and replay tests. No claim of graph support follows from fixed addresses alone.

### 8. Separate evidence levels and correctness gates

Produce four labeled result classes: fixed-set lookup analysis; frozen-timeline cache replay; calibrated timing simulation; live measurements. Replay needs real unique expert demands per batch/layer, replacement and in-flight events, finite transfer resources, and explicit policies for prefill and recompute. Layer timing offsets come from calibration, not invented values or miss-delayed timestamps. Timing simulation advances compute, copies, dependent tool starts, and request returns after injected stalls; otherwise it remains a frozen-timeline diagnostic. Report calibration error on held-out executions.

Correctness gates cover cache-off behavior, exact expert/shard mapping, unchanged router outputs on identical inputs, tensor output tolerances fixed before evaluation, and greedy token sequences under matched deterministic schedules. Forced-token runs localize numerical errors separately from free-running generation. Record repeat-baseline variability; do not reuse the collection capture gate's allowance as permission for cache-induced errors. A configuration cannot advertise token identity if its measured output diverges. Diagnose it or report it as unqualified for that claim rather than weakening thresholds after seeing results.

Full routing capture is used for diagnostic runs. Runtime feedback uses device-side compact counts and bounded asynchronous export; avoid per-layer Python `.cpu()` or `.item()` synchronization solely for prediction. Profile demand synchronization, copies, kernels, observer overhead, and graph effects separately. Exact residency replay starts from recorded initialization/load events or a complete snapshot of slots, generations, and in-flight transfers. Counter/detailed telemetry modes disclose dropped events; incomplete detailed traces cannot qualify for exact replay.

For live latency measurement, add request/output-token-index observations at the existing scheduler output-processing seam in `vllm/vllm/v1/core/sched/scheduler.py` at the pinned baseline. Label these host token-readiness times; network delivery requires its own observation. Tokens processed together share a timestamp and batch identifier rather than invented individual timings. Average TPOT derived from endpoints is labeled separately from observed inter-token distributions. This instrumentation is independently selectable from full routing capture and must not change agent-visible messages.

### 9. Predeclare a joint evaluation matrix and expansion gates

Local defaults are concurrency N=1 and N=4, uniform slot targets 16/32/64/96/128 where physically feasible, and TP=2. Record the actual byte budget including scratch/reserves; a slot target alone is not a memory comparison. Baselines include full residency, existing UVA/layer prefetch where supported, demand-only LRU and LFU, global/stage/temporal residency, previous+global/stage smoothing, and agent-conditioned policies on the same backend. Layer-ahead and semantic predictors are separate comparison arms with measured inference cost. Oracles remain offline.

For mechanism isolation hold KV allocation, scheduler, graph mode, workload, and warmup state constant. For service-capacity experiments permit reclaimed expert memory to become KV capacity, hold the total GPU limit constant, and label that source of gain. Pair task/seed/protocol runs, repeat at least three measured trials after a fixed warmup, rotate policy order, and report task-level uncertainty and sample counts. Freeze held-out groups, repetitions, budgets, outcome handling, and latency targets before measurement.

Metrics include engine TTFT, completion of all required tools to first returned token, token emission timing/TPOT with batching disclosed, end-to-end task time, output-token and completed-task throughput, per-rank/aggregate initialization and steady-state GPU peaks, host memory, transferred bytes, exposed miss waits, wasted prefetch, and controller cost. Keep task outcomes separate; higher throughput caused by prematurely failed tasks is not accepted as a performance gain. Do not count synchronous output delivery as individually observed token emission.

Expansion requires a new frozen protocol and separate artifacts: real multi-chunk tests/builds with observed progress; at least two versioned role templates for cross-role effects; paired role sets with changed dependencies for G2; held-out templates and role types for G3. Before collection, specify coverage minima for durations, early-output lead, result classes, and tasks; report unmet cells rather than silently substituting short calls. Larger-model selection is based on actual resources, supported weight layouts, and local backend evidence, with its own correctness and bandwidth calibration. Existing DeepSeek/Dots/GPT-OSS collection support does not imply expert-cache support.

Local backend correctness and the controlled local comparison are required milestones. Positive agent-feature speedup is a research outcome, not a software acceptance requirement. If the incremental effect is absent, retain that result and continue only the separately justified expansion experiments; do not re-label generic caching gains as validation of agent prediction.

## Risks / Trade-offs

- Weak agent signal or coarse shell labels -> include stage/history smoothing baselines, audit compound-command labels, use unseen confirmation tasks, and disclose negative ablations.
- Short tools offer little log lead time -> retain the simple duration predictor and collect genuine long-tool workflows before evaluating a log reader.
- Lookup frequency differs from transfer/stall cost -> use batch-aware replay and real measured bytes/waits; keep lookup coverage as a separate metric.
- Python demand handling or microbatching erases gains -> measure overhead early, compare equal execution modes, and optimize only measured bottlenecks within the Python-only backend constraint.
- Full model fits on A100 -> enforce an explicit expert byte budget and distinguish capacity-constrained results from full-resident reference performance.
- Prefetch contention or cache churn hurts active requests -> reserve demand resources, bound speculative traffic, expire stale plans, and report regressions.
- CPU-first loading or graph replay is incompatible with an existing kernel layout -> restrict the supported backend matrix, fail clearly, and require layout/numerical smoke tests before broadening support.
- Multi-role and larger-model resources are unavailable -> keep the corresponding coverage gate unfulfilled with the resource requirement recorded; do not mark those tasks complete or fabricate evidence.

## Migration Plan

1. Implement analysis and reproduce historical screening without changing source data. Add versioned experiment manifests and update the collection specification for newly emitted fields.
2. Implement the default-off CPU/GPU backend and demand-only policies; pass loading, exact-execution, TP, overflow, and shutdown checks.
3. Add optional live events, simple predictors, idle prefetch, and the bounded planner; measure incremental cost and run controlled local comparisons.
4. Calibrate replay, evaluate phase/tool/time ablations, and create the expanded-workload and larger-model manifests at their gates.
5. Keep every protocol/backend revision separately identifiable. Disabling the cache and event sink restores the existing execution path. A live rollback stops speculative admission, drains current copies at a safe boundary, and retains demand-only execution; a capacity-constrained engine must not switch to a full-resident layout that exceeds memory.

## Open Questions

- Does the packaged reproduction confirm the small tool-feature increment after stage priors and compound-command labeling are controlled? Resolve during analysis; do not assume a positive answer.
- Can the pinned modular backend process slot-backed weights and idle TP prefetch within the overhead budget? Resolve with eager/local smoke measurements before graph work.
- Which available tasks provide sufficient real progress streams and role variation? Freeze their coverage manifest before the expansion run.
- Which larger model and quantized/shared-expert backend fit the available hardware? Select at the migration gate; model support requires an adapter-specific spec update if it exceeds the initial exact backend contract.
