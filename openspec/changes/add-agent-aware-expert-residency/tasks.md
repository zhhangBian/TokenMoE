## 1. Protocols and package boundary

The implementing agent owns local setup, verification, and authorized runs. These unchecked tasks describe future implementation; authoring the specification does not execute them. Expansion milestones remain separate from local completion.

- [x] 1.1 Add the `optimization/` package and CPU-only command entry points for analysis, replay, and experiment planning; verify it can be imported without vLLM or GPU initialization.
- [x] 1.2 Define versioned input, split, predictor, cost-profile, and experiment manifests with model/template identities, source hashes, and explicit milestone status; reject accidental output-directory reuse.
- [x] 1.3 Extend `collection/data_collect.md` in the sections listed in the proposal for optional control/optimization artifacts, feature availability, cache events, and validation, preserving historical collection contracts.
- [x] 1.4 Freeze the local evaluation protocol: task groups, exploration versus untouched confirmation data, budgets, N1/N4, TP, KV/graph modes, warmup, at least three measured repetitions, latency targets, and task-outcome handling.

## 2. Demand analysis and reproducible evidence

- [x] 2.1 Implement validated trace ingestion and phase histograms using actual positions/layers; verify cached gaps, partial requests, the uncomputed final sampled token, pre-engine rejection, and explicit recompute accounting.
- [x] 2.2 Implement grouped outer splits and training-only inner tuning; verify that N1/N4/pilot/recovery copies cannot cross splits and that online adaptation cannot read future observations.
- [x] 2.3 Implement versioned causal features, independent first/no-tool/multiple-tool states, and conservative historical receipt proxies; audit representative compound shell commands and preserve label-rule versions.
- [x] 2.4 Implement global/stage/previous/EWMA and recent-window baselines plus global-, stage-, and tool-smoothed histories, sparse-key backoff, deterministic ranking, and model-specific predictor artifacts.
- [x] 2.5 Implement budget curves, phase/first-32-decode targets, layer/transition subgroups, task-macro and lookup-weighted summaries, task uncertainty, and explicitly labeled fixed-set hindsight results.
- [x] 2.6 Reproduce the exploratory Qwen comparison in a separate output directory, explain differences, and deliver the agent-feature increments over both history+global and history+stage; retain small or negative effects.

## 3. CPU storage and cache state

- [x] 3.1 Add default-off cache configuration, backend compatibility validation, config hashing, per-rank byte accounting, and supported TP settings; verify unsupported modes fail before weight loading and cache-off behavior is preserved.
- [x] 3.2 Implement CPU-first BF16 expert loading with correct TP shards/projection order and bounded pinned staging/layout transforms; verify it does not allocate a complete GPU expert bank.
- [x] 3.3 Implement fixed-address per-layer slots, generation tags, reservations, readiness events, demand pins, transfer coalescing, and LRU/LFU policies; verify eviction and stale-copy behavior using the cheapest adequate tests.
- [x] 3.4 Implement initialization and steady-state expert/total memory accounting, including staging, scratch, host memory, and all TP ranks; verify that a too-small budget is rejected rather than expanded silently.

## 4. Exact MoE execution and backend qualification

- [x] 4.1 Insert cache demand recovery between real router selection and modular expert execution; verify missing experts are loaded and never passed through as EP absent-expert sentinels.
- [x] 4.2 Implement deterministic capacity-fitting token microbatches, original top-k/coefficient preservation, output scattering, and reload accounting; verify prefill and batched decode whose expert union exceeds capacity.
- [x] 4.3 Coordinate TP plan epochs and microbatch order with rank-local readiness; verify delayed shards, duplicate demand, peer failure, and bounded shutdown without background-thread collectives.
- [x] 4.4 Add compact request/layer/phase feedback independent of full trace capture and implement cache/transfer and token-readiness observation modes with explicit loss accounting and asynchronous writing.
- [x] 4.5 Run the relevant CPU checks and verify the fork still installs with `VLLM_USE_PRECOMPILED=1`; retain Python-only source changes and record the fork revision/patches.
- [x] 4.6 Recheck actual local resources and run small Qwen TP1/TP2 eager qualification: CPU-first startup peak, forced-token tensors, real routing, greedy outputs, demand misses, overflow, cancellation, and shutdown under fixed numeric criteria.
- [x] 4.7 Record cache-off and repeated-baseline behavior, observer cost, demand synchronization, copy/compute cost, and native graph-enabled full-resident performance; do not proceed with a configuration that fails correctness.

## 5. Live agent events and arrival estimates

- [x] 5.1 Add the optional bounded event sink to the existing client/executor/adapter with IDs, sequences, source times, dependencies, and observed output ranges; verify unchanged prompts, tool observations, and harness independence from vLLM.
- [x] 5.2 Implement receipt timestamps, deduplication, gap/coalescing records, terminal snapshots, and conservative resynchronization; verify slow/disconnected consumers do not indefinitely block the agent.
- [x] 5.3 Implement session/request/tool states, cancellation, queued requests, and sequential/parallel tool joins; verify a completed tool alone cannot make an unresolved dependency ready.
- [x] 5.4 Implement historical-duration/survival prediction conditioned on elapsed time with censoring and unknown-tool fallback; evaluate interval calibration, coverage, and decision-time availability by held-out task.

## 6. Shared residency and idle prefetch

- [x] 6.1 Implement shared demand aggregation from phase distributions, expected token volumes, arrival/service probabilities, and actual scheduler state; verify shared experts and transfers are deduplicated per rank.
- [x] 6.2 Implement the measured-cost planner, existing-residency preference, displacement accounting, churn limits, demand priority, and per-rank/shared-link resource ledgers; verify speculative limits are enforced.
- [x] 6.3 Add bounded worker plan mailboxes and worker-owned asynchronous copy streams independent of model RPC; verify a real prefetch starts during an idle tool interval before the next request arrives.
- [x] 6.4 Implement plan expiry, controller-disconnect fallback, in-flight-copy resolution, and rollback to demand-only execution; verify live fallback never assumes enough memory for full residency.
- [x] 6.5 Integrate observable KV transfer reservations when supported; otherwise explicitly label arbitration unavailable and provide the separate contention-experiment configuration without changing KV eviction.

## 7. Replay and measurement integrity

- [ ] 7.1 Implement frozen-timeline replay using unique batch/layer expert demand, finite capacity/bandwidth, in-flight state, replacement, prefill microbatches, and an explicit recompute policy; reject unsupported missing routes.
- [ ] 7.2 Implement measured cost profiles and timing simulation that propagates delays through model work, tool starts, dependency joins, and request returns; mark uncalibrated output as diagnostic.
- [ ] 7.3 Validate simulation on held-out live executions and report error for bytes, cache events, and timing separately; verify it cannot use miss-delayed timestamps as extra prefetch lead.
- [x] 7.4 Implement event/feature validators, initial residency reconstruction, transfer and slot-generation checks, and incomplete-trace handling; preserve original dataset validation behavior.
- [x] 7.5 Implement metric extraction for true observation points, task outcomes, warmup/repetition boundaries, memory peaks, and uncertainty; distinguish engine token emission from batched client delivery.

## 8. Controlled local comparisons

- [ ] 8.1 Run a bounded pilot of full residency, existing supported offloaders, demand LRU/LFU, and global/stage/temporal policies; verify matched byte/KV/graph accounting and fix measurement faults before the matrix run.
- [ ] 8.2 Run the frozen N1/N4 and feasible budget matrix on the qualified backend, including history+global/stage and agent-conditioned policies, paired task/seed trials, and rotated order; preserve failures and actual overlap.
- [ ] 8.3 Implement and evaluate layer-ahead and semantic-prediction comparison arms where supported, recording predictor cost and explicit reasons for any unavailable arm.
- [ ] 8.4 Run stage/tool/causal-outcome/arrival/aggregation ablations, including degraded predictions and side-channel loss; quantify regressions and wasted prefetch without substituting hindsight features.
- [ ] 8.5 Report fixed-KV mechanism results separately from total-memory-limited service-capacity results, including latency/throughput/memory frontiers and predeclared latency-target feasibility.
- [ ] 8.6 If measured eager overhead justifies graph work, implement piecewise graph support with persistent-address/replay qualification and rerun only affected comparisons; otherwise retain an explicit unqualified graph-support status.

## 9. Expanded workloads and larger models

- [ ] 9.1 Freeze a separate real long-tool protocol with task IDs and duration/early-output/chunk/outcome coverage minima; collect and validate genuine test/build/tool streams while retaining synthetic mechanism tests separately.
- [ ] 9.2 Implement a generic observed-stream progress predictor and compare it with elapsed-time/history and available explicit progress; evaluate unseen-task calibration, useful lead time, and runtime cost.
- [ ] 9.3 Evaluate a bounded small-language-model log reader only under a preregistered supported coverage/resource profile; retain a measured-benefit or evidence-based nondeployment decision with its experiment status.
- [ ] 9.4 Freeze and collect versioned multi-role workflows, paired G2 dependency variants, and held-out G3 templates/role types; enforce event-time visibility and task/template grouping in their analyses.
- [ ] 9.5 Select a larger-model target from available resources and supported layouts, document adapter/spec changes if required, and run model-specific loading, correctness, and transfer calibration before comparing policies.
- [ ] 9.6 Execute the feasible preregistered larger-model comparisons and cross-model analyses; keep unexecuted resource-dependent cells and unsupported adapters visibly outstanding.

## 10. Delivery and completion checks

- [ ] 10.1 Produce reproducible commands, manifests, source versions, correctness/calibration records, numeric tables, and readable comparisons separating cache gains from agent-feature increments and all four evidence levels.
- [ ] 10.2 Update optimization usage documentation and relevant collection handoff/status references, preserving historical reports and marking each local/expansion/migration milestone by actual evidence.
- [ ] 10.3 Run affected package/fork checks and strict OpenSpec validation; verify no implementation or experiment task is marked complete solely because the specification or an output directory exists.
