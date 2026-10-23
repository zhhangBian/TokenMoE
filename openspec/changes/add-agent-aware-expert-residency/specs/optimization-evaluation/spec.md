## ADDED Requirements

### Requirement: Reproducible experiment manifest

Every experiment SHALL freeze task groups, model/weight/template identities, backend and source revisions, feature versions, policy parameters, expert and total memory budgets, KV allocation, graph mode, concurrency, TP layout, sampling, warmup, repetition count, latency targets, and metric definitions before measurement. It SHALL use a new output directory, preserve failures, and record actual device identities and host load. Local Qwen implementation runs SHALL be owned by the executing agent within the user's implementation authorization; specification authoring SHALL start no experiments.

#### Scenario: Hardware differs from historical collection
- **WHEN** the previously used GPUs are unavailable but another supported allocation is usable
- **THEN** the run records the new device/TP mapping and calibration under a distinct protocol without terminating unrelated processes or claiming identical hardware

### Requirement: Distinct replay and measurement evidence

Reports SHALL distinguish lookup analysis, frozen-timeline cache replay, calibrated timing simulation, and live measurements. Cache replay SHALL model unique batch/layer demands, actual resident/in-flight state, finite capacity, replacement, transfer setup/bytes, and the declared prefill/recompute policy. Missing recompute routes SHALL reject that replay case or trigger a separately identified recapture. Timing simulation SHALL use measured compute/copy profiles and advance dependent computation, tool starts, and request returns when stalls change time. Otherwise results SHALL remain frozen-timeline diagnostics. Calibration error SHALL be measured on executions excluded from fitting.

#### Scenario: A simulated miss delays an agent turn
- **WHEN** a timing simulation adds a stall before the model emits its tool call
- **THEN** the dependent tool start and next request are shifted according to that simulated execution, and the original trace's future timestamps are not reused as unaffected arrivals

### Requirement: Baseline and ablation matrix

Local comparisons SHALL include full residency, demand-only LRU/LFU, global/stage/temporal residency, previous+global and previous+stage smoothing, and agent-conditioned policies on the same cache backend. Existing UVA/layer prefetch, layer-ahead prediction, and semantic prediction SHALL be measured as separate comparison arms where supported, with unsupported arms and predictor overhead disclosed. Default Qwen sweeps SHALL cover N1/N4 and slot targets 16/32/64/96/128 when feasible, while reporting actual per-rank bytes including reserves. Oracles SHALL remain offline.

Ablations SHALL isolate request stage, tool category, causally observed outcome, arrival prediction, and shared aggregation. Outcome hindsight SHALL not qualify as a deployable arm. At least three measured repetitions per live comparison SHALL follow fixed warmup, with policy order rotated and task/seed pairing preserved.

#### Scenario: Agent predictor improves over LRU
- **WHEN** an agent-conditioned policy beats LRU but does not beat previous+stage smoothing
- **THEN** the report preserves both comparisons and does not attribute the full LRU improvement to agent metadata

### Requirement: Joint memory latency and throughput reporting

Experiments SHALL report engine TTFT, completion of required tools to first returned token, token emission timing and TPOT, task completion time, output-token and completed-task throughput, initialization/steady-state GPU peaks per rank and aggregate, host memory, transfer bytes, exposed miss wait, prefetch waste, and controller/observer cost. They SHALL disclose token-delivery batching, measurement clocks, missing fields, sample counts, and task-level uncertainty. Failed or truncated tasks SHALL not be silently removed or counted as successful completed-task throughput.

Mechanism comparisons SHALL hold KV allocation, scheduler, execution/graph mode, workload, and warmup constant. Service-capacity comparisons that convert freed expert memory to KV SHALL hold total GPU limits constant and identify that source of gain. Both fixed-memory performance and memory required to meet predeclared latency targets SHALL be reported, without requiring all metrics to improve simultaneously.

#### Scenario: More KV capacity avoids preemption
- **WHEN** an offload configuration improves throughput by using reclaimed expert memory for KV
- **THEN** the result is labeled a service-capacity gain, reports preemption/KV changes, and is separated from the fixed-KV mechanism comparison

### Requirement: Correctness and cost gates precede performance claims

Live policy comparisons SHALL require a qualified exact backend, a passing matching collection equivalence record when full capture is enabled, and measured copy/compute/observer costs. Cache correctness SHALL use its own stricter protocol and SHALL not inherit an allowance for cache-induced token differences from the historical capture gate. Comparisons SHALL include an equal-graph-mode full-resident baseline and disclose the native graph-enabled reference. A failed correctness gate SHALL preserve diagnostics and prevent that configuration from supporting speedup claims.

#### Scenario: Cache results diverge under fixed inputs
- **WHEN** qualification finds a cache-induced routing, tensor, or token mismatch outside the preregistered contract
- **THEN** the configuration remains unqualified, its evidence is retained, and performance numbers are not presented as an accepted exact-inference result

### Requirement: Workload and model expansion gates

Long-tool, multi-role, G2, G3, and larger-model studies SHALL use separate preregistered protocols and artifacts. Long-tool protocols SHALL specify minimum real-task coverage across duration, early-output lead, output chunks, and outcome classes before collection; artificial waiting SHALL be labeled synthetic and excluded from real-workload coverage. Multi-role studies SHALL use at least two versioned templates. G2 SHALL compare the same role set under different dependency structures; G3 SHALL hold out whole templates and distinguish unseen role types. Untouched task groups SHALL support confirmation beyond the explored Qwen tasks.

Larger-model migration SHALL follow local correctness and controlled measurement, select a model/backend from verified resource availability, and require new loading/layout/correctness and transfer calibration. An unsupported quantized/shared-expert adapter SHALL require an explicit capability/spec update before being claimed as supported.

#### Scenario: Existing data lacks progress streams
- **WHEN** the one-role Qwen collection is evaluated for general log-based arrival prediction or G2/G3
- **THEN** those coverage gates remain unmet and the report requires the corresponding expansion dataset rather than treating existing short bash calls as sufficient evidence

### Requirement: Evidence-based milestone delivery

Delivery SHALL contain runnable commands, input/split/protocol manifests, source hashes, correctness and calibration results, machine-readable metrics, readable comparisons, and explicit milestone status. Local analysis, backend qualification, local policy evaluation, workload expansion, and larger-model migration SHALL have independent completion states. Missing hardware or unsupported datasets SHALL not be marked complete. Software correctness SHALL not require a positive research result; absent agent-feature gains SHALL remain visible and SHALL not be replaced with generic cache gains.

#### Scenario: Local work succeeds but migration cannot run
- **WHEN** local milestones pass and a larger model lacks the required hardware or supported adapter
- **THEN** delivery marks only the local milestones complete and records the unmet migration prerequisites and outstanding tasks
