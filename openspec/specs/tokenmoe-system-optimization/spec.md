# tokenmoe-system-optimization Specification

## Purpose
Define the current-stage offline scheduler replay and optimization boundary for
TokenMoE, using predicted prompt-segment expert demand while excluding online
prefetch and EPLB runtime integration from active validation.

## Requirements

### Requirement: Expert-overlap-aware scheduling prototype
The system SHALL score candidate batches of ready agent nodes using predicted
expert demand aggregated from prompt segments, dependency constraints, waiting
cost, confidence gating, and starvation guards.

#### Scenario: Dependency would be violated
- **WHEN** an agent node depends on an unfinished predecessor
- **THEN** the scheduler prototype does not schedule that node regardless of predicted expert overlap

#### Scenario: Dependency graph is missing
- **WHEN** a workload does not provide dependency edges or ready times
- **THEN** scheduler replay marks agent-DAG scheduling unavailable or labels the result as non-agent prompt/domain locality replay

#### Scenario: Prediction-based replay scheduler runs
- **WHEN** offline scheduler replay is used before online vLLM integration
- **THEN** scheduler decisions use predicted per-segment/per-layer expert demand rather than true routed experts

#### Scenario: Replay batch is scored
- **WHEN** a predicted batch has been selected
- **THEN** the system reports actual active expert fanout and per-expert token counts from true vLLM routed prompt traces

#### Scenario: Non-MoE layer is present
- **WHEN** true vLLM routed traces include a layer not listed in persisted `moe_layer_ids`
- **THEN** scheduler replay excludes or marks that layer unavailable before computing fanout or token-density metrics

#### Scenario: Prediction confidence is low
- **WHEN** predictions for a candidate request or segment are missing or below the configured confidence threshold
- **THEN** the scheduler applies the configured baseline behavior or disables TokenMoE-specific prioritization for the affected request

#### Scenario: Node delay reaches the guard
- **WHEN** a ready node reaches the configured maximum delay bound
- **THEN** the scheduler prioritizes that node over further expert-overlap optimization

### Requirement: Constrained MoE execution cost replay
The system SHALL optimize a replay-only MoE execution cost proxy while
preserving agent DAG legality.

#### Scenario: Candidate batch is scored
- **WHEN** the scheduler evaluates a legal candidate batch
- **THEN** the score includes predicted active expert fanout cost, predicted low token-density cost derived from expert-token-label demand, added waiting cost, low-confidence penalty or gating, and starvation cost or guard

#### Scenario: True experts are used
- **WHEN** a batch decision has already been made
- **THEN** true routed experts are used only for post-decision scoring of actual fanout, actual token density, hit rate, and oracle gap

#### Scenario: Non-agent dataset is replayed
- **WHEN** replay uses a dataset marked as chat/prompt-only or domain-instruction scope
- **THEN** the report labels the result as prompt/domain locality replay and does not use it as evidence for agent-DAG scheduling claims

### Requirement: Prediction-vs-oracle replay comparison
The system SHALL report scheduler replay results separately for FIFO or baseline,
temporal baseline, RouteSig prediction, and oracle decision modes.

#### Scenario: Replay summary is generated
- **WHEN** scheduler replay completes
- **THEN** the report distinguishes FIFO baseline, temporal-window baseline, RouteSig prediction replay, and oracle replay using true routed experts

#### Scenario: Oracle outperforms prediction
- **WHEN** oracle replay has better expert fanout or token-density metrics than prediction replay
- **THEN** the report preserves both results instead of replacing prediction results with oracle values

### Requirement: Scheduler replay metrics
The system SHALL report MoE execution proxy, scheduling cost, policy comparison,
and failure/fallback metrics.

#### Scenario: MoE execution proxy is reported
- **WHEN** replay metrics are generated
- **THEN** the report includes actual active expert fanout per batch/layer, actual per-expert token count, predicted-vs-actual fanout error, and batch expert overlap

#### Scenario: Scheduling cost is reported
- **WHEN** replay metrics are generated
- **THEN** the report includes mean and p95 added waiting steps, maximum delay, deadline misses when deadlines exist, and dependency violations

#### Scenario: Failure and fallback metrics are reported
- **WHEN** replay metrics are generated
- **THEN** the report includes low-confidence gated fraction, missing block-span fraction, unavailable MoE-layer fraction, unavailable metric fraction, and any dataset/model pair where replay could not run

### Requirement: Prefetch and EPLB outputs excluded from current-stage reports
The system SHALL remove or quarantine legacy prefetch and EPLB replay outputs
from the main analysis and report path for this change.

#### Scenario: Main report is generated
- **WHEN** the current-stage TokenMoE report is generated
- **THEN** it does not present prefetch or EPLB replay metrics as active validation results

#### Scenario: Legacy output is retained
- **WHEN** historical prefetch or EPLB output remains available
- **THEN** it is labeled archived or excluded from this change's validation
