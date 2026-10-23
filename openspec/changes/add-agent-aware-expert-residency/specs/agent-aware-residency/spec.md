## ADDED Requirements

### Requirement: Causal session event state

The controller SHALL consume versioned session/request/tool events with unique event IDs, source sequence, observation and receipt times, and clock domains. It SHALL track active requests, queued work, pending tool dependencies, and terminal sessions. Duplicate events SHALL be idempotent; stale or missing events SHALL be recorded and cause conservative state handling. A decision SHALL depend only on events received by its decision time. Tool completion SHALL NOT imply that a request is already executing on the GPU.

#### Scenario: Two tools feed one request
- **WHEN** one of two required tool outputs is complete and the other is pending
- **THEN** the session remains waiting on the unresolved dependency and is not modeled as having a ready next request

#### Scenario: A delayed event follows cancellation
- **WHEN** an older tool-progress event arrives after the corresponding session has terminated
- **THEN** it is retained as delayed evidence but does not reactivate that session's predicted demand

### Requirement: Arrival distributions and bounded progress predictors

The initial arrival predictor SHALL use training-only duration histories, elapsed runtime, observed completion, and tool dependency structure to produce a remaining-time distribution with support and confidence. Timeouts SHALL be treated as censored or terminal outcomes according to their actual semantics. Unknown tools SHALL receive a declared fallback. Stream-feature and small-language-model predictors SHALL be optional alternatives that use only observed output and report inference overhead, resource use, lead-time coverage, and calibration on held-out tasks.

#### Scenario: A silent unfamiliar tool remains running
- **WHEN** no usable history or progress observation exists
- **THEN** the predictor returns its documented uncertain fallback and the controller can suppress speculative prefetch while preserving demand execution

### Requirement: Shared demand aggregation

The controller SHALL aggregate demand across sessions using model-specific phase distributions, expected token volume, arrival/service probability, and actual scheduler state. It SHALL distinguish expected lookup counts from unique weight loads. The same logical expert needed by several requests SHALL share a resident copy and an in-flight transfer per required rank. Terminal sessions SHALL not contribute future demand. N=1 SHALL use the same decision model.

#### Scenario: Concurrent requests share an expert
- **WHEN** two requests need the same layer/expert shard while its transfer is already in flight
- **THEN** the plan accounts for one stored shard and one transfer on that rank, with both consumers waiting on its readiness

### Requirement: Cost-aware bounded residency plans

Each plan SHALL include its epoch, model/backend identity, expiry, source predictions, desired resident experts, transfer reservations, and reasons. Candidate selection SHALL account for measured transfer/setup costs, exposed wait avoided, existing residency, displaced reuse, and available capacity. The controller SHALL enforce per-rank memory limits, a shared bandwidth ledger, configurable speculative byte/inflight limits, and demand priority. It SHALL protect in-use slots and apply a declared churn-control rule. Active requests SHALL NOT receive equal demand weights solely because they are active.

#### Scenario: A speculative load would displace active demand
- **WHEN** a proposed transfer requires an in-use slot or would exceed the declared speculative reservation
- **THEN** the controller defers or rejects that transfer and records the reason without blocking a feasible demand load behind new speculation

### Requirement: Idle delivery and safe reactive fallback

Plans SHALL reach worker-owned transfer execution while the engine is idle between requests, without waiting for a future model-execution RPC. Stale plans, unavailable predictions, controller failure, and side-channel loss SHALL stop new speculative admission and retain the configured demand-only policy. In-flight copies SHALL remain tracked until completion or a recorded failure; cancellation SHALL NOT imply instant DMA cancellation. No policy SHALL claim an unconditional no-regression guarantee.

#### Scenario: A plan arrives during tool execution
- **WHEN** the engine has no scheduled model work and receives a valid prefetch plan with available resources
- **THEN** the worker can start the planned copies before the next request arrives and records actual start/completion times

#### Scenario: Controller disconnects
- **WHEN** the controller connection fails during generation
- **THEN** the worker expires speculative plans, resolves existing transfers safely, and continues exact demand-driven inference if the storage backend remains healthy

### Requirement: Transfer contention boundary

The residency system SHALL identify shared host/device transfer resources and record whether KV transfers are included in their reservations. When KV arbitration is unavailable, reports SHALL label that limitation and SHALL NOT claim joint KV/expert scheduling. This capability SHALL not change the KV eviction policy.

#### Scenario: Independent KV offload uses the same link
- **WHEN** an experiment enables KV transfers without exposing their reservations to the expert planner
- **THEN** the run identifies uncoordinated contention and reports measured interaction separately from coordinated-transfer results
