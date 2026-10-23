## ADDED Requirements

### Requirement: Explicit opt-in and supported backend matrix

Expert caching SHALL be disabled by default and independently configurable from full trace capture. Initial support SHALL be Qwen3-30B-A3B BF16 with the verified unquantized modular Triton MoE backend, TP=1 or 2, EP=1, DP=1, synchronous scheduling, and no speculative decoding. Unsupported quantization, monolithic kernels, EP/EPLB, or conflicting weight offload SHALL fail configuration validation before loading. Effective cache settings SHALL participate in provenance and compilation/config hashing. Cache-disabled execution SHALL retain the existing inference path.

#### Scenario: Unsupported kernel is selected
- **WHEN** expert caching is requested with a monolithic or unsupported quantized backend
- **THEN** startup fails before weight allocation with the incompatible setting identified, without silently substituting an approximate path

### Requirement: CPU-first loading and bounded expert memory

The backend SHALL load expert shards into CPU storage without requiring a complete GPU expert bank during initialization. It SHALL preserve TP shard ownership, dtype, projection ordering, and required layout transformations. GPU expert memory SHALL stay within the configured per-rank budget during initialization, warmup, and execution, including resident/in-flight weight storage, staging, mapping state, and declared expert scratch. Nonexpert weights and other workspaces SHALL be reported separately and included in total peaks. Host pageable and pinned allocations SHALL be measured; pinning failure SHALL use a declared bounded fallback or fail clearly.

#### Scenario: Full expert weights exceed the GPU budget
- **WHEN** CPU memory can hold the checkpoint shards but complete GPU expert weights would exceed the configured limit
- **THEN** initialization loads through bounded storage and reaches a usable cache without first allocating the complete GPU expert bank

### Requirement: Exact router demand recovery

The backend SHALL execute the actual router-selected experts with their original weights and top-k coefficients. It SHALL resolve logical IDs to ready physical slots, coalesce duplicate demand, join existing transfers, and wait for all required shards before execution. It SHALL NOT drop experts, substitute predictions for routing, or encode unresolved cache misses as an EP absent-expert sentinel. Load failure SHALL produce an explicit execution error rather than plausible partial output.

#### Scenario: A prediction misses a selected expert
- **WHEN** the real router selects an expert absent from the predicted resident set
- **THEN** demand recovery loads that expert and waits for readiness before computing its contribution with the unchanged routing coefficient

### Requirement: Capacity-safe token microbatch execution

Every supported layer allocation SHALL provide at least top-k usable slots under the initial executor. If a scheduled batch's distinct expert union exceeds capacity, the backend SHALL partition tokens into deterministic disjoint microbatches whose individual expert unions fit. It SHALL preserve each token's original top-k order and coefficients, execute each token exactly once, restore output order, and account for reloads and scratch. It SHALL reject an allocation that cannot execute one token and SHALL NOT bypass the budget with an unreported full-layer allocation.

#### Scenario: Prefill touches most experts
- **WHEN** prefill selects more distinct experts than the layer's 32 usable slots
- **THEN** the backend processes capacity-fitting token microbatches, returns outputs in the original order, and records the actual repeated transfers and peak memory

### Requirement: Slot lifetime and asynchronous copy correctness

Slots SHALL track free/loading/ready/in-use states, logical identity, generation, and outstanding consumers. Copy completion SHALL precede mapping publication; computation completion SHALL precede reuse. In-use slots SHALL not be evicted. Stale completion events SHALL not publish into a slot reused for another generation. Worker-owned copy streams and bounded plan mailboxes SHALL operate during idle intervals as well as active inference. Shutdown SHALL stop new admission and resolve outstanding work before freeing resources.

#### Scenario: Old copy completes after a plan expires
- **WHEN** a transfer belongs to an expired plan or an obsolete slot generation
- **THEN** its completion is handled against that recorded generation and cannot overwrite the mapping for a newer occupant

### Requirement: Tensor-parallel execution agreement

All TP ranks SHALL agree on logical plan epochs and token-microbatch order while retaining rank-local weight shards and readiness events. Each rank SHALL wait for its required copies before entering the agreed computation/collective sequence. Background prefetch threads SHALL not issue unordered TP collectives. Peer failure, inconsistent plans, or unavailable shards SHALL terminate affected work with bounded failure handling rather than deadlock or return partial tensors.

#### Scenario: One shard finishes later
- **WHEN** TP rank 0 has loaded an expert shard and rank 1 has not
- **THEN** execution respects each rank's readiness and the common operation order, without treating rank 0 readiness as completion of the full expert

### Requirement: Numerical and graph qualification

The initial correctness profile SHALL use eager execution. Qualification SHALL compare unchanged router selections, expert outputs under predeclared numeric tolerances, forced-token execution, and greedy token sequences under matched deterministic schedules. Repeat-baseline variability SHALL be recorded separately. Piecewise graph support SHALL require dedicated buffer-lifetime, changing-residency, and replay-equivalence checks before enablement. A backend SHALL not claim token identity if cache-enabled execution diverges, and tolerances SHALL not be weakened after observing test results.

#### Scenario: A graph replays after slots change
- **WHEN** a graph-enabled profile changes resident expert identities between replays
- **THEN** qualification verifies current mappings and valid persistent buffers against eager results before that profile is allowed in performance claims

### Requirement: Independent compact feedback

The backend SHALL provide request-attributed, layer/phase-specific feedback from actual routing without requiring complete routing-file capture. Feedback SHALL use bounded aggregation/export and SHALL not introduce per-layer host synchronization solely to update predictions. Detailed tracing, compact counters, and disabled observation SHALL be distinguishable configurations with measured overhead. Demand-path synchronization needed for exact recovery SHALL be measured separately.

#### Scenario: Full recorder is disabled
- **WHEN** expert caching and predictor feedback are enabled without `TOKENMOE_TRACE_DIR`
- **THEN** inference and compact feedback operate without producing full routing artifacts or depending on the trace writer
