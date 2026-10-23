## ADDED Requirements

### Requirement: Versioned optimization artifacts

Optimization experiments SHALL preserve the existing raw and finalized routing formats and store new input/split/protocol manifests, feature-availability records, predictor/cost-profile versions, plan decisions, cache/transfer telemetry, token-readiness observations, and result tables as versioned additions in separate output directories. Source data, prior reports, and valid failure traces SHALL remain immutable. `collection/data_collect.md` SHALL be updated with any newly emitted collection fields, event schemas, storage locations, and validation rules in the same implementation change.

#### Scenario: Historical traces are analyzed
- **WHEN** analysis derives new labels or evaluates a new cache policy on the completed local Qwen dataset
- **THEN** it writes separate versioned outputs referencing source hashes and does not replace the original labels, routing, validation, or acceptance records

### Requirement: Cache event and feature validation

Validation SHALL check comparable clock domains, causal feature availability, event ordering, unique IDs, transfer lifecycle, rank-local bytes, slot-generation transitions, readiness before use, and memory-accounting consistency. Open transfers at a crash SHALL be marked incomplete. Explicit telemetry gaps SHALL make exact residency replay unavailable for the affected interval rather than inventing state. Cache telemetry SHALL not be required to validate historical collection runs that predate optimization experiments.

#### Scenario: A transfer completes for a reused slot
- **WHEN** an event stream publishes a ready expert using a slot generation superseded before that transfer completes
- **THEN** optimization validation fails and identifies the conflicting transfer/slot events

#### Scenario: Detailed telemetry is incomplete
- **WHEN** an otherwise valid inference run reports dropped cache events
- **THEN** its measured inference results retain their provenance but exact event-replay coverage is marked incomplete for the affected intervals
