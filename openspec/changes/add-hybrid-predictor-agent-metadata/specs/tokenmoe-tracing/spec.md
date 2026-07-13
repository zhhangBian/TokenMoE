# tokenmoe-tracing Delta

## MODIFIED Requirements

### Requirement: Router score trace option
The system SHALL support capture of real vLLM router top-k scores with the
same token, layer, and top-k indexing as selected expert IDs, using a
modified local vLLM capture path that mirrors routed-expert ID capture.

#### Scenario: Score capture disabled
- **WHEN** router-score tracing is disabled
- **THEN** selected expert ID tracing continues to work and score fields are omitted or marked unavailable

#### Scenario: Score capture enabled
- **WHEN** router-score tracing is enabled on a supported model
- **THEN** each captured score aligns index-for-index with the corresponding selected expert ID and the trace records the per-model score semantics

#### Scenario: Score capture is unsupported for a configuration
- **WHEN** score capture cannot be validated for a model or parallelism configuration
- **THEN** collection continues with expert IDs only and the trace marks router scores unavailable with an explicit reason instead of failing or emitting misaligned scores

#### Scenario: Score capture preflight runs
- **WHEN** a score-enabled collection run starts
- **THEN** a smoke check asserts scores are finite, within the documented range for the recorded semantics, aligned with expert IDs, and that generated token IDs are identical to a capture-disabled run

## ADDED Requirements

### Requirement: Trace schema v3 with router scores
The system SHALL define trace schema `tokenmoe.trace.v3` that extends v2 with
aligned router scores and score semantics while keeping v2 traces readable.

#### Scenario: v3 record is written
- **WHEN** score-enabled collection writes a trace record
- **THEN** the record uses schema version `tokenmoe.trace.v3` and includes per-token, per-MoE-layer, per-slot router scores aligned with selected expert IDs plus a `router_score_semantics` field

#### Scenario: Mixed trace versions are rejected
- **WHEN** an analysis run receives both v2 and v3 traces
- **THEN** the run is rejected with an explicit error instead of silently mixing score-bearing and score-free records

#### Scenario: Weighted coverage becomes available
- **WHEN** v3 traces with router scores are analyzed
- **THEN** weighted-coverage metrics are computed from real scores instead of being marked unavailable
