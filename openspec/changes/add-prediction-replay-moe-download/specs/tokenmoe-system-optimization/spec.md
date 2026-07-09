## MODIFIED Requirements

### Requirement: Expert-overlap-aware scheduling prototype
The system SHALL score candidate batches of ready agent nodes using predicted
expert overlap, active expert fanout estimates, waiting penalty, confidence,
and dependency constraints.

#### Scenario: Dependency would be violated
- **WHEN** an agent node depends on an unfinished predecessor
- **THEN** the scheduler prototype does not schedule that node regardless of predicted expert overlap

#### Scenario: Prediction-based replay scheduler runs
- **WHEN** offline replay is used before online vLLM integration
- **THEN** the scheduler decision uses predicted per-layer top-M expert sets rather than true routed experts

#### Scenario: Replay batch is scored
- **WHEN** a predicted batch has been selected
- **THEN** the system reports actual active expert fanout and per-expert token counts from real routed traces

#### Scenario: Prediction confidence is low
- **WHEN** predictions for a candidate request are missing or below the configured confidence threshold
- **THEN** the scheduler uses the configured baseline or disables TokenMoE-specific prioritization for that request

## ADDED Requirements

### Requirement: Prediction-vs-oracle replay comparison
The system SHALL report replay results separately for baseline, prediction, and
oracle decision modes.

#### Scenario: Replay summary is generated
- **WHEN** scheduler replay completes
- **THEN** the report distinguishes FIFO or baseline replay, RouteSig prediction replay, and oracle replay using true routed experts

#### Scenario: Oracle outperforms prediction
- **WHEN** oracle replay has better expert fanout or token-count metrics than prediction replay
- **THEN** the report preserves both results instead of replacing prediction results with oracle values

### Requirement: Temporal and spatial locality metrics
The system SHALL report temporal and spatial locality metrics derived from
ordered traces and workload metadata.

#### Scenario: Temporal locality is analyzed
- **WHEN** traces are analyzed in request order
- **THEN** the system reports reuse distance, phase-transition overlap, temporal Jaccard, or equivalent metrics where data is available

#### Scenario: Spatial locality is analyzed
- **WHEN** batch or replay metrics are analyzed
- **THEN** the system reports per-layer active expert fanout, per-expert token count, and batch expert overlap

#### Scenario: Metrics are unavailable
- **WHEN** required trace or workload fields are missing
- **THEN** the report marks the affected metric unavailable without failing the whole analysis
