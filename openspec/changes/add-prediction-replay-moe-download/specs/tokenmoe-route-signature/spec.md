## ADDED Requirements

### Requirement: Predictor-facing expert-set output
The system SHALL expose a prediction-facing interface that returns per-request,
per-layer top-M expert working sets from RouteSig and baseline predictors before
router execution.

#### Scenario: RouteSig prediction is available
- **WHEN** a RouteSig lookup has sufficient confidence for a request metadata key and layer
- **THEN** the prediction output includes request ID, layer ID, predicted expert IDs, confidence, source name, and fallback key

#### Scenario: RouteSig falls back to a broader key
- **WHEN** the most specific metadata key has insufficient samples
- **THEN** the prediction output records the broader fallback key that produced the expert set

#### Scenario: Baseline predictor is used for comparison
- **WHEN** replay requests predictions from global frequency, request LRU, or sequence history
- **THEN** the output uses the same prediction shape as RouteSig

### Requirement: Temporal expert feature derivation
The system SHALL derive temporal expert features from ordered traces without
requiring a learned model.

#### Scenario: Temporal features are computed
- **WHEN** ordered routed traces are analyzed
- **THEN** the system derives per-layer/per-expert features including recent access, reuse distance, window frequency, and burst count where data is available

#### Scenario: Trace history is too short
- **WHEN** a feature cannot be computed because trace history is insufficient
- **THEN** the system emits a neutral or missing value without failing prediction or replay
