## MODIFIED Requirements

### Requirement: Predictor-facing expert-set output
The system SHALL expose a prediction-facing interface that returns prompt
segment, per-layer top-M expert working sets from RouteSig and simple baseline
predictors before router execution.

#### Scenario: Segment prediction is available
- **WHEN** a predictor receives request metadata, prompt segment metadata, and layer ID
- **THEN** the prediction output includes request ID, segment ID, layer ID, predicted expert IDs, normalized expert weights, confidence, source name, fallback key, block type, segment position, token span, and optional scores

#### Scenario: RouteSig prediction is available
- **WHEN** a RouteSig lookup has sufficient confidence for a segment metadata key and layer
- **THEN** the prediction output uses the most specific available key in the configured fallback order

#### Scenario: RouteSig falls back to a broader key
- **WHEN** the most specific segment metadata key has insufficient samples
- **THEN** the prediction output records the broader fallback key that produced the expert set

#### Scenario: Baseline predictor is used for comparison
- **WHEN** replay requests predictions from global frequency, request LRU, sequence history, temporal window frequency, or oracle adapters
- **THEN** the output uses the same segment-aware prediction shape as RouteSig

#### Scenario: Expert weights are emitted
- **WHEN** any predictor returns a segment/layer prediction
- **THEN** it emits normalized expert weights aligned with predicted expert IDs

#### Scenario: Predictor has only a ranked expert set
- **WHEN** a predictor cannot derive frequency or probability weights for a ranked expert set
- **THEN** it uses an explicit uniform-weight conversion and records that conversion in the prediction output

### Requirement: Segment-aware RouteSig keys
The system SHALL maintain RouteSig statistics at prompt block/token-segment
granularity with hierarchical fallback.

#### Scenario: Segment fallback order is used
- **WHEN** RouteSig predicts experts for a prompt segment
- **THEN** lookup attempts keys in this order: `(agent_id, role, phase, block_type, segment_position)`, `(role, phase, block_type, segment_position)`, `(role, phase, block_type)`, `(role, block_type)`, `(block_type)`, `(role, phase)`, `(role)`, and `global`

#### Scenario: Segment alignment is unavailable
- **WHEN** a prompt segment cannot be mapped to a token span
- **THEN** RouteSig prediction and evaluation mark segment-level metrics unavailable for that segment instead of silently treating it as a normal aligned segment

#### Scenario: Layer is not an MoE layer
- **WHEN** a routed-experts array contains a layer ID that is not in the model's persisted `moe_layer_ids`
- **THEN** RouteSig ignores or marks that layer unavailable and never treats zero-filled entries as active expert IDs

### Requirement: Model-relative Top-M budgets
The system SHALL evaluate prediction budgets relative to each model's router
top-k instead of using a fixed Top-M across models.

#### Scenario: Main budget is selected
- **WHEN** prediction metrics are generated for a model
- **THEN** the main Top-M result uses `2 * router_top_k`

#### Scenario: Budget curve is reported
- **WHEN** prediction metrics are generated
- **THEN** the report includes budgets at `1x`, `1.5x`, `2x`, and `3x` router top-k

## ADDED Requirements

### Requirement: Prompt segment demand aggregation
The system SHALL aggregate segment-level expert probabilities into request and
batch-level predicted expert-token-label demand using prompt-token counts and
the model router top-k.

#### Scenario: Request demand is computed
- **WHEN** a request contains multiple prompt segments
- **THEN** per-layer request demand for each expert is the sum of each segment's normalized expert weight multiplied by that segment's prompt-token count and the model router top-k

#### Scenario: Batch demand is computed
- **WHEN** scheduler replay scores a candidate batch
- **THEN** per-layer batch demand is the sum of request-level demand across requests in the candidate batch

#### Scenario: Demand units are reported
- **WHEN** predicted demand is emitted for replay
- **THEN** demand units are labeled as expected expert-token labels so they can be compared with true routed expert-label counts across models with different router top-k values

### Requirement: Online time-ordered prediction evaluation
The system SHALL evaluate predictors in a time-ordered online protocol for each
dataset and model pair.

#### Scenario: Evaluation split is created by group
- **WHEN** traces are evaluated for a dataset/model pair
- **THEN** the first 70 percent of source groups in time or source-index order are used for training and the remaining 30 percent are used for evaluation

#### Scenario: Source groups are unavailable
- **WHEN** a dataset adapter cannot provide conversation, trajectory, workflow, or source-item grouping
- **THEN** evaluation for that dataset/model pair is marked unavailable instead of using a request-level split that could leak related records

#### Scenario: Online evaluation processes a trace
- **WHEN** an evaluation trace is processed
- **THEN** each predictor predicts before seeing the trace, scores against true vLLM routed experts, and updates only after scoring

### Requirement: Prediction quality reporting
The system SHALL report block-level prediction quality and fallback behavior.

#### Scenario: Prediction report is generated
- **WHEN** predictor evaluation completes
- **THEN** the report includes expert-label hit rate, exact-token hit rate, weighted coverage when router scores are available, and explicit unavailable status when scores are absent

#### Scenario: Block-level report is generated
- **WHEN** predictor evaluation completes
- **THEN** the report includes per-block-type, per-layer, segment-length bucket, confidence calibration, and fallback-key usage metrics
