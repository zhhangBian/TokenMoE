# tokenmoe-route-signature Delta

## MODIFIED Requirements

### Requirement: Segment-aware RouteSig keys
The system SHALL maintain RouteSig statistics at prompt block/token-segment
granularity with hierarchical fallback that includes enriched agent metadata
levels.

#### Scenario: Segment fallback order is used
- **WHEN** RouteSig predicts experts for a prompt segment
- **THEN** lookup attempts keys in this order: `(agent_id, role, trajectory_phase, tool_type, block_type, segment_position)`, `(role, trajectory_phase, tool_type, block_type, segment_position)`, `(role, trajectory_phase, block_type, segment_position)`, `(role, phase, block_type, segment_position)`, `(role, phase, block_type)`, `(role, block_type)`, `(block_type)`, `(role, phase)`, `(role)`, and `global`

#### Scenario: Enriched metadata is absent
- **WHEN** a workload does not provide `trajectory_phase` or `tool_type`
- **THEN** lookup skips the enriched levels and proceeds with the remaining fallback order without error

#### Scenario: Segment alignment is unavailable
- **WHEN** a prompt segment cannot be mapped to a token span
- **THEN** RouteSig prediction and evaluation mark segment-level metrics unavailable for that segment instead of silently treating it as a normal aligned segment

#### Scenario: Layer is not an MoE layer
- **WHEN** a routed-experts array contains a layer ID that is not in the model's persisted `moe_layer_ids`
- **THEN** RouteSig ignores or marks that layer unavailable and never treats zero-filled entries as active expert IDs

#### Scenario: Score-weighted statistics are used
- **WHEN** v3 traces with router scores are available
- **THEN** RouteSig supports accumulating score mass in addition to selection counts, and reports identify which statistics mode produced each result

### Requirement: Prediction quality reporting
The system SHALL report block-level prediction quality, fallback behavior, and
statistical significance for predictor comparisons.

#### Scenario: Prediction report is generated
- **WHEN** predictor evaluation completes
- **THEN** the report includes expert-label hit rate, exact-token hit rate, weighted coverage when router scores are available, and explicit unavailable status when scores are absent

#### Scenario: Block-level report is generated
- **WHEN** predictor evaluation completes
- **THEN** the report includes per-block-type, per-layer, segment-length bucket, confidence calibration, and fallback-key usage metrics

#### Scenario: Confidence intervals are reported
- **WHEN** predictor comparisons are reported for a dataset/model pair
- **THEN** hit-rate deltas between predictors include bootstrap confidence intervals computed over at least 1000 group-level resamples

#### Scenario: Multiple splits are reported
- **WHEN** predictor evaluation completes for a dataset/model pair
- **THEN** results are reported for at least two group-preserving time splits and headline claims are checked on both

## ADDED Requirements

### Requirement: Hybrid RouteSig-temporal predictor
The system SHALL provide a hybrid predictor that fuses RouteSig and
temporal-window distributions per layer using calibrated confidence, with
per-layer gating back to baselines.

#### Scenario: Hybrid prediction is produced
- **WHEN** the hybrid predictor predicts for a segment and layer
- **THEN** it emits a convex combination of the RouteSig and temporal-window expert distributions weighted by the calibrated RouteSig confidence of the resolved fallback key, in the standard segment-aware prediction shape

#### Scenario: Layer gating disables RouteSig
- **WHEN** a layer's train-split delta versus the global baseline is not positive with sufficient support
- **THEN** the hybrid predictor uses only the temporal/global path for that layer and the gate decision is persisted for reporting

#### Scenario: Sparse key is smoothed
- **WHEN** a RouteSig key has few samples
- **THEN** additive smoothing is applied and the prediction records that smoothing was used

#### Scenario: Confidence is calibrated
- **WHEN** confidence is computed for fusion weighting
- **THEN** calibration uses only train-split data bucketed by block type and layer

### Requirement: Learned ranker predictor
The system SHALL provide a lightweight learned ranker predictor trained only
on pre-router features.

#### Scenario: Ranker predicts experts
- **WHEN** the learned ranker predicts for a segment and layer
- **THEN** it emits a per-layer expert ranking in the standard prediction shape using only metadata, temporal statistics, segment position/length, and layer features available before router execution

#### Scenario: Ranker respects online protocol
- **WHEN** the ranker is evaluated
- **THEN** it is trained on the train split only and evaluated predict-before-update on the evaluation split

#### Scenario: Post-router features are excluded
- **WHEN** ranker features are constructed
- **THEN** no hidden states or current-request router outputs are used, and a test enforces this boundary

### Requirement: Metadata ablation evaluation
The system SHALL evaluate a metadata ablation matrix that isolates the
contribution of each agent metadata field.

#### Scenario: Ablation matrix is run
- **WHEN** ablation evaluation runs on an agent dataset
- **THEN** it reports the full hybrid plus variants removing role, phase, tool type, block type, and position fields, and length-only, temporal-only, and RouteSig-only baselines under identical splits and budgets

#### Scenario: Leakage controls are reported
- **WHEN** ablation results are reported
- **THEN** the report states whether observed gains persist beyond prompt-length-only and source-order-only variants

### Requirement: Agent-metadata increment gate
The system SHALL evaluate an explicit gate that decides whether agent metadata
adds predictive signal beyond temporal locality on real agent workloads.

#### Scenario: Gate passes
- **WHEN** the hybrid predictor beats the temporal-only baseline at the 2x budget on SWE-agent for both models with non-overlapping bootstrap confidence intervals on both splits, and at least one metadata ablation shows a significant drop
- **THEN** the gate decision record marks the agent-DAG claim as supported for the next stage

#### Scenario: Gate fails
- **WHEN** the gate criterion is not met
- **THEN** the gate decision record marks the agent-DAG claim unsupported and directs the next stage toward prompt/block-conditioned prefetch evidence, and this outcome still counts as valid completion of this change

#### Scenario: Gate record is persisted
- **WHEN** gate evaluation completes
- **THEN** a decision record with the numbers, splits, and criterion is written under the repository documentation directory
