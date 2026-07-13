# tokenmoe-evaluation-resources Delta

## MODIFIED Requirements

### Requirement: Real-model validation standard
The system SHALL treat real vLLM MoE prompt traces with router scores at
1k-record scale as the completion standard for this change.

#### Scenario: Fast tests pass without real traces
- **WHEN** only mock, synthetic, deterministic, or unit-test traces have been used
- **THEN** the change is not considered validated

#### Scenario: Required validation runs complete
- **WHEN** Qwen3-30B-A3B has run 1k prompt records for each of ShareGPT, LMSYS, SWE-agent, OpenCodeInstruct, and OpenMathInstruct-2, and DeepSeek-V2-Lite-Chat has run 1k prompt records for ShareGPT and SWE-agent, all with schema v3 router-score capture
- **THEN** the generated prediction, ablation, and gate reports can be used as completion evidence

#### Scenario: Optional scale runs are requested
- **WHEN** 5k-record runs for Qwen3 on ShareGPT and SWE-agent are requested
- **THEN** they may be executed and reported without blocking primary completion

#### Scenario: Existing baselines are preserved
- **WHEN** new 1k or 5k artifacts are produced
- **THEN** they are stored under distinct names and the 256-record v2 artifacts are retained as regression baselines

#### Scenario: Routed-experts preflight runs
- **WHEN** a required validation run starts
- **THEN** it asserts the selected model is MoE, routed-experts return is enabled, score capture status is explicit, pipeline parallelism is disabled, context parallelism is disabled, KV transfer/connectors are disabled, and TP/EP, dtype, max model length, and GPU memory settings are explicit

#### Scenario: Non-MoE layers are detected
- **WHEN** trace collection writes model metadata
- **THEN** it persists `moe_layer_ids` and downstream validation rejects metrics that treat non-MoE layers as active experts

#### Scenario: Segment token alignment is checked
- **WHEN** trace collection maps prompt block character spans to token spans
- **THEN** it compares the aligned token IDs with vLLM `prompt_token_ids` and marks the record segment-unavailable on mismatch

## ADDED Requirements

### Requirement: Enriched SWE-agent metadata workload
The system SHALL produce SWE-agent workloads with enriched typed agent
metadata sufficient for high-cardinality RouteSig keys and ablations.

#### Scenario: Enriched fields are emitted
- **WHEN** the SWE-agent adapter converts trajectory records
- **THEN** each record and its blocks include `trajectory_phase`, `tool_type`, `event_outcome`, `dag_depth`, `group_local_step_index`, `on_critical_path`, and refined segment roles (system, code_context, trajectory_event, tool_result)

#### Scenario: Heuristic provenance is recorded
- **WHEN** enriched fields are derived heuristically from action sequences
- **THEN** the manifest records the heuristic version used for derivation

#### Scenario: Underivable fields are marked
- **WHEN** an enriched field cannot be derived from the source data
- **THEN** it is marked unavailable in the manifest instead of silently defaulted

#### Scenario: Group scale increases
- **WHEN** the SWE-agent workload is produced at 1k records
- **THEN** it contains at least 64 source groups with valid dependency edges and ready times

### Requirement: Optional second agent dataset
The system SHALL treat a second agent-metadata dataset as optional evidence
that does not block completion.

#### Scenario: Second agent source is adopted
- **WHEN** an orchestrator-generated trajectory source or second public trajectory corpus is adopted
- **THEN** its adapter emits native role/phase/tool metadata, dependency edges, and ready times, and its results are reported as secondary agent evidence

#### Scenario: Second agent source is skipped
- **WHEN** no second agent source is adopted in this change
- **THEN** the decision and rationale are recorded and completion is not blocked
