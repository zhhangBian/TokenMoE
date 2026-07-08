# tokenmoe-tracing Specification

## Purpose
TBD - created by archiving change implement-tokenmoe-vllm-prototype. Update Purpose after archive.
## Requirements
### Requirement: Agent metadata capture
The system SHALL represent each agent LLM request with `AgentNodeMeta` containing `request_id`, `agent_id`, `role`, `phase`, `tool_type`, `graph_node_type`, and `prompt_block_types`.

#### Scenario: Metadata is attached to a trace record
- **WHEN** a trace record is emitted for an agent request
- **THEN** the record includes all required `AgentNodeMeta` fields for that `request_id`

#### Scenario: Optional tool type is absent
- **WHEN** a request is not associated with a tool
- **THEN** `tool_type` is recorded as null or an equivalent empty value without dropping the request

### Requirement: Routed expert ID trace
The system SHALL capture per-token, per-layer selected expert IDs after the MoE router computes top-k and before any correctness-changing substitution.

#### Scenario: Router executes normally
- **WHEN** a MoE layer routes a batch of tokens
- **THEN** the real router output is used for model execution and the selected logical expert IDs are available for tracing

#### Scenario: EPLB is enabled
- **WHEN** EPLB maps logical experts to physical expert replicas
- **THEN** the trace preserves logical expert IDs or records enough mapping metadata to recover logical expert IDs

### Requirement: Router score trace option
The system SHALL support optional capture of router top-k scores with the same token, layer, and top-k indexing as selected expert IDs.

#### Scenario: Score capture disabled
- **WHEN** router-score tracing is disabled
- **THEN** selected expert ID tracing continues to work and score fields are omitted or null

#### Scenario: Score capture enabled
- **WHEN** router-score tracing is enabled
- **THEN** each captured score aligns with the corresponding selected expert ID

### Requirement: Trace persistence
The system SHALL persist traces in JSONL and SHALL define a parquet-compatible schema for post-processing.

#### Scenario: JSONL trace output
- **WHEN** trace collection completes
- **THEN** the output contains one valid JSON object per request or trace segment

#### Scenario: Histogram derivation
- **WHEN** selected expert IDs are present
- **THEN** active expert histograms can be derived per layer and request

### Requirement: Trace correctness boundary
The system MUST NOT change model output semantics while tracing.

#### Scenario: Prediction or tracing fails
- **WHEN** tracing metadata is missing or a prediction is unavailable
- **THEN** vLLM execution falls back to normal routing and generation behavior

