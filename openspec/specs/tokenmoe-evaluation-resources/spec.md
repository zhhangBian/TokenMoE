# tokenmoe-evaluation-resources Specification

## Purpose
TBD - created by archiving change implement-tokenmoe-vllm-prototype. Update Purpose after archive.
## Requirements
### Requirement: Model preparation plan
The system SHALL document and script configurable preparation for runnable MoE models, including small-model and trace-only fallbacks.

#### Scenario: Small model run is selected
- **WHEN** the user selects a small model set
- **THEN** the preparation script targets models suitable for local smoke tests before large target models

#### Scenario: Target model run is selected
- **WHEN** the user selects target models such as Mixtral, Qwen3 MoE, or DeepSeek-V2-Lite
- **THEN** the preparation script uses configurable download and cache paths

### Requirement: Agent workload preparation
The system SHALL prepare workloads covering planner-coder-tester, search-summarize, tool-use, multi-agent discussion, and simplified SWE-agent/code repair workflows.

#### Scenario: Workloads are generated
- **WHEN** workload preparation runs
- **THEN** each workflow is available in a normalized JSONL format with agent metadata fields

#### Scenario: External datasets are unavailable
- **WHEN** an external dataset cannot be downloaded
- **THEN** synthetic or trace-only workload samples remain available for running the prototype

### Requirement: Dataset mapping documentation
The system SHALL document how each workload maps source fields to `agent_id`, `role`, `phase`, `graph_node_type`, `tool_type`, and `prompt_block_types`.

#### Scenario: New workload is inspected
- **WHEN** a user opens the dataset documentation
- **THEN** the mapping from source records to TokenMoE metadata is explicit

### Requirement: Reproducibility commands
The system SHALL provide commands for downloading models, preparing workloads, running trace collection, and running analysis.

#### Scenario: User follows README
- **WHEN** a user follows the documented commands in order
- **THEN** the prototype can be reproduced or clearly reports which optional resource is missing

