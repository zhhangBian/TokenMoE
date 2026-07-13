# tokenmoe-evaluation-resources Specification

## Purpose
Define the real-model, dataset, trace-collection, and reporting resources
required to validate TokenMoE prompt-only MoE prediction and replay experiments.

## Requirements

### Requirement: Model preparation plan
The system SHALL document and validate real local MoE models for vLLM
routed-experts prompt tracing, with Qwen3-30B-A3B as the primary target and
DeepSeek-V2-Lite-Chat as required cross-model validation.

#### Scenario: Primary model run is selected
- **WHEN** the primary model profile is used
- **THEN** the experiment targets `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B` for vLLM routed-experts prompt trace collection

#### Scenario: Cross-model run is selected
- **WHEN** cross-model validation is used
- **THEN** the experiment targets `/home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat` for ShareGPT and SWE-agent prompt trace collection

#### Scenario: Optional compatibility run is selected
- **WHEN** optional compatibility validation is requested
- **THEN** the experiment may target `/home/youwei/bzh/model/mistralai/Mixtral-8x7B-Instruct-v0.1` without blocking primary completion

#### Scenario: Dense model is selected
- **WHEN** a dense model is selected for the main experiment path
- **THEN** the system rejects it as unsuitable for routed expert capture rather than treating it as successful TokenMoE evidence

### Requirement: Reproducibility commands
The system SHALL provide commands for dataset download, dataset conversion,
prompt-only vLLM routed trace collection, prediction evaluation, and scheduler
replay.

#### Scenario: User follows README
- **WHEN** a user follows the documented commands in order
- **THEN** the prototype either produces the required real-model prompt traces and reports or clearly reports which model, dataset, permission, or runtime resource is missing

#### Scenario: vLLM trace collection is documented
- **WHEN** the user runs the documented trace collection command
- **THEN** the command uses vLLM routed-experts capture and stores prompt-only routed expert traces

#### Scenario: Validation command checks backend
- **WHEN** validation or report generation runs
- **THEN** it rejects traces whose backend is not `vllm-routed-experts`, whose fallback flag is true, whose schema versions are mixed, or whose routing is not prompt-only

### Requirement: Constrained external dataset download
The system SHALL update the user's local authenticated dataset download workflow
only by editing `DATASET_LIST` in `/home/youwei/bzh/dataset/download_dataset.py`.

#### Scenario: Dataset list is updated
- **WHEN** this change updates dataset download targets
- **THEN** only `DATASET_LIST` in `/home/youwei/bzh/dataset/download_dataset.py` is modified

#### Scenario: Dataset targets are listed
- **WHEN** `DATASET_LIST` is updated
- **THEN** it contains `anon8231489123/ShareGPT_Vicuna_unfiltered`, `lmsys/lmsys-chat-1m`, `nebius/SWE-agent-trajectories`, `nvidia/OpenCodeInstruct`, and `nvidia/OpenMathInstruct-2`

#### Scenario: Gated dataset access fails
- **WHEN** the existing download script cannot download a gated dataset
- **THEN** the failure is reported clearly without changing endpoint, token handling, retry behavior, or replacing the dataset silently

### Requirement: Isolated dataset adapters
The system SHALL convert external datasets to normalized prompt workloads using
a dedicated adapter folder outside core TokenMoE runtime modules.

#### Scenario: Dataset adapter runs
- **WHEN** a dataset converter processes a raw dataset under `/home/youwei/bzh/dataset/`
- **THEN** it emits one dataset-specific normalized workload JSONL under `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/`

#### Scenario: Prompt blocks are emitted
- **WHEN** a converter writes a normalized workload record
- **THEN** the record includes typed prompt blocks and character spans sufficient for later token-span alignment

#### Scenario: Provenance is emitted
- **WHEN** a converter writes a normalized workload record
- **THEN** the record includes schema version, source dataset, source index, source group ID, timestamp when available, claim scope, and unavailable field markers

#### Scenario: Agent DAG fields are emitted
- **WHEN** a converter processes SWE-agent trajectory records with reconstructable event order
- **THEN** it emits dependency edges and ready times for scheduler replay

#### Scenario: Agent DAG fields are unavailable
- **WHEN** a converter cannot reconstruct dependency edges or ready times from a source dataset
- **THEN** the manifest marks DAG scheduling unavailable for that output

#### Scenario: Manifest is written
- **WHEN** conversion completes
- **THEN** a manifest records source dataset path, output workload path, sample count, field mapping version, conversion command, unavailable source fields, license/access status, and redaction status

#### Scenario: Mixed workload is requested
- **WHEN** a mixed multi-dataset workload is requested in this change
- **THEN** it is treated as out of scope

### Requirement: Real-model validation standard
The system SHALL treat real vLLM MoE prompt traces and reports as the completion
standard for this change.

#### Scenario: Fast tests pass without real traces
- **WHEN** only mock, synthetic, deterministic, or unit-test traces have been used
- **THEN** the change is not considered validated

#### Scenario: Required validation runs complete
- **WHEN** Qwen3-30B-A3B has run 256 prompt records for each required dataset and DeepSeek-V2-Lite-Chat has run 256 prompt records for ShareGPT and SWE-agent trajectories
- **THEN** the generated prediction and scheduler replay reports can be used as completion evidence

#### Scenario: Routed-experts preflight runs
- **WHEN** a required validation run starts
- **THEN** it asserts the selected model is MoE, routed-experts return is enabled, pipeline parallelism is disabled, context parallelism is disabled, KV transfer/connectors are disabled, and TP/EP, dtype, max model length, and GPU memory settings are explicit

#### Scenario: Non-MoE layers are detected
- **WHEN** trace collection writes model metadata
- **THEN** it persists `moe_layer_ids` and downstream validation rejects metrics that treat non-MoE layers as active experts

#### Scenario: Segment token alignment is checked
- **WHEN** trace collection maps prompt block character spans to token spans
- **THEN** it compares the aligned token IDs with vLLM `prompt_token_ids` and marks the record segment-unavailable on mismatch

### Requirement: Claim-scope separated reporting
The system SHALL separate real agent metadata evidence from prompt-only and
domain-instruction evidence.

#### Scenario: Agent dataset is reported
- **WHEN** SWE-agent trajectory results are reported
- **THEN** the report may describe them as real-agent metadata evidence

#### Scenario: Non-agent dataset is reported
- **WHEN** ShareGPT, LMSYS, OpenCodeInstruct, or OpenMathInstruct-2 results are reported
- **THEN** the report labels them as chat/prompt-only or domain-instruction evidence and does not use them to support agent-DAG claims
