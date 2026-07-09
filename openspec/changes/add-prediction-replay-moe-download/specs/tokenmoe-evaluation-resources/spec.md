## MODIFIED Requirements

### Requirement: Model preparation plan
The system SHALL document and script configurable preparation for runnable MoE
models, with `moe-target` as the primary experiment profile and dense-baseline
or trace-only profiles available for comparison and fallback.

#### Scenario: Target model run is selected
- **WHEN** the user selects the target MoE profile
- **THEN** the preparation script targets real MoE models such as Mixtral, Qwen3 MoE, or DeepSeek-V2-Lite using configurable download and cache paths

#### Scenario: Optional small model run is selected
- **WHEN** the user explicitly selects a small model set for tests or fallback
- **THEN** the preparation script targets small MoE models without replacing the target MoE experiment profile

#### Scenario: Dense baseline run is selected
- **WHEN** the user selects a dense baseline profile
- **THEN** the preparation script can target dense Qwen models while marking them as not suitable for routed expert capture

#### Scenario: Download dry run is selected
- **WHEN** the user requests dry-run behavior
- **THEN** the preparation script prints the planned model IDs, local paths, endpoint, and profile without downloading model files

### Requirement: Reproducibility commands
The system SHALL provide commands for downloading models, preparing workloads,
running trace collection, and running analysis.

#### Scenario: User follows README
- **WHEN** a user follows the documented commands in order
- **THEN** the prototype can be reproduced or clearly reports which optional resource is missing

#### Scenario: MoE model profile is selected
- **WHEN** the user selects the `moe-target` download profile
- **THEN** the documented command prepares model paths intended for downstream vLLM routed expert capture

## ADDED Requirements

### Requirement: Experiment-local authenticated model download
The system SHALL support the user's local authenticated model download workflow
using `https://hf-mirror.com` and the hardcoded token already managed in
`/home/youwei/bzh/model/download.py`, without printing or reporting the token.

#### Scenario: Hardcoded token is used
- **WHEN** the model download script invokes the Hugging Face download API
- **THEN** the script passes the local hardcoded token without printing it in logs or dry-run output

#### Scenario: Mirror endpoint is used
- **WHEN** the user runs the model download script
- **THEN** the script uses `https://hf-mirror.com` as the default endpoint

#### Scenario: Dry run is used
- **WHEN** the user requests dry-run behavior
- **THEN** the script prints planned repo IDs, local paths, endpoint, and profile without printing the token or downloading model files
