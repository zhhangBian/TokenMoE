# tokenmoe-route-signature Specification

## Purpose
TBD - created by archiving change implement-tokenmoe-vllm-prototype. Update Purpose after archive.
## Requirements
### Requirement: RouteSignature data model
The system SHALL define a `RouteSignature` data structure keyed by agent metadata and layer ID that stores expert probabilities, top experts, entropy, confidence, miss costs, sample count, and update timestamp.

#### Scenario: Signature is created from traces
- **WHEN** routed expert traces are processed for a metadata key and layer
- **THEN** the system creates or updates a `RouteSignature` for that key and layer

### Requirement: Hierarchical fallback lookup
The system SHALL support RouteSig lookup fallback in this order: `(agent_id, role, phase, block_type)`, `(role, phase, block_type)`, `(role, phase)`, `role`, `global`.

#### Scenario: Specific key exists
- **WHEN** a signature exists for `(agent_id, role, phase, block_type)`
- **THEN** lookup returns the specific signature before checking broader fallbacks

#### Scenario: Cold start key is missing
- **WHEN** a specific metadata key has insufficient samples
- **THEN** lookup falls back to the most specific available broader signature

### Requirement: Online update
The system SHALL update RouteSig statistics online from real router traces without changing router behavior.

#### Scenario: New routed trace arrives
- **WHEN** a request finishes and routed expert IDs are available
- **THEN** the corresponding RouteSig distributions and sample counts are updated

#### Scenario: Low confidence signature
- **WHEN** a signature has low sample count, high entropy, or unstable recent traces
- **THEN** the system marks confidence low so optimizations can be disabled or fall back

### Requirement: Predictor evaluation
The system SHALL evaluate RouteSig predictions against global frequency, request/history LRU, and sequence-level history baselines.

#### Scenario: Top-M hit rate evaluation
- **WHEN** traces are split into training and evaluation windows
- **THEN** the system reports top-M hit rate for RouteSig and all required baselines

#### Scenario: Layer sensitivity evaluation
- **WHEN** evaluation completes
- **THEN** the system reports which MoE layers benefit most from agent-conditioned prediction

