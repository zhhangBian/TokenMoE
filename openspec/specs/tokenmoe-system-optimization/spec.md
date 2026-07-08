# tokenmoe-system-optimization Specification

## Purpose
TBD - created by archiving change implement-tokenmoe-vllm-prototype. Update Purpose after archive.
## Requirements
### Requirement: Expert prefetch and residency planning
The system SHALL use RouteSig predictions to estimate future expert demand and plan expert residency or prefetch actions without changing router correctness.

#### Scenario: Real expert offload is unavailable
- **WHEN** the current vLLM environment lacks a safe expert-weight prefetch/offload hook
- **THEN** the system provides a simulator or proxy that reports prefetch hit rate, wasted prefetch, and estimated stall reduction

#### Scenario: Prediction misses an expert
- **WHEN** a predicted expert set omits a true routed expert
- **THEN** execution falls back to normal expert loading or dispatch behavior

### Requirement: Expert-overlap-aware scheduling prototype
The system SHALL score candidate batches of ready agent nodes using predicted expert overlap, active expert fanout, waiting penalty, and dependency constraints.

#### Scenario: Dependency would be violated
- **WHEN** an agent node depends on an unfinished predecessor
- **THEN** the scheduler prototype does not schedule that node regardless of predicted expert overlap

#### Scenario: Replay scheduler runs
- **WHEN** offline replay is used before online vLLM integration
- **THEN** the system reports active expert fanout and per-expert token counts before and after scheduling

### Requirement: Proactive expert replica placement prototype
The system SHALL evaluate future expert demand from RouteSig as an input to proactive expert replica placement.

#### Scenario: EPLB integration is too invasive
- **WHEN** direct vLLM EPLB policy integration is not safe for the current phase
- **THEN** the system runs a replay simulator comparing moving-average EPLB and TokenMoE predicted demand

#### Scenario: Replica migration cost is high
- **WHEN** predicted benefit does not exceed migration or memory cost
- **THEN** the placement prototype avoids or rejects the proactive replica move

### Requirement: Fallback under prediction error
The system SHALL degrade to baseline vLLM behavior when RouteSig confidence is low, metadata is missing, or predictions are inaccurate.

#### Scenario: Confidence is low
- **WHEN** a RouteSig lookup returns low confidence
- **THEN** prefetch, scheduling, and placement optimizations are disabled or reduced for that request

