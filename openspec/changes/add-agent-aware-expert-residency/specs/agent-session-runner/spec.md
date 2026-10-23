## ADDED Requirements

### Requirement: Optional live residency event sink

The runner SHALL offer an opt-in bounded event sink for session lifecycle, request creation/completion, parsed tool issuance, tool start/output/completion, and cancellation. Events SHALL carry unique IDs, sequence, clock domain, source observation time, model/template/session/request/tool references as applicable, and dependency information known at that time. The controller SHALL record receipt time separately. The sink SHALL not import vLLM, alter prompts or tool observations, execute extra agent actions, or require a finalized trace before delivering events.

#### Scenario: Tool output becomes available during execution
- **WHEN** the streaming executor reads a tool-output chunk while the sink is enabled
- **THEN** it can emit a reference to exactly the observed byte range immediately while preserving the existing raw output and agent-visible observation behavior

### Requirement: Bounded side-channel failure handling

A stalled or unavailable sink SHALL not indefinitely block tool execution or model-request delivery. The runner SHALL use bounded buffering, record dropped/coalesced events and disconnects, and preserve terminal/dependency state through an observed snapshot or explicit resynchronization. Future bytes or final outcomes SHALL not be inserted into earlier snapshots. Side-channel overhead SHALL be measured in live experiments.

#### Scenario: Event consumer falls behind
- **WHEN** output events exhaust the sink buffer
- **THEN** the runner follows its declared loss/coalescing policy, retains the authoritative tool trace, records the gap, and enables conservative controller recovery without changing the tool result
