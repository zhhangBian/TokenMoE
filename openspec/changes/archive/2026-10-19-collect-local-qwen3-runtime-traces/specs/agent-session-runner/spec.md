## ADDED Requirements

### Requirement: Local collection execution ownership

During implementation of the local Qwen profile, the executing agent SHALL own resource preparation, necessary downloads, server/container execution, log diagnosis, collection-fault repair, targeted verification, and resumption of remaining tasks. The user's current execution authorization SHALL take precedence over historical handoff instructions reserving GPU runs/downloads for the user. Proposal/spec authoring SHALL NOT start actual collection.

#### Scenario: Collection infrastructure fails
- **WHEN** implementation encounters a locally repairable container-entry-point, dependency, tool-parsing, serving, or storage fault
- **THEN** the executing agent preserves evidence, repairs and verifies the issue, and continues without delegating routine repairs to the user

### Requirement: Task outcomes are independent of trace validity

The runner SHALL record mini-swe-agent terminal states/exit_reason separately from collection-data validity. Incorrect solutions, failing tests, nonzero tool exits, exhausted budgets, and model format errors SHALL retain their actual outputs. A session with real model requests whose computation traces, terminal state, and linked records satisfy the specification SHALL be eligible as valid collection data. Startup failures without actual model execution, lost engine records for accepted requests, misattributed routing, and damaged files SHALL be classified as collection faults requiring recollection.

The external executing agent SHALL NOT inject benchmark solutions or alter actual command exit codes, model responses, or terminal states to manufacture success. Model-generated format errors SHALL be distinguished from collector loss or misparsing of valid responses using the original response and parser behavior; collector defects SHALL be repaired.

A context-budget rejection before engine admission SHALL retain its actual failed task outcome. A session with earlier valid computed routing SHALL remain eligible as collection data when the rejection is evidenced and contains no fabricated engine timings, prompt token IDs, or routing. A session with no actual computed rows SHALL NOT count as coverage.

#### Scenario: Step limit with a complete trace
- **WHEN** a session reaches its 20-step limit with complete real requests, tool outputs, routing, and terminal-state records
- **THEN** the runner preserves timeout/LimitsExceeded and marks the planned unit valid without automatically rerunning it to improve submission rates

#### Scenario: Submission with missing routing
- **WHEN** an agent submits normally but a routing file is missing for one computed request
- **THEN** the submission state is preserved, the collection unit is marked needs_recollection, and the executing agent repairs the cause before recollecting

### Requirement: Append-only recovery provenance

HTTP retries SHALL retain existing step_index semantics, increment attempt_id, and allocate a new llm_request_id. Server restarts and task recollection SHALL allocate new application/session/request IDs and output directories, without representing a new execution as later turns of the same live session. The executing agent SHALL use `recovery_events.jsonl` to link the original planned unit, old/new directories and IDs, fault categories, evidence paths, repair commit/patch hashes, changed parameters, and verification results. Existing raw failure data SHALL NOT be overwritten.

When the same fault recurs unchanged, the executing agent SHALL change its diagnosis or repair approach rather than retry indefinitely. Resumption SHALL skip only planned units with validated data; a session or directory's existence SHALL NOT establish validity.

#### Scenario: Server crashes halfway through N4
- **WHEN** 12 N4 units have valid traces and 8 units are missing or invalid
- **THEN** recovery creates new executions only for those 8 units while preserving original errors, provenance links, and the 12 completed units

#### Scenario: Transport retry inside one agent step
- **WHEN** the first HTTP attempt in a step encounters a retryable transport error and the second succeeds
- **THEN** both requests retain the same step_index with distinct attempt_id/llm_request_id values, preserving the transport error without rewriting the first request as successful

### Requirement: Preserve request and tool visibility

The local runner SHALL preserve complete assistant content/reasoning/tool_calls, actual outgoing messages and provenance, and each tool execution's original command, raw merged output, chunk-read times/offsets, actual terminal state, and agent-visible observation. Reasoning SHALL be passed back according to the existing protocol. Parser adaptations or configuration optimization SHALL NOT silently disable thinking, remove or rewrite reasoning, replace tool results, or bypass the actual router.

HTTP errors SHALL preserve `http_status_code` and the actual error response body. Recovery of historical missing error bodies SHALL disclose exact-input replay evidence separately, including original/probe request IDs, matching model/messages/tools/sampling/seed and engine configuration, and the absence of engine records for both rejected attempts. Replayed responses SHALL NOT be presented as captured original responses.

#### Scenario: Tool output is truncated for the agent
- **WHEN** tool output exceeds the observation template's limit
- **THEN** all raw bytes are retained, chunks cover the raw file, provenance identifies the payload actually received by the agent, and reports count raw/seen bytes separately
