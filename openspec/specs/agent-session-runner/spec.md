# Agent Session Runner

## Purpose
Define reproducible mini-swe-agent sessions, request and tool provenance, container execution, and host load recording for TokenMoE data collection.

## Requirements

### Requirement: Run command
`python -m tokenmoe_collect run --config <run.yaml>` SHALL:
- run the configured SWE-bench Verified instances with mini-swe-agent 2.4.6 against an already running fork server;
- keep at most `num_concurrent_sessions` sessions active at once;
- write every harness-side record under the run directory.

The run config SHALL name:
- the model config file and the server URL;
- the instance list;
- the concurrency and the base seed;
- the step and time limits;
- the container runtime;
- the output directory.

#### Scenario: Concurrency bound
- **WHEN** a run with 20 instances and `num_concurrent_sessions: 4` executes
- **THEN** no more than 4 sessions are active at any time, and the application run records share one `concurrency_group_id`

#### Scenario: Output directory must be new
- **WHEN** the configured output directory already contains a run
- **THEN** the command exits without writing anything

### Requirement: Request ID propagation
The runner SHALL give every model call a new `llm_request_id` and send it as `vllm_xargs.tokenmoe_llm_request_id` in `POST /v1/chat/completions`. A retried call SHALL reuse the `step_index`, increment `attempt_id`, and get a new `llm_request_id`.

#### Scenario: Retry after a transport error
- **WHEN** the first attempt for step 7 fails with a connection error and the second attempt succeeds
- **THEN** two request records exist for step 7, with attempt ids 0 and 1 and distinct `llm_request_id`s

### Requirement: Reasoning pass-back
When the model returns reasoning text, the runner SHALL keep it on the assistant message. It SHALL send that reasoning back in the `reasoning` field of the message in every later request of the session.

#### Scenario: Thinking kept across turns
- **WHEN** step 3's response contains reasoning and step 4 is sent
- **THEN** step 4's request body contains step 3's assistant message with the same `reasoning` text

### Requirement: Harness request records
For each model call, the runner SHALL record:
- `llm_request_id`, `application_run_id`, `session_id`, `step_index`, `attempt_id` and `prev_tool_call_ids`;
- `request_created_at`, `request_sent_at` and `response_received_at`;
- the messages appended since the session's previous request, with per-message provenance: `segment_type`, `source_tool_call_id`, and the payload offsets inside the message content;
- the full assistant response (content, reasoning, tool calls), the finish reason and the request status.

If a request's message list is not an append-extension of the previous request's list, the runner SHALL record that request's full message list.

#### Scenario: Tool output provenance
- **WHEN** a request contains the observation for tool call `tc_x`, wrapped in mini-swe-agent's observation template
- **THEN** that message's provenance marks it `tool_output` with `source_tool_call_id = tc_x`, plus the character range of the tool payload inside the message

### Requirement: Streaming tool executor
The runner SHALL execute each agent command with `<runtime> exec` in the session's container, with stdout and stderr merged into one pipe. It SHALL tee every read of that pipe to `tool_outputs/<tool_call_id>.out` and record one chunk (`chunk_index`, `chunk_time`, `byte_offset`, `byte_length`) per read. It SHALL return to mini-swe-agent the same output and the same error-result dictionaries that mini-swe-agent's own environment returns.

#### Scenario: Chunk timing
- **WHEN** a command prints one line per second for 5 seconds
- **THEN** its output file holds all 5 lines, at least 5 chunk records exist with increasing `chunk_time`, and the offsets tile the file

#### Scenario: Timeout
- **WHEN** a command exceeds the configured timeout
- **THEN** the exec process is killed, the tool call is recorded with `timed_out = true`, the output received so far is kept, and the agent receives mini-swe-agent's standard timeout observation

### Requirement: Tool call records
For each executed command, the runner SHALL record:
- `tool_call_id`, `session_id` and `llm_request_id_issuing`;
- `call_index_within_request` and `tool_name`;
- the full command as `tool_args_ref`;
- a harness-side `parsed_at`, plus `started_at` and `finished_at`;
- `exit_status`, `timed_out`, `output_bytes_raw` and `output_bytes_seen`.

`output_bytes_seen` SHALL be the UTF-8 byte length of the observation text that is passed to the agent.

#### Scenario: Truncated observation
- **WHEN** a command prints 50,000 characters and mini-swe-agent's template keeps the first and last 5,000
- **THEN** `output_bytes_raw` equals the output file size and `output_bytes_seen` is smaller than it

### Requirement: Container lifecycle
The runner SHALL start one container per session from the SWE-bench Verified instance image, with the configured runtime (`docker` or `podman`). It SHALL remove the container when the session ends, whether the session succeeds, fails or times out.

#### Scenario: Session crash cleanup
- **WHEN** the agent raises an unexpected exception mid-session
- **THEN** the session is recorded with a failed final status and its container is removed

### Requirement: Session and application run records
For each instance, the runner SHALL write one application run record and one session record (§4.1, §4.2), with:
- start and finish times;
- the final status: `succeeded`, `failed`, `cancelled` or `timeout`;
- the mini-swe-agent exit reason;
- the per-run seed;
- `benchmark_score = null`.

#### Scenario: Step limit reached
- **WHEN** a session hits mini-swe-agent's step limit
- **THEN** the session's final status is `timeout` and its exit reason names the limit

### Requirement: Host load sampling
While a run is active, the runner SHALL sample host load once per second and append a row with:
- time and `host_id`;
- 1-minute load average, CPU utilization and memory utilization;
- the number of live tool exec processes;
- per-GPU utilization and memory used;
- network rx/tx bytes.

#### Scenario: GPU metrics unavailable
- **WHEN** NVML cannot be initialized
- **THEN** sampling continues with the GPU columns null, and a warning is logged once

### Requirement: Deterministic seeds
The runner SHALL derive each request's sampling seed from `(base_seed, benchmark_item_id, step_index, attempt_id)` by a stable hash, and SHALL record it.

#### Scenario: Seed reproducibility
- **WHEN** the same run config is executed twice
- **THEN** corresponding requests carry identical seeds

### Requirement: Server-independent harness
The `tokenmoe_collect` package SHALL NOT import `vllm`. Its tests SHALL run on CPU without a model server.

#### Scenario: Harness tests without GPUs
- **WHEN** `pytest collection/tests` runs in the harness venv on a machine without GPUs or a server
- **THEN** all tests pass, using a fake server and a local subprocess runtime

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
