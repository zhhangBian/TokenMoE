## ADDED Requirements

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
The runner SHALL execute each agent command with `<runtime> exec` in the session's container, with stdout and stderr merged into one pipe. It SHALL tee every read of that pipe to `tool_outputs/<tool_call_id>.out` and record one chunk (`chunk_index`, `chunk_time`, `byte_offset`, `byte_length`) per read. It SHALL return to mini-swe-agent the same output and the same exceptions that mini-swe-agent's own environment returns.

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
