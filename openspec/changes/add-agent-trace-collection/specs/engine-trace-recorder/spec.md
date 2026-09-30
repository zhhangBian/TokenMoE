## ADDED Requirements

### Requirement: Recorder enablement
The vLLM fork SHALL create the trace recorder only when the environment variable `TOKENMOE_TRACE_DIR` is set. Each engine lifetime SHALL write into its own subdirectory `<TOKENMOE_TRACE_DIR>/<engine_instance_id>/`. When `TOKENMOE_TRACE_DIR` is unset, the fork SHALL behave exactly like upstream v0.30.0 with the same flags.

#### Scenario: Recorder disabled
- **WHEN** the engine starts without `TOKENMOE_TRACE_DIR`
- **THEN** no recorder is created, no trace files are written, and API responses match upstream behavior

#### Scenario: Recorder enabled
- **WHEN** the engine starts with `TOKENMOE_TRACE_DIR=/x` and `--enable-return-routed-experts`
- **THEN** the directory `/x/<engine_instance_id>/` exists and contains `engine_meta.json` before the first request is accepted

### Requirement: Startup configuration validation
When the recorder is enabled, the engine SHALL refuse to start if any of the following holds:
- `--enable-return-routed-experts` is off;
- prefix caching is disabled;
- a speculative decoding config is set;
- async scheduling is enabled;
- the data-parallel size is greater than 1.

The check SHALL run before model weights are loaded. The error message SHALL name the offending setting.

#### Scenario: Async scheduling left on
- **WHEN** the recorder is enabled and async scheduling resolves to true
- **THEN** startup fails with an error naming async scheduling and suggesting `--no-async-scheduling`

#### Scenario: Prefix caching disabled
- **WHEN** the recorder is enabled and the engine is started with `--no-enable-prefix-caching`
- **THEN** startup fails before any model weights load, with an error naming prefix caching

### Requirement: Engine metadata
The recorder SHALL write `engine_meta.json` containing:
- `raw_format_version`, `engine_instance_id`, the vLLM version and fork commit;
- the effective engine configuration: tensor, expert and data parallel sizes, dtype, quantization, `max_model_len`, `gpu_memory_utilization`, `max_num_seqs`, `max_num_batched_tokens`, KV block size, prefix caching, async scheduling, model runner version, scheduling policy, capture flags;
- `clock_domain_id` (the host boot id), one pair of monotonic and wall-clock anchor timestamps, and the process id.

#### Scenario: Metadata reflects resolved values
- **WHEN** the engine resolves `max_num_batched_tokens` to a default rather than a CLI value
- **THEN** `engine_meta.json` records the resolved value

### Requirement: Captured-layer map
The recorder SHALL write `layer_map.json`, listing every MoE layer bound for routed-experts capture, in ascending layer-id order. Each entry SHALL give the layer id and the capture path: `router`, `monolithic` or `capture_source`. With tensor parallelism, only TP rank 0 SHALL write it.

Routing arrays SHALL contain exactly these layers, in this order.

#### Scenario: Dense layers excluded
- **WHEN** a model has dense MLP layers before its first MoE layer
- **THEN** those layer ids are absent from `layer_map.json` and from the layer axis of every routing array

#### Scenario: Unbound MoE path
- **WHEN** no MoE layer can be bound for capture
- **THEN** engine startup fails instead of recording all-zero routing

### Requirement: Engine step records
For every engine step that computes at least one token, the recorder SHALL append one record to `steps.jsonl` with:
- `step_index`, incrementing from 0 within the engine lifetime;
- the step timestamps on the monotonic clock: scheduling start, dispatch (`step_started_at`), model output ready (`step_finished_at`), output processed;
- the total number of scheduled tokens, the numbers of running and waiting requests, and KV cache usage;
- `entries`: every computed range as `(llm_request_id or null, vllm_request_id, phase, token_start, token_end)`, right-open.

Phase SHALL be one of `recompute`, `new_prefill` or `decode`. Entries SHALL cover exactly the tokens the scheduler scheduled in that step. A range that crosses a phase boundary SHALL be split into one entry per phase.

#### Scenario: Chunked prefill
- **WHEN** a 10,000-token prompt with 2,000 cached tokens is prefilled in chunks of 4,096 and 3,904 tokens
- **THEN** two steps carry new_prefill entries `[2000, 6096)` and `[6096, 10000)` for that request

#### Scenario: Prefill completes and decode starts in the same step
- **WHEN** a request's remaining prompt tokens and its first decode position are scheduled together
- **THEN** that step holds one new_prefill entry and one decode entry for the request, with adjacent ranges

#### Scenario: Request without a TokenMoE id
- **WHEN** a request is served without `tokenmoe_llm_request_id`
- **THEN** its computed ranges still appear in step entries, with a null `llm_request_id`

### Requirement: Phase classification and preemption
For each request, the recorder SHALL track the highest token index already computed. It SHALL classify scheduled tokens below that index as `recompute`, tokens from that index up to `num_prompt_tokens` as `new_prefill`, and all later tokens as `decode`. `num_cached_tokens` SHALL be the start of the request's first scheduled range. The recorder SHALL count preemptions per request.

#### Scenario: Preempted request is recomputed
- **WHEN** a request that has computed 5,000 tokens is preempted and rescheduled from token 0 with no prefix hit
- **THEN** the rescheduled range `[0, 5000)` is recorded as `recompute`, the request's `num_preemptions` is 1, and its routing keeps the rows from the first computation

#### Scenario: Prefix-cache hit
- **WHEN** a request's first scheduled range starts at token 3,072
- **THEN** its `num_cached_tokens` is 3,072 and no step entry covers `[0, 3072)` for that request

### Requirement: Request records
When a request finishes, the recorder SHALL append one record to `requests.jsonl`. This covers normal stops, length limits, aborts and errors. The record SHALL contain:
- `llm_request_id` and `vllm_request_id`;
- `engine_received_at`, `first_scheduled_at`, `first_token_at` and `inference_finished_at`;
- `num_prompt_tokens`, `num_output_tokens`, `num_cached_tokens` and `num_preemptions`;
- the engine finish status, the sampling seed, and the routing file name (or null).

#### Scenario: Aborted request
- **WHEN** the client disconnects while a request is decoding
- **THEN** a request record is written with an abort finish status, and routing for its computed tokens is still written

### Requirement: Per-request routing arrays
For every finished request that carries `tokenmoe_llm_request_id`, the recorder SHALL write `routing/<llm_request_id>.npz` with:
- `token_ids` of length `T = num_prompt_tokens + num_output_tokens`;
- `row_start = num_cached_tokens`;
- `experts` of shape `[T - 1 - row_start, L, K]`;
- `step_index` (one value per row);
- `layer_ids`.

Row `i` SHALL hold the logical expert IDs selected for token position `row_start + i`. These IDs SHALL be taken before any EPLB mapping, from the step slice in which the token was first computed. Expert IDs SHALL be stored as uint8 when the model has at most 256 routed experts, and as uint16 otherwise. Router weights (scores) SHALL NOT be recorded.

#### Scenario: Shape of a normal request
- **WHEN** a request with 1,000 prompt tokens, 200 cached tokens and 50 output tokens finishes
- **THEN** `experts` has 849 rows, row 0 corresponds to token position 200, and `step_index` has 849 entries

#### Scenario: Cached prefix not re-exposed
- **WHEN** a request hits the prefix cache for its first 4,096 tokens
- **THEN** its routing file contains no rows for positions below 4,096

### Requirement: GPT-OSS monolithic Triton MXFP4 capture
The fork SHALL support routed-experts capture on the monolithic Triton MXFP4 expert kernel that GPT-OSS uses on Hopper. The kernel's routing arithmetic SHALL stay unchanged. Binding capture on that kernel SHALL NOT raise.

#### Scenario: GPT-OSS-120B starts with capture
- **WHEN** GPT-OSS-120B is served on Hopper with the recorder enabled
- **THEN** startup succeeds, `layer_map.json` lists 36 layers on the `monolithic` path, and routing rows hold 4 distinct IDs in `[0, 128)`

### Requirement: API response suppression
While the recorder is enabled, the fork SHALL NOT assemble routed experts into engine outputs. OpenAI-compatible responses SHALL therefore carry no `routed_experts` field.

#### Scenario: Chat completion under the recorder
- **WHEN** a chat completion is served with the recorder enabled
- **THEN** the response has no `routed_experts` payload, and the routing is only in the trace directory

### Requirement: Non-blocking trace writing
All trace file I/O SHALL happen on a writer thread. On engine shutdown, the recorder SHALL flush and close every file.

#### Scenario: Clean shutdown
- **WHEN** the server receives SIGTERM after serving requests
- **THEN** all finished requests have complete request records and routing files, and `steps.jsonl` ends with a complete line

### Requirement: Fork unit tests
The fork SHALL include tests under `tests/tokenmoe/` covering:
- phase classification, including preemption and mixed prefill/decode steps;
- step-slice routing attribution;
- routing file shape and dtype;
- startup validation;
- API suppression.

These tests SHALL run on CPU without model weights.

#### Scenario: Tests run on CPU
- **WHEN** `pytest tests/tokenmoe` runs in the fork venv on a machine without GPUs
- **THEN** all tests pass
