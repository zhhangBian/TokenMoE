## Context

Stage A of TokenMoE needs agent traces whose records join on shared IDs and one clock (`collection/data_collect.md` §1–§4). Three producers write them:
- the agent harness (sessions, requests, prompt composition);
- the tool executor (tool calls, raw output streams);
- the vLLM fork (engine steps, routing).

**Current state**
- The repo has no collection code; commit 6bcc6b6 removed it.
- The `vllm/` submodule sits on upstream main 0b3ba88f1 (≈ v0.22.0) plus the July score patch 90025dce2. That base cannot serve dots3-note.
- This change moves the fork to upstream tag **v0.30.0** (ced6857afa0e).

All fork paths below are inside the `vllm/` submodule, at tag v0.30.0.

**What v0.30.0 already does for routed-experts capture**
- Worker side: each step, top-k IDs are captured per layer into a device buffer. That buffer is copied to host together with the step's KV slot mapping (`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:46-248`).
- Scheduler side: `update_from_output` stores the step into a slot-indexed buffer. It already slices the step's rows per request, in model-runner order (`vllm/v1/core/sched/scheduler.py:1923-1937`).
- Capture hooks exist in two places:
  - `BaseRouter.select_experts`, before EPLB mapping (`router/base_router.py:291-305`);
  - monolithic kernels that expose `routing_replay_out` (`modular_kernel.py:1011-1067`).
- Routing is returned per request through the OpenAI API as base64 (`entrypoints/openai/chat_completion/serving.py:1077-1098`).

**What it lacks for our spec**
- engine-step records with timestamps and per-request token ranges;
- routing limited to computed tokens and keyed by our request ID;
- capture on GPT-OSS's Hopper MoE kernel, which errors out at startup;
- request lifecycle times.

**Constraints**
- Everything lives in `collection/`, plus Python-only fork edits.
- The user executes every GPU run, install, download and image pull.
- Real runs happen on Hopper with DeepSeek-V4-Flash, dots3-note and GPT-OSS-120B.
- The local 7xA100 host is for debugging only, with Qwen3-30B-A3B.

## Goals / Non-Goals

**Goals:**
1. A fork-side recorder that writes, without changing model outputs:
   - engine-step records;
   - request records;
   - per-request routing (expert IDs);
   - a captured-layer map.
2. Capture works on the MoE path each model actually uses:
   - Qwen3-30B-A3B on A100;
   - GPT-OSS-120B, DeepSeek-V4-Flash and dots3-note on Hopper.
3. A mini-swe-agent 2.4.6 session runner for SWE-bench Verified. It records application runs, sessions, requests, tool calls, raw tool-output streams and host load.
4. `finalize` into the §6 layout, plus a validator for the §7 rules that apply to single-loop, single-host runs.
5. Per model: serve configs, a smoke script, the capture on/off equivalence gate, and a pilot script. A README states the storage estimate.
6. `collection/data_collect.md` matches what the code emits.

**Non-Goals:**
- Offline labels (§5), predictors, progress reader, residency simulator.
- Layer-latency profiles: §3.2 `layer_latency_profile_ref` stays null.
- Cache events (§4.10), graph events (§4.11), multi-host clock sync.
- Harnesses other than mini-swe-agent.
- SWE-bench scoring: `benchmark_score` stays null.
- Expert or data parallelism, speculative decoding, multimodal input.

## Decisions

### D1. `collection/` layout; the package and the fork share only a file format

```text
collection/
  README.md                 setup, exact commands, storage estimate
  data_collect.md           authoritative spec (moved from /#data_collect.md)
  pyproject.toml            package tokenmoe_collect; extras [harness], [dev]
  tokenmoe_collect/
    ids.py clock.py records.py jsonl.py
    client.py               chat-completions transport: vllm_xargs, reasoning, timestamps
    executor.py             streaming tool executor around docker/podman exec
    hostload.py             1 Hz host sampler
    minisweagent_adapter.py model + environment classes for mini-swe-agent 2.4.6
    launcher.py             static records, container lifecycle, N concurrent sessions
    static.py               experiment config, model profile, role template, benchmark items
    align.py finalize.py validate.py equivalence.py estimate.py
    cli.py                  python -m tokenmoe_collect {run,finalize,validate,equiv,profile,estimate}
  configs/models/*.yaml     one file per model: serve flags, parsers, sampling, model facts
  configs/runs/*.yaml       local-debug, pilot, full
  scripts/                  serve.sh smoke.sh equivalence.sh pilot.sh pull_images.sh
  tests/                    CPU-only tests; never import vllm
```

- Fork code lives in the fork: `vllm/tokenmoe_trace.py` and `tests/tokenmoe/`.
- `tokenmoe_collect` never imports vllm, and the fork never imports `tokenmoe_collect`.
- They share only the raw file format. That format is documented in `data_collect.md` §8.1 and versioned by `raw_format_version`.
- Alternative: a repo-wide `tokenmoe/` package. The user rejected it on 2026-09-30.

### D2. The recorder is owned by the Scheduler in the EngineCore process

**Enablement.**
- It is created only when `TOKENMOE_TRACE_DIR` is set.
- It requires `--enable-return-routed-experts`.

**Why the Scheduler.** Everything it needs is already in that process:
- batch composition and per-request token counts (`SchedulerOutput`);
- per-step routing (`ModelRunnerOutput.routed_experts`);
- request state.

EngineCore adds only the early setup and the step timestamps.

| Hook | v0.30.0 location | Recorder action |
|---|---|---|
| engine init | `v1/engine/core.py:108-134`, `EngineCore.__init__`, before the executor is created (134) | Validate the resolved config (D9), so a bad launch fails before any weights load. Create `engine_instance_id` and its trace directory. Export `TOKENMOE_TRACE_ENGINE_DIR`, so the worker processes spawned at 134 write `layer_map.json` there (D7). |
| scheduler init | `scheduler.py:387-405`, after `RoutedExpertsManager` | Write `engine_meta.json` with the resolved KV and scheduler settings, read `layer_map.json`, and start the writer thread. |
| add request | `scheduler.py:2470` | Record `engine_received_at`. Read `llm_request_id` from `sampling_params.extra_args["tokenmoe_llm_request_id"]`. Keep `vllm_request_id`, which has a random suffix (`v1/engine/input_processor.py:266-279`). Keep the prompt token IDs. |
| schedule tail | `scheduler.py:1520-1533`, `_update_after_schedule`, before `num_computed_tokens += n` (1533); called at 1455 | For each request, record range `[num_computed_tokens, +n)`, split by phase (D4), as pending step entries. |
| step timestamps | `v1/engine/core.py:589-628`, `step()` | `t_sched` before `schedule()` (600); `t_dispatch` before `execute_model` (601); `t_output` after the model output / `sample_tokens` (605-609); `t_end` after `update_from_output` (614-616). |
| routing | `scheduler.py:1923-1937`, where `routing_offsets` come from `model_runner_output.req_ids` | Slice rows per request. Keep new_prefill/decode rows; drop recompute rows. |
| first token | `scheduler.py:2025-2031` (`new_token_ids`, `num_output_tokens_before`) | Record `first_token_at`. |
| preempt | `scheduler.py:1477-1518` | Count and time. `num_computed_tokens` resets at 1497. |
| finish | `scheduler.py:2561`, `_free_request` | Queue the request record and routing file. Every finish goes through here, aborts included (`finish_requests`, 2498). |
| shutdown | `scheduler.py:2817`, called from `core.py:756-762` | Flush the writer. |
| API suppression | `scheduler.py:2067-2105` | Skip routed-experts assembly for `EngineCoreOutput`, so the API omits it (`chat_completion/serving.py:1077-1098`, `completion/serving.py:573-592`). |

**Alternatives.**
- A `--scheduler-cls` plugin: the user chose direct fork edits.
- Returning routing in HTTP responses (C3): payloads reach tens of MB and carry no step attribution.

### D3. Attribute routing from each step's slice, not from the slot buffer

Upstream reads a finished prefill's routing back from the slot-indexed buffer through the request's block IDs (`scheduler.py:2076-2095`). The recorder instead takes every row from the step slice `routing_data[off : off+n]`, where `off` follows `model_runner_output.req_ids` (`scheduler.py:1935`).

Benefits:
- Step and phase attribution is exact.
- It does not depend on slot contents surviving preemption or eviction.
- Cached-prefix rows are never collected, which is what spec change C1 needs.

Upstream's slot buffer stays in place; the recorder simply does not read it.

### D4. Phases, entries and `num_cached_tokens` (decision C7a)

- Per request, retain the actual covered intervals. The initial cached prefix
  `[0, num_cached_tokens)` starts as covered. The highest end remains `hwm`,
  but a later cache hit may leave holes below it (Q9).
- Split scheduled ranges by coverage and by `P = num_prompt_tokens`:

  | Tokens | Phase |
  |---|---|
  | Already covered | `recompute` |
  | First computation below P | `new_prefill` |
  | First computation at/above P | `decode` |

- A range that crosses a boundary produces several entries for the same request in the same step.
- `num_cached_tokens` = the `s` of the request's first entry. This is the local prefix-cache hit, since KV connectors are refused. v0.30.0 keeps that number only inside `prefill_stats` (`scheduler.py:999-1006`).
- Recompute rows are not stored; the first computation's rows are kept.

Changes to §7:
- Rule 7: every new_prefill and decode token appears exactly once; any extra computation appears only as `recompute`.
- Rule 8: "requests within a step are distinct" becomes "one request's entries within a step do not overlap".

### D5. Clock and timestamps

- **Clock.**
  - Every producer uses `time.monotonic_ns()` (CLOCK_MONOTONIC), which is system-wide on one host.
  - `clock_domain_id` is `/proc/sys/kernel/random/boot_id`.
  - Each producer writes one (monotonic, wall) anchor pair at startup.
  - Tool timestamps are taken on the host, never inside containers.
- **Step window.**
  - `step_started_at` := `t_dispatch`.
  - `step_finished_at` := `t_output`.
  - `t_sched` and `t_end` are kept to measure recorder and scheduler overhead.
- **Request lifecycle.**
  - `engine_received_at`: at add request.
  - `first_scheduled_at`: `t_dispatch` of the first step with an entry for the request.
  - `first_token_at`: `t_output` of the step that produced output token 0.
  - `inference_finished_at`: `t_output` of the finishing step; the abort-processing time for aborts.
- **Tool calls.** `finalize` sets `issued_at` from the issuing request's `inference_finished_at`, so rule 4 holds by construction. The harness also keeps its own `parsed_at` for diagnostics.
- **Async scheduling must be off (C5).** With it on:
  - `max_concurrent_batches` becomes 2 (`config/vllm.py:589-599`);
  - EngineCore switches to `step_with_batch_queue` (`core.py:213-216, 235-237`), which overlaps scheduling of step N+1 with execution of step N;
  - so `t_dispatch`/`t_output` no longer bracket exactly one forward.

  v0.30.0 turns async scheduling on by default (`config/vllm.py:1438-1487`); serve configs pass `--no-async-scheduling`.

### D6. No router scores

Decided by the user on 2026-09-30: the recorder stores expert IDs only.

- **Why.** Residency simulation, prefetch hit rate, transfer volume and the §5 demand histograms depend only on which experts are selected; a selected expert must be resident whatever its weight. Scores would add `2·L·K` bytes per computed token, twice the uint8 IDs.
- **Consequences.**
  - The July score patch (90025dce2) is not carried over.
  - The fork does not touch `v1/outputs.py` or `router/base_router.py`.
  - The recorder does not depend on the model runner version. At v0.30.0, MRv2 is the default (`config/vllm.py:675-720`), and routed-experts capture is not on its unsupported list (`config/vllm.py:2815`). `engine_meta.json` records which runner vLLM picked.
- **If scores are ever needed**, they can be regenerated approximately:
  - replay each request's stored `token_ids`, prefill only, through the same fork, engine config and GPU type, with router weights captured;
  - expect small numeric differences and rare top-K flips at near-ties; flipped rows show up when replayed IDs are compared with the recorded ones.

### D7. Captured-layer map

- `bind_routed_experts_capturer` (`routed_experts_capturer.py:251-304`) records, for each bound module:
  - the layer id;
  - the capture path: router, monolithic, or capture-source module.
- TP rank 0 writes `layer_map.json` into the directory named by `TOKENMOE_TRACE_ENGINE_DIR` (D2, engine init).
- Binding happens in `initialize_from_config`, before the Scheduler is constructed (`gpu_worker.py:761-762`; `core.py:134, 145, 162`). The recorder can therefore read the map at init.
- The capture buffer is indexed by all hidden layers (`routed_experts_capturer.py:32-43`), so dense layers would be all-zero rows. dots3's first `first_k_dense_replace` layers are one example (`models/dots3_note/nvidia/model.py:540`).
- The recorder writes only bound layers, in ascending id order. That list is §3.2 `moe_layer_ids`.

### D8. GPT-OSS capture on the Hopper Triton MXFP4 kernel

**The problem.**
- On SM90 the MXFP4 oracle chooses the TRITON backend and returns `OAITritonMxfp4ExpertsMonolithic` first (`oracle/mxfp4.py:196-203, 332-351`). The device gate is at `gpt_oss_triton_kernels_moe.py:36-58`.
- That class does not implement `supports_routing_replay_capture`, so binding raises `ValueError` (`routed_experts_capturer.py:283-292`).

**The fix.** It goes inside `triton_kernel_moe_forward` (`gpt_oss_triton_kernels_moe.py:541-640`) and applies only when a capture function is set:
- **Legacy triton_kernels (v3.5.1, the bundled version).** Replace the fused `routing()` call (566-573) with its two constituent calls, `topk()` then `routing_from_bitmatrix()`, and capture `expt_indx`. Those two calls are exactly the body of `routing()` (`triton_kernels/routing.py:292-303` in the copy installed now).
- **triton_kernels v3.6+.** Capture `topk_result.indx` (574-590).
- The class then reports `supports_routing_replay_capture() == True`.
- The kernel's routing does not change bit-for-bit; the equivalence gate (D17) confirms it.

**Alternatives.**
- Force the modular `OAITritonExperts` when capture is on: this changes the served kernel and its routing arithmetic.
- Recompute top-k from router logits just for recording: this is the recomputed-router path the project rules out.

**Other models need no fix.**
- DSV4's non-MegaMoE path uses `FusedTopKBiasRouter`, hash layers included (`router/router_factory.py:227-241`; hash lookup at `fused_topk_bias_router.py:314-324`; MoE block at `models/deepseek_v4/nvidia/model.py:846-858, 995-1015`).
- dots3's `Dots3NoteMoE` inherits from `DeepseekV2MoE` (`models/dots3_note/nvidia/model.py:99`), whose router is a `BaseRouter`.
- MegaMoE carries its own capture source (`models/deepseek_v4/nvidia/model.py:199, 665, 696-697`).
- The monolithic Triton class only accepts Renormalize routing with SWIGLUOAI activation (`gpt_oss_triton_kernels_moe.py:1357-1380`), so DSV4 and dots3 never reach it.

### D9. Startup validation

When `TOKENMOE_TRACE_DIR` is set, the engine refuses to start if any of these holds:
- `--enable-return-routed-experts` is off;
- prefix caching is off;
- a speculative config is set;
- async scheduling is on;
- `data_parallel_size > 1`.

The check runs in `EngineCore.__init__` before the executor starts (D2). `VllmConfig.__post_init__` (`config/vllm.py:1254`) has already resolved every setting, async scheduling included, so a bad launch fails before any weights load.

Upstream already refuses PP > 1, DCP/PCP and KV connectors when capture is on (`config/vllm.py:1279-1310`).

### D10. Fork raw output (`raw_format_version = 1`)

Everything goes under `$TOKENMOE_TRACE_DIR/<engine_instance_id>/`, the directory exported to workers as `TOKENMOE_TRACE_ENGINE_DIR`:

- **`engine_meta.json`**
  - format version, engine_instance_id, vLLM version and commit;
  - the effective config:
    - parallelism: TP, EP, DP;
    - model: dtype, quantization, max_model_len;
    - memory and batching: gpu_memory_utilization, max_num_seqs, max_num_batched_tokens, block size;
    - scheduling: prefix caching, async scheduling, model runner version, scheduling policy;
    - capture flags;
  - clock anchors, boot_id, pid.
- **`layer_map.json`** (D7).
- **`steps.jsonl`**
  - step_index, the four timestamps, num_tokens_total, num_running, num_waiting, KV usage;
  - `entries`: a list of `[llm_request_id|null, vllm_request_id, phase, token_start, token_end]`.
- **`requests.jsonl`**
  - IDs and lifecycle times;
  - num_prompt_tokens, num_output_tokens, num_cached_tokens, num_preemptions;
  - engine finish status, seed, routing file name.
- **`routing/<llm_request_id>.npz`** (uncompressed)

  | Array | Shape and dtype |
  |---|---|
  | `token_ids` | `[T]` int32 |
  | `row_start` | scalar, = `num_cached_tokens` |
  | `experts` | `[R, L, K]`; uint8 when E ≤ 256, else uint16 (Q5) |
  | `row_end` | scalar, actual captured end; `T-1` for a complete generation |
  | `routing_complete` | bool; false for aborts/errors |
  | `token_positions` | `[R]` int32, strictly increasing absolute positions |
  | `step_index` | `[R]` int32 |
  | `layer_ids` | `[L]` |

**Runtime behaviour.**
- A writer thread does all file I/O. The engine loop only slices arrays, appends to in-memory buffers and enqueues.
- Requests without `tokenmoe_llm_request_id` still appear in step entries, because they are real batch members. They get no routing file.
- One run directory owns one engine lifetime: the run script starts and stops `vllm serve`.

### D11. Harness integration with mini-swe-agent 2.4.6

We drive `DefaultAgent` ourselves instead of using `mini-extra swebench`, with two classes of our own.

**`TracedModel`**
- It keeps mini-swe-agent's tool-calling action parsing and message formatting, but replaces the transport with `client.py`.
- `client.py` sends `POST /v1/chat/completions` with:
  - the messages;
  - the bash tool schema;
  - sampling parameters and a per-request seed;
  - `vllm_xargs = {"tokenmoe_llm_request_id": ...}` (`chat_completion/protocol.py:494, 716-758`).
- Assistant messages carry `reasoning`. vLLM passes it to templates as both `reasoning` and `reasoning_content` (`entrypoints/chat_utils.py:2013-2042`), so thinking survives across turns.
- It records `request_created_at`, `request_sent_at` and `response_received_at`.

**`TracedEnvironment`**
- It keeps mini-swe-agent's docker environment container lifecycle, with the runtime binary configurable (`podman` locally).
- It replaces `execute()` with the streaming executor (D12). It returns the same result shape and returns the same timeout error dictionary, so agent logic and observation templates stay unchanged.

**Other points.**
- The launcher passes the image name itself. It also starts, stops and removes one container per session.
- Why our own transport instead of litellm:
  - we need exact control over `vllm_xargs`, reasoning pass-back and timestamps;
  - how litellm handles `reasoning` for `hosted_vllm` is unverified.
- The exact override points (class and method names) are assumptions until checked against the installed 2.4.6 source (task 3.1).

### A1 source verification (2026-09-30)

Installed `mini-swe-agent==2.4.6` and the D19 dependencies into
`/home/youwei/bzh/venvs/tokenmoe-harness`; `pip check` passed. Source paths
below are relative to its `lib/python3.12/site-packages/minisweagent/`.

- `agents/default.py`: `DefaultAgent.run(task, **kwargs)` returns the exit
  message's `extra` dict. `query()` checks limits and increments `n_calls`;
  `execute_actions()` executes parsed action dicts **sequentially**, then calls
  `model.format_observation_messages`. Limits are `LimitsExceeded` and
  `TimeExceeded`; submission is `Submitted`, an `InterruptAgentFlow` subtype.
- `models/litellm_model.py`: transport is `_query()`, preparation is
  `_prepare_messages_for_api()`, parsing is `_parse_actions()`. Reusing
  `query()` would retain its own retry/cost logic, so the adapter must replace
  `query()` and transport while using the installed parser/formatter helpers.
- `models/utils/actions_toolcall.py`: `BASH_TOOL`,
  `parse_toolcall_actions` (expects tool-call objects with `.function` and `.id`),
  and `format_toolcall_observation_messages` are the reusable hooks. Action
  dicts contain `command` and the model's `tool_call_id`; preserve this API ID
  separately from the collector's globally unique `tc_` ID.
- `environments/docker.py`: `execute(action: dict, cwd="", *, timeout=None)`
  merges stderr into stdout with UTF-8 replacement and universal newlines.
  It catches `subprocess.TimeoutExpired` and other exceptions and returns
  `output`, `returncode=-1`, `exception_info`, plus `extra.exception_type` and
  `extra.exception`. It does **not** propagate TimeoutExpired. `_check_finished`
  raises Submitted for the completion sentinel. `config.executable` selects
  docker/podman. The stock `cleanup()` backgrounds a shell command, so a
  guaranteed-cleanup wrapper must wait for container removal.
- `config/benchmarks/swebench.yaml`: installed tool-calling benchmark config;
  cwd `/testbed`, timeout 60, interpreter `[bash, -c]`,
  `BASH_ENV=/root/.bashrc`. Its observation template keeps output under 10,000
  characters; otherwise it keeps the first/last 5,000 characters in separate
  `<output_head>` and `<output_tail>` blocks with intervening scaffold text.
- `run/benchmarks/swebench.py:get_swebench_docker_image_name`: explicit
  `image_name`, then `docker_image`, otherwise
  `docker.io/swebench/sweb.eval.x86_64.<instance_id with __ replaced by _1776_>:latest`,
  lowercased.

The user confirmed the installed timeout/interpreter contract and multi-span
provenance on 2026-09-30. The adapter uses Jinja string-conversion markers to
locate each payload slice and verifies that removing markers exactly restores
the installed formatter output; it does not alter the observation.

### D12. Streaming tool executor

- It runs `Popen([runtime, "exec", "-w", cwd, <container>, "bash", "-c", cmd], stdout=PIPE, stderr=STDOUT, bufsize=0)`. This is the same merged stream mini-swe-agent uses (C2).
- A `select` + `os.read(fd, 65536)` loop with a deadline handles output. Every read:
  - appends to `tool_outputs/<tool_call_id>.out`;
  - writes one chunk record: `chunk_index`, `chunk_time`, `byte_offset`, `byte_length`.
- On the deadline it kills the exec process, marks `timed_out`, and returns the installed environment's error result dictionary, carrying the partial output.
- `started_at` is taken just before `Popen`; `finished_at` after `wait()`.
- `output_bytes_raw` is the file size. `output_bytes_seen` is the byte length of the observation the agent actually receives.

### D13. Harness raw records

- Each harness process appends to its own jsonl shards under `raw/harness/<host>-<pid>/`.
- Request records store:
  - `messages_delta`: the messages appended since the session's previous request (Q2);
  - per-message provenance: `segment_type`, `source_tool_call_id`, and payload offsets inside the message content (for example, where the raw tool output sits inside the observation template);
  - the full assistant response: content, reasoning, tool_calls.
- Tools and sampling parameters are stored once, in the role template.
- If a message list is not an append-extension of the previous request's list, that request stores its full list.

### D14. IDs

- Runtime IDs have a typed prefix plus a time-ordered random suffix:
  - `run_`, `ses_`, `req_`, `tc_`, `eng_`;
  - `step_id = <engine_instance_id>:<step_index>`.
- Static IDs are content hashes (sha256 of canonical JSON, first 16 hex digits):
  - `experiment_config_id`, `engine_config_id`, `model_profile_id`;
  - `agent_template_id` + `version_hash`.

### D15. `finalize` and prompt segments

**Join and output.**
- `finalize` joins harness and engine requests on `llm_request_id`.
- It fills cross-producer fields: `issued_at`, `llm_request_ids_consuming`, `first_scheduled_at`, and so on.
- It writes the §6 layout:
  - jsonl for small tables;
  - parquet for `prompt_segments`, `tool_output_chunks`, `engine_steps` and `host_load`;
  - routing and tool output files hardlinked from `raw/`, copied only when a hardlink is impossible. `raw/` is kept, and near-uniform expert IDs compress poorly, so a recompressed copy would add space rather than save it;
  - `prompts/<llm_request_id>.prompt.txt`: the decoded engine prompt tokens, the same text that segment character ranges index into (Q3);
  - `prompts/<llm_request_id>.output.txt`: the decoded output tokens, reasoning and tool-call markup included. The parsed assistant message stays in the request record.
- It then runs the validator.

**Segments: decode-and-locate (Q1).**
1. The prompt text is `tokenizer.decode(engine prompt_token_ids, skip_special_tokens=False)`, the exact tokens the engine computed.
2. Per-token character offsets come from incremental decoding.
3. Messages are located in order with a cursor: exact match first, then a whitespace-stripped match.
4. Payload sub-spans come from the harness provenance offsets.
5. Text not claimed by any message becomes `other`: role headers, special tokens, rendered tool-call syntax.
6. A token that straddles a boundary goes to the segment holding its first character.

If a message cannot be located:
- its segment gets null ranges and a non-null `alignment_error`;
- its text stays inside an `other` segment, so rule 9's coverage and non-overlap still hold.

**Alternative.** Re-render each request offline with the model's renderer and compare tokens (C4b as first written). Rejected because:
- the four models use three renderers at v0.30.0: HF jinja, `vllm/renderers/deepseek_v4.py`, and harmony for GPT-OSS;
- no re-render can be more exact than the engine's own tokens.

**Pilot metric.** The share of message segments located. Below 99%, fall back to C4a: client-side rendering, then `/v1/completions` with token IDs.

### D16. Validator

- Applicable rules: 1–9, 11, 12; rules 6–9 in their revised form.
- Rules 10 (sub-agents), 13 (graph events) and 14 (multi-host) report `not_applicable`.
- Rule 12 checks for a passing equivalence record for the experiment config's `engine_config_id` (Q6).

### D17. Equivalence gate (rule 12)

**Setup.**
- 50 fixed first-step prompts: system prompt plus instance template for 50 fixed SWE-bench Verified instances.
- Sent sequentially (batch 1), greedy, `max_tokens` 256.
- Four runs with the same serve config: capture off, off, on, on.

**Pass criteria.**
- Token-sequence disagreement between on and off does not exceed the off/off floor, plus a tolerance of one prompt.
- Every captured row of a bound layer holds K distinct IDs in `[0, E)`.
- The two capture-on runs agree on routing over the positions where their tokens agree. This also checks capture under TP > 1.

**Output.** `static/equivalence/<engine_config_id>.json`.

### D18. Serve configuration

`configs/models/<model>.yaml` holds:
- model path, revision, TP;
- max_model_len, dtype and quantization;
- tool-call and reasoning parsers;
- sampling defaults and max_tokens;
- per-GPU-type overrides;
- facts used by the model profile.

**Common flags.**
- `--enable-prefix-caching --no-async-scheduling --enable-return-routed-experts --generation-config vllm`.
- Environment: `TOKENMOE_TRACE_DIR`.

**Qwen3-30B-A3B (local).**
- TP=2.
- YaRN factor 4 to 128K via `--hf-overrides`.
- `--reasoning-parser qwen3 --tool-call-parser hermes`.

**Hopper models.** Parser flags and TP sizes come from each model's vLLM recipe and are confirmed at smoke time (A6).

### D19. Environments

**Fork venv.**
- A new venv with an editable install of the fork branch under `VLLM_USE_PRECOMPILED=1`.
- Stack: torch 2.13, CUDA 13; needs NVIDIA driver ≥ 580 (the local driver is 595.71).
- The old venv stays, so the July baselines remain reproducible (Q7). It is an editable install of `vllm/` that loads the git-ignored compiled files in that tree, so it first moves to a preserved copy of the old tree (Migration Plan step 1).

**Harness venv.**
- Install command: `pip install -e collection[harness]`.
- Packages: mini-swe-agent==2.4.6, httpx, numpy, pyarrow, tokenizers, psutil, nvidia-ml-py, pyyaml.

### D20. Storage estimate

`estimate.py` implements the formulas below; the pilot replaces the assumptions with measured values.

**Per computed token.** Expert IDs take `L·K` bytes (uint8 when E ≤ 256). The int32 step indices and absolute token positions add 8 bytes per computed token. No scores are stored (D6).

**Planning assumptions.** They drive every number below:
- 60 steps per session;
- about 1.3K computed tokens per step, so C ≈ 78K per session;
- context grows from 3K to about 63K tokens, so the sum of `T` over all requests ≈ 2.0M tokens per session;
- one concurrency config at N=1, uncompressed, with `raw/` kept.

**Terms that do not depend on the model**, about 24 MB per session:
- These grow with the square of session length:
  - `token_ids` in the routing files: about 8 MB.
  - Prompt text, materialized at finalize: about 7 MB (Q3).
- Engine steps: about 6 MB of raw jsonl plus about 1 MB of parquet. At N=8, sessions share steps.
- Tool outputs: about 1 MB, stored once (hardlinked).
- Harness records: under 1 MB, with message deltas (Q2).

| Model | L × K | Expert IDs, per session | Total, per session | Total, 500 sessions |
|---|---|---|---|---|
| Qwen3-30B-A3B (debug) | 48 × 8 | 30 MB | 54 MB | 27 GB |
| GPT-OSS-120B | 36 × 4 | 11 MB | 35 MB | 18 GB |
| DeepSeek-V4-Flash | 43 × 6 | 20 MB | 44 MB | 22 GB |
| dots3-note | L × 8 (L from config) | 0.62·L MB | 0.62·L + 24 MB | 0.31·L + 12 GB |

Each model needs N=1 plus at least one multi-session config (§8.2), which roughly doubles these totals.

## Verified facts vs assumptions

**Verified in code at v0.30.0** (file:line as cited above):
- **Capture pipeline.**
  - Worker capture to host copy to per-step slice (D2, D3).
  - Logical IDs are captured before EPLB mapping.
- **Engine behaviour.**
  - MRv2 is the default, and routed-experts capture is not on its unsupported list.
  - Async scheduling is on by default.
  - The step loop, preemption, finish and shutdown paths (D2).
  - `vllm_xargs` reaches `SamplingParams.extra_args`.
  - The engine request ID carries a random suffix.
- **Chat and API.**
  - Assistant `reasoning` reaches templates.
  - API routing output is conditional on `routed_experts is not None`.
- **Kernels and models.**
  - GPT-OSS's Hopper kernel raises at bind (D8).
  - DSV4's hash layers and non-MegaMoE path reach `BaseRouter`.
  - dots3 inherits `DeepseekV2MoE`.
  - The capturer binds before the Scheduler is created.

**Assumptions that need a smoke test or source check:**

| # | Assumption | Checked by |
|---|---|---|
| A1 | mini-swe-agent 2.4.6 structure: `DefaultAgent`, the tool-calling model's query/parse split, the docker environment's `execute` and timeout exception, the SWE-bench config file, image naming. All of it comes from search snippets. | task 3.1 after [user] install |
| A2 | The triton_kernels bundled with v0.30.0 is v3.5.1, and there `routing()` == `topk()` + `routing_from_bitmatrix()`. Verified only on the copy from the currently installed (older) wheel. | re-read after install; GPU unit test on Hopper |
| A3 | DSV4-Flash and dots3 have a full-attention KV group, which capture init requires (`routed_experts_capturer.py:307-312`), and serve with prefix caching on. | Hopper smoke |
| A4 | Their default Hopper MoE backends are modular and go through `BaseRouter.select_experts`. | layer map shows the router path |
| A5 | Capture under TP > 1 gives identical IDs on all ranks. | D17 routing agreement |
| A6 | Parser flags and TP per model; GPT-OSS harmony chat with tools on v0.30.0. | Hopper smoke |
| A7 | Rootless podman: exec streaming, resource limits, and SWE-bench images all work. | local smoke |
| A8 | Recorder overhead is below 5%. | pilot |
| A9 | Decode-and-locate reaches at least 99%. | pilot |
| A10 | `--no-async-scheduling` with capture on serves all four models. | smoke |

## Risks / Trade-offs

- **[Recorder work inside the engine loop slows serving]** → The loop only slices and appends; the writer thread does all I/O. The pilot measures overhead against the 5% gate.
- **[In-flight routing buffers use host memory]** → A 30K-token Qwen3 prefill needs about 11.5 MB of IDs. The total is bounded by concurrency.
- **[An engine crash loses in-flight requests]** → The validator reports missing routing; the affected sessions are marked failed.
- **[With async scheduling off, throughput is below production]** → This is recorded in the engine config. Only timing realism is affected; routing is not.
- **[The server re-renders earlier tool calls differently from what the model emitted, breaking the prefix cache]** → `num_cached_tokens` records it. This is also what real deployments do.
- **[Upstream changes the capture design again]** → The tag is pinned. The AuxOutput redesign (vllm#45635) is not in v0.30.0. The recorder uses only the per-step slice and a few hooks, so a rebase stays small.
- **[mini-swe-agent internals differ from the search snippets]** → Task 3.1 verifies them before the adapter is written. All coupling sits in one module.
- **[Decode-and-locate misses content that a template rewrites]** → It tries stripped variants, the pilot measures the hit rate, and C4a is the fallback.
- **[`token_ids` storage grows with the square of session length]** → The README states it; deduplication can come later.
- **[The equivalence gate raises false alarms from run-to-run nondeterminism]** → The off/off floor absorbs it.

## Migration Plan

1. **[user] Preserve the old environment first.**
   - The old venv (torch 2.11) is an editable install of `vllm/`. It loads compiled files that sit, git-ignored, in that working tree: `vllm/*.so`, the `vllm/vllm_flash_attn/` build files, `vllm/third_party/{deep_gemm,triton_kernels}` and `vllm/_version.py` (987 files, about 540 MB).
   - Checking out v0.30.0 in `vllm/`, or installing the new venv from it, would break the old venv.
   - Before either, add a detached worktree of 90025dce2 outside the repo and copy those ignored files into it (commands in task 2.1). From then on, run the old venv with `PYTHONPATH` pointing at that worktree.
2. **Fork branch.**
   - Create branch `tokenmoe-v0.30.0-trace` in the submodule from tag v0.30.0 (already present locally).
   - Implement the fork edits and `tests/tokenmoe/`, then commit.
   - [user] Push the branch to origin.
   - Point the parent repo's submodule at the new commit.
   - The old branch `tokenmoe/router-score-capture` stays untouched.
3. **Parent repo.**
   - `git mv '#data_collect.md' collection/data_collect.md`, then apply the write-backs.
   - Delete the root `pyproject.toml` and the empty `tokenmoe/` and `docs/` directories.
   - Rewrite the root README as a short index.
   - Leave `#idea.md` and `datasets/` as they are.
4. **Environments.** [user] Create the new fork venv and the harness venv (D19).
5. **Rollback.**
   - Set the submodule pointer back to 90025dce2. The old venv keeps working from the preserved worktree.
   - Run data lives only in new directories, so nothing needs cleaning.

## Decision Log

Answered by the user on 2026-09-30:

| # | Decision |
|---|---|
| Q1 | Prompt segments use decode-and-locate on the engine's tokens (D15). If the pilot locates fewer than 99% of messages, fall back to C4a. |
| Q2 | The harness stores per-request `messages_delta` plus provenance (D13). |
| Q3 | `finalize` materializes `prompts/<id>.prompt.txt` and `prompts/<id>.output.txt` (D15), about 7 MB per session. |
| Q4 | No router scores; expert IDs only (D6). |
| Q5 | Expert IDs are uint8 when E ≤ 256, uint16 otherwise (D10). |
| Q6 | The equivalence record is keyed by `engine_config_id`, and rule 12 is reworded to match (D16, D17). |
| Q7 | A new venv for the v0.30.0 fork; the old venv stays (D19). |

### Q8. Partial routing on aborted/error requests (confirmed 2026-09-30 during implementation)

The user approved adding `row_end` and `routing_complete`. Normal completed generations retain `row_end = T-1`. Aborted/error requests keep only their actual captured interval `[row_start, row_end)` and set `routing_complete = false`; rules 6 and 7 validate that interval. A never-scheduled request has an empty interval starting at zero. An abort processed after model execution may have computed the final known token without appending a new sampled token, so its `row_end` may equal `T`. When an abort frees a request before `update_from_output`, defer recorder finalization until the pending step routing has been collected. The engine request snapshot must survive that free.

### Q9. Non-contiguous routing after a later cache hit (confirmed 2026-09-30)

The user approved `token_positions[R]`: each saved row explicitly names its
absolute token position. `row_start` remains the first-schedule cache hit and
`row_end` the captured end. Normal contiguous requests retain the old interval
invariant; a request can have fewer rows if a later, longer prefix-cache hit
skips positions it has never computed. See `scheduler.py:_get_local_prefix_cache_hit`
and the waiting/preempted request scheduling path (upstream lines 889–1006).

Phase classification must therefore track actual covered intervals, rather
than only hwm: previously captured positions and the initial cached prefix are
recompute; a previously skipped hole, when first executed, is new_prefill or
decode. The writer sorts rows by token position while retaining their original
step indices. Rules 6/7 join entries to token_positions. For a normal generation,
routing_complete means all actually computed rows were captured; cache holes
are not missing capture. Aborts/errors still set it false (Q8).

## Open Questions

None at design time. The remaining unknowns are assumptions A1–A10, each with the check that settles it.

### Implementation decisions confirmed 2026-09-30

- Q10: Keep mini-swe-agent 2.4.6 timeout-result dictionaries and the SWE-bench bash -c interpreter with BASH_ENV.
- Q11: Tool output provenance carries multiple payload ranges; a call may produce several tool_output segments with scaffold between them.
- Q12: llm_request_ids_consuming is a list, including every retry attempt that carries the observation.
- Local verification scope: use the existing Qwen3-30B-A3B weights (the user accepted this in place of Instruct-2507). Docker Hub timed out, so the user approved three local Python-container smoke tasks, explicitly labeled tokenmoe/local-smoke. This does not constitute SWE-bench scoring or a Hopper pilot.
- v0.30.0 treats YaRN max_position_embeddings as already scaled; the Qwen HF overrides therefore set it to 131072 alongside factor 4 and original_max_position_embeddings 32768.

- Q13: Local smoke containers use --network=none; the tasks do not require networking and rootless /dev/net/tun is unavailable.
- Q14: API suppression means no non-null routing payload; retain upstream routed_experts: null serialization.
