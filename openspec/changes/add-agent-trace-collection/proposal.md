## Why

`#idea.md` needs agent traces that join four things by shared IDs on one clock:
- the routed experts of every computed token;
- the engine step that computed each token;
- the tool call that put each piece of prompt content there;
- that tool's timed output stream.

Nothing in the repo can collect them today:
- Commit 6bcc6b6 removed all collection code.
- The vLLM fork base (≈ v0.22.0) cannot serve dots3-note.
- On GPT-OSS-120B's default MXFP4 kernel, routed-experts capture fails at startup.

The pipeline therefore has to be rebuilt on vLLM v0.30.0 before any Stage A analysis can run.

## What Changes

- **New `collection/` directory.** It is self-contained and holds the collection package, configs, scripts, tests, README and the collection spec. No collection code or docs stay at the repo root.
- **vLLM fork: trace recorder.** A new branch from upstream tag v0.30.0 adds a Python-only recorder, enabled by `TOKENMOE_TRACE_DIR` plus `--enable-return-routed-experts`. It writes:
  - one record per engine step: timestamps, plus per-request entries tagged `recompute`, `new_prefill` or `decode`;
  - one record per request: lifecycle times, cache and preemption counts;
  - one routing array per request: expert IDs;
  - a map of the captured MoE layers.

  At startup it refuses any engine configuration that would make these records wrong:
  - prefix caching must be on;
  - speculative decoding must be off;
  - async scheduling must be off;
  - data parallelism must be 1.
- **vLLM fork: GPT-OSS capture fix.** Routed-experts capture currently raises at bind on GPT-OSS's monolithic Triton MXFP4 kernel. The fix makes that kernel capture.
- **vLLM fork: API suppression.** While the recorder is active, the fork stops assembling routing into API responses.
- **Session runner** for mini-swe-agent 2.4.6 × SWE-bench Verified:
  - a model class that tags each call with its `llm_request_id` and passes reasoning back to the server;
  - a streaming tool executor around `docker exec` / `podman exec` that tees raw output with chunk timestamps;
  - a 1 Hz host-load sampler;
  - run configs for N concurrent sessions.
- **Offline tooling:**
  - static records: experiment config, model profile, role templates, benchmark items;
  - prompt-segment alignment;
  - `finalize`, which turns raw producer files into the spec's storage layout;
  - a validator for the spec's rules, limited to single-loop agents.
- **Per-model serve configs and scripts:**
  - Qwen3-30B-A3B for local A100 debugging: hybrid thinking, YaRN to 128K.
  - DeepSeek-V4-Flash, dots3-note and GPT-OSS-120B for Hopper.
  - Smoke, capture on/off equivalence and pilot scripts for every model.
- **Repo cleanup:**
  - delete the stale root `pyproject.toml` and the empty `tokenmoe/` and `docs/`;
  - rewrite the root README as a short index.
- **BREAKING (collection spec):**
  - The spec moves: `#data_collect.md` → `collection/data_collect.md`.
  - Routing arrays hold only tokens that the request itself computed: rows cover `[num_cached_tokens, T-1)`. Cached-prefix routing is no longer re-exposed.
  - Request metadata changes from `extra_body.tokenmoe` to a single key, `vllm_xargs.tokenmoe_llm_request_id`. Session, template and step metadata stay on the harness side and are joined offline.
  - Each tool call gets one merged raw output file instead of separate `stdout` and `stderr`.
  - Expert IDs are stored as uint8 when a model has at most 256 routed experts (uint16 otherwise).
  - Rule 12's equivalence record is keyed by the engine configuration instead of the experiment configuration.
  - Router scores are no longer collected: `scores` (§4.8), `capture_router_scores` (§3.1) and `router_score_semantics` (§3.2) are removed.
- **BREAKING (fork):** the July score-capture patch on 0b3ba88f1 is not carried over, and router scores are no longer captured.

## Capabilities

### New Capabilities
- `engine-trace-recorder`: fork-side recording of engine steps, request lifecycle and per-request routing (expert IDs). Also covers:
  - enablement and startup validation;
  - the captured-layer map;
  - capture coverage for the MoE kernels the four models use;
  - suppression of routing in API responses;
  - the capture on/off equivalence gate.
- `agent-session-runner`: running mini-swe-agent sessions on SWE-bench Verified against the fork:
  - request-ID generation and propagation;
  - reasoning pass-back;
  - the streaming tool executor;
  - host-load sampling;
  - harness-side lifecycle records;
  - run configs: model, concurrency, instances, seeds.
- `trace-dataset`:
  - static records;
  - offline prompt segments;
  - `finalize` into the storage layout;
  - validation;
  - the storage estimate;
  - keeping `collection/data_collect.md` consistent with what the code emits.

### Modified Capabilities
None. `openspec/specs/` is empty.

## Impact

**`collection/data_collect.md` sections touched** (all edits in Chinese, like the rest of the spec):

| Section | Change |
|---|---|
| §1 item 2 | Routing covers computed tokens only. |
| §1 item 7 | Routing arrays hold expert IDs only. |
| §3.1, §3.2 | Remove `capture_router_scores` and `router_score_semantics`. |
| §4.3 | Engine-side fields `vllm_request_id` and `num_preemptions`; defines `num_cached_tokens` as the count at first schedule. |
| §4.6 | One merged output stream per tool call. |
| §4.7 | Adds the `recompute` phase and the step timestamps. |
| §4.8 | Row range, `step_ids` semantics, where `model_profile_id` lives, expert ID dtype; `scores` removed. |
| §6 | `tool_outputs/` layout; storage estimate without scores. |
| §7 | Rules 6 (score check removed), 7, 8 and 12. |
| §8 | First-batch order: mini-SWE-agent now, other harnesses later. |
| §8.1 | Instrumentation points: `vllm_xargs`, recorder raw files, reasoning pass-back. |
| §9 | Model table: Qwen3 is local-debug only; per-model capture path; GPT-OSS score note removed. |

**vLLM fork (`vllm/` submodule).** New branch `tokenmoe/v0.30.0-trace` from tag v0.30.0; the submodule pointer in this repo moves to it.
- New module: `vllm/tokenmoe_trace.py`.
- Edited files:
  - `vllm/v1/engine/core.py`
  - `vllm/v1/core/sched/scheduler.py`
  - `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`
  - `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`
- New tests: `tests/tokenmoe/`.

**Dependencies.**
- Fork venv: vLLM v0.30.0 precompiled wheel (torch 2.13, CUDA 13, NVIDIA driver ≥ 580).
- Harness venv: mini-swe-agent 2.4.6 (brings litellm), numpy, pyarrow, tokenizers, psutil, nvidia-ml-py, pyyaml.
- Benchmark: SWE-bench Verified, fetched with the user's download script.
- Images: SWE-bench instance images from Docker Hub; rootless podman on the local host.

**Existing environment.** The old venv is an editable install of `vllm/`. Before the fork branch is checked out, it moves to a preserved worktree of 90025dce2 (design.md Migration Plan step 1).

**Data.** New run directories under `/home/youwei/bzh/dataset/tokenmoe_artifacts/`. Existing artifacts, including the 256-record v2 baselines, are not touched.

**Removed.** Root `pyproject.toml`, empty `tokenmoe/` and `docs/`, and the stale README content.
