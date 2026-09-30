## 1. Repo restructure and spec write-back

- [ ] 1.1 Create `collection/`, then `git mv '#data_collect.md' collection/data_collect.md`
- [ ] 1.2 Delete the root `pyproject.toml` and the empty `tokenmoe/` and `docs/` directories. Rewrite the root README as a short index that points to `#idea.md` and `collection/`
- [ ] 1.3 Write C1–C7 and the decisions in design.md's Decision Log back into `collection/data_collect.md` (in Chinese), in these sections: §1 items 2 and 7, §3.1, §3.2, §4.3, §4.4, §4.6, §4.7, §4.8, §6, §7 rules 6–8 and 12, §8, §8.1, §9. Remove router scores everywhere: `capture_router_scores` (§3.1), `router_score_semantics` (§3.2), `scores` (§4.8), the score terms of the §6 estimate, the score check of rule 6, and the GPT-OSS score note (§9)
- [ ] 1.4 Update `openspec/config.yaml` to the final paths, and run `openspec validate add-agent-trace-collection --strict`

## 2. vLLM fork: trace recorder on v0.30.0

- [ ] 2.1 [user] Preserve the old environment before `vllm/` changes (design.md Migration Plan step 1):
  - `git -C /home/youwei/bzh/project/TokenMoE/vllm worktree add --detach /home/youwei/bzh/project/TokenMoE-vllm-0722 90025dce2`
  - `cd /home/youwei/bzh/project/TokenMoE/vllm && git ls-files -z -o -i --exclude-standard -- vllm/ | grep -zv __pycache__ | xargs -0 cp --parents -t /home/youwei/bzh/project/TokenMoE-vllm-0722/`
  - From then on, run the old venv with `PYTHONPATH=/home/youwei/bzh/project/TokenMoE-vllm-0722`
- [ ] 2.2 In the submodule, create and check out branch `tokenmoe/v0.30.0-trace` from tag v0.30.0 (ced6857afa0e). Leave `tokenmoe/router-score-capture` untouched
- [ ] 2.3 Add `vllm/tokenmoe_trace.py`:
  - enablement from the environment, and startup validation (D9)
  - `engine_meta.json`
  - the writer thread
  - phase classification (D4)
  - pending step entries
  - per-request routing accumulation
  - npz writing (D10)
- [ ] 2.4 EngineCore hooks in `vllm/v1/engine/core.py`:
  - before the executor is created (108-134): validation, `engine_instance_id`, the trace directory, and the export of `TOKENMOE_TRACE_ENGINE_DIR`
  - step timestamps in `step()` (589-628), passed to the recorder
- [ ] 2.5 Scheduler hooks in `vllm/v1/core/sched/scheduler.py`: init (387-405), `add_request` (2470), `_update_after_schedule` (1520-1533), routing slice (1923-1937), first token (2025-2031), `_preempt_request` (1477-1518), `_free_request` (2561), `shutdown` (2817)
- [ ] 2.6 Record `layer_map.json` from `bind_routed_experts_capturer` (`routed_experts_capturer.py:251-304`), written by TP rank 0 into `TOKENMOE_TRACE_ENGINE_DIR`. Restrict routing arrays to bound layers (D7)
- [ ] 2.7 API suppression: skip routed-experts assembly in `update_from_output` (2067-2105) while the recorder is active
- [ ] 2.8 GPT-OSS monolithic Triton MXFP4 capture in `gpt_oss_triton_kernels_moe.py` (D8):
  - split the legacy `routing()` call into `topk` + `routing_from_bitmatrix` when a capture function is set, and capture `expt_indx`
  - capture `topk_result.indx` on v3.6+
  - `supports_routing_replay_capture() = True`
- [ ] 2.9 CPU tests in `tests/tokenmoe/`:
  - phase classification, including preemption and mixed steps
  - step-slice attribution
  - npz shape and dtype
  - startup validation
  - API suppression
- [ ] 2.10 [user] Push branch `tokenmoe/v0.30.0-trace` to origin. Then update the submodule pointer in the parent repo

## 3. Harness package core (`collection/tokenmoe_collect`)

- [ ] 3.1 [user] Create the harness venv. Run `pip install mini-swe-agent==2.4.6` plus the extras from D19. Then verify, against the installed source (A1), the exact override points, and record the findings in `design.md`:
  - `DefaultAgent`
  - the tool-calling model's query/parse split
  - `execute` and the timeout exception in the docker environment
  - the SWE-bench config path
  - image naming
- [ ] 3.2 Add `collection/pyproject.toml` (package `tokenmoe_collect`, extras `[harness]`, `[dev]`) and `ids.py`, `clock.py`, `records.py`, `jsonl.py`, with unit tests
- [ ] 3.3 `client.py`: chat-completions transport with `vllm_xargs`, reasoning pass-back, seeds, timestamps, and retries with new attempt ids. Test it against a fake HTTP server
- [ ] 3.4 `executor.py`: merged-pipe streaming tee, chunk records, timeout kill with partial output, and a pluggable runtime (`docker`/`podman`/local subprocess for tests). Test with a local subprocess
- [ ] 3.5 `hostload.py`: 1 Hz sampler with null GPU columns when NVML is unavailable. Test the sampler loop with fake sources
- [ ] 3.6 `minisweagent_adapter.py`:
  - `TracedModel` and `TracedEnvironment` on the override points from 3.1
  - message provenance and deltas (D13)
  - `output_bytes_seen`
- [ ] 3.7 `static.py`: experiment config, model profile from `engine_meta.json` + `layer_map.json` + model yaml, role template with content-hash `version_hash`, and benchmark items from SWE-bench Verified
- [ ] 3.8 `launcher.py` + `cli.py run`:
  - run-config loading, and refusal to reuse an output dir
  - a container per session with guaranteed cleanup
  - the concurrency bound
  - application-run and session records
- [ ] 3.9 End-to-end CPU test: a fake chat server scripted to issue bash tool calls, plus a local-subprocess runtime, and one session through `run`. It checks every harness record

## 4. Offline tooling

- [ ] 4.1 `finalize.py`: join on `llm_request_id`, fill cross-producer fields, emit jsonl/parquet, hardlink routing and tool output files from `raw/` (copy only across filesystems), write `prompts/<id>.{prompt,output}.txt` from the decoded engine tokens, keep `raw/`
- [ ] 4.2 `align.py`: decode-and-locate prompt segments with payload splitting, `other` gap filling and `alignment_error` (D15). Unit tests with a small real tokenizer from the local model directory
- [ ] 4.3 `validate.py`: rules 1–9, 11 and 12 (revised), `not_applicable` for 10/13/14, and `validation.json` with examples. Build a synthetic run fixture that violates each rule once
- [ ] 4.4 `equivalence.py` + `cli.py equiv`: 50 fixed prompts, off/off/on/on protocol, pass criteria and record file (D17)
- [ ] 4.5 `estimate.py` + `cli.py estimate`: formula mode and measured mode (D20)

## 5. Configs, scripts and docs

- [ ] 5.1 `configs/models/qwen3-30b-a3b.yaml` for local debug: TP=2, YaRN ×4 to 128K, qwen3 reasoning parser, hermes tool parser, sampling
- [ ] 5.2 `configs/models/{gpt-oss-120b,deepseek-v4-flash,dots3-note}.yaml` for Hopper. Parsers and TP follow the vLLM recipes, marked "confirm at smoke" (A6)
- [ ] 5.3 `configs/runs/{local-debug,pilot,full}.yaml`: instances, concurrency, seeds, limits
- [ ] 5.4 `scripts/serve.sh`, `smoke.sh`, `equivalence.sh`, `pilot.sh`, `pull_images.sh`. Each prints the exact commands it runs
- [ ] 5.5 `collection/README.md`: venv setup, the command sequence per model, output layout and storage estimate table, and the list of `[user]` steps

## 6. Local debug on A100 (Qwen3-30B-A3B)

- [ ] 6.1 [user] Create the fork venv and install the fork branch with `VLLM_USE_PRECOMPILED=1`, then run `pytest tests/tokenmoe` in the fork
- [ ] 6.2 [user] Add `princeton-nlp/SWE-bench_Verified` to `DATASET_LIST` in `download_dataset.py` and run it. Pull the images for the local-debug instances with podman
- [ ] 6.3 [user] Run `scripts/smoke.sh qwen3-30b-a3b`. Check `layer_map.json` (48 router-path layers), one routed request, and the step records
- [ ] 6.4 [user] Run `scripts/equivalence.sh qwen3-30b-a3b` (TP=2)
- [ ] 6.5 [user] Run `configs/runs/local-debug.yaml` on 3 instances, then finalize and validate. Report the outputs back for review

## 7. Hopper pilot

- [ ] 7.1 [user] Set up the fork and harness venvs on the Hopper host, download model weights, and pull the pilot images
- [ ] 7.2 [user] Run smoke and equivalence for GPT-OSS-120B, DeepSeek-V4-Flash and dots3-note. Check the capture path of each model in `layer_map.json` (A3–A6, A10)
- [ ] 7.3 [user] Run the pilot: 20 repo-stratified instances at N=1 and N=4 per model. Then finalize, validate and estimate
- [ ] 7.4 Review the pilot against the acceptance criteria:
  - all applicable rules pass, and the equivalence gate passes
  - ≥ 90% of sessions end normally
  - recorder overhead < 5%
  - tee output matches what the agent received before truncation
  - segment location ≥ 99%
  - measured storage matches the estimate

  Record the results and fix the full-run parameters in `configs/runs/full.yaml`
