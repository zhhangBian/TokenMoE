# Expert residency experiments

CPU analysis, predictors, protocol planning and frozen-timeline diagnostics live here. GPU execution belongs to `../vllm`. Importing this package does not import torch, Triton or vLLM.

```bash
uv pip install --python /home/youwei/bzh/venvs/tokenmoe-harness/bin/python -e './collection[dev]' -e './optimization[dev]'
DATA=/home/youwei/bzh/dataset/tokenmoe_artifacts
PY=/home/youwei/bzh/venvs/tokenmoe-harness/bin/python
CUDA_VISIBLE_DEVICES=2,3 "$PY" -m tokenmoe_opt analyze \
  --run "$DATA/qwen3-runtime-20261019-063626/collection/N1/attempt-01/N1" \
  --run "$DATA/qwen3-runtime-20261019-063626/collection/N4/attempt-01/N4" \
  --label-version command-outcome-v1 --output "$DATA/NEW-screening-run"
"$PY" -m tokenmoe_opt plan --inputs "$DATA/NEW-screening-run/inputs.json" \
  --confirmation-split "$DATA/NEW-confirmation-split.json" \
  --output "$DATA/NEW-protocol-run"
```

Every output directory must be new. Use `shell-command-v2` in a separate run for conservative executable-word categories. Optional `--tune` tunes history mixture weights using training-only inner task folds; fixed screening uses 0.5. All N1/N4/pilot/recovery copies remain grouped by task. `plan` requires a split manifest with matching identity and nonempty untouched `groups` disjoint from exploration. Record the dataset hash and selection rule in its `selection` field; confirmation tasks stay outside fitting and policy selection.

For concurrent live evaluation, repeat `analyze --held-out-task TASK` for every task sharing the engine. This produces one joint held-out fold and excludes all copies of every selected task from its predictor and inner tuning. Using separate leave-one-task-out predictors for other sessions on the same shared controller would leak those other tasks into its prior. The default analysis keeps leave-one-task-out behavior.

Outputs include input SHA-256, read-only validation, model/weight/tokenizer/template identity, source hashes, split groups, fitted per-fold priors, feature/provenance records, per-request coverage, task bootstrap intervals, layer/transition subgroups, and tool increments against history+global and history+stage. Empty targets are null with a reason. `first32_decode` counts computed positions and does not measure TTFT. Hindsight uses target routes and cannot be deployed.

`tokenmoe-opt replay --demands demands.jsonl --cost-profile costs.json --capacity 32 --output NEW-DIR` accepts chronological `time_seconds`, `layer`, `tokens` (lists of top-k expert IDs), and optional `phase`. The cost manifest has `kind=cost-profile`, exact `identity`, and `copy` containing `bytes_per_expert`, `bandwidth_bytes_second`, and `setup_seconds`, with optional rank-specific expert sizes. Replay tracks finite slots, generations, in-flight copies, coalescing, LRU/LFU, shared-link bandwidth and capacity-fitting partitions of at most 64 tokens. An initial `snapshot` can supply complete slot and replacement history; otherwise initialization is explicitly cold. Unrecorded recompute is rejected. Speculative plans require `--timeline-kind neutral`; timestamps already delayed by misses cannot supply extra prefetch time.

`simulate --graph graph.json --cost-profile costs.json --capacity 64 --output NEW-DIR` advances model/tool/join/return dependency nodes through simulated completion. Only roots accept absolute release times; dependent nodes use dependencies and lags. Model service must be declared or reference a matching measured native-service geometry. Tools retain declared durations and fixed execution trajectories; deadline-induced cancellation or changed model outputs require separate modeling. Output remains an uncalibrated diagnostic. `calibrate --run HELD-OUT-RUN --cost-profile COSTS --protocol FROZEN-PROTOCOL-DIR --output NEW-DIR` compares separately fitted, source-frozen N1 fixed-input measurements against reconstructed warmup state, cache counts and service timing. The first held-out check reproduced all cache counts but failed the timing gate; it does not qualify simulated latency or complete agent workflows.

`plan` freezes the local N1/N4, slot/TP, sampling, KV/graph, warmup, repetition, numeric tolerance, outcome and latency-target protocol. Status is preregistered and unexecuted. Plans never mark backend, workload or migration milestones complete. Read [screening evidence](docs/screening-evidence.md) for reproduced results and retained diagnostics.

`tokenmoe-opt arrivals --inputs ANALYSIS/inputs.json --output NEW-DIR` evaluates task-held-out empirical survival on validated historical tools. The local 1,021-tool diagnostic produced 1,973 elapsed-time decisions and 74.52% task-macro coverage for nominal 80% intervals, with 1,955 finite uncensored intervals. The report retains unknown tails and calibration errors; this is not live timing-benefit evidence.

The optional collection flag `control_events: true` exports bounded live control records without changing agent prompts or tool observations. `SessionState.receive` records controller receipt; gaps suppress predictions until an observed snapshot, duplicate events are idempotent, terminal sessions remain terminal, and sequential/parallel joins require every dependency. No harness module imports vLLM. The cost-aware planner accepts explicit model-specific distributions, token volumes, arrival/service probabilities and actual scheduler/session states, with atomic per-rank and shared-link reservations.

The default-off Python cache and diagnostic commands are described in [backend evidence](docs/backend-evidence.md). `fit-costs` requires a matching native/cache pair, complete detailed events and exact model/device identities. `control` requires a frozen predictor with training-task provenance, validated duration-training sources, measured compatible costs, client events, engine scheduling/readiness events, compact feedback and fresh worker snapshots. It rejects training-task sessions. Worker delivery uses private local datagram mailboxes with bounded plans and five-second leases; demand always takes priority. `diagnostic-metrics` reports actual token observation points, transfers and sampled resources, keeping missing or partial fields explicit. These commands do not establish a controlled policy gain or calibrated timing simulation.

The experimental `prefill_execution=expert-major` mode is default-off and requires its own correctness and cost records. It groups actual current-batch token/expert pairs, keeps every contribution and its coefficient, and reduces at the original top-k positions. Batches above 64 tokens use bounded pair chunks; smaller batches and decode retain the token executor. At a 4,096-token scheduler limit, explicitly reserve 192 MiB scratch for original BF16 route outputs plus execution workspace. `qualify_qwen.py --cache --prefill-execution expert-major` accounts for that reservation. Cost profiles and live snapshots identify the execution mode and maximum pair chunk; controllers reject mismatches, with legacy profiles interpreted as token execution.

`optimization/scripts/qualify_qwen.py` supports fixed warmup and repetition counts plus native graph, UVA and layer-prefetch references. Keep eight-token diagnostics separate from the 4096-token agent protocol. `optimization/scripts/summarize_qualification.py --run NATIVE --run CACHE --output NEW-DIR` validates matching runtime sources, devices, KV/scheduler settings, warmup/repetitions and greedy outputs before writing a short-cost comparison. It labels aggregate cache counters as whole-engine measurements, including warmup. The current cache's short-trial costs exceed native and UVA; the evidence document retains these negative results.

`optimization/scripts/run_agent_pilot.py` reuses the real collection launcher with explicit `optimization-agent-v1` metadata-only provenance, fixed warmup, measured boundaries and raw task outcomes. It accepts exploration tasks only and rejects overlap with predictor training groups. `agent-metrics --run PILOT --output NEW-DIR` extracts actual engine/client observations, failed outcomes, windowed transfer/resource data and whole-task uncertainty. These runs cannot be finalized as full routing datasets. Read [agent pilot evidence](docs/agent-pilot-evidence.md) for the real N1 run, its failed task outcome and the shutdown fault being repaired before controlled comparisons.

`compare-agents --trial PILOT METRICS-DIR --trial OTHER-PILOT OTHER-METRICS-DIR --output NEW-DIR` writes descriptive cells and checks sampling/KV/graph, devices, tasks, runtime sources, frozen protocols, complete exports and repetition coverage. It verifies that trial metadata still matches the metric extraction hashes. It does not infer a rotated matrix from a collection of pilot directories or publish a paired speed ratio; those comparisons remain unqualified until the complete matrix is verified.

```bash
CUDA_VISIBLE_DEVICES=2,3 "$PY" -m pytest optimization/tests collection/tests -q
"$PY" -m ruff check optimization collection/tokenmoe_collect/validate.py collection/tests/test_pipeline.py
openspec validate add-agent-aware-expert-residency --strict
```
