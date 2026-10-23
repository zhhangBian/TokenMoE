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
  --output "$DATA/NEW-protocol-run"
```

Every output directory must be new. Use `shell-command-v2` in a separate run for conservative executable-word categories. Optional `--tune` tunes history mixture weights using training-only inner task folds; fixed screening uses 0.5. All N1/N4/pilot/recovery copies remain grouped by task. Current local tasks are exploratory; untouched confirmation groups remain outstanding.

Outputs include input SHA-256, read-only validation, model/weight/tokenizer/template identity, source hashes, split groups, fitted per-fold priors, feature/provenance records, per-request coverage, task bootstrap intervals, layer/transition subgroups, and tool increments against history+global and history+stage. Empty targets are null with a reason. `first32_decode` counts computed positions and does not measure TTFT. Hindsight uses target routes and cannot be deployed.

`tokenmoe-opt replay --demands demands.jsonl --cost-profile costs.json --capacity 32 --output NEW-DIR` accepts chronological `time_seconds`, `layer`, `tokens` (lists of top-k expert IDs), and optional `phase`. The v1 cost manifest has `kind=cost-profile`, exact `identity`, and `copy` containing `bytes_per_expert`, `bandwidth_bytes_second`, and `setup_seconds`. This is a serialized frozen-timeline diagnostic with capacity-fitting partitions and explicit rejection of unrecorded recompute. It lacks held-out live calibration and propagated model/tool timing; it cannot establish latency improvement. Historical traces have no measured layer timing offsets.

`plan` freezes the local N1/N4, slot/TP, sampling, KV/graph, warmup, repetition, numeric tolerance, outcome and latency-target protocol. Status is preregistered and unexecuted. Plans never mark backend, workload or migration milestones complete. Read [screening evidence](docs/screening-evidence.md) for reproduced results and retained diagnostics.

`tokenmoe-opt arrivals --inputs ANALYSIS/inputs.json --output NEW-DIR` evaluates task-held-out empirical survival on validated historical tools. The local 1,021-tool diagnostic produced 1,973 elapsed-time decisions and 74.52% task-macro coverage for nominal 80% intervals, with 1,955 finite uncensored intervals. The report retains unknown tails and calibration errors; this is not live timing-benefit evidence.

The optional collection flag `control_events: true` exports bounded live control records without changing agent prompts or tool observations. `SessionState.receive` records controller receipt; gaps suppress predictions until an observed snapshot, duplicate events are idempotent, terminal sessions remain terminal, and sequential/parallel joins require every dependency. No harness module imports vLLM. The cost-aware planner accepts explicit model-specific distributions, token volumes, arrival/service probabilities and actual scheduler/session states, with atomic per-rank and shared-link reservations.

The default-off Python cache and diagnostic commands are described in [backend evidence](docs/backend-evidence.md). `fit-costs` requires a matching native/cache pair, complete detailed events and exact model/device identities. `control` requires a frozen predictor with training-task provenance, validated duration-training sources, measured compatible costs, client events, engine scheduling/readiness events, compact feedback and fresh worker snapshots. It rejects training-task sessions. Worker delivery uses private local datagram mailboxes with bounded plans and five-second leases; demand always takes priority. `diagnostic-metrics` reports actual token observation points, transfers and sampled resources, keeping missing or partial fields explicit. These commands do not establish a controlled policy gain or calibrated timing simulation.

```bash
CUDA_VISIBLE_DEVICES=2,3 "$PY" -m pytest optimization/tests collection/tests -q
"$PY" -m ruff check optimization collection/tokenmoe_collect/validate.py collection/tests/test_pipeline.py
openspec validate add-agent-aware-expert-residency --strict
```
