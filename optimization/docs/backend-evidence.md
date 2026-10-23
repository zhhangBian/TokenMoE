# Qwen exact demand cache: local evidence

The Python-only backend supports BF16 Qwen3-30B-A3B, eager Triton, TP1/TP2,
EP1/DP1/PP1, synchronous scheduling, and fixed per-layer slots. It is off by
default. Unsupported layouts, graphs, quantization, overlapping weight offload,
and insufficient expert budgets fail before checkpoint loading. Dense weights
remain on GPU; routed expert shards load into CPU storage with one pinned
expert staging buffer per layer. Demand recovery preserves router IDs and
coefficients through capacity-fitting token microbatches.

All artifacts below are under
`/home/youwei/bzh/dataset/tokenmoe_artifacts`; physical GPUs 2 and 3 are A100 80GB
devices ending in `9018` and `6973`. The resource monitor records complete UUIDs,
initialization and steady phases, process identities, and sampled physical memory.
Samples are observations at 100 ms intervals, not guaranteed instantaneous maxima.

| Check | Artifact | Observed result |
|---|---|---|
| Real layer-0 forced tensors, TP1 and both TP2 shards | `qwen-expert-tensors-20261022-v1` | 32 inputs, 97 unique experts, slot capacity 16; max absolute error 0 |
| Full-model matching tensors and routing | `qwen-cache-probe-tp2-20261022-v2` paired with `qwen-native-probe-tp2-20261022-v2` | 288 files across 48 layers, both ranks, prefill and two decode frames; inputs, coefficients and outputs equal; no routing or greedy-token mismatch |
| Idle worker delivery and expiry | same cache run | Real pytest tool, copy starts on both ranks before next request; follow-up greedy outputs equal |
| Delayed TP shard | `qwen-cache-stress-tp2-20261022-v1` | Rank 1 delayed 48 demand loads by 50 ms each; greedy outputs equal |
| Cancellation and subsequent request | same stress run | Follow-up greedy outputs equal |
| Peer exit during actual demand | same stress run | Expected `EngineDeadError`; all workers released in 11.019 s, watchdog unused |
| Native N4 reference | `qwen-native-n4-tp2-20261022-v1` | First two trial token sequences equal to the N4 cache stress run |
| Frozen predictor → controller → idle worker copies | `qwen-cache-live-control-tp2-20261023-v2` | Both ranks start copies during real pytest; terminal session withdraws pending plans; source stability passes |

The live-controller fixture declares its session synthetic and has no agent task
outcome. The earlier `...-20261023-v1` is retained as failed evidence: initial
residency age was first recorded when a prediction appeared, delaying admission
by two control cycles; copy starts followed the short tool's end by about 19 ms.
Residency age now advances during idle observations without predicted work.
The successful run also uses the complete real optimization test suite as its
tool. Neither run qualifies long-tool workload coverage.

The intentional peer-failure log includes a Python resource-tracker warning about
three shared-memory objects cleaned at shutdown. This fault test establishes a
bounded failure path, not a general leak-free claim.

With equal 1 GiB KV allocation, the full-model detailed-probe pair recorded
sampled physical GPU peaks of 34,441,658,368 / 34,401,812,480 bytes for native
residency and 9,074,507,776 / 9,034,661,888 bytes for 16-slot demand caching.
The cache process-tree RSS peak was 68,232,687,616 bytes; summed process RSS can
count shared pages repeatedly. Cache rank logs each contain 162,971 events,
zero reported drops or writer errors, and 75,601,281,024 started/completed bytes.
Both complete lifecycle reconstructions passed.

`qwen-cache-integrity-20261023-v2` retains a separate current-validator run for
both rank lifecycles and all held-out request-feature provenance. Missing memory
fields, layout mismatches and scratch-inclusive rank budget violations are
errors; incomplete export and compact-only observation cannot qualify exact replay.

Memory savings do not establish a speedup. The paired second short trials took
8.340 s with caching and 0.624 s with native residency, with detailed observation
and tensor probes enabled. Startup took 274.729 s and 28.848 s respectively.
These diagnostic runs have two short repetitions, no agent task outcomes, and
no native graph reference; they cannot support controlled performance claims.

`qwen-cache-costs-20261022-v1/cost-profile.json` fits actual demand copies and
host eviction-to-readiness recovery from that pair. The rank-conservative shared
bandwidth is 21.374 GB/s, device-copy p90 is 0.373 ms, and host recovery p90 is
1.963 ms. Short-prompt prefill/decode service medians remain diagnostics.
Propagated timing simulation and held-out calibration remain outstanding.

For local reproduction, use the fork venv and a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 \
  /home/youwei/bzh/venvs/tokenmoe-vllm030/bin/python \
  optimization/scripts/qualify_qwen.py --tp 2 --cache --probe \
  --observation detailed --control-directory NEW-DIR/mailboxes --output NEW-DIR
```

`--stress` instead of `--probe` exercises delayed shards, cancellation and
intentional worker failure. It must run as a separate diagnostic. Output
directories contain source patches (including untracked files), runtime hashes,
package freeze, exact config, resources, criteria, logs and failure status.
`source_stable` must pass; the script snapshots every recorded Python module.

Fork revision `3a8651074d7be272f0c147744fe24f0f21c8e883` installs successfully
with `VLLM_USE_PRECOMPILED=1`, pinned native-wheel base
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`, variant `cu130`, and `--no-deps`.
The installed version is `0.30.1.dev2+g3a8651074.precompiled`. Fork pre-commit
checks and manual mypy 3.12 pass. The test lockfile's Transformers 5.16.1 is
required by the existing scheduler fixtures; 5.17.0 changes a Pixtral symbol.

The follow-up fork fix `3a244b074b` rechecks plan epochs under the cache lock so
TP fallback also stops queued speculative admission. Eleven cache regressions,
pre-commit and mypy 3.12 pass. The same precompiled install procedure was repeated
successfully, producing `0.30.1.dev3+g3a244b074.precompiled`.

The CPU planner caches layer victims and stops fresh candidate admission when
the in-flight ledger is full, retaining fully coalesced transfers. A regression
checks that joining existing work still succeeds under an exhausted ledger.
Idle predictions retain their original bounded lease during active generation;
they are not repeatedly renewed without a fresh idle prediction. Terminal or
unreliable sessions withdraw pending plans immediately; existing DMA stays tracked.

`qwen-residency-confirmation-20261023-v1/split.json` freezes ten unused task
groups, one from each repository with remaining tasks, selected by a fixed
SHA-256 rule using task IDs only. Dataset hash, excluded exploration groups and
the rule are retained. Protocol v3 requires nonempty disjoint confirmation groups
and declares a separate native/expert/KV/both contention experiment with KV
arbitration unavailable. These confirmation and contention studies are unexecuted.

Controlled agent comparisons, long-tool/multi-role coverage, and larger-model
migration have independent outstanding gates. The formal agent protocol uses
4096 generated tokens, engine TTFT p95 at most 1 s, and required-tool completion
to fully received batched response p95 at most 30 s. The 8/64-token diagnostics
cannot be substituted for that protocol.

## Repeated short cost reference (2026-10-23)

`qwen-backend-cost-comparison-20261023-v1` checks matching KV, scheduling,
input/output tokens, GPU UUIDs and fork runtime hashes for seven source-stable
runs. Each uses two warmup batches (four requests), then three measured batches
of two prompts generating eight tokens each. The output directory retains the
comparison JSON and its reproducible summary script. Engine logs are archived
as `engine.log` beside each source result. These remain backend diagnostics;
they have no agent task outcomes or task uncertainty.

| Backend | Median batch seconds | Sampled GPU peak bytes, rank 0 / rank 1 |
|---|---:|---:|
| Native eager | 0.6963 | 34,441,658,368 / 34,401,812,480 |
| Cache, observation off | 7.3723 | 9,074,507,776 / 9,034,661,888 |
| Cache, compact feedback | 9.0165 | 9,074,507,776 / 9,034,661,888 |
| Cache, detailed events | 7.4366 | 9,074,507,776 / 9,034,661,888 |
| Existing UVA, 24 GiB/rank expert offload | 2.6824 | 8,738,963,456 / 8,699,117,568 |
| Existing layer prefetch, 7/8 expert layers | 18.0973 | 9,810,608,128 / 9,770,762,240 |
| Native graph reference | 0.1049 | 34,533,933,056 / 34,494,087,168 |

All three measured outputs in every arm equal the native reference. Native
graph v2 uses the same runtime source hashes as the other six runs; graph v1 is
retained separately. Mode order was sequential, so observer timing differences
also include host-load/order variation. They do not establish a causal claim
that compact feedback is intrinsically more expensive than detailed export.

The cache-off-observation run counted 118,063,890,432 transferred bytes per rank
across the whole engine lifetime, including warmup/profiling. Rank 0 recorded
7.319 s of readiness waits, 1.683 s of router-to-host synchronization, and
7.141 s of device copies. Those clocks can overlap; they must not be summed or
attributed to a measured repetition. Current caching is slower than existing
UVA at this short workload and memory range. Full-resident graph performance is
a separate stronger reference. No speedup or latency-target feasibility is claimed.

UVA and layer-prefetch shutdown logs contain forced engine termination after
the grace period. The following arm's GPU process check showed no surviving
workers from the previous arm. This is disclosed alongside successful inference.

`qwen-cache-costs-20261023-v2` refits copy costs from the repeated detailed run.
Cost fitting rejects unequal KV/graph/scheduler configurations and verifies the
current checkpoint revision and tokenizer hash against the predictor identity.
Held-out propagated timing calibration remains outstanding.

Formal agent execution also requires resolving the sampling/template identity:
historical predictors use temperature 0.6, top-p 0.95 and top-k 20; protocol v3
currently declares temperature 0. Reusing those predictors under a new template
version is rejected by the controller. The requested choice is to retain the
historical sampling or collect separate temperature-zero training data. No
formal task comparison has started while this choice is pending.
