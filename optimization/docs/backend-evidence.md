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

Controlled agent comparisons, observer overhead with three measured repetitions,
native graph performance, long-tool/multi-role coverage, and larger-model
migration have independent outstanding gates. The formal agent protocol uses
4096 generated tokens, engine TTFT p95 at most 1 s, and required-tool completion
to fully received batched response p95 at most 30 s. The 8/64-token diagnostics
cannot be substituted for that protocol.
