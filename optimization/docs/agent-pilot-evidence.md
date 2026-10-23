# Agent pilot and measurement integrity

The first real agent pilot uses Qwen3-30B-A3B BF16, TP2 on physical GPUs 2/3,
eager Triton, synchronous scheduling, 8 GiB KV per rank, context 32,768,
maximum scheduled batch 4,096, and the fitted template's temperature 0.6,
top-p 0.95, top-k 20 and 4,096-token generation budget. Two independent warmup
requests precede the measured window. Task `astropy__astropy-12907` belongs to
the explored data; the ten confirmation tasks remain untouched.

Source root commit `4b24d00` and fork `ed375549fc` are frozen in independent
worktrees at `qwen-agent-pilot-sources-20261023-v1`, under
`/home/youwei/bzh/dataset/tokenmoe_artifacts`. This lets development proceed
without editing executing Python files. Precompiled native runtime artifacts
are retained beside the frozen Python sources.

The run directory is `qwen-agent-pilot-full-n1-20261023-v1`; extracted metrics
are in `qwen-agent-pilot-full-n1-metrics-20261023-v4` under the same artifact root.
The source-stability check passed. The metadata-only engine exported 48 actual
MoE layer identities. A real Podman preflight checked the common boot clock and
the task's 2-core/4-GiB kernel limits.

| Observed value | Native eager pilot |
|---|---:|
| Measured requests, excluding two warmups | 20 |
| Engine TTFT p95 | 0.3490 s |
| Required-tool completion to complete client response p95 | 27.1000 s (19 requests) |
| Endpoint average TPOT p50 | 0.05544 s/token |
| Measured window | 430.5106 s |
| Observed output-token throughput | 16.0809 token/s |
| Sampled measured GPU peaks, rank 0 / rank 1 | 42,333,241,344 / 42,293,395,456 bytes |
| Simultaneously sampled aggregate GPU peak | 84,626,636,800 bytes |
| Measured process-tree RSS peak | 7,586,512,896 bytes |
| Agent terminal outcome | `timeout`, `LimitsExceeded` at 20 steps |
| Submitted or benchmark-solved tasks | 0 submissions; benchmark score unmeasured |

This is one exploratory task and one repetition, with no cache/predictor policy
comparison. The observed request p95 values fall below the declared targets in
this run. They do not establish workload-wide feasibility or successful-task
throughput. A single task has no task-level confidence interval.

Shutdown exposed a measurement fault: the API default's zero-second shutdown
deadline and the harness's group-wide SIGTERM could kill EngineCore before the
token observer wrote its final status. The raw per-request observations are
retained, but the missing `token-readiness.status.json` makes complete side-channel
export unverified. The metric output reports that status explicitly as missing.
Future runs give the API a 30-second shutdown deadline and signal the API parent
first; forced group termination remains a bounded fallback. The separate
`qwen-api-shutdown-20261023-v1` live check passed in 5.877 s with 19 ordered
events, complete flush, zero drops/errors, no forced group cleanup and no
remaining owned child processes.

`tokenmoe-opt agent-metrics --run RUN --output NEW-DIR` reads the unchanged
harness records from a metadata-only pilot or a finalized historical collection.
It checks exact model/template identity, entity uniqueness, clock agreement,
dependency joins, warmup/measured boundaries and immutable source hashes. It
retains failed/rejected/truncated requests and task outcomes. It reports engine
TTFT separately from the nonstreaming client's complete-payload delivery, with
no invented network first-byte time or interpolated token times. Historical
inference-finished TPOT is labeled an upper bound. Emissions after the measured
window do not inflate throughput. Task bootstrap weights whole task groups.

Resource extraction reconstructs simultaneous aggregate peaks from actual
samples, rather than summing unrelated per-device maxima. Process RSS may count
shared pages repeatedly. Transfer extraction excludes warmup admissions, retains
unused-prefetch bytes at the observation boundary, and requires complete detailed
lifecycles for exact counters. New backend events record one demand-recovery
duration per microbatch and separate router-to-host synchronization; per-expert
wait endpoints are not summed into a fictitious exposed stall.

The qualification entry point accepts saved prompt token IDs, larger scheduled
batches, model context and slot capacities so the real agent prefix can receive
a matching forced-tensor check and copy calibration before cached policy trials.

`qwen-agent-prefix-native-tp2-20261023-v1` and
`qwen-agent-prefix-c64-tp2-20261023-v1` use the same 1,472-token captured first-turn
prefix under the formal 32K/4K/8-GiB settings. The fixed-input tensor gate in
`qwen-agent-prefix-tensors-20261023-v1` passes all 288 rank/layer/frame files,
with zero router differences and equal greedy tokens. Three warm-prefix short
trials took 0.580/0.579/0.490 s natively and 1.843/1.993/2.020 s with 64 slots.
These eight-output-token checks do not replace agent comparisons.

The first detailed 64-slot trace reported 1,737 dropped events on rank 0 and 947
on rank 1 out of 563,339 produced events per rank. Cost fitting correctly rejects
this loss, even though tensor correctness passes. The failed fit remains at
`qwen-agent-prefix-costs-20261023-v1`. The writer's per-event flush is replaced
by bounded batches of at most 64 records with a 10-ms idle flush target, keeping
the 4,096-item admission bound and nonblocking drop behavior. Writer parameters
are exported with status and configuration.

The repeated pair `qwen-agent-prefix-{native,c64}-tp2-20261023-v2` freezes root
`c64e1c5` and fork `71d35f96bf` in source worktree v3. Each rank exported all
563,339 events with zero drops/errors and complete flush. The matching tensor
report passes all 288 files with zero router differences and equal greedy tokens.
Native short trials took 0.467/0.469/0.500 s; cached trials took
1.625/2.006/1.992 s. `qwen-agent-prefix-costs-20261023-v2` records these measured
costs with exact model/config/device identities and native-service geometry.

The real demand-LRU64 N1 run `qwen-agent-pilot-lru64-n1-20261023-v1` used that
loss-free source and cost profile. Its metrics are in
`qwen-agent-pilot-lru64-n1-metrics-20261023-v1`. All ten requests completed at
the transport level; the task ended with `timeout/TimeExceeded`, without a
submission or benchmark score. All side channels flushed without loss.

| Observed value | LRU64 eager pilot |
|---|---:|
| Engine TTFT p95 | 23.5589 s (10 requests) |
| Required-tool completion to complete client response p95 | 97.6936 s (9 requests) |
| Endpoint average TPOT p50 | 0.20423 s/token |
| Measured window | 993.3943 s |
| Sampled measured GPU peaks, rank 0 / rank 1 | 27,510,571,008 / 27,470,725,120 bytes |
| Measured demand-copy bytes per rank | 663,089,577,984 bytes |
| Demand-recovery time, rank 0 / rank 1 | 282.0470 / 282.2645 s |
| Router-to-host synchronization, rank 0 / rank 1 | 54.3370 / 41.6156 s |

This uses about 35% less sampled GPU memory than the native pilot and fails
both the 1-second TTFT and 30-second complete-response targets. One task and one
repetition, with different generated trajectories, cannot establish a paired
task speed ratio or agent-feature gain. Recovery, copy and synchronization
intervals can overlap and must not be summed as independent exposed stalls.

The separate preregistration `qwen-timing-calibration-protocol-20261023-v1`
freezes source hashes and a held-out fixed-input run before measurement. The run
`qwen-agent-prefix-c64-tp2-20261023-v3` has no tensor probes and is excluded from
cost fitting. The report `qwen-timing-calibration-20261023-v2` reconstructs the
closed N1 warmup state and reproduces 1,727,004,672 transferred bytes, 28,914
unique-demand hits, 366 misses, 366 evictions, 366 reloads and 1,155 logical
microbatches, each with zero absolute error. Service predictions of
0.491/0.493/0.524 s undershoot observed 1.789/1.798/1.627 s. Median relative
error is 72.5580% and p95 error 72.5732%, exceeding the preregistered 20%/30%
limits. The fixed-input service gate fails; full-workflow timing remains
unqualified. Cache-specific host/control costs require further measurement.
The failed v1 adapter run is retained separately; its treatment of internal
warmup request IDs was corrected without changing the frozen timing model.

Runtime host-path experiments retain their negative results. Frozen source v5
(root `70390e5`, fork `a3b2276864`) replaces the device mapping table with
host-resolved physical IDs, avoids repeated polling on resident hits, and tests
CPU-group TP agreement. `qwen-agent-prefix-tensors-20261023-v5` passes all 288
files. Its stress run passes delayed-shard and post-cancellation equality; a
deliberately exited worker raises `EngineDeadError` and releases both workers
within 10.845 s without watchdog intervention. The existing shared-memory
resource-tracker warning remains recorded after this intentional failure.

The normal-exit repeat `qwen-agent-prefix-c64-tp2-20261023-v6` takes
2.507/2.445/2.557 s and drops 6,987/4,511 detailed events on ranks 0/1.
`qwen-agent-prefix-costs-20261023-v6` correctly rejects fitting. Across the same
1,968 layer forwards, median router-sync-end to demand-start host time grows
from 0.714 ms in v2 to 1.011 ms in v6. This CPU-group variant is rejected for
performance qualification. Subsequent code retains NCCL agreement with one
host read for both route/order and plan-epoch checks, and uses the existing
msgspec dependency for buffered binary JSON encoding. It requires a fresh
loss-free matched run before use in agent comparisons.

Frozen source v6 (root `5acf923`, fork `d26f2fb47e`) passes the repeated 288-file
gate at `qwen-agent-prefix-tensors-20261023-v7`. Short cached trials fall to
1.241/1.154/1.237 s, but ranks 0/1 still drop 4,838/4,827 detailed records.
The corresponding cost fit remains rejected. The next writer revision raises
its finite admission bound to 16,384 and exports that capacity in configuration
and status; a stalled consumer still triggers nonblocking, counted drops.
`qwen-forced-layer-hostmap-20261023-v1` independently verifies real layer-0
weights with 97 selected logical experts at capacity 16: TP1 and both TP2
shards have zero output error under forced identical inputs.
