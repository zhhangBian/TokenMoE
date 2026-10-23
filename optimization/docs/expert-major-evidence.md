# Expert-major prefill qualification

This is a separate demand-only mechanism profile on physical GPUs 2/3. The
token-executor agent-policy protocol remains separate. Frozen source worktree
`qwen-agent-pilot-sources-20261023-v9` uses root `5d1e243` and fork
`a24971a1c2`; all artifacts below are under
`/home/youwei/bzh/dataset/tokenmoe_artifacts`.

The default-off implementation groups only the current batch's selected
token/expert pairs. It loads capacity-fitting expert groups, retains their pins
through bounded chunks, scatters every contribution to its original token/top-k
position, and uses the existing `moe_sum` reduction. Smaller batches and decode
retain the original token executor. The default chunk bound is 256 pairs. At
4,096 scheduled tokens, 192 MiB of rank-local scratch is explicitly reserved;
an insufficient reservation fails configuration before loading.

`qwen-expertmajor-{native,c64}-tp2-20261023-v1` uses BF16 Qwen3-30B-A3B, TP2,
eager Triton, 8 GiB KV per rank, 32K context and the same saved 1,472-token
prefix. Two warmups precede three eight-output-token short trials. The report
`qwen-expertmajor-tensors-20261023-v1` passes all 288 rank/layer/frame files:
input, coefficient and output errors are zero, routes match and greedy tokens
are equal. Both sources remain stable. Each rank exports all 148,461 detailed
records with zero drops/errors and complete flush. The matching measured cost
profile is `qwen-expertmajor-costs-20261023-v1`.

| Diagnostic | Token executor, prefix v8 | Expert-major, v1 |
|---|---:|---:|
| Measured short trials | 1.279 / 1.169 / 1.141 s | 1.239 / 1.249 / 1.221 s |
| Whole-engine admitted bytes per rank | 121,239,502,848 | 35,710,304,256 |
| Whole-engine transfers per rank | 25,694 | 7,568 |
| Whole-engine acquisition groups | 3,644 | 2,164 |
| Expert-major routed contributions per rank | 0 | 565,248 |
| Expert-major kernel chunks per rank | 0 | 2,256 |

Whole-engine counts include startup/internal work and warmup. The first prefix
request's observed TTFT is 30.847 s versus 11.941 s; it also includes first-run
JIT and tensor-probe overhead. These sequential diagnostics have no rotated
mode order and do not establish an agent latency gain. Repeated short requests
reuse the prefix and mostly execute the original token path. Full agent
latency targets remain unverified for expert-major execution.

`qwen-expertmajor-layer-shards-20261023-v1` checks actual checkpoint weights at
capacity 16, with 128 forced tokens selecting 127 unique experts. TP1 and both
TP2 shards have zero output error. `qwen-expertmajor-stress-tp2-20261023-v1`
passes delayed-shard equality, cancellation followed by identical generation,
and peer failure. Cancellation and peer failure force fresh 1,472-token
prefills. Rank 1 incurs 48 injected shard delays; a deliberate peer exit raises
`EngineDeadError` and releases both workers in 10.828 s without the watchdog.
The existing resource-tracker shared-memory warning after intentional failure
is retained. The fork's 18 cache GPU tests and pre-commit/mypy checks pass.

The old transfer-benefit heuristic estimates repeated token-microbatch loads.
Expert-major prefill instead loads an expert once per scheduled batch and
changes which speculative copies survive until use. Reusing that heuristic is
unsupported: the controller rejects expert-major predictive prefetch. The
separate mechanism protocol permits full/offloader/LRU/LFU comparisons while
agent-conditioned comparisons continue under the token protocol. Numeric
qualification does not qualify the timing simulator or the predictive planner.

`qwen-prefix-route-order-20261023-v2` separately evaluates row-order variants in
CPU cache-count replay under a hypothetical empty initial cache. It reduces
64-slot bytes by about 22–23% for expert-set/bitmask sorting; top-1 sorting
regresses. The v1 initial-state description was incorrect and is retained with
the correction in v2. `qwen-route-pair-algebra-20261023-v1` is a one-layer
full-bank algebra check, not a bounded-storage experiment. Neither diagnostic
substitutes for the live qualification above.

The first agent attempt `qwen-agent-pilot-expertmajor-lru64-n1-20261023-v1`
fails before any agent request: its longer output directory exceeds the UNIX
socket pathname bound. The workers terminate and the failure is preserved.
The repaired mailbox binds through an open private-directory descriptor and
advertises its short address in worker state, retaining the original socket
and state-file locations and permissions. A cross-process regression test uses
a directory longer than 108 bytes and checks delivery and repeated cleanup.
