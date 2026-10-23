# Real test/build stream protocol

`qwen-long-tools-protocol-20261023-v3` freezes 48 authored commands across
12 already explored benchmark tasks. Four jobs per task run the full suite,
a focused suite, official regression tests and bytecode compilation. The
protocol selects only instance/base/test metadata columns and requested task
rows from the local Parquet dataset; solution patches are excluded. Jobs use
fresh cached task containers, reset tracked files to the benchmark base, and
apply the official test patch during setup. Existing installed and untracked
build artifacts remain available. No confirmation task is selected.

The actual image digest and original image commit are recorded. A normal
SWE-bench preparation commit may change executable modes/build configuration;
the benchmark base object must exist locally. The failed v1/v2 metadata
attempts are retained: v1 required exact image HEAD equality; v2 did not handle
Django data-only test patches. Both failed before tool measurement.

Two containers run concurrently, each with 2 CPU cores, 4 GiB and no network or
GPU access. Full/focused/regression/compile deadlines are 180/300/120/90 seconds.
The planned minima require ten task groups, 36 valid tools, 16 tools over five
seconds, eight over 30 seconds, three over 90 seconds and three uncensored
completions over 30 seconds. At least 16/8 tools must provide first output more
than 0.5/5 seconds before the endpoint, with 16 multi-chunk and four zero-chunk
tools, six successes, three failures and three timeouts. These thresholds are
frozen before measurement and are not reduced if collection misses them.

The collector reuses the existing merged-pipe executor, raw chunk records,
bounded control sink, container resource probe and cleanup. Tool/process-group
termination is confirmed before reporting a timeout. Timeouts are right-censored
at the configured deadline, excluding cleanup delay from the lower bound.
Validation checks immutable command/image identity, clocks, complete byte ranges,
chunk order and export status. Repeated jobs cannot increase unique task coverage.
The corpus reports actual coverage and source hashes; failed captures remain
visible. These are real repository tools in authored workflows, with no LLM
decisions, solved-task throughput or live prefetch gain implied.

```bash
PY=/home/youwei/bzh/venvs/tokenmoe-harness/bin/python
"$PY" -m tokenmoe_opt tools-plan --dataset /home/youwei/bzh/dataset/princeton-nlp/SWE-bench_Verified/data/test-00000-of-00001.parquet --output NEW-PROTOCOL
"$PY" -m tokenmoe_opt tools-collect --protocol NEW-PROTOCOL/protocol.json --output NEW-CORPUS
```

Stream progress, elapsed/history and explicit-progress predictors require a
separate task-held-out evaluation after the actual coverage gate. Small-model
log reading, multi-role workflows and larger-model execution remain outstanding.
