# Local log-reader resource qualification

`qwen-log-reader-protocol-20261023-v2` freezes an exploratory qualification of
the locally available Qwen2.5-7B-Instruct checkpoint. It uses only observed
focused-tool prefixes at elapsed 1/30 seconds and historical priors fitted
without the queried task. Test patches, final outcomes and future log bytes
are excluded from prompts. The resource stage selects the first three eligible
task/job/elapsed-sorted prefixes, uses two warmups and three measured repetitions.

The isolated Transformers eager-SDPA BF16 profile runs on physical GPU 3, with
20 GiB GPU and 32 GiB host-RSS limits, four Torch CPU threads, 1,536 input tokens
and 96 generated tokens. The inline p95 budget is 0.5 seconds, matching the
existing planning horizon. Every measured reply must provide ordered finite
numeric p10/median/p90 values before full quality scoring is eligible.

Frozen source v16 uses root `f2eee36` and fork `13d9223886`. The result
`qwen-log-reader-resource-20261023-v2` records nine measurements:

| Observation | Measured |
|---|---:|
| Warm reader latency p50 / p95 | 0.864 / 1.287 s |
| Observed GPU peak | 17,089,757,184 bytes |
| Process-tree RSS peak | 5,329,944,576 bytes |
| Model/tokenizer startup | 4.654 s |
| Finite numeric interval replies | 3 / 9 |

Memory fits the declared limits; latency and the numeric-only output contract
fail. The other replies return an unknown upper bound rather than a finite
p90, and are retained verbatim. The decision is nondeployment for this
checkpoint/backend/inline profile. Full quality scoring and concurrent solver
interference remain unexecuted. This result does not establish that all small
readers lack useful predictions; a different resource or output contract needs
its own frozen experiment.

The v1 startup failure is retained separately: importing dataset-report code
required PyArrow in the GPU environment before any model loaded. Distribution
statistics now live in a lightweight shared module, preserving the report API;
the reader also loads without Accelerate. Effective Python/Torch/Transformers
versions and package freeze are saved with the successful resource experiment.
