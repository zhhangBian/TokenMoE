# TokenMoE vLLM/Qwen End-to-End Report

## Setup

- Model: `/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct`
- vLLM: `0.8.5.post1` from `/home/youwei/anaconda3/envs/tokenmoe/lib/python3.10/site-packages/vllm/__init__.py`
- CUDA_VISIBLE_DEVICES: `0`
- Unique workload requests: 216
- Batch sizes: [1, 2, 4, 8]
- Modes: ['baseline', 'tokenmoe_sidecar']
- Model type: `qwen2`
- Architectures: `['Qwen2ForCausalLM']`
- MoE routed expert capture supported: `False`
- Fallback reason: `dense_model_no_moe_router`

## Key Results

- Best observed throughput was 20.45 requests/s at batch size 8 in `baseline` mode.
- TokenMoE sidecar metadata decisions averaged 27.30 us/request (p95 50.44 us).
- Sidecar fallback reasons: `{'dense_model_no_moe_router': 864}`.
- Baseline vs sidecar exact output match averaged 0.897 across separately sharded batch-size runs.

The requested Qwen2.5-7B-Instruct model is a dense Qwen2 CausalLM model, so the vLLM path was exercised as a real serving run while routed-expert capture was correctly disabled by capability detection. The sidecar does not alter prompts, sampling parameters, or model weights; exact-output matching is reported as a measurement because vLLM batch execution is not bitwise stable across independently restarted shards.

## Figures

![Latency by batch](analysis/vllm_qwen/figures/latency_by_batch.png)

![Throughput by batch](analysis/vllm_qwen/figures/throughput_by_batch.png)

![Sidecar overhead](analysis/vllm_qwen/figures/sidecar_overhead.png)

![Latency by role](analysis/vllm_qwen/figures/latency_by_role.png)

## Summary Table

| mode             |   batch_size |   requests |   mean_latency_ms |   p95_latency_ms |   throughput_req_s |   throughput_output_tok_s |   mean_sidecar_us |
|:-----------------|-------------:|-----------:|------------------:|-----------------:|-------------------:|--------------------------:|------------------:|
| baseline         |            1 |        216 |           364.523 |          373.065 |              2.743 |                    87.786 |             0.000 |
| baseline         |            2 |        216 |           181.362 |          186.847 |              5.514 |                   176.442 |             0.000 |
| baseline         |            4 |        216 |            92.961 |           97.085 |             10.757 |                   344.229 |             0.000 |
| baseline         |            8 |        216 |            48.888 |           49.787 |             20.455 |                   654.558 |             0.000 |
| tokenmoe_sidecar |            1 |        216 |           351.724 |          359.161 |              2.843 |                    90.980 |            41.581 |
| tokenmoe_sidecar |            2 |        216 |           181.528 |          187.328 |              5.509 |                   176.281 |            42.746 |
| tokenmoe_sidecar |            4 |        216 |            93.328 |           94.763 |             10.715 |                   342.877 |            14.904 |
| tokenmoe_sidecar |            8 |        216 |            48.924 |           49.753 |             20.440 |                   654.080 |             9.968 |

## Output Stability

|   batch_size |   requests |   exact_output_match_rate |
|-------------:|-----------:|--------------------------:|
|        1.000 |    216.000 |                     1.000 |
|        2.000 |    216.000 |                     0.870 |
|        4.000 |    216.000 |                     0.852 |
|        8.000 |    216.000 |                     0.866 |

## Data

- Raw generations: `data/vllm_qwen/generations.jsonl`
- Sidecar decisions: `data/vllm_qwen/sidecar_decisions.jsonl`
- Latency CSV: `data/vllm_qwen/latency.csv`
- Capability JSON: `analysis/vllm_qwen/model_capability.json`
- Summary JSON: `analysis/vllm_qwen/experiment_summary.json`