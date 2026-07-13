# TokenMoE vLLM Model Profiles

## Qwen3-30B-A3B

- Path: `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B`
- Model type: `qwen3_moe`
- Hidden layers: 48
- MoE layers: 0-47
- Routed experts: 128
- Router top-k: 8

Example profile:

```bash
source /home/youwei/bzh/venvs/tokenmoe-vllm/bin/activate
cd /tmp
TOKENMOE_ROOT=/home/youwei/bzh/project/TokenMoE
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT \
python -u $TOKENMOE_ROOT/scripts/collect_traces.py \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/sharegpt_prompt_workloads.jsonl \
  --model /home/youwei/bzh/model/Qwen/Qwen3-30B-A3B \
  --limit 256 \
  --batch-size 32 \
  --tensor-parallel-size 2 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.90 \
  --output /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/qwen3_sharegpt.jsonl \
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/qwen3_sharegpt_env.json
```

Shorter SWE-agent prompts were also validated with TP1 and
`--max-model-len 2048`.

## DeepSeek-V2-Lite-Chat

- Path: `/home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat`
- Model type: `deepseek_v2`
- Hidden layers: 27
- MoE layers: 1-26 (`first_k_dense_replace = 1`)
- Routed experts: 64
- Router top-k: 6

Example profile:

```bash
source /home/youwei/bzh/venvs/tokenmoe-vllm/bin/activate
cd /tmp
TOKENMOE_ROOT=/home/youwei/bzh/project/TokenMoE
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT \
python -u $TOKENMOE_ROOT/scripts/collect_traces.py \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/swe_agent_prompt_workloads.jsonl \
  --model /home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat \
  --limit 256 \
  --batch-size 32 \
  --tensor-parallel-size 1 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --output /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/deepseek_swe_agent.jsonl \
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/deepseek_swe_agent_env.json
```

ShareGPT validation used the same DeepSeek profile with
`--max-model-len 10240`.

## Preflight Requirements

The collector requires:

- MoE model config
- `enable_return_routed_experts=True`
- pipeline parallel size 1
- context parallel size 1
- KV transfer/connectors disabled
- explicit tensor parallelism, expert parallelism, dtype, max model length, and
  GPU memory utilization
- derived `moe_layer_ids` and router top-k

Validated experiment environment:

- Python environment: `/home/youwei/bzh/venvs/tokenmoe-vllm`
- vLLM source checkout: `/home/youwei/bzh/project/TokenMoE/vllm`
- vLLM commit: `0b3ba88f165976e77ca5e6a7a3f5bba4562b80af`
- Import path: run from outside the repository root, or put
  `/home/youwei/bzh/project/TokenMoE/vllm` before
  `/home/youwei/bzh/project/TokenMoE` in `PYTHONPATH`.
- Environment reports:
  `/home/youwei/bzh/dataset/tokenmoe_artifacts/logs/*_env.json`
