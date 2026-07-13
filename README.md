# TokenMoE Prompt MoE Replay Prototype

TokenMoE tests whether metadata visible before MoE routing, especially agent
role/phase and prompt-block structure, predicts real MoE expert working sets.
This stage is deliberately offline: it collects prompt-only vLLM routed-expert
traces, evaluates predictors, and replays scheduler decisions. It does not
modify vLLM's live scheduler or model outputs.

## Data

The external dataset download script is outside this repository:

```bash
python /home/youwei/bzh/dataset/download_dataset.py
```

This change only edits `DATASET_LIST` in that script. Gated datasets are not
replaced silently; failures are recorded in `logs/experiments/`.

Convert raw datasets into local prompt workloads:

```bash
PYTHONPATH=. python -m dataset_adapters.convert_all --limit 256
```

Outputs are written under
`/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/` with manifests under
`/home/youwei/bzh/dataset/tokenmoe_artifacts/manifests/`.

## vLLM Trace Collection

The active collector is vLLM routed-experts only:

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

DeepSeek cross-model profile:

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

The collector rejects dense models, non-vLLM backends, fallback traces, missing
MoE layer IDs, mixed prompt/decode routing, and missing explicit launch profile
settings.

## Analysis

After real traces are collected:

```bash
PYTHONPATH=. python scripts/analyze_traces.py \
  --traces /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/qwen3_sharegpt.jsonl \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/sharegpt_prompt_workloads.jsonl \
  --analysis-dir /home/youwei/bzh/dataset/tokenmoe_artifacts/analysis/qwen3_sharegpt \
  --report-dir /home/youwei/bzh/dataset/tokenmoe_artifacts/reports/qwen3_sharegpt
```

Reports separate global/temporal/RouteSig/oracle predictors and label claim
scopes as real-agent metadata, chat/prompt-only, or domain-instruction.
Prefetch replay and EPLB replay are excluded from current-stage validation.

## Validation

Fast logic tests:

```bash
PYTHONPATH=. pytest -q tests
```

Completion evidence still requires real vLLM MoE runs: Qwen3-30B-A3B over the
five workload files and DeepSeek-V2-Lite-Chat over ShareGPT plus SWE-agent
workloads. Mock, synthetic, deterministic, and unit traces are not sufficient.
Validated run logs are kept under `logs/experiments/`; trace environment
reports are kept under `/home/youwei/bzh/dataset/tokenmoe_artifacts/logs/`.

Next-stage work: online expert prefetch, vLLM scheduler modification, and
proactive EPLB replica placement.
