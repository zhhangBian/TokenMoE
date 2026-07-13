# TokenMoE Real vLLM MoE Prediction/Replay Handoff

Date: 2026-07-13

Change: `add-prediction-replay-moe-download`

Repository: `/home/youwei/bzh/project/TokenMoE`

Artifact root: `/home/youwei/bzh/dataset/tokenmoe_artifacts`

This document is intended to be enough context for a future session with no
chat history. It records the research intent, the current implementation, how
to rerun the experiments, what the present conclusions are, and where the next
research work should continue.

## TL;DR

This stage is complete. OpenSpec reports `65/65` tasks complete and
`add-prediction-replay-moe-download` is ready to archive.

The current-stage goal was not online acceleration. The goal was to build a
real evidence loop:

1. Convert external datasets into prompt-block workloads.
2. Capture real prompt-only MoE routed experts from vLLM.
3. Align prompt blocks to prompt token spans.
4. Predict per segment and per MoE layer expert working sets before router
   execution.
5. Compare RouteSig against simple baselines and oracle.
6. Run offline scheduler replay using predicted demand, while true routed
   experts are used only after decisions for scoring.

That loop now works on real local MoE models:

- Qwen3-30B-A3B over five datasets, 256 prompt records each.
- DeepSeek-V2-Lite-Chat over ShareGPT and SWE-agent, 256 prompt records each.
- All required traces are `vllm-routed-experts`, `prompt_only`, v2 schema, and
  have `backend_fallback_used = false`.
- SWE-agent now has a real dependency graph for replay: 256 records, 32 source
  groups, 8 records per group, 224 dependency edges, ready times present, zero
  dependency violations in replay.

Main result:

RouteSig shows real prompt/block metadata signal on most prompt/domain
datasets, but it is not yet a strong agent-DAG result. On SWE-agent, RouteSig
does not consistently beat temporal/global baselines. The next research stage
should improve predictors and SWE metadata before modifying vLLM runtime
scheduling or prefetching.

## Current Research Claim Boundary

What can be claimed now:

- Real vLLM MoE prompt traces can be captured and analyzed end to end.
- Prompt block metadata has measurable predictive signal for MoE expert sets.
- RouteSig beats simple frequency/temporal baselines on most non-agent
  prompt/domain datasets in this 256-record validation.
- Offline replay now uses prediction-based decisions rather than true experts
  as the decision signal.
- SWE-agent replay is dependency-safe and can support agent-DAG experiments.

What cannot be claimed yet:

- Do not claim online runtime acceleration.
- Do not claim vLLM scheduler modification has been implemented.
- Do not claim expert prefetch/offload/EPLB placement is implemented.
- Do not claim agent metadata clearly beats temporal locality on SWE-agent.
- Do not claim scheduler replay currently gives stable fanout or all-to-all
  proxy improvement.

The most accurate current statement is:

TokenMoE has a validated real-vLLM prompt-routing prediction and offline replay
pipeline. RouteSig demonstrates useful prompt/block locality signal, especially
on chat/domain workloads. SWE-agent now has legal agent-DAG replay, but the
agent-specific predictor needs stronger metadata and modeling before it can
support a strong systems optimization claim.

## OpenSpec State

Change directory:

```text
openspec/changes/add-prediction-replay-moe-download/
```

Important files:

- `proposal.md`
- `design.md`
- `tasks.md`
- `specs/tokenmoe-route-signature/spec.md`
- `specs/tokenmoe-evaluation-resources/spec.md`
- `specs/tokenmoe-system-optimization/spec.md`

Status command:

```bash
openspec instructions apply --change add-prediction-replay-moe-download --json
```

Expected state:

```text
progress: 65/65
state: all_done
```

Validation command:

```bash
openspec validate add-prediction-replay-moe-download --strict
```

Expected result:

```text
Change 'add-prediction-replay-moe-download' is valid
```

Mixtral note:

`4.11` is marked complete as an optional compatibility run skipped for this
stage. Primary completion depends on Qwen3 and DeepSeek only. This matches the
spec decision that Mixtral is optional and must not block completion.

## Code Layout

### Dataset adapters

Folder:

```text
dataset_adapters/
```

Purpose:

- Dedicated conversion code outside core `tokenmoe/`.
- One converter per external dataset.
- Emits normalized v2 prompt workloads under the artifact root.
- Emits manifests with provenance, access/license/redaction status, mapping
  version, unavailable fields, claim scope, and DAG availability.

Key files:

- `dataset_adapters/common.py`
- `dataset_adapters/sharegpt.py`
- `dataset_adapters/lmsys.py`
- `dataset_adapters/swe_agent.py`
- `dataset_adapters/opencode.py`
- `dataset_adapters/openmath.py`
- `dataset_adapters/convert_all.py`

Important SWE-agent fix:

- `MAX_EVENTS_PER_TRAJECTORY = 8`
- `source_group_id` is now based on the base trajectory group plus source
  index, for example `AnalogJ__lexicon-336:0`.
- This prevents all 256 SWE records from collapsing into one source group.
- Final SWE workload has 32 groups, 8 records per group, and 224 dependency
  edges.

### Trace collection

Main file:

```text
scripts/collect_traces.py
```

Current behavior:

- vLLM routed-experts only.
- No active Transformers router-logit backend.
- Uses `enable_return_routed_experts=True`.
- Uses `SamplingParams(..., routed_experts_prompt_start=0)`.
- Generates only `max_tokens=1` by default to trigger vLLM output.
- Keeps only prompt-token routing.
- Validates tokenizer-aligned prompt spans against vLLM `prompt_token_ids`.
- Persists `moe_layer_ids`, `router_top_k`, `backend`, fallback flags,
  `routing_scope`, prompt segments, and source provenance.
- Supports `--batch-size`; real experiments used `--batch-size 32`.

Important support file:

```text
tokenmoe/vllm_sidecar.py
```

Purpose:

- Load HuggingFace config.
- Infer MoE capability.
- Derive `moe_layer_ids`, router top-k, and expert count.
- Validate explicit vLLM launch profile.

### Schema and trace logic

Key files:

- `tokenmoe/schema.py`
- `tokenmoe/trace.py`

Current schema versions:

- Workload: `tokenmoe.workload.v2`
- Trace: `tokenmoe.trace.v2`

Trace hard checks:

- Backend must be `vllm-routed-experts`.
- Fallback traces are rejected.
- Routing scope must be `prompt_only`.
- Mixed v1/v2 analysis is rejected.
- `moe_layer_ids` must be present.
- Dense/non-MoE layers must not be treated as expert-0 demand.
- Segment alignment failure is explicit, not silent.

### Prediction and RouteSig

Key files:

- `tokenmoe/prediction.py`
- `tokenmoe/routesig.py`
- `tokenmoe/metrics.py`

Predictors currently evaluated:

- `global_frequency`
- `request_lru`
- `sequence_history`
- `temporal_window_frequency`
- `routesig`
- `oracle_future_demand`

Prediction granularity:

- request ID
- prompt segment ID
- MoE layer ID
- top-M expert set
- normalized expert weights
- confidence
- predictor source
- fallback key
- block type
- segment position
- token span

RouteSig fallback order:

```text
(agent_id, role, phase, block_type, segment_position)
  -> (role, phase, block_type, segment_position)
  -> (role, phase, block_type)
  -> (role, block_type)
  -> (block_type)
  -> (role, phase)
  -> (role)
  -> global
```

Top-M budgets are model-relative:

- `1x router_top_k`
- `1.5x router_top_k`
- `2x router_top_k` as main report setting
- `3x router_top_k`

Concrete budgets:

- Qwen3 top-k 8: 8, 12, 16, 24
- DeepSeek top-k 6: 6, 9, 12, 18

### Scheduler replay and reporting

Key files:

- `tokenmoe/simulators.py`
- `tokenmoe/reporting.py`
- `scripts/analyze_traces.py`

Current replay behavior:

- Scheduler decisions use predicted expert-token-label demand.
- True routed experts are used only after batch selection for scoring.
- SWE-agent with dependency edges is reported as `agent_dag`.
- Non-agent datasets are reported as `prompt_or_domain_locality`.
- Dependency violations are counted.
- Prefetch and EPLB are archived/excluded, not active current-stage outputs.

## External Data

Dataset download script is outside this repo:

```text
/home/youwei/bzh/dataset/download_dataset.py
```

Important constraint:

Only `DATASET_LIST` may be edited in that script. Do not change endpoint logic,
token handling, retry logic, or replacement dataset behavior.

Expected dataset list:

```python
DATASET_LIST = [
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "lmsys/lmsys-chat-1m",
    "nebius/SWE-agent-trajectories",
    "nvidia/OpenCodeInstruct",
    "nvidia/OpenMathInstruct-2",
]
```

Run download:

```bash
python /home/youwei/bzh/dataset/download_dataset.py
```

Download logs from this run:

- `logs/experiments/20260713_dataset_download.log`
- `logs/experiments/20260713_dataset_download_user_config.log`

## Artifact Layout

Artifact root:

```text
/home/youwei/bzh/dataset/tokenmoe_artifacts/
```

Workloads:

```text
workloads/sharegpt_prompt_workloads.jsonl
workloads/lmsys_prompt_workloads.jsonl
workloads/swe_agent_prompt_workloads.jsonl
workloads/opencode_prompt_workloads.jsonl
workloads/openmath_prompt_workloads.jsonl
```

Manifests:

```text
manifests/sharegpt_manifest.json
manifests/lmsys_manifest.json
manifests/swe_agent_manifest.json
manifests/opencode_manifest.json
manifests/openmath_manifest.json
```

Required traces:

```text
traces/qwen3_sharegpt.jsonl
traces/qwen3_lmsys.jsonl
traces/qwen3_swe_agent.jsonl
traces/qwen3_opencode.jsonl
traces/qwen3_openmath.jsonl
traces/deepseek_sharegpt.jsonl
traces/deepseek_swe_agent.jsonl
```

Environment reports:

```text
logs/qwen3_sharegpt_env.json
logs/qwen3_lmsys_env.json
logs/qwen3_swe_agent_env.json
logs/qwen3_opencode_env.json
logs/qwen3_openmath_env.json
logs/deepseek_sharegpt_env.json
logs/deepseek_swe_agent_env.json
```

Analysis metrics:

```text
analysis/<dataset-model>/locality_metrics.json
analysis/<dataset-model>/locality_report.md
```

Final reports:

```text
reports/qwen3_sharegpt/tokenmoe_report.md
reports/qwen3_lmsys/tokenmoe_report.md
reports/qwen3_swe_agent/tokenmoe_report.md
reports/qwen3_opencode/tokenmoe_report.md
reports/qwen3_openmath/tokenmoe_report.md
reports/deepseek_sharegpt/tokenmoe_report.md
reports/deepseek_swe_agent/tokenmoe_report.md
```

Experiment log index:

```text
logs/experiments/20260713_real_vllm_validation_summary.md
```

## Environment

Validated vLLM environment:

```text
venv: /home/youwei/bzh/venvs/tokenmoe-vllm
vLLM source: /home/youwei/bzh/project/TokenMoE/vllm
vLLM commit: 0b3ba88f165976e77ca5e6a7a3f5bba4562b80af
```

When running vLLM commands, do not run from the repo root unless import
shadowing has been checked. The validated pattern is:

```bash
source /home/youwei/bzh/venvs/tokenmoe-vllm/bin/activate
cd /tmp
TOKENMOE_ROOT=/home/youwei/bzh/project/TokenMoE
export PYTHONPATH=$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT
```

The vLLM source path must precede the TokenMoE repo path in `PYTHONPATH`.

GPU notes from this run:

- Initial filler processes on GPU0/GPU1 were killed with user permission.
- Later GPU0 was occupied by another user's process, so final SWE reruns used
  GPU1 only.
- Do not assume the same GPU occupancy in future runs.
- Qwen3 ShareGPT/LMSYS/OpenCode/OpenMath were run TP2 on GPU0+1.
- Qwen3 SWE was run TP1 on GPU1 because the prompt lengths fit and GPU0 was
  occupied.
- DeepSeek was run TP1 on GPU1.

## Model Profiles

### Qwen3-30B-A3B

Path:

```text
/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B
```

Capability:

- model type: `qwen3_moe`
- hidden layers: 48
- MoE layers: 48, IDs `0..47`
- routed experts: 128
- router top-k: 8

Validated profiles:

- ShareGPT, LMSYS, OpenCode, OpenMath:
  - `tensor_parallel_size = 2`
  - `expert_parallel = on`
  - `dtype = bfloat16`
  - `max_model_len = 10240`
  - `gpu_memory_utilization = 0.90`
  - `batch_size = 32`
- SWE-agent:
  - `tensor_parallel_size = 1`
  - `expert_parallel = on`
  - `dtype = bfloat16`
  - `max_model_len = 2048`
  - `gpu_memory_utilization = 0.90`
  - `batch_size = 32`

### DeepSeek-V2-Lite-Chat

Path:

```text
/home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat
```

Capability:

- model type: `deepseek_v2`
- hidden layers: 27
- MoE layers: 26, IDs `1..26`
- routed experts: 64
- router top-k: 6
- requires `--trust-remote-code`

Validated profiles:

- ShareGPT:
  - `tensor_parallel_size = 1`
  - `expert_parallel = on`
  - `dtype = bfloat16`
  - `max_model_len = 10240`
  - `gpu_memory_utilization = 0.90`
  - `batch_size = 32`
- SWE-agent:
  - `tensor_parallel_size = 1`
  - `expert_parallel = on`
  - `dtype = bfloat16`
  - `max_model_len = 2048`
  - `gpu_memory_utilization = 0.90`
  - `batch_size = 32`

## Reproduction Commands

All commands below assume:

```bash
cd /home/youwei/bzh/project/TokenMoE
```

### Convert datasets

```bash
PYTHONPATH=. /home/youwei/anaconda3/bin/python -u -m dataset_adapters.convert_all --limit 256 \
  2>&1 | tee logs/experiments/$(date +%Y%m%d)_dataset_convert_all.log
```

Expected workload count:

- 256 rows for each of the five workload files.
- SWE-agent: 256 rows, 32 groups, group size 8, 224 dependency edges.

### Collect Qwen3 traces

Use the vLLM environment pattern:

```bash
source /home/youwei/bzh/venvs/tokenmoe-vllm/bin/activate
cd /tmp
TOKENMOE_ROOT=/home/youwei/bzh/project/TokenMoE
export PYTHONPATH=$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT
```

Qwen3 ShareGPT:

```bash
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=0,1 \
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
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/qwen3_sharegpt_env.json \
  2>&1 | tee $TOKENMOE_ROOT/logs/experiments/$(date +%Y%m%d)_qwen3_sharegpt_256_trace.log
```

Qwen3 LMSYS:

```bash
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=0,1 \
python -u $TOKENMOE_ROOT/scripts/collect_traces.py \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/lmsys_prompt_workloads.jsonl \
  --model /home/youwei/bzh/model/Qwen/Qwen3-30B-A3B \
  --limit 256 \
  --batch-size 32 \
  --tensor-parallel-size 2 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.90 \
  --output /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/qwen3_lmsys.jsonl \
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/qwen3_lmsys_env.json \
  2>&1 | tee $TOKENMOE_ROOT/logs/experiments/$(date +%Y%m%d)_qwen3_lmsys_256_trace.log
```

Qwen3 SWE-agent:

```bash
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 \
python -u $TOKENMOE_ROOT/scripts/collect_traces.py \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/swe_agent_prompt_workloads.jsonl \
  --model /home/youwei/bzh/model/Qwen/Qwen3-30B-A3B \
  --limit 256 \
  --batch-size 32 \
  --tensor-parallel-size 1 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --output /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/qwen3_swe_agent.jsonl \
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/qwen3_swe_agent_env.json \
  2>&1 | tee $TOKENMOE_ROOT/logs/experiments/$(date +%Y%m%d)_qwen3_swe_agent_256_trace.log
```

Qwen3 OpenCode:

```bash
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=0,1 \
python -u $TOKENMOE_ROOT/scripts/collect_traces.py \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/opencode_prompt_workloads.jsonl \
  --model /home/youwei/bzh/model/Qwen/Qwen3-30B-A3B \
  --limit 256 \
  --batch-size 32 \
  --tensor-parallel-size 2 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.90 \
  --output /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/qwen3_opencode.jsonl \
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/qwen3_opencode_env.json \
  2>&1 | tee $TOKENMOE_ROOT/logs/experiments/$(date +%Y%m%d)_qwen3_opencode_256_trace.log
```

Qwen3 OpenMath:

```bash
/usr/bin/time -v env CUDA_VISIBLE_DEVICES=0,1 \
python -u $TOKENMOE_ROOT/scripts/collect_traces.py \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/openmath_prompt_workloads.jsonl \
  --model /home/youwei/bzh/model/Qwen/Qwen3-30B-A3B \
  --limit 256 \
  --batch-size 32 \
  --tensor-parallel-size 2 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.90 \
  --output /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/qwen3_openmath.jsonl \
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/qwen3_openmath_env.json \
  2>&1 | tee $TOKENMOE_ROOT/logs/experiments/$(date +%Y%m%d)_qwen3_openmath_256_trace.log
```

### Collect DeepSeek traces

DeepSeek ShareGPT:

```bash
source /home/youwei/bzh/venvs/tokenmoe-vllm/bin/activate
cd /tmp
TOKENMOE_ROOT=/home/youwei/bzh/project/TokenMoE
export PYTHONPATH=$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT

/usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 \
python -u $TOKENMOE_ROOT/scripts/collect_traces.py \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/sharegpt_prompt_workloads.jsonl \
  --model /home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat \
  --limit 256 \
  --batch-size 32 \
  --tensor-parallel-size 1 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --output /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/deepseek_sharegpt.jsonl \
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/deepseek_sharegpt_env.json \
  2>&1 | tee $TOKENMOE_ROOT/logs/experiments/$(date +%Y%m%d)_deepseek_sharegpt_256_trace.log
```

DeepSeek SWE-agent:

```bash
source /home/youwei/bzh/venvs/tokenmoe-vllm/bin/activate
cd /tmp
TOKENMOE_ROOT=/home/youwei/bzh/project/TokenMoE
export PYTHONPATH=$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT

/usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 \
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
  --env-report /home/youwei/bzh/dataset/tokenmoe_artifacts/logs/deepseek_swe_agent_env.json \
  2>&1 | tee $TOKENMOE_ROOT/logs/experiments/$(date +%Y%m%d)_deepseek_swe_agent_256_trace.log
```

### Run analysis/reporting

Run from the repo root with the normal Python environment:

```bash
cd /home/youwei/bzh/project/TokenMoE
```

Template:

```bash
/usr/bin/time -v env MPLBACKEND=Agg PYTHONPATH=. /home/youwei/anaconda3/bin/python -u scripts/analyze_traces.py \
  --traces /home/youwei/bzh/dataset/tokenmoe_artifacts/traces/<trace>.jsonl \
  --workload /home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/<workload>.jsonl \
  --analysis-dir /home/youwei/bzh/dataset/tokenmoe_artifacts/analysis/<name> \
  --report-dir /home/youwei/bzh/dataset/tokenmoe_artifacts/reports/<name> \
  2>&1 | tee logs/experiments/$(date +%Y%m%d)_<name>_analysis.log
```

Concrete mapping:

```text
qwen3_sharegpt:
  trace: qwen3_sharegpt.jsonl
  workload: sharegpt_prompt_workloads.jsonl

qwen3_lmsys:
  trace: qwen3_lmsys.jsonl
  workload: lmsys_prompt_workloads.jsonl

qwen3_swe_agent:
  trace: qwen3_swe_agent.jsonl
  workload: swe_agent_prompt_workloads.jsonl

qwen3_opencode:
  trace: qwen3_opencode.jsonl
  workload: opencode_prompt_workloads.jsonl

qwen3_openmath:
  trace: qwen3_openmath.jsonl
  workload: openmath_prompt_workloads.jsonl

deepseek_sharegpt:
  trace: deepseek_sharegpt.jsonl
  workload: sharegpt_prompt_workloads.jsonl

deepseek_swe_agent:
  trace: deepseek_swe_agent.jsonl
  workload: swe_agent_prompt_workloads.jsonl
```

Final successful analysis logs from this run:

```text
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v8.log
logs/experiments/20260713_qwen3_lmsys_analysis_optimized_v8.log
logs/experiments/20260713_qwen3_swe_agent_analysis_groupfix.log
logs/experiments/20260713_qwen3_opencode_analysis_optimized_v8.log
logs/experiments/20260713_qwen3_openmath_analysis_optimized_v8.log
logs/experiments/20260713_deepseek_sharegpt_analysis_optimized_v8.log
logs/experiments/20260713_deepseek_swe_agent_analysis_groupfix.log
```

## Final Results

All numbers below are from
`/home/youwei/bzh/dataset/tokenmoe_artifacts/analysis/*/locality_metrics.json`.

### Prediction, main 2x budget

| Dataset/model | Global | Temporal | RouteSig | Oracle | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `qwen3_sharegpt` | 0.333202 | 0.326389 | 0.340535 | 0.587052 | RouteSig beats global/temporal |
| `qwen3_lmsys` | 0.350937 | 0.314470 | 0.370612 | 0.613590 | RouteSig beats global/temporal |
| `qwen3_swe_agent` | 0.441582 | 0.443449 | 0.439890 | 0.544593 | RouteSig slightly below temporal/global |
| `qwen3_opencode` | 0.394782 | 0.394442 | 0.428792 | 0.516572 | RouteSig beats global/temporal |
| `qwen3_openmath` | 0.440887 | 0.439864 | 0.491650 | 0.630145 | RouteSig beats global/temporal |
| `deepseek_sharegpt` | 0.239133 | 0.230431 | 0.248640 | 0.382351 | RouteSig beats global/temporal |
| `deepseek_swe_agent` | 0.369003 | 0.373778 | 0.345094 | 0.445222 | RouteSig below temporal/global |

Interpretation:

- Prompt/domain datasets show consistent metadata/block locality signal.
- SWE-agent is now structurally valid for DAG replay, but RouteSig is not yet
  the strongest predictor there.
- Oracle gap remains large, which means there is still modeling headroom.

### Scheduler replay fanout

| Dataset/model | Replay kind | DAG available | FIFO | Temporal | RouteSig | Oracle |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `qwen3_sharegpt` | `prompt_or_domain_locality` | false | 5643.45 | 5653.00 | 5654.10 | 5577.95 |
| `qwen3_lmsys` | `prompt_or_domain_locality` | false | 5476.85 | 5361.85 | 5461.95 | 5274.60 |
| `qwen3_swe_agent` | `agent_dag` | true | 5060.60 | 5143.70 | 5115.85 | 5121.75 |
| `qwen3_opencode` | `prompt_or_domain_locality` | false | 5627.60 | 5603.95 | 5616.10 | 5574.90 |
| `qwen3_openmath` | `prompt_or_domain_locality` | false | 4695.15 | 4676.60 | 4676.60 | 4630.35 |
| `deepseek_sharegpt` | `prompt_or_domain_locality` | false | 1664.00 | 1664.00 | 1664.00 | 1664.00 |
| `deepseek_swe_agent` | `agent_dag` | true | 1657.50 | 1659.30 | 1662.00 | 1660.05 |

Interpretation:

- Replay is legal and prediction-based.
- Non-agent datasets are not used to make agent-DAG claims.
- Current scheduler objective does not yet give stable fanout improvement.
- This is evidence that the next stage should improve predictor/scoring before
  runtime integration.

## Validation Commands

Fast tests:

```bash
cd /home/youwei/bzh/project/TokenMoE
PYTHONPATH=. pytest -q tests
```

Expected result from this run:

```text
12 passed
```

OpenSpec validation:

```bash
openspec validate add-prediction-replay-moe-download --strict
```

Expected:

```text
Change 'add-prediction-replay-moe-download' is valid
```

Whitespace check:

```bash
git diff --check
```

Expected: no output.

Real artifact validation snippet:

```bash
cd /home/youwei/bzh/project/TokenMoE
python - <<'PY'
import json
from pathlib import Path

root = Path('/home/youwei/bzh/dataset/tokenmoe_artifacts')
required = {
    'qwen3_sharegpt': ('/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B', 'chat_prompt_only', 8, 48),
    'qwen3_lmsys': ('/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B', 'chat_prompt_only', 8, 48),
    'qwen3_swe_agent': ('/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B', 'real_agent_metadata', 8, 48),
    'qwen3_opencode': ('/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B', 'domain_instruction', 8, 48),
    'qwen3_openmath': ('/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B', 'domain_instruction', 8, 48),
    'deepseek_sharegpt': ('/home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat', 'chat_prompt_only', 6, 26),
    'deepseek_swe_agent': ('/home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat', 'real_agent_metadata', 6, 26),
}

for name, (model, claim, topk, moe_layers) in required.items():
    count = 0
    for line in (root / 'traces' / f'{name}.jsonl').open():
        rec = json.loads(line)
        count += 1
        assert rec['schema_version'] == 'tokenmoe.trace.v2', name
        assert rec['backend'] == 'vllm-routed-experts', name
        assert rec['backend_fallback_used'] is False, name
        assert rec['routing_scope'] == 'prompt_only', name
        assert rec['model_id'] == model, name
        assert rec['claim_scope'] == claim, name
        assert rec['router_top_k'] == topk, name
        assert len(rec['moe_layer_ids']) == moe_layers, name
        assert not rec.get('segment_unavailable_reason'), name
        allowed = set(rec['moe_layer_ids'])
        assert all(layer['layer_id'] in allowed for layer in rec['layers']), name
    assert count == 256, (name, count)

    metrics = json.loads((root / 'analysis' / name / 'locality_metrics.json').read_text())
    assert metrics['locality']['num_records'] == 256, name
    assert metrics['prediction']['available'] is True, name
    assert metrics['prediction']['split']['available'] is True, name
    sched = metrics['simulators']['scheduler']
    if 'swe_agent' in name:
        assert sched['replay_kind'] == 'agent_dag', name
        assert sched['dag_scheduling_available'] is True, name
    else:
        assert sched['replay_kind'] == 'prompt_or_domain_locality', name
        assert sched['dag_scheduling_available'] is False, name
    assert sched['dependency_violations'] == 0, name

print('real artifact validation passed for', len(required), 'dataset/model pairs')
PY
```

Expected:

```text
real artifact validation passed for 7 dataset/model pairs
```

## Performance/Optimization History

The first full analysis attempts were too slow, especially on Qwen3 ShareGPT.
Those logs are intentionally preserved:

```text
logs/experiments/20260713_qwen3_sharegpt_analysis.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v2.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v3.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v4.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v5.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v6.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v7.log
logs/experiments/20260713_qwen3_sharegpt_analysis_optimized_v8.log
```

Key optimizations added:

- Batched vLLM generation with `--batch-size`.
- Cached selected expert arrays per trace record.
- Vectorized many `np.unique`/histogram computations.
- Added segment/layer count cache.
- Avoided legacy request-level predictor evaluation in locality summary.
- Added streaming `ScoreAccumulator` for segment predictor metrics.
- Cached RouteSig signatures and scheduler candidate demand.
- Added timing logs inside report generation and predictor loops.

These optimizations matter because future runs at 1k/5k/10k records will
otherwise become impractical.

## Known Issues and Caveats

### SWE-agent evidence is structurally valid but weak as a predictor result

SWE-agent now supports agent-DAG replay correctly, but RouteSig does not beat
temporal/global baselines:

- Qwen3 SWE-agent 2x hit rate:
  - global: 0.441582
  - temporal: 0.443449
  - RouteSig: 0.439890
- DeepSeek SWE-agent 2x hit rate:
  - global: 0.369003
  - temporal: 0.373778
  - RouteSig: 0.345094

This means the central future question is not whether the pipeline runs; it is
whether richer agent metadata and stronger predictors can beat temporal
locality in real agent workloads.

### Dataset size is still small

Each dataset/model pair uses 256 prompt records. This is enough for pipeline
validation, but not enough for a strong paper claim.

Need next:

- 1k, 5k, 10k scale runs.
- More SWE-agent trajectory groups.
- Confidence intervals.
- Multiple time/group splits.
- Ablations for metadata leakage.

### Scheduler replay objective is not mature

Replay is legal and prediction-based, but fanout improvement is not stable.
The current score is a first systems proxy, not a final scheduling objective.

Need next:

- Better all-to-all communication proxy.
- Better token density/load imbalance proxy.
- Better critical-path delay handling.
- Confidence-aware fallback decisions.
- RouteSig/temporal hybrid scheduling.

### Router scores are unavailable

Current vLLM routed-experts output provides selected expert IDs. Router weights
or probabilities are not captured, so `weighted_coverage` is marked unavailable
with:

```text
unavailable_router_scores_not_captured_by_vllm
```

This is acceptable for the current stage. Capturing router scores inside fused
vLLM kernels is future work.

### Do not revive Transformers as a main backend

Transformers router-logit capture was explicitly removed from the active
experiment path. It may be useful for debugging or tiny fixtures, but it cannot
count as completion evidence.

## Next Research Stage

Recommended order:

### 1. Improve predictors before modifying vLLM runtime

The current weakness is predictor strength, not system integration.

Concrete ideas:

- `RouteSig + temporal_window_frequency` hybrid predictor.
- Per-layer fallback weighting.
- Smoothing for sparse metadata keys.
- Confidence calibration by block type/layer.
- Predict expert distributions, not only top-M sets.
- Add a lightweight learned ranker:
  - logistic regression
  - GBDT
  - small MLP
- Keep all inputs pre-router.

Primary research question:

Does agent/prompt metadata add information beyond temporal locality?

### 2. Add richer SWE-agent metadata

Current SWE metadata is too coarse. Add fields such as:

- trajectory phase:
  - issue understanding
  - edit
  - test
  - debug
  - finalize
- tool type:
  - read file
  - search
  - edit
  - run test
  - inspect error
- event outcome:
  - test failed
  - patch applied
  - error observed
- repository/language/file type
- DAG depth and critical-path position
- source-group-local step index
- prompt segment role:
  - system
  - code context
  - trajectory event
  - tool result

The next SWE adapter revision should produce richer typed blocks and richer
metadata, not just more records.

### 3. Scale real traces

Recommended scale milestones:

- 1k records per dataset/model.
- 5k records for Qwen3 on the most informative datasets.
- 10k records for final statistics if runtime is acceptable.
- Larger SWE-agent group count.

Keep:

- group-preserving split
- time/source-index order
- predict-before-update evaluation
- no request-level random split

### 4. Strengthen scheduler replay objective

Target a constrained objective:

Under DAG legality and delay bounds, improve expert overlap/token density and
reduce communication burstiness.

Metrics to add or refine:

- active expert fanout per batch/layer
- per-expert token density
- load imbalance
- predicted-vs-actual fanout error
- all-to-all proxy
- critical path delay
- mean/p95 waiting
- low-confidence gated fraction
- oracle gap

### 5. Add ablations

Needed for a credible research story:

- remove role
- remove phase
- remove block type
- remove segment position
- remove source group position
- prompt length only
- temporal only
- RouteSig only
- RouteSig plus temporal
- per-layer RouteSig vs shared RouteSig
- agent datasets vs non-agent datasets

The goal is to prove the signal is not just prompt length, source order, or
dataset identity leakage.

### 6. Runtime integration only after predictor/replay improve

Next-stage runtime items:

- online expert prefetch/offload
- vLLM scheduler modification
- proactive EPLB replica placement
- expert residency policy
- DeepEP/all-to-all-aware batching

Do not start here unless the predictor and replay evidence are stronger.
Runtime integration adds complexity and should consume a validated signal.

## Suggested Next Commands

If the goal is archival:

```bash
openspec validate add-prediction-replay-moe-download --strict
```

Then archive with the appropriate OpenSpec archive command.

If the goal is next research iteration:

1. Create a new OpenSpec change for stronger predictor and SWE metadata.
2. Keep this completed change as the real-vLLM baseline.
3. Start from `tokenmoe/prediction.py`, `tokenmoe/routesig.py`,
   `tokenmoe/metrics.py`, `tokenmoe/simulators.py`, and
   `dataset_adapters/swe_agent.py`.
4. Use the existing 7 trace/report pairs as regression baselines.

## File Checklist for Future Review

Implementation files most likely relevant:

```text
dataset_adapters/
scripts/collect_traces.py
scripts/analyze_traces.py
tokenmoe/schema.py
tokenmoe/trace.py
tokenmoe/vllm_sidecar.py
tokenmoe/prediction.py
tokenmoe/routesig.py
tokenmoe/metrics.py
tokenmoe/simulators.py
tokenmoe/reporting.py
tests/test_current_stage.py
```

Documentation files:

```text
README.md
docs/datasets.md
docs/models.md
docs/research.md
docs/trace_format.md
docs/handoff_add_prediction_replay_moe_download.md
logs/experiments/20260713_real_vllm_validation_summary.md
```

OpenSpec files:

```text
openspec/changes/add-prediction-replay-moe-download/proposal.md
openspec/changes/add-prediction-replay-moe-download/design.md
openspec/changes/add-prediction-replay-moe-download/tasks.md
openspec/changes/add-prediction-replay-moe-download/specs/
```

## Final Verification Snapshot

Last verified in this handoff state:

```text
PYTHONPATH=. pytest -q tests
  12 passed

openspec validate add-prediction-replay-moe-download --strict
  Change 'add-prediction-replay-moe-download' is valid

git diff --check
  no output

real artifact validation
  passed for 7 dataset/model pairs
```

The repo worktree was not clean at the time of this document; implementation
changes were present but not committed.
