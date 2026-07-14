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

1k-scale conversion keeps the 256-record baselines by suffixing outputs
(the manifest name is derived from the output filename):

```bash
for name in sharegpt lmsys swe_agent opencode openmath; do
  PYTHONPATH=. python -m dataset_adapters.$name --limit 1000 \
    --output /home/youwei/bzh/project/dataset/tokenmoe_artifacts/workloads/${name}_prompt_workloads_1k.jsonl
done
```

The SWE-agent adapter emits enriched pre-router metadata
(`trajectory_phase`, `tool_type`, `event_outcome`, `dag_depth`,
`group_local_step_index`, `on_critical_path`; heuristic version
`swe_agent.metadata_heuristics.v1` recorded in the manifest). Underivable
fields are marked unavailable rather than defaulted.

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

### Router-score capture (trace schema v3)

The local vLLM fork supports capturing per-token router scores alongside
routed expert IDs behind `enable_return_routed_expert_scores` (CLI:
`--enable-return-routed-expert-scores`). Pass `--router-scores on` to
`collect_traces.py` to emit `tokenmoe.trace.v3` records with aligned
`router_scores` and per-model `router_score_semantics`
(Qwen3-MoE: `softmax_topk_renormalized`; DeepSeek-V2:
`softmax_topk_scaled_unnormalized`). If score capture is unsupported for a
config, collection fails closed to ID-only capture and records an explicit
`router_scores_unavailable_reason`. v2 and v3 traces cannot be mixed in one
analysis run. Add `_1k`/`_5k` suffixes to `--output`/`--env-report` paths so
256-record baselines are preserved, e.g.
`traces/qwen3_sharegpt_1k.jsonl` with `--limit 1000 --router-scores on`.

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

Predictors evaluated online (predict-before-update):

- `global_frequency`, `lru_expert_cache`, `sequence_position_frequency`,
  `temporal_window_frequency`, `routesig` — baselines.
- `routesig_score_weighted` — RouteSig with router-score-mass statistics
  (only when all train traces are v3 with scores).
- `hybrid_routesig_temporal` — per-layer gating on train-split
  delta-vs-global plus confidence-calibrated convex fusion of RouteSig and
  temporal; per-layer gate decisions are persisted in `gate_report`.
- `learned_ranker` — logistic regression over pre-router feature-group
  probabilities; no post-router inputs.
- `oracle_upper_bound` — reported separately, never as a prediction result.

Analysis artifacts (`analysis/locality_metrics.json`) include bootstrap 95%
CIs (`delta_vs` on every predictor row, >=1000 group-level resamples), a
second group-preserving time split (`prediction_second_split`,
train fraction 0.5), the metadata ablation matrix (`metadata_ablations`:
full / -role / -phase / -tool_type / -block_type / -positions / length_only /
temporal_only / routesig_only at the 2x budget), per-layer signal maps
(`per_layer_signal_map`), and `weighted_coverage` computed from v3 router
scores (explicitly marked unavailable for v2 traces).

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
