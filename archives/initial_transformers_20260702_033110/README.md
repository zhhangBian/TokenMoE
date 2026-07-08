# TokenMoE vLLM Prototype

TokenMoE is a trace-first prototype for testing whether agent metadata predicts
MoE expert working sets early enough to help serving systems. It does not change
router choices or model outputs.

## Reproduce

```bash
MODEL_SET=small DRY_RUN=1 scripts/download_models.sh
bash scripts/run_trace_collection.sh
PYTHONPATH=. python scripts/validate_routesig.py --traces data/traces/tokenmoe_traces.jsonl
pytest -q tests
```

Default reproduction uses `TitanML/tiny-mixtral` through Transformers router
logits. On this machine the run collected 80 real router traces with
`fallback_used=false`.

Outputs:

- `data/workloads/agent_workloads.jsonl`
- `data/traces/tokenmoe_traces.jsonl`
- `data/traces/tokenmoe_traces.parquet`
- `analysis/locality_report.md`
- `analysis/figures/*.png`
- `reports/tokenmoe_report.md`

## vLLM Path

The repository checkout already contains routed expert capture support in vLLM:

- `BaseRouter.select_experts()` captures logical top-k IDs before EPLB mapping.
- `RoutedExpertsCapturer` stores per-token/per-layer selected experts.
- `CompletionOutput.routed_experts` exposes the final request array.

Use this when a complete vLLM install and compiled extensions are available:

```bash
PYTHONPATH=. python scripts/collect_traces.py \
  --backend vllm \
  --model TitanML/tiny-mixtral \
  --workload data/workloads/agent_workloads.jsonl \
  --output data/traces/vllm_traces.jsonl \
  --allow-fallback
```

Known local gap: this Python environment does not have the full vLLM dependency
set or compiled `vllm._C` extension installed, so the completed experiment used
the real Transformers Mixtral router-logit path.

## Reports

Open `reports/tokenmoe_report.md` for the final illustrated report. It includes
role/expert heatmaps, RouteSig-vs-baseline top-M hit rates, layer sensitivity,
and replay results for prefetch, scheduling, and EPLB.
