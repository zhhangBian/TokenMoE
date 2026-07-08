#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="${WORKLOAD:-data/workloads/agent_workloads.jsonl}"
TRACE_OUTPUT="${TRACE_OUTPUT:-data/traces/tokenmoe_traces.jsonl}"
PARQUET_OUTPUT="${PARQUET_OUTPUT:-data/traces/tokenmoe_traces.parquet}"
MODEL="${MODEL:-TitanML/tiny-mixtral}"
LIMIT="${LIMIT:-80}"
DEVICE="${DEVICE:-cuda:0}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

python scripts/download_workloads.py --output "${WORKLOAD}" --per-workflow "${PER_WORKFLOW:-12}"
python scripts/collect_traces.py \
  --workload "${WORKLOAD}" \
  --output "${TRACE_OUTPUT}" \
  --parquet-output "${PARQUET_OUTPUT}" \
  --backend transformers \
  --model "${MODEL}" \
  --limit "${LIMIT}" \
  --device "${DEVICE}" \
  --allow-fallback
python scripts/analyze_traces.py --traces "${TRACE_OUTPUT}" --workload "${WORKLOAD}"
