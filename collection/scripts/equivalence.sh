#!/usr/bin/env bash
set -euo pipefail
collection_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
harness_python="${TOKENMOE_HARNESS_VENV:-/home/youwei/bzh/venvs/tokenmoe-harness}/bin/python"
model="${1:?model name required}"
set -x
exec "$harness_python" -m tokenmoe_collect equiv \
  --model-config "$collection_dir/configs/models/$model.yaml" \
  --dataset-path "${TOKENMOE_SWEBENCH_DIR:-/home/youwei/bzh/dataset/princeton-nlp/SWE-bench_Verified}" \
  --output-dir "${TOKENMOE_EQUIV_DIR:?set TOKENMOE_EQUIV_DIR to a new directory}" \
  --port "${TOKENMOE_PORT:-8001}"
