#!/usr/bin/env bash
set -euo pipefail
collection_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
harness_python="${TOKENMOE_HARNESS_VENV:-/home/youwei/bzh/venvs/tokenmoe-harness}/bin/python"
model="${1:?model name required}"
shift
set -x
exec "$harness_python" -m tokenmoe_collect serve --model-config "$collection_dir/configs/models/$model.yaml" --trace-dir "${TOKENMOE_TRACE_DIR:?set TOKENMOE_TRACE_DIR}" "$@"
