#!/usr/bin/env bash
set -euo pipefail
collection_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
harness_python="${TOKENMOE_HARNESS_VENV:-/home/youwei/bzh/venvs/tokenmoe-harness}/bin/python"
config="${1:-$collection_dir/configs/runs/pilot.yaml}"
set -x
exec "$harness_python" -m tokenmoe_collect pilot --config "$config" \
  --equivalence-record "${TOKENMOE_EQUIVALENCE_RECORD:?set a passing equivalence JSON}" \
  --port "${TOKENMOE_PORT:-8000}"
