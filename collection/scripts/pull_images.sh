#!/usr/bin/env bash
set -euo pipefail
harness_python="${TOKENMOE_HARNESS_VENV:-/home/youwei/bzh/venvs/tokenmoe-harness}/bin/python"
set -x
exec "$harness_python" -m tokenmoe_collect pull-images --config "${1:?run config required}"
