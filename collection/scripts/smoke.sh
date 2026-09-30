#!/usr/bin/env bash
set -euo pipefail
collection_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
harness_python="${TOKENMOE_HARNESS_VENV:-/home/youwei/bzh/venvs/tokenmoe-harness}/bin/python"
model="${1:?model name required}"
trace_dir="${TOKENMOE_TRACE_DIR:?set TOKENMOE_TRACE_DIR to a new experiment directory}"
port="${TOKENMOE_PORT:-8000}"
set -x
mkdir -p "$trace_dir"
if [[ "${TOKENMOE_CONNECT_ONLY:-0}" != 1 ]]; then
  "$collection_dir/scripts/serve.sh" "$model" --port "$port" > "$trace_dir/server.log" 2>&1 &
  server_pid=$!
  trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" || true' EXIT
fi
"$harness_python" -m tokenmoe_collect smoke --model-config "$collection_dir/configs/models/$model.yaml" --trace-dir "$trace_dir" --url "http://127.0.0.1:$port"
