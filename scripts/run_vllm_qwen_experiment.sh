#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${TOKENMOE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
cd /tmp

conda run -n tokenmoe python -u "${REPO_ROOT}/scripts/run_vllm_qwen_experiment.py" "$@" \
  2>&1 | tee "${LOG_DIR}/vllm_qwen_${STAMP}.log"
