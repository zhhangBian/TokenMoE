#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1

TOTAL="${TOKENMOE_TOTAL:-216}"
MAX_CALLS_PER_PROCESS="${TOKENMOE_MAX_CALLS_PER_PROCESS:-48}"
BATCH_SIZES_CSV="${TOKENMOE_BATCH_SIZES:-1,2,4,8}"
MODES_CSV="${TOKENMOE_MODES:-baseline,tokenmoe_sidecar}"
STAMP="${TOKENMOE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${TOKENMOE_OUTPUT_DIR:-${REPO_ROOT}/data/vllm_qwen}"
ANALYSIS_DIR="${TOKENMOE_ANALYSIS_DIR:-${REPO_ROOT}/analysis/vllm_qwen}"
RESUME="${TOKENMOE_RESUME:-0}"
WAIT_FOR_GPU="${TOKENMOE_WAIT_FOR_GPU:-1}"
MIN_FREE_MB="${TOKENMOE_MIN_FREE_MB:-24000}"
STABLE_CHECKS="${TOKENMOE_GPU_STABLE_CHECKS:-3}"
GPU_INDEX="${CUDA_VISIBLE_DEVICES%%,*}"

IFS=',' read -r -a BATCH_SIZES <<< "${BATCH_SIZES_CSV}"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"

mkdir -p "${OUTPUT_DIR}" "${ANALYSIS_DIR}"
if [[ "${RESUME}" != "1" ]]; then
  rm -f "${OUTPUT_DIR}/generations.jsonl" \
        "${OUTPUT_DIR}/sidecar_decisions.jsonl" \
        "${OUTPUT_DIR}/latency.csv"
fi

wait_for_gpu() {
  if [[ "${WAIT_FOR_GPU}" != "1" ]]; then
    return 0
  fi
  stable=0
  while true; do
    free_mb="$(nvidia-smi --id="${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      stable=$((stable + 1))
      if (( stable >= STABLE_CHECKS )); then
        return 0
      fi
      echo "GPU ${GPU_INDEX} free_mb=${free_mb}; stable_check=${stable}/${STABLE_CHECKS}" >&2
      sleep 10
      continue
    fi
    stable=0
    echo "waiting for GPU ${GPU_INDEX}: free_mb=${free_mb}, need=${MIN_FREE_MB}" >&2
    sleep 30
  done
}

cd /tmp

for batch_size in "${BATCH_SIZES[@]}"; do
  chunk=$((batch_size * MAX_CALLS_PER_PROCESS))
  if (( chunk < batch_size )); then
    chunk="${batch_size}"
  fi
  for mode in "${MODES[@]}"; do
    offset=0
    while (( offset < TOTAL )); do
      limit="${chunk}"
      if (( offset + limit > TOTAL )); then
        limit=$((TOTAL - offset))
      fi
      summary_file="${ANALYSIS_DIR}/shards/summary_offset_${offset}_limit_${limit}_batches_${batch_size}_modes_${mode}.json"
      if [[ "${RESUME}" == "1" && -s "${summary_file}" ]]; then
        echo "skip existing shard batch_size=${batch_size} mode=${mode} offset=${offset} limit=${limit}"
        offset=$((offset + limit))
        continue
      fi
      shard_log="${LOG_DIR}/vllm_qwen_${STAMP}_b${batch_size}_${mode}_off${offset}.log"
      echo "shard batch_size=${batch_size} mode=${mode} offset=${offset} limit=${limit}" | tee "${shard_log}"
      wait_for_gpu
      conda run -n tokenmoe python -u "${REPO_ROOT}/scripts/run_vllm_qwen_experiment.py" \
        --offset "${offset}" \
        --limit "${limit}" \
        --batch-sizes "${batch_size}" \
        --modes "${mode}" \
        --append \
        --output-dir "${OUTPUT_DIR}" \
        --analysis-dir "${ANALYSIS_DIR}" \
        "$@" 2>&1 | tee -a "${shard_log}"
      offset=$((offset + limit))
    done
  done
done

conda run -n tokenmoe python "${REPO_ROOT}/scripts/analyze_vllm_qwen.py" \
  --data-dir "${OUTPUT_DIR}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  --report-dir "${REPO_ROOT}/reports/vllm_qwen" \
  2>&1 | tee "${LOG_DIR}/vllm_qwen_${STAMP}_analysis.log"
