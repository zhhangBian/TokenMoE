#!/usr/bin/env bash
set -euo pipefail

MODEL_SET="${MODEL_SET:-small}"
CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"
DRY_RUN="${DRY_RUN:-0}"
HF_TOKEN_ARG=()

if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARG=(--token "${HF_TOKEN}")
fi

case "${MODEL_SET}" in
  small)
    MODELS=(
      "TitanML/tiny-mixtral"
    )
    ;;
  target)
    MODELS=(
      "mistralai/Mixtral-8x7B-Instruct-v0.1"
      "Qwen/Qwen3-30B-A3B"
      "deepseek-ai/DeepSeek-V2-Lite-Chat"
    )
    ;;
  trace-only)
    MODELS=()
    ;;
  all)
    MODELS=(
      "TitanML/tiny-mixtral"
      "mistralai/Mixtral-8x7B-Instruct-v0.1"
      "Qwen/Qwen3-30B-A3B"
      "Qwen/Qwen3-235B-A22B"
      "deepseek-ai/DeepSeek-V2-Lite-Chat"
    )
    ;;
  *)
    echo "unknown MODEL_SET=${MODEL_SET}; expected small, target, trace-only, or all" >&2
    exit 2
    ;;
esac

echo "MODEL_SET=${MODEL_SET}"
echo "CACHE_DIR=${CACHE_DIR}"
if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "trace-only selected: no model files are required"
  exit 0
fi

for model in "${MODELS[@]}"; do
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: huggingface-cli download ${model} --cache-dir ${CACHE_DIR}"
  else
    huggingface-cli download "${model}" --cache-dir "${CACHE_DIR}" "${HF_TOKEN_ARG[@]}"
  fi
done
