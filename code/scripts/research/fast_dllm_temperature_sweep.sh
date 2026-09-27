#!/usr/bin/env bash

# Baseline-decoding temperature sweep for large diffusion LMs.
# This intentionally skips cache and confidence-aware parallel decoding so the
# experiment isolates sampling temperature / precision effects.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=4 bash code/scripts/research/fast_dllm_temperature_sweep.sh llada gsm8k
#   CUDA_VISIBLE_DEVICES=5 bash code/scripts/research/fast_dllm_temperature_sweep.sh dream humaneval
#
# Defaults:
#   - GSM8K uses LIMIT=256 for a fast but meaningful scout.
#   - HumanEval uses the full 164 tasks.
#   - greedy runs once; stochastic float32 cells run three explicit seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/fast_dllm_env.sh"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 llada|dream gsm8k|humaneval" >&2
  exit 2
fi

MODEL="$1"
TASK="$2"

BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
LENGTH_GSM8K="${LENGTH_GSM8K:-256}"
LENGTH_HUMANEVAL="${LENGTH_HUMANEVAL:-256}"
GSM8K_LIMIT="${GSM8K_LIMIT:-256}"
HUMANEVAL_LIMIT="${HUMANEVAL_LIMIT:-}"
RUN_GROUP="${RUN_GROUP:-fast_dllm_temperature_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FAST_DLLM_SCRATCH}/evals/${RUN_GROUP}}"
LOG_ROOT="${LOG_ROOT:-${FAST_DLLM_SCRATCH}/logs/${RUN_GROUP}}"

# Format: label:temperature:sampling_dtype
IFS=',' read -r -a TEMP_SPECS <<< "${TEMP_SPECS:-default_t0:0.0:default,g32_t0p25:0.25:float32,g32_t0p5:0.5:float32,g32_t0p75:0.75:float32,g32_t1p0:1.0:float32,g32_t1p5:1.5:float32,g32_t2p0:2.0:float32}"
IFS=',' read -r -a SEED_VALUES <<< "${GENERATION_SEEDS:-0,1,2}"
FIRST_SEED="${SEED_VALUES[0]}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

case "${TASK}" in
  gsm8k)
    LENGTH="${LENGTH_GSM8K}"
    TASK_LIMIT="${GSM8K_LIMIT}"
    ;;
  humaneval)
    LENGTH="${LENGTH_HUMANEVAL}"
    TASK_LIMIT="${HUMANEVAL_LIMIT}"
    ;;
  *)
    echo "Unsupported task: ${TASK}" >&2
    exit 2
    ;;
esac

echo "Fast-dLLM temperature sweep"
echo "  model:       ${MODEL}"
echo "  task:        ${TASK}"
echo "  run_group:   ${RUN_GROUP}"
echo "  output_root: ${OUTPUT_ROOT}"
echo "  log_root:    ${LOG_ROOT}"
echo "  length:      ${LENGTH}"
echo "  block:       ${BLOCK_LENGTH}"
echo "  limit:       ${TASK_LIMIT:-full}"
echo "  specs:       ${TEMP_SPECS[*]}"
echo "  seeds:       ${SEED_VALUES[*]}"
echo "  cuda:        ${CUDA_VISIBLE_DEVICES:-all visible GPUs}"
echo

for spec in "${TEMP_SPECS[@]}"; do
  IFS=':' read -r LABEL TEMPERATURE SAMPLING_DTYPE <<< "${spec}"
  for seed in "${SEED_VALUES[@]}"; do
    if [[ "${TEMPERATURE}" =~ ^0+([.]0+)?$ ]]; then
      [[ "${seed}" == "${FIRST_SEED}" ]] || continue
    fi

    RUN_NAME="${MODEL}_${TASK}_baseline_${LABEL}_seed${seed}_len${LENGTH}_block${BLOCK_LENGTH}"
    LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"

    echo "[$(date -Is)] starting ${RUN_NAME}"
    if [[ -n "${TASK_LIMIT}" ]]; then
      LIMIT="${TASK_LIMIT}" \
      RUN_NAME="${RUN_NAME}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      TEMPERATURE="${TEMPERATURE}" \
      SAMPLING_DTYPE="${SAMPLING_DTYPE}" \
      GENERATION_SEED="${seed}" \
      bash "${SCRIPT_DIR}/fast_dllm_eval.sh" "${MODEL}" "${TASK}" baseline "${LENGTH}" "${BLOCK_LENGTH}" \
        2>&1 | tee "${LOG_PATH}"
    else
      RUN_NAME="${RUN_NAME}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      TEMPERATURE="${TEMPERATURE}" \
      SAMPLING_DTYPE="${SAMPLING_DTYPE}" \
      GENERATION_SEED="${seed}" \
      bash "${SCRIPT_DIR}/fast_dllm_eval.sh" "${MODEL}" "${TASK}" baseline "${LENGTH}" "${BLOCK_LENGTH}" \
        2>&1 | tee "${LOG_PATH}"
    fi
    echo "[$(date -Is)] finished ${RUN_NAME}"
  done
done
