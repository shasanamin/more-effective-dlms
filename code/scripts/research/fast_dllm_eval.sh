#!/usr/bin/env bash

# Scratch-safe Fast-dLLM evaluation wrapper.
#
# Positional args:
#   model: llada | dream
#   task: gsm8k | humaneval
#   mode: baseline | cache | parallel | cache_parallel | dual_cache_parallel
#   length: generation length, default 256
#   block_length: block length, default 32
#
# Common environment overrides:
#   CUDA_VISIBLE_DEVICES=3
#   MODEL_PATH=...
#   NUM_FEWSHOT=...
#   BATCH_SIZE=1
#   THRESHOLD=0.9
#   TEMPERATURE=0.0
#   SAMPLING_DTYPE=default | float64 | float32
#   GENERATION_SEED=0
#   LIMIT=10
#   RUN_NAME=my_run
#   OUTPUT_ROOT=/path/to/large-storage/fast_dllm/evals
#   EXTRA_MODEL_ARGS=...
#   EXTRA_LM_EVAL_ARGS=...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/fast_dllm_env.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {llada|dream} {gsm8k|humaneval} {baseline|cache|parallel|cache_parallel|dual_cache_parallel} [length] [block_length]" >&2
  exit 2
fi

MODEL="$1"
TASK="${2:-gsm8k}"
MODE="${3:-baseline}"
LENGTH="${4:-256}"
BLOCK_LENGTH="${5:-32}"

THRESHOLD="${THRESHOLD:-0.9}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SAMPLING_DTYPE="${SAMPLING_DTYPE:-default}"
GENERATION_SEED="${GENERATION_SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MONITOR_GPU="${MONITOR_GPU:-1}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FAST_DLLM_SCRATCH}/evals}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-${MODEL}_${TASK}_${MODE}_len${LENGTH}_block${BLOCK_LENGTH}_${DATE_TAG}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_DIR="${OUTPUT_DIR}/generated"
STEPS_PARALLEL=$(( (LENGTH + BLOCK_LENGTH - 1) / BLOCK_LENGTH ))

mkdir -p "${OUTPUT_DIR}" "${SAVE_DIR}"

case "${TASK}" in
  gsm8k)
    NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
    ;;
  humaneval)
    NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
    ;;
  *)
    echo "Unsupported task: ${TASK}" >&2
    exit 2
    ;;
esac

case "${MODEL}" in
  llada)
    MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
    WORKDIR="${PROJECT_ROOT}/fast-dllm/llada"
    ENTRYPOINT="eval_llada.py"
    MODEL_NAME="llada_dist"
    MODEL_ARGS="model_path=${MODEL_PATH},gen_length=${LENGTH},block_length=${BLOCK_LENGTH},temperature=${TEMPERATURE},sampling_dtype=${SAMPLING_DTYPE},generation_seed=${GENERATION_SEED},show_speed=True,save_dir=${SAVE_DIR}"
    case "${MODE}" in
      baseline)
        MODEL_ARGS="${MODEL_ARGS},steps=${LENGTH}"
        ;;
      cache)
        MODEL_ARGS="${MODEL_ARGS},steps=${LENGTH},use_cache=True"
        ;;
      parallel)
        MODEL_ARGS="${MODEL_ARGS},steps=${STEPS_PARALLEL},threshold=${THRESHOLD}"
        ;;
      cache_parallel)
        MODEL_ARGS="${MODEL_ARGS},steps=${STEPS_PARALLEL},use_cache=True,threshold=${THRESHOLD}"
        ;;
      dual_cache_parallel)
        MODEL_ARGS="${MODEL_ARGS},steps=${STEPS_PARALLEL},use_cache=True,dual_cache=True,threshold=${THRESHOLD}"
        ;;
      *)
        echo "Unsupported mode: ${MODE}" >&2
        exit 2
        ;;
    esac
    ;;
  dream)
    MODEL_PATH="${MODEL_PATH:-Dream-org/Dream-v0-Base-7B}"
    WORKDIR="${PROJECT_ROOT}/fast-dllm/dream"
    ENTRYPOINT="eval.py"
    MODEL_NAME="dream"
    MODEL_ARGS="pretrained=${MODEL_PATH},max_new_tokens=${LENGTH},add_bos_token=true,temperature=${TEMPERATURE},sampling_dtype=${SAMPLING_DTYPE},generation_seed=${GENERATION_SEED},show_speed=True,save_dir=${SAVE_DIR}"
    case "${MODE}" in
      baseline)
        MODEL_ARGS="${MODEL_ARGS},diffusion_steps=${LENGTH},alg=entropy"
        ;;
      cache)
        MODEL_ARGS="${MODEL_ARGS},diffusion_steps=${LENGTH},alg=entropy,use_cache=true"
        ;;
      parallel)
        MODEL_ARGS="${MODEL_ARGS},diffusion_steps=${STEPS_PARALLEL},alg=confidence_threshold,threshold=${THRESHOLD}"
        ;;
      cache_parallel)
        MODEL_ARGS="${MODEL_ARGS},diffusion_steps=${STEPS_PARALLEL},alg=confidence_threshold,threshold=${THRESHOLD},use_cache=true"
        ;;
      dual_cache_parallel)
        MODEL_ARGS="${MODEL_ARGS},diffusion_steps=${STEPS_PARALLEL},alg=confidence_threshold,threshold=${THRESHOLD},use_cache=true,dual_cache=true"
        ;;
      *)
        echo "Unsupported mode: ${MODE}" >&2
        exit 2
        ;;
    esac
    if [[ "${TASK}" == "humaneval" ]]; then
      MODEL_ARGS="${MODEL_ARGS},escape_until=true"
    fi
    ;;
  *)
    echo "Unsupported model: ${MODEL}" >&2
    exit 2
    ;;
esac

if [[ -n "${EXTRA_MODEL_ARGS:-}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

LM_EVAL_ARGS=(
  launch
  --num_processes 1
  "${ENTRYPOINT}"
  --model "${MODEL_NAME}"
  --model_args "${MODEL_ARGS}"
  --tasks "${TASK}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_DIR}/lm_eval"
  --confirm_run_unsafe_code
)

if [[ "${TASK}" == "gsm8k" ]]; then
  LM_EVAL_ARGS+=(--num_fewshot "${NUM_FEWSHOT}")
fi

if [[ "${TASK}" == "humaneval" || "${LOG_SAMPLES:-0}" == "1" ]]; then
  LM_EVAL_ARGS+=(--log_samples)
fi

if [[ -n "${LIMIT:-}" ]]; then
  LM_EVAL_ARGS+=(--limit "${LIMIT}")
fi

if [[ -n "${EXTRA_LM_EVAL_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_LM_EVAL_ARGS} )
  LM_EVAL_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Fast-dLLM eval"
echo "  model:       ${MODEL}"
echo "  task:        ${TASK}"
echo "  mode:        ${MODE}"
echo "  model_path:  ${MODEL_PATH}"
echo "  length:      ${LENGTH}"
echo "  block:       ${BLOCK_LENGTH}"
echo "  temperature: ${TEMPERATURE}"
echo "  sample dtype:${SAMPLING_DTYPE}"
echo "  generation seed: ${GENERATION_SEED}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "  hf_home:     ${HF_HOME}"
echo "  cuda:        ${CUDA_VISIBLE_DEVICES:-all visible GPUs}"
echo

cd "${WORKDIR}"

GPU_MONITOR_ID="${GPU_MONITOR_ID:-${CUDA_VISIBLE_DEVICES:-}}"
GPU_MONITOR_PATH="${OUTPUT_DIR}/gpu_monitor.csv"
MONITOR_PID=""
if [[ "${MONITOR_GPU}" == "1" && -n "${GPU_MONITOR_ID}" ]]; then
  echo "timestamp,index,utilization_gpu_pct,memory_used_mib,power_draw_w" > "${GPU_MONITOR_PATH}"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits \
        -i "${GPU_MONITOR_ID}" >> "${GPU_MONITOR_PATH}" 2>/dev/null
      sleep "${GPU_MONITOR_INTERVAL}"
    done
  ) &
  MONITOR_PID="$!"
fi

START_EPOCH="$(date +%s)"
START_ISO="$(date -Is)"
set +e
"${FAST_DLLM_ENV}/bin/accelerate" "${LM_EVAL_ARGS[@]}"
STATUS="$?"
set -e
END_EPOCH="$(date +%s)"
END_ISO="$(date -Is)"

if [[ -n "${MONITOR_PID}" ]]; then
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
fi

{
  echo "model=${MODEL}"
  echo "task=${TASK}"
  echo "mode=${MODE}"
  echo "model_path=${MODEL_PATH}"
  echo "length=${LENGTH}"
  echo "block_length=${BLOCK_LENGTH}"
  echo "steps_parallel=${STEPS_PARALLEL}"
  echo "threshold=${THRESHOLD}"
  echo "temperature=${TEMPERATURE}"
  echo "sampling_dtype=${SAMPLING_DTYPE}"
  echo "generation_seed=${GENERATION_SEED}"
  echo "batch_size=${BATCH_SIZE}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "start=${START_ISO}"
  echo "end=${END_ISO}"
  echo "wall_seconds=$((END_EPOCH - START_EPOCH))"
  echo "status=${STATUS}"
} > "${OUTPUT_DIR}/run_metadata.txt"

exit "${STATUS}"
