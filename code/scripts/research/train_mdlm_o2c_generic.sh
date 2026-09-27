#!/usr/bin/env bash
set -euo pipefail
set -f

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-${OUTPUT_ROOT:-$PROJECT_ROOT/../outputs}}"
ENV_DIR="${ENV_DIR:-$SCRATCH_ROOT/envs/dlm}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

VISIBLE_DEVICES="${VISIBLE_DEVICES:-${GPU_ID:-0}}"
RUN_NAME="${RUN_NAME:-mdlm_tiny_o2c}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_ROOT/dlm_asym/runs}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_NAME}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH_ROOT/dlm_asym/cache}"
SHARED_HF_HOME="${SHARED_HF_HOME:-$SCRATCH_ROOT/cache/huggingface}"
HF_HOME="${HF_HOME:-$SHARED_HF_HOME}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
WANDB_DIR="${WANDB_DIR:-$SCRATCH_ROOT/dlm_asym/wandb}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/mpl}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/cache/xdg}"
TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-$TORCH_HOME/kernels}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$XDG_CACHE_HOME/nv/ComputeCache}"

DATA_CONFIG="${DATA_CONFIG:-openwebtext-split}"
MODEL="${MODEL:-tiny}"
SEQ_LEN="${SEQ_LEN:-1024}"
NUM_NODES="${NUM_NODES:-1}"
TRAINER_DEVICES="${TRAINER_DEVICES:-}"

MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-}"
EVAL_MICRO_BATCH_SIZE="${EVAL_MICRO_BATCH_SIZE:-}"
EVAL_GLOBAL_BATCH_SIZE="${EVAL_GLOBAL_BATCH_SIZE:-16}"

MAX_STEPS="${MAX_STEPS-8000}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
VAL_INTERVAL="${VAL_INTERVAL:-250}"
LOG_INTERVAL="${LOG_INTERVAL:-$VAL_INTERVAL}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-1}"
NUM_SAMPLE_LOG="${NUM_SAMPLE_LOG:-4}"
GEN_PPL_MODEL="${GEN_PPL_MODEL:-gpt2}"
MAIN_SAMPLING_STEPS="${MAIN_SAMPLING_STEPS:-128}"
SAMPLE_STEPS="${SAMPLE_STEPS:-[2,8,32,128]}"
TRAIN_T="${TRAIN_T:-0}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16}"
LOADER_WORKERS="${LOADER_WORKERS:-8}"
PREPROCESS_NUM_PROC="${PREPROCESS_NUM_PROC:-$LOADER_WORKERS}"
TRAIN_MAX_EXAMPLES="${TRAIN_MAX_EXAMPLES:-}"
VALID_MAX_EXAMPLES="${VALID_MAX_EXAMPLES:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-}"
VALID_SPLIT="${VALID_SPLIT:-}"
DATA_STREAMING="${DATA_STREAMING:-}"
OPTIM_NAME="${OPTIM_NAME:-adamw}"

SENSITIVITY_ENABLED="${SENSITIVITY_ENABLED:-1}"
SENSITIVITY_NUM_BATCHES="${SENSITIVITY_NUM_BATCHES:-1}"
SENSITIVITY_TIME_POINTS="${SENSITIVITY_TIME_POINTS:-[0.15,0.35,0.55]}"
SENSITIVITY_DELTA_T="${SENSITIVITY_DELTA_T:-0.05}"
SENSITIVITY_MAX_MASKED_POSITIONS="${SENSITIVITY_MAX_MASKED_POSITIONS:-256}"
SENSITIVITY_PATHWISE_ENABLED="${SENSITIVITY_PATHWISE_ENABLED:-1}"
SENSITIVITY_FLIP_ENABLED="${SENSITIVITY_FLIP_ENABLED:-1}"
SENSITIVITY_FLIP_FRACTION="${SENSITIVITY_FLIP_FRACTION:-0.02}"
SENSITIVITY_MAX_FLIP_POSITIONS="${SENSITIVITY_MAX_FLIP_POSITIONS:-16}"
SENSITIVITY_LOG_PER_TIMEPOINT="${SENSITIVITY_LOG_PER_TIMEPOINT:-0}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

if [ -z "$TRAINER_DEVICES" ]; then
  if [ -n "$VISIBLE_DEVICES" ]; then
    TRAINER_DEVICES="$(awk -F, '{print NF}' <<< "$VISIBLE_DEVICES")"
  else
    TRAINER_DEVICES=1
  fi
fi

if [ -n "$MICRO_BATCH_SIZE" ]; then
  if [ -z "$ACCUMULATE_GRAD_BATCHES" ]; then
    ACCUMULATE_GRAD_BATCHES=1
  fi
  GLOBAL_BATCH_SIZE="$((MICRO_BATCH_SIZE * TRAINER_DEVICES * NUM_NODES * ACCUMULATE_GRAD_BATCHES))"
fi

if [ -n "$EVAL_MICRO_BATCH_SIZE" ]; then
  EVAL_GLOBAL_BATCH_SIZE="$((EVAL_MICRO_BATCH_SIZE * TRAINER_DEVICES * NUM_NODES))"
fi

mkdir -p \
  "$RUN_DIR" \
  "$CACHE_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$WANDB_DIR" \
  "$MPLCONFIGDIR" \
  "$XDG_CACHE_HOME" \
  "$TORCH_HOME" \
  "$PYTORCH_KERNEL_CACHE_PATH" \
  "$CUDA_CACHE_PATH"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Expected project python at $PYTHON_BIN but it was not found." >&2
  exit 1
fi

case "$SENSITIVITY_ENABLED" in
  1|true|True|TRUE|yes|YES) sensitivity_enabled_value=true ;;
  *) sensitivity_enabled_value=false ;;
esac

case "$SENSITIVITY_LOG_PER_TIMEPOINT" in
  1|true|True|TRUE|yes|YES) sensitivity_log_per_timepoint=true ;;
  *) sensitivity_log_per_timepoint=false ;;
esac

case "$SENSITIVITY_PATHWISE_ENABLED" in
  1|true|True|TRUE|yes|YES) sensitivity_pathwise_enabled=true ;;
  *) sensitivity_pathwise_enabled=false ;;
esac

case "$SENSITIVITY_FLIP_ENABLED" in
  1|true|True|TRUE|yes|YES) sensitivity_flip_enabled=true ;;
  *) sensitivity_flip_enabled=false ;;
esac

export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export WANDB_DIR
export MPLCONFIGDIR
export XDG_CACHE_HOME
export TORCH_HOME
export PYTORCH_KERNEL_CACHE_PATH
export CUDA_CACHE_PATH
export TOKENIZERS_PARALLELISM=False
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_ROOT"

echo "Launching $RUN_NAME"
echo "  visible_devices: $VISIBLE_DEVICES"
echo "  trainer.devices: $TRAINER_DEVICES"
echo "  trainer.num_nodes: $NUM_NODES"
echo "  model: $MODEL"
echo "  seq_len: $SEQ_LEN"
echo "  optimizer: $OPTIM_NAME"
echo "  train_T: $TRAIN_T"
echo "  global_batch: $GLOBAL_BATCH_SIZE"
if [ -n "$MICRO_BATCH_SIZE" ]; then
  echo "  micro_batch_per_device: $MICRO_BATCH_SIZE"
fi
if [ -n "$ACCUMULATE_GRAD_BATCHES" ]; then
  echo "  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES"
fi
echo "  eval_global_batch: $EVAL_GLOBAL_BATCH_SIZE"
if [ -n "$EVAL_MICRO_BATCH_SIZE" ]; then
  echo "  eval_batch_per_device: $EVAL_MICRO_BATCH_SIZE"
fi
echo "  run_dir: $RUN_DIR"

cmd=(
  "$PYTHON_BIN" -u main.py
  "algo=mdlm"
  "model=$MODEL"
  "model.length=$SEQ_LEN"
  "data=$DATA_CONFIG"
  "data.cache_dir=$CACHE_ROOT"
  "++data.preprocessing_num_proc=$PREPROCESS_NUM_PROC"
  "trainer.devices=$TRAINER_DEVICES"
  "trainer.num_nodes=$NUM_NODES"
  "trainer.precision=$TRAINER_PRECISION"
  "trainer.num_sanity_val_steps=$NUM_SANITY_VAL_STEPS"
  "trainer.val_check_interval=$VAL_INTERVAL"
  "trainer.log_every_n_steps=$LOG_INTERVAL"
  "trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES"
  "trainer.limit_val_batches=$LIMIT_VAL_BATCHES"
  "loader.global_batch_size=$GLOBAL_BATCH_SIZE"
  "loader.eval_global_batch_size=$EVAL_GLOBAL_BATCH_SIZE"
  "loader.num_workers=$LOADER_WORKERS"
  "callbacks.checkpoint_every_n_steps.every_n_train_steps=$CHECKPOINT_EVERY"
  "checkpointing.save_dir=$RUN_DIR"
  "checkpointing.resume_from_ckpt=True"
  "checkpointing.resume_ckpt_path=$RUN_DIR/checkpoints/last.ckpt"
  "hydra.run.dir=$RUN_DIR"
  "sampling.predictor=ancestral_cache"
  "sampling.steps=$MAIN_SAMPLING_STEPS"
  "sampling.num_sample_batches=$NUM_SAMPLE_BATCHES"
  "sampling.num_sample_log=$NUM_SAMPLE_LOG"
  "sampling.use_float64=False"
  "eval.compute_generative_perplexity=True"
  "eval.gen_ppl_eval_model_name_or_path=$GEN_PPL_MODEL"
  "eval.sample_eval_steps=$SAMPLE_STEPS"
  "eval.sensitivity.enabled=$sensitivity_enabled_value"
  "eval.sensitivity.num_batches=$SENSITIVITY_NUM_BATCHES"
  "eval.sensitivity.time_points=$SENSITIVITY_TIME_POINTS"
  "eval.sensitivity.delta_t=$SENSITIVITY_DELTA_T"
  "eval.sensitivity.max_masked_positions=$SENSITIVITY_MAX_MASKED_POSITIONS"
  "eval.sensitivity.pathwise_enabled=$sensitivity_pathwise_enabled"
  "eval.sensitivity.flip_enabled=$sensitivity_flip_enabled"
  "eval.sensitivity.flip_fraction=$SENSITIVITY_FLIP_FRACTION"
  "eval.sensitivity.max_flip_positions=$SENSITIVITY_MAX_FLIP_POSITIONS"
  "eval.sensitivity.log_per_timepoint=$sensitivity_log_per_timepoint"
  "optim.name=$OPTIM_NAME"
  "algo.T=$TRAIN_T"
  "wandb.name=$RUN_NAME"
  "+wandb.offline=True"
)

if [ -n "$MAX_STEPS" ]; then
  cmd+=("trainer.max_steps=$MAX_STEPS")
fi

if [ -n "$MAX_EPOCHS" ]; then
  cmd+=("trainer.max_epochs=$MAX_EPOCHS")
fi

if [ -n "$MICRO_BATCH_SIZE" ]; then
  cmd+=("loader.batch_size=$MICRO_BATCH_SIZE")
fi

if [ -n "$ACCUMULATE_GRAD_BATCHES" ]; then
  cmd+=("trainer.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES")
fi

if [ -n "$EVAL_MICRO_BATCH_SIZE" ]; then
  cmd+=("loader.eval_batch_size=$EVAL_MICRO_BATCH_SIZE")
fi

if [ -n "$TRAIN_MAX_EXAMPLES" ]; then
  cmd+=("++data.train_max_examples=$TRAIN_MAX_EXAMPLES")
fi

if [ -n "$VALID_MAX_EXAMPLES" ]; then
  cmd+=("++data.valid_max_examples=$VALID_MAX_EXAMPLES")
fi

if [ -n "$TRAIN_SPLIT" ]; then
  cmd+=("++data.train_split=$TRAIN_SPLIT")
fi

if [ -n "$VALID_SPLIT" ]; then
  cmd+=("++data.valid_split=$VALID_SPLIT")
fi

if [ -n "$DATA_STREAMING" ]; then
  cmd+=("data.streaming=$DATA_STREAMING")
fi

if [ -n "$EXTRA_OVERRIDES" ]; then
  read -r -a extra_override_args <<< "$EXTRA_OVERRIDES"
  cmd+=("${extra_override_args[@]}")
fi

"${cmd[@]}"
