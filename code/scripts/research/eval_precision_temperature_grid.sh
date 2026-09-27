#!/usr/bin/env bash
# Precision x temperature x configured-update grid for MDLM.
#
# Variables (with defaults):
#   CHECKPOINT       (required) absolute .ckpt path
#   OUT_DIR          (required) output dir
#   GPU_ID           CUDA device (default 0)
#   MODEL_SIZE       small / tiny (default small)
#   BUDGETS          space-separated M (default "2 4 8 16 32 64 128 256 512 1024")
#   TEMPERATURES     space-separated tau (default "0.85 0.9 1.0")
#   PRECISIONS       space-separated precision tags (default "fp32 fp64")
#                    fp32 -> gumbel_precision=float32 use_float64=False
#                    fp64 -> gumbel_precision=float64 use_float64=True
#   EVAL_BATCH_SIZE  (default 4)
#   NUM_SAMPLE_BATCHES (default 32) -> total_samples = batch * num_batches = 128
#   SEED             generation seed (default 1)
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-${OUTPUT_ROOT:-$PACKAGE_ROOT/outputs}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-$SCRATCH_ROOT/dlm_asym/cache}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a .ckpt path}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
GPU_ID="${GPU_ID:-0}"
MODEL_SIZE="${MODEL_SIZE:-small}"
SEQ_LEN="${SEQ_LEN:-1024}"

BUDGETS="${BUDGETS:-2 4 8 16 32 64 128 256 512 1024}"
TEMPERATURES="${TEMPERATURES:-0.85 0.9 1.0}"
PRECISIONS="${PRECISIONS:-fp32 fp64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-32}"
SEED="${SEED:-1}"
GEN_PPL_MODEL="${GEN_PPL_MODEL:-gpt2}"
REVEAL_STRATEGY="${REVEAL_STRATEGY:-independent}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(realpath "$OUT_DIR")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=False
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ckpt_stem="$(basename "${CHECKPOINT%.ckpt}")"

for prec in $PRECISIONS; do
  case "$prec" in
    fp32) gumbel="float32"; use64="False" ;;
    fp64) gumbel="float64"; use64="True"  ;;
    *) echo "unknown PRECISION=$prec"; exit 1 ;;
  esac
  for budget in $BUDGETS; do
    for temp in $TEMPERATURES; do
      temp_tag="${temp//./p}"
      out_file="$OUT_DIR/${ckpt_stem}_s${budget}_temp${temp_tag}_${prec}.json"
      hydra_dir="$OUT_DIR/hydra_${ckpt_stem}_s${budget}_temp${temp_tag}_${prec}"
      if [ -f "$out_file" ]; then
        echo "[skip] $out_file"
        continue
      fi
      echo "[$(date '+%H:%M:%S')] ckpt=$ckpt_stem M=$budget temp=$temp prec=$prec gpu=$GPU_ID"
      "$PYTHON_BIN" -u "$PACKAGE_ROOT/code/main.py" \
        mode=sample_eval \
        seed="$SEED" \
        algo=mdlm \
        model="$MODEL_SIZE" \
        model.length="$SEQ_LEN" \
        data=openwebtext-split \
        data.cache_dir="$CACHE_ROOT" \
        loader.batch_size="$EVAL_BATCH_SIZE" \
        loader.eval_batch_size="$EVAL_BATCH_SIZE" \
        sampling.predictor=ancestral_cache \
        sampling.steps="$budget" \
        sampling.x0_temperature="$temp" \
        sampling.x0_temperature_schedule=constant \
        sampling.ancestral_reveal_strategy="$REVEAL_STRATEGY" \
        sampling.num_sample_batches="$NUM_SAMPLE_BATCHES" \
        sampling.gumbel_precision="$gumbel" \
        sampling.use_float64="$use64" \
        eval.compute_generative_perplexity=True \
        eval.gen_ppl_eval_model_name_or_path="$GEN_PPL_MODEL" \
        eval.checkpoint_path="$CHECKPOINT" \
        eval.generated_samples_path="$out_file" \
        hydra.run.dir="$hydra_dir" \
        wandb=null
    done
  done
done
