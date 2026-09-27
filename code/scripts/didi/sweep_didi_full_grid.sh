#!/bin/bash
# Full DiDi-Instruct sampler calibration grid.
#
# Grid:
#   sampler in {ancestral, guided}
#   use_float64 in {true, false}  # true=fp64 categorical sampling, false=fp32
#   tau in {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.10}
#   configured updates M in {1,2,4,8,16,32,64,128,256,512,1024}
#
# Modes:
#   MODE=manifest       write manifest.tsv and gpu*.tsv shards
#   MODE=run GPU=<id>   run one GPU shard sequentially; skips completed JSONs
#   MODE=combine        scan JSONs and write summary/table artifacts
#   MODE=status         combine, then print a compact completion report
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
PY=${PY:-python}
DIDI_DIR=${DIDI_DIR:-${PROJECT_ROOT}/didi-instruct}
CKPT=${CKPT:-${OUTPUT_ROOT}/dlm_asym/runs/didi-instruct.sanitized.ckpt}
OUT_DIR=${OUT_DIR:-${OUTPUT_ROOT}/dlm_asym/runs/didi_full_grid_may01}
EVAL_BS=${EVAL_BS:-8}
NUM_BATCHES=${NUM_BATCHES:-5}
TAUS=${TAUS:-"0.8 0.85 0.9 0.95 1.0 1.05 1.10"}
FPS=${FPS:-"true false"}
SAMPLERS=${SAMPLERS:-"ancestral guided"}
STEPS=${STEPS:-"1 2 4 8 16 32 64 128 256 512 1024"}
GPUS=${GPUS:-"0"}
GUIDED_VARIANT=${GUIDED_VARIANT:-current}
USE_STUDENT_EMA=${USE_STUDENT_EMA:-true}
WAIT_FOR_GPU_FREE=${WAIT_FOR_GPU_FREE:-false}
WAIT_MEM_MIB=${WAIT_MEM_MIB:-12000}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-60}
MODE=${MODE:-manifest}

mkdir -p "$OUT_DIR"

tau_to_str() {
  local t="$1"
  echo "${t//./p}"
}

fp_to_short() {
  case "$1" in
    true|True|TRUE|1) echo "fp64" ;;
    false|False|FALSE|0) echo "fp32" ;;
    *) echo "fp${1}" ;;
  esac
}

json_complete() {
  local json="$1"
  [[ -s "$json" ]] || return 1
  "$PY" - "$json" >/dev/null 2>&1 <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)

for key in ("generative_ppl", "entropy"):
    val = d.get(key)
    if val is None or val == "NA":
        raise SystemExit(1)
PY
}

wait_for_gpu_free() {
  local gpu="$1"
  if [[ "$WAIT_FOR_GPU_FREE" != "true" ]]; then
    return 0
  fi

  echo "[gpu${gpu}][wait] waiting for memory.used <= ${WAIT_MEM_MIB} MiB"
  while true; do
    local line used util
    line=$(nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "$gpu" 2>/dev/null || true)
    used=${line%%,*}
    util=${line##*,}
    used=${used// /}
    util=${util// /}
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= WAIT_MEM_MIB )); then
      echo "[gpu${gpu}][wait] available: memory.used=${used} MiB, util=${util}%"
      return 0
    fi
    echo "[gpu${gpu}][wait] busy: memory.used=${used:-NA} MiB, util=${util:-NA}%; polling again in ${WAIT_POLL_SECONDS}s"
    sleep "$WAIT_POLL_SECONDS"
  done
}

build_manifest() {
  local manifest="$OUT_DIR/manifest.tsv"
  "$PY" - "$manifest" "$OUT_DIR" "$GPUS" "$TAUS" "$FPS" "$SAMPLERS" "$STEPS" \
    "$GUIDED_VARIANT" "$EVAL_BS" "$NUM_BATCHES" <<'PY'
from pathlib import Path
import sys

manifest = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
gpus = sys.argv[3].split()
taus = sys.argv[4].split()
fps = sys.argv[5].split()
samplers = sys.argv[6].split()
steps = sys.argv[7].split()
guided_variant = sys.argv[8]
eval_bs = sys.argv[9]
num_batches = sys.argv[10]

def tau_to_str(t):
    return t.replace(".", "p")

def fp_to_short(fp):
    return "fp64" if fp.lower() in ("true", "1") else "fp32"

def cost(step, sampler):
    # Guided sampling does discriminator gradient tilting / reranking, so use a
    # conservative multiplier to keep high-budget guided jobs spread across GPUs.
    mult = 3.0 if sampler == "guided" else 1.0
    return float(step) * mult

jobs = []
for step in sorted(steps, key=lambda x: int(x), reverse=True):
    for sampler in samplers:
        for fp in fps:
            for tau in taus:
                fp_short = fp_to_short(fp)
                tau_short = tau_to_str(tau)
                job_id = (
                    f"{sampler}_{guided_variant}_ema_"
                    f"t{tau_short}_{fp_short}_s{step}"
                )
                jobs.append({
                    "job_id": job_id,
                    "sampler": sampler,
                    "guided_variant": guided_variant,
                    "use_float64": fp,
                    "precision": fp_short,
                    "temperature": tau,
                    "steps": step,
                    "eval_bs": eval_bs,
                    "num_batches": num_batches,
                    "cost": cost(step, sampler),
                    "json": str(out_dir / f"{job_id}.json"),
                    "log": str(out_dir / "logs" / f"{job_id}.log"),
                })

loads = {gpu: 0.0 for gpu in gpus}
counts = {gpu: 0 for gpu in gpus}
for job in jobs:
    gpu = min(gpus, key=lambda g: (loads[g], counts[g], int(g)))
    job["gpu"] = gpu
    loads[gpu] += job["cost"]
    counts[gpu] += 1

header = [
    "job_id", "gpu", "sampler", "guided_variant", "use_float64",
    "precision", "temperature", "steps", "eval_bs", "num_batches",
    "cost", "json", "log",
]
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "logs").mkdir(parents=True, exist_ok=True)
with manifest.open("w") as f:
    f.write("\t".join(header) + "\n")
    for job in jobs:
        f.write("\t".join(str(job[col]) for col in header) + "\n")

for gpu in gpus:
    with (out_dir / f"gpu{gpu}.tsv").open("w") as f:
        f.write("\t".join(header) + "\n")
        for job in jobs:
            if job["gpu"] == gpu:
                f.write("\t".join(str(job[col]) for col in header) + "\n")

print(f"Manifest written: {manifest} ({len(jobs)} jobs)")
for gpu in gpus:
    print(
        f"  GPU {gpu}: {counts[gpu]} jobs, "
        f"estimated cost {loads[gpu]:.0f}"
    )
PY
}

run_shard() {
  local gpu="$1"
  local shard="$OUT_DIR/gpu${gpu}.tsv"
  if [[ ! -f "$shard" ]]; then
    echo "Shard $shard not found; run MODE=manifest first." >&2
    exit 1
  fi

  mkdir -p "$OUT_DIR/logs"
  wait_for_gpu_free "$gpu"
  cd "$DIDI_DIR"

  tail -n +2 "$shard" | while IFS=$'\t' read -r job_id assigned_gpu sampler guided_variant use_fp64 precision temperature steps eval_bs num_batches cost json log; do
    if [[ -z "${job_id:-}" ]]; then
      continue
    fi
    if json_complete "$json"; then
      echo "[gpu${gpu}][skip] $job_id"
      continue
    fi

    echo "============================================================"
    echo "[gpu${gpu}][run] $job_id"
    echo "  sampler=$sampler guided_variant=$guided_variant precision=$precision tau=$temperature steps=$steps"
    echo "  json=$json"
    echo "  log=$log"
    echo "============================================================"

    set +e
    CUDA_VISIBLE_DEVICES=$gpu \
    HF_HOME=${OUTPUT_ROOT}/cache/huggingface \
    HF_DATASETS_CACHE=${OUTPUT_ROOT}/cache/huggingface/datasets \
    XDG_CACHE_HOME=${OUTPUT_ROOT}/cache/xdg \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HYDRA_FULL_ERROR=1 \
    "$PY" main.py \
      mode=sample_eval \
      loader.batch_size="$eval_bs" \
      loader.eval_batch_size="$eval_bs" \
      model=small \
      algo=didi_instruct \
      eval.checkpoint_path="$CKPT" \
      eval.compute_generative_perplexity=true \
      eval.gen_ppl_eval_model_name_or_path=gpt2-large \
      eval.generated_samples_path="$json" \
      sampling.steps="$steps" \
      sampling.num_sample_batches="$num_batches" \
      sampling.predictor="$sampler" \
      sampling.guided_variant="$guided_variant" \
      sampling.noise_removal=ancestral \
      sampling.use_float64="$use_fp64" \
      sampling.temperature="$temperature" \
      sampling.use_student_ema="$USE_STUDENT_EMA" \
      trainer.devices=1 \
      wandb.offline=true > "$log" 2>&1
    rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
      echo "[gpu${gpu}][error] $job_id failed with exit code $rc; see $log" >&2
      continue
    fi
    if ! json_complete "$json"; then
      echo "[gpu${gpu}][error] $job_id finished but did not write complete metrics; see $log" >&2
      continue
    fi
    echo "[gpu${gpu}][done] $job_id"
  done

  echo "DONE gpu${gpu}"
}

combine_results() {
  local manifest="$OUT_DIR/manifest.tsv"
  if [[ ! -f "$manifest" ]]; then
    echo "Manifest $manifest not found; run MODE=manifest first." >&2
    exit 1
  fi

  "$PY" - "$manifest" "$OUT_DIR" "$TAUS" "$SAMPLERS" "$FPS" "$STEPS" <<'PY'
from collections import defaultdict
from pathlib import Path
import csv
import json
import sys

manifest = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
taus = sys.argv[3].split()
samplers = sys.argv[4].split()
fps = sys.argv[5].split()
steps = sys.argv[6].split()

def fp_to_short(fp):
    return "fp64" if fp.lower() in ("true", "1") else "fp32"

def fmt_cell(row):
    if row is None:
        return "--"
    return f"{row['gen_ppl']:.2f} [{row['entropy']:.2f}]"

rows = []
complete = []
missing = []
with manifest.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for job in reader:
        row = dict(job)
        json_path = Path(job["json"])
        try:
            with json_path.open() as jf:
                data = json.load(jf)
            row["gen_ppl"] = float(data["generative_ppl"])
            row["entropy"] = float(data["entropy"])
            row["status"] = "complete"
            complete.append(row)
        except Exception:
            row["gen_ppl"] = "NA"
            row["entropy"] = "NA"
            row["status"] = "missing"
            missing.append(row)
        rows.append(row)

summary_cols = [
    "job_id", "gpu", "sampler", "guided_variant", "precision",
    "use_float64", "temperature", "steps", "eval_bs", "num_batches",
    "gen_ppl", "entropy", "status", "json", "log",
]
summary = out_dir / "summary.tsv"
with summary.open("w") as f:
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=summary_cols)
    writer.writeheader()
    for row in sorted(
        rows,
        key=lambda r: (
            r["sampler"],
            r["precision"],
            int(r["steps"]),
            float(r["temperature"]),
        ),
    ):
        writer.writerow({col: row[col] for col in summary_cols})

complete_tsv = out_dir / "summary_complete.tsv"
with complete_tsv.open("w") as f:
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=summary_cols)
    writer.writeheader()
    for row in sorted(
        complete,
        key=lambda r: (
            r["sampler"],
            r["precision"],
            int(r["steps"]),
            float(r["temperature"]),
        ),
    ):
        writer.writerow({col: row[col] for col in summary_cols})

missing_tsv = out_dir / "summary_missing.tsv"
with missing_tsv.open("w") as f:
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=summary_cols)
    writer.writeheader()
    for row in sorted(
        missing,
        key=lambda r: (
            int(r["steps"]),
            r["sampler"],
            r["precision"],
            float(r["temperature"]),
        ),
    ):
        writer.writerow({col: row[col] for col in summary_cols})

by_key = {}
for row in complete:
    by_key[
        (
            row["sampler"],
            row["precision"],
            row["temperature"],
            row["steps"],
        )
    ] = row

md = out_dir / "didi_full_grid_table.md"
with md.open("w") as f:
    f.write("# DiDi-Instruct Full Grid\n\n")
    f.write("Each cell is `generative PPL [token-frequency entropy]`.\n\n")
    for sampler in samplers:
        for fp in [fp_to_short(x) for x in fps]:
            f.write(f"## {sampler}, {fp}\n\n")
            f.write("| M | " + " | ".join(f"tau={t}" for t in taus) + " |\n")
            f.write("|---:|" + "|".join("---:" for _ in taus) + "|\n")
            for step in steps:
                cells = [
                    fmt_cell(by_key.get((sampler, fp, tau, step)))
                    for tau in taus
                ]
                f.write(f"| {step} | " + " | ".join(cells) + " |\n")
            f.write("\n")

tex = out_dir / "didi_full_grid_table.tex"
with tex.open("w") as f:
    f.write("% Auto-generated by code/scripts/didi/sweep_didi_full_grid.sh MODE=combine\n")
    f.write("% Each cell is generative PPL [token-frequency entropy].\n")
    f.write("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    f.write("\\caption{DiDi-Instruct full sampler calibration grid. Each cell reports generative perplexity and token-frequency entropy.}\n")
    f.write("\\label{tab:didi-full-grid}\n")
    for sampler in samplers:
        for fp in [fp_to_short(x) for x in fps]:
            title = f"{sampler}, {fp}"
            f.write(f"\\paragraph{{{title}.}}\n")
            colspec = "r" + "r" * len(taus)
            f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
            f.write("$M$ & " + " & ".join(f"$\\tau={t}$" for t in taus) + " \\\\\n")
            f.write("\\midrule\n")
            for step in steps:
                cells = [
                    fmt_cell(by_key.get((sampler, fp, tau, step)))
                    for tau in taus
                ]
                f.write(f"{step} & " + " & ".join(cells) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\n")
    f.write("\\end{table*}\n")

print(f"Summary: {summary} ({len(complete)}/{len(rows)} complete)")
print(f"Complete rows: {complete_tsv}")
print(f"Missing rows:  {missing_tsv}")
print(f"Markdown table: {md}")
print(f"LaTeX table:    {tex}")
PY
}

print_status() {
  combine_results
  "$PY" - "$OUT_DIR/summary.tsv" <<'PY'
from collections import Counter
import csv
import sys

summary = sys.argv[1]
rows = list(csv.DictReader(open(summary), delimiter="\t"))
status = Counter(r["status"] for r in rows)
by_gpu = Counter((r["gpu"], r["status"]) for r in rows)
by_group = Counter((r["sampler"], r["precision"], r["status"]) for r in rows)

print("")
print("Completion:")
print(f"  complete: {status['complete']}")
print(f"  missing:  {status['missing']}")
print(f"  total:    {len(rows)}")
print("")
print("By GPU:")
for gpu in sorted({r["gpu"] for r in rows}, key=int):
    print(
        f"  gpu{gpu}: "
        f"{by_gpu[(gpu, 'complete')]} complete, "
        f"{by_gpu[(gpu, 'missing')]} missing"
    )
print("")
print("By sampler/precision:")
for sampler in sorted({r["sampler"] for r in rows}):
    for precision in sorted({r["precision"] for r in rows}, reverse=True):
        print(
            f"  {sampler:9s} {precision}: "
            f"{by_group[(sampler, precision, 'complete')]} complete, "
            f"{by_group[(sampler, precision, 'missing')]} missing"
        )
PY
}

case "$MODE" in
  manifest)
    build_manifest
    ;;
  run)
    : "${GPU:?GPU env var required for MODE=run}"
    run_shard "$GPU"
    ;;
  combine)
    combine_results
    ;;
  status)
    print_status
    ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
