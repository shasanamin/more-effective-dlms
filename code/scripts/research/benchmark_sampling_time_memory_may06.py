#!/usr/bin/env python
"""Benchmark sampling wallclock and CUDA peak memory for fp32/fp64 sampling.

This script measures generation only.  It deliberately does not compute GPT-2
generative perplexity, so the timing is not contaminated by evaluator forward
passes.  Run it separately for the native MDLM codebase and the vendored
DiDi-Instruct codebase to avoid module-name collisions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from _project_paths import SCRATCH_ROOT

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = ROOT / "code"
DIDI_DIR = ROOT / "didi-instruct"
RUNS_DIR = SCRATCH_ROOT / "dlm_asym" / "runs"
DEFAULT_OUT_DIR = RUNS_DIR / "sampling_time_memory_may06"
TINY_LATE_CKPT = (
    RUNS_DIR
    / "mdlm_tiny_o2c_apr20_fullowt_10e_adamw_fresh"
    / "checkpoints"
    / "9-375155.ckpt"
)


OmegaConf.register_new_resolver("cwd", os.getcwd, replace=True)
OmegaConf.register_new_resolver("device_count", torch.cuda.device_count, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "div_up", lambda x, y: (x + y - 1) // y, replace=True)

if hasattr(torch, "set_float32_matmul_precision"):
  torch.set_float32_matmul_precision("high")


def _resolve_local_hf_path(name_or_path: str) -> str:
  """Use a cached snapshot when offline, otherwise preserve the model ID."""
  path = Path(name_or_path)
  if path.exists():
    return str(path)
  model_key = name_or_path.replace("/", "--")
  cache_roots = []
  if os.environ.get("TRANSFORMERS_CACHE"):
    cache_roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
  if os.environ.get("HF_HOME"):
    cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
  for cache_root in cache_roots:
    snapshot_root = cache_root / f"models--{model_key}" / "snapshots"
    if snapshot_root.exists():
      snapshots = sorted(
          (item for item in snapshot_root.iterdir() if item.is_dir()),
          key=lambda item: item.stat().st_mtime,
          reverse=True)
      if snapshots:
        return str(snapshots[0])
  return name_or_path


@dataclass(frozen=True)
class BenchSpec:
  label: str
  display_name: str
  codebase: str
  checkpoint: Path
  model_size: str
  algo_name: str
  predictor: str
  use_float64: bool
  gumbel_precision: str = "float32"
  x0_temperature: float = 1.0
  didi_temperature: float = 1.0
  didi_use_student_ema: bool = True
  didi_guided_variant: str = "current"


def _json_complete(path: Path) -> bool:
  if not path.exists() or path.stat().st_size == 0:
    return False
  try:
    with path.open() as f:
      data = json.load(f)
    perf = data.get("performance", {})
    return (
        np.isfinite(float(perf["wall_seconds_total"]))
        and np.isfinite(float(perf["peak_allocated_mib"])))
  except Exception:
    return False


def _specs(codebase: str) -> list[BenchSpec]:
  if codebase == "mdlm":
    return [
        BenchSpec(
            label="mdlm_small_fp64",
            display_name="MDLM-Small fp64",
            codebase="mdlm",
            checkpoint=RUNS_DIR / "mdlm.ckpt",
            model_size="small",
            algo_name="mdlm",
            predictor="ancestral_cache",
            use_float64=True,
            gumbel_precision="float64"),
        BenchSpec(
            label="mdlm_small_fp32",
            display_name="MDLM-Small fp32",
            codebase="mdlm",
            checkpoint=RUNS_DIR / "mdlm.ckpt",
            model_size="small",
            algo_name="mdlm",
            predictor="ancestral_cache",
            use_float64=False,
            gumbel_precision="float32"),
        BenchSpec(
            label="mdlm_small_tau0p9_fp32",
            display_name="MDLM-Small tau=0.9 fp32",
            codebase="mdlm",
            checkpoint=RUNS_DIR / "mdlm.ckpt",
            model_size="small",
            algo_name="mdlm",
            predictor="ancestral_cache",
            use_float64=False,
            gumbel_precision="float32",
            x0_temperature=0.9),
        BenchSpec(
            label="mdlm_small_fp16",
            display_name="MDLM-Small fp16",
            codebase="mdlm",
            checkpoint=RUNS_DIR / "mdlm.ckpt",
            model_size="small",
            algo_name="mdlm",
            predictor="ancestral_cache",
            use_float64=False,
            gumbel_precision="float16"),
        BenchSpec(
            label="mdlm_tiny_fp64",
            display_name="MDLM-Tiny fp64",
            codebase="mdlm",
            checkpoint=TINY_LATE_CKPT,
            model_size="tiny",
            algo_name="mdlm",
            predictor="ancestral_cache",
            use_float64=True,
            gumbel_precision="float64"),
        BenchSpec(
            label="mdlm_tiny_fp32",
            display_name="MDLM-Tiny fp32",
            codebase="mdlm",
            checkpoint=TINY_LATE_CKPT,
            model_size="tiny",
            algo_name="mdlm",
            predictor="ancestral_cache",
            use_float64=False,
            gumbel_precision="float32"),
    ]
  if codebase == "didi":
    return [
        BenchSpec(
            label="didi_instruct_fp64",
            display_name="DiDi-Instruct fp64",
            codebase="didi",
            checkpoint=RUNS_DIR / "didi-instruct.sanitized.ckpt",
            model_size="small",
            algo_name="didi_instruct",
            predictor="guided",
            use_float64=True,
            gumbel_precision="float64"),
        BenchSpec(
            label="didi_instruct_fp32",
            display_name="DiDi-Instruct fp32",
            codebase="didi",
            checkpoint=RUNS_DIR / "didi-instruct.sanitized.ckpt",
            model_size="small",
            algo_name="didi_instruct",
            predictor="guided",
            use_float64=False,
            gumbel_precision="float32"),
    ]
  raise ValueError(f"Unknown codebase: {codebase}")


def _configure_imports(codebase: str):
  base = CODE_DIR if codebase == "mdlm" else DIDI_DIR
  sys.path.insert(0, str(base))
  import algo  # pylint: disable=import-outside-toplevel
  import dataloader  # pylint: disable=import-outside-toplevel
  return base, algo, dataloader


def _diffusion_class(algo_module, spec: BenchSpec):
  if spec.codebase == "mdlm":
    return {"mdlm": algo_module.MDLM}[spec.algo_name]
  return {"didi_instruct": algo_module.DiDiInstruct}[spec.algo_name]


def _compose_config(spec: BenchSpec, args: argparse.Namespace, step: int,
                    config_dir: Path):
  if spec.codebase == "mdlm":
    tokenizer_path = _resolve_local_hf_path("gpt2")
    evaluator_path = _resolve_local_hf_path("gpt2-large")
    overrides = [
        "mode=sample_eval",
        f"seed={args.seed}",
        f"loader.batch_size={args.eval_batch_size}",
        f"loader.eval_batch_size={args.eval_batch_size}",
      "data=openwebtext-split",
      f"data.cache_dir={SCRATCH_ROOT / 'dlm_asym' / 'cache'}",
      f"data.tokenizer_name_or_path={tokenizer_path}",
        "algo=mdlm",
        f"model={spec.model_size}",
        f"model.length={args.seq_len}",
        f"eval.checkpoint_path={spec.checkpoint}",
      "eval.compute_generative_perplexity=false",
      f"eval.gen_ppl_eval_model_name_or_path={evaluator_path}",
        f"sampling.predictor={spec.predictor}",
        f"sampling.steps={step}",
        f"sampling.num_sample_batches={args.measure_batches}",
        "sampling.noise_removal=ancestral",
        f"sampling.use_float64={str(spec.use_float64).lower()}",
        f"sampling.gumbel_precision={spec.gumbel_precision}",
        f"sampling.x0_temperature={spec.x0_temperature}",
        "sampling.x0_temperature_schedule=constant",
        "sampling.ancestral_reveal_strategy=independent",
        "trainer.devices=1",
        "wandb=null",
        f"hydra.run.dir={args.out_dir / 'hydra' / spec.label}",
        f"checkpointing.save_dir={args.out_dir / 'hydra' / spec.label}",
    ]
  else:
    # Avoid mode=sample_eval because DiDi's generate_samples prints one line per
    # reverse step in that mode, which distorts high-budget wallclock logging.
    overrides = [
        "mode=benchmark",
        f"seed={args.seed}",
        f"loader.batch_size={args.eval_batch_size}",
        f"loader.eval_batch_size={args.eval_batch_size}",
        "algo=didi_instruct",
        "model=small",
        f"model.length={args.seq_len}",
        f"eval.checkpoint_path={spec.checkpoint}",
        "eval.compute_generative_perplexity=false",
        f"sampling.predictor={spec.predictor}",
        f"sampling.steps={step}",
        f"sampling.num_sample_batches={args.measure_batches}",
        "sampling.noise_removal=ancestral",
        f"sampling.use_float64={str(spec.use_float64).lower()}",
        f"sampling.temperature={spec.didi_temperature}",
        f"sampling.use_student_ema={str(spec.didi_use_student_ema).lower()}",
        f"sampling.guided_variant={spec.didi_guided_variant}",
        "trainer.devices=1",
        "wandb=null",
        f"hydra.run.dir={args.out_dir / 'hydra' / spec.label}",
        f"checkpointing.save_dir={args.out_dir / 'hydra' / spec.label}",
    ]

  if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
  with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    return compose(config_name="config", overrides=overrides)


def _load_model(spec: BenchSpec, args: argparse.Namespace, config_dir: Path,
                algo_module, dataloader_module):
  config = _compose_config(spec, args, args.steps[0], config_dir)
  tokenizer = dataloader_module.get_tokenizer(config)
  cls = _diffusion_class(algo_module, spec)
  kwargs = {
      "tokenizer": tokenizer,
      "config": config,
  }
  if spec.codebase == "didi":
    kwargs["strict"] = False
  model = cls.load_from_checkpoint(str(spec.checkpoint), **kwargs)
  model = model.to("cuda")
  model.eval()
  return model, config


def _mib(num_bytes: int | float) -> float:
  return float(num_bytes) / (1024.0 * 1024.0)


def _zero_model_grads(model) -> None:
  try:
    model.zero_grad(set_to_none=True)
  except TypeError:
    model.zero_grad()


def _run_one_batch(model, step: int) -> tuple[float, float, int, int]:
  if torch.cuda.is_available():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
  wall_start = time.perf_counter()
  samples = model.restore_model_and_sample(num_steps=step)
  if torch.cuda.is_available():
    end_event.record()
    torch.cuda.synchronize()
    cuda_seconds = start_event.elapsed_time(end_event) / 1000.0
  else:
    cuda_seconds = float("nan")
  wall_seconds = time.perf_counter() - wall_start
  batch_size = int(samples.shape[0])
  num_tokens = int(samples.numel())
  del samples
  return wall_seconds, cuda_seconds, batch_size, num_tokens


def _benchmark_step(model, config, spec: BenchSpec, args: argparse.Namespace,
                    step: int, out_file: Path):
  del config
  torch.cuda.empty_cache()
  _zero_model_grads(model)

  for _ in range(args.warmup_batches):
    _run_one_batch(model, step)
    _zero_model_grads(model)
    torch.cuda.empty_cache()

  torch.cuda.synchronize()
  baseline_allocated = torch.cuda.memory_allocated()
  baseline_reserved = torch.cuda.memory_reserved()

  wall_times = []
  cuda_times = []
  peak_allocated = []
  peak_reserved = []
  total_samples = 0
  total_tokens = 0

  for _ in range(args.measure_batches):
    torch.cuda.empty_cache()
    _zero_model_grads(model)
    torch.cuda.reset_peak_memory_stats()
    wall_seconds, cuda_seconds, batch_size, num_tokens = _run_one_batch(
        model, step)
    wall_times.append(wall_seconds)
    cuda_times.append(cuda_seconds)
    peak_allocated.append(_mib(torch.cuda.max_memory_allocated()))
    peak_reserved.append(_mib(torch.cuda.max_memory_reserved()))
    total_samples += batch_size
    total_tokens += num_tokens
    _zero_model_grads(model)
    gc.collect()
    torch.cuda.empty_cache()

  wall_total = float(np.sum(wall_times))
  cuda_total = float(np.sum(cuda_times))
  peak_alloc_mib = float(np.max(peak_allocated))
  peak_reserved_mib = float(np.max(peak_reserved))
  baseline_alloc_mib = _mib(baseline_allocated)
  baseline_reserved_mib = _mib(baseline_reserved)

  payload = {
      "metadata": {
          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "label": spec.label,
          "display_name": spec.display_name,
          "codebase": spec.codebase,
          "checkpoint": str(spec.checkpoint),
          "model_size": spec.model_size,
          "algo": spec.algo_name,
          "sampler": spec.predictor,
          "sample_steps": step,
          "seq_len": args.seq_len,
          "eval_batch_size": args.eval_batch_size,
          "warmup_batches": args.warmup_batches,
          "measure_batches": args.measure_batches,
          "total_samples": total_samples,
          "total_tokens": total_tokens,
          "use_float64": spec.use_float64,
          "gumbel_precision": spec.gumbel_precision,
          "x0_temperature": spec.x0_temperature,
          "didi_temperature": spec.didi_temperature,
          "didi_use_student_ema": spec.didi_use_student_ema,
          "didi_guided_variant": spec.didi_guided_variant,
          "device": torch.cuda.get_device_name(0),
      },
      "performance": {
          "wall_seconds_total": wall_total,
          "cuda_seconds_total": cuda_total,
          "wall_seconds_per_generation": wall_total / total_samples,
          "cuda_seconds_per_generation": cuda_total / total_samples,
          "wall_seconds_per_sample": wall_total / total_samples,
          "cuda_seconds_per_sample": cuda_total / total_samples,
          "wall_generations_per_second": total_samples / wall_total,
          "cuda_generations_per_second": total_samples / cuda_total,
          "wall_samples_per_second": total_samples / wall_total,
          "cuda_samples_per_second": total_samples / cuda_total,
          "wall_tokens_per_second": total_tokens / wall_total,
          "cuda_tokens_per_second": total_tokens / cuda_total,
          "wall_seconds_per_40_samples": wall_total / total_samples * 40.0,
          "cuda_seconds_per_40_samples": cuda_total / total_samples * 40.0,
          "wall_seconds_per_batch": float(np.mean(wall_times)),
          "cuda_seconds_per_batch": float(np.mean(cuda_times)),
          "wall_batch_seconds_mean": float(np.mean(wall_times)),
          "wall_batch_seconds_std": float(np.std(wall_times, ddof=0)),
          "cuda_batch_seconds_mean": float(np.mean(cuda_times)),
          "cuda_batch_seconds_std": float(np.std(cuda_times, ddof=0)),
          "baseline_allocated_mib": baseline_alloc_mib,
          "baseline_reserved_mib": baseline_reserved_mib,
          "peak_allocated_mib": peak_alloc_mib,
          "peak_reserved_mib": peak_reserved_mib,
          "peak_incremental_allocated_mib": peak_alloc_mib - baseline_alloc_mib,
          "peak_incremental_reserved_mib": peak_reserved_mib - baseline_reserved_mib,
      },
      "per_batch": [
          {
              "wall_seconds": float(wall),
              "cuda_seconds": float(cuda),
              "peak_allocated_mib": float(alloc),
              "peak_reserved_mib": float(reserved),
          }
          for wall, cuda, alloc, reserved in zip(
              wall_times, cuda_times, peak_allocated, peak_reserved)
      ],
  }

  out_file.parent.mkdir(parents=True, exist_ok=True)
  with out_file.open("w") as f:
    json.dump(payload, f, indent=2)
  return payload


def _append_summary(summary_path: Path, rows: list[list[object]], force: bool):
  header = [
      "label", "display_name", "codebase", "model_size", "sampler",
      "steps", "temperature", "precision", "use_float64", "eval_batch_size",
      "measure_batches", "total_samples", "total_tokens",
      "wall_seconds_per_generation", "cuda_seconds_per_generation",
      "wall_seconds_per_batch", "cuda_seconds_per_batch",
      "wall_seconds_per_40_samples", "cuda_seconds_per_40_samples",
      "wall_generations_per_second", "cuda_generations_per_second",
      "wall_tokens_per_second", "cuda_tokens_per_second",
      "peak_allocated_mib", "peak_incremental_allocated_mib",
      "baseline_allocated_mib", "peak_reserved_mib",
      "peak_incremental_reserved_mib", "json",
  ]
  if force or not summary_path.exists():
    summary_path.write_text("\t".join(header) + "\n")
  with summary_path.open("a") as f:
    for row in rows:
      f.write("\t".join(str(item) for item in row) + "\n")


def _precision_label(spec: BenchSpec) -> str:
  if spec.gumbel_precision in {"float64", "fp64"}:
    return "fp64"
  if spec.gumbel_precision in {"float32", "fp32"}:
    return "fp32"
  if spec.gumbel_precision in {"float16", "fp16", "half"}:
    return "fp16"
  return str(spec.gumbel_precision)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--codebase", choices=["mdlm", "didi"], required=True)
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--steps", type=int, nargs="+",
                      default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
  parser.add_argument("--labels", nargs="*", default=None)
  parser.add_argument("--eval-batch-size", type=int, default=8)
  parser.add_argument("--warmup-batches", type=int, default=1)
  parser.add_argument("--measure-batches", type=int, default=3)
  parser.add_argument("--seq-len", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("This benchmark needs CUDA for timing/memory claims.")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  (args.out_dir / "json").mkdir(exist_ok=True)
  args.steps = [int(s) for s in args.steps]

  base, algo_module, dataloader_module = _configure_imports(args.codebase)
  config_dir = base / "configs"
  selected = set(args.labels) if args.labels else None
  specs = [
      spec for spec in _specs(args.codebase)
      if selected is None or spec.label in selected
  ]
  if not specs:
    raise ValueError(f"No specs selected for codebase={args.codebase}")

  summary_path = args.out_dir / f"summary_{args.codebase}.tsv"
  if args.force and summary_path.exists():
    summary_path.unlink()

  for spec in specs:
    missing_steps = [
        step for step in args.steps
        if args.force or not _json_complete(
            args.out_dir / "json" / f"{spec.label}_s{step}.json")
    ]
    if not missing_steps:
      print(f"[skip] {spec.label}: all requested JSONs already complete")
      continue
    print(f"[load] {spec.display_name}: {spec.checkpoint}")
    model, config = _load_model(
        spec, args, config_dir, algo_module, dataloader_module)

    rows = []
    for step in missing_steps:
      out_file = args.out_dir / "json" / f"{spec.label}_s{step}.json"
      print(f"[run] {spec.label} N={step} -> {out_file}", flush=True)
      payload = _benchmark_step(model, config, spec, args, step, out_file)
      perf = payload["performance"]
      meta = payload["metadata"]
      rows.append([
          spec.label,
          spec.display_name,
          spec.codebase,
          spec.model_size,
          spec.predictor,
          step,
          spec.x0_temperature,
          _precision_label(spec),
          spec.use_float64,
          meta["eval_batch_size"],
          meta["measure_batches"],
          meta["total_samples"],
          meta["total_tokens"],
          perf["wall_seconds_per_generation"],
          perf["cuda_seconds_per_generation"],
          perf["wall_seconds_per_batch"],
          perf["cuda_seconds_per_batch"],
          perf["wall_seconds_per_40_samples"],
          perf["cuda_seconds_per_40_samples"],
          perf["wall_generations_per_second"],
          perf["cuda_generations_per_second"],
          perf["wall_tokens_per_second"],
          perf["cuda_tokens_per_second"],
          perf["peak_allocated_mib"],
          perf["peak_incremental_allocated_mib"],
          perf["baseline_allocated_mib"],
          perf["peak_reserved_mib"],
          perf["peak_incremental_reserved_mib"],
          str(out_file),
      ])
      print(
          f"[done] {spec.label} N={step}: "
          f"wall/40={perf['wall_seconds_per_40_samples']:.3f}s, "
          f"peak={perf['peak_allocated_mib'] / 1024.0:.2f}GiB, "
          f"delta={perf['peak_incremental_allocated_mib'] / 1024.0:.2f}GiB",
          flush=True)

    _append_summary(summary_path, rows, force=False)
    del model
    del config
    gc.collect()
    torch.cuda.empty_cache()

  print(f"[summary] {summary_path}")


if __name__ == "__main__":
  main()
