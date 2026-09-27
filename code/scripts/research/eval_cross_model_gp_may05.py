#!/usr/bin/env python
"""Cross-model generative-PPL sweep for the paper comparison plot.

This driver intentionally loads each model once and evaluates many sampling
budgets in-process.  It is for sample evaluation only; it does not train or
modify checkpoints.
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
RUNS_DIR = SCRATCH_ROOT / "dlm_asym" / "runs"
DEFAULT_OUT_DIR = RUNS_DIR / "cross_model_may05"

sys.path.insert(0, str(CODE_DIR))

import algo  # noqa: E402
import dataloader  # noqa: E402
import metrics  # noqa: E402
import trainer_base  # noqa: E402
import utils  # noqa: E402


OmegaConf.register_new_resolver("cwd", os.getcwd, replace=True)
OmegaConf.register_new_resolver("device_count", torch.cuda.device_count, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "div_up", lambda x, y: (x + y - 1) // y, replace=True)

if hasattr(torch, "set_float32_matmul_precision"):
  torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class EvalSpec:
  label: str
  display_name: str
  checkpoint: Path
  algo_name: str
  predictor: str
  use_float64: bool
  gumbel_precision: str
  x0_temperature: float = 1.0
  x0_temperature_schedule: str = "constant"
  steps: tuple[int, ...] | None = None
  flat_ar: bool = False


def _diffusion_class(algo_name: str):
  return {
      "ar": algo.AR,
      "mdlm": algo.MDLM,
      "duo_base": algo.DUO_BASE,
      "sedd": algo.SEDDAbsorb,
      "duo": algo.DUO,
  }[algo_name]


def _json_complete(path: Path) -> bool:
  if not path.exists() or path.stat().st_size == 0:
    return False
  try:
    with path.open() as f:
      data = json.load(f)
    perf = data.get("performance", {})
    value = perf.get("generative_perplexity")
    return value is not None and np.isfinite(float(value))
  except Exception:
    return False


def _sequence_entropy(token_ids, vocab_size: int) -> float:
  if isinstance(token_ids, torch.Tensor):
    token_ids = token_ids.detach().cpu().numpy()
  unique, counts = np.unique(token_ids, return_counts=True)
  probs = counts / len(token_ids)
  return float(-np.sum(probs * np.log(probs + 1e-10)))


def _compose_config(spec: EvalSpec, args: argparse.Namespace, step: int):
  overrides = [
      "mode=sample_eval",
      f"seed={args.seed}",
      f"loader.batch_size={args.eval_batch_size}",
      f"loader.eval_batch_size={args.eval_batch_size}",
      "data=openwebtext-split",
      f"data.cache_dir={SCRATCH_ROOT / 'dlm_asym' / 'cache'}",
      f"algo={spec.algo_name}",
      "model=small",
      f"model.length={args.seq_len}",
      f"eval.checkpoint_path={spec.checkpoint}",
      "eval.compute_generative_perplexity=true",
      f"eval.gen_ppl_eval_model_name_or_path={args.gen_ppl_model}",
      f"eval.perplexity_batch_size={args.perplexity_batch_size}",
      f"sampling.predictor={spec.predictor}",
      f"sampling.steps={step}",
      f"sampling.num_sample_batches={args.num_sample_batches}",
      "sampling.noise_removal=ancestral",
      f"sampling.use_float64={str(spec.use_float64).lower()}",
      f"sampling.gumbel_precision={spec.gumbel_precision}",
      f"sampling.x0_temperature={spec.x0_temperature}",
      f"sampling.x0_temperature_schedule={spec.x0_temperature_schedule}",
      "sampling.ancestral_reveal_strategy=independent",
      "trainer.devices=1",
      "wandb=null",
      f"hydra.run.dir={args.out_dir / 'hydra' / spec.label}",
      f"checkpointing.save_dir={args.out_dir / 'hydra' / spec.label}",
  ]
  if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
  with initialize_config_dir(
      version_base=None,
      config_dir=str(CODE_DIR / "configs")):
    return compose(config_name="config", overrides=overrides)


def _load_model(spec: EvalSpec, args: argparse.Namespace, step: int):
  config = _compose_config(spec, args, step)
  tokenizer = dataloader.get_tokenizer(config)
  cls = _diffusion_class(spec.algo_name)
  model = cls.load_from_checkpoint(
      str(spec.checkpoint),
      tokenizer=tokenizer,
      config=config)
  model = model.to("cuda")
  model.eval()
  return model, config


@torch.no_grad()
def _evaluate_step(model, config, spec: EvalSpec, step: int, out_file: Path):
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  all_samples: list[str] = []
  all_entropy: list[float] = []
  all_nfe: list[int] = []
  total_sampling_time = 0.0

  for _ in range(int(config.sampling.num_sample_batches)):
    if torch.cuda.is_available():
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
    samples = model.restore_model_and_sample(num_steps=step)
    if torch.cuda.is_available():
      end.record()
      torch.cuda.synchronize()
      total_sampling_time += start.elapsed_time(end) / 1000.0

    model.metrics.record_entropy(samples)
    text_samples = model.tokenizer.batch_decode(samples)
    model.metrics.record_generative_perplexity(
        text_samples, config.model.length, device=model.device)
    all_samples.extend(list(text_samples))
    all_entropy.extend([
        _sequence_entropy(sample, model.tokenizer.vocab_size)
        for sample in samples
    ])
    all_nfe.extend([step] * len(text_samples))

  generative_ppl = float(model.metrics.gen_ppl.compute().item())
  entropy = float(model.metrics.sample_entropy.compute().item())
  total_samples = len(all_samples)
  avg_time_per_sample = (
      total_sampling_time / total_samples if total_samples else 0.0)
  samples_per_second = (
      total_samples / total_sampling_time if total_sampling_time > 0 else 0.0)

  out_file.parent.mkdir(parents=True, exist_ok=True)
  payload = {
      "metadata": {
          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "label": spec.label,
          "display_name": spec.display_name,
          "model_path": str(spec.checkpoint),
          "algo": spec.algo_name,
          "sampler": spec.predictor,
          "sample_steps": step,
          "num_sample_batches": int(config.sampling.num_sample_batches),
          "batch_size": int(config.loader.eval_batch_size),
          "total_samples": total_samples,
          "use_float64": bool(config.sampling.use_float64),
          "gumbel_precision": str(config.sampling.gumbel_precision),
          "x0_temperature": float(config.sampling.x0_temperature),
          "x0_temperature_schedule": str(
              config.sampling.x0_temperature_schedule),
          "gen_ppl_model": str(config.eval.gen_ppl_eval_model_name_or_path),
      },
      "performance": {
          "generative_perplexity": generative_ppl,
          "total_generation_time_seconds": total_sampling_time,
          "average_time_per_sample": avg_time_per_sample,
          "samples_per_second": samples_per_second,
          "avg_nfe": float(np.mean(all_nfe)) if all_nfe else 0.0,
          "avg_entropy": entropy,
          "avg_entropy_per_sequence": (
              float(np.mean(all_entropy)) if all_entropy else 0.0),
      },
      "samples": all_samples,
      "nfe": all_nfe,
  }
  with out_file.open("w") as f:
    json.dump(payload, f, indent=2)
  return payload


def _default_specs(steps: tuple[int, ...]) -> list[EvalSpec]:
  return [
      EvalSpec(
          label="mdlm_default_tau1_fp64",
          display_name="MDLM (default)",
          checkpoint=RUNS_DIR / "mdlm.ckpt",
          algo_name="mdlm",
          predictor="ancestral_cache",
          use_float64=True,
          gumbel_precision="float64",
          x0_temperature=1.0,
          steps=steps),
      EvalSpec(
          label="mdlm_calibrated_tau0p9_fp32",
          display_name="MDLM (calibrated)",
          checkpoint=RUNS_DIR / "mdlm.ckpt",
          algo_name="mdlm",
          predictor="ancestral_cache",
          use_float64=False,
          gumbel_precision="float32",
          x0_temperature=0.9,
          steps=steps),
      EvalSpec(
          label="mdlm_default_tau1_fp16",
          display_name="MDLM (fp16 categorical)",
          checkpoint=RUNS_DIR / "mdlm.ckpt",
          algo_name="mdlm",
          predictor="ancestral_cache",
          use_float64=False,
          gumbel_precision="float16",
          x0_temperature=1.0,
          steps=steps),
      EvalSpec(
          label="duo_default_tau1_fp64",
          display_name="DUO",
          checkpoint=RUNS_DIR / "duo.ckpt",
          algo_name="duo_base",
          predictor="ancestral",
          use_float64=True,
          gumbel_precision="float64",
          steps=steps),
      EvalSpec(
          label="duo_distilled_default_tau1_fp64",
          display_name="DUO-Distilled",
          checkpoint=RUNS_DIR / "duo-distilled.ckpt",
          algo_name="duo_base",
          predictor="ancestral",
          use_float64=True,
          gumbel_precision="float64",
          steps=steps),
      EvalSpec(
          label="sedd_default_tau1_fp64",
          display_name="SEDD",
          checkpoint=RUNS_DIR / "sedd.ckpt",
          algo_name="sedd",
          predictor="analytic",
          use_float64=True,
          gumbel_precision="float64",
          steps=steps),
      EvalSpec(
          label="ar_default_fp64",
          display_name="Autoregressive",
          checkpoint=RUNS_DIR / "ar.ckpt",
          algo_name="ar",
          predictor="ancestral",
          use_float64=True,
          gumbel_precision="float64",
          steps=(steps[-1],),
          flat_ar=True),
  ]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
  parser.add_argument("--steps", type=int, nargs="+",
                      default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
  parser.add_argument("--labels", nargs="*", default=None,
                      help="Optional subset of spec labels to run.")
  parser.add_argument("--eval-batch-size", type=int, default=8)
  parser.add_argument("--num-sample-batches", type=int, default=5)
  parser.add_argument("--perplexity-batch-size", type=int, default=8)
  parser.add_argument("--seq-len", type=int, default=1024)
  parser.add_argument("--gen-ppl-model", default="gpt2-large")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--force", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  (args.out_dir / "json").mkdir(exist_ok=True)
  steps = tuple(int(s) for s in args.steps)
  selected = set(args.labels) if args.labels else None
  specs = [
      spec for spec in _default_specs(steps)
      if selected is None or spec.label in selected
  ]
  summary_path = args.out_dir / "summary.tsv"
  summary_header = [
      "label", "display_name", "steps", "gen_ppl", "entropy",
      "total_samples", "sampler", "algo", "temperature", "precision",
      "use_float64", "json",
  ]
  if not summary_path.exists() or args.force:
    summary_path.write_text("\t".join(summary_header) + "\n")

  for spec in specs:
    missing_steps = [
        step for step in spec.steps
        if args.force or not _json_complete(
            args.out_dir / "json" / f"{spec.label}_s{step}.json")
    ]
    if not missing_steps:
      print(f"[skip] {spec.label}: all requested JSONs already complete")
      continue

    print(f"[load] {spec.display_name}: {spec.checkpoint}")
    model, config = _load_model(spec, args, missing_steps[0])

    rows = []
    for step in missing_steps:
      out_file = args.out_dir / "json" / f"{spec.label}_s{step}.json"
      print(f"[run] {spec.label} N={step} -> {out_file}")
      payload = _evaluate_step(model, config, spec, step, out_file)
      perf = payload["performance"]
      meta = payload["metadata"]
      rows.append([
          spec.label,
          spec.display_name,
          step,
          perf["generative_perplexity"],
          perf["avg_entropy"],
          meta["total_samples"],
          meta["sampler"],
          meta["algo"],
          meta["x0_temperature"],
          meta["gumbel_precision"],
          meta["use_float64"],
          str(out_file),
      ])
      print(
          f"[done] {spec.label} N={step}: "
          f"gPPL={perf['generative_perplexity']:.4g}, "
          f"H={perf['avg_entropy']:.3f}")

    with summary_path.open("a") as f:
      for row in rows:
        f.write("\t".join(str(item) for item in row) + "\n")

    del model
    del config
    gc.collect()
    torch.cuda.empty_cache()

  print(f"[summary] {summary_path}")


if __name__ == "__main__":
  main()
