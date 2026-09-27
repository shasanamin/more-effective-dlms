#!/usr/bin/env python3
"""Measure the finite-grid MDLM temperature-calibration certificate.

This implements the forced-mask estimator from the paper and matches the
deployed MDLM schedule.  For a configured budget M, the implementation makes
M stochastic updates with total reveal mass 1-epsilon and one terminal
ancestral cleanup with reveal mass epsilon.  Each held-out sequence therefore
contributes:

1. sample a main reveal round K uniformly from {1, ..., M};
2. sample one target position I uniformly;
3. retain every non-target token with probability
   (K - 1)(1-epsilon)/M;
4. force I to be masked and cache the denoiser logits at I.

An independently forced cleanup context at retention 1-epsilon supplies the
small residual term.  The two losses are combined with weights 1-epsilon and
epsilon, respectively.  Cached logits support an entire temperature sweep
without generation.  The script also measures a mask-level cross-entropy
profile on a fixed grid.  All reported uncertainty intervals resample complete
sequences.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SCRATCH_ROOT = os.environ.get(
    "SCRATCH_ROOT",
    os.environ.get("OUTPUT_ROOT", str(PACKAGE_ROOT / "outputs")),
)
HF_HOME = os.environ.setdefault("HF_HOME", f"{SCRATCH_ROOT}/cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", f"{HF_HOME}/datasets")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{HF_HOME}/transformers")
os.environ.setdefault("XDG_CACHE_HOME", f"{SCRATCH_ROOT}/cache/xdg")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import algo  # noqa: E402
import dataloader  # noqa: E402


def _get_diffusion_model(algo_name: str):
    mapping = {
        "mdlm": getattr(algo, "MDLM", None),
        "duo_base": getattr(algo, "DUO_BASE", None),
        "cdlm": getattr(algo, "CDLM", None),
        "d3pm": getattr(algo, "D3PMAbsorb", None),
        "sedd": getattr(algo, "SEDDAbsorb", None),
        "duo": getattr(algo, "DUO", None),
    }
    if algo_name not in mapping or mapping[algo_name] is None:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    return mapping[algo_name]


def _maybe_resolve_local_hf_path(name_or_path: str) -> str:
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    cache_root = Path(os.environ["TRANSFORMERS_CACHE"])
    model_key = name_or_path.replace("/", "--")
    snapshot_root = cache_root / f"models--{model_key}" / "snapshots"
    if not snapshot_root.exists():
        return name_or_path
    snapshots = sorted(
        (p for p in snapshot_root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(snapshots[0]) if snapshots else name_or_path


def _resolve_config(
    run_dir: Path,
    checkpoint_path: Path,
    eval_batch_size: int,
    num_workers: int,
):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    config = OmegaConf.load(config_path)
    config.mode = "ppl_eval"
    config.trainer.devices = 1
    config.trainer.num_nodes = 1
    config.loader.global_batch_size = eval_batch_size
    config.loader.eval_global_batch_size = eval_batch_size
    config.loader.batch_size = eval_batch_size
    config.loader.eval_batch_size = eval_batch_size
    config.loader.num_workers = num_workers
    config.data.tokenizer_name_or_path = _maybe_resolve_local_hf_path(
        str(config.data.tokenizer_name_or_path)
    )
    config.eval.checkpoint_path = str(checkpoint_path)
    config.eval.generate_samples = False
    config.eval.compute_generative_perplexity = False
    if hasattr(config.eval, "sensitivity"):
        config.eval.sensitivity.enabled = False
    return config


def _collect_sequences(
    loader: Iterable[dict[str, torch.Tensor]],
    num_sequences: int,
) -> torch.Tensor:
    chunks = []
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].detach().cpu()
        take = min(input_ids.shape[0], num_sequences - count)
        chunks.append(input_ids[:take])
        count += take
        if count >= num_sequences:
            break
    if count < num_sequences:
        raise RuntimeError(
            f"Validation loader yielded {count} sequences; requested {num_sequences}"
        )
    return torch.cat(chunks, dim=0)


def _bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    point = float(values.mean())
    if values.size < 2 or draws <= 0:
        return point, point, point
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return point, float(lo), float(hi)


def _temperature_summary(
    ce: np.ndarray,
    temperatures: np.ndarray,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = ce.shape[0]
    if n == 0:
        raise ValueError("Cannot summarize an empty observation matrix")
    point_means = ce.mean(axis=0)
    bootstrap_indices = rng.integers(0, n, size=(bootstrap_draws, n))
    bootstrap_means = ce[bootstrap_indices].mean(axis=1)
    lower = np.quantile(bootstrap_means, 0.025, axis=0)
    upper = np.quantile(bootstrap_means, 0.975, axis=0)
    best_index = int(np.argmin(point_means))
    bootstrap_best = temperatures[np.argmin(bootstrap_means, axis=1)]

    tau_one_index = int(np.argmin(np.abs(temperatures - 1.0)))
    if not np.isclose(temperatures[tau_one_index], 1.0):
        raise ValueError("Temperature grid must contain tau=1")
    delta = ce - ce[:, [tau_one_index]]
    delta_bootstrap = delta[bootstrap_indices].mean(axis=1)
    delta_lower = np.quantile(delta_bootstrap, 0.025, axis=0)
    delta_upper = np.quantile(delta_bootstrap, 0.975, axis=0)

    curve = []
    for idx, tau in enumerate(temperatures):
        curve.append(
            {
                "temperature": float(tau),
                "cross_entropy": float(point_means[idx]),
                "ci95": [float(lower[idx]), float(upper[idx])],
                "delta_from_tau_1": float(delta[:, idx].mean()),
                "delta_ci95": [
                    float(delta_lower[idx]),
                    float(delta_upper[idx]),
                ],
            }
        )
    return {
        "curve": curve,
        "certificate_optimal_temperature_on_grid": float(
            temperatures[best_index]
        ),
        "certificate_optimal_temperature_bootstrap_ci95": [
            float(np.quantile(bootstrap_best, 0.025)),
            float(np.quantile(bootstrap_best, 0.975)),
        ],
    }


@torch.inference_mode()
def _forced_mask_batch(
    model,
    x0_cpu: torch.Tensor,
    retain_probability: np.ndarray,
    positions: np.ndarray,
    temperatures: np.ndarray,
    mask_uniforms: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x0 = x0_cpu.to(device, non_blocking=True)
    batch_size, length = x0.shape
    retain = torch.as_tensor(
        retain_probability, device=device, dtype=torch.float32
    ).view(batch_size, 1)
    uniforms = torch.as_tensor(mask_uniforms, device=device, dtype=torch.float32)
    keep = uniforms < retain
    z = torch.where(keep, x0, model.mask_index)
    rows = torch.arange(batch_size, device=device)
    cols = torch.as_tensor(positions, device=device, dtype=torch.long)
    z[rows, cols] = model.mask_index
    if getattr(model, "ignore_bos", False):
        z[:, 0] = x0[:, 0]

    # The audited MDLM checkpoints use SUBS without time conditioning.  This
    # makes the forced-mask context, rather than a clipped time embedding, the
    # complete model input relevant to the theorem.
    if bool(getattr(model, "time_conditioning", False)):
        raise ValueError(
            "This implementation currently targets the paper's time-agnostic "
            "MDLM checkpoints; a time-conditioned model needs an explicit "
            "clean-probability-to-time map."
        )
    alpha_for_api = retain.clamp_min(1e-5)
    sigma = model._sigma_from_alphat(alpha_for_api)
    log_probabilities = model.forward(z, sigma)
    forced_logits = log_probabilities[rows, cols].float()
    targets = x0[rows, cols]

    ce_columns = []
    for temperature in temperatures:
        scaled_log_probabilities = F.log_softmax(
            forced_logits / float(temperature), dim=-1
        )
        ce_columns.append(
            -scaled_log_probabilities[rows, targets].detach().cpu().numpy()
        )
    cross_entropy = np.stack(ce_columns, axis=1)

    base_probabilities = forced_logits.exp()
    expected_logit = (base_probabilities * forced_logits).sum(dim=-1)
    target_logit = forced_logits[rows, targets]
    derivative = (expected_logit - target_logit).detach().cpu().numpy()
    return cross_entropy, derivative


def _iter_slices(size: int, batch_size: int):
    for start in range(0, size, batch_size):
        yield slice(start, min(start + batch_size, size))


def measure(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    observations_path = (
        Path(args.observations_output).resolve()
        if args.observations_output
        else output_path.with_suffix(".npz")
    )
    device = torch.device(args.device)
    budgets = np.asarray(sorted(set(args.budgets)), dtype=np.int64)
    temperatures = np.asarray(sorted(set(args.temperatures)), dtype=np.float64)
    profile_a = np.arange(args.profile_bins, dtype=np.float64) / args.profile_bins

    if np.any(budgets < 1):
        raise ValueError("All budgets must be positive")
    if np.any(temperatures <= 0) or not np.any(np.isclose(temperatures, 1.0)):
        raise ValueError("Temperatures must be positive and include 1.0")
    if args.profile_sequences > args.num_sequences:
        raise ValueError("profile_sequences cannot exceed num_sequences")

    config = _resolve_config(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = dataloader.get_tokenizer(config)
    diffusion_model = _get_diffusion_model(str(config.algo.name))
    model = diffusion_model.load_from_checkpoint(
        str(checkpoint_path),
        tokenizer=tokenizer,
        config=config,
    ).to(device)
    # Paper generation enters TrainerBase._eval_mode(), which copies the EMA
    # shadow parameters into the backbone.  Apply that copy once and then drop
    # the EMA helper so subsequent forwards cannot restore raw parameters.
    model._eval_mode()
    model.ema = None

    _, valid_loader = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=args.seed
    )
    sequences = _collect_sequences(valid_loader, args.num_sequences)
    length = int(sequences.shape[1])
    position_low = 1 if getattr(model, "ignore_bos", False) else 0

    certificate_ce = np.empty(
        (len(budgets), args.num_sequences, len(temperatures)), dtype=np.float32
    )
    certificate_derivative = np.empty(
        (len(budgets), args.num_sequences), dtype=np.float32
    )
    certificate_round = np.empty(
        (len(budgets), args.num_sequences), dtype=np.int32
    )
    certificate_position = np.empty(
        (len(budgets), args.num_sequences), dtype=np.int32
    )
    cleanup_ce = np.empty(
        (len(budgets), args.num_sequences, len(temperatures)), dtype=np.float32
    )
    cleanup_derivative = np.empty(
        (len(budgets), args.num_sequences), dtype=np.float32
    )
    cleanup_position = np.empty(
        (len(budgets), args.num_sequences), dtype=np.int32
    )

    for budget_index, budget in enumerate(budgets):
        rng = np.random.default_rng(args.seed + 100_003 * int(budget))
        rounds = rng.integers(1, int(budget) + 1, size=args.num_sequences)
        positions = rng.integers(position_low, length, size=args.num_sequences)
        mask_uniforms = rng.random((args.num_sequences, length), dtype=np.float32)
        retain_probability = (
            (rounds - 1) * (1.0 - args.cleanup_mass) / float(budget)
        )

        for batch_slice in _iter_slices(args.num_sequences, args.eval_batch_size):
            ce, derivative = _forced_mask_batch(
                model=model,
                x0_cpu=sequences[batch_slice],
                retain_probability=retain_probability[batch_slice],
                positions=positions[batch_slice],
                temperatures=temperatures,
                mask_uniforms=mask_uniforms[batch_slice],
                device=device,
            )
            certificate_ce[budget_index, batch_slice] = ce
            certificate_derivative[budget_index, batch_slice] = derivative

        cleanup_positions = rng.integers(
            position_low, length, size=args.num_sequences
        )
        cleanup_uniforms = rng.random(
            (args.num_sequences, length), dtype=np.float32
        )
        cleanup_retain = np.full(
            args.num_sequences, 1.0 - args.cleanup_mass, dtype=np.float64
        )
        for batch_slice in _iter_slices(args.num_sequences, args.eval_batch_size):
            ce, derivative = _forced_mask_batch(
                model=model,
                x0_cpu=sequences[batch_slice],
                retain_probability=cleanup_retain[batch_slice],
                positions=cleanup_positions[batch_slice],
                temperatures=temperatures,
                mask_uniforms=cleanup_uniforms[batch_slice],
                device=device,
            )
            cleanup_ce[budget_index, batch_slice] = ce
            cleanup_derivative[budget_index, batch_slice] = derivative

        certificate_ce[budget_index] = (
            (1.0 - args.cleanup_mass) * certificate_ce[budget_index]
            + args.cleanup_mass * cleanup_ce[budget_index]
        )
        certificate_derivative[budget_index] = (
            (1.0 - args.cleanup_mass) * certificate_derivative[budget_index]
            + args.cleanup_mass * cleanup_derivative[budget_index]
        )
        certificate_round[budget_index] = rounds
        certificate_position[budget_index] = positions
        cleanup_position[budget_index] = cleanup_positions
        print(f"[certificate] {args.label}: completed N={budget}", flush=True)

    profile_ce = np.empty(
        (
            len(profile_a),
            args.profile_sequences,
            len(temperatures),
        ),
        dtype=np.float32,
    )
    profile_position = np.empty(
        (len(profile_a), args.profile_sequences), dtype=np.int32
    )
    profile_sequences = sequences[: args.profile_sequences]
    for level_index, a_value in enumerate(profile_a):
        rng = np.random.default_rng(args.seed + 1_000_003 + level_index)
        positions = rng.integers(
            position_low, length, size=args.profile_sequences
        )
        mask_uniforms = rng.random(
            (args.profile_sequences, length), dtype=np.float32
        )
        retain_probability = np.full(args.profile_sequences, a_value)
        for batch_slice in _iter_slices(
            args.profile_sequences, args.eval_batch_size
        ):
            ce, _ = _forced_mask_batch(
                model=model,
                x0_cpu=profile_sequences[batch_slice],
                retain_probability=retain_probability[batch_slice],
                positions=positions[batch_slice],
                temperatures=temperatures,
                mask_uniforms=mask_uniforms[batch_slice],
                device=device,
            )
            profile_ce[level_index, batch_slice] = ce
        profile_position[level_index] = positions
        print(
            f"[profile] {args.label}: completed "
            f"a={a_value:.6f} ({level_index + 1}/{len(profile_a)})",
            flush=True,
        )

    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)
    budget_summaries = {}
    for budget_index, budget in enumerate(budgets):
        derivative_point, derivative_lo, derivative_hi = _bootstrap_interval(
            -certificate_derivative[budget_index],
            bootstrap_rng,
            args.bootstrap_samples,
        )
        summary = _temperature_summary(
            certificate_ce[budget_index],
            temperatures,
            bootstrap_draws=args.bootstrap_samples,
            seed=args.bootstrap_seed + int(budget),
        )
        summary["negative_derivative_at_tau_1"] = derivative_point
        summary["negative_derivative_at_tau_1_ci95"] = [
            derivative_lo,
            derivative_hi,
        ]
        budget_summaries[str(int(budget))] = summary

    profile_tau_one_index = int(np.argmin(np.abs(temperatures - 1.0)))
    profile_summary = []
    for level_index, a_value in enumerate(profile_a):
        point, lo, hi = _bootstrap_interval(
            profile_ce[level_index, :, profile_tau_one_index],
            bootstrap_rng,
            args.bootstrap_samples,
        )
        profile_summary.append(
            {
                "clean_probability": float(a_value),
                "cross_entropy_at_tau_1": point,
                "ci95": [lo, hi],
            }
        )

    output = {
        "metadata": {
            "label": args.label,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "seed": args.seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "num_sequences": args.num_sequences,
            "profile_sequences": args.profile_sequences,
            "profile_bins": args.profile_bins,
            "sequence_length": length,
            "budgets": budgets.tolist(),
            "temperatures": temperatures.tolist(),
            "weights": "EMA",
            "configured_budget_semantics": (
                "M reverse-update invocations plus terminal cleanup; "
                "M+1 reveal blocks for the certificate, not a model-call count"
            ),
            "cleanup_mass": args.cleanup_mass,
            "main_reveal_mass_per_round": {
                str(int(budget)): (
                    (1.0 - args.cleanup_mass) / float(budget)
                )
                for budget in budgets
            },
            "estimator_scale": (
                "Per-forced-position. Multiply cross-entropy and derivative "
                f"by L={length} for the paper's summed J_N and C_N'."
            ),
        },
        "budgets": budget_summaries,
        "profile_at_tau_1": profile_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    observations_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        observations_path,
        label=np.asarray(args.label),
        budgets=budgets,
        temperatures=temperatures,
        profile_a=profile_a,
        certificate_ce=certificate_ce,
        certificate_derivative=certificate_derivative,
        certificate_round=certificate_round,
        certificate_position=certificate_position,
        cleanup_ce=cleanup_ce,
        cleanup_derivative=cleanup_derivative,
        cleanup_position=cleanup_position,
        profile_ce=profile_ce,
        profile_position=profile_position,
    )
    print(f"[ok] wrote summary to {output_path}")
    print(f"[ok] wrote observations to {observations_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory with .hydra/")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations-output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-sequences", type=int, default=512)
    parser.add_argument("--profile-sequences", type=int, default=128)
    parser.add_argument("--profile-bins", type=int, default=32)
    parser.add_argument(
        "--cleanup-mass",
        type=float,
        default=1e-5,
        help=(
            "Residual reveal mass assigned to the terminal cleanup. The "
            "generation driver defaults to eps=1e-5."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[2, 8, 32, 128, 1024]
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[
            0.50,
            0.60,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            1.05,
            1.10,
            1.20,
            1.30,
            1.40,
            1.50,
        ],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    args = parser.parse_args()
    if not 0.0 <= args.cleanup_mass < 1.0:
        parser.error("--cleanup-mass must lie in [0, 1)")
    return args


if __name__ == "__main__":
    measure(parse_args())
