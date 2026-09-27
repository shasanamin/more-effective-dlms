#!/usr/bin/env python3
"""Summarize paired schedule-matched calibration-certificate measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LABELS = ("public_mdlm_small", "early_mdlm_small", "late_mdlm_small")


def _paired_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(lo), float(hi)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> None:
    result_dir = args.results_dir.resolve()
    summaries = {
        label: json.loads((result_dir / f"{label}.json").read_text())
        for label in LABELS
    }
    arrays = {
        label: np.load(result_dir / f"{label}.npz")
        for label in LABELS
    }

    reference = arrays[LABELS[0]]
    for label in LABELS[1:]:
        current = arrays[label]
        for key in (
            "budgets",
            "temperatures",
            "certificate_round",
            "certificate_position",
            "profile_a",
            "profile_position",
        ):
            if not np.array_equal(reference[key], current[key]):
                raise ValueError(f"Measurements are not paired for key={key}: {label}")

    temperatures = reference["temperatures"]
    budgets = reference["budgets"]
    tau_one = int(np.flatnonzero(np.isclose(temperatures, 1.0))[0])
    tau_point = int(np.flatnonzero(np.isclose(temperatures, args.temperature))[0])

    certificate_rows: list[dict[str, object]] = []
    for label in LABELS:
        for budget_index, budget in enumerate(budgets):
            item = summaries[label]["budgets"][str(int(budget))]
            curve = {
                float(row["temperature"]): row
                for row in item["curve"]
            }
            row = curve[float(args.temperature)]
            certificate_rows.append(
                {
                    "checkpoint": label,
                    "configured_updates": int(budget),
                    "temperature": float(args.temperature),
                    "delta_cross_entropy_vs_tau1": row["delta_from_tau_1"],
                    "delta_ci95_low": row["delta_ci95"][0],
                    "delta_ci95_high": row["delta_ci95"][1],
                    "grid_optimal_temperature": item[
                        "certificate_optimal_temperature_on_grid"
                    ],
                    "grid_optimal_temperature_ci95_low": item[
                        "certificate_optimal_temperature_bootstrap_ci95"
                    ][0],
                    "grid_optimal_temperature_ci95_high": item[
                        "certificate_optimal_temperature_bootstrap_ci95"
                    ][1],
                    "num_sequences": summaries[label]["metadata"]["num_sequences"],
                    "weights": summaries[label]["metadata"]["weights"],
                    "cleanup_mass": summaries[label]["metadata"]["cleanup_mass"],
                }
            )

    rng = np.random.default_rng(args.bootstrap_seed)
    early = arrays["early_mdlm_small"]
    late = arrays["late_mdlm_small"]
    checkpoint_rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(budgets):
        difference = (
            late["certificate_ce"][budget_index, :, tau_one]
            - early["certificate_ce"][budget_index, :, tau_one]
        )
        point, low, high = _paired_ci(difference, rng, args.bootstrap_samples)
        checkpoint_rows.append(
            {
                "configured_updates": int(budget),
                "temperature": 1.0,
                "late_minus_early_cross_entropy": point,
                "ci95_low": low,
                "ci95_high": high,
                "num_paired_sequences": int(difference.size),
            }
        )

    profile_difference = (
        late["profile_ce"][:, :, tau_one]
        - early["profile_ce"][:, :, tau_one]
    )
    profile_rows: list[dict[str, object]] = []
    for level, values in zip(reference["profile_a"], profile_difference):
        point, low, high = _paired_ci(values, rng, args.bootstrap_samples)
        profile_rows.append(
            {
                "clean_probability": float(level),
                "late_minus_early_cross_entropy": point,
                "ci95_low": low,
                "ci95_high": high,
                "num_paired_sequences": int(values.size),
            }
        )

    integrated = profile_difference.mean(axis=0)
    integrated_point, integrated_low, integrated_high = _paired_ci(
        integrated, rng, args.bootstrap_samples
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "temperature_certificate.csv", certificate_rows)
    _write_csv(args.output_dir / "checkpoint_certificate.csv", checkpoint_rows)
    _write_csv(args.output_dir / "checkpoint_profile.csv", profile_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "temperature_contrast": float(args.temperature),
                "temperature_reference": 1.0,
                "paired_profile_left32_late_minus_early": integrated_point,
                "paired_profile_left32_ci95": [
                    integrated_low,
                    integrated_high,
                ],
                "interpretation": (
                    "Positive temperature contrasts mean the sharpened "
                    "temperature increases the teacher-forced certificate. "
                    "Negative checkpoint contrasts mean the late checkpoint "
                    "has lower certificate cross-entropy."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    # A compact human-readable audit printout.
    print("checkpoint\tM\tdelta_J(tau)-J(1)\t95% CI\tgrid optimum")
    for row in certificate_rows:
        print(
            f"{row['checkpoint']}\t{row['configured_updates']}\t"
            f"{row['delta_cross_entropy_vs_tau1']:.4f}\t"
            f"[{row['delta_ci95_low']:.4f}, "
            f"{row['delta_ci95_high']:.4f}]\t"
            f"{row['grid_optimal_temperature']:.2f}"
        )
    print(
        "late-minus-early profile left-32 mean: "
        f"{integrated_point:.4f} "
        f"[{integrated_low:.4f}, {integrated_high:.4f}]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    return parser.parse_args()


if __name__ == "__main__":
    summarize(parse_args())
