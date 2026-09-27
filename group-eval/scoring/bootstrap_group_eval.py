#!/usr/bin/env python3
"""Crossed Bayesian bootstrap for GroupEval scores and method comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from score_group_eval import (
    DIMENSIONS,
    aggregate_systems,
    compute_group_metrics,
    flatten_judgments,
    read_jsonl,
)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - level
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def one_sided_lower(values: np.ndarray, level: float = 0.95) -> float:
    return float(np.quantile(values, 1.0 - level))


def _build_method_data(
    sample_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}
    rows_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        rows_by_method[str(row["method_code"])].append(row)

    for method, rows in rows_by_method.items():
        by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        partitions: set[str] = set()
        for row in rows:
            sample_index = str(row["sample_index"])
            partition = str(row["partition_id"])
            group_id = str(row["group_id"])
            by_sample[sample_index].append(row)
            groups[(partition, group_id)].append(row)
            partitions.add(partition)
        methods[method] = {
            "rows": rows,
            "by_sample": by_sample,
            "groups": groups,
            "partitions": sorted(partitions),
            "sample_indices": sorted(by_sample),
        }
    return methods


def _row_to_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "valid": bool(row["valid"]),
        "mode_id": row["mode_id"],
        "hybrid": bool(row["hybrid"]),
        "severe_degeneration": bool(row["severe_degeneration"]),
        "quality": {dimension: int(row[dimension]) for dimension in DIMENSIONS},
    }


def _weighted_method_scores(
    data: Mapping[str, Any],
    sample_weights: Mapping[str, float],
    partition_weights: Mapping[str, float],
) -> tuple[float, float]:
    # GEQ: partition-weighted context mean within sample, then sample-weighted mean.
    geq_num = 0.0
    geq_den = 0.0
    for sample_index, contexts in data["by_sample"].items():
        context_num = 0.0
        context_den = 0.0
        for row in contexts:
            weight = float(partition_weights[str(row["partition_id"])])
            context_num += weight * float(row["quality"])
            context_den += weight
        if context_den <= 0:
            continue
        sample_weight = float(sample_weights[sample_index])
        geq_num += sample_weight * (context_num / context_den)
        geq_den += sample_weight
    geq = geq_num / geq_den

    # GED: recompute weighted mode entropy inside every stored group. Sample weights
    # propagate across both frozen partitions; partition weights represent grouping context.
    ged_partition_num: dict[str, float] = defaultdict(float)
    ged_partition_den: dict[str, float] = defaultdict(float)
    for (partition, _group_id), rows in data["groups"].items():
        samples = [_row_to_sample(row) for row in rows]
        weights = [float(sample_weights[str(row["sample_index"])]) for row in rows]
        group_metrics = compute_group_metrics(samples, weights=weights)
        group_mass = float(sum(weights))
        ged_partition_num[partition] += group_mass * float(group_metrics["ged"])
        ged_partition_den[partition] += group_mass

    ged_num = 0.0
    ged_den = 0.0
    for partition in data["partitions"]:
        if ged_partition_den[partition] <= 0:
            continue
        partition_score = ged_partition_num[partition] / ged_partition_den[partition]
        weight = float(partition_weights[partition])
        ged_num += weight * partition_score
        ged_den += weight
    ged = ged_num / ged_den
    return float(geq), float(ged)


def _point_estimates(
    sample_rows: Sequence[Mapping[str, Any]], group_rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    systems = aggregate_systems(sample_rows, group_rows)
    return {
        str(row["method_code"]): {"geq": float(row["geq"]), "ged": float(row["ged"])}
        for row in systems
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--pairing-mode",
        choices=("none", "partitions", "samples-and-partitions"),
        default="partitions",
        help=(
            "Dependence used for method comparisons. 'partitions' shares only random-"
            "partition weights across aligned methods (canonical default). 'samples-and-"
            "partitions' additionally shares sample weights and is valid only when sample "
            "indices are substantively paired by common random numbers. 'none' treats "
            "methods independently."
        ),
    )
    parser.add_argument("--baseline", default=None, help="Optional baseline method code")
    parser.add_argument("--ged-threshold", type=float, default=None)
    parser.add_argument("--ged-ni-margin", type=float, default=None)
    args = parser.parse_args()

    if args.replicates < 100:
        raise ValueError("use at least 100 replicates for a smoke run; protocol default is 10,000")
    records = read_jsonl(args.judgments)
    sample_rows, group_rows = flatten_judgments(records)
    method_data = _build_method_data(sample_rows)
    methods = sorted(method_data)
    if not methods:
        raise ValueError("no methods found")
    if args.baseline is not None and args.baseline not in method_data:
        raise ValueError(f"baseline {args.baseline!r} not found; methods={methods}")

    sample_sets = [set(method_data[method]["sample_indices"]) for method in methods]
    partition_sets = [set(method_data[method]["partitions"]) for method in methods]
    samples_aligned = all(sample_set == sample_sets[0] for sample_set in sample_sets[1:])
    partitions_aligned = all(
        partition_set == partition_sets[0] for partition_set in partition_sets[1:]
    )
    if args.pairing_mode == "partitions" and not partitions_aligned:
        raise ValueError("partition pairing requested but partition IDs are not aligned")
    if args.pairing_mode == "samples-and-partitions" and not (samples_aligned and partitions_aligned):
        raise ValueError(
            "sample/partition pairing requested but sample indices or partition IDs are not aligned"
        )
    share_samples = args.pairing_mode == "samples-and-partitions"
    share_partitions = args.pairing_mode in {"partitions", "samples-and-partitions"}

    rng = np.random.default_rng(args.seed)
    distributions: dict[str, dict[str, np.ndarray]] = {
        method: {
            "geq": np.empty(args.replicates, dtype=float),
            "ged": np.empty(args.replicates, dtype=float),
        }
        for method in methods
    }

    common_samples = sorted(sample_sets[0]) if share_samples else []
    common_partitions = sorted(partition_sets[0]) if share_partitions else []

    for replicate in range(args.replicates):
        shared_sample_weights = (
            dict(zip(common_samples, rng.exponential(size=len(common_samples)), strict=True))
            if share_samples
            else None
        )
        shared_partition_weights = (
            dict(
                zip(
                    common_partitions,
                    rng.exponential(size=len(common_partitions)),
                    strict=True,
                )
            )
            if share_partitions
            else None
        )
        for method in methods:
            data = method_data[method]
            if share_samples:
                sample_weights = shared_sample_weights
            else:
                indices = data["sample_indices"]
                sample_weights = dict(
                    zip(indices, rng.exponential(size=len(indices)), strict=True)
                )
            if share_partitions:
                partition_weights = shared_partition_weights
            else:
                partitions = data["partitions"]
                partition_weights = dict(
                    zip(partitions, rng.exponential(size=len(partitions)), strict=True)
                )
            assert sample_weights is not None and partition_weights is not None
            geq, ged = _weighted_method_scores(data, sample_weights, partition_weights)
            distributions[method]["geq"][replicate] = geq
            distributions[method]["ged"][replicate] = ged

    points = _point_estimates(sample_rows, group_rows)
    score_rows: list[dict[str, Any]] = []
    for method in methods:
        geq_ci = percentile_interval(distributions[method]["geq"])
        ged_ci = percentile_interval(distributions[method]["ged"])
        row: dict[str, Any] = {
            "method_code": method,
            "geq": points[method]["geq"],
            "geq_ci_low": geq_ci[0],
            "geq_ci_high": geq_ci[1],
            "geq_lcb_95_one_sided": one_sided_lower(distributions[method]["geq"]),
            "ged": points[method]["ged"],
            "ged_ci_low": ged_ci[0],
            "ged_ci_high": ged_ci[1],
            "ged_lcb_95_one_sided": one_sided_lower(distributions[method]["ged"]),
        }
        if args.ged_threshold is not None:
            row["ged_sufficient"] = row["ged_lcb_95_one_sided"] >= args.ged_threshold
        score_rows.append(row)

    comparisons: list[dict[str, Any]] = []
    for candidate in methods:
        for reference in methods:
            if candidate == reference:
                continue
            delta_geq = distributions[candidate]["geq"] - distributions[reference]["geq"]
            delta_ged = distributions[candidate]["ged"] - distributions[reference]["ged"]
            geq_ci = percentile_interval(delta_geq)
            ged_ci = percentile_interval(delta_ged)
            geq_lcb = one_sided_lower(delta_geq)
            ged_lcb = one_sided_lower(delta_ged)
            comparison: dict[str, Any] = {
                "candidate": candidate,
                "reference": reference,
                "delta_geq": points[candidate]["geq"] - points[reference]["geq"],
                "delta_geq_ci_low": geq_ci[0],
                "delta_geq_ci_high": geq_ci[1],
                "delta_geq_lcb_95": geq_lcb,
                "quality_superior": geq_lcb > 0.0,
                "delta_ged": points[candidate]["ged"] - points[reference]["ged"],
                "delta_ged_ci_low": ged_ci[0],
                "delta_ged_ci_high": ged_ci[1],
                "delta_ged_lcb_95": ged_lcb,
            }
            if args.ged_ni_margin is not None:
                comparison["ged_noninferior"] = ged_lcb > -args.ged_ni_margin
                comparison["quality_superior_and_ged_noninferior"] = (
                    geq_lcb > 0.0 and ged_lcb > -args.ged_ni_margin
                )
            comparisons.append(comparison)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "bootstrap_scores.csv", score_rows)
    write_csv(args.out_dir / "bootstrap_comparisons.csv", comparisons)
    payload = {
        "protocol_version": "1.0.0",
        "bootstrap": {
            "method": "crossed_bayesian_bootstrap",
            "replicates": args.replicates,
            "seed": args.seed,
            "pairing_mode": args.pairing_mode,
            "samples_aligned": samples_aligned,
            "partitions_aligned": partitions_aligned,
            "confidence_level": 0.95,
            "notes": [
                "Partition weights are shared across methods by default because the same frozen partitions are used as a blocking factor.",
                "Sample weights are shared only under samples-and-partitions mode, which requires substantively paired outputs generated with common random numbers.",
                "GED is recomputed from weighted valid-mode masses inside each stored group.",
                "Mode assignments are treated as observed judge outputs; this bootstrap does not model latent clustering uncertainty.",
                "GED weighted ranges are sensitivity summaries, not calibrated confidence intervals; no threshold is supplied.",
            ],
        },
        "ged_threshold": args.ged_threshold,
        "ged_noninferiority_margin": args.ged_ni_margin,
        "scores": score_rows,
        "comparisons": comparisons,
    }
    (args.out_dir / "bootstrap.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
