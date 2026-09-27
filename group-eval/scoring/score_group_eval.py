#!/usr/bin/env python3
"""Compute deterministic GroupEval GEQ/GED scores from validated judge outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DIMENSIONS = (
    "readability",
    "coherence",
    "substance_naturalness",
    "nondegeneration_structure",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            records.append(value)
    return records


def sample_quality(sample: Mapping[str, Any]) -> float:
    if not bool(sample["valid"]):
        return 0.0
    quality = sample["quality"]
    return 1.25 * sum(float(quality[dimension]) for dimension in DIMENSIONS)


def _weighted_entropy(masses: Sequence[float]) -> float:
    total = float(sum(masses))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for mass in masses:
        if mass <= 0:
            continue
        probability = mass / total
        entropy -= probability * math.log(probability)
    return entropy


def compute_group_metrics(
    samples: Sequence[Mapping[str, Any]],
    weights: Sequence[float] | None = None,
) -> dict[str, float | int | bool]:
    """Compute group metrics; weights enable the Bayesian bootstrap GED analogue."""
    n = len(samples)
    if n == 0:
        raise ValueError("group must contain at least one sample")
    if weights is None:
        weights_arr = np.ones(n, dtype=float)
    else:
        if len(weights) != n:
            raise ValueError("weights length must equal group size")
        weights_arr = np.asarray(weights, dtype=float)
        if np.any(weights_arr < 0) or float(weights_arr.sum()) <= 0:
            raise ValueError("weights must be nonnegative with positive total")

    q = np.asarray([sample_quality(sample) for sample in samples], dtype=float)
    total_weight = float(weights_arr.sum())
    geq = float(np.dot(weights_arr, q) / total_weight)

    valid_mask = np.asarray([bool(sample["valid"]) for sample in samples], dtype=bool)
    valid_count = int(valid_mask.sum())
    valid_weight = float(weights_arr[valid_mask].sum())
    valid_fraction = valid_weight / total_weight
    mode_masses: Counter[str] = Counter()
    mode_member_counts: Counter[str] = Counter()
    for sample, weight in zip(samples, weights_arr, strict=True):
        if bool(sample["valid"]):
            mode_id = str(sample["mode_id"])
            mode_masses[mode_id] += float(weight)
            mode_member_counts[mode_id] += 1

    entropy = _weighted_entropy(list(mode_masses.values()))
    represented_modes = sum(mass > 0 for mass in mode_masses.values())
    if valid_weight <= 0 or represented_modes < 2:
        ged = 0.0
        effective_modes = 1.0 if valid_weight > 0 else 0.0
        dominant_share = 1.0 if valid_weight > 0 else 0.0
    else:
        ged = 10.0 * valid_fraction * entropy / math.log(n)
        effective_modes = math.exp(entropy)
        dominant_share = max(mode_masses.values()) / valid_weight

    invalid_rate = 1.0 - valid_fraction
    all_singleton_modes = (
        valid_count >= 2
        and len(mode_member_counts) == valid_count
        and all(count == 1 for count in mode_member_counts.values())
    )
    ged_at_ceiling = valid_count == n and all_singleton_modes
    severe_rate = float(
        np.dot(
            weights_arr,
            np.asarray([bool(sample["severe_degeneration"]) for sample in samples], dtype=float),
        )
        / total_weight
    )
    hybrid_rate = (
        float(
            np.dot(
                weights_arr[valid_mask],
                np.asarray(
                    [bool(sample["hybrid"]) for sample in samples if bool(sample["valid"])],
                    dtype=float,
                ),
            )
            / valid_weight
        )
        if valid_weight > 0
        else 0.0
    )

    return {
        "group_size": n,
        "geq": geq,
        "ged": float(max(0.0, min(10.0, ged))),
        "invalid_rate": invalid_rate,
        "valid_count": valid_count,
        "represented_mode_count": int(represented_modes),
        "effective_mode_count": float(effective_modes),
        "dominant_mode_share": float(dominant_share),
        "all_singleton_modes": all_singleton_modes,
        "ged_at_ceiling": ged_at_ceiling,
        "severe_degeneration_rate": severe_rate,
        "hybrid_rate": hybrid_rate,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def _bottom_fraction_mean(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    count = max(1, int(math.ceil(len(values) * fraction)))
    return float(statistics.fmean(sorted(values)[:count]))


def flatten_judgments(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return sample-context rows and group rows from wrapper records."""
    sample_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_sample_contexts: set[tuple[str, str, str]] = set()

    for record in records:
        metadata = record.get("request_metadata", {})
        output = record.get("output", record)
        if not isinstance(metadata, Mapping) or not isinstance(output, Mapping):
            raise ValueError("each record requires object request_metadata and output")
        method = str(metadata.get("method_code", "unknown"))
        partition = str(metadata.get("partition_id", "P0"))
        group_id = str(metadata.get("group_id", output.get("group_id", "unknown")))
        group_index = int(metadata.get("group_index", 0))
        if group_id in seen_group_ids:
            raise ValueError(f"duplicate judgment group_id {group_id!r}")
        seen_group_ids.add(group_id)
        sample_map: dict[str, Mapping[str, Any]] = {}
        for item in metadata.get("sample_map", []):
            if not isinstance(item, Mapping):
                continue
            display_id = str(item["display_id"])
            if display_id in sample_map:
                raise ValueError(
                    f"group {group_id}: duplicate display_id {display_id!r} in sample_map"
                )
            sample_map[display_id] = item
        samples = output.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"group {group_id}: output.samples must be a list")

        group_samples: list[Mapping[str, Any]] = []
        seen_display_ids: set[str] = set()
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ValueError(f"group {group_id}: sample output must be an object")
            display_id = str(sample["sample_id"])
            if display_id in seen_display_ids:
                raise ValueError(
                    f"group {group_id}: duplicate output sample_id {display_id!r}"
                )
            seen_display_ids.add(display_id)
            mapping = sample_map.get(display_id, {})
            true_sample_id = str(mapping.get("sample_id", display_id))
            sample_index_raw = mapping.get("sample_index", display_id)
            sample_index = str(sample_index_raw)
            context_key = (method, partition, sample_index)
            if context_key in seen_sample_contexts:
                raise ValueError(
                    "duplicate method/partition/sample_index judgment context "
                    f"{method}/{partition}/{sample_index}"
                )
            seen_sample_contexts.add(context_key)
            row: dict[str, Any] = {
                "method_code": method,
                "partition_id": partition,
                "group_id": group_id,
                "group_index": group_index,
                "display_id": display_id,
                "sample_id": true_sample_id,
                "sample_index": sample_index,
                "text_sha256": mapping.get("text_sha256"),
                "valid": bool(sample["valid"]),
                "invalid_reason": sample.get("invalid_reason"),
                "quality": sample_quality(sample),
                "mode_id": sample.get("mode_id"),
                "hybrid": bool(sample["hybrid"]),
                "severe_degeneration": bool(sample["severe_degeneration"]),
                "prompt_injection_detected": bool(sample["prompt_injection_detected"]),
                "clustering_confidence": output.get("clustering_confidence"),
            }
            for dimension in DIMENSIONS:
                row[dimension] = int(sample["quality"][dimension])
            sample_rows.append(row)
            group_samples.append(sample)

        metrics = compute_group_metrics(group_samples)
        confidence = output.get("clustering_confidence")
        group_rows.append(
            {
                "method_code": method,
                "partition_id": partition,
                "group_id": group_id,
                "group_index": group_index,
                "clustering_confidence": confidence,
                "low_clustering_confidence": confidence == "low",
                "nonhigh_clustering_confidence": confidence in {"low", "medium"},
                **metrics,
            }
        )

    return sample_rows, group_rows


def aggregate_systems(
    sample_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    samples_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    groups_by_method: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        samples_by_method[str(row["method_code"])].append(row)
    for row in group_rows:
        groups_by_method[str(row["method_code"])].append(row)

    summaries: list[dict[str, Any]] = []
    for method in sorted(samples_by_method):
        rows = samples_by_method[method]
        by_sample: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_sample[str(row["sample_index"])].append(row)

        sample_quality_means = [
            statistics.fmean(float(row["quality"]) for row in contexts)
            for contexts in by_sample.values()
        ]
        invalid_context = [1.0 - float(row["valid"]) for row in rows]
        severe_context = [float(row["severe_degeneration"]) for row in rows]
        group_method_rows = groups_by_method.get(method, [])

        summary: dict[str, Any] = {
            "method_code": method,
            "unique_samples": len(by_sample),
            "judged_sample_contexts": len(rows),
            "groups": len(group_method_rows),
            "partitions": len({str(row["partition_id"]) for row in rows}),
            "geq": float(statistics.fmean(sample_quality_means)),
            "ged": float(statistics.fmean(float(row["ged"]) for row in group_method_rows)),
            "invalid_rate": float(statistics.fmean(invalid_context)),
            "quality_p10": _quantile(sample_quality_means, 0.10),
            "bottom_quartile_quality": _bottom_fraction_mean(sample_quality_means, 0.25),
            "severe_degeneration_rate": float(statistics.fmean(severe_context)),
            "mean_effective_mode_count": float(
                statistics.fmean(float(row["effective_mode_count"]) for row in group_method_rows)
            ),
            "mean_represented_mode_count": float(
                statistics.fmean(
                    float(row["represented_mode_count"]) for row in group_method_rows
                )
            ),
            "mean_dominant_mode_share": float(
                statistics.fmean(float(row["dominant_mode_share"]) for row in group_method_rows)
            ),
            "all_singleton_group_rate": float(
                statistics.fmean(
                    float(bool(row["all_singleton_modes"])) for row in group_method_rows
                )
            ),
            "ged_ceiling_rate": float(
                statistics.fmean(
                    float(bool(row["ged_at_ceiling"])) for row in group_method_rows
                )
            ),
            "low_clustering_confidence_rate": float(
                statistics.fmean(
                    float(bool(row["low_clustering_confidence"]))
                    for row in group_method_rows
                )
            ),
            "nonhigh_clustering_confidence_rate": float(
                statistics.fmean(
                    float(bool(row["nonhigh_clustering_confidence"]))
                    for row in group_method_rows
                )
            ),
            "hybrid_rate": float(statistics.fmean(float(row["hybrid"]) for row in rows if row["valid"]))
            if any(bool(row["valid"]) for row in rows)
            else 0.0,
        }
        for dimension in DIMENSIONS:
            summary[f"mean_{dimension}"] = float(
                statistics.fmean(float(row[dimension]) for row in rows)
            )
        summaries.append(summary)
    return summaries


def pairwise_win_rate(decisions: Iterable[str]) -> dict[str, float | int]:
    normalized = [decision.lower() for decision in decisions]
    wins = sum(decision in {"a", "win", "left"} for decision in normalized)
    losses = sum(decision in {"b", "loss", "right"} for decision in normalized)
    ties = sum(decision in {"tie", "inconsistent"} for decision in normalized)
    total = wins + losses + ties
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "comparisons": total,
        "win_rate": (wins + 0.5 * ties) / total if total else float("nan"),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=None,
        help="Optional descriptive GED threshold; no calibrated threshold is supplied",
    )
    args = parser.parse_args()

    records = read_jsonl(args.judgments)
    sample_rows, group_rows = flatten_judgments(records)
    system_rows = aggregate_systems(sample_rows, group_rows)

    if args.sufficiency_threshold is not None:
        for row in system_rows:
            row["ged_point_above_threshold"] = float(row["ged"]) >= args.sufficiency_threshold

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "sample_scores.csv", sample_rows)
    write_csv(args.out_dir / "group_scores.csv", group_rows)
    write_csv(args.out_dir / "system_scores.csv", system_rows)
    summary = {
        "protocol_version": "1.0.0",
        "judgments": str(args.judgments),
        "systems": system_rows,
        "notes": [
            "GEQ is the sample-mean score after averaging repeated partition contexts.",
            "GED is the equal-weight mean of fixed-size group GED values.",
            "Report GED alongside GEQ and mode diagnostics; no calibrated sufficiency threshold is supplied.",
            "All-singleton and ceiling rates diagnose whether local GED has saturated.",
            "Point thresholds are descriptive. Bootstrap GEQ intervals are conditional; GED weighted ranges are sensitivity summaries.",
        ],
    }
    (args.out_dir / "scores.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
