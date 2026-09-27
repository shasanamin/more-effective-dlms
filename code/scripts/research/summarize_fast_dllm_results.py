#!/usr/bin/env python3
"""Summarize scratch Fast-dLLM eval outputs into a compact CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_metadata(path: Path) -> dict[str, str]:
    meta = {}
    if not path.exists():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key] = value
    return meta


def latest_result_json(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "lm_eval").glob("**/results_*.json"))
    return candidates[-1] if candidates else None


def read_metrics(path: Path | None) -> dict[str, str | float | int]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    task = next(iter(data.get("results", {}) or {}), "")
    task_metrics = data.get("results", {}).get(task, {}) if task else {}
    metrics: dict[str, str | float | int] = {
        "task_from_json": task,
        "effective_samples": data.get("n-samples", {}).get(task, {}).get("effective", ""),
        "eval_seconds": data.get("total_evaluation_time_seconds", ""),
    }
    for key, value in task_metrics.items():
        if key == "alias":
            continue
        metrics[key.replace(",", "__")] = value
    return metrics


def read_gpu_monitor(path: Path) -> dict[str, str | float]:
    if not path.exists():
        return {}
    max_mem = 0.0
    max_util = 0.0
    max_power = 0.0
    rows = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            try:
                max_util = max(max_util, float(row["utilization_gpu_pct"]))
                max_mem = max(max_mem, float(row["memory_used_mib"]))
                max_power = max(max_power, float(row["power_draw_w"]))
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "gpu_samples": rows,
        "max_gpu_util_pct": max_util,
        "max_gpu_mem_mib": max_mem,
        "max_power_w": max_power,
    }


def summarize(root: Path) -> list[dict[str, str | float | int]]:
    rows = []
    if (root / "run_metadata.txt").exists() or (root / "lm_eval").exists():
        run_dirs = [root]
    else:
        run_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for run_dir in run_dirs:
        meta = read_metadata(run_dir / "run_metadata.txt")
        result_json = latest_result_json(run_dir)
        row: dict[str, str | float | int] = {"run": run_dir.name}
        row.update(meta)
        row.update(read_metrics(result_json))
        row.update(read_gpu_monitor(run_dir / "gpu_monitor.csv"))
        row["result_json"] = str(result_json) if result_json else ""
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Run-group directory, e.g. /large-storage/fast_dllm/evals/run-group",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize(args.root)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    out_f = args.output.open("w", newline="") if args.output else None
    try:
        writer = csv.DictWriter(out_f or __import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()
