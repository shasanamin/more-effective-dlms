#!/usr/bin/env python3
"""Validate and score the synthetic fixture, then render one request offline."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = (args.output_dir or Path(tempfile.mkdtemp(prefix="groupeval-example-"))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def run(*arguments: object) -> None:
        subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, check=True)

    run("scoring/validate_outputs.py", "--input", "examples/example_judgments.jsonl")
    run("scoring/score_group_eval.py", "--judgments", "examples/example_judgments.jsonl",
        "--out-dir", out / "scores")
    expected = json.loads((ROOT / "examples/example_scores.json").read_text())["systems"][0]
    observed = json.loads((out / "scores/scores.json").read_text())["systems"][0]
    for metric in ("geq", "ged"):
        if not math.isclose(observed[metric], expected[metric], rel_tol=1e-12):
            raise ValueError(f"Unexpected {metric}: {observed[metric]}")

    fixture = json.loads((ROOT / "examples/example_judgments.jsonl").read_text())
    group = json.loads((ROOT / "examples/example_input_group.json").read_text())
    groups_path = out / "groups.jsonl"
    groups_path.write_text(json.dumps({"request_metadata": fixture["request_metadata"],
                                      "group_input": group}) + "\n", encoding="utf-8")
    run("experiment/judge_runner_template.py", "--groups", groups_path,
        "--output", out / "request_preview.jsonl", "--provider", "inst",
        "--model", "gpt-oss:120b", "--target-profile", "protocol/target_profile_owt1024.md",
        "--dry-run")
    print(f"Offline example passed. GEQ={observed['geq']:.5f}; GED={observed['ged']:.7f}.")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
