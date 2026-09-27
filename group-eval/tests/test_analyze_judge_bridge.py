#!/usr/bin/env python3
"""Focused synthetic tests for the strict three-judge bridge combiner."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiment" / "analyze_judge_bridge.py"
FIXTURE = ROOT / "results" / "mdlm_pilot_gpt_oss_120b_20260725"

BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "groupeval_analyze_judge_bridge", SCRIPT
)
assert BRIDGE_SPEC is not None and BRIDGE_SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(BRIDGE_SPEC)
sys.modules[BRIDGE_SPEC.name] = BRIDGE
BRIDGE_SPEC.loader.exec_module(BRIDGE)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class JudgeBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="groupeval-bridge-test-")
        self.root = Path(self.temp.name)
        self.base_summary = json.loads(
            (FIXTURE / "pilot_summary.json").read_text(encoding="utf-8")
        )
        self.base_summary["judge_diagnostics"]["protocol_identity"] = {
            "protocol_version": "1.0.0",
            "protocol_sha256": "1" * 64,
            "target_profile_id": "owt1024-fullstream-v1",
            "target_profile_sha256": "2" * 64,
            "prompt_sha256": "3" * 64,
            "schema_sha256": "4" * 64,
            "strict_json_schema": True,
            "local_validation_uses_unmodified_schema": True,
            "tools_enabled": False,
            "provider_schema_omitted_keywords": ["uniqueItems"],
        }
        with (FIXTURE / "pilot_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            self.csv_fields = list(reader.fieldnames or [])
            self.base_csv = list(reader)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_analysis(
        self,
        name: str,
        model: str,
        index: int,
        *,
        reverse: bool = False,
        source_shift: float = 0.0,
        protocol_shift: bool = False,
    ) -> Path:
        directory = self.root / name
        directory.mkdir()
        summary = copy.deepcopy(self.base_summary)
        summary["judge_diagnostics"]["judge_model"] = model
        summary["judge_diagnostics"]["system_fingerprint"] = f"fixture-{index}"
        if protocol_shift:
            summary["judge_diagnostics"]["protocol_identity"][
                "prompt_sha256"
            ] = "f" * 64
        summary["input_artifacts"]["judgments"]["sha256"] = f"{index + 1:064x}"
        csv_rows = copy.deepcopy(self.base_csv)
        for row_index, (summary_row, csv_row) in enumerate(
            zip(summary["results"], csv_rows, strict=True)
        ):
            metric_index = 9 - row_index if reverse else row_index
            geq = float(metric_index)
            ged = float(9 - metric_index)
            summary_row["geq"] = geq
            summary_row["ged"] = ged
            csv_row["geq"] = str(geq)
            csv_row["ged"] = str(ged)
            if source_shift and row_index == 0:
                shifted = (
                    float(summary_row["gpt2_generative_perplexity"]) + source_shift
                )
                summary_row["gpt2_generative_perplexity"] = shifted
                csv_row["gpt2_generative_perplexity"] = str(shifted)
        result_by_key = {
            (int(row["updates_M"]), str(row["operating_point"])): row
            for row in summary["results"]
        }
        for comparison in summary["within_budget_comparisons"]:
            budget = int(comparison["updates_M"])
            for metric in ("geq", "ged"):
                comparison[f"delta_{metric}"] = (
                    float(result_by_key[(budget, "selected")][metric])
                    - float(result_by_key[(budget, "reference")][metric])
                )
        (directory / "pilot_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(directory / "pilot_results.csv", csv_rows, self.csv_fields)
        return directory

    def run_bridge(
        self,
        directories: list[tuple[str, Path]],
        output: Path,
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT)]
        for alias, directory in directories:
            command.extend(["--input", f"{alias}={directory}"])
        command.extend(["--output-dir", str(output)])
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_success_emits_separate_scores_and_sensitivity(self) -> None:
        first = self.make_analysis("first", "judge:model-a", 0)
        second = self.make_analysis("second", "judge:model-b", 1)
        third = self.make_analysis("third", "judge:model-c", 2, reverse=True)
        output = self.root / "bridge"
        process = self.run_bridge(
            [("Judge A", first), ("Judge B", second), ("Judge C", third)],
            output,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        summary = json.loads(
            (output / "judge_bridge_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(summary["results"]), 10)
        self.assertIn("no pooling", summary["analysis"]["score_handling"])
        self.assertEqual(
            summary["common_judge_protocol_identity"]["prompt_sha256"],
            "3" * 64,
        )
        pairs = summary["analysis"]["sensitivity_statistics"][
            "pairwise_spearman_across_cells"
        ]
        self.assertEqual(pairs[0]["geq"]["rho"], 1.0)
        self.assertEqual(pairs[-1]["geq"]["rho"], -1.0)
        signs = summary["analysis"]["sensitivity_statistics"][
            "selected_gt_reference_sign_counts"
        ]
        self.assertEqual(signs["Judge A"]["geq"]["selected_gt_reference_count"], 5)
        self.assertEqual(signs["Judge A"]["ged"]["selected_gt_reference_count"], 0)
        self.assertEqual(signs["Judge C"]["geq"]["selected_gt_reference_count"], 0)
        table = (output / "judge_bridge_table.tex").read_text(encoding="utf-8")
        self.assertNotIn(r"\Delta", table)
        self.assertNotIn("delta", table.lower())
        self.assertIn(r"\begin{tabular}{rcccccccc}", table)
        self.assertIn(r"\cmidrule(lr){8-9}", table)
        csv_header = (output / "judge_bridge.csv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        self.assertIn("judge_a_geq", csv_header)
        self.assertIn("judge_c_ged", csv_header)

    def test_spearman_uses_average_ranks_and_handles_constants(self) -> None:
        self.assertEqual(BRIDGE.average_ranks([1.0, 1.0, 2.0, 3.0]), [1.5, 1.5, 3.0, 4.0])
        self.assertAlmostEqual(
            BRIDGE.spearman(
                [1.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 2.0, 3.0],
            ),
            5.0 / 6.0,
        )
        self.assertIsNone(
            BRIDGE.spearman(
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
            )
        )

    def test_rejects_duplicate_underlying_judge_alias(self) -> None:
        first = self.make_analysis("first", "judge:model-a", 0)
        second = self.make_analysis("second", "judge:model-a", 1)
        third = self.make_analysis("third", "judge:model-c", 2)
        process = self.run_bridge(
            [("Judge A", first), ("Judge B", second), ("Judge C", third)],
            self.root / "duplicate-model",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("judge model aliases must be distinct", process.stderr)

    def test_rejects_source_metric_mismatch(self) -> None:
        first = self.make_analysis("first", "judge:model-a", 0)
        second = self.make_analysis("second", "judge:model-b", 1)
        third = self.make_analysis(
            "third", "judge:model-c", 2, source_shift=0.25
        )
        process = self.run_bridge(
            [("Judge A", first), ("Judge B", second), ("Judge C", third)],
            self.root / "source-mismatch",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("source gPPL/entropy do not exactly match", process.stderr)

    def test_rejects_protocol_identity_mismatch(self) -> None:
        first = self.make_analysis("first", "judge:model-a", 0)
        second = self.make_analysis(
            "second",
            "judge:model-b",
            1,
            protocol_shift=True,
        )
        third = self.make_analysis("third", "judge:model-c", 2)
        process = self.run_bridge(
            [("Judge A", first), ("Judge B", second), ("Judge C", third)],
            self.root / "protocol-mismatch",
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("does not exactly match", process.stderr)
        self.assertIn("prompt, schema, protocol", process.stderr)


if __name__ == "__main__":
    unittest.main()
