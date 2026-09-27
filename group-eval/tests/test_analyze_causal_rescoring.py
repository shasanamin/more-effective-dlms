#!/usr/bin/env python3
"""Synthetic tests for the reviewer-safe causal-rescoring analyzer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "experiment" / "analyze_causal_rescoring.py"
SPEC = importlib.util.spec_from_file_location("analyze_causal_rescoring", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def synthetic_summary(spec: object) -> dict[str, object]:
    series_offset = {
        "gpt2_large_full_stream": 1.0,
        "gpt2_large_first_document": 1.1,
        "qwen3_8b_full_stream": 0.9,
        "qwen3_8b_first_document": 1.2,
    }[spec.series_id]
    is_full = spec.span_policy == "full_stream_segmented"
    is_qwen = spec.evaluator == "Qwen3-8B-Base"
    cells: list[dict[str, object]] = []
    for index, expected in enumerate(MODULE.CELL_SPECS):
        source_bytes = 1000 + index
        source_chars = 980 + index
        included_bytes = source_bytes if is_full else 500 + index
        included_chars = source_chars if is_full else 490 + index
        scored_bytes = included_bytes - int(is_qwen)
        scored_chars = included_chars - int(is_qwen)
        source_boundaries = 20
        included_boundaries = source_boundaries if is_full else 10
        scored_boundaries = included_boundaries - int(is_qwen)
        token_count = 500 + index
        perplexity = series_offset * (10.0 + index)
        token_mean = math.log(perplexity)
        nll_sum = token_mean * token_count
        bpb = 1.0 + 0.01 * index + (0.02 if is_qwen else 0.0)
        content_nll = bpb * math.log(2.0) * scored_bytes
        macro_ppl = perplexity * 1.05
        macro_mean = math.log(macro_ppl)
        cell = {
            **expected,
            "expected_count": 40,
            "source_sha256": digest(str(expected["cell_id"])),
            "output_count": 40,
            "scorable_output_count": 40,
            "source_segment_count": 60,
            "included_segment_count": 60 if is_full else 40,
            "scorable_segment_count": 50 if is_full else 40,
            "source_boundary_count": source_boundaries,
            "included_boundary_count": included_boundaries,
            "scored_boundary_count": scored_boundaries,
            "truncated_segment_count": int(is_qwen),
            "leading_sentinel_removed_count": 39,
            "raw_character_count": source_chars + 80,
            "raw_utf8_bytes": source_bytes + 100,
            "source_content_character_count": source_chars,
            "source_content_utf8_bytes": source_bytes,
            "included_content_character_count": included_chars,
            "included_content_utf8_bytes": included_bytes,
            "scored_content_character_count": scored_chars,
            "scored_content_utf8_bytes": scored_bytes,
            "scored_evaluator_token_count": token_count,
            "nll_sum": nll_sum,
            "content_nll_sum_for_bpb": content_nll,
            "token_micro_mean_nll": token_mean,
            "token_micro_perplexity": perplexity,
            "output_macro_mean_nll": macro_mean,
            "output_macro_geometric_perplexity": macro_ppl,
            "output_macro_mean_nll_bootstrap_95": [macro_mean - 0.1, macro_mean + 0.1],
            "output_macro_geometric_perplexity_bootstrap_95": [
                math.exp(macro_mean - 0.1),
                math.exp(macro_mean + 0.1),
            ],
            "corpus_bits_per_utf8_byte": bpb,
        }
        cells.append(cell)

    overall: dict[str, object] = {
        field: sum(int(cell[field]) for cell in cells)
        for field in MODULE.ADDITIVE_FIELDS
    }
    total_tokens = int(overall["scored_evaluator_token_count"])
    total_nll = sum(float(cell["nll_sum"]) for cell in cells)
    total_content_nll = sum(float(cell["content_nll_sum_for_bpb"]) for cell in cells)
    total_bytes = int(overall["scored_content_utf8_bytes"])
    macro_mean = sum(float(cell["output_macro_mean_nll"]) for cell in cells) / len(cells)
    overall.update(
        {
            "nll_sum": total_nll,
            "content_nll_sum_for_bpb": total_content_nll,
            "token_micro_mean_nll": total_nll / total_tokens,
            "token_micro_perplexity": math.exp(total_nll / total_tokens),
            "output_macro_mean_nll": macro_mean,
            "output_macro_geometric_perplexity": math.exp(macro_mean),
            "output_macro_mean_nll_bootstrap_95": [macro_mean - 0.1, macro_mean + 0.1],
            "output_macro_geometric_perplexity_bootstrap_95": [
                math.exp(macro_mean - 0.1),
                math.exp(macro_mean + 0.1),
            ],
            "corpus_bits_per_utf8_byte": total_content_nll / (math.log(2.0) * total_bytes),
        }
    )
    return {
        "artifact": MODULE.EXPECTED_ARTIFACT,
        "protocol_version": MODULE.PROTOCOL_VERSION,
        "input_manifest_sha256": digest("manifest"),
        "private_per_output_artifact_sha256": digest(spec.series_id),
        "total_cells": 20,
        "model": {
            "identifier": spec.model_identifier,
            "requested_revision": spec.revision,
            "resolved_model_revision": spec.revision,
            "resolved_tokenizer_revision": None,
        },
        "scoring": {
            "span_policy": spec.span_policy,
            "max_evaluator_tokens_per_segment_including_primer": 1024,
            "bootstrap_unit": "output",
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 20260731,
            "add_special_tokens": False,
            "common_contract_marker": "identical",
        },
        "overall": overall,
        "cells": cells,
    }


class AnalyzerTest(unittest.TestCase):
    def write_inputs(self, root: Path) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for spec in MODULE.SERIES_SPECS:
            path = root / f"{spec.series_id}.json"
            path.write_text(json.dumps(synthetic_summary(spec)), encoding="utf-8")
            paths[spec.series_id] = path
        return paths

    def make_args(
        self,
        paths: dict[str, Path],
        output_dir: Path,
        group_eval_csv: Path | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            gpt2_full_summary=paths["gpt2_large_full_stream"],
            gpt2_first_summary=paths["gpt2_large_first_document"],
            qwen_full_summary=paths["qwen3_8b_full_stream"],
            qwen_first_summary=paths["qwen3_8b_first_document"],
            group_eval_csv=group_eval_csv,
            output_dir=output_dir,
        )

    def write_group_eval(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("model", "operating_point", "updates_M", "geq"),
                lineterminator="\n",
            )
            writer.writeheader()
            for index, cell in enumerate(MODULE.CELL_SPECS):
                writer.writerow(
                    {
                        "model": cell["model_family"],
                        "operating_point": cell["operating_point"],
                        "updates_M": cell["steps"],
                        "geq": 9.5 - 0.1 * index,
                    }
                )

    def test_emits_complete_artifact_without_group_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_inputs(root)
            output = root / "analysis"
            result = MODULE.analyze(self.make_args(paths, output))
            self.assertEqual(result["cell_metric_rows"], 80)
            self.assertEqual(result["matched_rows"], 40)
            self.assertEqual(result["rank_correlation_rows"], 72)
            self.assertEqual(result["geq_association_rows"], 0)
            for filename in (
                "cell_metrics.csv",
                "matched_mdlm_didi.csv",
                "matched_sign_summary.csv",
                "rank_correlations.csv",
                "diagnostic_coverage.csv",
                "geq_gppl_associations.csv",
                "summary.json",
                "results.md",
                "tables.tex",
                "SHA256SUMS",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(summary["group_eval_input"])
            self.assertEqual(len(summary["cell_metrics"]), 80)
            self.assertIn("explicitly pending", (output / "results.md").read_text())

    def test_group_eval_associations_include_null_or_adverse_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_inputs(root)
            group_eval = root / "cell_results.csv"
            self.write_group_eval(group_eval)
            output = root / "analysis"
            result = MODULE.analyze(self.make_args(paths, output, group_eval))
            self.assertEqual(result["geq_association_rows"], 48)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            primary_pooled = [
                row
                for row in summary["geq_gppl_associations"]
                if row["series_id"] == "gpt2_large_full_stream"
                and row["gppl_metric"] == "token_micro_perplexity"
                and row["subset"] == "primary_M_ge_16"
                and row["model_scope"] == "pooled_unadjusted"
            ]
            self.assertEqual(len(primary_pooled), 1)
            self.assertAlmostEqual(
                primary_pooled[0]["spearman_rho_geq_vs_log_gppl"], -1.0, places=12
            )
            self.assertIn("descriptive", primary_pooled[0]["inference_scope"])

    def test_cross_summary_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_inputs(root)
            path = paths["qwen3_8b_first_document"]
            value = json.loads(path.read_text(encoding="utf-8"))
            value["cells"][0]["source_sha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "common source_sha256"):
                MODULE.analyze(self.make_args(paths, root / "analysis"))

    def test_wrong_evaluator_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.write_inputs(root)
            path = paths["qwen3_8b_full_stream"]
            value = json.loads(path.read_text(encoding="utf-8"))
            value["model"]["requested_revision"] = "0" * 40
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requested_revision"):
                MODULE.analyze(self.make_args(paths, root / "analysis"))


if __name__ == "__main__":
    unittest.main()
