#!/usr/bin/env python3
"""Unit checks for causal-LM span handling; no checkpoint is loaded."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Optional causal-LM accounting tests require torch") from exc


SCRIPT = Path(__file__).parents[1] / "experiment" / "rescore_causal_lm.py"
SPEC = importlib.util.spec_from_file_location("rescore_causal_lm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    """Small fast-tokenizer stand-in with one token per Python character."""

    is_fast = True
    bos_token_id = 100
    eos_token_id = 101
    pad_token_id = 101

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        return {
            "input_ids": [index + 1 for index, _ in enumerate(text)],
            "offset_mapping": [(index, index + 1) for index, _ in enumerate(text)],
        }


class UniformToyModel:
    """Returns uniform next-token logits for deterministic accounting tests."""

    def __call__(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        shape = (*input_ids.shape, 128)
        return SimpleNamespace(logits=torch.zeros(shape, device=input_ids.device))


class SpanPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()

    def test_full_stream_resets_and_preserves_terminal_empty_fragment(self) -> None:
        sentinel = MODULE.GENERATOR_SENTINEL
        encoded = MODULE.encode_output(
            f"{sentinel}ab{sentinel}{sentinel}c{sentinel}",
            self.tokenizer,
            max_length=16,
            span_policy="full_stream_segmented",
        )
        self.assertTrue(encoded.leading_sentinel_removed)
        self.assertEqual(encoded.source_boundary_count, 3)
        self.assertEqual(encoded.source_segment_count, 4)
        self.assertEqual(encoded.included_segment_count, 4)
        self.assertEqual(
            [segment.source_boundary_after for segment in encoded.segments],
            [True, True, True, False],
        )
        self.assertEqual(
            [segment.scored_content_token_count for segment in encoded.segments],
            [2, 0, 1, 0],
        )
        self.assertEqual(sum(segment.boundary_scored for segment in encoded.segments), 3)
        self.assertEqual(encoded.segments[1].input_ids, [100, 101])
        self.assertEqual(encoded.segments[-1].input_ids, [100])

    def test_first_document_keeps_only_first_generated_boundary(self) -> None:
        sentinel = MODULE.GENERATOR_SENTINEL
        encoded = MODULE.encode_output(
            f"{sentinel}ab{sentinel}cd{sentinel}ef",
            self.tokenizer,
            max_length=16,
            span_policy="first_document",
        )
        self.assertEqual(encoded.source_segment_count, 3)
        self.assertEqual(encoded.included_segment_count, 1)
        self.assertEqual(encoded.source_boundary_count, 2)
        self.assertEqual(encoded.included_boundary_count, 1)
        self.assertEqual(encoded.segments[0].input_ids, [100, 1, 2, 101])

    def test_limit_is_per_segment_and_boundary_is_not_skipped(self) -> None:
        encoded = MODULE.encode_segment(
            MODULE.SourceSegment(content="abcd", boundary_after=True),
            self.tokenizer,
            max_length=4,
        )
        self.assertEqual(encoded.input_ids, [100, 1, 2, 3])
        self.assertEqual(encoded.scored_content_utf8_bytes, 3)
        self.assertFalse(encoded.boundary_scored)
        self.assertTrue(encoded.truncated)

    def test_unicode_partial_offset_cut_backs_off(self) -> None:
        offsets = [(0, 1), (1, 2), (1, 2), (2, 3)]
        self.assertEqual(MODULE._avoid_partial_character_cut(2, offsets), 1)

    def test_segment_scores_aggregate_back_to_one_output(self) -> None:
        sentinel = MODULE.GENERATOR_SENTINEL
        encoded = MODULE.encode_output(
            f"ab{sentinel}",
            self.tokenizer,
            max_length=16,
            span_policy="full_stream_segmented",
        )
        flattened = [
            (0, segment_index, segment)
            for segment_index, segment in enumerate(encoded.segments)
        ]
        scores = MODULE._score_segments(
            flattened,
            UniformToyModel(),
            self.tokenizer,
            device="cpu",
            batch_size=2,
        )
        row = MODULE._output_row("cell", 0, encoded, scores)
        summary = MODULE._summarize_rows([row], 1000, 7)
        self.assertEqual(row["token_count"], 3)
        self.assertEqual(row["scorable_segment_count"], 1)
        self.assertAlmostEqual(row["perplexity"], 128.0, places=4)
        self.assertAlmostEqual(row["content_bits_per_utf8_byte"], 7.0, places=5)
        self.assertAlmostEqual(summary["token_micro_perplexity"], 128.0, places=4)
        self.assertAlmostEqual(
            summary["output_macro_geometric_perplexity"], 128.0, places=4
        )
        self.assertAlmostEqual(summary["corpus_bits_per_utf8_byte"], 7.0, places=5)


class PortablePathTest(unittest.TestCase):
    def test_cli_root_overrides_environment_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest = base / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"OUTPUT_ROOT": str(base / "wrong")}):
                resolved = MODULE.resolve_source_path(
                    "${OUTPUT_ROOT}/cell.json", manifest, base / "right"
                )
            self.assertEqual(resolved, (base / "right" / "cell.json").resolve())

    def test_only_output_root_placeholder_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            with self.assertRaisesRegex(ValueError, "only.*OUTPUT_ROOT"):
                MODULE.resolve_source_path("${HOME}/cell.json", manifest, None)

    def test_source_hash_mismatch_fails(self) -> None:
        cell = {
            "cell_id": "cell",
            "source_sha256_expected": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            MODULE.verify_source_sha256(cell, Path("cell.json"), b"not expected")


if __name__ == "__main__":
    unittest.main()
