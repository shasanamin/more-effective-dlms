#!/usr/bin/env python3
"""Focused tests for the conditional large-DLM GroupEval path."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiment"
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))

from analyze_large_dlm_conditional import (  # noqa: E402
    auc_binary,
    pass_fail_association,
    validate_frozen_provenance,
    validate_judgments,
    verify_prepared_bundle,
)
from conditional_large_dlm_common import (  # noqa: E402
    EVALUATION_PROTOCOL,
    PROTOCOL_VERSION,
    conditional_geq,
    conditional_semantic_errors,
)
from judge_runner_template import (  # noqa: E402
    parse_prompt,
    render_request,
    strip_profile_title,
    validate_output,
)
from prepare_large_dlm_conditional import (  # noqa: E402
    EXPECTED_SOURCE_SHA256,
    MODEL_CHECKPOINTS,
    MODEL_REVISIONS,
)


class ConditionalLargeDLMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "group_id": "CGE-test",
            "task_name": "gsm8k",
            "condition_id": "gsm8k:0",
            "sample_map": [
                {"display_id": "S1", "model_code": "llada"},
                {"display_id": "S2", "model_code": "dream"},
            ],
        }
        self.output = {
            "protocol_version": PROTOCOL_VERSION,
            "group_id": "CGE-test",
            "samples": [
                {
                    "sample_id": sample_id,
                    "valid": True,
                    "invalid_reason": None,
                    "quality": {
                        "readability": 2,
                        "reasoning_coherence": 1,
                        "task_success": 2,
                        "nondegeneration_structure": 2,
                    },
                    "evidence": ["coherent candidate"],
                    "severe_degeneration": False,
                    "prompt_injection_detected": False,
                }
                for sample_id in ("S1", "S2")
            ],
            "group_warnings": [],
        }

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _run_metadata(
        self,
        *,
        fingerprint: str | None = "test-fingerprint",
        response_model: str | None = "gpt-oss:120b",
    ) -> dict[str, object]:
        return {
            "judge_provider": "institutional_genai",
            "judge_model_snapshot": "gpt-oss:120b",
            "judge_endpoint": "https://example.invalid/api/chat/completions",
            "provider_response_model": response_model,
            "system_fingerprint": fingerprint,
            "model_identifier_is_provider_alias": True,
            "reasoning_effort": "provider_default",
            "tools_enabled": False,
            "strict_json_schema": True,
            "provider_schema_omitted_keywords": ["uniqueItems"],
            "provider_schema_normalizations": [],
            "local_validation_uses_unmodified_schema": True,
            "prompt_sha256": "0" * 64,
            "schema_sha256": "1" * 64,
            "target_profile_sha256": "2" * 64,
        }

    def test_conditional_score_and_semantics(self) -> None:
        self.assertEqual(conditional_geq(self.output["samples"][0]), 8.75)
        self.assertEqual(
            conditional_semantic_errors(
                {"request_metadata": self.metadata, "output": self.output}
            ),
            [],
        )
        invalid = json.loads(json.dumps(self.output))
        invalid["samples"][0]["valid"] = False
        invalid["samples"][0]["invalid_reason"] = "no_interpretable_task_response"
        self.assertTrue(
            any(
                "zero" in message
                for message in conditional_semantic_errors(
                    {"request_metadata": self.metadata, "output": invalid}
                )
            )
        )

    def test_schema_and_runner_use_conditional_validator(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/conditional_geq_output.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            validate_output(
                self.output,
                self.metadata,
                Draft202012Validator(schema),
            ),
            [],
        )

    def test_condition_is_rendered_separately_and_candidates_remain_blinded(self) -> None:
        system, user = parse_prompt(ROOT / "prompts/group_eval_conditional_task.md")
        profile = strip_profile_title(
            (ROOT / "protocol/target_profile_large_dlm_tasks.md").read_text(
                encoding="utf-8"
            )
        )
        record = {
            "request_metadata": self.metadata,
            "group_input": {
                "protocol_version": PROTOCOL_VERSION,
                "group_id": "CGE-test",
                "target_profile_id": "large-dlm-tasks-v1",
                "task_name": "gsm8k",
                "condition": {
                    "condition_id": "gsm8k:0",
                    "text": "What is 2 + 2?",
                },
                "samples": [
                    {"sample_id": "S1", "response": "The answer is 4."},
                    {"sample_id": "S2", "response": "It is 5."},
                ],
            },
        }
        rendered_system, rendered_user = render_request(record, system, user, profile)
        self.assertIn("GROUPEVAL_CONDITION", rendered_user)
        self.assertIn("GROUPEVAL_DATA", rendered_user)
        self.assertLess(
            rendered_user.index("GROUPEVAL_CONDITION"),
            rendered_user.index("GROUPEVAL_DATA"),
        )
        combined = rendered_system + rendered_user
        self.assertNotIn("LLaDA", combined)
        self.assertNotIn("Dream-v0", combined)
        self.assertIn("Do not cluster responses or report diversity/GED", rendered_system)

    def test_auc_and_association_handle_ties(self) -> None:
        labels = np.array([False, False, True, True])
        scores = np.array([0.0, 5.0, 5.0, 10.0])
        self.assertAlmostEqual(auc_binary(labels, scores), 0.875)
        self.assertAlmostEqual(pass_fail_association(labels, scores), 5.0)

    def test_frozen_bundle_and_resource_hashes_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            resources = bundle / "resources"
            resources.mkdir()
            prompt = resources / "prompt.md"
            input_schema = resources / "input.json"
            output_schema = resources / "output.json"
            profile = resources / "profile.md"
            for path, text in (
                (prompt, "prompt\n"),
                (input_schema, "{}\n"),
                (output_schema, "{}\n"),
                (profile, "profile\n"),
            ):
                path.write_text(text, encoding="utf-8")
            provenance = {
                "protocol_version": PROTOCOL_VERSION,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "model_checkpoints": MODEL_CHECKPOINTS,
                "cached_model_revisions": MODEL_REVISIONS,
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "generation_operating_point": {
                    "temperature": 0.0,
                    "generation_seed": 0,
                    "maximum_new_tokens": 256,
                    "baseline_diffusion_steps": 256,
                    "block_length": 32,
                },
                "privacy": {
                    "judge_inputs_include_reference_answers": False,
                    "judge_inputs_include_humaneval_tests": False,
                    "judge_inputs_include_canonical_solutions": False,
                    "binary_labels_stored_only_in_private_label_map": True,
                },
                "resource_sha256": {
                    "prompt": self._sha(prompt),
                    "input_schema": self._sha(input_schema),
                    "output_schema": self._sha(output_schema),
                    "target_profile": self._sha(profile),
                },
            }
            artifacts = {
                "conditional_groups.jsonl": "{}\n",
                "private_label_map.json": "{}\n",
                "generation_provenance.json": json.dumps(provenance) + "\n",
                "preparation_summary.json": "{}\n",
            }
            for name, text in artifacts.items():
                (bundle / name).write_text(text, encoding="utf-8")
            (bundle / "SHA256SUMS").write_text(
                "".join(
                    f"{self._sha(bundle / name)}  {name}\n" for name in artifacts
                ),
                encoding="utf-8",
            )
            fixture_hashes = {
                name: self._sha(bundle / name) for name in artifacts
            }
            verify_prepared_bundle(
                bundle / "conditional_groups.jsonl",
                bundle / "private_label_map.json",
                bundle / "generation_provenance.json",
                bundle / "preparation_summary.json",
                bundle / "SHA256SUMS",
                expected_hashes=fixture_hashes,
            )
            fixture_hashes["conditional_groups.jsonl"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "pinned frozen bundle"):
                verify_prepared_bundle(
                    bundle / "conditional_groups.jsonl",
                    bundle / "private_label_map.json",
                    bundle / "generation_provenance.json",
                    bundle / "preparation_summary.json",
                    bundle / "SHA256SUMS",
                    expected_hashes=fixture_hashes,
                )
            observed = validate_frozen_provenance(
                bundle / "generation_provenance.json",
                prompt,
                input_schema,
                output_schema,
                profile,
            )
            self.assertEqual(observed["prompt"], self._sha(prompt))
            prompt.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt hash mismatch"):
                validate_frozen_provenance(
                    bundle / "generation_provenance.json",
                    prompt,
                    input_schema,
                    output_schema,
                    profile,
                )

    def test_judgment_serving_identity_must_be_consistent(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/conditional_geq_output.schema.json").read_text(
                encoding="utf-8"
            )
        )
        metadata_two = json.loads(json.dumps(self.metadata))
        metadata_two["group_id"] = "CGE-test-2"
        output_two = json.loads(json.dumps(self.output))
        output_two["group_id"] = "CGE-test-2"
        rows = [
            {
                "request_metadata": self.metadata,
                "run_metadata": self._run_metadata(),
                "output": self.output,
            },
            {
                "request_metadata": metadata_two,
                "run_metadata": self._run_metadata(fingerprint="changed-fingerprint"),
                "output": output_two,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            judgments = Path(temporary) / "judgments.jsonl"
            judgments.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            groups = {
                "CGE-test": {"request_metadata": self.metadata},
                "CGE-test-2": {"request_metadata": metadata_two},
            }
            with self.assertRaisesRegex(ValueError, "mixed judge configuration"):
                validate_judgments(
                    judgments,
                    groups,
                    Draft202012Validator(schema),
                    {
                        "prompt": "0" * 64,
                        "output_schema": "1" * 64,
                        "target_profile": "2" * 64,
                    },
                )
            rows[1]["run_metadata"] = self._run_metadata()
            judgments.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            _, identity = validate_judgments(
                judgments,
                groups,
                Draft202012Validator(schema),
                {
                    "prompt": "0" * 64,
                    "output_schema": "1" * 64,
                    "target_profile": "2" * 64,
                },
            )
            self.assertEqual(identity["provider_response_model"], "gpt-oss:120b")
            self.assertEqual(identity["system_fingerprint"], "test-fingerprint")


if __name__ == "__main__":
    unittest.main()
