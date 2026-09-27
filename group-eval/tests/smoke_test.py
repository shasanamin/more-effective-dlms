#!/usr/bin/env python3
"""End-to-end local smoke test for the GroupEval artifact; no API calls."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(*args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [PYTHON, *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def run_expect_failure(*args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [PYTHON, *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if proc.returncode == 0:
        raise RuntimeError(
            f"command unexpectedly succeeded: {' '.join(args)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def synthetic_output(group_record: dict[str, Any]) -> dict[str, Any]:
    meta = group_record["request_metadata"]
    group = group_record["group_input"]
    method = meta["method_code"]
    mode_map: dict[str, str] = {}
    mode_members: dict[str, list[str]] = {}
    samples: list[dict[str, Any]] = []
    for item, mapping in zip(group["samples"], meta["sample_map"], strict=True):
        display = item["sample_id"]
        underlying = "one" if method == "method_b" else f"topic_{mapping['sample_index']}"
        if underlying not in mode_map:
            mode_map[underlying] = f"M{len(mode_map) + 1}"
        mode_id = mode_map[underlying]
        mode_members.setdefault(mode_id, []).append(display)
        score = 1 if method == "method_b" else 2
        samples.append(
            {
                "sample_id": display,
                "valid": True,
                "invalid_reason": None,
                "quality": {
                    "readability": score,
                    "coherence": score,
                    "substance_naturalness": score,
                    "nondegeneration_structure": score,
                },
                "evidence": ["synthetic smoke-test judgment"],
                "mode_id": mode_id,
                "hybrid": False,
                "severe_degeneration": False,
                "prompt_injection_detected": False,
            }
        )
    modes = [
        {
            "mode_id": mode_id,
            "label": f"Synthetic mode {mode_id}",
            "member_sample_ids": members,
            "distinguishing_basis": "deterministic fixture assignment",
        }
        for mode_id, members in mode_members.items()
    ]
    return {
        "protocol_version": "1.0.0",
        "group_id": group["group_id"],
        "samples": samples,
        "modes": modes,
        "group_warnings": [],
        "clustering_confidence": "low" if method == "method_b" else "high",
    }


def main() -> int:
    pilot_bootstrap_replicates = int(
        os.environ.get("GROUPEVAL_SMOKE_BOOTSTRAP_REPLICATES", "100")
    )
    assert pilot_bootstrap_replicates >= 100
    for schema_path in sorted((ROOT / "schemas").glob("*.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)

    didi_help = run("experiment/prepare_didi_pilot.py", "--help").stdout
    for required_option in (
        "--source-root",
        "--checkpoint",
        "--recorded-source-root",
        "--recorded-checkpoint-path",
        "--output-dir",
    ):
        assert required_option in didi_help
    didi_preparer_source = (
        ROOT / "experiment" / "prepare_didi_pilot.py"
    ).read_text(encoding="utf-8")
    hard_coded_private_paths = [
        node.value
        for node in ast.walk(ast.parse(didi_preparer_source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(("/home/", "/scratch/"))
    ]
    assert not hard_coded_private_paths

    runner_spec = importlib.util.spec_from_file_location(
        "groupeval_judge_runner",
        ROOT / "experiment" / "judge_runner_template.py",
    )
    assert runner_spec is not None and runner_spec.loader is not None
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    absolute_schema = json.loads(
        (ROOT / "schemas" / "absolute_output.schema.json").read_text(
            encoding="utf-8"
        )
    )
    inst_body = runner.make_inst_body(
        "gpt-oss:120b", "system", "user", absolute_schema, 6000
    )
    inst_provider_schema = inst_body["response_format"]["json_schema"]["schema"]
    openai_body = runner.make_openai_body(
        "gpt-5.6-luna", "system", "user", absolute_schema, 6000
    )
    openai_default_reasoning_body = runner.make_openai_body(
        "gpt-5.6-terra",
        "system",
        "user",
        absolute_schema,
        6000,
        reasoning_effort=None,
    )
    openai_provider_schema = openai_body["text"]["format"]["schema"]

    def schema_contains_key(value: Any, key: str) -> bool:
        if isinstance(value, dict):
            return key in value or any(
                schema_contains_key(item, key) for item in value.values()
            )
        if isinstance(value, list):
            return any(schema_contains_key(item, key) for item in value)
        return False

    assert schema_contains_key(absolute_schema, "uniqueItems")
    assert not schema_contains_key(inst_provider_schema, "uniqueItems")
    assert schema_contains_key(inst_provider_schema, "minItems")
    assert not schema_contains_key(openai_provider_schema, "uniqueItems")
    assert schema_contains_key(openai_provider_schema, "minItems")
    assert openai_provider_schema["properties"]["protocol_version"]["type"] == "string"
    assert openai_provider_schema["properties"]["clustering_confidence"]["type"] == "string"
    invalid_reason_schema = openai_provider_schema["$defs"]["sample"]["properties"][
        "invalid_reason"
    ]
    assert invalid_reason_schema["anyOf"][1]["type"] == "string"
    assert openai_body["reasoning"] == {"effort": "none"}
    assert "reasoning" not in openai_default_reasoning_body
    assert openai_body["store"] is False
    member_schema = absolute_schema["$defs"]["mode"]["properties"][
        "member_sample_ids"
    ]
    assert list(Draft202012Validator(member_schema).iter_errors(["S1", "S1"]))
    retry_feedback = runner.retry_notice(
        [
            "semantic:valid sample S1: nondegeneration_structure=0 "
            "requires severe_degeneration=true",
            "schema:samples/0/mode_id:None is not valid",
        ]
    )
    assert "Return a complete replacement JSON object" in retry_feedback
    assert "Local validation failures to correct" in retry_feedback
    assert "nondegeneration_structure=0 requires severe_degeneration=true" in retry_feedback
    assert "Emit compact, complete JSON" in retry_feedback
    assert "Every valid sample must have invalid_reason=null" in retry_feedback
    assert "List mode IDs contiguously" in retry_feedback
    bounded_feedback = runner.retry_notice(["x" * 1000] * 20)
    assert "additional validation failure(s) omitted" in bounded_feedback
    assert "x" * runner.MAX_RETRY_FEEDBACK_CHARS not in bounded_feedback

    run(
        "scoring/validate_outputs.py",
        "--input",
        "examples/example_judgments.jsonl",
    )

    with tempfile.TemporaryDirectory(prefix="groupeval-smoke-") as tmp_name:
        tmp = Path(tmp_name)
        relative_recorded_path = run_expect_failure(
            "experiment/prepare_didi_pilot.py",
            "--source-root",
            str(ROOT),
            "--checkpoint",
            str(ROOT / "LICENSE"),
            "--recorded-source-root",
            "relative/frozen-grid",
            "--recorded-checkpoint-path",
            "/recorded/archive/checkpoint.ckpt",
            "--output-dir",
            str(tmp / "invalid-didi-preparation"),
        )
        assert (
            "--recorded-source-root must be the absolute path literal"
            in relative_recorded_path.stderr
        )
        relative_recorded_checkpoint = run_expect_failure(
            "experiment/prepare_didi_pilot.py",
            "--source-root",
            str(ROOT),
            "--checkpoint",
            str(ROOT / "LICENSE"),
            "--recorded-source-root",
            "/recorded/archive/frozen-grid",
            "--recorded-checkpoint-path",
            "relative/checkpoint.ckpt",
            "--output-dir",
            str(tmp / "invalid-didi-checkpoint-preparation"),
        )
        assert (
            "--recorded-checkpoint-path must be the absolute path literal"
            in relative_recorded_checkpoint.stderr
        )

        samples_dir = tmp / "samples"
        samples_dir.mkdir()
        manifest = tmp / "manifest.csv"
        fields = [
            "method_code",
            "sample_index",
            "sample_id",
            "target_profile_id",
            "text_path",
            "generation_seed",
        ]
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method in ("method_a", "method_b"):
                for index in range(16):
                    path = samples_dir / f"{method}_{index:03d}.txt"
                    unicode_separators = (
                        " It retains Unicode line\u2028and paragraph\u2029separators."
                        if index == 0
                        else ""
                    )
                    path.write_text(
                        f"Synthetic web document {index}. It contains several coherent sentences. "
                        "Method-blinded fixture text supports grouping and scoring tests."
                        f"{unicode_separators}",
                        encoding="utf-8",
                    )
                    writer.writerow(
                        {
                            "method_code": method,
                            "sample_index": index,
                            "sample_id": f"{method}_{index:03d}",
                            "target_profile_id": "owt1024-v1",
                            "text_path": path,
                            "generation_seed": 9000 + index,
                        }
                    )

        groups = tmp / "groups.jsonl"
        run(
            "experiment/grouping.py",
            "--manifest",
            str(manifest),
            "--output",
            str(groups),
            "--group-size",
            "8",
            "--partition-seeds",
            "11,22",
            "--optimization-candidates",
            "50",
            "--allow-noncanonical",
        )
        group_rows = read_jsonl(groups)
        assert len(group_rows) == 8
        assert all(len(row["group_input"]["samples"]) == 8 for row in group_rows)

        # A schema-constrained provider can still violate cross-field invariants.
        # Verify that the runner requests a complete new object with bounded,
        # actionable validator feedback rather than mutating the invalid output.
        invalid_provider_output = synthetic_output(group_rows[0])
        invalid_provider_output["samples"][0]["quality"][
            "nondegeneration_structure"
        ] = 0
        invalid_provider_output["samples"][0]["severe_degeneration"] = False
        valid_provider_output = synthetic_output(group_rows[0])
        provider_outputs = [invalid_provider_output, valid_provider_output]
        captured_bodies: list[dict[str, Any]] = []

        class FakeInstResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self, output: dict[str, Any], index: int) -> None:
                self._data = {
                    "id": f"fake-response-{index}",
                    "model": "fake-model",
                    "system_fingerprint": "fake-fingerprint",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                    "choices": [
                        {
                            "message": {"content": json.dumps(output)},
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.text = json.dumps(self._data)

            def json(self) -> dict[str, Any]:
                return self._data

        import requests

        original_post = requests.post

        def fake_post(
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: float,
        ) -> FakeInstResponse:
            del url, headers, timeout
            captured_bodies.append(json)
            return FakeInstResponse(
                provider_outputs[len(captured_bodies) - 1],
                len(captured_bodies),
            )

        requests.post = fake_post
        try:
            replacement_output, replacement_metadata = runner.run_one_inst(
                "https://example.invalid/chat/completions",
                "test-key-not-a-secret",
                runner.make_inst_body(
                    "fake-model",
                    "system",
                    "user",
                    absolute_schema,
                    6000,
                ),
                group_rows[0]["request_metadata"],
                Draft202012Validator(absolute_schema),
                max_transport_retries=0,
                max_content_retries=1,
                timeout=1.0,
                rate_limiter=runner.SerialRateLimiter(0.0),
            )
        finally:
            requests.post = original_post
        assert replacement_output == valid_provider_output
        assert replacement_metadata["content_retries"] == 1
        assert len(replacement_metadata["attempt_body_sha256s"]) == 2
        assert len(captured_bodies) == 2
        assert captured_bodies[0]["messages"][-1]["content"] == "user"
        replacement_prompt = captured_bodies[1]["messages"][-1]["content"]
        assert "Return a complete replacement JSON object" in replacement_prompt
        assert (
            "nondegeneration_structure=0 requires severe_degeneration=true"
            in replacement_prompt
        )

        # GenAI Studio sometimes reports its gateway rate limit as HTTP 400
        # instead of 429.  That exact provider message must be retried, while
        # arbitrary client-side 400 errors remain non-retriable.
        class FakeRateLimitResponse:
            status_code = 400
            headers: dict[str, str] = {}
            text = '{"detail":"Rate limit exceeded. Please try again later."}'

        rate_limit_calls = 0

        def fake_rate_limited_post(
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: float,
        ) -> FakeRateLimitResponse | FakeInstResponse:
            nonlocal rate_limit_calls
            del url, headers, json, timeout
            rate_limit_calls += 1
            if rate_limit_calls == 1:
                return FakeRateLimitResponse()
            return FakeInstResponse(valid_provider_output, rate_limit_calls)

        original_sleep = runner.time.sleep
        requests.post = fake_rate_limited_post
        runner.time.sleep = lambda _: None
        try:
            rate_limit_output, rate_limit_metadata = runner.run_one_inst(
                "https://example.invalid/chat/completions",
                "test-key-not-a-secret",
                runner.make_inst_body(
                    "fake-model",
                    "system",
                    "user",
                    absolute_schema,
                    6000,
                ),
                group_rows[0]["request_metadata"],
                Draft202012Validator(absolute_schema),
                max_transport_retries=1,
                max_content_retries=0,
                timeout=1.0,
                rate_limiter=runner.SerialRateLimiter(0.0),
            )
        finally:
            requests.post = original_post
            runner.time.sleep = original_sleep
        assert rate_limit_output == valid_provider_output
        assert rate_limit_metadata["transport_retries"] == 1
        assert rate_limit_calls == 2

        preview = tmp / "preview.jsonl"
        dry = tmp / "dry.jsonl"
        run(
            "experiment/judge_runner_template.py",
            "--groups",
            str(groups),
            "--output",
            str(dry),
            "--dry-run",
            "--limit",
            "1",
            "--request-preview",
            str(preview),
        )
        assert read_jsonl(dry)[0]["dry_run"] is True
        rendered_input = read_jsonl(preview)[0]["input"]
        assert "### Candidate S1" in rendered_input
        assert "Output `sample_id` (copy exactly): `S1`" in rendered_input
        assert "_S1_BEGIN>>>" not in rendered_input

        judgments = tmp / "judgments.jsonl"
        with judgments.open("w", encoding="utf-8") as handle:
            for row in group_rows:
                wrapper = {
                    "request_metadata": row["request_metadata"],
                    "run_metadata": {
                        "judge_provider": "synthetic",
                        "judge_model_snapshot": "smoke-test",
                        "system_fingerprint": "synthetic-fingerprint",
                        "prompt_sha256": "a" * 64,
                        "schema_sha256": "b" * 64,
                        "target_profile_sha256": "c" * 64,
                        "strict_json_schema": True,
                        "local_validation_uses_unmodified_schema": True,
                        "tools_enabled": False,
                        "provider_schema_omitted_keywords": ["uniqueItems"],
                        "response_id": f"response-{row['request_metadata']['group_id']}",
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        },
                        "transport_retries": 0,
                        "content_retries": 0,
                        "finish_reason": "stop",
                    },
                    "output": synthetic_output(row),
                }
                handle.write(json.dumps(wrapper) + "\n")

        # Production shards are round-robin and therefore cannot be concatenated
        # directly when the canonical groups order matters.  Exercise the strict
        # merger with deliberately reversed shard arguments.
        judgment_rows_for_merge = read_jsonl(judgments)
        merge_shards: list[Path] = []
        for shard_index in range(3):
            shard = tmp / f"judgments_shard_{shard_index}.jsonl"
            shard.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in judgment_rows_for_merge[shard_index::3]
                )
                + "\n",
                encoding="utf-8",
            )
            merge_shards.append(shard)
        merged = tmp / "judgments_merged.jsonl"
        run(
            "experiment/merge_judgment_shards.py",
            "--groups",
            str(groups),
            "--shards",
            *(str(path) for path in reversed(merge_shards)),
            "--output",
            str(merged),
            "--expected-model",
            "smoke-test",
            "--expected-provider",
            "synthetic",
        )
        assert [
            row["request_metadata"]["group_id"] for row in read_jsonl(merged)
        ] == [row["request_metadata"]["group_id"] for row in group_rows]

        merge_existing_proc = run_expect_failure(
            "experiment/merge_judgment_shards.py",
            "--groups",
            str(groups),
            "--shards",
            *(str(path) for path in merge_shards),
            "--output",
            str(merged),
        )
        assert "refusing to overwrite" in merge_existing_proc.stderr

        duplicate_shard = tmp / "judgments_duplicate_shard.jsonl"
        duplicate_shard.write_text(
            json.dumps(judgment_rows_for_merge[0]) + "\n",
            encoding="utf-8",
        )
        merge_duplicate_proc = run_expect_failure(
            "experiment/merge_judgment_shards.py",
            "--groups",
            str(groups),
            "--shards",
            *(str(path) for path in merge_shards),
            str(duplicate_shard),
            "--output",
            str(tmp / "merge_duplicate.jsonl"),
        )
        assert "duplicate judgment group_id" in merge_duplicate_proc.stderr

        merge_missing_shard = tmp / "judgments_merge_missing.jsonl"
        merge_missing_shard.write_text(
            "\n".join(
                json.dumps(row) for row in judgment_rows_for_merge[:-1]
            )
            + "\n",
            encoding="utf-8",
        )
        merge_missing_proc = run_expect_failure(
            "experiment/merge_judgment_shards.py",
            "--groups",
            str(groups),
            "--shards",
            str(merge_missing_shard),
            "--output",
            str(tmp / "merge_missing.jsonl"),
        )
        assert "missing expected judgment group IDs" in merge_missing_proc.stderr

        merge_wrong_model_proc = run_expect_failure(
            "experiment/merge_judgment_shards.py",
            "--groups",
            str(groups),
            "--shards",
            *(str(path) for path in merge_shards),
            "--output",
            str(tmp / "merge_wrong_model.jsonl"),
            "--expected-model",
            "not-the-smoke-model",
        )
        assert "judge_model_snapshot must be" in merge_wrong_model_proc.stderr

        report = tmp / "validation.json"
        run(
            "scoring/validate_outputs.py",
            "--input",
            str(judgments),
            "--groups",
            str(groups),
            "--report",
            str(report),
        )
        missing = tmp / "judgments_missing.jsonl"
        missing.write_text(
            "\n".join(json.dumps(row) for row in read_jsonl(judgments)[:-1]) + "\n",
            encoding="utf-8",
        )
        missing_proc = run_expect_failure(
            "scoring/validate_outputs.py",
            "--input",
            str(missing),
            "--groups",
            str(groups),
        )
        assert "missing expected judgment group_id" in missing_proc.stderr

        duplicate = tmp / "judgments_duplicate.jsonl"
        judgment_rows = read_jsonl(judgments)
        duplicate.write_text(
            "\n".join(json.dumps(row) for row in [*judgment_rows, judgment_rows[0]]) + "\n",
            encoding="utf-8",
        )
        duplicate_proc = run_expect_failure(
            "scoring/validate_outputs.py",
            "--input",
            str(duplicate),
            "--groups",
            str(groups),
        )
        assert "duplicate judgment group_id" in duplicate_proc.stderr

        drift = tmp / "judgments_metadata_drift.jsonl"
        drift_rows = read_jsonl(judgments)
        drift_rows[0]["run_metadata"]["judge_model_snapshot"] = "different-snapshot"
        drift.write_text(
            "\n".join(json.dumps(row) for row in drift_rows) + "\n",
            encoding="utf-8",
        )
        drift_proc = run_expect_failure(
            "scoring/validate_outputs.py",
            "--input",
            str(drift),
            "--groups",
            str(groups),
        )
        assert "judge_model_snapshot is inconsistent" in drift_proc.stderr

        private_map = tmp / "private_label_map.json"
        private_map.write_text(
            json.dumps(
                {
                    "artifact": "GroupEval MDLM-Small pilot private method map",
                    "preparation_version": "test",
                    "target_profile_id": "owt1024-v1",
                    "keep_private_until_judging_complete": True,
                    "methods": {
                        "method_a": {
                            "model": "public MDLM-Small",
                            "configured_updates_M": 16,
                            "sampler": "ancestral_cache",
                            "temperature": 0.9,
                            "categorical_precision": "fp32",
                            "condition": "selected",
                            "source_result": "/scratch/private/method_a.json",
                            "source_config": "/scratch/private/method_a.yaml",
                        },
                        "method_b": {
                            "model": "public MDLM-Small",
                            "configured_updates_M": 16,
                            "sampler": "ancestral_cache",
                            "temperature": 1.0,
                            "categorical_precision": "fp64",
                            "condition": "unscaled-fp64-reference",
                            "source_result": "/scratch/private/method_b.json",
                            "source_config": "/scratch/private/method_b.yaml",
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = tmp / "generation_provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "artifact": "GroupEval MDLM-Small two-operating-point pilot",
                    "preparation_version": "test",
                    "source_root": "/scratch/private/source",
                    "target_profile": {
                        "id": "owt1024-v1",
                        "path": "/home/private/target_profile.md",
                        "sha256": "d" * 64,
                    },
                    "checkpoint": {
                        "configured_path": "/scratch/private/model.ckpt",
                        "verified_path": "/scratch/private/model.ckpt",
                        "sha256": "e" * 64,
                        "bytes": 123,
                    },
                    "normalization": {"truncation": False},
                    "seed_and_pairing": {
                        "global_lightning_seed": 1,
                        "per_sample_seeds_available": False,
                        "sample_indices": "decoded-list positions",
                        "sample_level_common_random_numbers_declared": False,
                        "allowed_bootstrap_pairing_mode": "partitions",
                    },
                    "archive_limitations": {"raw_text_redistributed": False},
                    "design": {
                        "methods": 2,
                        "samples_per_method": 16,
                        "total_samples": 32,
                        "group_size": 8,
                        "partition_seeds": [11, 22],
                    },
                    "cells": [
                        {
                            "method_code": "method_a",
                            "configured_updates_M": 16,
                            "temperature": 0.9,
                            "categorical_precision": "fp32",
                            "global_lightning_seed": 1,
                            "sample_count": 16,
                            "source_timestamp": "2026-01-01T00:00:00",
                            "result_file": {"path": "/scratch/a.json", "sha256": "f" * 64},
                            "hydra_config": {"path": "/scratch/a.yaml", "sha256": "1" * 64},
                            "normalized_sample_hash_list_sha256": "2" * 64,
                            "performance_for_cross_check_only": {
                                "generative_perplexity": 40.0,
                                "avg_entropy": 5.2,
                                "avg_entropy_per_sequence": 5.2,
                            },
                        },
                        {
                            "method_code": "method_b",
                            "configured_updates_M": 16,
                            "temperature": 1.0,
                            "categorical_precision": "fp64",
                            "global_lightning_seed": 1,
                            "sample_count": 16,
                            "source_timestamp": "2026-01-01T00:00:00",
                            "result_file": {"path": "/scratch/b.json", "sha256": "3" * 64},
                            "hydra_config": {"path": "/scratch/b.yaml", "sha256": "4" * 64},
                            "normalized_sample_hash_list_sha256": "5" * 64,
                            "performance_for_cross_check_only": {
                                "generative_perplexity": 100.0,
                                "avg_entropy": 5.4,
                                "avg_entropy_per_sequence": 5.4,
                            },
                        },
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        pilot_analysis = tmp / "pilot_analysis"
        run(
            "experiment/analyze_mdlm_pilot.py",
            "--judgments",
            str(judgments),
            "--groups",
            str(groups),
            "--generation-provenance",
            str(provenance),
            "--private-label-map",
            str(private_map),
            "--output-dir",
            str(pilot_analysis),
            "--bootstrap-replicates",
            str(pilot_bootstrap_replicates),
        )
        pilot_summary = json.loads(
            (pilot_analysis / "pilot_summary.json").read_text(encoding="utf-8")
        )
        assert pilot_summary["analysis"]["judgment_completeness"]["complete"] is True
        assert pilot_summary["analysis"]["bootstrap_pairing_mode"] == "partitions"
        assert (
            pilot_summary["analysis"]["bootstrap_replicates"]
            == pilot_bootstrap_replicates
        )
        assert (
            pilot_summary["judge_diagnostics"]["token_usage_scope"]
            == "accepted_response_attempts_only"
        )
        assert (
            pilot_summary["judge_diagnostics"]["retry_attempt_token_usage_included"]
            is False
        )
        protocol_identity = pilot_summary["judge_diagnostics"]["protocol_identity"]
        assert protocol_identity["prompt_sha256"] == "a" * 64
        assert protocol_identity["schema_sha256"] == "b" * 64
        assert protocol_identity["target_profile_sha256"] == "c" * 64
        assert protocol_identity["strict_json_schema"] is True
        assert protocol_identity["local_validation_uses_unmodified_schema"] is True
        assert protocol_identity["tools_enabled"] is False
        assert protocol_identity["provider_schema_omitted_keywords"] == [
            "uniqueItems"
        ]
        assert len(pilot_summary["within_budget_comparisons"]) == 1
        assert (pilot_analysis / "pilot_results.csv").is_file()
        assert (pilot_analysis / "pilot_deltas.csv").is_file()
        table_text = (pilot_analysis / "pilot_table.tex").read_text(encoding="utf-8")
        assert "GED$\\uparrow$ (ref./sel.)" in table_text
        assert f"{pilot_bootstrap_replicates:,}-replicate" in table_text
        for safe_path in (
            pilot_analysis / "pilot_summary.json",
            pilot_analysis / "pilot_results.csv",
            pilot_analysis / "pilot_deltas.csv",
            pilot_analysis / "pilot_table.tex",
        ):
            safe_text = safe_path.read_text(encoding="utf-8")
            assert "/scratch/" not in safe_text
            assert "/home/" not in safe_text

        # The archived DiDi provenance uses `entropy`, whereas the MDLM
        # provenance uses `avg_entropy` plus a cross-implementation check.
        # Exercise that study-specific input shape rather than only changing
        # the presentation profile on the MDLM-shaped fixture.
        didi_provenance = tmp / "didi_generation_provenance.json"
        didi_provenance_payload = json.loads(
            provenance.read_text(encoding="utf-8")
        )
        didi_provenance_payload["artifact"] = (
            "GroupEval DiDi-Instruct reward-guided two-operating-point pilot"
        )
        didi_provenance_payload["design"]["noncanonical"] = True
        for cell in didi_provenance_payload["cells"]:
            performance = cell["performance_for_cross_check_only"]
            performance["entropy"] = performance.pop("avg_entropy")
            performance.pop("avg_entropy_per_sequence")
            performance["evaluator"] = "gpt2-large"
        didi_provenance.write_text(
            json.dumps(didi_provenance_payload) + "\n",
            encoding="utf-8",
        )
        didi_private_map = tmp / "didi_private_label_map.json"
        didi_private_map_payload = json.loads(
            private_map.read_text(encoding="utf-8")
        )
        didi_private_map_payload["artifact"] = (
            "GroupEval DiDi-Instruct pilot private method map"
        )
        for label in didi_private_map_payload["methods"].values():
            label["model"] = "DiDi-Instruct EMA student"
            label["sampler"] = "reward-guided ancestral"
        didi_private_map.write_text(
            json.dumps(didi_private_map_payload) + "\n",
            encoding="utf-8",
        )

        didi_analysis = tmp / "didi_analysis"
        run(
            "experiment/analyze_mdlm_pilot.py",
            "--judgments",
            str(judgments),
            "--groups",
            str(groups),
            "--generation-provenance",
            str(didi_provenance),
            "--private-label-map",
            str(didi_private_map),
            "--output-dir",
            str(didi_analysis),
            "--bootstrap-replicates",
            str(pilot_bootstrap_replicates),
            "--study",
            "didi-instruct",
        )
        didi_summary = json.loads(
            (didi_analysis / "pilot_summary.json").read_text(encoding="utf-8")
        )
        assert didi_summary["analysis"]["study"] == "didi-instruct"
        assert didi_summary["analysis"]["gppl_evaluator"] == "GPT-2-Large"
        if didi_summary["analysis"]["figures_generated"]:
            assert (didi_analysis / "didi_pilot_groupeval.pdf").is_file()
        didi_table = (didi_analysis / "pilot_table.tex").read_text(
            encoding="utf-8"
        )
        assert "reward-guided DiDi-Instruct EMA student" in didi_table
        assert "16-sample-per-operating-point" in didi_table
        assert r"\label{tab:groupeval-didi-pilot}" in didi_table

        mislabeled_analysis = run_expect_failure(
            "experiment/analyze_mdlm_pilot.py",
            "--judgments",
            str(judgments),
            "--groups",
            str(groups),
            "--generation-provenance",
            str(provenance),
            "--private-label-map",
            str(private_map),
            "--output-dir",
            str(tmp / "mislabeled_analysis"),
            "--bootstrap-replicates",
            str(pilot_bootstrap_replicates),
            "--study",
            "didi-instruct",
        )
        assert (
            "generation_provenance.artifact does not match the selected study"
            in mislabeled_analysis.stderr
        )

        incomplete_analysis = run_expect_failure(
            "experiment/analyze_mdlm_pilot.py",
            "--judgments",
            str(missing),
            "--groups",
            str(groups),
            "--generation-provenance",
            str(provenance),
            "--private-label-map",
            str(private_map),
            "--output-dir",
            str(tmp / "incomplete_analysis"),
            "--bootstrap-replicates",
            str(pilot_bootstrap_replicates),
        )
        assert "incomplete judgments" in incomplete_analysis.stderr

        score_dir = tmp / "scores"
        run(
            "scoring/score_group_eval.py",
            "--judgments",
            str(judgments),
            "--out-dir",
            str(score_dir),
        )
        with (score_dir / "system_scores.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            system_scores = {
                row["method_code"]: row for row in csv.DictReader(handle)
            }
        assert float(system_scores["method_a"]["all_singleton_group_rate"]) == 1.0
        assert float(system_scores["method_a"]["ged_ceiling_rate"]) == 1.0
        assert float(system_scores["method_a"]["low_clustering_confidence_rate"]) == 0.0
        assert float(system_scores["method_b"]["all_singleton_group_rate"]) == 0.0
        assert float(system_scores["method_b"]["ged_ceiling_rate"]) == 0.0
        assert float(system_scores["method_b"]["low_clustering_confidence_rate"]) == 1.0

        duplicate_score_proc = run_expect_failure(
            "scoring/score_group_eval.py",
            "--judgments",
            str(duplicate),
            "--out-dir",
            str(tmp / "duplicate_scores"),
        )
        assert "duplicate judgment group_id" in duplicate_score_proc.stderr

        bootstrap_dir = tmp / "bootstrap"
        run(
            "scoring/bootstrap_group_eval.py",
            "--judgments",
            str(judgments),
            "--out-dir",
            str(bootstrap_dir),
            "--replicates",
            "100",
            "--pairing-mode",
            "partitions",
            "--baseline",
            "method_b",
            "--ged-ni-margin",
            "0.5",
        )
        with (bootstrap_dir / "bootstrap_comparisons.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            bootstrap_comparisons = list(csv.DictReader(handle))
        assert bootstrap_comparisons
        assert "point_pareto_dominates" not in bootstrap_comparisons[0]
        assert "statistically_supported_pareto" not in bootstrap_comparisons[0]
        table = tmp / "results.tex"
        run(
            "experiment/make_paper_tables.py",
            "--scores",
            str(bootstrap_dir / "bootstrap_scores.csv"),
            "--comparisons",
            str(bootstrap_dir / "bootstrap_comparisons.csv"),
            "--output",
            str(table),
            "--baseline",
            "method_b",
        )
        assert "Operating point" in table.read_text(encoding="utf-8")

    print("GroupEval smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
