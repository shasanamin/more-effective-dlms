#!/usr/bin/env python3
"""Validate and analyze the frozen conditional large-DLM GEQ study.

The judge sees only a public task condition and two blinded responses.  This
script joins accepted judgments to a private deterministic label map, computes
task-conditional GEQ, and evaluates association with exact-match/functional
success.  It never reads or emits reference answers, HumanEval tests, or
canonical solutions, and it deliberately does not define GED for this design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from conditional_large_dlm_common import (
    EVALUATION_PROTOCOL,
    PROTOCOL_VERSION,
    QUALITY_DIMENSIONS,
    TARGET_PROFILE_ID,
    conditional_geq,
    conditional_semantic_errors,
)
from prepare_large_dlm_conditional import EXPECTED_PASS_COUNTS, MODELS
from prepare_large_dlm_conditional import (  # noqa: E402
    EXPECTED_SOURCE_SHA256,
    MODEL_CHECKPOINTS,
    MODEL_REVISIONS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_SCHEMA = ROOT / "schemas/conditional_group_input.schema.json"
DEFAULT_OUTPUT_SCHEMA = ROOT / "schemas/conditional_geq_output.schema.json"
DEFAULT_PROMPT = ROOT / "prompts/group_eval_conditional_task.md"
DEFAULT_PROFILE = ROOT / "protocol/target_profile_large_dlm_tasks.md"
TASKS = ("gsm8k", "humaneval")
EXPECTED_CONDITIONS = {"gsm8k": 256, "humaneval": 164}
MODEL_NAMES = {
    "llada": "LLaDA-8B-Instruct",
    "dream": "Dream-v0-Base-7B",
}
MODEL_INDEX = {model: index for index, model in enumerate(MODELS)}
EXPECTED_JUDGE_PROVIDER = "institutional_genai"
EXPECTED_JUDGE_MODEL = "gpt-oss:120b"
EXPECTED_BUNDLE_FILES = {
    "conditional_groups.jsonl",
    "private_label_map.json",
    "generation_provenance.json",
    "preparation_summary.json",
}
EXPECTED_BUNDLE_SHA256 = {
    "conditional_groups.jsonl":
        "7f48595fec5b182f489ba6d37f47edf2a0d7bdbd353ac6bc2082c3317dfca3e9",
    "private_label_map.json":
        "140fdf57a5f8ba85a307c95599a4948a7f26ff4d3f2aa40ce53dd251bcce973a",
    "generation_provenance.json":
        "e3c8d1cfbf7f7c7992cd5c4c887acd3b6512b302ff85a3c8c8c096715774e711",
    "preparation_summary.json":
        "624b9c01a5a52804843c1ba41b5d212c9d764798774dc3e5117a7ae9786e58ca",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def load_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("  ", 1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number}: malformed SHA256SUMS row")
            digest, name = parts
            if (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or Path(name).name != name
            ):
                raise ValueError(f"{path}:{line_number}: invalid digest or file name")
            if name in entries:
                raise ValueError(f"{path}:{line_number}: duplicate entry {name!r}")
            entries[name] = digest
    if set(entries) != EXPECTED_BUNDLE_FILES:
        raise ValueError(
            f"{path}: expected exact frozen bundle entries "
            f"{sorted(EXPECTED_BUNDLE_FILES)}, found {sorted(entries)}"
        )
    return entries


def verify_prepared_bundle(
    groups_path: Path,
    labels_path: Path,
    provenance_path: Path,
    summary_path: Path,
    sums_path: Path,
    *,
    expected_hashes: Mapping[str, str] = EXPECTED_BUNDLE_SHA256,
) -> None:
    supplied = {
        "conditional_groups.jsonl": groups_path,
        "private_label_map.json": labels_path,
        "generation_provenance.json": provenance_path,
        "preparation_summary.json": summary_path,
    }
    entries = load_sha256_manifest(sums_path)
    if entries != dict(expected_hashes):
        raise ValueError(
            f"{sums_path}: manifest does not match the pinned frozen bundle digests"
        )
    bundle_root = sums_path.resolve().parent
    for name, path in supplied.items():
        if path.resolve() != bundle_root / name:
            raise ValueError(
                f"{name}: supplied path is outside or renamed from the frozen bundle"
            )
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen bundle artifact: {path}")
        observed = sha256_path(path)
        if observed != entries[name]:
            raise ValueError(
                f"{path}: SHA-256 mismatch; expected {entries[name]}, observed {observed}"
            )


def validate_frozen_provenance(
    path: Path,
    prompt_path: Path,
    input_schema_path: Path,
    output_schema_path: Path,
    profile_path: Path,
) -> dict[str, str]:
    provenance = require_mapping(load_json(path), "generation provenance")
    if provenance.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("generation provenance protocol_version mismatch")
    if provenance.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ValueError("generation provenance evaluation_protocol mismatch")
    if provenance.get("model_checkpoints") != MODEL_CHECKPOINTS:
        raise ValueError("generation provenance model checkpoints mismatch")
    if provenance.get("cached_model_revisions") != MODEL_REVISIONS:
        raise ValueError("generation provenance model revisions mismatch")
    if provenance.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("generation provenance source hashes mismatch")
    if provenance.get("generation_operating_point") != {
        "temperature": 0.0,
        "generation_seed": 0,
        "maximum_new_tokens": 256,
        "baseline_diffusion_steps": 256,
        "block_length": 32,
    }:
        raise ValueError("generation provenance operating point mismatch")
    privacy = require_mapping(provenance.get("privacy"), "generation provenance.privacy")
    expected_privacy = {
        "judge_inputs_include_reference_answers": False,
        "judge_inputs_include_humaneval_tests": False,
        "judge_inputs_include_canonical_solutions": False,
        "binary_labels_stored_only_in_private_label_map": True,
    }
    if dict(privacy) != expected_privacy:
        raise ValueError("generation provenance privacy declaration mismatch")

    resource_paths = {
        "prompt": prompt_path,
        "input_schema": input_schema_path,
        "output_schema": output_schema_path,
        "target_profile": profile_path,
    }
    declared = require_mapping(
        provenance.get("resource_sha256"), "generation provenance.resource_sha256"
    )
    if set(declared) != set(resource_paths):
        raise ValueError("generation provenance resource hash keys mismatch")
    observed: dict[str, str] = {}
    for name, resource_path in resource_paths.items():
        if not resource_path.is_file():
            raise FileNotFoundError(f"missing frozen resource: {resource_path}")
        observed[name] = sha256_path(resource_path)
        if declared.get(name) != observed[name]:
            raise ValueError(
                f"frozen {name} hash mismatch: declared {declared.get(name)}, "
                f"observed {observed[name]}"
            )
    return observed


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def require_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return value


def schema_errors(
    validator: Draft202012Validator, value: Any, context: str
) -> list[str]:
    return [
        f"{context}:{'/'.join(map(str, error.absolute_path)) or '<root>'}:"
        f"{error.message}"
        for error in validator.iter_errors(value)
    ]


def condition_number(condition_id: str) -> int:
    try:
        return int(condition_id.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid condition_id: {condition_id}") from exc


def validate_groups(
    path: Path, validator: Draft202012Validator
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_jsonl(path)
    if len(rows) != sum(EXPECTED_CONDITIONS.values()):
        raise ValueError(f"expected 420 groups, found {len(rows)}")

    by_id: dict[str, dict[str, Any]] = {}
    task_ids: dict[str, set[int]] = {task: set() for task in TASKS}
    for row_number, row in enumerate(rows, 1):
        metadata = require_mapping(row.get("request_metadata"), f"group {row_number}.metadata")
        group = require_mapping(row.get("group_input"), f"group {row_number}.group_input")
        errors = schema_errors(validator, group, f"group {row_number}")
        if errors:
            raise ValueError(" | ".join(errors[:8]))

        group_id = str(group["group_id"])
        task = str(group["task_name"])
        condition = require_mapping(group["condition"], f"{group_id}.condition")
        condition_id = str(condition["condition_id"])
        if group_id in by_id:
            raise ValueError(f"duplicate group_id: {group_id}")
        if task not in TASKS or not condition_id.startswith(f"{task}:"):
            raise ValueError(f"{group_id}: task/condition mismatch")
        numeric_id = condition_number(condition_id)
        if numeric_id in task_ids[task]:
            raise ValueError(f"duplicate condition_id: {condition_id}")
        task_ids[task].add(numeric_id)

        required_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "group_id": group_id,
            "task_name": task,
            "condition_id": condition_id,
            "target_profile_id": TARGET_PROFILE_ID,
        }
        for field, expected in required_metadata.items():
            if metadata.get(field) != expected:
                raise ValueError(
                    f"{group_id}: metadata.{field} expected {expected!r}, "
                    f"found {metadata.get(field)!r}"
                )
        condition_sha = hashlib.sha256(str(condition["text"]).encode("utf-8")).hexdigest()
        if metadata.get("condition_sha256") != condition_sha:
            raise ValueError(f"{group_id}: condition SHA-256 mismatch")

        samples = require_sequence(group["samples"], f"{group_id}.samples")
        sample_map = require_sequence(metadata.get("sample_map"), f"{group_id}.sample_map")
        samples_by_id = {
            str(require_mapping(item, f"{group_id}.sample")["sample_id"]): item
            for item in samples
        }
        maps_by_id = {
            str(require_mapping(item, f"{group_id}.sample_map item")["display_id"]): item
            for item in sample_map
        }
        if set(samples_by_id) != {"S1", "S2"} or set(maps_by_id) != {"S1", "S2"}:
            raise ValueError(f"{group_id}: expected exactly S1 and S2")
        if {str(item.get("model_code")) for item in maps_by_id.values()} != set(MODELS):
            raise ValueError(f"{group_id}: sample map must contain both models")
        for display_id, item in maps_by_id.items():
            if item.get("condition_id") != condition_id:
                raise ValueError(f"{group_id}/{display_id}: condition mismatch")
            response = str(samples_by_id[display_id]["response"])
            response_sha = hashlib.sha256(response.encode("utf-8")).hexdigest()
            if item.get("response_sha256") != response_sha:
                raise ValueError(f"{group_id}/{display_id}: response SHA-256 mismatch")
        by_id[group_id] = row

    for task, expected_n in EXPECTED_CONDITIONS.items():
        if task_ids[task] != set(range(expected_n)):
            raise ValueError(f"{task}: conditions are not the exact 0..{expected_n - 1} set")
    rows.sort(
        key=lambda row: (
            TASKS.index(str(row["request_metadata"]["task_name"])),
            condition_number(str(row["request_metadata"]["condition_id"])),
        )
    )
    return rows, by_id


def validate_labels(
    path: Path, groups_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    document = require_mapping(load_json(path), "private label map")
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("private label map protocol_version mismatch")
    if document.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        raise ValueError("private label map evaluation_protocol mismatch")
    records = require_sequence(document.get("records"), "private label map.records")
    if len(records) != len(groups_by_id):
        raise ValueError("private label map must contain exactly one record per group")

    by_id: dict[str, dict[str, Any]] = {}
    pass_counts = {task: {model: 0 for model in MODELS} for task in TASKS}
    totals = {task: {model: 0 for model in MODELS} for task in TASKS}
    for record_index, raw_record in enumerate(records, 1):
        record = dict(require_mapping(raw_record, f"label record {record_index}"))
        group_id = str(record.get("group_id", ""))
        if group_id in by_id:
            raise ValueError(f"duplicate private label group_id: {group_id}")
        if group_id not in groups_by_id:
            raise ValueError(f"private label has unexpected group_id: {group_id}")
        metadata = groups_by_id[group_id]["request_metadata"]
        if record.get("task_name") != metadata["task_name"]:
            raise ValueError(f"{group_id}: private task mismatch")
        if record.get("condition_id") != metadata["condition_id"]:
            raise ValueError(f"{group_id}: private condition mismatch")
        labels = require_sequence(record.get("labels"), f"{group_id}.labels")
        if len(labels) != 2:
            raise ValueError(f"{group_id}: expected two private labels")
        label_by_display: dict[str, dict[str, Any]] = {}
        for raw_label in labels:
            label = dict(require_mapping(raw_label, f"{group_id}.label"))
            display = str(label.get("display_id", ""))
            if display in label_by_display:
                raise ValueError(f"{group_id}: duplicate label display ID")
            if type(label.get("deterministic_pass")) is not bool:
                raise ValueError(f"{group_id}/{display}: deterministic_pass must be boolean")
            label_by_display[display] = label
        sample_map = {item["display_id"]: item for item in metadata["sample_map"]}
        if set(label_by_display) != set(sample_map):
            raise ValueError(f"{group_id}: label/sample display IDs differ")
        if {label["model_code"] for label in label_by_display.values()} != set(MODELS):
            raise ValueError(f"{group_id}: labels must contain both models")
        for display, label in label_by_display.items():
            if label.get("model_code") != sample_map[display]["model_code"]:
                raise ValueError(f"{group_id}/{display}: private model map mismatch")
            task = str(record["task_name"])
            model = str(label["model_code"])
            totals[task][model] += 1
            pass_counts[task][model] += int(label["deterministic_pass"])
        record["labels_by_display"] = label_by_display
        by_id[group_id] = record
    if set(by_id) != set(groups_by_id):
        raise ValueError("private label group IDs do not match prepared groups")
    for task in TASKS:
        for model in MODELS:
            expected_passed, expected_n = EXPECTED_PASS_COUNTS[task][model]
            if (pass_counts[task][model], totals[task][model]) != (expected_passed, expected_n):
                raise ValueError(
                    f"{task}/{model}: expected deterministic labels "
                    f"{expected_passed}/{expected_n}, found "
                    f"{pass_counts[task][model]}/{totals[task][model]}"
                )
    return by_id


def validate_judgments(
    path: Path,
    groups_by_id: Mapping[str, Mapping[str, Any]],
    validator: Draft202012Validator,
    frozen_resource_hashes: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(path)
    if len(rows) != len(groups_by_id):
        raise ValueError(
            f"judgments must contain exactly {len(groups_by_id)} rows; found {len(rows)}"
        )
    by_id: dict[str, dict[str, Any]] = {}
    judge_identity: dict[str, Any] | None = None
    identity_fields = (
        "judge_provider",
        "judge_model_snapshot",
        "judge_endpoint",
        "provider_response_model",
        "system_fingerprint",
        "model_identifier_is_provider_alias",
        "reasoning_effort",
        "tools_enabled",
        "strict_json_schema",
        "provider_schema_omitted_keywords",
        "provider_schema_normalizations",
        "local_validation_uses_unmodified_schema",
        "prompt_sha256",
        "schema_sha256",
        "target_profile_sha256",
    )
    for row_number, row in enumerate(rows, 1):
        metadata = require_mapping(row.get("request_metadata"), f"judgment {row_number}.metadata")
        group_id = str(metadata.get("group_id", ""))
        if group_id in by_id:
            raise ValueError(f"duplicate judgment group_id: {group_id}")
        if group_id not in groups_by_id:
            raise ValueError(f"unexpected judgment group_id: {group_id}")
        canonical_metadata = groups_by_id[group_id]["request_metadata"]
        if canonical_json(metadata) != canonical_json(canonical_metadata):
            raise ValueError(f"{group_id}: judgment request_metadata is not canonical")
        output = require_mapping(row.get("output"), f"{group_id}.output")
        errors = schema_errors(validator, output, f"{group_id}.output")
        errors.extend(conditional_semantic_errors({"request_metadata": metadata, "output": output}))
        if errors:
            raise ValueError(" | ".join(errors[:8]))

        run = require_mapping(row.get("run_metadata"), f"{group_id}.run_metadata")
        missing_identity_fields = [field for field in identity_fields if field not in run]
        if missing_identity_fields:
            raise ValueError(
                f"{group_id}: missing run metadata fields {missing_identity_fields}"
            )
        identity = {field: run.get(field) for field in identity_fields}
        if identity["judge_provider"] != EXPECTED_JUDGE_PROVIDER:
            raise ValueError(f"{group_id}: this study requires GenAI-Inst")
        if identity["judge_model_snapshot"] != EXPECTED_JUDGE_MODEL:
            raise ValueError(f"{group_id}: this study requires gpt-oss:120b")
        if not str(identity["judge_endpoint"]).startswith("https://"):
            raise ValueError(f"{group_id}: judge endpoint must use HTTPS")
        if identity["model_identifier_is_provider_alias"] is not True:
            raise ValueError(f"{group_id}: mutable provider alias must be declared")
        if identity["reasoning_effort"] != "provider_default":
            raise ValueError(f"{group_id}: unexpected GenAI-Inst reasoning metadata")
        provider_response_model = identity["provider_response_model"]
        if provider_response_model is not None and not str(provider_response_model).strip():
            raise ValueError(f"{group_id}: empty provider_response_model")
        system_fingerprint = identity["system_fingerprint"]
        if system_fingerprint is not None and not str(system_fingerprint).strip():
            raise ValueError(f"{group_id}: empty system_fingerprint")
        if identity["tools_enabled"] is not False:
            raise ValueError(f"{group_id}: judge tools must be disabled")
        if identity["strict_json_schema"] is not True:
            raise ValueError(f"{group_id}: strict_json_schema must be true")
        if identity["provider_schema_omitted_keywords"] != ["uniqueItems"]:
            raise ValueError(f"{group_id}: unexpected provider schema adaptation")
        if identity["provider_schema_normalizations"] != []:
            raise ValueError(f"{group_id}: unexpected provider schema normalization")
        if identity["local_validation_uses_unmodified_schema"] is not True:
            raise ValueError(f"{group_id}: canonical local schema validation is required")
        for hash_field in ("prompt_sha256", "schema_sha256", "target_profile_sha256"):
            value = identity[hash_field]
            if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{group_id}: invalid {hash_field}")
        expected_run_hashes = {
            "prompt_sha256": frozen_resource_hashes["prompt"],
            "schema_sha256": frozen_resource_hashes["output_schema"],
            "target_profile_sha256": frozen_resource_hashes["target_profile"],
        }
        for hash_field, expected_hash in expected_run_hashes.items():
            if identity[hash_field] != expected_hash:
                raise ValueError(
                    f"{group_id}: run {hash_field} does not match the frozen resource"
                )
        if judge_identity is None:
            judge_identity = identity
        elif canonical_json(identity) != canonical_json(judge_identity):
            raise ValueError(f"{group_id}: mixed judge configuration in one analysis")
        by_id[group_id] = row
    if set(by_id) != set(groups_by_id):
        raise ValueError("judgment group IDs do not match prepared groups")
    assert judge_identity is not None
    return by_id, judge_identity


def assemble_samples(
    groups: Sequence[Mapping[str, Any]],
    labels_by_id: Mapping[str, Mapping[str, Any]],
    judgments_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        metadata = group["request_metadata"]
        group_id = str(metadata["group_id"])
        task = str(metadata["task_name"])
        condition_id = str(metadata["condition_id"])
        map_by_display = {item["display_id"]: item for item in metadata["sample_map"]}
        private = labels_by_id[group_id]["labels_by_display"]
        output_samples = {
            item["sample_id"]: item for item in judgments_by_id[group_id]["output"]["samples"]
        }
        if set(output_samples) != set(map_by_display):
            raise ValueError(f"{group_id}: output/sample-map IDs differ after validation")
        for display_id, sample in output_samples.items():
            model = str(map_by_display[display_id]["model_code"])
            quality = require_mapping(sample["quality"], f"{group_id}/{display_id}.quality")
            row = {
                "group_id": group_id,
                "task_name": task,
                "condition_id": condition_id,
                "condition_number": condition_number(condition_id),
                "display_id": display_id,
                "model_code": model,
                "model_name": MODEL_NAMES[model],
                "deterministic_pass": bool(private[display_id]["deterministic_pass"]),
                "valid": bool(sample["valid"]),
                "geq": float(conditional_geq(sample)),
                "severe_degeneration": bool(sample["severe_degeneration"]),
                "prompt_injection_detected": bool(sample["prompt_injection_detected"]),
            }
            row.update({dimension: int(quality[dimension]) for dimension in QUALITY_DIMENSIONS})
            rows.append(row)
    rows.sort(
        key=lambda row: (
            TASKS.index(str(row["task_name"])),
            int(row["condition_number"]),
            MODEL_INDEX[str(row["model_code"])],
        )
    )
    return rows


def task_matrices(
    rows: Sequence[Mapping[str, Any]], task: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [row for row in rows if row["task_name"] == task]
    by_condition: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in selected:
        by_condition.setdefault(str(row["condition_id"]), {})[str(row["model_code"])] = row
    ordered_ids = sorted(by_condition, key=condition_number)
    if len(ordered_ids) != EXPECTED_CONDITIONS[task]:
        raise ValueError(f"{task}: unexpected condition count during analysis")
    scores = np.empty((len(ordered_ids), len(MODELS)), dtype=np.float64)
    labels = np.empty((len(ordered_ids), len(MODELS)), dtype=np.bool_)
    for index, condition_id in enumerate(ordered_ids):
        if set(by_condition[condition_id]) != set(MODELS):
            raise ValueError(f"{condition_id}: missing model response")
        for model_index, model in enumerate(MODELS):
            row = by_condition[condition_id][model]
            scores[index, model_index] = float(row["geq"])
            labels[index, model_index] = bool(row["deterministic_pass"])
    return scores, labels, ordered_ids


def auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return math.nan
    # Conditional GEQ lies on the exact nine-point grid 0, 1.25, ..., 10.
    bins = np.rint(scores / 1.25).astype(np.int64)
    if np.any(bins < 0) or np.any(bins > 8) or not np.allclose(scores, 1.25 * bins):
        raise ValueError("GEQ scores are not on the declared 0--10 grid")
    positive_counts = np.bincount(bins[labels], minlength=9).astype(np.float64)
    negative_counts = np.bincount(bins[~labels], minlength=9).astype(np.float64)
    negative_below = np.cumsum(negative_counts) - negative_counts
    wins = np.sum(positive_counts * (negative_below + 0.5 * negative_counts))
    return float(wins / (positive * negative))


def pass_fail_association(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.any(labels) or np.all(labels):
        return math.nan
    return float(scores[labels].mean() - scores[~labels].mean())


def interval(draws: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(draws, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def construct_point(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float, int, int]:
    flat_labels = labels.reshape(-1)
    flat_scores = scores.reshape(-1)
    return (
        auc_binary(flat_labels, flat_scores),
        pass_fail_association(flat_labels, flat_scores),
        int(flat_labels.sum()),
        int(flat_labels.size - flat_labels.sum()),
    )


def bootstrap_task(
    scores: np.ndarray,
    labels: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    n = scores.shape[0]
    rng = np.random.default_rng(seed)
    model_mean_draws = np.empty((replicates, len(MODELS)), dtype=np.float64)
    paired_difference_draws = np.empty(replicates, dtype=np.float64)
    scopes = ("pooled", *MODELS)
    auc_draws = {scope: np.empty(replicates, dtype=np.float64) for scope in scopes}
    association_draws = {
        scope: np.empty(replicates, dtype=np.float64) for scope in scopes
    }
    for replicate in range(replicates):
        indices = rng.integers(0, n, size=n)
        sampled_scores = scores[indices]
        sampled_labels = labels[indices]
        model_mean_draws[replicate] = sampled_scores.mean(axis=0)
        paired_difference_draws[replicate] = (
            sampled_scores[:, MODEL_INDEX["dream"]]
            - sampled_scores[:, MODEL_INDEX["llada"]]
        ).mean()
        for scope in scopes:
            if scope == "pooled":
                scope_scores = sampled_scores
                scope_labels = sampled_labels
            else:
                column = MODEL_INDEX[scope]
                scope_scores = sampled_scores[:, column]
                scope_labels = sampled_labels[:, column]
            auc_draws[scope][replicate] = auc_binary(scope_labels, scope_scores)
            association_draws[scope][replicate] = pass_fail_association(
                scope_labels, scope_scores
            )

    discordant = labels[:, 0] != labels[:, 1]
    discordant_indices = np.flatnonzero(discordant)
    signed_differences = np.empty(discordant_indices.size, dtype=np.float64)
    for position, condition_index in enumerate(discordant_indices):
        passing_column = int(np.argmax(labels[condition_index]))
        failing_column = 1 - passing_column
        signed_differences[position] = (
            scores[condition_index, passing_column]
            - scores[condition_index, failing_column]
        )
    discordant_rng = np.random.default_rng(seed + 1_000_003)
    discordant_mean_draws = np.empty(replicates, dtype=np.float64)
    discordant_direction_draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = discordant_rng.integers(
            0, signed_differences.size, size=signed_differences.size
        )
        draw = signed_differences[indices]
        discordant_mean_draws[replicate] = draw.mean()
        discordant_direction_draws[replicate] = np.mean(draw > 0.0)

    return {
        "model_mean_draws": model_mean_draws,
        "paired_difference_draws": paired_difference_draws,
        "auc_draws": auc_draws,
        "association_draws": association_draws,
        "discordant_indices": discordant_indices,
        "discordant_signed_differences": signed_differences,
        "discordant_mean_draws": discordant_mean_draws,
        "discordant_direction_draws": discordant_direction_draws,
    }


def format_ci(point: float, low: float, high: float, digits: int = 2) -> str:
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def make_latex(
    task_model_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    construct_rows: Sequence[Mapping[str, Any]],
    discordant_rows: Sequence[Mapping[str, Any]],
    replicates: int,
) -> str:
    lines = [
        "% Generated by analyze_large_dlm_conditional.py; do not edit manually.",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Task & Model & Deterministic success & Conditional GEQ (95\\% CI) \\\\",
        "\\midrule",
    ]
    for row in task_model_rows:
        lines.append(
            f"{row['task_name']} & {row['model_name']} & "
            f"{100.0 * float(row['deterministic_accuracy']):.1f}\\% & "
            f"{format_ci(float(row['geq_mean']), float(row['geq_ci_low']), float(row['geq_ci_high']))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Task & $\\Delta$GEQ (Dream $-$ LLaDA) & Pooled AUROC (95\\% CI) \\\\",
            "\\midrule",
        ]
    )
    pooled_by_task = {
        str(row["task_name"]): row
        for row in construct_rows
        if row["scope"] == "pooled"
    }
    for paired in paired_rows:
        task = str(paired["task_name"])
        construct = pooled_by_task[task]
        lines.append(
            f"{task} & "
            f"{format_ci(float(paired['dream_minus_llada_geq']), float(paired['ci_low']), float(paired['ci_high']))} & "
            f"{format_ci(float(construct['auroc']), float(construct['auroc_ci_low']), float(construct['auroc_ci_high']))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            "Task & Discordant conditions & GEQ(pass) $-$ GEQ(fail) (95\\% CI) \\\\",
            "\\midrule",
        ]
    )
    for row in discordant_rows:
        lines.append(
            f"{row['task_name']} & {row['discordant_conditions']} & "
            f"{format_ci(float(row['mean_geq_pass_minus_fail']), float(row['ci_low']), float(row['ci_high']))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            f"% Intervals use a {replicates:,}-replicate condition-cluster bootstrap.",
            "% GEQ is task-conditional; GED is undefined for one response per model/condition.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--private-label-map", type=Path, required=True)
    parser.add_argument("--input-schema", type=Path, default=DEFAULT_INPUT_SCHEMA)
    parser.add_argument("--output-schema", type=Path, default=DEFAULT_OUTPUT_SCHEMA)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--target-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--generation-provenance",
        type=Path,
        default=None,
        help="Default: generation_provenance.json beside --groups",
    )
    parser.add_argument(
        "--preparation-summary",
        type=Path,
        default=None,
        help="Default: preparation_summary.json beside --groups",
    )
    parser.add_argument(
        "--sha256sums",
        type=Path,
        default=None,
        help="Default: SHA256SUMS beside --groups",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap-replicates must be at least 1,000")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    bundle_root = args.groups.resolve().parent
    provenance_path = (
        args.generation_provenance.resolve()
        if args.generation_provenance is not None
        else bundle_root / "generation_provenance.json"
    )
    preparation_summary_path = (
        args.preparation_summary.resolve()
        if args.preparation_summary is not None
        else bundle_root / "preparation_summary.json"
    )
    sums_path = (
        args.sha256sums.resolve()
        if args.sha256sums is not None
        else bundle_root / "SHA256SUMS"
    )
    verify_prepared_bundle(
        args.groups.resolve(),
        args.private_label_map.resolve(),
        provenance_path,
        preparation_summary_path,
        sums_path,
    )
    frozen_resource_hashes = validate_frozen_provenance(
        provenance_path,
        args.prompt.resolve(),
        args.input_schema.resolve(),
        args.output_schema.resolve(),
        args.target_profile.resolve(),
    )

    input_schema = load_json(args.input_schema)
    output_schema = load_json(args.output_schema)
    Draft202012Validator.check_schema(input_schema)
    Draft202012Validator.check_schema(output_schema)
    groups, groups_by_id = validate_groups(
        args.groups, Draft202012Validator(input_schema)
    )
    labels_by_id = validate_labels(args.private_label_map, groups_by_id)
    judgments_by_id, judge_identity = validate_judgments(
        args.judgments,
        groups_by_id,
        Draft202012Validator(output_schema),
        frozen_resource_hashes,
    )
    sample_rows = assemble_samples(groups, labels_by_id, judgments_by_id)

    task_model_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    construct_rows: list[dict[str, Any]] = []
    discordant_rows: list[dict[str, Any]] = []
    task_payload: dict[str, Any] = {}
    for task_index, task in enumerate(TASKS):
        scores, deterministic_labels, condition_ids = task_matrices(sample_rows, task)
        bootstrap = bootstrap_task(
            scores,
            deterministic_labels,
            args.bootstrap_replicates,
            args.bootstrap_seed + task_index * 10_000_019,
        )
        model_payload: dict[str, Any] = {}
        for model_index, model in enumerate(MODELS):
            geq_mean = float(scores[:, model_index].mean())
            geq_low, geq_high = interval(bootstrap["model_mean_draws"][:, model_index])
            passed = int(deterministic_labels[:, model_index].sum())
            total = int(deterministic_labels.shape[0])
            row = {
                "task_name": task,
                "model_code": model,
                "model_name": MODEL_NAMES[model],
                "conditions": total,
                "deterministic_passed": passed,
                "deterministic_accuracy": passed / total,
                "geq_mean": geq_mean,
                "geq_ci_low": geq_low,
                "geq_ci_high": geq_high,
            }
            task_model_rows.append(row)
            model_payload[model] = row

        paired_values = scores[:, MODEL_INDEX["dream"]] - scores[:, MODEL_INDEX["llada"]]
        paired_low, paired_high = interval(bootstrap["paired_difference_draws"])
        deterministic_difference = float(
            deterministic_labels[:, MODEL_INDEX["dream"]].mean()
            - deterministic_labels[:, MODEL_INDEX["llada"]].mean()
        )
        paired_row = {
            "task_name": task,
            "conditions": len(condition_ids),
            "difference_orientation": "dream_minus_llada",
            "dream_minus_llada_geq": float(paired_values.mean()),
            "ci_low": paired_low,
            "ci_high": paired_high,
            "dream_minus_llada_deterministic_accuracy": deterministic_difference,
            "geq_and_deterministic_differences_same_direction": (
                float(paired_values.mean()) * deterministic_difference > 0.0
            ),
        }
        paired_rows.append(paired_row)

        construct_payload: dict[str, Any] = {}
        for scope in ("pooled", *MODELS):
            if scope == "pooled":
                scope_scores = scores
                scope_labels = deterministic_labels
            else:
                column = MODEL_INDEX[scope]
                scope_scores = scores[:, column]
                scope_labels = deterministic_labels[:, column]
            auc, association, passed, failed = construct_point(scope_labels, scope_scores)
            auc_low, auc_high = interval(bootstrap["auc_draws"][scope])
            association_low, association_high = interval(
                bootstrap["association_draws"][scope]
            )
            construct_row = {
                "task_name": task,
                "scope": scope,
                "responses": passed + failed,
                "deterministic_passed": passed,
                "deterministic_failed": failed,
                "auroc": auc,
                "auroc_ci_low": auc_low,
                "auroc_ci_high": auc_high,
                "mean_geq_pass_minus_fail": association,
                "association_ci_low": association_low,
                "association_ci_high": association_high,
            }
            construct_rows.append(construct_row)
            construct_payload[scope] = construct_row

        signed = bootstrap["discordant_signed_differences"]
        discordant_low, discordant_high = interval(bootstrap["discordant_mean_draws"])
        direction_low, direction_high = interval(bootstrap["discordant_direction_draws"])
        discordant_row = {
            "task_name": task,
            "discordant_conditions": int(signed.size),
            "mean_geq_pass_minus_fail": float(signed.mean()),
            "ci_low": discordant_low,
            "ci_high": discordant_high,
            "fraction_pass_geq_greater_than_fail": float(np.mean(signed > 0.0)),
            "fraction_ci_low": direction_low,
            "fraction_ci_high": direction_high,
            "fraction_tied": float(np.mean(signed == 0.0)),
            "fraction_pass_geq_lower_than_fail": float(np.mean(signed < 0.0)),
        }
        discordant_rows.append(discordant_row)
        task_payload[task] = {
            "models": model_payload,
            "paired_model_comparison": paired_row,
            "construct_validity": construct_payload,
            "discordant_pass_vs_fail": discordant_row,
        }

    safe_sample_rows = [
        {
            field: row[field]
            for field in (
                "group_id",
                "task_name",
                "condition_id",
                "display_id",
                "model_code",
                "model_name",
                "deterministic_pass",
                "valid",
                "geq",
                *QUALITY_DIMENSIONS,
                "severe_degeneration",
                "prompt_injection_detected",
            )
        }
        for row in sample_rows
    ]
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "target_profile_id": TARGET_PROFILE_ID,
        "judge": judge_identity,
        "design": {
            "groups": len(groups),
            "responses": len(sample_rows),
            "tasks": list(TASKS),
            "models": list(MODELS),
            "judge_input_blinded_to_model": True,
            "judge_input_includes_hidden_reference_answers": False,
            "judge_input_includes_humaneval_tests_or_canonical_solutions": False,
            "ged_defined_or_reported": False,
        },
        "analysis": {
            "geq_range": [0.0, 10.0],
            "geq_dimensions": list(QUALITY_DIMENSIONS),
            "bootstrap_method": "ordinary_condition_cluster_bootstrap",
            "bootstrap_unit": "task condition",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence_interval": "two-sided percentile 95%",
            "paired_model_difference_orientation": "Dream minus LLaDA",
            "auroc_positive_class": "deterministic pass",
            "auroc_chance_value": 0.5,
            "association_orientation": "mean GEQ(pass) minus mean GEQ(fail)",
        },
        "tasks": task_payload,
        "source_sha256": {
            "groups": sha256_path(args.groups),
            "judgments": sha256_path(args.judgments),
            "private_label_map": sha256_path(args.private_label_map),
            "generation_provenance": sha256_path(provenance_path),
            "preparation_summary": sha256_path(preparation_summary_path),
            "prepared_bundle_sha256sums": sha256_path(sums_path),
            "prompt": frozen_resource_hashes["prompt"],
            "input_schema": sha256_path(args.input_schema),
            "output_schema": sha256_path(args.output_schema),
            "target_profile": frozen_resource_hashes["target_profile"],
        },
        "interpretation_limits": [
            "GEQ is task-conditional here and is not directly comparable to unconditional GroupEval pilots.",
            "AUROC and pass/fail association are construct-validity diagnostics, not human validation.",
            "The judge did not receive hidden answers, HumanEval tests, or canonical solutions and did not execute code.",
            "GSM8K exact-match and HumanEval functional success establish different task-specific orderings in these frozen runs.",
            "One response per model per condition does not identify within-model diversity, so GED is undefined and unreported.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sample_scores.csv": safe_sample_rows,
        "task_model_summary.csv": task_model_rows,
        "paired_model_comparison.csv": paired_rows,
        "construct_validity.csv": construct_rows,
        "discordant_comparison.csv": discordant_rows,
    }
    for filename, rows in paths.items():
        write_csv(args.output_dir / filename, rows)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "conditional_geq_table.tex").write_text(
        make_latex(
            task_model_rows,
            paired_rows,
            construct_rows,
            discordant_rows,
            args.bootstrap_replicates,
        ),
        encoding="utf-8",
    )
    manifest_files = [*paths, "summary.json", "conditional_geq_table.tex"]
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8", newline="") as handle:
        for filename in manifest_files:
            handle.write(f"{sha256_path(args.output_dir / filename)}  {filename}\n")
    os.chmod(args.output_dir / "SHA256SUMS", 0o644)
    print(
        json.dumps(
            {
                "groups": len(groups),
                "responses": len(sample_rows),
                "output_dir": str(args.output_dir),
                "tasks": task_payload,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
