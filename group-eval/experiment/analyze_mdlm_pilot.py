#!/usr/bin/env python3
"""Validate, score, bootstrap, and package a frozen GroupEval pilot.

The supported study profiles are the public MDLM-Small operating-point pilot
and the reward-guided DiDi-Instruct EMA-student operating-point pilot. The
input labels and generation provenance may contain private scratch paths.
Only aggregate, reviewer-safe outputs are published. Raw generations, blinded
method codes, API response IDs, and local paths are never copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORING = ROOT / "scoring"
VALIDATOR = SCORING / "validate_outputs.py"
SCORER = SCORING / "score_group_eval.py"
BOOTSTRAP = SCORING / "bootstrap_group_eval.py"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_PAIRING_MODE = "partitions"
DEFAULT_BOOTSTRAP_SEED = 20260724
GED_INTERPRETATION = (
    "Higher GED denotes more valid, balanced local semantic-discourse modes under "
    "the named judge. No human-calibrated sufficiency threshold or non-inferiority "
    "margin is applied."
)


@dataclass(frozen=True)
class StudyProfile:
    """Identity checks and presentation metadata for a frozen pilot family."""

    display_name: str
    artifact_name: str
    generation_artifact_name: str
    private_map_artifact_name: str
    model_label: str
    sampler_label: str
    source_metric_evaluator: str | None
    requires_noncanonical_design: bool
    figure_stem: str
    figure_title: str
    gppl_evaluator: str
    gppl_axis_label: str
    table_label: str
    table_caption: str


STUDY_PROFILES = {
    "mdlm-small": StudyProfile(
        display_name="public MDLM-Small",
        artifact_name="Reviewer-safe MDLM-Small GroupEval pilot analysis",
        generation_artifact_name="GroupEval MDLM-Small two-operating-point pilot",
        private_map_artifact_name="GroupEval MDLM-Small pilot private method map",
        model_label="public MDLM-Small",
        sampler_label="ancestral_cache",
        source_metric_evaluator=None,
        requires_noncanonical_design=False,
        figure_stem="mdlm_pilot_groupeval",
        figure_title="MDLM-Small GroupEval quality and diversity pilot",
        gppl_evaluator="GPT-2",
        gppl_axis_label="GPT-2 generative perplexity (log scale)",
        table_label="tab:groupeval-pilot",
        table_caption=(
            "GroupEval pilot on public MDLM-Small. ``Ref.'' is "
            "$\\tau=1.0$/fp64 and ``sel.'' is $\\tau=0.9$/fp32."
        ),
    ),
    "didi-instruct": StudyProfile(
        display_name="DiDi-Instruct EMA student",
        artifact_name="Reviewer-safe DiDi-Instruct GroupEval pilot analysis",
        generation_artifact_name=(
            "GroupEval DiDi-Instruct reward-guided two-operating-point pilot"
        ),
        private_map_artifact_name="GroupEval DiDi-Instruct pilot private method map",
        model_label="DiDi-Instruct EMA student",
        sampler_label="reward-guided ancestral",
        source_metric_evaluator="gpt2-large",
        requires_noncanonical_design=True,
        figure_stem="didi_pilot_groupeval",
        figure_title="DiDi-Instruct GroupEval quality and diversity pilot",
        gppl_evaluator="GPT-2-Large",
        gppl_axis_label="GPT-2-Large generative perplexity (log scale)",
        table_label="tab:groupeval-didi-pilot",
        table_caption=(
            "Exploratory {samples_per_method}-sample-per-operating-point "
            "GroupEval pilot on the "
            "reward-guided DiDi-Instruct EMA student. ``Ref.'' is "
            "$\\tau=1.0$/fp64 and ``sel.'' is $\\tau=0.9$/fp32."
        ),
    ),
}


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered = {
                field: (
                    f"{value:.8g}"
                    if isinstance((value := row.get(field)), float)
                    else value
                )
                for field in fieldnames
            }
            writer.writerow(rendered)


def run_tool(arguments: Sequence[str]) -> None:
    process = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout)[-6000:]
        raise RuntimeError(f"{Path(arguments[0]).name} failed:\n{detail}")


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected an object")
    return value


def require_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected an array")
    return value


def finite_float(value: Any, context: str, *, positive: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite number") from exc
    if not math.isfinite(converted) or (positive and converted <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{context}: expected a {qualifier} number, found {value!r}")
    return converted


def integer(value: Any, context: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: expected an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected an integer") from exc
    if str(converted) != str(value) and not isinstance(value, int):
        try:
            if float(value) != converted:
                raise ValueError(f"{context}: expected an integer")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: expected an integer") from exc
    if nonnegative and converted < 0:
        raise ValueError(f"{context}: expected a nonnegative integer")
    return converted


def boolean_from_csv(value: str, context: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{context}: expected a boolean, found {value!r}")


def public_condition(label: Mapping[str, Any], context: str) -> str:
    temperature = finite_float(label.get("temperature"), f"{context}.temperature")
    precision = str(label.get("categorical_precision", "")).lower()
    condition = str(label.get("condition", "")).lower()
    if math.isclose(temperature, 0.9) and precision == "fp32" and condition in {
        "selected",
        "calibrated",
    }:
        return "selected"
    if (
        math.isclose(temperature, 1.0)
        and precision == "fp64"
        and condition in {"reference", "unscaled-fp64-reference"}
    ):
        return "reference"
    raise ValueError(
        f"{context}: expected selected (T=0.9/fp32) or reference "
        f"(T=1.0/fp64), found temperature={temperature}, precision={precision!r}, "
        f"condition={condition!r}"
    )


def input_artifact(path: Path, records: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        descriptor["records"] = records
    return descriptor


def validate_study_identity(
    provenance: Mapping[str, Any],
    private_map: Mapping[str, Any],
    study: StudyProfile,
) -> int:
    """Prevent a valid archive from being silently relabeled as another study."""
    if provenance.get("artifact") != study.generation_artifact_name:
        raise ValueError(
            "generation_provenance.artifact does not match the selected study: "
            f"expected {study.generation_artifact_name!r}"
        )
    if private_map.get("artifact") != study.private_map_artifact_name:
        raise ValueError(
            "private_label_map.artifact does not match the selected study: "
            f"expected {study.private_map_artifact_name!r}"
        )

    design = require_mapping(provenance.get("design"), "generation_provenance.design")
    samples_per_method = integer(
        design.get("samples_per_method"), "design.samples_per_method"
    )
    if samples_per_method <= 0:
        raise ValueError("design.samples_per_method must be positive")
    noncanonical = design.get("noncanonical", False)
    if not isinstance(noncanonical, bool):
        raise ValueError("design.noncanonical must be a boolean when present")
    if study.requires_noncanonical_design and not noncanonical:
        raise ValueError(
            "the selected study requires provenance.design.noncanonical=true"
        )

    methods = require_mapping(private_map.get("methods"), "private_label_map.methods")
    for method, raw_label in methods.items():
        label = require_mapping(raw_label, f"private_label_map.methods.{method}")
        if label.get("model") != study.model_label:
            raise ValueError(
                f"private_label_map.methods.{method}.model does not match the "
                f"selected study ({study.model_label!r})"
            )
        if label.get("sampler") != study.sampler_label:
            raise ValueError(
                f"private_label_map.methods.{method}.sampler does not match the "
                f"selected study ({study.sampler_label!r})"
            )

    if study.source_metric_evaluator is not None:
        cells = require_sequence(
            provenance.get("cells"), "generation_provenance.cells"
        )
        for index, raw_cell in enumerate(cells):
            cell = require_mapping(raw_cell, f"generation_provenance.cells[{index}]")
            performance = require_mapping(
                cell.get("performance_for_cross_check_only"),
                f"generation_provenance.cells[{index}].performance",
            )
            evaluator = str(performance.get("evaluator", "")).strip().lower()
            if evaluator != study.source_metric_evaluator:
                raise ValueError(
                    f"generation_provenance.cells[{index}].performance.evaluator "
                    f"must be {study.source_metric_evaluator!r}"
                )
    return samples_per_method


def validate_input_alignment(
    groups: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    private_map: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[int, dict[str, str]],
]:
    methods_raw = require_mapping(
        private_map.get("methods"), "private_label_map.methods"
    )
    labels = {
        str(method): require_mapping(value, f"methods.{method}")
        for method, value in methods_raw.items()
    }
    cells_raw = require_sequence(provenance.get("cells"), "generation_provenance.cells")
    cells: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(cells_raw):
        cell = require_mapping(value, f"generation_provenance.cells[{index}]")
        method = str(cell.get("method_code", ""))
        if not method:
            raise ValueError(f"generation_provenance.cells[{index}].method_code is required")
        if method in cells:
            raise ValueError(f"duplicate provenance method_code {method}")
        cells[method] = cell

    def collect_group_ids(
        rows: Sequence[Mapping[str, Any]], source: str
    ) -> tuple[set[str], set[str]]:
        group_ids: set[str] = set()
        methods: set[str] = set()
        for index, row in enumerate(rows):
            metadata = require_mapping(
                row.get("request_metadata"), f"{source}[{index}].request_metadata"
            )
            group_id = str(metadata.get("group_id", ""))
            method = str(metadata.get("method_code", ""))
            if not group_id or not method:
                raise ValueError(f"{source}[{index}]: group_id and method_code are required")
            if group_id in group_ids:
                raise ValueError(f"{source}: duplicate group_id {group_id}")
            group_ids.add(group_id)
            methods.add(method)
        return group_ids, methods

    group_ids, group_methods = collect_group_ids(groups, "groups")
    judgment_ids, judgment_methods = collect_group_ids(judgments, "judgments")
    if not groups:
        raise ValueError("groups file is empty")
    if group_ids != judgment_ids:
        missing = sorted(group_ids - judgment_ids)
        unexpected = sorted(judgment_ids - group_ids)
        raise ValueError(
            f"incomplete judgments: missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    method_sets = {
        "groups": group_methods,
        "judgments": judgment_methods,
        "provenance": set(cells),
        "private labels": set(labels),
    }
    expected_methods = set(labels)
    for source, observed in method_sets.items():
        if observed != expected_methods:
            raise ValueError(
                f"{source} method set mismatch: missing={sorted(expected_methods-observed)}, "
                f"unexpected={sorted(observed-expected_methods)}"
            )

    design = require_mapping(provenance.get("design"), "generation_provenance.design")
    if integer(design.get("methods"), "design.methods") != len(expected_methods):
        raise ValueError("generation_provenance.design.methods is inconsistent")
    expected_group_size = integer(design.get("group_size"), "design.group_size")
    expected_samples = integer(
        design.get("samples_per_method"), "design.samples_per_method"
    )
    partition_seeds = require_sequence(
        design.get("partition_seeds"), "design.partition_seeds"
    )
    if not partition_seeds:
        raise ValueError("design.partition_seeds must be nonempty")

    partitions_by_method: dict[str, set[str]] = defaultdict(set)
    contexts_by_method_partition: dict[tuple[str, str], Counter[str]] = defaultdict(
        Counter
    )
    groups_by_method_partition: Counter[tuple[str, str]] = Counter()
    layouts_by_method: dict[
        str, dict[str, dict[int, tuple[str, ...]]]
    ] = defaultdict(lambda: defaultdict(dict))
    grouping_seeds_by_method: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(groups):
        metadata = require_mapping(row["request_metadata"], f"groups[{index}].request_metadata")
        method = str(metadata["method_code"])
        partition = str(metadata.get("partition_id", ""))
        if not partition:
            raise ValueError(f"groups[{index}].request_metadata.partition_id is required")
        group_index = integer(
            metadata.get("group_index"),
            f"groups[{index}].request_metadata.group_index",
            nonnegative=True,
        )
        grouping_seed = integer(
            metadata.get("grouping_seed"),
            f"groups[{index}].request_metadata.grouping_seed",
        )
        group = require_mapping(row.get("group_input"), f"groups[{index}].group_input")
        samples = require_sequence(group.get("samples"), f"groups[{index}].group_input.samples")
        sample_map = require_sequence(
            metadata.get("sample_map"), f"groups[{index}].request_metadata.sample_map"
        )
        if len(samples) != expected_group_size or len(sample_map) != expected_group_size:
            raise ValueError(f"groups[{index}]: unexpected group/sample-map size")
        partitions_by_method[method].add(partition)
        groups_by_method_partition[(method, partition)] += 1
        if group_index in layouts_by_method[method][partition]:
            raise ValueError(
                f"{method}/{partition}: duplicate group_index {group_index}"
            )
        sample_indices: list[str] = []
        for mapping in sample_map:
            item = require_mapping(mapping, f"groups[{index}].sample_map item")
            sample_index = str(item.get("sample_index"))
            sample_indices.append(sample_index)
            contexts_by_method_partition[(method, partition)][sample_index] += 1
        layouts_by_method[method][partition][group_index] = tuple(
            sorted(sample_indices, key=int)
        )
        prior_seed = grouping_seeds_by_method[method].get(partition)
        if prior_seed is not None and prior_seed != grouping_seed:
            raise ValueError(
                f"{method}/{partition}: grouping_seed varies across groups"
            )
        grouping_seeds_by_method[method][partition] = grouping_seed
    if expected_samples % expected_group_size:
        raise ValueError("samples_per_method is not divisible by group_size")
    expected_groups_per_partition = expected_samples // expected_group_size
    expected_sample_indices = {str(index) for index in range(expected_samples)}
    expected_total_groups = (
        len(expected_methods) * len(partition_seeds) * expected_groups_per_partition
    )
    if len(groups) != expected_total_groups:
        raise ValueError(
            f"expected {expected_total_groups} frozen groups, found {len(groups)}"
        )
    expected_partition_ids = {
        f"P{index}" for index in range(1, len(partition_seeds) + 1)
    }
    for method in expected_methods:
        if partitions_by_method[method] != expected_partition_ids:
            raise ValueError(
                f"{method}: partition IDs are inconsistent with provenance; "
                f"expected={sorted(expected_partition_ids)}, "
                f"found={sorted(partitions_by_method[method])}"
            )
        for partition in partitions_by_method[method]:
            context_counts = contexts_by_method_partition[(method, partition)]
            duplicated = sorted(
                sample_index
                for sample_index, count in context_counts.items()
                if count != 1
            )
            missing_indices = sorted(
                expected_sample_indices - set(context_counts), key=int
            )
            unexpected_indices = sorted(
                set(context_counts) - expected_sample_indices
            )
            if (
                len(context_counts) != expected_samples
                or duplicated
                or missing_indices
                or unexpected_indices
            ):
                raise ValueError(
                    f"{method}/{partition}: expected each of {expected_samples} sample "
                    f"contexts exactly once; unique={len(context_counts)}, "
                    f"nonunit_counts={duplicated[:12]}, missing={missing_indices[:12]}, "
                    f"unexpected={unexpected_indices[:12]}"
                )
            observed_groups = groups_by_method_partition[(method, partition)]
            if observed_groups != expected_groups_per_partition:
                raise ValueError(
                    f"{method}/{partition}: expected {expected_groups_per_partition} "
                    f"groups, found {observed_groups}"
                )
            observed_group_indices = set(layouts_by_method[method][partition])
            expected_group_indices = set(range(expected_groups_per_partition))
            if observed_group_indices != expected_group_indices:
                raise ValueError(
                    f"{method}/{partition}: group indices are inconsistent; "
                    f"expected={sorted(expected_group_indices)}, "
                    f"found={sorted(observed_group_indices)}"
                )

    # Partition-only paired inference is valid only if a partition label denotes
    # the same grouping of sample positions for every method. Sample texts remain
    # independent across methods; only this grouping-context block is shared.
    reference_method = sorted(expected_methods)[0]
    reference_layout = dict(layouts_by_method[reference_method])
    reference_seeds = dict(grouping_seeds_by_method[reference_method])
    for method in sorted(expected_methods - {reference_method}):
        if dict(layouts_by_method[method]) != reference_layout:
            raise ValueError(
                f"{method}: partition layouts are not aligned with "
                f"{reference_method}; partition-only pairing is invalid"
            )
        if dict(grouping_seeds_by_method[method]) != reference_seeds:
            raise ValueError(
                f"{method}: grouping seeds are not aligned with "
                f"{reference_method}; partition-only pairing is invalid"
            )

    seed_info = require_mapping(
        provenance.get("seed_and_pairing"), "generation_provenance.seed_and_pairing"
    )
    if seed_info.get("allowed_bootstrap_pairing_mode") != BOOTSTRAP_PAIRING_MODE:
        raise ValueError(
            "generation provenance does not permit partition-only bootstrap pairing"
        )
    if bool(seed_info.get("sample_level_common_random_numbers_declared")):
        raise ValueError(
            "this pilot wrapper requires independent sample weights; provenance "
            "unexpectedly declares sample-level common random numbers"
        )

    profile = require_mapping(
        provenance.get("target_profile"), "generation_provenance.target_profile"
    )
    profile_id = str(profile.get("id", ""))
    if str(private_map.get("target_profile_id", "")) != profile_id:
        raise ValueError("private label map and provenance target profiles differ")
    request_profiles = {
        str(require_mapping(row["request_metadata"], "request metadata").get("target_profile_id"))
        for row in groups
    }
    if request_profiles != {profile_id}:
        raise ValueError("groups and provenance target profiles differ")

    pairs: dict[int, dict[str, str]] = defaultdict(dict)
    for method in sorted(expected_methods):
        label = labels[method]
        cell = cells[method]
        updates = integer(
            label.get("configured_updates_M"),
            f"methods.{method}.configured_updates_M",
        )
        cell_updates = integer(
            cell.get("configured_updates_M"),
            f"cells.{method}.configured_updates_M",
        )
        if updates != cell_updates:
            raise ValueError(f"{method}: label/provenance update count mismatch")
        condition = public_condition(label, f"methods.{method}")
        if condition in pairs[updates]:
            raise ValueError(f"M={updates}: duplicate {condition} operating point")
        pairs[updates][condition] = method
        for field in ("temperature", "categorical_precision"):
            if str(label.get(field)) != str(cell.get(field)):
                raise ValueError(f"{method}: label/provenance {field} mismatch")
        if integer(cell.get("sample_count"), f"cells.{method}.sample_count") != expected_samples:
            raise ValueError(f"{method}: provenance sample count mismatch")
    for updates, conditions in pairs.items():
        if set(conditions) != {"reference", "selected"}:
            raise ValueError(
                f"M={updates}: expected one reference and one selected cell, "
                f"found {sorted(conditions)}"
            )
    if len(pairs) * 2 != len(expected_methods):
        raise ValueError("operating-point pairs do not cover every method")
    return labels, cells, dict(pairs)


def usage_value(usage: Mapping[str, Any], names: Sequence[str], context: str) -> int:
    present = [name for name in names if usage.get(name) is not None]
    if not present:
        raise ValueError(f"{context}: missing token usage field from {list(names)}")
    values = {
        integer(usage[name], f"{context}.{name}", nonnegative=True) for name in present
    }
    if len(values) != 1:
        raise ValueError(f"{context}: inconsistent synonymous token usage fields")
    return next(iter(values))


def judge_diagnostics(
    judgments: Sequence[Mapping[str, Any]], methods: Iterable[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_method: dict[str, dict[str, Any]] = {
        method: {
            "judge_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "transport_retries": 0,
            "content_retries": 0,
            "calls_with_retry": 0,
        }
        for method in methods
    }
    provider_values: set[str] = set()
    model_values: set[str] = set()
    response_model_values: set[str] = set()
    endpoint_values: set[str] = set()
    reasoning_effort_values: set[str] = set()
    reasoning_effort_source_values: set[str] = set()
    reasoning_effort_requested_values: set[str] = set()
    fingerprint_values: set[str] = set()
    reasoning_token_records = 0
    finish_reasons: Counter[str] = Counter()
    protocol_identity_values: dict[str, set[str]] = {
        field: set()
        for field in (
            "protocol_version",
            "protocol_sha256",
            "target_profile_id",
            "target_profile_sha256",
            "prompt_sha256",
            "schema_sha256",
            "strict_json_schema",
            "local_validation_uses_unmodified_schema",
            "tools_enabled",
            "provider_schema_omitted_keywords",
            "provider_schema_normalizations",
        )
    }

    for index, row in enumerate(judgments):
        metadata = require_mapping(
            row.get("request_metadata"), f"judgments[{index}].request_metadata"
        )
        method = str(metadata["method_code"])
        run = require_mapping(row.get("run_metadata"), f"judgments[{index}].run_metadata")
        usage = require_mapping(run.get("usage"), f"judgments[{index}].run_metadata.usage")
        prompt_tokens = usage_value(
            usage,
            ("prompt_tokens", "input_tokens"),
            f"judgments[{index}].usage",
        )
        completion_tokens = usage_value(
            usage,
            ("completion_tokens", "output_tokens"),
            f"judgments[{index}].usage",
        )
        total_tokens = usage_value(
            usage, ("total_tokens",), f"judgments[{index}].usage"
        )
        output_details = usage.get("output_tokens_details")
        reasoning_tokens = 0
        if output_details is not None:
            output_details = require_mapping(
                output_details, f"judgments[{index}].usage.output_tokens_details"
            )
            if output_details.get("reasoning_tokens") is not None:
                reasoning_tokens = integer(
                    output_details["reasoning_tokens"],
                    f"judgments[{index}].usage.output_tokens_details.reasoning_tokens",
                    nonnegative=True,
                )
                reasoning_token_records += 1
        if total_tokens < prompt_tokens + completion_tokens:
            raise ValueError(f"judgments[{index}]: total token count is inconsistent")
        transport = integer(
            run.get("transport_retries"),
            f"judgments[{index}].transport_retries",
            nonnegative=True,
        )
        content = integer(
            run.get("content_retries"),
            f"judgments[{index}].content_retries",
            nonnegative=True,
        )
        aggregate = by_method[method]
        aggregate["judge_calls"] += 1
        aggregate["prompt_tokens"] += prompt_tokens
        aggregate["completion_tokens"] += completion_tokens
        aggregate["reasoning_tokens"] += reasoning_tokens
        aggregate["total_tokens"] += total_tokens
        aggregate["transport_retries"] += transport
        aggregate["content_retries"] += content
        aggregate["calls_with_retry"] += int(transport + content > 0)
        provider_values.add(str(run.get("judge_provider", "")))
        model_values.add(str(run.get("judge_model_snapshot", "")))
        if run.get("provider_response_model") is not None:
            response_model_values.add(str(run["provider_response_model"]))
        if run.get("judge_endpoint") is not None:
            endpoint_values.add(str(run["judge_endpoint"]))
        if run.get("reasoning_effort") is not None:
            reasoning_effort_values.add(str(run["reasoning_effort"]))
        if run.get("reasoning_effort_source") is not None:
            reasoning_effort_source_values.add(str(run["reasoning_effort_source"]))
        if "reasoning_effort_requested" in run:
            reasoning_effort_requested_values.add(
                json.dumps(run["reasoning_effort_requested"], sort_keys=True)
            )
        fingerprint = run.get("system_fingerprint")
        if fingerprint is not None:
            fingerprint_values.add(str(fingerprint))
        finish_reason = run.get("finish_reason")
        finish_reasons[
            "not_available" if finish_reason in (None, "") else str(finish_reason)
        ] += 1
        protocol_identity = {
            "protocol_version": metadata.get("protocol_version"),
            "protocol_sha256": metadata.get("protocol_sha256"),
            "target_profile_id": metadata.get("target_profile_id"),
            "target_profile_sha256": run.get("target_profile_sha256"),
            "prompt_sha256": run.get("prompt_sha256"),
            "schema_sha256": run.get("schema_sha256"),
            "strict_json_schema": run.get("strict_json_schema"),
            "local_validation_uses_unmodified_schema": run.get(
                "local_validation_uses_unmodified_schema"
            ),
            "tools_enabled": run.get("tools_enabled"),
            "provider_schema_omitted_keywords": run.get(
                "provider_schema_omitted_keywords"
            ),
            "provider_schema_normalizations": run.get(
                "provider_schema_normalizations", []
            ),
        }
        for field, value in protocol_identity.items():
            if value is None:
                raise ValueError(
                    f"judgments[{index}]: missing protocol identity field {field}"
                )
            if field.endswith("_sha256"):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(
                        f"judgments[{index}].{field}: expected a lowercase SHA-256"
                    )
            protocol_identity_values[field].add(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )

    if len(provider_values) != 1 or "" in provider_values:
        raise ValueError(f"judgments contain inconsistent judge providers: {provider_values}")
    if len(model_values) != 1 or "" in model_values:
        raise ValueError(f"judgments contain inconsistent judge models: {model_values}")
    for field, values in (
        ("provider response models", response_model_values),
        ("judge endpoints", endpoint_values),
        ("reasoning efforts", reasoning_effort_values),
        ("reasoning effort sources", reasoning_effort_source_values),
        ("requested reasoning efforts", reasoning_effort_requested_values),
    ):
        if len(values) > 1 or "" in values:
            raise ValueError(f"judgments contain inconsistent {field}: {values}")
    provider = next(iter(provider_values))
    if provider == "openai":
        if (
            not response_model_values
            or not endpoint_values
            or not reasoning_effort_values
        ):
            raise ValueError(
                "OpenAI judgments must record provider_response_model, judge_endpoint, "
                "and reasoning_effort"
            )
        if bool(reasoning_effort_source_values) != bool(
            reasoning_effort_requested_values
        ):
            raise ValueError(
                "OpenAI judgments must record reasoning_effort_source and "
                "reasoning_effort_requested together"
            )
        if reasoning_token_records != len(judgments):
            raise ValueError(
                "OpenAI judgments must record reasoning_tokens for every response"
            )
    if len(fingerprint_values) > 1:
        raise ValueError(
            f"judgments contain inconsistent serving fingerprints: {fingerprint_values}"
        )
    for field, values in protocol_identity_values.items():
        if len(values) != 1:
            raise ValueError(
                f"judgments contain inconsistent protocol identity field "
                f"{field}: {values}"
            )
    for method, aggregate in by_method.items():
        calls = int(aggregate["judge_calls"])
        if calls <= 0:
            raise ValueError(f"{method}: no judge calls found")
        aggregate["mean_prompt_tokens_per_call"] = aggregate["prompt_tokens"] / calls
        aggregate["mean_completion_tokens_per_call"] = (
            aggregate["completion_tokens"] / calls
        )

    totals = {
        key: sum(int(values[key]) for values in by_method.values())
        for key in (
            "judge_calls",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "transport_retries",
            "content_retries",
            "calls_with_retry",
        )
    }
    global_summary = {
        "judge_provider": provider,
        "judge_model": next(iter(model_values)),
        "provider_response_model": (
            next(iter(response_model_values)) if response_model_values else None
        ),
        "judge_endpoint": next(iter(endpoint_values)) if endpoint_values else None,
        "reasoning_effort": (
            next(iter(reasoning_effort_values)) if reasoning_effort_values else None
        ),
        "reasoning_effort_source": (
            next(iter(reasoning_effort_source_values))
            if reasoning_effort_source_values
            else None
        ),
        "reasoning_effort_requested": (
            json.loads(next(iter(reasoning_effort_requested_values)))
            if reasoning_effort_requested_values
            else None
        ),
        "system_fingerprint": (
            next(iter(fingerprint_values)) if fingerprint_values else None
        ),
        "protocol_identity": {
            field: json.loads(next(iter(values)))
            for field, values in protocol_identity_values.items()
        },
        **totals,
        "token_usage_scope": "accepted_response_attempts_only",
        "retry_attempt_token_usage_included": False,
        "token_usage_limitation": (
            "Provider token usage is retained for each accepted response attempt only. "
            "Retry counts include prior rejected content attempts and transport "
            "retries, whose token usage is not inferred or added."
        ),
        "retry_call_rate": totals["calls_with_retry"] / totals["judge_calls"],
        "finish_reasons": dict(sorted(finish_reasons.items())),
    }
    return by_method, global_summary


def source_metrics(cell: Mapping[str, Any], context: str) -> tuple[float, float]:
    performance = require_mapping(
        cell.get("performance_for_cross_check_only"), f"{context}.performance"
    )
    gppl = finite_float(
        performance.get("generative_perplexity"),
        f"{context}.generative_perplexity",
        positive=True,
    )
    entropy_key = "avg_entropy" if "avg_entropy" in performance else "entropy"
    entropy = finite_float(
        performance.get(entropy_key), f"{context}.{entropy_key}"
    )
    alternate = performance.get("avg_entropy_per_sequence")
    if alternate is not None:
        alternate_value = finite_float(alternate, f"{context}.avg_entropy_per_sequence")
        if not math.isclose(entropy, alternate_value, rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError(f"{context}: archived entropy implementations disagree")
    return gppl, entropy


def build_results(
    labels: Mapping[str, Mapping[str, Any]],
    cells: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[int, Mapping[str, str]],
    system_rows: Sequence[Mapping[str, str]],
    bootstrap_scores: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
    usage_by_method: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    systems = {str(row["method_code"]): row for row in system_rows}
    boot = {str(row["method_code"]): row for row in bootstrap_scores}
    if set(systems) != set(labels) or set(boot) != set(labels):
        raise ValueError("score/bootstrap method sets differ from private labels")
    comparison_map = {
        (str(row["candidate"]), str(row["reference"])): row for row in comparisons
    }

    results: list[dict[str, Any]] = []
    for updates in sorted(pairs):
        for condition in ("reference", "selected"):
            method = pairs[updates][condition]
            label = labels[method]
            cell = cells[method]
            system = systems[method]
            score = boot[method]
            usage = usage_by_method[method]
            gppl, entropy = source_metrics(cell, f"cell {method}")
            results.append(
                {
                    "cell": f"M{updates}-{condition}",
                    "updates_M": updates,
                    "operating_point": condition,
                    "temperature": finite_float(label["temperature"], f"{method}.temperature"),
                    "categorical_precision": str(label["categorical_precision"]),
                    "gpt2_generative_perplexity": gppl,
                    "within_sequence_entropy": entropy,
                    "geq": finite_float(score["geq"], f"{method}.geq"),
                    "geq_ci_low": finite_float(score["geq_ci_low"], f"{method}.geq_ci_low"),
                    "geq_ci_high": finite_float(score["geq_ci_high"], f"{method}.geq_ci_high"),
                    "ged": finite_float(score["ged"], f"{method}.ged"),
                    "ged_ci_low": finite_float(score["ged_ci_low"], f"{method}.ged_ci_low"),
                    "ged_ci_high": finite_float(score["ged_ci_high"], f"{method}.ged_ci_high"),
                    "invalid_rate": finite_float(system["invalid_rate"], f"{method}.invalid_rate"),
                    "severe_degeneration_rate": finite_float(
                        system["severe_degeneration_rate"],
                        f"{method}.severe_degeneration_rate",
                    ),
                    "quality_p10": finite_float(system["quality_p10"], f"{method}.quality_p10"),
                    "bottom_quartile_quality": finite_float(
                        system["bottom_quartile_quality"],
                        f"{method}.bottom_quartile_quality",
                    ),
                    "mean_effective_mode_count": finite_float(
                        system["mean_effective_mode_count"],
                        f"{method}.mean_effective_mode_count",
                    ),
                    "mean_dominant_mode_share": finite_float(
                        system["mean_dominant_mode_share"],
                        f"{method}.mean_dominant_mode_share",
                    ),
                    "all_singleton_group_rate": finite_float(
                        system["all_singleton_group_rate"],
                        f"{method}.all_singleton_group_rate",
                    ),
                    "ged_ceiling_rate": finite_float(
                        system["ged_ceiling_rate"], f"{method}.ged_ceiling_rate"
                    ),
                    "hybrid_rate": finite_float(system["hybrid_rate"], f"{method}.hybrid_rate"),
                    "low_clustering_confidence_rate": finite_float(
                        system["low_clustering_confidence_rate"],
                        f"{method}.low_clustering_confidence_rate",
                    ),
                    "nonhigh_clustering_confidence_rate": finite_float(
                        system["nonhigh_clustering_confidence_rate"],
                        f"{method}.nonhigh_clustering_confidence_rate",
                    ),
                    "judge_calls": int(usage["judge_calls"]),
                    "judge_prompt_tokens": int(usage["prompt_tokens"]),
                    "judge_completion_tokens": int(usage["completion_tokens"]),
                    "transport_retries": int(usage["transport_retries"]),
                    "content_retries": int(usage["content_retries"]),
                }
            )

    result_by_key = {
        (int(row["updates_M"]), str(row["operating_point"])): row for row in results
    }
    deltas: list[dict[str, Any]] = []
    for updates in sorted(pairs):
        candidate = pairs[updates]["selected"]
        reference = pairs[updates]["reference"]
        comparison = comparison_map.get((candidate, reference))
        if comparison is None:
            raise ValueError(f"missing selected-minus-reference comparison for M={updates}")
        calibrated = result_by_key[(updates, "selected")]
        baseline = result_by_key[(updates, "reference")]
        deltas.append(
            {
                "updates_M": updates,
                "comparison": "selected-minus-reference",
                "delta_gpt2_generative_perplexity": (
                    calibrated["gpt2_generative_perplexity"]
                    - baseline["gpt2_generative_perplexity"]
                ),
                "gpt2_perplexity_ratio": (
                    calibrated["gpt2_generative_perplexity"]
                    / baseline["gpt2_generative_perplexity"]
                ),
                "delta_within_sequence_entropy": (
                    calibrated["within_sequence_entropy"]
                    - baseline["within_sequence_entropy"]
                ),
                "delta_geq": finite_float(comparison["delta_geq"], f"M={updates}.delta_geq"),
                "delta_geq_ci_low": finite_float(
                    comparison["delta_geq_ci_low"], f"M={updates}.delta_geq_ci_low"
                ),
                "delta_geq_ci_high": finite_float(
                    comparison["delta_geq_ci_high"], f"M={updates}.delta_geq_ci_high"
                ),
                "delta_geq_lcb_95": finite_float(
                    comparison["delta_geq_lcb_95"], f"M={updates}.delta_geq_lcb"
                ),
                "quality_superior_at_95pct_one_sided": boolean_from_csv(
                    comparison["quality_superior"], f"M={updates}.quality_superior"
                ),
                "delta_ged": finite_float(comparison["delta_ged"], f"M={updates}.delta_ged"),
                "delta_ged_ci_low": finite_float(
                    comparison["delta_ged_ci_low"], f"M={updates}.delta_ged_ci_low"
                ),
                "delta_ged_ci_high": finite_float(
                    comparison["delta_ged_ci_high"], f"M={updates}.delta_ged_ci_high"
                ),
                "delta_ged_lcb_95": finite_float(
                    comparison["delta_ged_lcb_95"], f"M={updates}.delta_ged_lcb"
                ),
                "ged_interpretation": GED_INTERPRETATION,
            }
        )
    return results, deltas


def reviewer_provenance(
    provenance: Mapping[str, Any],
    labels: Mapping[str, Mapping[str, Any]],
    cells: Mapping[str, Mapping[str, Any]],
    pairs: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    profile = require_mapping(provenance["target_profile"], "target_profile")
    checkpoint = require_mapping(provenance["checkpoint"], "checkpoint")
    public_cells: list[dict[str, Any]] = []
    for updates in sorted(pairs):
        for condition in ("reference", "selected"):
            method = pairs[updates][condition]
            label = labels[method]
            cell = cells[method]
            result_file = require_mapping(cell.get("result_file"), f"{method}.result_file")
            generation_config = require_mapping(
                cell.get(
                    "generation_config",
                    cell.get("hydra_config", cell.get("run_log")),
                ),
                f"{method}.generation_config",
            )
            public_cells.append(
                {
                    "cell": f"M{updates}-{condition}",
                    "updates_M": updates,
                    "operating_point": condition,
                    "model": str(label.get("model")),
                    "sampler": str(label.get("sampler")),
                    "temperature": finite_float(label.get("temperature"), f"{method}.temperature"),
                    "categorical_precision": str(label.get("categorical_precision")),
                    "sample_count": integer(cell.get("sample_count"), f"{method}.sample_count"),
                    "global_lightning_seed": integer(
                        cell.get("global_lightning_seed"), f"{method}.global_lightning_seed"
                    ),
                    "source_timestamp": str(cell.get("source_timestamp")),
                    "result_file_sha256": str(result_file.get("sha256")),
                    "generation_config_sha256": str(
                        generation_config.get("sha256")
                    ),
                    "normalized_sample_hash_list_sha256": str(
                        cell.get("normalized_sample_hash_list_sha256")
                    ),
                }
            )
    return {
        "artifact": str(provenance.get("artifact")),
        "preparation_version": str(provenance.get("preparation_version")),
        "target_profile": {
            "id": str(profile.get("id")),
            "sha256": str(profile.get("sha256")),
        },
        "checkpoint": {
            "sha256": str(checkpoint.get("sha256")),
            "bytes": integer(checkpoint.get("bytes"), "checkpoint.bytes"),
        },
        "normalization": provenance.get("normalization"),
        "seed_and_pairing": provenance.get("seed_and_pairing"),
        "design": provenance.get("design"),
        "archive_limitations": provenance.get("archive_limitations"),
        "cells": public_cells,
    }


def descriptive_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 3 or np.std(x_array) == 0 or np.std(y_array) == 0:
        return None
    return float(np.corrcoef(x_array, y_array)[0, 1])


def make_figure(
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    study: StudyProfile,
) -> bool:
    cache_root = Path(tempfile.gettempdir()) / f"groupeval-matplotlib-{os.getuid()}"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return False

    updates_values = sorted({int(row["updates_M"]) for row in results})
    color_map = plt.get_cmap("viridis")
    colors = {
        updates: color_map(index / max(1, len(updates_values) - 1))
        for index, updates in enumerate(updates_values)
    }
    by_key = {
        (int(row["updates_M"]), str(row["operating_point"])): row for row in results
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    for updates in updates_values:
        pair = [by_key[(updates, condition)] for condition in ("reference", "selected")]
        axes[0].plot(
            [float(row["gpt2_generative_perplexity"]) for row in pair],
            [float(row["geq"]) for row in pair],
            color=colors[updates],
            linewidth=1.0,
            alpha=0.65,
        )
        axes[1].plot(
            [float(row["within_sequence_entropy"]) for row in pair],
            [float(row["ged"]) for row in pair],
            color=colors[updates],
            linewidth=1.0,
            alpha=0.65,
        )
        for condition, marker, filled in (
            ("reference", "o", False),
            ("selected", "s", True),
        ):
            row = by_key[(updates, condition)]
            face = colors[updates] if filled else "white"
            gppl = float(row["gpt2_generative_perplexity"])
            geq = float(row["geq"])
            axes[0].plot(
                gppl,
                geq,
                marker=marker,
                markersize=5,
                markerfacecolor=face,
                markeredgecolor=colors[updates],
                color=colors[updates],
                linewidth=0,
            )
            entropy = float(row["within_sequence_entropy"])
            ged = float(row["ged"])
            axes[1].plot(
                entropy,
                ged,
                marker=marker,
                markersize=5,
                markerfacecolor=face,
                markeredgecolor=colors[updates],
                color=colors[updates],
                linewidth=0,
            )
        calibrated = by_key[(updates, "selected")]
        axes[0].annotate(
            f"{updates}",
            (float(calibrated["gpt2_generative_perplexity"]), float(calibrated["geq"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
        axes[1].annotate(
            f"{updates}",
            (float(calibrated["within_sequence_entropy"]), float(calibrated["ged"])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel(study.gppl_axis_label)
    axes[0].set_ylabel("GEQ perceived quality")
    axes[1].set_xlabel("Within-sequence entropy $H$")
    axes[1].set_ylabel("GED local semantic diversity")
    for axis in axes:
        axis.margins(y=0.08)
        axis.grid(alpha=0.22, linewidth=0.6)
    condition_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.35",
            markerfacecolor="white",
            linewidth=0,
            label=r"Reference: $\tau=1.0$, fp64",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="0.35",
            markerfacecolor="0.35",
            linewidth=0,
            label=r"Selected: $\tau=0.9$, fp32",
        ),
    ]
    axes[0].legend(handles=condition_handles, fontsize=7, loc="best", frameon=False)
    fig.tight_layout(pad=0.8)
    stem = output_dir / study.figure_stem
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Title": study.figure_title,
            "Subject": "Aggregate reviewer-safe GroupEval results",
        },
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        metadata={"Description": "Aggregate reviewer-safe GroupEval results"},
    )
    plt.close(fig)
    return True


def latex_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def make_latex(
    results: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int,
    study: StudyProfile,
    samples_per_method: int,
) -> str:
    by_key = {
        (int(row["updates_M"]), str(row["operating_point"])): row for row in results
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rcccccc}",
        r"\toprule",
        (
            r"$M$ & gPPL (ref./sel.) & $H$ (ref./sel.) & "
            r"GEQ$\uparrow$ (ref./sel.) & $\Delta$GEQ [95\% BB int.] & "
            r"GED$\uparrow$ (ref./sel.) & $\Delta$GED [95\% BB int.] \\"
        ),
        r"\midrule",
    ]
    for delta in deltas:
        updates = int(delta["updates_M"])
        reference = by_key[(updates, "reference")]
        calibrated = by_key[(updates, "selected")]
        lines.append(
            " & ".join(
                [
                    str(updates),
                    (
                        f"{latex_number(float(reference['gpt2_generative_perplexity']), 1)} / "
                        f"{latex_number(float(calibrated['gpt2_generative_perplexity']), 1)}"
                    ),
                    (
                        f"{latex_number(float(reference['within_sequence_entropy']))} / "
                        f"{latex_number(float(calibrated['within_sequence_entropy']))}"
                    ),
                    (
                        f"{latex_number(float(reference['geq']))} / "
                        f"{latex_number(float(calibrated['geq']))}"
                    ),
                    (
                        f"{latex_number(float(delta['delta_geq']))} "
                        f"[{latex_number(float(delta['delta_geq_ci_low']))}, "
                        f"{latex_number(float(delta['delta_geq_ci_high']))}]"
                    ),
                    (
                        f"{latex_number(float(reference['ged']))} / "
                        f"{latex_number(float(calibrated['ged']))}"
                    ),
                    (
                        f"{latex_number(float(delta['delta_ged']))} "
                        f"[{latex_number(float(delta['delta_ged_ci_low']))}, "
                        f"{latex_number(float(delta['delta_ged_ci_high']))}]"
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            (
                rf"\caption{{{study.table_caption.format(samples_per_method=samples_per_method)} "
                r"Deltas are selected "
                r"minus reference within the same sampling budget; intervals use a "
                f"{bootstrap_replicates:,}-replicate partition-blocked Bayesian bootstrap sharing "
                r"partition weights only. Higher GED denotes more valid, balanced "
                r"judged semantic--discourse modes; no human-calibrated sufficiency "
                r"threshold is imposed.}"
            ),
            rf"\label{{{study.table_label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def assert_reviewer_safe(payload: Any, context: str = "payload") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert_reviewer_safe(value, f"{context}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_reviewer_safe(value, f"{context}[{index}]")
    elif isinstance(payload, str):
        normalized = payload.replace("\\", "/")
        if (
            normalized.startswith("/")
            or "/scratch/" in normalized
            or "/home/" in normalized
            or normalized.startswith("~/")
        ):
            raise ValueError(f"{context}: reviewer output contains a private path")


RESULT_FIELDS = (
    "cell",
    "updates_M",
    "operating_point",
    "temperature",
    "categorical_precision",
    "gpt2_generative_perplexity",
    "within_sequence_entropy",
    "geq",
    "geq_ci_low",
    "geq_ci_high",
    "ged",
    "ged_ci_low",
    "ged_ci_high",
    "invalid_rate",
    "severe_degeneration_rate",
    "quality_p10",
    "bottom_quartile_quality",
    "mean_effective_mode_count",
    "mean_dominant_mode_share",
    "all_singleton_group_rate",
    "ged_ceiling_rate",
    "hybrid_rate",
    "low_clustering_confidence_rate",
    "nonhigh_clustering_confidence_rate",
    "judge_calls",
    "judge_prompt_tokens",
    "judge_completion_tokens",
    "transport_retries",
    "content_retries",
)

DELTA_FIELDS = (
    "updates_M",
    "comparison",
    "delta_gpt2_generative_perplexity",
    "gpt2_perplexity_ratio",
    "delta_within_sequence_entropy",
    "delta_geq",
    "delta_geq_ci_low",
    "delta_geq_ci_high",
    "delta_geq_lcb_95",
    "quality_superior_at_95pct_one_sided",
    "delta_ged",
    "delta_ged_ci_low",
    "delta_ged_ci_high",
    "delta_ged_lcb_95",
    "ged_interpretation",
)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    study = STUDY_PROFILES[args.study]
    inputs = {
        "judgments": args.judgments.resolve(),
        "groups": args.groups.resolve(),
        "generation_provenance": args.generation_provenance.resolve(),
        "private_label_map": args.private_label_map.resolve(),
    }
    for role, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role} file not found: {path}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a fresh path"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    groups = load_jsonl(inputs["groups"])
    judgments = load_jsonl(inputs["judgments"])
    provenance = load_json(inputs["generation_provenance"])
    private_map = load_json(inputs["private_label_map"])
    samples_per_method = validate_study_identity(provenance, private_map, study)
    labels, cells, pairs = validate_input_alignment(
        groups, judgments, provenance, private_map
    )
    usage_by_method, judge_summary = judge_diagnostics(judgments, labels)

    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        with tempfile.TemporaryDirectory(prefix="groupeval-pilot-analysis-") as work_name:
            work = Path(work_name)
            validation_report = work / "validation_report.json"
            score_dir = work / "scores"
            bootstrap_dir = work / "bootstrap"
            run_tool(
                [
                    str(VALIDATOR),
                    "--input",
                    str(inputs["judgments"]),
                    "--groups",
                    str(inputs["groups"]),
                    "--report",
                    str(validation_report),
                ]
            )
            validation = load_json(validation_report)
            if (
                validation.get("invalid_records") != 0
                or validation.get("dataset_errors")
                or validation.get("valid_records") != len(groups)
            ):
                raise ValueError("judgment validation did not establish completeness")
            run_tool(
                [
                    str(SCORER),
                    "--judgments",
                    str(inputs["judgments"]),
                    "--out-dir",
                    str(score_dir),
                ]
            )
            run_tool(
                [
                    str(BOOTSTRAP),
                    "--judgments",
                    str(inputs["judgments"]),
                    "--out-dir",
                    str(bootstrap_dir),
                    "--replicates",
                    str(args.bootstrap_replicates),
                    "--seed",
                    str(args.bootstrap_seed),
                    "--pairing-mode",
                    BOOTSTRAP_PAIRING_MODE,
                ]
            )
            bootstrap_metadata = load_json(bootstrap_dir / "bootstrap.json")
            bootstrap_config = require_mapping(
                bootstrap_metadata.get("bootstrap"), "bootstrap.bootstrap"
            )
            if bootstrap_config.get("pairing_mode") != BOOTSTRAP_PAIRING_MODE:
                raise ValueError("bootstrap did not use partition-only pairing")
            if integer(
                bootstrap_config.get("replicates"), "bootstrap.replicates"
            ) != args.bootstrap_replicates:
                raise ValueError("bootstrap replicate count mismatch")
            if (
                bootstrap_metadata.get("ged_threshold") is not None
                or bootstrap_metadata.get("ged_noninferiority_margin") is not None
            ):
                raise ValueError("pilot bootstrap unexpectedly applied a GED decision threshold")

            results, deltas = build_results(
                labels,
                cells,
                pairs,
                load_csv(score_dir / "system_scores.csv"),
                load_csv(bootstrap_dir / "bootstrap_scores.csv"),
                load_csv(bootstrap_dir / "bootstrap_comparisons.csv"),
                usage_by_method,
            )

        assert staging is not None
        write_csv(staging / "pilot_results.csv", results, RESULT_FIELDS)
        write_csv(staging / "pilot_deltas.csv", deltas, DELTA_FIELDS)
        latex = make_latex(
            results,
            deltas,
            args.bootstrap_replicates,
            study,
            samples_per_method,
        )
        (staging / "pilot_table.tex").write_text(latex, encoding="utf-8")
        figures_generated = make_figure(results, staging, study)

        log_gppl = [
            math.log(float(row["gpt2_generative_perplexity"])) for row in results
        ]
        geq = [float(row["geq"]) for row in results]
        entropy = [float(row["within_sequence_entropy"]) for row in results]
        ged = [float(row["ged"]) for row in results]
        summary: dict[str, Any] = {
            "artifact": study.artifact_name,
            "analysis": {
                "study": args.study,
                "generator": study.display_name,
                "gppl_evaluator": study.gppl_evaluator,
                "samples_per_operating_point": samples_per_method,
                "bootstrap_method": "crossed_bayesian_bootstrap",
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_pairing_mode": BOOTSTRAP_PAIRING_MODE,
                "canonical_replicate_count": (
                    args.bootstrap_replicates == BOOTSTRAP_REPLICATES
                ),
                "judgment_completeness": {
                    "expected_groups": len(groups),
                    "valid_judgments": len(judgments),
                    "complete": len(groups) == len(judgments),
                },
                "ged_interpretation": GED_INTERPRETATION,
                "figures_generated": figures_generated,
                "descriptive_correlations": {
                    "pearson_log_gpt2_gppl_vs_geq": descriptive_correlation(
                        log_gppl, geq
                    ),
                    "pearson_entropy_vs_ged": descriptive_correlation(entropy, ged),
                    "note": (
                        f"Descriptive across-cell associations only; {len(results)} "
                        "operating points do not support strong correlational claims."
                    ),
                },
            },
            "input_artifacts": {
                role: input_artifact(
                    path,
                    records=(
                        len(judgments)
                        if role == "judgments"
                        else len(groups)
                        if role == "groups"
                        else None
                    ),
                )
                for role, path in inputs.items()
            },
            "judge_diagnostics": judge_summary,
            "generation_provenance": reviewer_provenance(
                provenance, labels, cells, pairs
            ),
            "results": results,
            "within_budget_comparisons": deltas,
            "interpretation_notes": [
                (
                    f"GEQ is the quality outcome; {study.gppl_evaluator} gPPL is an "
                    "evaluator-likelihood diagnostic and is shown on a logarithmic axis."
                ),
                GED_INTERPRETATION,
                (
                    "All-singleton, GED-ceiling, dominant-mode, hybrid, and clustering-"
                    "confidence rates must accompany GED because local semantic modes "
                    "can saturate on heterogeneous web text."
                ),
                (
                    "Sample weights are independent across methods because the archive "
                    "does not establish sample-level common random numbers; only shared "
                    "partition weights are used."
                ),
                (
                    "Bayesian-bootstrap intervals condition on the accepted judge "
                    "outputs, their fixed local mode assignments, and two frozen "
                    "overlap-reduced partitions. They do not model re-judging, "
                    "re-clustering, service drift, or arbitrary future partitions."
                ),
                (
                    "GED is a nonlinear weighted entropy functional. Its percentile "
                    "Bayesian-bootstrap interval need not be centered on, or contain, "
                    "the equal-weight plug-in estimate; this is not a confidence "
                    "interval for a linear sample mean."
                ),
                (
                    "Reported judge token totals cover accepted response attempts only; "
                    "retry counts include prior rejected/transport attempts, and omitted "
                    "retry usage is not estimated."
                ),
            ],
        }
        assert_reviewer_safe(summary)
        (staging / "pilot_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for path in (
            staging / "pilot_results.csv",
            staging / "pilot_deltas.csv",
            staging / "pilot_table.tex",
            staging / "pilot_summary.json",
        ):
            text = path.read_text(encoding="utf-8")
            if "/scratch/" in text or "/home/" in text:
                raise ValueError(f"{path.name}: reviewer package leaked a private path")
        os.replace(staging, output_dir)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)

    return {
        "output_dir": str(output_dir),
        "cells": len(labels),
        "within_budget_comparisons": len(pairs),
        "judgments": len(judgments),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_pairing_mode": BOOTSTRAP_PAIRING_MODE,
        "figures_generated": (output_dir / f"{study.figure_stem}.pdf").is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--generation-provenance", required=True, type=Path)
    parser.add_argument("--private-label-map", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--study",
        choices=tuple(STUDY_PROFILES),
        default="mdlm-small",
        help="Presentation/provenance profile (default preserves the MDLM-Small pilot).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    result = analyze(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
