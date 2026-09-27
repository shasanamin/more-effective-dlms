#!/usr/bin/env python3
"""Combine three reviewer-safe MDLM pilot analyses without pooling judges.

Each input must be an output directory created by ``analyze_mdlm_pilot.py``.
The bridge verifies that all three analyses use the exact same frozen groups,
generation provenance, ten cells, and source gPPL/entropy measurements.  It
then emits judge-separated GEQ/GED values and descriptive judge-sensitivity
statistics.  It never averages, calibrates, or otherwise pools judge scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


BUDGETS = (4, 16, 64, 256, 1024)
OPERATING_POINTS = ("reference", "selected")
EXPECTED_CELLS = tuple(
    (budget, operating_point)
    for budget in BUDGETS
    for operating_point in OPERATING_POINTS
)
REQUIRED_INPUT_FILES = ("pilot_summary.json", "pilot_results.csv")
CORE_RESULT_FIELDS = (
    "cell",
    "updates_M",
    "operating_point",
    "temperature",
    "categorical_precision",
    "gpt2_generative_perplexity",
    "within_sequence_entropy",
    "geq",
    "ged",
)
SOURCE_FIELDS = (
    "cell",
    "updates_M",
    "operating_point",
    "temperature",
    "categorical_precision",
    "gpt2_generative_perplexity",
    "within_sequence_entropy",
)
PROTOCOL_IDENTITY_FIELDS = (
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
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
TABLE_LABEL_RE = re.compile(r"^[A-Za-z0-9:._-]+$")
DISPLAY_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .:+/_-]{0,39}$")


@dataclass(frozen=True)
class InputSpec:
    display_alias: str
    key: str
    directory: Path


@dataclass
class JudgeAnalysis:
    spec: InputSpec
    summary: dict[str, Any]
    rows: dict[tuple[int, str], dict[str, Any]]
    judge_model: str
    judge_provider: str
    system_fingerprint: str | None
    summary_sha256: str
    results_sha256: str
    judgments_sha256: str
    protocol_identity: dict[str, Any]


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: expected at least one data row")
    return rows


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected an object")
    return value


def require_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected an array")
    return value


def require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a non-empty string")
    return value.strip()


def finite_float(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}: expected a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{context}: expected a finite number")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{context}: expected >= {minimum}, found {converted}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{context}: expected <= {maximum}, found {converted}")
    return converted


def integer(value: Any, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: expected an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected an integer") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected an integer") from exc
    if not math.isfinite(numeric) or numeric != converted:
        raise ValueError(f"{context}: expected an integer")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{context}: expected >= {minimum}, found {converted}")
    return converted


def require_sha256(value: Any, context: str) -> str:
    text = require_nonempty_string(value, context)
    if not HASH_RE.fullmatch(text):
        raise ValueError(f"{context}: expected a lowercase SHA-256 digest")
    return text


def slugify_alias(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not key:
        raise ValueError(f"display alias {value!r} does not produce a usable key")
    return key


def parse_input_spec(value: str) -> InputSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--input must have the form DISPLAY_ALIAS=/path/to/analyzer-output"
        )
    alias, path_text = value.split("=", 1)
    alias = alias.strip()
    path_text = path_text.strip()
    if not alias or not path_text:
        raise argparse.ArgumentTypeError(
            "--input must have a non-empty alias and directory"
        )
    if not DISPLAY_ALIAS_RE.fullmatch(alias):
        raise argparse.ArgumentTypeError(
            "DISPLAY_ALIAS must be 1--40 characters using letters, digits, "
            "spaces, or .:+/_-"
        )
    return InputSpec(
        display_alias=alias,
        key=slugify_alias(alias),
        directory=Path(path_text),
    )


def close_float(left: Any, right: Any, context: str) -> None:
    left_value = finite_float(left, f"{context}.left")
    right_value = finite_float(right, f"{context}.right")
    if not math.isclose(left_value, right_value, rel_tol=2e-7, abs_tol=2e-7):
        raise ValueError(
            f"{context}: inconsistent values {left_value!r} and {right_value!r}"
        )


def validate_result_row(
    value: Mapping[str, Any],
    context: str,
) -> tuple[tuple[int, str], dict[str, Any]]:
    missing = [field for field in CORE_RESULT_FIELDS if field not in value]
    if missing:
        raise ValueError(f"{context}: missing required fields {missing}")
    updates = integer(value["updates_M"], f"{context}.updates_M")
    operating_point = require_nonempty_string(
        value["operating_point"], f"{context}.operating_point"
    )
    key = (updates, operating_point)
    if key not in EXPECTED_CELLS:
        raise ValueError(f"{context}: unexpected design cell {key}")
    expected_cell = f"M{updates}-{operating_point}"
    cell = require_nonempty_string(value["cell"], f"{context}.cell")
    if cell != expected_cell:
        raise ValueError(
            f"{context}.cell: expected {expected_cell!r}, found {cell!r}"
        )
    temperature = finite_float(value["temperature"], f"{context}.temperature")
    precision = require_nonempty_string(
        value["categorical_precision"], f"{context}.categorical_precision"
    ).lower()
    if operating_point == "reference":
        if not math.isclose(temperature, 1.0) or precision != "fp64":
            raise ValueError(
                f"{context}: reference must be temperature=1.0/fp64"
            )
    else:
        if not math.isclose(temperature, 0.9) or precision != "fp32":
            raise ValueError(
                f"{context}: selected must be temperature=0.9/fp32"
            )
    normalized = dict(value)
    normalized["cell"] = cell
    normalized["updates_M"] = updates
    normalized["operating_point"] = operating_point
    normalized["temperature"] = temperature
    normalized["categorical_precision"] = precision
    normalized["gpt2_generative_perplexity"] = finite_float(
        value["gpt2_generative_perplexity"],
        f"{context}.gpt2_generative_perplexity",
        minimum=0.0,
    )
    if normalized["gpt2_generative_perplexity"] <= 0:
        raise ValueError(f"{context}.gpt2_generative_perplexity: expected > 0")
    normalized["within_sequence_entropy"] = finite_float(
        value["within_sequence_entropy"],
        f"{context}.within_sequence_entropy",
    )
    normalized["geq"] = finite_float(
        value["geq"], f"{context}.geq", minimum=0.0, maximum=10.0
    )
    normalized["ged"] = finite_float(
        value["ged"], f"{context}.ged", minimum=0.0, maximum=10.0
    )
    calls = integer(value.get("judge_calls"), f"{context}.judge_calls", minimum=1)
    if calls != 32:
        raise ValueError(f"{context}.judge_calls: expected 32, found {calls}")
    normalized["judge_calls"] = calls
    return key, normalized


def validate_summary_csv_consistency(
    summary_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    csv_rows: Sequence[Mapping[str, str]],
    context: str,
) -> None:
    csv_by_key: dict[tuple[int, str], Mapping[str, str]] = {}
    for index, row in enumerate(csv_rows):
        row_context = f"{context}.pilot_results.csv[{index}]"
        key, _ = validate_result_row(row, row_context)
        if key in csv_by_key:
            raise ValueError(f"{row_context}: duplicate design cell {key}")
        csv_by_key[key] = row
    if set(csv_by_key) != set(EXPECTED_CELLS):
        raise ValueError(
            f"{context}.pilot_results.csv: expected exactly {EXPECTED_CELLS}, "
            f"found {tuple(sorted(csv_by_key))}"
        )
    for key in EXPECTED_CELLS:
        summary_row = summary_rows[key]
        csv_row = csv_by_key[key]
        for field in CORE_RESULT_FIELDS:
            field_context = f"{context}.{key}.{field}"
            if field in {
                "updates_M",
                "temperature",
                "gpt2_generative_perplexity",
                "within_sequence_entropy",
                "geq",
                "ged",
            }:
                close_float(summary_row[field], csv_row[field], field_context)
            elif str(summary_row[field]) != str(csv_row[field]):
                raise ValueError(
                    f"{field_context}: summary/CSV mismatch "
                    f"{summary_row[field]!r} != {csv_row[field]!r}"
                )


def validate_input_artifacts(
    summary: Mapping[str, Any],
    context: str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    artifacts = require_mapping(summary.get("input_artifacts"), f"{context}.input_artifacts")
    required = ("judgments", "groups", "generation_provenance", "private_label_map")
    normalized: dict[str, dict[str, Any]] = {}
    for role in required:
        descriptor = require_mapping(
            artifacts.get(role), f"{context}.input_artifacts.{role}"
        )
        normalized[role] = {
            "sha256": require_sha256(
                descriptor.get("sha256"),
                f"{context}.input_artifacts.{role}.sha256",
            ),
            "bytes": integer(
                descriptor.get("bytes"),
                f"{context}.input_artifacts.{role}.bytes",
                minimum=1,
            ),
        }
        if "records" in descriptor:
            normalized[role]["records"] = integer(
                descriptor["records"],
                f"{context}.input_artifacts.{role}.records",
                minimum=1,
            )
    for role in ("judgments", "groups"):
        if normalized[role].get("records") != 320:
            raise ValueError(
                f"{context}.input_artifacts.{role}.records: expected 320"
            )
    return normalized["judgments"]["sha256"], normalized


def validate_protocol_identity(
    diagnostics: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    identity = require_mapping(
        diagnostics.get("protocol_identity"),
        f"{context}.judge_diagnostics.protocol_identity",
    )
    missing = [field for field in PROTOCOL_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(
            f"{context}.judge_diagnostics.protocol_identity: "
            f"missing required fields {missing}"
        )
    normalized: dict[str, Any] = {
        "protocol_version": require_nonempty_string(
            identity["protocol_version"],
            f"{context}.protocol_identity.protocol_version",
        ),
        "protocol_sha256": require_sha256(
            identity["protocol_sha256"],
            f"{context}.protocol_identity.protocol_sha256",
        ),
        "target_profile_id": require_nonempty_string(
            identity["target_profile_id"],
            f"{context}.protocol_identity.target_profile_id",
        ),
        "target_profile_sha256": require_sha256(
            identity["target_profile_sha256"],
            f"{context}.protocol_identity.target_profile_sha256",
        ),
        "prompt_sha256": require_sha256(
            identity["prompt_sha256"],
            f"{context}.protocol_identity.prompt_sha256",
        ),
        "schema_sha256": require_sha256(
            identity["schema_sha256"],
            f"{context}.protocol_identity.schema_sha256",
        ),
    }
    for field in (
        "strict_json_schema",
        "local_validation_uses_unmodified_schema",
        "tools_enabled",
    ):
        value = identity[field]
        if not isinstance(value, bool):
            raise ValueError(
                f"{context}.protocol_identity.{field}: expected a boolean"
            )
        normalized[field] = value
    omitted = require_sequence(
        identity["provider_schema_omitted_keywords"],
        f"{context}.protocol_identity.provider_schema_omitted_keywords",
    )
    if any(not isinstance(value, str) or not value for value in omitted):
        raise ValueError(
            f"{context}.protocol_identity.provider_schema_omitted_keywords: "
            "expected non-empty strings"
        )
    if len(set(omitted)) != len(omitted):
        raise ValueError(
            f"{context}.protocol_identity.provider_schema_omitted_keywords: "
            "duplicate values"
        )
    normalized["provider_schema_omitted_keywords"] = list(omitted)
    return normalized


def validate_mdlm_provenance(summary: Mapping[str, Any], context: str) -> None:
    provenance = require_mapping(
        summary.get("generation_provenance"),
        f"{context}.generation_provenance",
    )
    if provenance.get("artifact") != "GroupEval MDLM-Small two-operating-point pilot":
        raise ValueError(f"{context}: unexpected generation-provenance artifact")
    design = require_mapping(
        provenance.get("design"), f"{context}.generation_provenance.design"
    )
    expected_design = {
        "methods": 10,
        "samples_per_method": 128,
        "total_samples": 1280,
        "group_size": 8,
    }
    for field, expected in expected_design.items():
        if integer(
            design.get(field),
            f"{context}.generation_provenance.design.{field}",
        ) != expected:
            raise ValueError(
                f"{context}.generation_provenance.design.{field}: "
                f"expected {expected}"
            )
    if design.get("noncanonical") not in (None, False):
        raise ValueError(f"{context}: MDLM bridge requires the canonical design")
    partition_seeds = require_sequence(
        design.get("partition_seeds"),
        f"{context}.generation_provenance.design.partition_seeds",
    )
    if tuple(
        integer(value, f"{context}.partition_seeds[{index}]")
        for index, value in enumerate(partition_seeds)
    ) != (271828, 314159):
        raise ValueError(f"{context}: unexpected frozen partition seeds")

    cells = require_sequence(
        provenance.get("cells"), f"{context}.generation_provenance.cells"
    )
    if len(cells) != len(EXPECTED_CELLS):
        raise ValueError(
            f"{context}.generation_provenance.cells: expected "
            f"{len(EXPECTED_CELLS)} cells"
        )
    for index, raw_cell in enumerate(cells):
        cell = require_mapping(
            raw_cell, f"{context}.generation_provenance.cells[{index}]"
        )
        if cell.get("model") != "public MDLM-Small":
            raise ValueError(
                f"{context}.generation_provenance.cells[{index}]: "
                "unexpected generator"
            )
        if cell.get("sampler") != "ancestral_cache":
            raise ValueError(
                f"{context}.generation_provenance.cells[{index}]: "
                "unexpected sampler"
            )
        if integer(
            cell.get("sample_count"),
            f"{context}.generation_provenance.cells[{index}].sample_count",
        ) != 128:
            raise ValueError(
                f"{context}.generation_provenance.cells[{index}]: "
                "expected 128 samples"
            )


def validate_within_budget_comparisons(
    summary: Mapping[str, Any],
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    context: str,
) -> None:
    comparisons = require_sequence(
        summary.get("within_budget_comparisons"),
        f"{context}.within_budget_comparisons",
    )
    by_budget: dict[int, Mapping[str, Any]] = {}
    for index, raw_comparison in enumerate(comparisons):
        comparison = require_mapping(
            raw_comparison, f"{context}.within_budget_comparisons[{index}]"
        )
        budget = integer(
            comparison.get("updates_M"),
            f"{context}.within_budget_comparisons[{index}].updates_M",
        )
        if budget not in BUDGETS or budget in by_budget:
            raise ValueError(
                f"{context}.within_budget_comparisons[{index}]: "
                f"unexpected or duplicate M={budget}"
            )
        if comparison.get("comparison") != "selected-minus-reference":
            raise ValueError(
                f"{context}.within_budget_comparisons[{index}]: "
                "expected selected-minus-reference"
            )
        for metric in ("geq", "ged"):
            delta = finite_float(
                comparison.get(f"delta_{metric}"),
                f"{context}.within_budget_comparisons[{index}].delta_{metric}",
            )
            expected_delta = (
                float(rows[(budget, "selected")][metric])
                - float(rows[(budget, "reference")][metric])
            )
            close_float(
                delta,
                expected_delta,
                f"{context}.within_budget_comparisons[{index}].delta_{metric}",
            )
            low = finite_float(
                comparison.get(f"delta_{metric}_ci_low"),
                f"{context}.within_budget_comparisons[{index}]."
                f"delta_{metric}_ci_low",
            )
            high = finite_float(
                comparison.get(f"delta_{metric}_ci_high"),
                f"{context}.within_budget_comparisons[{index}]."
                f"delta_{metric}_ci_high",
            )
            finite_float(
                comparison.get(f"delta_{metric}_lcb_95"),
                f"{context}.within_budget_comparisons[{index}]."
                f"delta_{metric}_lcb_95",
            )
            if low > high:
                raise ValueError(
                    f"{context}.within_budget_comparisons[{index}]."
                    f"delta_{metric}: CI endpoints are reversed"
                )
        if not isinstance(
            comparison.get("quality_superior_at_95pct_one_sided"), bool
        ):
            raise ValueError(
                f"{context}.within_budget_comparisons[{index}]: "
                "quality-superiority decision must be boolean"
            )
        by_budget[budget] = comparison
    if set(by_budget) != set(BUDGETS):
        raise ValueError(
            f"{context}.within_budget_comparisons: expected exactly M={BUDGETS}"
        )


def load_analysis(spec: InputSpec) -> JudgeAnalysis:
    directory = spec.directory.expanduser().resolve()
    context = f"input[{spec.display_alias}]"
    if not directory.is_dir():
        raise FileNotFoundError(f"{context}: directory not found: {directory}")
    for filename in REQUIRED_INPUT_FILES:
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"{context}: missing {filename}")
    summary_path = directory / "pilot_summary.json"
    results_path = directory / "pilot_results.csv"
    summary = load_json(summary_path)
    if summary.get("artifact") != "Reviewer-safe MDLM-Small GroupEval pilot analysis":
        raise ValueError(
            f"{context}.artifact: not an analyze_mdlm_pilot.py reviewer-safe output"
        )

    analysis = require_mapping(summary.get("analysis"), f"{context}.analysis")
    if analysis.get("study") not in (None, "mdlm-small"):
        raise ValueError(f"{context}: analyzer study is not mdlm-small")
    if analysis.get("generator") not in (None, "public MDLM-Small"):
        raise ValueError(f"{context}: analyzer generator is not public MDLM-Small")
    if analysis.get("gppl_evaluator") not in (None, "GPT-2"):
        raise ValueError(f"{context}: analyzer gPPL evaluator is not GPT-2")
    completeness = require_mapping(
        analysis.get("judgment_completeness"),
        f"{context}.analysis.judgment_completeness",
    )
    expected_groups = integer(
        completeness.get("expected_groups"),
        f"{context}.analysis.judgment_completeness.expected_groups",
    )
    valid_judgments = integer(
        completeness.get("valid_judgments"),
        f"{context}.analysis.judgment_completeness.valid_judgments",
    )
    if completeness.get("complete") is not True:
        raise ValueError(f"{context}: analyzer did not mark judgments complete")
    if expected_groups != 320 or valid_judgments != 320:
        raise ValueError(
            f"{context}: expected a complete 320-call MDLM pilot, found "
            f"{expected_groups}/{valid_judgments}"
        )
    if analysis.get("bootstrap_method") != "crossed_bayesian_bootstrap":
        raise ValueError(f"{context}: unexpected bootstrap method")
    if analysis.get("bootstrap_pairing_mode") != "partitions":
        raise ValueError(f"{context}: unexpected bootstrap pairing mode")
    if integer(
        analysis.get("bootstrap_replicates"),
        f"{context}.analysis.bootstrap_replicates",
    ) != 10_000:
        raise ValueError(f"{context}: expected 10,000 bootstrap replicates")

    diagnostics = require_mapping(
        summary.get("judge_diagnostics"), f"{context}.judge_diagnostics"
    )
    judge_model = require_nonempty_string(
        diagnostics.get("judge_model"), f"{context}.judge_diagnostics.judge_model"
    )
    judge_provider = require_nonempty_string(
        diagnostics.get("judge_provider"),
        f"{context}.judge_diagnostics.judge_provider",
    )
    system_fingerprint_raw = diagnostics.get("system_fingerprint")
    system_fingerprint = (
        None
        if system_fingerprint_raw is None
        else require_nonempty_string(
            system_fingerprint_raw,
            f"{context}.judge_diagnostics.system_fingerprint",
        )
    )
    protocol_identity = validate_protocol_identity(diagnostics, context)
    if integer(
        diagnostics.get("judge_calls"),
        f"{context}.judge_diagnostics.judge_calls",
    ) != 320:
        raise ValueError(f"{context}: judge diagnostics must contain 320 calls")

    summary_results = require_sequence(summary.get("results"), f"{context}.results")
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for index, value in enumerate(summary_results):
        mapping = require_mapping(value, f"{context}.results[{index}]")
        key, normalized = validate_result_row(
            mapping, f"{context}.results[{index}]"
        )
        if key in rows:
            raise ValueError(f"{context}.results[{index}]: duplicate cell {key}")
        rows[key] = normalized
    if set(rows) != set(EXPECTED_CELLS):
        raise ValueError(
            f"{context}.results: expected exactly {EXPECTED_CELLS}, "
            f"found {tuple(sorted(rows))}"
        )
    validate_summary_csv_consistency(
        rows, load_csv(results_path), context
    )
    validate_mdlm_provenance(summary, context)
    validate_within_budget_comparisons(summary, rows, context)
    judgments_sha256, _ = validate_input_artifacts(summary, context)

    return JudgeAnalysis(
        spec=InputSpec(spec.display_alias, spec.key, directory),
        summary=summary,
        rows=rows,
        judge_model=judge_model,
        judge_provider=judge_provider,
        system_fingerprint=system_fingerprint,
        summary_sha256=sha256_file(summary_path),
        results_sha256=sha256_file(results_path),
        judgments_sha256=judgments_sha256,
        protocol_identity=protocol_identity,
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def source_signature(analysis: JudgeAnalysis) -> dict[str, Any]:
    artifacts = require_mapping(
        analysis.summary["input_artifacts"],
        f"input[{analysis.spec.display_alias}].input_artifacts",
    )
    input_hashes = {
        role: {
            "sha256": require_sha256(
                require_mapping(artifacts[role], f"input_artifacts.{role}").get(
                    "sha256"
                ),
                f"input_artifacts.{role}.sha256",
            ),
            "bytes": integer(
                require_mapping(artifacts[role], f"input_artifacts.{role}").get(
                    "bytes"
                ),
                f"input_artifacts.{role}.bytes",
                minimum=1,
            ),
        }
        for role in ("groups", "generation_provenance", "private_label_map")
    }
    source_rows = [
        {
            field: analysis.rows[key][field]
            for field in SOURCE_FIELDS
        }
        for key in EXPECTED_CELLS
    ]
    return {
        "input_artifacts": input_hashes,
        "generation_provenance": analysis.summary.get("generation_provenance"),
        "source_rows": source_rows,
    }


def validate_cross_judge_alignment(analyses: Sequence[JudgeAnalysis]) -> None:
    if len(analyses) != 3:
        raise ValueError(f"exactly three judge analyses are required, found {len(analyses)}")
    aliases = [analysis.spec.display_alias for analysis in analyses]
    keys = [analysis.spec.key for analysis in analyses]
    directories = [analysis.spec.directory for analysis in analyses]
    judge_models = [analysis.judge_model for analysis in analyses]
    judgment_hashes = [analysis.judgments_sha256 for analysis in analyses]
    for name, values in (
        ("display aliases", aliases),
        ("alias keys", keys),
        ("input directories", directories),
        ("judge model aliases", judge_models),
        ("judgment artifacts", judgment_hashes),
    ):
        if len(set(values)) != 3:
            raise ValueError(f"the three {name} must be distinct: {values}")

    baseline = canonical_json(source_signature(analyses[0]))
    for analysis in analyses[1:]:
        if canonical_json(source_signature(analysis)) != baseline:
            raise ValueError(
                f"input[{analysis.spec.display_alias}]: frozen groups, generation "
                "provenance, cells, or source gPPL/entropy do not exactly match "
                f"input[{analyses[0].spec.display_alias}]"
            )
    baseline_protocol = canonical_json(analyses[0].protocol_identity)
    for analysis in analyses[1:]:
        if canonical_json(analysis.protocol_identity) != baseline_protocol:
            raise ValueError(
                f"input[{analysis.spec.display_alias}]: prompt, schema, protocol, "
                "target profile, or validation configuration does not exactly "
                f"match input[{analyses[0].spec.display_alias}]"
            )


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for index, _ in indexed[start:stop]:
            ranks[index] = average_rank
        start = stop
    return ranks


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation vectors must have the same length >= 2")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    value = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_centered, right_centered, strict=True
        )
    ) / denominator
    return max(-1.0, min(1.0, value))


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def correlation_record(
    left: JudgeAnalysis,
    right: JudgeAnalysis,
    metric: str,
) -> dict[str, Any]:
    left_values = [float(left.rows[key][metric]) for key in EXPECTED_CELLS]
    right_values = [float(right.rows[key][metric]) for key in EXPECTED_CELLS]
    rho = spearman(left_values, right_values)
    return {
        "rho": rho,
        "defined": rho is not None,
        "reason": (
            None
            if rho is not None
            else "undefined because at least one rank vector is constant"
        ),
        "cells": len(EXPECTED_CELLS),
        "rank_ties": "average ranks",
    }


def sign_summary(analysis: JudgeAnalysis, metric: str) -> dict[str, Any]:
    selected_gt: list[int] = []
    selected_eq: list[int] = []
    selected_lt: list[int] = []
    for budget in BUDGETS:
        reference = float(analysis.rows[(budget, "reference")][metric])
        selected = float(analysis.rows[(budget, "selected")][metric])
        if math.isclose(selected, reference, rel_tol=0.0, abs_tol=1e-12):
            selected_eq.append(budget)
        elif selected > reference:
            selected_gt.append(budget)
        else:
            selected_lt.append(budget)
    return {
        "selected_gt_reference_count": len(selected_gt),
        "selected_eq_reference_count": len(selected_eq),
        "selected_lt_reference_count": len(selected_lt),
        "selected_gt_reference_M": selected_gt,
        "selected_eq_reference_M": selected_eq,
        "selected_lt_reference_M": selected_lt,
        "comparisons": len(BUDGETS),
        "basis": (
            "Point-estimate directions only; ties use absolute tolerance 1e-12. "
            "Counts are not significance tests."
        ),
    }


def bridge_rows(analyses: Sequence[JudgeAnalysis]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in EXPECTED_CELLS:
        common = analyses[0].rows[key]
        row: dict[str, Any] = {
            field: common[field]
            for field in SOURCE_FIELDS
        }
        for analysis in analyses:
            row[f"{analysis.spec.key}_geq"] = analysis.rows[key]["geq"]
            row[f"{analysis.spec.key}_ged"] = analysis.rows[key]["ged"]
        rows.append(row)
    return rows


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered: dict[str, Any] = {}
            for field in fieldnames:
                value = row[field]
                rendered[field] = (
                    f"{value:.8g}" if isinstance(value, float) else value
                )
            writer.writerow(rendered)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def make_latex_table(
    analyses: Sequence[JudgeAnalysis],
    rows: Sequence[Mapping[str, Any]],
    table_label: str,
) -> str:
    by_key = {
        (int(row["updates_M"]), str(row["operating_point"])): row for row in rows
    }
    headers = [latex_escape(analysis.spec.display_alias) for analysis in analyses]
    model_mapping = "; ".join(
        f"{latex_escape(analysis.spec.display_alias)} is "
        rf"\texttt{{{latex_escape(analysis.judge_model)}}}"
        for analysis in analyses
    )
    lines = [
        "% Auto-generated by experiment/analyze_judge_bridge.py",
        r"\begin{table}[!ht]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.25pt}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{rcccccccc}",
        r"    \toprule",
        (
            r"    & & & "
            + " & ".join(
                rf"\multicolumn{{2}}{{c}}{{{header}}}" for header in headers
            )
            + r" \\"
        ),
        r"    \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        (
            r"    \(M\) & gPPL (ref./sel.) & \(H\) (ref./sel.)"
            r" & GEQ\(\uparrow\) & GED\(\uparrow\)"
            r" & GEQ\(\uparrow\) & GED\(\uparrow\)"
            r" & GEQ\(\uparrow\) & GED\(\uparrow\) \\"
        ),
        r"    \midrule",
    ]
    for budget in BUDGETS:
        reference = by_key[(budget, "reference")]
        selected = by_key[(budget, "selected")]
        cells = [
            str(budget),
            (
                f"{float(reference['gpt2_generative_perplexity']):.1f} / "
                f"{float(selected['gpt2_generative_perplexity']):.1f}"
            ),
            (
                f"{float(reference['within_sequence_entropy']):.2f} / "
                f"{float(selected['within_sequence_entropy']):.2f}"
            ),
        ]
        for analysis in analyses:
            cells.extend(
                [
                    (
                        f"{float(reference[f'{analysis.spec.key}_geq']):.2f} / "
                        f"{float(selected[f'{analysis.spec.key}_geq']):.2f}"
                    ),
                    (
                        f"{float(reference[f'{analysis.spec.key}_ged']):.2f} / "
                        f"{float(selected[f'{analysis.spec.key}_ged']):.2f}"
                    ),
                ]
            )
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            (
                r"  \caption{\textbf{Exploratory GroupEval judge comparison on public "
                r"MDLM-Small.} Every entry is reference / selected at fixed \(M\); "
                r"the operating points are \((\tau{=}1,\mathrm{fp64})\) and "
                r"\((\tau{=}.9,\mathrm{fp32})\), respectively. "
                + model_mapping
                + r". All three judges receive the identical frozen blinded groups. "
                r"Higher GED denotes more valid, balanced semantic--discourse modes "
                r"under the named judge. GEQ/GED values are judge-specific: they are "
                r"neither pooled nor "
                r"treated as interchangeable scales. gPPL and \(H\) are the existing "
                r"128-sample measurements; per-judge intervals, matched differences, "
                r"and diagnostics remain in the released aggregates.}"
            ),
            rf"  \label{{{table_label}}}",
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


def analyze(
    specs: Sequence[InputSpec],
    output_dir: Path,
    table_label: str,
) -> dict[str, Any]:
    analyses = [load_analysis(spec) for spec in specs]
    validate_cross_judge_alignment(analyses)
    rows = bridge_rows(analyses)

    source = source_signature(analyses[0])
    common_artifacts = source["input_artifacts"]
    pairwise: list[dict[str, Any]] = []
    for left_index in range(len(analyses)):
        for right_index in range(left_index + 1, len(analyses)):
            left = analyses[left_index]
            right = analyses[right_index]
            pairwise.append(
                {
                    "judge_a": left.spec.display_alias,
                    "judge_b": right.spec.display_alias,
                    "geq": correlation_record(left, right, "geq"),
                    "ged": correlation_record(left, right, "ged"),
                }
            )

    judges = [
        {
            "display_alias": analysis.spec.display_alias,
            "key": analysis.spec.key,
            "judge_model_alias": analysis.judge_model,
            "judge_provider": analysis.judge_provider,
            "system_fingerprint": analysis.system_fingerprint,
            "judge_calls": integer(
                require_mapping(
                    analysis.summary["judge_diagnostics"], "judge_diagnostics"
                )["judge_calls"],
                "judge_diagnostics.judge_calls",
            ),
            "source_artifacts": {
                "pilot_summary_sha256": analysis.summary_sha256,
                "pilot_results_csv_sha256": analysis.results_sha256,
                "private_judgments_sha256": analysis.judgments_sha256,
            },
            "selected_vs_reference_signs": {
                "geq": sign_summary(analysis, "geq"),
                "ged": sign_summary(analysis, "ged"),
            },
        }
        for analysis in analyses
    ]
    summary: dict[str, Any] = {
        "artifact": "Reviewer-safe MDLM-Small three-judge GroupEval comparison",
        "analysis": {
            "design_cells": len(EXPECTED_CELLS),
            "budgets_M": list(BUDGETS),
            "operating_points": list(OPERATING_POINTS),
            "judge_count": len(analyses),
            "source_alignment": "exact",
            "score_handling": "judge-separated; no pooling, averaging, or scale calibration",
            "sensitivity_statistics": {
                "pairwise_spearman_across_cells": pairwise,
                "selected_gt_reference_sign_counts": {
                    judge["display_alias"]: judge["selected_vs_reference_signs"]
                    for judge in judges
                },
                "interpretation": (
                    "These are descriptive sensitivities to judge choice. They do not "
                    "establish human calibration, test sign-count significance, or make "
                    "service-alias scales interchangeable."
                ),
            },
        },
        "common_judge_protocol_identity": analyses[0].protocol_identity,
        "common_source_artifacts": common_artifacts,
        "judges_in_table_order": judges,
        "results": rows,
        "interpretation_notes": [
            "Every GEQ/GED value remains attached to its named judge alias.",
            "No cross-judge mean, pooled score, affine alignment, or consensus score is computed.",
            (
                "Selected-versus-reference sign counts describe point-estimate "
                "directions within judge and fixed M; they are not significance tests."
            ),
            (
                "Pairwise Spearman correlations summarize rank correspondence "
                "across ten confounded cells only."
            ),
            (
                "Higher GED denotes more valid, balanced local semantic-discourse "
                "modes under the named judge; no human-calibrated sufficiency "
                "threshold or non-inferiority margin is available."
            ),
        ],
    }
    assert_reviewer_safe(summary)

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a fresh path"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        fieldnames = list(SOURCE_FIELDS)
        for analysis in analyses:
            fieldnames.extend(
                [f"{analysis.spec.key}_geq", f"{analysis.spec.key}_ged"]
            )
        write_csv(staging / "judge_bridge.csv", rows, fieldnames)
        (staging / "judge_bridge_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        latex = make_latex_table(analyses, rows, table_label)
        (staging / "judge_bridge_table.tex").write_text(latex, encoding="utf-8")
        for path in staging.iterdir():
            if path.is_file():
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
        "judges": [analysis.judge_model for analysis in analyses],
        "cells": len(rows),
        "pooled_scores_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input_spec,
        metavar="DISPLAY_ALIAS=ANALYZER_OUTPUT_DIR",
        help=(
            "Reviewer-safe analyze_mdlm_pilot.py output. Repeat exactly three "
            "times in the desired table order."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--table-label", default="tab:groupeval-pilot")
    args = parser.parse_args()
    if len(args.input) != 3:
        parser.error(f"--input must be supplied exactly three times, found {len(args.input)}")
    if not TABLE_LABEL_RE.fullmatch(args.table_label):
        parser.error("--table-label contains unsupported characters")
    result = analyze(args.input, args.output_dir, args.table_label)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
