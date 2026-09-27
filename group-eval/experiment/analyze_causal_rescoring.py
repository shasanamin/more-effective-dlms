#!/usr/bin/env python3
"""Validate and compare the four common causal-LM rescoring summaries.

Inputs are the reviewer-safe summaries from ``rescore_causal_lm.py`` for
GPT-2-Large and Qwen3-8B-Base under full-stream and first-document policies.
The optional GroupEval CSV must be the 20-cell ``cell_results.csv`` emitted by
``analyze_cross_model_pilot.py``.  This analyzer never reads raw generations
or private per-output likelihoods.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ANALYSIS_VERSION = "1.0.0"
PROTOCOL_VERSION = "common-causal-gppl-v2"
EXPECTED_ARTIFACT = "reviewer-safe aggregate causal-LM gPPL rescoring"
EXPECTED_UPDATES = (4, 16, 64, 256, 1024)
EXPECTED_OUTPUTS_PER_CELL = 40
PRIMARY_MIN_UPDATES = 16

METRICS = (
    "token_micro_perplexity",
    "output_macro_geometric_perplexity",
    "corpus_bits_per_utf8_byte",
)
GPPL_METRICS = (
    "token_micro_perplexity",
    "output_macro_geometric_perplexity",
)
ADDITIVE_FIELDS = (
    "output_count",
    "scorable_output_count",
    "source_segment_count",
    "included_segment_count",
    "scorable_segment_count",
    "source_boundary_count",
    "included_boundary_count",
    "scored_boundary_count",
    "truncated_segment_count",
    "leading_sentinel_removed_count",
    "raw_character_count",
    "raw_utf8_bytes",
    "source_content_character_count",
    "source_content_utf8_bytes",
    "included_content_character_count",
    "included_content_utf8_bytes",
    "scored_content_character_count",
    "scored_content_utf8_bytes",
    "scored_evaluator_token_count",
)
SOURCE_IDENTITY_FIELDS = (
    "expected_count",
    "model_family",
    "operating_point",
    "precision",
    "steps",
    "temperature",
    "output_count",
    "source_segment_count",
    "source_boundary_count",
    "leading_sentinel_removed_count",
    "raw_character_count",
    "raw_utf8_bytes",
    "source_content_character_count",
    "source_content_utf8_bytes",
    "source_sha256",
)


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    evaluator: str
    evaluator_short: str
    model_identifier: str
    revision: str
    span_policy: str


SERIES_SPECS = (
    SeriesSpec(
        series_id="gpt2_large_full_stream",
        evaluator="GPT-2-Large",
        evaluator_short="GPT-2-L",
        model_identifier="gpt2-large",
        revision="32b71b12589c2f8d625668d2335a01cac3249519",
        span_policy="full_stream_segmented",
    ),
    SeriesSpec(
        series_id="gpt2_large_first_document",
        evaluator="GPT-2-Large",
        evaluator_short="GPT-2-L",
        model_identifier="gpt2-large",
        revision="32b71b12589c2f8d625668d2335a01cac3249519",
        span_policy="first_document",
    ),
    SeriesSpec(
        series_id="qwen3_8b_full_stream",
        evaluator="Qwen3-8B-Base",
        evaluator_short="Qwen3-8B",
        model_identifier="Qwen/Qwen3-8B-Base",
        revision="b53affe9dcff71ea989116a20abf368cbe4f2bdd",
        span_policy="full_stream_segmented",
    ),
    SeriesSpec(
        series_id="qwen3_8b_first_document",
        evaluator="Qwen3-8B-Base",
        evaluator_short="Qwen3-8B",
        model_identifier="Qwen/Qwen3-8B-Base",
        revision="b53affe9dcff71ea989116a20abf368cbe4f2bdd",
        span_policy="first_document",
    ),
)
SPEC_BY_ID = {spec.series_id: spec for spec in SERIES_SPECS}


def expected_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for prefix, model in (("mdlm", "MDLM-Small"), ("didi", "DiDi-Instruct")):
        for short, operating_point, temperature, precision in (
            ("ref", "reference", 1.0, "fp64"),
            ("sel", "selected", 0.9, "fp32"),
        ):
            for updates in EXPECTED_UPDATES:
                cells.append(
                    {
                        "cell_id": f"{prefix}_{short}_m{updates}",
                        "model_family": model,
                        "operating_point": operating_point,
                        "steps": updates,
                        "temperature": temperature,
                        "precision": precision,
                    }
                )
    return cells


CELL_SPECS = expected_cells()
CELL_SPEC_BY_ID = {cell["cell_id"]: cell for cell in CELL_SPECS}


@dataclass
class LoadedSeries:
    spec: SeriesSpec
    filename: str
    file_sha256: str
    summary: dict[str, Any]
    cells: dict[str, dict[str, Any]]


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


def finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context}: expected an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected an integer") from exc
    if isinstance(value, float) and value != result:
        raise ValueError(f"{context}: expected an integer")
    return result


def require_equal(context: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{context}: expected {expected!r}, observed {observed!r}")


def require_sha256(value: Any, context: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context}: expected a SHA-256 digest")
    return text


def positive_float(value: Any, context: str) -> float:
    result = finite_float(value, context)
    if result <= 0.0:
        raise ValueError(f"{context}: expected a positive number")
    return result


def close_enough(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=2e-9, abs_tol=2e-9)


def validate_cell(cell: dict[str, Any], expected: Mapping[str, Any], context: str) -> None:
    for field in ("cell_id", "model_family", "operating_point", "precision"):
        require_equal(f"{context}.{field}", str(cell.get(field)), str(expected[field]))
    require_equal(f"{context}.steps", integer(cell.get("steps"), context), expected["steps"])
    temperature = finite_float(cell.get("temperature"), f"{context}.temperature")
    if not close_enough(temperature, float(expected["temperature"])):
        raise ValueError(f"{context}.temperature: unexpected value {temperature}")
    require_equal(
        f"{context}.expected_count",
        integer(cell.get("expected_count"), context),
        EXPECTED_OUTPUTS_PER_CELL,
    )
    require_equal(
        f"{context}.output_count",
        integer(cell.get("output_count"), context),
        EXPECTED_OUTPUTS_PER_CELL,
    )
    require_sha256(cell.get("source_sha256"), f"{context}.source_sha256")

    counts = {field: integer(cell.get(field), f"{context}.{field}") for field in ADDITIVE_FIELDS}
    if any(value < 0 for value in counts.values()):
        raise ValueError(f"{context}: diagnostic counts must be nonnegative")
    inequalities = (
        ("scorable_output_count", "output_count"),
        ("included_segment_count", "source_segment_count"),
        ("scorable_segment_count", "included_segment_count"),
        ("included_boundary_count", "source_boundary_count"),
        ("scored_boundary_count", "included_boundary_count"),
        ("truncated_segment_count", "included_segment_count"),
        ("included_content_character_count", "source_content_character_count"),
        ("included_content_utf8_bytes", "source_content_utf8_bytes"),
        ("scored_content_character_count", "included_content_character_count"),
        ("scored_content_utf8_bytes", "included_content_utf8_bytes"),
    )
    for smaller, larger in inequalities:
        if counts[smaller] > counts[larger]:
            raise ValueError(f"{context}: {smaller} exceeds {larger}")

    token_count = counts["scored_evaluator_token_count"]
    nll_sum = finite_float(cell.get("nll_sum"), f"{context}.nll_sum")
    content_nll = finite_float(
        cell.get("content_nll_sum_for_bpb"), f"{context}.content_nll_sum_for_bpb"
    )
    if nll_sum < 0.0 or content_nll < 0.0:
        raise ValueError(f"{context}: NLL sums must be nonnegative")
    token_mean = finite_float(cell.get("token_micro_mean_nll"), f"{context}.token_micro_mean_nll")
    token_ppl = positive_float(cell.get("token_micro_perplexity"), f"{context}.token_micro_perplexity")
    if not close_enough(token_mean, nll_sum / token_count):
        raise ValueError(f"{context}: token-micro mean NLL is inconsistent")
    if not close_enough(token_ppl, math.exp(token_mean)):
        raise ValueError(f"{context}: token-micro perplexity is inconsistent")

    macro_mean = finite_float(cell.get("output_macro_mean_nll"), f"{context}.output_macro_mean_nll")
    macro_ppl = positive_float(
        cell.get("output_macro_geometric_perplexity"),
        f"{context}.output_macro_geometric_perplexity",
    )
    if not close_enough(macro_ppl, math.exp(macro_mean)):
        raise ValueError(f"{context}: output-macro perplexity is inconsistent")
    scored_bytes = counts["scored_content_utf8_bytes"]
    bpb = positive_float(
        cell.get("corpus_bits_per_utf8_byte"), f"{context}.corpus_bits_per_utf8_byte"
    )
    expected_bpb = content_nll / (math.log(2.0) * scored_bytes)
    if not close_enough(bpb, expected_bpb):
        raise ValueError(f"{context}: corpus bits per byte is inconsistent")


def validate_summary(path: Path, spec: SeriesSpec) -> LoadedSeries:
    summary = load_json(path)
    context = f"{path.name}/{spec.series_id}"
    require_equal(f"{context}.artifact", summary.get("artifact"), EXPECTED_ARTIFACT)
    require_equal(f"{context}.protocol_version", summary.get("protocol_version"), PROTOCOL_VERSION)
    require_equal(f"{context}.total_cells", integer(summary.get("total_cells"), context), len(CELL_SPECS))
    require_sha256(summary.get("input_manifest_sha256"), f"{context}.input_manifest_sha256")
    require_sha256(
        summary.get("private_per_output_artifact_sha256"),
        f"{context}.private_per_output_artifact_sha256",
    )

    model = summary.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{context}.model: expected an object")
    require_equal(f"{context}.model.identifier", model.get("identifier"), spec.model_identifier)
    require_equal(
        f"{context}.model.requested_revision",
        str(model.get("requested_revision", "")).lower(),
        spec.revision,
    )
    require_equal(
        f"{context}.model.resolved_model_revision",
        str(model.get("resolved_model_revision", "")).lower(),
        spec.revision,
    )
    scoring = summary.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError(f"{context}.scoring: expected an object")
    require_equal(f"{context}.scoring.span_policy", scoring.get("span_policy"), spec.span_policy)
    require_equal(
        f"{context}.scoring.max_length",
        integer(scoring.get("max_evaluator_tokens_per_segment_including_primer"), context),
        1024,
    )
    require_equal(f"{context}.scoring.bootstrap_unit", scoring.get("bootstrap_unit"), "output")
    require_equal(
        f"{context}.scoring.bootstrap_replicates",
        integer(scoring.get("bootstrap_replicates"), context),
        10_000,
    )
    require_equal(f"{context}.scoring.add_special_tokens", scoring.get("add_special_tokens"), False)

    raw_cells = summary.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError(f"{context}.cells: expected an array")
    cells: dict[str, dict[str, Any]] = {}
    for index, raw_cell in enumerate(raw_cells):
        if not isinstance(raw_cell, dict):
            raise ValueError(f"{context}.cells[{index}]: expected an object")
        cell_id = str(raw_cell.get("cell_id", ""))
        if not cell_id or cell_id in cells:
            raise ValueError(f"{context}.cells[{index}]: missing or duplicate cell_id")
        if cell_id not in CELL_SPEC_BY_ID:
            raise ValueError(f"{context}: unexpected cell_id {cell_id!r}")
        validate_cell(raw_cell, CELL_SPEC_BY_ID[cell_id], f"{context}.{cell_id}")
        cells[cell_id] = raw_cell
    require_equal(f"{context}.cell identities", set(cells), set(CELL_SPEC_BY_ID))

    overall = summary.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"{context}.overall: expected an object")
    for field in ADDITIVE_FIELDS:
        observed = integer(overall.get(field), f"{context}.overall.{field}")
        expected_sum = sum(integer(cell[field], field) for cell in cells.values())
        require_equal(f"{context}.overall.{field}", observed, expected_sum)
    return LoadedSeries(
        spec=spec,
        filename=path.name,
        file_sha256=sha256_file(path),
        summary=summary,
        cells=cells,
    )


def validate_common_inputs(series: Mapping[str, LoadedSeries]) -> None:
    require_equal("series roles", set(series), set(SPEC_BY_ID))
    manifest_hashes = {
        str(loaded.summary["input_manifest_sha256"]) for loaded in series.values()
    }
    if len(manifest_hashes) != 1:
        raise ValueError("the four summaries do not share one input manifest hash")

    baseline = series[SERIES_SPECS[0].series_id]
    baseline_scoring = dict(baseline.summary["scoring"])
    baseline_scoring.pop("span_policy", None)
    for loaded in series.values():
        scoring = dict(loaded.summary["scoring"])
        scoring.pop("span_policy", None)
        require_equal(f"{loaded.spec.series_id}: common scoring contract", scoring, baseline_scoring)

    for cell_id in CELL_SPEC_BY_ID:
        reference = baseline.cells[cell_id]
        for loaded in series.values():
            cell = loaded.cells[cell_id]
            for field in SOURCE_IDENTITY_FIELDS:
                require_equal(
                    f"{loaded.spec.series_id}/{cell_id}: common {field}",
                    cell.get(field),
                    reference.get(field),
                )
            policy = loaded.spec.span_policy
            if policy == "full_stream_segmented":
                for included, source in (
                    ("included_segment_count", "source_segment_count"),
                    ("included_boundary_count", "source_boundary_count"),
                    ("included_content_character_count", "source_content_character_count"),
                    ("included_content_utf8_bytes", "source_content_utf8_bytes"),
                ):
                    require_equal(
                        f"{loaded.spec.series_id}/{cell_id}: full-stream {included}",
                        cell[included],
                        cell[source],
                    )
            else:
                require_equal(
                    f"{loaded.spec.series_id}/{cell_id}: first-document segment count",
                    integer(cell["included_segment_count"], cell_id),
                    EXPECTED_OUTPUTS_PER_CELL,
                )


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("ratio denominator must be positive")
    return numerator / denominator


def coverage(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def flatten_interval(row: dict[str, Any], key: str) -> None:
    value = row.pop(key, None)
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key}: expected a two-element interval")
    row[f"{key}_low"] = finite_float(value[0], f"{key}[0]")
    row[f"{key}_high"] = finite_float(value[1], f"{key}[1]")


def make_cell_rows(series: Mapping[str, LoadedSeries]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SERIES_SPECS:
        loaded = series[spec.series_id]
        for expected in CELL_SPECS:
            cell = dict(loaded.cells[expected["cell_id"]])
            flatten_interval(cell, "output_macro_mean_nll_bootstrap_95")
            flatten_interval(cell, "output_macro_geometric_perplexity_bootstrap_95")
            cell.update(
                {
                    "series_id": spec.series_id,
                    "evaluator": spec.evaluator,
                    "evaluator_model_identifier": spec.model_identifier,
                    "evaluator_revision": spec.revision,
                    "span_policy": spec.span_policy,
                    "primary_included": int(cell["steps"]) >= PRIMARY_MIN_UPDATES,
                    "policy_content_byte_coverage": coverage(
                        cell["included_content_utf8_bytes"], cell["source_content_utf8_bytes"]
                    ),
                    "evaluator_content_byte_coverage": coverage(
                        cell["scored_content_utf8_bytes"], cell["included_content_utf8_bytes"]
                    ),
                    "policy_boundary_coverage": coverage(
                        cell["included_boundary_count"], cell["source_boundary_count"]
                    ),
                    "evaluator_boundary_coverage": coverage(
                        cell["scored_boundary_count"], cell["included_boundary_count"]
                    ),
                    "scorable_output_fraction": coverage(
                        cell["scorable_output_count"], cell["output_count"]
                    ),
                }
            )
            ordered = {
                "series_id": cell.pop("series_id"),
                "evaluator": cell.pop("evaluator"),
                "evaluator_model_identifier": cell.pop("evaluator_model_identifier"),
                "evaluator_revision": cell.pop("evaluator_revision"),
                "span_policy": cell.pop("span_policy"),
                **cell,
            }
            rows.append(ordered)
    return rows


def metric_sign(mdlm: float, didi: float) -> tuple[str, str]:
    if math.isclose(mdlm, didi, rel_tol=1e-12, abs_tol=1e-12):
        return "tie", "tie"
    if mdlm < didi:
        return "MDLM-lower", "MDLM-Small"
    return "MDLM-higher", "DiDi-Instruct"


def make_matched_rows(series: Mapping[str, LoadedSeries]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SERIES_SPECS:
        loaded = series[spec.series_id]
        for operating_point, short in (("reference", "ref"), ("selected", "sel")):
            for updates in EXPECTED_UPDATES:
                mdlm = loaded.cells[f"mdlm_{short}_m{updates}"]
                didi = loaded.cells[f"didi_{short}_m{updates}"]
                row: dict[str, Any] = {
                    "series_id": spec.series_id,
                    "evaluator": spec.evaluator,
                    "span_policy": spec.span_policy,
                    "operating_point": operating_point,
                    "updates_M": updates,
                    "primary_included": updates >= PRIMARY_MIN_UPDATES,
                    "contrast": "MDLM-Small_over_DiDi-Instruct",
                }
                for metric in METRICS:
                    mdlm_value = positive_float(mdlm[metric], f"{mdlm['cell_id']}.{metric}")
                    didi_value = positive_float(didi[metric], f"{didi['cell_id']}.{metric}")
                    value_ratio = ratio(mdlm_value, didi_value)
                    sign, lower_model = metric_sign(mdlm_value, didi_value)
                    row.update(
                        {
                            f"mdlm_{metric}": mdlm_value,
                            f"didi_{metric}": didi_value,
                            f"mdlm_over_didi_{metric}_ratio": value_ratio,
                            f"log_mdlm_over_didi_{metric}_ratio": math.log(value_ratio),
                            f"{metric}_sign": sign,
                            f"{metric}_lower_model": lower_model,
                        }
                    )
                rows.append(row)
    return rows


def make_sign_summaries(matched_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in SERIES_SPECS:
        rows = [
            row
            for row in matched_rows
            if row["series_id"] == spec.series_id and bool(row["primary_included"])
        ]
        for metric in METRICS:
            signs = [str(row[f"{metric}_sign"]) for row in rows]
            output.append(
                {
                    "series_id": spec.series_id,
                    "evaluator": spec.evaluator,
                    "span_policy": spec.span_policy,
                    "metric": metric,
                    "subset": "primary_M_ge_16",
                    "matched_pair_count": len(rows),
                    "mdlm_lower_count": signs.count("MDLM-lower"),
                    "mdlm_higher_count": signs.count("MDLM-higher"),
                    "tie_count": signs.count("tie"),
                }
            )
    return output


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    norm_a = math.sqrt(sum(value * value for value in centered_a))
    norm_b = math.sqrt(sum(value * value for value in centered_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_a, centered_b)) / (norm_a * norm_b)


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    return pearson(average_ranks(values_a), average_ranks(values_b))


def scoped_cell_ids(model_scope: str, primary: bool) -> list[str]:
    return [
        cell["cell_id"]
        for cell in CELL_SPECS
        if (model_scope == "pooled_unadjusted" or cell["model_family"] == model_scope)
        and (not primary or int(cell["steps"]) >= PRIMARY_MIN_UPDATES)
    ]


def make_rank_correlations(series: Mapping[str, LoadedSeries]) -> list[dict[str, Any]]:
    comparisons = (
        (
            "evaluator_at_full_stream",
            "gpt2_large_full_stream",
            "qwen3_8b_full_stream",
        ),
        (
            "evaluator_at_first_document",
            "gpt2_large_first_document",
            "qwen3_8b_first_document",
        ),
        (
            "span_policy_with_gpt2_large",
            "gpt2_large_full_stream",
            "gpt2_large_first_document",
        ),
        (
            "span_policy_with_qwen3_8b",
            "qwen3_8b_full_stream",
            "qwen3_8b_first_document",
        ),
    )
    rows: list[dict[str, Any]] = []
    for subset, primary in (("primary_M_ge_16", True), ("sensitivity_all_M", False)):
        for scope in ("pooled_unadjusted", "MDLM-Small", "DiDi-Instruct"):
            cell_ids = scoped_cell_ids(scope, primary)
            for comparison, left_id, right_id in comparisons:
                for metric in METRICS:
                    left = [positive_float(series[left_id].cells[cell][metric], metric) for cell in cell_ids]
                    right = [positive_float(series[right_id].cells[cell][metric], metric) for cell in cell_ids]
                    rows.append(
                        {
                            "comparison": comparison,
                            "left_series_id": left_id,
                            "right_series_id": right_id,
                            "metric": metric,
                            "subset": subset,
                            "model_scope": scope,
                            "n_cells": len(cell_ids),
                            "spearman_rho": spearman(left, right),
                            "transform_note": "ranks of positive raw values; log gives identical Spearman rho",
                        }
                    )
    return rows


def make_diagnostics(series: Mapping[str, LoadedSeries]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SERIES_SPECS:
        cells = list(series[spec.series_id].cells.values())
        totals = {
            field: sum(integer(cell[field], f"{spec.series_id}.{field}") for cell in cells)
            for field in ADDITIVE_FIELDS
        }
        per_cell_eval_bytes = [
            coverage(cell["scored_content_utf8_bytes"], cell["included_content_utf8_bytes"])
            for cell in cells
        ]
        per_cell_eval_boundaries = [
            coverage(cell["scored_boundary_count"], cell["included_boundary_count"])
            for cell in cells
            if integer(cell["included_boundary_count"], "included_boundary_count") > 0
        ]
        overall = series[spec.series_id].summary["overall"]
        rows.append(
            {
                "series_id": spec.series_id,
                "evaluator": spec.evaluator,
                "span_policy": spec.span_policy,
                "cell_count": len(cells),
                **totals,
                "scorable_output_fraction": coverage(
                    totals["scorable_output_count"], totals["output_count"]
                ),
                "policy_content_byte_coverage": coverage(
                    totals["included_content_utf8_bytes"], totals["source_content_utf8_bytes"]
                ),
                "evaluator_content_byte_coverage": coverage(
                    totals["scored_content_utf8_bytes"], totals["included_content_utf8_bytes"]
                ),
                "minimum_cell_evaluator_content_byte_coverage": min(
                    value for value in per_cell_eval_bytes if value is not None
                ),
                "policy_boundary_coverage": coverage(
                    totals["included_boundary_count"], totals["source_boundary_count"]
                ),
                "evaluator_boundary_coverage": coverage(
                    totals["scored_boundary_count"], totals["included_boundary_count"]
                ),
                "minimum_cell_evaluator_boundary_coverage": min(
                    value for value in per_cell_eval_boundaries if value is not None
                ),
                "cells_with_truncation": sum(
                    integer(cell["truncated_segment_count"], "truncated_segment_count") > 0
                    for cell in cells
                ),
                "token_micro_perplexity": positive_float(
                    overall["token_micro_perplexity"], "overall.token_micro_perplexity"
                ),
                "output_macro_geometric_perplexity": positive_float(
                    overall["output_macro_geometric_perplexity"],
                    "overall.output_macro_geometric_perplexity",
                ),
                "corpus_bits_per_utf8_byte": positive_float(
                    overall["corpus_bits_per_utf8_byte"], "overall.corpus_bits_per_utf8_byte"
                ),
            }
        )
    return rows


def load_group_eval(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {"model", "operating_point", "updates_M", "geq"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path}: missing GroupEval columns {missing}")
        rows = list(reader)
    if len(rows) != len(CELL_SPECS):
        raise ValueError(f"{path}: expected 20 GroupEval cells, found {len(rows)}")
    expected_lookup = {
        (cell["model_family"], cell["operating_point"], cell["steps"]): cell["cell_id"]
        for cell in CELL_SPECS
    }
    values: dict[str, float] = {}
    for index, row in enumerate(rows, 2):
        key = (
            str(row["model"]),
            str(row["operating_point"]),
            integer(row["updates_M"], f"{path}:{index}.updates_M"),
        )
        if key not in expected_lookup:
            raise ValueError(f"{path}:{index}: unexpected GroupEval cell {key}")
        cell_id = expected_lookup[key]
        if cell_id in values:
            raise ValueError(f"{path}:{index}: duplicate GroupEval cell {key}")
        geq = finite_float(row["geq"], f"{path}:{index}.geq")
        if not 0.0 <= geq <= 10.0:
            raise ValueError(f"{path}:{index}.geq: expected a score in [0,10]")
        values[cell_id] = geq
    require_equal(f"{path}: GroupEval identities", set(values), set(CELL_SPEC_BY_ID))
    return values, {
        "filename": path.name,
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "required_columns": sorted(required),
    }


def association_direction(rho: float | None) -> str:
    if rho is None:
        return "undefined"
    if rho < 0.0:
        return "inverse (higher GEQ accompanies lower log gPPL)"
    if rho > 0.0:
        return "positive (higher GEQ accompanies higher log gPPL)"
    return "zero monotone association"


def make_geq_associations(
    series: Mapping[str, LoadedSeries],
    geq: Mapping[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, primary in (("primary_M_ge_16", True), ("sensitivity_all_M", False)):
        for scope in ("pooled_unadjusted", "MDLM-Small", "DiDi-Instruct"):
            cell_ids = scoped_cell_ids(scope, primary)
            geq_values = [geq[cell_id] for cell_id in cell_ids]
            for spec in SERIES_SPECS:
                for metric in GPPL_METRICS:
                    log_gppl = [
                        math.log(positive_float(series[spec.series_id].cells[cell_id][metric], metric))
                        for cell_id in cell_ids
                    ]
                    rho = spearman(geq_values, log_gppl)
                    rows.append(
                        {
                            "series_id": spec.series_id,
                            "evaluator": spec.evaluator,
                            "span_policy": spec.span_policy,
                            "gppl_metric": metric,
                            "metric_role": (
                                "primary_token_micro" if metric == "token_micro_perplexity" else "macro_sensitivity"
                            ),
                            "subset": subset,
                            "model_scope": scope,
                            "n_cells": len(cell_ids),
                            "spearman_rho_geq_vs_log_gppl": rho,
                            "direction": association_direction(rho),
                            "inference_scope": "descriptive; no p-value or causal interpretation",
                        }
                    )
    return rows


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    empty_fieldnames: Sequence[str] = (),
) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    if not fieldnames:
        fieldnames = list(empty_fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted: dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float):
                    formatted[field] = f"{value:.12g}"
                elif value is None:
                    formatted[field] = ""
                else:
                    formatted[field] = value
            writer.writerow(formatted)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if abs(number) >= 1000:
        return f"{number:.1f}"
    if abs(number) >= 100:
        return f"{number:.2f}"
    return f"{number:.{digits}f}"


def series_matrix(series: Mapping[str, LoadedSeries], metric: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for cell in CELL_SPECS:
        rows.append(
            [
                cell["model_family"],
                cell["operating_point"],
                str(cell["steps"]),
                *(fmt(series[spec.series_id].cells[cell["cell_id"]][metric], 4) for spec in SERIES_SPECS),
            ]
        )
    return rows


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def make_markdown(
    series: Mapping[str, LoadedSeries],
    matched: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    geq_associations: Sequence[Mapping[str, Any]],
    geq_input: Mapping[str, Any] | None,
) -> str:
    headers = ["Model", "Operating point", "M", "GPT-2-L/full", "GPT-2-L/first", "Qwen/full", "Qwen/first"]
    lines = [
        "# Common causal-LM rescoring analysis",
        "",
        "All four summaries passed the exact 20-cell, source-hash, manifest, protocol, evaluator-revision, and span-policy checks. M=4 is retained as a declared invalidity-floor sensitivity cell but excluded from primary correlation/sign summaries.",
        "",
        "Lower perplexity and bits per byte are better evaluator likelihood; ratios below are MDLM-Small divided by DiDi-Instruct, so a ratio above one means DiDi has the lower value. Raw perplexities are interpreted only within an evaluator; bits per byte supports cross-tokenizer sensitivity checks.",
        "",
        "## Diagnostic coverage",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Evaluator", "Policy", "Outputs scored", "Policy byte coverage", "Evaluator byte coverage", "Boundary coverage", "Truncated segments"],
            (
                [
                    str(row["evaluator"]),
                    str(row["span_policy"]),
                    f"{row['scorable_output_count']}/{row['output_count']}",
                    fmt(row["policy_content_byte_coverage"], 4),
                    fmt(row["evaluator_content_byte_coverage"], 4),
                    fmt(row["evaluator_boundary_coverage"], 4),
                    str(row["truncated_segment_count"]),
                ]
                for row in diagnostics
            ),
        )
    )
    titles = {
        "token_micro_perplexity": "Token-micro perplexity",
        "output_macro_geometric_perplexity": "Equal-output macro geometric perplexity",
        "corpus_bits_per_utf8_byte": "Content bits per UTF-8 byte",
    }
    for metric in METRICS:
        lines.extend(["", f"## {titles[metric]}", ""])
        lines.extend(markdown_table(headers, series_matrix(series, metric)))

    compact_matched: dict[tuple[str, int], dict[str, Any]] = {}
    for row in matched:
        compact_matched.setdefault((str(row["operating_point"]), int(row["updates_M"])), {})[
            str(row["series_id"])
        ] = row
    lines.extend(["", "## Matched MDLM/DiDi token-micro perplexity ratios", ""])
    lines.extend(
        markdown_table(
            ["Operating point", "M", "Primary", "GPT-2-L/full", "GPT-2-L/first", "Qwen/full", "Qwen/first"],
            (
                [
                    operating_point,
                    str(updates),
                    "yes" if updates >= PRIMARY_MIN_UPDATES else "no",
                    *(
                        fmt(
                            compact_matched[(operating_point, updates)][spec.series_id][
                                "mdlm_over_didi_token_micro_perplexity_ratio"
                            ],
                            3,
                        )
                        for spec in SERIES_SPECS
                    ),
                ]
                for operating_point in ("reference", "selected")
                for updates in EXPECTED_UPDATES
            ),
        )
    )

    primary_rank = [
        row
        for row in rank_rows
        if row["subset"] == "primary_M_ge_16" and row["metric"] == "token_micro_perplexity"
    ]
    lines.extend(["", "## Primary rank stability (token-micro perplexity)", ""])
    lines.extend(
        markdown_table(
            ["Comparison", "Scope", "n", "Spearman rho"],
            (
                [
                    str(row["comparison"]),
                    str(row["model_scope"]),
                    str(row["n_cells"]),
                    fmt(row["spearman_rho"], 3),
                ]
                for row in primary_rank
            ),
        )
    )

    lines.extend(["", "## GroupEval association", ""])
    if geq_input is None:
        lines.append("No matched GroupEval CSV was supplied. The analysis is explicitly pending; no association is inferred from older unmatched pilots.")
    else:
        lines.append(
            f"Loaded 20 matched cells from `{geq_input['filename']}` (SHA-256 `{geq_input['sha256']}`). Negative rho means higher GEQ accompanies lower log gPPL. These are descriptive cell-level associations; the pooled scope is unadjusted for generator identity."
        )
        lines.append("")
        primary_geq = [
            row
            for row in geq_associations
            if row["subset"] == "primary_M_ge_16" and row["metric_role"] == "primary_token_micro"
        ]
        lines.extend(
            markdown_table(
                ["Evaluator", "Policy", "Scope", "n", "rho(GEQ, log gPPL)"],
                (
                    [
                        str(row["evaluator"]),
                        str(row["span_policy"]),
                        str(row["model_scope"]),
                        str(row["n_cells"]),
                        fmt(row["spearman_rho_geq_vs_log_gppl"], 3),
                    ]
                    for row in primary_geq
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "Correlations are descriptive rank summaries over predeclared cells, not independent-sample significance tests. The complete all-M sensitivity analyses, macro-PPL checks, signs, counts, hashes, and cell diagnostics are retained in the CSV and JSON artifacts regardless of direction.",
            "",
        ]
    )
    return "\n".join(lines)


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_table(
    columns: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    caption: str,
    label: str,
) -> list[str]:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(latex_escape(value) for value in row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    return lines


def make_latex(
    series: Mapping[str, LoadedSeries],
    matched: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    geq_associations: Sequence[Mapping[str, Any]],
) -> str:
    output = [
        "% Auto-generated by analyze_causal_rescoring.py; reviewer-safe aggregate data only.",
        "% M=4 is shown but excluded from primary correlation/sign summaries.",
        "",
    ]
    titles = {
        "token_micro_perplexity": ("Token-micro causal-LM perplexity for every cell.", "tab:causal_rescore_token_micro"),
        "output_macro_geometric_perplexity": ("Equal-output macro geometric causal-LM perplexity for every cell.", "tab:causal_rescore_output_macro"),
        "corpus_bits_per_utf8_byte": ("Content bits per UTF-8 byte for every cell; mapped EOS loss is excluded.", "tab:causal_rescore_bpb"),
    }
    headers = ["Model", "OP", "M", "GPT-2-L/full", "GPT-2-L/first", "Qwen/full", "Qwen/first"]
    for metric in METRICS:
        caption, label = titles[metric]
        output.extend(
            latex_table("llrrrrr", headers, series_matrix(series, metric), caption, label)
        )

    compact_matched: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in matched:
        compact_matched.setdefault((str(row["operating_point"]), int(row["updates_M"])), {})[
            str(row["series_id"])
        ] = row
    ratio_rows = [
        [
            operating_point,
            str(updates),
            "yes" if updates >= PRIMARY_MIN_UPDATES else "no",
            *(
                fmt(
                    compact_matched[(operating_point, updates)][spec.series_id][
                        "mdlm_over_didi_token_micro_perplexity_ratio"
                    ],
                    3,
                )
                for spec in SERIES_SPECS
            ),
        ]
        for operating_point in ("reference", "selected")
        for updates in EXPECTED_UPDATES
    ]
    output.extend(
        latex_table(
            "lrrrrrr",
            ["OP", "M", "Primary", "GPT-2-L/full", "GPT-2-L/first", "Qwen/full", "Qwen/first"],
            ratio_rows,
            "Matched MDLM-Small over DiDi-Instruct token-micro perplexity ratios. Values above one indicate lower DiDi evaluator perplexity.",
            "tab:causal_rescore_ratios",
        )
    )
    diagnostic_rows = [
        [
            str(row["evaluator"]),
            str(row["span_policy"]),
            f"{row['scorable_output_count']}/{row['output_count']}",
            fmt(row["policy_content_byte_coverage"], 4),
            fmt(row["evaluator_content_byte_coverage"], 4),
            fmt(row["evaluator_boundary_coverage"], 4),
            str(row["truncated_segment_count"]),
        ]
        for row in diagnostics
    ]
    output.extend(
        latex_table(
            "llrrrrr",
            ["Evaluator", "Policy", "Outputs", "Policy bytes", "Scored bytes", "Boundaries", "Trunc."],
            diagnostic_rows,
            "Coverage diagnostics. Policy-byte coverage is included/source content; scored-byte and boundary coverage are scored/included.",
            "tab:causal_rescore_coverage",
        )
    )
    primary_rank = [
        row
        for row in rank_rows
        if row["subset"] == "primary_M_ge_16" and row["metric"] == "token_micro_perplexity"
    ]
    output.extend(
        latex_table(
            "llrr",
            ["Comparison", "Scope", "n", "Spearman rho"],
            (
                [str(row["comparison"]), str(row["model_scope"]), str(row["n_cells"]), fmt(row["spearman_rho"], 3)]
                for row in primary_rank
            ),
            "Primary rank-stability diagnostics for token-micro perplexity, excluding M=4.",
            "tab:causal_rescore_rank",
        )
    )
    if geq_associations:
        primary_geq = [
            row
            for row in geq_associations
            if row["subset"] == "primary_M_ge_16" and row["metric_role"] == "primary_token_micro"
        ]
        output.extend(
            latex_table(
                "lllrr",
                ["Evaluator", "Policy", "Scope", "n", "rho(GEQ, log gPPL)"],
                (
                    [
                        str(row["evaluator"]),
                        str(row["span_policy"]),
                        str(row["model_scope"]),
                        str(row["n_cells"]),
                        fmt(row["spearman_rho_geq_vs_log_gppl"], 3),
                    ]
                    for row in primary_geq
                ),
                "Descriptive GroupEval-quality association with log token-micro gPPL, excluding M=4. Negative values indicate higher GEQ with lower gPPL; pooled estimates are unadjusted for generator identity.",
                "tab:causal_rescore_geq",
            )
        )
    else:
        output.append("% No matched GroupEval CSV supplied; association table intentionally omitted.")
    return "\n".join(output) + "\n"


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = {
        "gpt2_large_full_stream": args.gpt2_full_summary.resolve(),
        "gpt2_large_first_document": args.gpt2_first_summary.resolve(),
        "qwen3_8b_full_stream": args.qwen_full_summary.resolve(),
        "qwen3_8b_first_document": args.qwen_first_summary.resolve(),
    }
    if len(set(input_paths.values())) != len(input_paths):
        raise ValueError("the four summary paths must be distinct")
    series = {
        series_id: validate_summary(path, SPEC_BY_ID[series_id])
        for series_id, path in input_paths.items()
    }
    validate_common_inputs(series)
    cell_rows = make_cell_rows(series)
    matched_rows = make_matched_rows(series)
    sign_summaries = make_sign_summaries(matched_rows)
    rank_rows = make_rank_correlations(series)
    diagnostics = make_diagnostics(series)

    geq_values: dict[str, float] | None = None
    geq_input: dict[str, Any] | None = None
    if args.group_eval_csv is not None:
        geq_values, geq_input = load_group_eval(args.group_eval_csv.resolve())
    geq_associations = (
        make_geq_associations(series, geq_values) if geq_values is not None else []
    )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cell_metrics.csv", cell_rows)
    write_csv(output_dir / "matched_mdlm_didi.csv", matched_rows)
    write_csv(output_dir / "matched_sign_summary.csv", sign_summaries)
    write_csv(output_dir / "rank_correlations.csv", rank_rows)
    write_csv(output_dir / "diagnostic_coverage.csv", diagnostics)
    write_csv(
        output_dir / "geq_gppl_associations.csv",
        geq_associations,
        empty_fieldnames=(
            "series_id",
            "evaluator",
            "span_policy",
            "gppl_metric",
            "metric_role",
            "subset",
            "model_scope",
            "n_cells",
            "spearman_rho_geq_vs_log_gppl",
            "direction",
            "inference_scope",
        ),
    )

    manifest_hash = next(
        iter(str(loaded.summary["input_manifest_sha256"]) for loaded in series.values())
    )
    summary = {
        "artifact": "Common causal-LM rescoring comparison",
        "analysis_version": ANALYSIS_VERSION,
        "source_protocol_version": PROTOCOL_VERSION,
        "input_manifest_sha256": manifest_hash,
        "input_summaries": [
            {
                "series_id": spec.series_id,
                "filename": series[spec.series_id].filename,
                "sha256": series[spec.series_id].file_sha256,
                "evaluator": spec.evaluator,
                "model_identifier": spec.model_identifier,
                "revision": spec.revision,
                "span_policy": spec.span_policy,
                "private_per_output_artifact_sha256": series[spec.series_id].summary[
                    "private_per_output_artifact_sha256"
                ],
            }
            for spec in SERIES_SPECS
        ],
        "validation": {
            "exact_cell_count": len(CELL_SPECS),
            "outputs_per_cell": EXPECTED_OUTPUTS_PER_CELL,
            "common_source_hashes_verified": True,
            "common_source_metadata_verified": True,
            "common_scoring_contract_verified": True,
            "exact_model_revisions_verified": True,
            "primary_subset": "M >= 16; M=4 retained only as sensitivity",
        },
        "metric_direction": {
            "perplexity_and_bits_per_byte": "lower is better evaluator likelihood",
            "matched_ratio": "MDLM-Small divided by DiDi-Instruct",
            "geq": "higher is better judged quality",
            "geq_log_gppl_concordance": "negative Spearman rho",
        },
        "group_eval_input": geq_input,
        "cell_metrics": cell_rows,
        "matched_mdlm_didi": matched_rows,
        "matched_primary_sign_summary": sign_summaries,
        "rank_correlations": rank_rows,
        "diagnostic_coverage": diagnostics,
        "geq_gppl_associations": geq_associations,
        "interpretation_guardrails": [
            "Raw perplexity magnitudes are tokenizer-specific and are not compared directly across evaluators.",
            "Pooled GroupEval associations are unadjusted descriptive summaries across generator families.",
            "No p-values, independent-cell assumptions, or outcome-dependent metric selection are used.",
            "Null, positive, negative, and undefined correlations are emitted without filtering.",
        ],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    (output_dir / "results.md").write_text(
        make_markdown(series, matched_rows, rank_rows, diagnostics, geq_associations, geq_input),
        encoding="utf-8",
    )
    (output_dir / "tables.tex").write_text(
        make_latex(series, matched_rows, rank_rows, diagnostics, geq_associations),
        encoding="utf-8",
    )
    artifact_paths = sorted(path for path in output_dir.iterdir() if path.is_file())
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in artifact_paths),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "cell_metric_rows": len(cell_rows),
        "matched_rows": len(matched_rows),
        "rank_correlation_rows": len(rank_rows),
        "geq_association_rows": len(geq_associations),
        "group_eval_included": geq_input is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt2-full-summary", required=True, type=Path)
    parser.add_argument("--gpt2-first-summary", required=True, type=Path)
    parser.add_argument("--qwen-full-summary", required=True, type=Path)
    parser.add_argument("--qwen-first-summary", required=True, type=Path)
    parser.add_argument(
        "--group-eval-csv",
        type=Path,
        help="optional matched 20-cell GroupEval cell_results.csv",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
