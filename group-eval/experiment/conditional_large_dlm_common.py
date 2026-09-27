#!/usr/bin/env python3
"""Shared constants and invariants for conditional large-DLM GEQ validation.

This is deliberately separate from unconditional GroupEval v1.  The conditional
study has one task condition and one response per model, so it evaluates GEQ but
does not define or report GED.
"""

from __future__ import annotations

from typing import Any, Mapping


PROTOCOL_VERSION = "1.0.0-conditional-large-dlm"
EVALUATION_PROTOCOL = "conditional-large-dlm-geq-v1"
TARGET_PROFILE_ID = "large-dlm-tasks-v1"
QUALITY_DIMENSIONS = (
    "readability",
    "reasoning_coherence",
    "task_success",
    "nondegeneration_structure",
)


def is_conditional_request(metadata: Mapping[str, Any]) -> bool:
    """Return whether request metadata selects the conditional validator."""

    return metadata.get("evaluation_protocol") == EVALUATION_PROTOCOL


def conditional_geq(sample: Mapping[str, Any]) -> float:
    """Compute the declared 0--10 conditional GEQ sample score."""

    if not bool(sample.get("valid")):
        return 0.0
    quality = sample.get("quality")
    if not isinstance(quality, Mapping):
        raise ValueError("sample.quality must be an object")
    return 1.25 * sum(int(quality[name]) for name in QUALITY_DIMENSIONS)


def conditional_semantic_errors(record: Mapping[str, Any]) -> list[str]:
    """Check conditional cross-field invariants not encoded by JSON Schema."""

    output = record.get("output", record)
    if not isinstance(output, Mapping):
        return ["record.output must be an object"]
    metadata = record.get("request_metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}

    errors: list[str] = []
    if output.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(
            f"protocol_version must be {PROTOCOL_VERSION!r}"
        )
    if metadata.get("protocol_version") is not None and (
        output.get("protocol_version") != metadata.get("protocol_version")
    ):
        errors.append("output.protocol_version does not match request metadata")
    if metadata.get("group_id") is not None and (
        output.get("group_id") != metadata.get("group_id")
    ):
        errors.append("output.group_id does not match request metadata")

    samples = output.get("samples")
    if not isinstance(samples, list):
        return errors + ["output.samples must be an array"]
    sample_ids = [
        sample.get("sample_id") if isinstance(sample, Mapping) else None
        for sample in samples
    ]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample IDs are not unique")

    sample_map = metadata.get("sample_map")
    if isinstance(sample_map, list):
        expected_ids = [
            item.get("display_id") if isinstance(item, Mapping) else None
            for item in sample_map
        ]
        if len(sample_ids) != len(expected_ids) or set(sample_ids) != set(expected_ids):
            errors.append(
                "output sample IDs do not match request_metadata.sample_map"
            )

    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        sample_id = str(sample.get("sample_id"))
        valid = bool(sample.get("valid"))
        quality = sample.get("quality")
        if not isinstance(quality, Mapping):
            errors.append(f"sample {sample_id}: quality must be an object")
            continue
        scores = [quality.get(name) for name in QUALITY_DIMENSIONS]
        if valid:
            if sample.get("invalid_reason") is not None:
                errors.append(
                    f"valid sample {sample_id} must have invalid_reason=null"
                )
        else:
            if sample.get("invalid_reason") is None:
                errors.append(
                    f"invalid sample {sample_id} requires an invalid_reason"
                )
            if any(score != 0 for score in scores):
                errors.append(
                    f"invalid sample {sample_id} must have zero on all quality dimensions"
                )

        severe = bool(sample.get("severe_degeneration"))
        nondeg = quality.get("nondegeneration_structure")
        if severe and nondeg != 0:
            errors.append(
                f"sample {sample_id}: severe_degeneration=true requires "
                "nondegeneration_structure=0"
            )
        if valid and nondeg == 0 and not severe:
            errors.append(
                f"valid sample {sample_id}: nondegeneration_structure=0 requires "
                "severe_degeneration=true"
            )

    return errors
