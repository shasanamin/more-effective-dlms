#!/usr/bin/env python3
"""Validate GroupEval judge outputs against JSON Schema and semantic invariants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield line_number, value


def unwrap_output(record: dict[str, Any]) -> dict[str, Any]:
    output = record.get("output", record)
    if not isinstance(output, dict):
        raise ValueError("record.output must be an object")
    return output


def expected_display_ids(record: dict[str, Any]) -> list[str] | None:
    metadata = record.get("request_metadata")
    if not isinstance(metadata, dict):
        return None
    sample_map = metadata.get("sample_map")
    if not isinstance(sample_map, list):
        return None
    ids: list[str] = []
    for item in sample_map:
        if not isinstance(item, dict) or not isinstance(item.get("display_id"), str):
            raise ValueError("request_metadata.sample_map entries require display_id")
        ids.append(item["display_id"])
    return ids


def semantic_errors(record: dict[str, Any]) -> list[str]:
    """Return cross-field errors not expressible conveniently in JSON Schema."""
    output = unwrap_output(record)
    errors: list[str] = []
    samples = output.get("samples", [])
    modes = output.get("modes", [])

    sample_ids = [item.get("sample_id") for item in samples if isinstance(item, dict)]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample IDs are not unique")

    expected = expected_display_ids(record)
    if expected is not None and set(sample_ids) != set(expected):
        errors.append(
            f"output sample IDs do not match request sample_map: expected={sorted(expected)}, "
            f"observed={sorted(str(x) for x in sample_ids)}"
        )
    if expected is not None and len(sample_ids) != len(expected):
        errors.append("output sample count does not match request sample_map")

    mode_ids = [item.get("mode_id") for item in modes if isinstance(item, dict)]
    if len(mode_ids) != len(set(mode_ids)):
        errors.append("mode IDs are not unique")

    mode_membership: dict[str, list[str]] = {}
    for mode in modes:
        if not isinstance(mode, dict):
            continue
        mode_id = mode.get("mode_id")
        for member in mode.get("member_sample_ids", []):
            mode_membership.setdefault(str(member), []).append(str(mode_id))

    sample_by_id = {
        str(sample.get("sample_id")): sample
        for sample in samples
        if isinstance(sample, dict) and sample.get("sample_id") is not None
    }

    unknown_members = sorted(set(mode_membership) - set(sample_by_id))
    if unknown_members:
        errors.append(f"mode records contain unknown sample IDs: {unknown_members}")

    for sample_id, sample in sample_by_id.items():
        valid = bool(sample.get("valid"))
        assigned = mode_membership.get(sample_id, [])
        declared_mode = sample.get("mode_id")
        quality = sample.get("quality", {})
        scores = [
            quality.get("readability"),
            quality.get("coherence"),
            quality.get("substance_naturalness"),
            quality.get("nondegeneration_structure"),
        ]

        if valid:
            if sample.get("invalid_reason") is not None:
                errors.append(f"valid sample {sample_id} must have invalid_reason=null")
            if not isinstance(declared_mode, str):
                errors.append(f"valid sample {sample_id} must declare a non-null mode_id")
            if len(assigned) != 1:
                errors.append(f"valid sample {sample_id} must appear in exactly one mode record")
            elif assigned[0] != declared_mode:
                errors.append(
                    f"sample {sample_id} declares {declared_mode} but mode record assigns {assigned[0]}"
                )
        else:
            if sample.get("invalid_reason") is None:
                errors.append(f"invalid sample {sample_id} requires an invalid_reason")
            if declared_mode is not None:
                errors.append(f"invalid sample {sample_id} must have mode_id=null")
            if bool(sample.get("hybrid")):
                errors.append(f"invalid sample {sample_id} must have hybrid=false")
            if assigned:
                errors.append(f"invalid sample {sample_id} appears in a mode record")
            if any(score != 0 for score in scores):
                errors.append(f"invalid sample {sample_id} must have zero on all quality dimensions")

        severe = bool(sample.get("severe_degeneration"))
        nondeg = quality.get("nondegeneration_structure")
        if severe and nondeg != 0:
            errors.append(
                f"sample {sample_id}: severe_degeneration=true requires nondegeneration_structure=0"
            )
        # A valid sample with a zero non-degeneration score must be an observed severe
        # loop/template/structural failure. An invalid sample can have all dimensions
        # zero for unrelated reasons (for example, token corruption), so the converse
        # is not enforced for invalid samples.
        if valid and nondeg == 0 and not severe:
            errors.append(
                f"valid sample {sample_id}: nondegeneration_structure=0 requires "
                "severe_degeneration=true"
            )

    # Enforce contiguous mode IDs M1..MK in the order listed. This limits arbitrary labels.
    expected_mode_ids = [f"M{i}" for i in range(1, len(mode_ids) + 1)]
    if mode_ids != expected_mode_ids:
        errors.append(f"mode IDs must be contiguous and listed as {expected_mode_ids}")

    first_appearance: list[str] = []
    for sample in samples:
        if isinstance(sample, dict) and bool(sample.get("valid")):
            mode_id = sample.get("mode_id")
            if isinstance(mode_id, str) and mode_id not in first_appearance:
                first_appearance.append(mode_id)
    if first_appearance != expected_mode_ids:
        errors.append(
            f"mode IDs must follow first appearance in samples: expected={expected_mode_ids}, "
            f"observed={first_appearance}"
        )

    if output.get("group_id") is not None:
        metadata = record.get("request_metadata")
        if isinstance(metadata, dict) and metadata.get("group_id") is not None:
            if output["group_id"] != metadata["group_id"]:
                errors.append("output.group_id does not match request_metadata.group_id")
        if isinstance(metadata, dict) and metadata.get("protocol_version") is not None:
            if output.get("protocol_version") != metadata["protocol_version"]:
                errors.append(
                    "output.protocol_version does not match "
                    "request_metadata.protocol_version"
                )

    return errors


def validate_record(
    record: dict[str, Any], validator: Draft202012Validator
) -> list[str]:
    output = unwrap_output(record)
    errors = [
        f"schema: {'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(output), key=lambda err: list(err.absolute_path))
    ]
    try:
        errors.extend(f"semantic: {message}" for message in semantic_errors(record))
    except (TypeError, ValueError) as exc:
        errors.append(f"semantic: {exc}")
    return errors


def _metadata_group_id(record: Mapping[str, Any]) -> str | None:
    metadata = record.get("request_metadata")
    if not isinstance(metadata, Mapping):
        return None
    group_id = metadata.get("group_id")
    return str(group_id) if group_id is not None and str(group_id) else None


def _expected_group_map(
    rows: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    expected: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for line_number, record in rows:
        metadata = record.get("request_metadata")
        group_input = record.get("group_input")
        if not isinstance(metadata, dict):
            errors.append(f"groups line {line_number}: request_metadata must be an object")
            continue
        group_id = _metadata_group_id(record)
        if group_id is None:
            errors.append(f"groups line {line_number}: request_metadata.group_id is required")
            continue
        if group_id in expected:
            errors.append(f"groups file contains duplicate group_id {group_id!r}")
            continue
        if not isinstance(group_input, dict):
            errors.append(f"groups line {line_number}: group_input must be an object")
        elif str(group_input.get("group_id", "")) != group_id:
            errors.append(
                f"groups line {line_number}: group_input.group_id does not match "
                "request_metadata.group_id"
            )
        expected[group_id] = record
    return expected, errors


def _summarize_ids(values: Sequence[str], limit: int = 8) -> str:
    preview = ", ".join(repr(value) for value in values[:limit])
    suffix = "" if len(values) <= limit else f", ... ({len(values)} total)"
    return preview + suffix


def dataset_errors(
    rows: Sequence[tuple[int, dict[str, Any]]],
    expected_rows: Sequence[tuple[int, dict[str, Any]]] | None = None,
) -> list[str]:
    """Validate completeness, uniqueness, and run-level metadata consistency.

    Record-level schema/semantic validation cannot detect a cleanly formed but
    incomplete or duplicated API run. When ``expected_rows`` is supplied from the
    frozen groups JSONL, this function treats its group IDs and request metadata as
    the run manifest.
    """
    errors: list[str] = []
    observed_by_group: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    missing_metadata_lines: list[int] = []
    sample_contexts: Counter[tuple[str, str, str]] = Counter()

    for line_number, record in rows:
        group_id = _metadata_group_id(record)
        if group_id is None:
            missing_metadata_lines.append(line_number)
        else:
            observed_by_group[group_id].append((line_number, record))

        metadata = record.get("request_metadata")
        if not isinstance(metadata, Mapping):
            continue
        method = str(metadata.get("method_code", ""))
        partition = str(metadata.get("partition_id", ""))
        sample_map = metadata.get("sample_map")
        if not isinstance(sample_map, list):
            continue
        for item in sample_map:
            if not isinstance(item, Mapping) or item.get("sample_index") is None:
                continue
            sample_contexts[(method, partition, str(item["sample_index"]))] += 1

    if missing_metadata_lines:
        errors.append(
            "judgments missing request_metadata.group_id on lines "
            + ", ".join(map(str, missing_metadata_lines[:8]))
        )
    duplicate_groups = sorted(
        group_id for group_id, records in observed_by_group.items() if len(records) != 1
    )
    if duplicate_groups:
        errors.append(
            "duplicate judgment group_id values: " + _summarize_ids(duplicate_groups)
        )
    duplicate_contexts = sorted(
        key for key, count in sample_contexts.items() if count != 1
    )
    if duplicate_contexts:
        preview = ["/".join(key) for key in duplicate_contexts]
        errors.append(
            "duplicate method/partition/sample_index judgment contexts: "
            + _summarize_ids(preview)
        )

    # A production run should use one coherent judge/prompt/schema/profile series.
    run_fields = (
        "judge_provider",
        "judge_model_snapshot",
        "reasoning_effort",
        "tools_enabled",
        "prompt_sha256",
        "schema_sha256",
        "target_profile_sha256",
    )
    run_metadata_present = [
        isinstance(record.get("run_metadata"), Mapping) for _, record in rows
    ]
    if any(run_metadata_present) and not all(run_metadata_present):
        errors.append("run_metadata is present for only a subset of judgment records")
    if all(run_metadata_present) and rows:
        for field in run_fields:
            present = [
                record["run_metadata"].get(field) is not None for _, record in rows
            ]
            if any(present) and not all(present):
                errors.append(
                    f"run_metadata.{field} is present for only a subset of judgments"
                )
            values = {
                json.dumps(record["run_metadata"].get(field), sort_keys=True)
                for _, record in rows
                if record["run_metadata"].get(field) is not None
            }
            if len(values) > 1:
                errors.append(f"run_metadata.{field} is inconsistent across judgments")
        for field in ("judge_provider", "judge_model_snapshot"):
            missing = [
                line_number
                for line_number, record in rows
                if not str(record["run_metadata"].get(field, "")).strip()
            ]
            if missing:
                errors.append(
                    f"run_metadata.{field} is missing on lines "
                    + ", ".join(map(str, missing[:8]))
                )
        response_ids = [
            str(record["run_metadata"]["response_id"])
            for _, record in rows
            if record["run_metadata"].get("response_id")
        ]
        if response_ids and len(response_ids) != len(rows):
            errors.append(
                "run_metadata.response_id is present for only a subset of judgments"
            )
        duplicate_response_ids = sorted(
            response_id
            for response_id, count in Counter(response_ids).items()
            if count > 1
        )
        if duplicate_response_ids:
            errors.append(
                "duplicate run_metadata.response_id values: "
                + _summarize_ids(duplicate_response_ids)
            )

    request_metadata_rows = [
        record["request_metadata"]
        for _, record in rows
        if isinstance(record.get("request_metadata"), Mapping)
    ]
    protocol_versions_present = [
        metadata.get("protocol_version") is not None
        for metadata in request_metadata_rows
    ]
    if any(protocol_versions_present) and not all(protocol_versions_present):
        errors.append(
            "request_metadata.protocol_version is present for only a subset of judgments"
        )
    request_protocols = {
        str(metadata["protocol_version"])
        for metadata in request_metadata_rows
        if metadata.get("protocol_version") is not None
    }
    if len(request_protocols) > 1:
        errors.append("request_metadata.protocol_version is inconsistent across judgments")
    protocol_hashes_present = [
        metadata.get("protocol_sha256") is not None for metadata in request_metadata_rows
    ]
    if any(protocol_hashes_present) and not all(protocol_hashes_present):
        errors.append(
            "request_metadata.protocol_sha256 is present for only a subset of judgments"
        )
    request_protocol_hashes = {
        str(metadata["protocol_sha256"])
        for metadata in request_metadata_rows
        if metadata.get("protocol_sha256") is not None
    }
    if len(request_protocol_hashes) > 1:
        errors.append("request_metadata.protocol_sha256 is inconsistent across judgments")

    if expected_rows is None:
        return errors

    expected, expected_errors = _expected_group_map(expected_rows)
    errors.extend(expected_errors)
    expected_ids = set(expected)
    observed_ids = set(observed_by_group)
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    if missing:
        errors.append(
            "missing expected judgment group_id values: " + _summarize_ids(missing)
        )
    if unexpected:
        errors.append(
            "unexpected judgment group_id values: " + _summarize_ids(unexpected)
        )

    immutable_fields = (
        "protocol_version",
        "protocol_sha256",
        "method_code",
        "partition_id",
        "group_index",
        "group_id",
        "grouping_seed",
        "order_seed",
        "target_profile_id",
        "sample_map",
    )
    for group_id in sorted(expected_ids & observed_ids):
        if len(observed_by_group[group_id]) != 1:
            continue
        line_number, observed = observed_by_group[group_id][0]
        observed_metadata = observed.get("request_metadata")
        expected_metadata = expected[group_id].get("request_metadata")
        if not isinstance(observed_metadata, Mapping) or not isinstance(
            expected_metadata, Mapping
        ):
            errors.append(f"judgment line {line_number}: request_metadata must be an object")
            continue
        for field in immutable_fields:
            if observed_metadata.get(field) != expected_metadata.get(field):
                errors.append(
                    f"judgment line {line_number} group {group_id}: "
                    f"request_metadata.{field} differs from groups manifest"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Judgment JSONL file")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "schemas" / "absolute_output.schema.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path; stdout is always concise",
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help=(
            "Optional frozen groups JSONL. When supplied, require exactly one judgment "
            "for every expected group and matching immutable request metadata."
        ),
    )
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    report: list[dict[str, Any]] = []
    total = 0
    invalid = 0

    try:
        rows = list(read_jsonl(args.input))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for line_number, record in rows:
        total += 1
        errors = validate_record(record, validator)
        if errors:
            invalid += 1
            report.append({"line": line_number, "errors": errors})
            for message in errors:
                print(f"{args.input}:{line_number}: {message}", file=sys.stderr)

    expected_rows: list[tuple[int, dict[str, Any]]] | None = None
    if args.groups is not None:
        try:
            expected_rows = list(read_jsonl(args.groups))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    global_errors = dataset_errors(rows, expected_rows=expected_rows)
    for message in global_errors:
        print(f"{args.input}: dataset: {message}", file=sys.stderr)

    result = {
        "input": str(args.input),
        "schema": str(args.schema),
        "groups": str(args.groups) if args.groups is not None else None,
        "records": total,
        "valid_records": total - invalid,
        "invalid_records": invalid,
        "errors": report,
        "dataset_errors": global_errors,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if invalid or global_errors:
        print(
            f"FAILED: {invalid}/{total} records invalid; "
            f"{len(global_errors)} dataset-level errors",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {total} records valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
