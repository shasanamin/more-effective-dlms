#!/usr/bin/env python3
"""Merge GroupEval judgment shards in the frozen groups' canonical order.

The input shard order is deliberately irrelevant.  Every expected group ID
must occur exactly once across the shards, and no unexpected ID is accepted.
The completed file is published atomically without overwriting an existing
output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line, parse_constant=_reject_nonfinite)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON object: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: each JSONL row must be an object"
                )
            yield line_number, value, raw_line


def metadata_group_id(record: Mapping[str, Any], context: str) -> str:
    metadata = record.get("request_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{context}: request_metadata must be an object")
    value = metadata.get("group_id")
    if value is None or not str(value).strip():
        raise ValueError(f"{context}: request_metadata.group_id is required")
    return str(value)


def canonical_group_ids(path: Path) -> list[str]:
    ordered_ids: list[str] = []
    seen: dict[str, int] = {}
    for line_number, record, _ in read_jsonl(path):
        group_id = metadata_group_id(record, f"{path}:{line_number}")
        if group_id in seen:
            raise ValueError(
                f"{path}:{line_number}: duplicate group_id {group_id!r}; "
                f"first seen on line {seen[group_id]}"
            )
        group_input = record.get("group_input")
        if isinstance(group_input, Mapping):
            input_group_id = group_input.get("group_id")
            if input_group_id is not None and str(input_group_id) != group_id:
                raise ValueError(
                    f"{path}:{line_number}: group_input.group_id does not match "
                    "request_metadata.group_id"
                )
        seen[group_id] = line_number
        ordered_ids.append(group_id)
    if not ordered_ids:
        raise ValueError(f"{path}: no groups found")
    return ordered_ids


def _run_value(
    record: Mapping[str, Any],
    field: str,
    context: str,
    expected: str | None,
) -> str | None:
    run_metadata = record.get("run_metadata")
    value: Any = None
    if isinstance(run_metadata, Mapping):
        value = run_metadata.get(field)
    normalized = str(value) if value is not None and str(value).strip() else None
    if expected is not None and normalized != expected:
        raise ValueError(
            f"{context}: run_metadata.{field} must be {expected!r}, "
            f"found {normalized!r}"
        )
    return normalized


def load_shards(
    shard_paths: list[Path],
    expected_ids: set[str],
    expected_model: str | None,
    expected_provider: str | None,
) -> tuple[dict[str, str], str | None, str | None]:
    rows_by_id: dict[str, str] = {}
    first_seen: dict[str, str] = {}
    model_values: set[str] = set()
    provider_values: set[str] = set()
    model_missing: list[str] = []
    provider_missing: list[str] = []

    resolved_paths: set[Path] = set()
    for shard_path in shard_paths:
        resolved = shard_path.resolve()
        if resolved in resolved_paths:
            raise ValueError(f"shard path was supplied more than once: {shard_path}")
        resolved_paths.add(resolved)

        shard_rows = 0
        for line_number, record, raw_line in read_jsonl(shard_path):
            shard_rows += 1
            context = f"{shard_path}:{line_number}"
            group_id = metadata_group_id(record, context)
            if group_id not in expected_ids:
                raise ValueError(f"{context}: unexpected group_id {group_id!r}")
            if group_id in rows_by_id:
                raise ValueError(
                    f"{context}: duplicate judgment group_id {group_id!r}; "
                    f"first seen at {first_seen[group_id]}"
                )

            model = _run_value(
                record,
                "judge_model_snapshot",
                context,
                expected_model,
            )
            provider = _run_value(
                record,
                "judge_provider",
                context,
                expected_provider,
            )
            if model is None:
                model_missing.append(context)
            else:
                model_values.add(model)
            if provider is None:
                provider_missing.append(context)
            else:
                provider_values.add(provider)

            rows_by_id[group_id] = (
                raw_line if raw_line.endswith("\n") else raw_line + "\n"
            )
            first_seen[group_id] = context
        if shard_rows == 0:
            raise ValueError(f"{shard_path}: shard contains no judgment records")

    missing = sorted(expected_ids - set(rows_by_id))
    if missing:
        preview = ", ".join(repr(group_id) for group_id in missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(f"missing expected judgment group IDs: {preview}{suffix}")

    # If either run field is recorded, require it on every row and require one
    # coherent series.  Expected-value flags additionally make the field
    # mandatory even when every shard omitted it.
    if model_values and model_missing:
        raise ValueError(
            "run_metadata.judge_model_snapshot is missing for only a subset "
            f"of judgments (first missing at {model_missing[0]})"
        )
    if provider_values and provider_missing:
        raise ValueError(
            "run_metadata.judge_provider is missing for only a subset "
            f"of judgments (first missing at {provider_missing[0]})"
        )
    if expected_model is not None and model_missing:
        raise ValueError(
            "run_metadata.judge_model_snapshot is required when --expected-model "
            f"is used (first missing at {model_missing[0]})"
        )
    if expected_provider is not None and provider_missing:
        raise ValueError(
            "run_metadata.judge_provider is required when --expected-provider "
            f"is used (first missing at {provider_missing[0]})"
        )
    if len(model_values) > 1:
        raise ValueError(
            "mixed run_metadata.judge_model_snapshot values across shards: "
            + ", ".join(repr(value) for value in sorted(model_values))
        )
    if len(provider_values) > 1:
        raise ValueError(
            "mixed run_metadata.judge_provider values across shards: "
            + ", ".join(repr(value) for value in sorted(provider_values))
        )

    model = next(iter(model_values), None)
    provider = next(iter(provider_values), None)
    return rows_by_id, model, provider


def publish_fresh_atomic(output: Path, ordered_lines: Iterable[str]) -> None:
    """Publish a complete file atomically and never replace an existing path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"{output} already exists; refusing to overwrite it")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output.name}.tmp-",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for line in ordered_lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

        # A hard link publishes the fully written inode without the overwrite
        # behavior of os.replace().  It also fails atomically if another process
        # created the requested fresh output after the existence check above.
        os.link(temporary_name, output)
        os.unlink(temporary_name)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        required=True,
        type=Path,
        help="Frozen groups JSONL whose order defines the merged output order",
    )
    parser.add_argument(
        "--shards",
        required=True,
        type=Path,
        nargs="+",
        help="One or more completed judgment shard JSONLs",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-model",
        default=None,
        help="Require this exact run_metadata.judge_model_snapshot value",
    )
    parser.add_argument(
        "--expected-provider",
        default=None,
        help="Require this exact run_metadata.judge_provider value",
    )
    args = parser.parse_args()

    output_resolved = args.output.resolve()
    if args.groups.resolve() == output_resolved:
        raise ValueError("--groups and --output must be different paths")
    if any(path.resolve() == output_resolved for path in args.shards):
        raise ValueError("--output must not be one of the input shards")
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists; refusing to overwrite it"
        )

    ordered_ids = canonical_group_ids(args.groups)
    rows_by_id, model, provider = load_shards(
        args.shards,
        set(ordered_ids),
        args.expected_model,
        args.expected_provider,
    )
    publish_fresh_atomic(
        args.output,
        (rows_by_id[group_id] for group_id in ordered_ids),
    )
    print(
        json.dumps(
            {
                "groups": str(args.groups),
                "shards": [str(path) for path in args.shards],
                "output": str(args.output),
                "records": len(ordered_ids),
                "judge_provider": provider,
                "judge_model_snapshot": model,
                "sha256": sha256_file(args.output),
                "canonical_group_order": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
