#!/usr/bin/env python3
"""Validate and analyze the matched MDLM-Small/DiDi-Instruct GroupEval run.

The analyzer requires a complete 200-call judgment file produced from
``prepare_cross_model_pilot.py``.  It validates the frozen group design,
request ordering, method blindness, judge schema/semantics, source provenance,
and exact request-metadata round trip before scoring.

Inference uses the protocol's crossed Bayesian bootstrap.  Partition weights
are shared across methods because every method uses the same two frozen
grouping layouts; sample weights remain independent because the two models do
not have common-random-number outputs.  Reported MDLM--DiDi contrasts are thus
matched by operating point and update budget and blocked by grouping context,
but are deliberately not described as sample-paired.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator


GROUPEVAL_ROOT = Path(__file__).resolve().parents[1]
SCORING_ROOT = GROUPEVAL_ROOT / "scoring"
if str(SCORING_ROOT) not in sys.path:
    sys.path.insert(0, str(SCORING_ROOT))

from bootstrap_group_eval import (  # noqa: E402
    _build_method_data,
    _weighted_method_scores,
    percentile_interval,
)
from score_group_eval import (  # noqa: E402
    DIMENSIONS,
    aggregate_systems,
    flatten_judgments,
)
from validate_outputs import dataset_errors, validate_record  # noqa: E402


ANALYSIS_VERSION = "1.0.0"
EXPECTED_ARTIFACT = "Matched MDLM-Small/DiDi-Instruct GroupEval pilot"
EXPECTED_METHODS = 20
EXPECTED_SAMPLES = 40
EXPECTED_GROUP_SIZE = 8
EXPECTED_PARTITIONS = ("P1", "P2")
EXPECTED_GROUPS_PER_PARTITION = 5
EXPECTED_CALLS_PER_METHOD = 10
EXPECTED_CALLS = 200
EXPECTED_SHARDS = 4
EXPECTED_SHARD_ALGORITHM = "rotating-opaque-method-rank-v1"
EXPECTED_UPDATES = (4, 16, 64, 256, 1024)
DEFAULT_REPLICATES = 10_000
DEFAULT_SEED = 20260731
EXPECTED_PROVIDER = "institutional_genai"
EXPECTED_MODEL = "gpt-oss:120b"
PAIRING_MODE = "partitions"
EXPECTED_PROTOCOL_VERSION = "1.0.0"
EXPECTED_TARGET_PROFILE_ID = "owt1024-fullstream-v1"
SHARD_METADATA_FIELDS = frozenset({
    "shard_id",
    "shard_count",
    "shard_sequence_index",
    "shard_algorithm",
})


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected an object")
    return value


def require_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected an array")
    return value


def require_equal(context: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{context}: expected {expected!r}, observed {observed!r}")


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


def verify_sha256s(
    sums_path: Path,
    required: Mapping[str, Path],
) -> dict[str, str]:
    """Verify every listed prepared artifact and bind required paths to it."""
    if not sums_path.is_file():
        raise FileNotFoundError(f"prepared-artifact hash manifest not found: {sums_path}")
    bundle_root = sums_path.parent.resolve()
    entries: dict[str, str] = {}
    for line_number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is None:
            raise ValueError(f"{sums_path}:{line_number}: malformed SHA256SUMS row")
        digest, name = parts[0].lower(), parts[1]
        relative = Path(name)
        if (
            relative.is_absolute()
            or name in {"", "."}
            or ".." in relative.parts
            or relative.as_posix() != name
        ):
            raise ValueError(f"{sums_path}:{line_number}: unsafe hash entry {name!r}")
        if name in entries:
            raise ValueError(f"{sums_path}: duplicate hash entry {name!r}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"{sums_path}: empty SHA256SUMS manifest")
    for name, expected_digest in entries.items():
        candidate = bundle_root / name
        if candidate.is_symlink():
            raise ValueError(f"{sums_path}: hash entry must not be a symlink: {name!r}")
        resolved = candidate.resolve()
        try:
            resolved.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(f"{sums_path}: hash entry escapes bundle root: {name!r}") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"{sums_path}: listed file not found: {name!r}")
        require_equal(f"{resolved}:SHA256SUMS", sha256_file(resolved), expected_digest)
    for name, path in required.items():
        if name not in entries:
            raise ValueError(f"{sums_path}: missing required entry {name}")
        require_equal(f"{name}:required path", path.resolve(), (bundle_root / name).resolve())
    return entries


def validate_identity(
    provenance: Mapping[str, Any],
    private_map: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    require_equal("provenance.artifact", provenance.get("artifact"), EXPECTED_ARTIFACT)
    require_equal("private-map target profile", private_map.get("target_profile_id"), "owt1024-fullstream-v1")
    if not bool(private_map.get("keep_private_until_judging_complete")):
        raise ValueError("private label map does not preserve the predeclared blind")
    methods_raw = require_mapping(private_map.get("methods"), "private_label_map.methods")
    methods = {
        str(code): require_mapping(label, f"private_label_map.methods.{code}")
        for code, label in methods_raw.items()
    }
    cells_raw = require_sequence(provenance.get("cells"), "generation_provenance.cells")
    cells: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(cells_raw):
        cell = require_mapping(item, f"generation_provenance.cells[{index}]")
        code = str(cell.get("method_code", ""))
        if not code or code in cells:
            raise ValueError(f"generation_provenance.cells[{index}]: missing/duplicate method_code")
        cells[code] = cell
    require_equal("private/provenance method sets", set(methods), set(cells))
    require_equal("method count", len(methods), EXPECTED_METHODS)

    combinations: set[tuple[str, str, int]] = set()
    for code in sorted(methods):
        label, cell = methods[code], cells[code]
        model = str(label.get("model"))
        operating_point = str(label.get("operating_point"))
        updates = integer(label.get("configured_updates_M"), f"{code}.configured_updates_M")
        if model not in {"MDLM-Small", "DiDi-Instruct"}:
            raise ValueError(f"{code}: unsupported model label {model!r}")
        if operating_point not in {"reference", "selected"}:
            raise ValueError(f"{code}: unsupported operating point {operating_point!r}")
        if updates not in EXPECTED_UPDATES:
            raise ValueError(f"{code}: unsupported update budget {updates}")
        combination = (model, operating_point, updates)
        if combination in combinations:
            raise ValueError(f"duplicate method cell {combination}")
        combinations.add(combination)
        for field in (
            "model",
            "configured_updates_M",
            "operating_point",
            "condition",
            "temperature",
            "categorical_precision",
            "sampler",
        ):
            require_equal(f"{code}:label/provenance {field}", str(label.get(field)), str(cell.get(field)))
        require_equal(f"{code}:sample_count", integer(cell.get("sample_count"), f"{code}.sample_count"), EXPECTED_SAMPLES)
        metrics = require_mapping(cell.get("source_metrics"), f"{code}.source_metrics")
        require_equal(f"{code}:source evaluator", metrics.get("evaluator"), "gpt2-large")
        require_equal(
            f"{code}:source evaluator revision",
            metrics.get("evaluator_revision"),
            "32b71b12589c2f8d625668d2335a01cac3249519",
        )
        finite_float(metrics.get("generative_perplexity"), f"{code}:source gPPL")
        finite_float(metrics.get("within_sequence_entropy"), f"{code}:source entropy")
    expected_combinations = {
        (model, operating_point, updates)
        for model in ("MDLM-Small", "DiDi-Instruct")
        for operating_point in ("reference", "selected")
        for updates in EXPECTED_UPDATES
    }
    require_equal("model/operating-point/update cells", combinations, expected_combinations)

    design = require_mapping(provenance.get("design"), "generation_provenance.design")
    expected_design = {
        "methods": EXPECTED_METHODS,
        "samples_per_method": EXPECTED_SAMPLES,
        "group_size": EXPECTED_GROUP_SIZE,
        "groups_per_partition_per_method": EXPECTED_GROUPS_PER_PARTITION,
        "calls_per_method": EXPECTED_CALLS_PER_METHOD,
        "total_judge_calls": EXPECTED_CALLS,
    }
    for field, expected in expected_design.items():
        require_equal(f"design.{field}", integer(design.get(field), f"design.{field}"), expected)
    require_equal("design.update_budgets", tuple(design.get("update_budgets", [])), EXPECTED_UPDATES)
    require_equal("design.partition_seeds count", len(design.get("partition_seeds", [])), len(EXPECTED_PARTITIONS))

    pairing = require_mapping(provenance.get("seed_and_pairing"), "seed_and_pairing")
    require_equal("allowed bootstrap pairing", pairing.get("allowed_bootstrap_pairing_mode"), PAIRING_MODE)
    if bool(pairing.get("sample_level_common_random_numbers_declared")):
        raise ValueError("sample-level pairing is invalid for this cross-model study")
    return methods, cells


def validate_groups(
    groups: Sequence[Mapping[str, Any]],
    methods: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    require_equal("group record count", len(groups), EXPECTED_CALLS)
    method_set = set(methods)
    seen_group_ids: set[str] = set()
    contexts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    layouts: dict[str, dict[str, dict[int, tuple[str, ...]]]] = defaultdict(lambda: defaultdict(dict))
    calls_by_method: Counter[str] = Counter()
    calls_by_shard: Counter[str] = Counter()
    calls_by_shard_round: Counter[tuple[str, int]] = Counter()
    calls_by_method_shard: Counter[tuple[str, str]] = Counter()
    order = require_mapping(provenance.get("request_order"), "request_order")
    sharding_raw = order.get("sharding")
    sharding = (
        require_mapping(sharding_raw, "request_order.sharding")
        if sharding_raw is not None
        else None
    )
    shard_field_presence = [
        SHARD_METADATA_FIELDS.intersection(
            require_mapping(record.get("request_metadata"), f"groups[{index}].request_metadata")
        )
        for index, record in enumerate(groups)
    ]
    any_shard_metadata = any(shard_field_presence)
    all_shard_metadata = all(fields == SHARD_METADATA_FIELDS for fields in shard_field_presence)
    if any_shard_metadata and not all_shard_metadata:
        raise ValueError("shard metadata is only partially present across group records")
    if (sharding is None) != (not any_shard_metadata):
        raise ValueError("request_order.sharding and per-record shard metadata must be present together")
    if sharding is not None:
        require_equal(
            "request-order shard count",
            integer(sharding.get("shards"), "request_order.sharding.shards"),
            EXPECTED_SHARDS,
        )
        require_equal("request-order shard algorithm", sharding.get("algorithm"), EXPECTED_SHARD_ALGORITHM)
    local_protocol_hash = sha256_file(GROUPEVAL_ROOT / "protocol" / "protocol_v1.yaml")

    for sequence_index, record in enumerate(groups):
        metadata = require_mapping(record.get("request_metadata"), f"groups[{sequence_index}].request_metadata")
        group_input = require_mapping(record.get("group_input"), f"groups[{sequence_index}].group_input")
        method = str(metadata.get("method_code", ""))
        partition = str(metadata.get("partition_id", ""))
        group_id = str(metadata.get("group_id", ""))
        if method not in method_set:
            raise ValueError(f"groups[{sequence_index}]: unknown method {method!r}")
        if partition not in EXPECTED_PARTITIONS:
            raise ValueError(f"groups[{sequence_index}]: unexpected partition {partition!r}")
        if not group_id or group_id in seen_group_ids:
            raise ValueError(f"groups[{sequence_index}]: missing/duplicate group_id {group_id!r}")
        seen_group_ids.add(group_id)
        calls_by_method[method] += 1
        require_equal(f"groups[{sequence_index}]:group_input.group_id", group_input.get("group_id"), group_id)
        require_equal(f"groups[{sequence_index}]:target profile", group_input.get("target_profile_id"), "owt1024-fullstream-v1")
        require_equal(f"groups[{sequence_index}]:request sequence", integer(metadata.get("request_sequence_index"), "request sequence"), sequence_index)
        require_equal(f"groups[{sequence_index}]:interleave round", integer(metadata.get("interleave_round"), "interleave round"), sequence_index // EXPECTED_METHODS)
        require_equal(f"groups[{sequence_index}]:interleave position", integer(metadata.get("interleave_position_in_round"), "interleave position"), sequence_index % EXPECTED_METHODS)
        require_equal(
            f"groups[{sequence_index}]:interleave algorithm",
            metadata.get("interleave_algorithm"),
            order.get("algorithm"),
        )
        require_equal(
            f"groups[{sequence_index}]:interleave seed",
            integer(metadata.get("interleave_seed"), "interleave seed"),
            integer(order.get("seed"), "request_order.seed"),
        )
        require_equal(
            f"groups[{sequence_index}]:protocol version",
            metadata.get("protocol_version"),
            EXPECTED_PROTOCOL_VERSION,
        )
        require_equal(
            f"groups[{sequence_index}]:protocol hash",
            metadata.get("protocol_sha256"),
            local_protocol_hash,
        )
        if sharding is not None:
            shard_id = str(metadata.get("shard_id", ""))
            if shard_id not in {f"S{i}" for i in range(1, EXPECTED_SHARDS + 1)}:
                raise ValueError(f"groups[{sequence_index}]: invalid shard_id {shard_id!r}")
            require_equal(
                f"groups[{sequence_index}]:shard count",
                integer(metadata.get("shard_count"), "shard count"),
                EXPECTED_SHARDS,
            )
            require_equal(
                f"groups[{sequence_index}]:shard algorithm",
                metadata.get("shard_algorithm"),
                EXPECTED_SHARD_ALGORITHM,
            )
            require_equal(
                f"groups[{sequence_index}]:shard sequence",
                integer(metadata.get("shard_sequence_index"), "shard sequence"),
                calls_by_shard[shard_id],
            )
            calls_by_shard[shard_id] += 1
            calls_by_shard_round[(shard_id, sequence_index // EXPECTED_METHODS)] += 1
            calls_by_method_shard[(method, shard_id)] += 1
        if require_mapping(metadata.get("private_method_metadata", {}), "private_method_metadata"):
            raise ValueError("paper_groups.jsonl unexpectedly embeds private method labels")

        samples = require_sequence(group_input.get("samples"), f"groups[{sequence_index}].samples")
        sample_map = require_sequence(metadata.get("sample_map"), f"groups[{sequence_index}].sample_map")
        require_equal(f"groups[{sequence_index}]:group size", len(samples), EXPECTED_GROUP_SIZE)
        require_equal(f"groups[{sequence_index}]:sample-map size", len(sample_map), EXPECTED_GROUP_SIZE)
        by_display = {str(item.get("sample_id")): item for item in samples if isinstance(item, Mapping)}
        require_equal(f"groups[{sequence_index}]:display IDs", set(by_display), {f"S{i}" for i in range(1, EXPECTED_GROUP_SIZE + 1)})
        indices: list[str] = []
        for mapping_raw in sample_map:
            mapping = require_mapping(mapping_raw, f"groups[{sequence_index}].sample_map item")
            display_id = str(mapping.get("display_id"))
            sample_index = str(mapping.get("sample_index"))
            if sample_index not in {str(i) for i in range(EXPECTED_SAMPLES)}:
                raise ValueError(f"groups[{sequence_index}]: invalid sample index {sample_index}")
            candidate = require_mapping(by_display.get(display_id), f"groups[{sequence_index}]:{display_id}")
            text = str(candidate.get("text", ""))
            if not text.strip():
                raise ValueError(f"groups[{sequence_index}]: empty candidate {display_id}")
            require_equal(f"groups[{sequence_index}]:{display_id}:text hash", hashlib.sha256(text.encode("utf-8")).hexdigest(), mapping.get("text_sha256"))
            contexts[(method, partition)][sample_index] += 1
            indices.append(sample_index)
        group_index = integer(metadata.get("group_index"), f"groups[{sequence_index}].group_index")
        if group_index in layouts[method][partition]:
            raise ValueError(f"{method}/{partition}: duplicate group index {group_index}")
        layouts[method][partition][group_index] = tuple(sorted(indices, key=int))

    require_equal("calls per method", dict(calls_by_method), {method: EXPECTED_CALLS_PER_METHOD for method in method_set})
    if sharding is not None:
        require_equal(
            "calls per shard",
            dict(calls_by_shard),
            {f"S{i}": EXPECTED_CALLS // EXPECTED_SHARDS for i in range(1, EXPECTED_SHARDS + 1)},
        )
        for shard_id in sorted(calls_by_shard):
            for round_index in range(EXPECTED_CALLS_PER_METHOD):
                require_equal(
                    f"{shard_id}:round {round_index}:calls",
                    calls_by_shard_round[(shard_id, round_index)],
                    EXPECTED_METHODS // EXPECTED_SHARDS,
                )
        for method in sorted(method_set):
            per_shard = [
                calls_by_method_shard[(method, f"S{i}")]
                for i in range(1, EXPECTED_SHARDS + 1)
            ]
            require_equal(f"{method}:per-shard range", (min(per_shard), max(per_shard)), (2, 3))
    all_indices = {str(i) for i in range(EXPECTED_SAMPLES)}
    for method in sorted(method_set):
        for partition in EXPECTED_PARTITIONS:
            require_equal(f"{method}/{partition}:sample set", set(contexts[(method, partition)]), all_indices)
            require_equal(f"{method}/{partition}:context multiplicity", set(contexts[(method, partition)].values()), {1})
            require_equal(f"{method}/{partition}:group count", len(layouts[method][partition]), EXPECTED_GROUPS_PER_PARTITION)
            require_equal(f"{method}/{partition}:group indices", set(layouts[method][partition]), set(range(EXPECTED_GROUPS_PER_PARTITION)))
    reference = layouts[sorted(method_set)[0]]
    for method in sorted(method_set)[1:]:
        require_equal(f"{method}:partition layouts", layouts[method], reference)

    require_equal("request-order rounds", integer(order.get("rounds"), "request_order.rounds"), EXPECTED_CALLS_PER_METHOD)
    for round_index in range(EXPECTED_CALLS_PER_METHOD):
        round_methods = {
            str(groups[index]["request_metadata"]["method_code"])
            for index in range(round_index * EXPECTED_METHODS, (round_index + 1) * EXPECTED_METHODS)
        }
        require_equal(f"request-order round {round_index}", round_methods, method_set)


def validate_judgments(
    groups: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    expected_provider: str,
    expected_model: str,
) -> dict[str, Any]:
    require_equal("judgment count", len(judgments), EXPECTED_CALLS)
    schema_path = GROUPEVAL_ROOT / "schemas" / "absolute_output.schema.json"
    prompt_path = GROUPEVAL_ROOT / "prompts" / "group_eval_absolute.md"
    schema = load_json(schema_path)
    validator = Draft202012Validator(schema)
    dataset_failures = dataset_errors(
        list(enumerate(judgments, 1)),
        list(enumerate(groups, 1)),
    )
    if dataset_failures:
        raise ValueError("dataset validation failed:\n- " + "\n- ".join(dataset_failures))

    expected_ids = [str(row["request_metadata"]["group_id"]) for row in groups]
    observed_ids = [str(row.get("request_metadata", {}).get("group_id", "")) for row in judgments]
    require_equal("judgment request order", observed_ids, expected_ids)
    profile = require_mapping(provenance.get("target_profile"), "target_profile")
    protocol = require_mapping(provenance.get("protocol"), "protocol")
    profile_path = GROUPEVAL_ROOT / "protocol" / "target_profile_owt1024_fullstream.md"
    protocol_path = GROUPEVAL_ROOT / "protocol" / "protocol_v1.yaml"
    profile_hash = sha256_file(profile_path)
    protocol_hash = sha256_file(protocol_path)
    prompt_hash = sha256_file(prompt_path)
    schema_hash = sha256_file(schema_path)
    require_equal("target-profile id", profile.get("id"), EXPECTED_TARGET_PROFILE_ID)
    require_equal("target-profile local/provenance hash", profile.get("sha256"), profile_hash)
    require_equal("protocol version", protocol.get("version"), EXPECTED_PROTOCOL_VERSION)
    require_equal("protocol local/provenance hash", protocol.get("sha256"), protocol_hash)

    providers: set[str] = set()
    models: set[str] = set()
    response_models: set[str] = set()
    fingerprints: set[str] = set()
    finish_reasons: Counter[str] = Counter()
    retries = Counter()
    for index, (group, judgment) in enumerate(zip(groups, judgments, strict=True)):
        expected_metadata = require_mapping(group.get("request_metadata"), f"groups[{index}].request_metadata")
        observed_metadata = require_mapping(judgment.get("request_metadata"), f"judgments[{index}].request_metadata")
        require_equal(f"judgments[{index}]:exact request metadata", canonical_json(observed_metadata), canonical_json(expected_metadata))
        failures = validate_record(dict(judgment), validator)
        if failures:
            raise ValueError(f"judgments[{index}] invalid:\n- " + "\n- ".join(failures))
        run = require_mapping(judgment.get("run_metadata"), f"judgments[{index}].run_metadata")
        provider = str(run.get("judge_provider", ""))
        model = str(run.get("judge_model_snapshot", ""))
        providers.add(provider)
        models.add(model)
        response_model = str(run.get("provider_response_model") or "")
        fingerprint = str(run.get("system_fingerprint") or "")
        if not response_model:
            raise ValueError(f"judgments[{index}]: missing provider_response_model")
        if not fingerprint:
            raise ValueError(f"judgments[{index}]: missing system_fingerprint")
        response_models.add(response_model)
        fingerprints.add(fingerprint)
        finish_reasons[str(run.get("finish_reason") or "not_available")] += 1
        retries["transport"] += integer(run.get("transport_retries", 0), f"judgments[{index}].transport_retries")
        retries["content"] += integer(run.get("content_retries", 0), f"judgments[{index}].content_retries")
        require_equal(f"judgments[{index}]:tools", run.get("tools_enabled"), False)
        require_equal(f"judgments[{index}]:strict schema", run.get("strict_json_schema"), True)
        require_equal(f"judgments[{index}]:local schema", run.get("local_validation_uses_unmodified_schema"), True)
        require_equal(f"judgments[{index}]:prompt hash", run.get("prompt_sha256"), prompt_hash)
        require_equal(f"judgments[{index}]:schema hash", run.get("schema_sha256"), schema_hash)
        require_equal(f"judgments[{index}]:profile hash", run.get("target_profile_sha256"), profile_hash)
    require_equal("judge provider", providers, {expected_provider})
    require_equal("judge model", models, {expected_model})
    require_equal("provider response model", response_models, {expected_model})
    if len(fingerprints) != 1:
        raise ValueError(f"serving fingerprint was not stable within run: {sorted(fingerprints)}")
    return {
        "provider": expected_provider,
        "model": expected_model,
        "provider_response_models": sorted(response_models),
        "system_fingerprints": sorted(fingerprints),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "transport_retries": retries["transport"],
        "content_retries": retries["content"],
        "calls": len(judgments),
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "target_profile_sha256": profile_hash,
        "protocol_sha256": protocol_hash,
    }


def bootstrap_scores(
    sample_rows: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    if replicates < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    method_data = _build_method_data(sample_rows)
    methods = sorted(method_data)
    require_equal("bootstrap method count", len(methods), EXPECTED_METHODS)
    for method, data in method_data.items():
        require_equal(f"{method}:bootstrap samples", len(data["sample_indices"]), EXPECTED_SAMPLES)
        require_equal(f"{method}:bootstrap partitions", tuple(data["partitions"]), EXPECTED_PARTITIONS)
    distributions = {
        method: {
            "geq": np.empty(replicates, dtype=float),
            "ged": np.empty(replicates, dtype=float),
        }
        for method in methods
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        shared_partition_weights = dict(
            zip(EXPECTED_PARTITIONS, rng.exponential(size=len(EXPECTED_PARTITIONS)), strict=True)
        )
        for method in methods:
            data = method_data[method]
            indices = data["sample_indices"]
            independent_sample_weights = dict(
                zip(indices, rng.exponential(size=len(indices)), strict=True)
            )
            geq, ged = _weighted_method_scores(
                data,
                independent_sample_weights,
                shared_partition_weights,
            )
            distributions[method]["geq"][replicate] = geq
            distributions[method]["ged"][replicate] = ged
    return distributions


def method_lookup(methods: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str, int], str]:
    return {
        (
            str(label["model"]),
            str(label["operating_point"]),
            integer(label["configured_updates_M"], f"{code}.updates"),
        ): code
        for code, label in methods.items()
    }


def make_per_output_rows(
    sample_rows: Sequence[Mapping[str, Any]],
    methods: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    contexts: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        contexts[(str(row["method_code"]), str(row["sample_index"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, sample_index), rows in contexts.items():
        require_equal(f"{method}/{sample_index}:contexts", len(rows), len(EXPECTED_PARTITIONS))
        require_equal(f"{method}/{sample_index}:partitions", {str(row["partition_id"]) for row in rows}, set(EXPECTED_PARTITIONS))
        label = methods[method]
        output.append({
            "model": label["model"],
            "operating_point": label["operating_point"],
            "updates_M": integer(label["configured_updates_M"], f"{method}.updates"),
            "method_code": method,
            "sample_index": int(sample_index),
            "contexts": len(rows),
            "geq": statistics.fmean(float(row["quality"]) for row in rows),
            "invalid_context_rate": statistics.fmean(1.0 - float(bool(row["valid"])) for row in rows),
            "severe_degeneration_context_rate": statistics.fmean(float(bool(row["severe_degeneration"])) for row in rows),
            **{
                f"mean_{dimension}": statistics.fmean(float(row[dimension]) for row in rows)
                for dimension in DIMENSIONS
            },
        })
    output.sort(key=lambda row: (str(row["model"]), str(row["operating_point"]), int(row["updates_M"]), int(row["sample_index"])))
    require_equal("per-output rows", len(output), EXPECTED_METHODS * EXPECTED_SAMPLES)
    return output


def make_group_rows(
    group_rows: Sequence[Mapping[str, Any]],
    methods: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in group_rows:
        method = str(row["method_code"])
        label = methods[method]
        output.append({
            "model": label["model"],
            "operating_point": label["operating_point"],
            "updates_M": integer(label["configured_updates_M"], f"{method}.updates"),
            "method_code": method,
            "partition_id": row["partition_id"],
            "group_index": row["group_index"],
            "group_id": row["group_id"],
            "geq": row["geq"],
            "ged": row["ged"],
            "invalid_rate": row["invalid_rate"],
            "severe_degeneration_rate": row["severe_degeneration_rate"],
            "represented_mode_count": row["represented_mode_count"],
            "effective_mode_count": row["effective_mode_count"],
            "dominant_mode_share": row["dominant_mode_share"],
            "clustering_confidence": row["clustering_confidence"],
        })
    output.sort(key=lambda row: (str(row["model"]), str(row["operating_point"]), int(row["updates_M"]), str(row["partition_id"]), int(row["group_index"])))
    require_equal("group-score rows", len(output), EXPECTED_CALLS)
    return output


def make_cell_rows(
    systems: Sequence[Mapping[str, Any]],
    methods: Mapping[str, Mapping[str, Any]],
    cells: Mapping[str, Mapping[str, Any]],
    distributions: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    system_map = {str(row["method_code"]): row for row in systems}
    require_equal("scored method set", set(system_map), set(methods))
    output: list[dict[str, Any]] = []
    for method, label in methods.items():
        system = system_map[method]
        metrics = require_mapping(cells[method].get("source_metrics"), f"{method}.source_metrics")
        geq_ci = percentile_interval(distributions[method]["geq"])
        ged_ci = percentile_interval(distributions[method]["ged"])
        row = {
            "model": label["model"],
            "operating_point": label["operating_point"],
            "updates_M": integer(label["configured_updates_M"], f"{method}.updates"),
            "temperature": finite_float(label["temperature"], f"{method}.temperature"),
            "categorical_precision": label["categorical_precision"],
            "method_code": method,
            "gpt2_large_generative_perplexity": finite_float(metrics["generative_perplexity"], f"{method}.gPPL"),
            "within_sequence_entropy": finite_float(metrics["within_sequence_entropy"], f"{method}.entropy"),
            "geq": float(system["geq"]),
            "geq_ci_low": geq_ci[0],
            "geq_ci_high": geq_ci[1],
            "ged": float(system["ged"]),
            "ged_ci_low": ged_ci[0],
            "ged_ci_high": ged_ci[1],
            "invalid_rate": float(system["invalid_rate"]),
            "severe_degeneration_rate": float(system["severe_degeneration_rate"]),
            "quality_p10": float(system["quality_p10"]),
            "bottom_quartile_quality": float(system["bottom_quartile_quality"]),
            "mean_effective_mode_count": float(system["mean_effective_mode_count"]),
            "mean_dominant_mode_share": float(system["mean_dominant_mode_share"]),
            "ged_ceiling_rate": float(system["ged_ceiling_rate"]),
            "low_clustering_confidence_rate": float(system["low_clustering_confidence_rate"]),
            **{f"mean_{dimension}": float(system[f"mean_{dimension}"]) for dimension in DIMENSIONS},
        }
        output.append(row)
    output.sort(key=lambda row: (int(row["updates_M"]), str(row["operating_point"]), str(row["model"])))
    return output


def make_contrasts(
    cells: Sequence[Mapping[str, Any]],
    methods: Mapping[str, Mapping[str, Any]],
    distributions: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    lookup = method_lookup(methods)
    cell_map = {
        (str(row["model"]), str(row["operating_point"]), int(row["updates_M"])): row
        for row in cells
    }
    output: list[dict[str, Any]] = []
    for operating_point in ("reference", "selected"):
        for updates in EXPECTED_UPDATES:
            md_key = ("MDLM-Small", operating_point, updates)
            di_key = ("DiDi-Instruct", operating_point, updates)
            md_method, di_method = lookup[md_key], lookup[di_key]
            md, di = cell_map[md_key], cell_map[di_key]
            delta_geq_draws = distributions[md_method]["geq"] - distributions[di_method]["geq"]
            delta_ged_draws = distributions[md_method]["ged"] - distributions[di_method]["ged"]
            geq_ci = percentile_interval(delta_geq_draws)
            ged_ci = percentile_interval(delta_ged_draws)
            row: dict[str, Any] = {
                "contrast_direction": "MDLM-Small-minus-DiDi-Instruct",
                "operating_point": operating_point,
                "updates_M": updates,
                "delta_gpt2_large_generative_perplexity": float(md["gpt2_large_generative_perplexity"]) - float(di["gpt2_large_generative_perplexity"]),
                "gpt2_large_perplexity_ratio_mdlm_over_didi": float(md["gpt2_large_generative_perplexity"]) / float(di["gpt2_large_generative_perplexity"]),
                "delta_within_sequence_entropy": float(md["within_sequence_entropy"]) - float(di["within_sequence_entropy"]),
                "delta_geq": float(md["geq"]) - float(di["geq"]),
                "delta_geq_ci_low": geq_ci[0],
                "delta_geq_ci_high": geq_ci[1],
                "bootstrap_probability_delta_geq_gt_0": float(np.mean(delta_geq_draws > 0.0)),
                "delta_ged": float(md["ged"]) - float(di["ged"]),
                "delta_ged_ci_low": ged_ci[0],
                "delta_ged_ci_high": ged_ci[1],
                "bootstrap_probability_delta_ged_gt_0": float(np.mean(delta_ged_draws > 0.0)),
                "delta_invalid_rate": float(md["invalid_rate"]) - float(di["invalid_rate"]),
                "delta_severe_degeneration_rate": float(md["severe_degeneration_rate"]) - float(di["severe_degeneration_rate"]),
                **{
                    f"delta_mean_{dimension}": float(md[f"mean_{dimension}"]) - float(di[f"mean_{dimension}"])
                    for dimension in DIMENSIONS
                },
            }
            output.append(row)
    output.sort(key=lambda row: (int(row["updates_M"]), str(row["operating_point"])))
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: (f"{value:.10g}" if isinstance((value := row.get(field)), float) else value)
                for field in fieldnames
            })


def ci_text(point: float, low: float, high: float) -> str:
    return f"{point:.2f} [{low:.2f}, {high:.2f}]"


def make_markdown(
    cells: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    judge: Mapping[str, Any],
) -> str:
    lines = [
        "# Matched cross-model GroupEval results",
        "",
        f"Validated {judge['calls']} judgments from `{judge['provider']}/{judge['model']}`. "
        f"Intervals are 95% crossed-Bayesian-bootstrap intervals ({bootstrap['replicates']:,} replicates).",
        "",
        "GEQ is higher-is-better judged quality on [0,10]. GED is higher-is-more-valid/balanced local semantic-discourse diversity on [0,10]; it is not a human-calibrated adequacy threshold. Each output's GEQ is averaged over its two independently grouped contexts.",
        "Because GED is a nonlinear plug-in entropy statistic, its weighted Bayesian-bootstrap interval need not be centered on, or contain, the unweighted plug-in point estimate. The released paper table therefore uses GED point estimates directionally and does not treat these intervals as calibrated diversity thresholds.",
        "",
        "## Cell estimates",
        "",
        "| M | Operating point | Model | GPT-2-L gPPL | H | GEQ (95% CI) | GED (95% CI) | Invalid |",
        "|---:|:--|:--|--:|--:|:--|:--|--:|",
    ]
    for row in cells:
        lines.append(
            f"| {row['updates_M']} | {row['operating_point']} | {row['model']} | "
            f"{row['gpt2_large_generative_perplexity']:.2f} | {row['within_sequence_entropy']:.3f} | "
            f"{ci_text(row['geq'], row['geq_ci_low'], row['geq_ci_high'])} | "
            f"{ci_text(row['ged'], row['ged_ci_low'], row['ged_ci_high'])} | {row['invalid_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Matched cross-model contrasts",
        "",
        "All differences below are MDLM-Small minus DiDi-Instruct at the same operating point and M. Positive GEQ/GED values favor MDLM on the named metric.",
        "",
        "| M | Operating point | Delta GEQ (95% CI) | Delta GED (95% CI) | Delta invalid |",
        "|---:|:--|:--|:--|--:|",
    ])
    for row in contrasts:
        lines.append(
            f"| {row['updates_M']} | {row['operating_point']} | "
            f"{ci_text(row['delta_geq'], row['delta_geq_ci_low'], row['delta_geq_ci_high'])} | "
            f"{ci_text(row['delta_ged'], row['delta_ged_ci_low'], row['delta_ged_ci_high'])} | {row['delta_invalid_rate']:.3f} |"
        )
    lines.extend([
        "",
        "## Inference scope",
        "",
        "The cells are matched by operating point and update budget. Shared partition weights block on the two common grouping layouts; sample weights are independent across models because sample index is decoded-list order, not a common-random-number pairing. The bootstrap conditions on the judge's observed mode assignments and does not model judge or latent-clustering uncertainty.",
        "",
    ])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.replicates < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    groups_path = args.groups.resolve()
    provenance_path = args.generation_provenance.resolve()
    private_map_path = args.private_label_map.resolve()
    sums_path = (args.sha256sums or groups_path.parent / "SHA256SUMS").resolve()
    verify_sha256s(sums_path, {
        "paper_groups.jsonl": groups_path,
        "generation_provenance.json": provenance_path,
        "private_label_map.json": private_map_path,
    })
    provenance = load_json(provenance_path)
    private_map = load_json(private_map_path)
    methods, cells = validate_identity(provenance, private_map)
    groups = load_jsonl(groups_path)
    validate_groups(groups, methods, provenance)
    judgments = load_jsonl(args.judgments.resolve())
    judge = validate_judgments(groups, judgments, provenance, args.expected_provider, args.expected_model)

    sample_context_rows, raw_group_rows = flatten_judgments(judgments)
    systems = aggregate_systems(sample_context_rows, raw_group_rows)
    distributions = bootstrap_scores(sample_context_rows, args.replicates, args.seed)
    per_output = make_per_output_rows(sample_context_rows, methods)
    group_scores = make_group_rows(raw_group_rows, methods)
    cell_results = make_cell_rows(systems, methods, cells, distributions)
    contrasts = make_contrasts(cell_results, methods, distributions)

    bootstrap = {
        "method": "crossed_bayesian_bootstrap",
        "replicates": args.replicates,
        "seed": args.seed,
        "confidence_level": 0.95,
        "pairing_mode": PAIRING_MODE,
        "shared_random_weights": "partition contexts only",
        "independent_random_weights": "model/method output samples",
        "mode_assignments": "conditioned on observed judge assignments",
    }
    summary = {
        "artifact": "Matched MDLM-Small/DiDi-Instruct GroupEval analysis",
        "analysis_version": ANALYSIS_VERSION,
        "source_artifact": EXPECTED_ARTIFACT,
        "input_artifacts": {
            "judgments": {"sha256": sha256_file(args.judgments.resolve()), "records": len(judgments)},
            "groups": {"sha256": sha256_file(groups_path), "records": len(groups)},
            "generation_provenance": {"sha256": sha256_file(provenance_path)},
            "private_label_map": {"sha256": sha256_file(private_map_path)},
            "prepared_sha256sums": {"sha256": sha256_file(sums_path)},
        },
        "design": {
            "methods": EXPECTED_METHODS,
            "samples_per_method": EXPECTED_SAMPLES,
            "sample_contexts_per_method": EXPECTED_SAMPLES * len(EXPECTED_PARTITIONS),
            "groups_per_method": EXPECTED_CALLS_PER_METHOD,
            "judge_calls": EXPECTED_CALLS,
            "group_size": EXPECTED_GROUP_SIZE,
        },
        "judge": judge,
        "bootstrap": bootstrap,
        "metric_direction": {
            "geq": "higher is better judged quality",
            "ged": "higher is more valid and balanced local semantic-discourse diversity",
            "contrast": "MDLM-Small minus DiDi-Instruct",
        },
        "source_metric_scope": provenance.get("source_metric_scope"),
        "cell_results": cell_results,
        "matched_cross_model_contrasts": contrasts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_output_geq.csv", per_output)
    write_csv(output_dir / "group_scores.csv", group_scores)
    write_csv(output_dir / "cell_results.csv", cell_results)
    write_csv(output_dir / "matched_cross_model_contrasts.csv", contrasts)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "results.md").write_text(make_markdown(cell_results, contrasts, bootstrap, judge), encoding="utf-8")
    hash_paths = sorted(path for path in output_dir.iterdir() if path.is_file())
    (output_dir / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in hash_paths), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "judge_calls": EXPECTED_CALLS,
        "cell_rows": len(cell_results),
        "contrast_rows": len(contrasts),
        "per_output_rows": len(per_output),
        "group_rows": len(group_scores),
        "bootstrap_replicates": args.replicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--generation-provenance", required=True, type=Path)
    parser.add_argument("--private-label-map", required=True, type=Path)
    parser.add_argument("--sha256sums", type=Path, help="Defaults to SHA256SUMS beside --groups")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-provider", default=EXPECTED_PROVIDER)
    parser.add_argument("--expected-model", default=EXPECTED_MODEL)
    args = parser.parse_args()
    print(json.dumps(analyze(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
