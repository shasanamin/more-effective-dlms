#!/usr/bin/env python3
"""Create blinded GroupEval groups from a sample manifest.

Canonical design: 128 samples per method, groups of 8, and two seeded
overlap-reduced partitions.
The same sample-index groups are used for every method to block on grouping context;
sample-level pairing is used only when common-random-number outputs are explicitly declared.
Within-group display order is randomized independently for each method and partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

PROTOCOL_VERSION = "1.0.0"
DEFAULT_SEEDS = (271828, 314159)
PRIVATE_COLUMNS = (
    "system_label_private",
    "budget",
    "sampler_temperature",
    "categorical_precision",
    "generation_seed",
    "checkpoint_id",
    "sampler_notes",
)


@dataclass(frozen=True)
class Sample:
    method_code: str
    sample_index: str
    sample_id: str
    text: str
    text_sha256: str
    target_profile_id: str
    metadata: dict[str, str]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def read_text(row: dict[str, str], manifest_dir: Path) -> str:
    inline = row.get("text", "")
    path_value = row.get("text_path", "")
    if inline and path_value:
        raise ValueError("manifest row must use either text or text_path, not both")
    if inline:
        return inline
    if not path_value:
        raise ValueError("manifest row requires text or text_path")
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_dir / path
    if not path.is_file():
        raise FileNotFoundError(f"sample text file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_manifest(path: Path) -> dict[str, list[Sample]]:
    by_method: dict[str, list[Sample]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"method_code", "sample_index", "sample_id", "target_profile_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing required columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            method = row.get("method_code", "").strip()
            index = row.get("sample_index", "").strip()
            sample_id = row.get("sample_id", "").strip()
            profile = row.get("target_profile_id", "").strip()
            if not all((method, index, sample_id, profile)):
                raise ValueError(f"{path}:{line_number}: required field is blank")
            text = read_text(row, path.parent)
            if not text.strip():
                raise ValueError(f"{path}:{line_number}: sample text is empty")
            observed_hash = sha256_text(text)
            supplied_hash = row.get("text_sha256", "").strip().lower()
            if supplied_hash and supplied_hash != observed_hash:
                raise ValueError(
                    f"{path}:{line_number}: text_sha256 mismatch: "
                    f"expected {supplied_hash}, observed {observed_hash}"
                )
            metadata = {key: row.get(key, "").strip() for key in PRIVATE_COLUMNS if row.get(key)}
            by_method[method].append(
                Sample(
                    method_code=method,
                    sample_index=index,
                    sample_id=sample_id,
                    text=text,
                    text_sha256=observed_hash,
                    target_profile_id=profile,
                    metadata=metadata,
                )
            )
    if not by_method:
        raise ValueError("manifest contains no samples")
    return dict(by_method)


def natural_index_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def validate_design(
    by_method: dict[str, list[Sample]],
    group_size: int,
    expected_samples: int,
    allow_noncanonical: bool,
) -> list[str]:
    index_sets: dict[str, set[str]] = {}
    profile_ids: set[str] = set()
    for method, samples in by_method.items():
        indices = [sample.sample_index for sample in samples]
        ids = [sample.sample_id for sample in samples]
        if len(indices) != len(set(indices)):
            raise ValueError(f"method {method}: duplicate sample_index values")
        if len(ids) != len(set(ids)):
            raise ValueError(f"method {method}: duplicate sample_id values")
        if len(samples) % group_size:
            raise ValueError(
                f"method {method}: sample count {len(samples)} is not divisible by group size {group_size}"
            )
        if not allow_noncanonical and len(samples) != expected_samples:
            raise ValueError(
                f"method {method}: canonical protocol requires {expected_samples} samples; "
                "use --allow-noncanonical to override"
            )
        index_sets[method] = set(indices)
        profile_ids.update(sample.target_profile_id for sample in samples)
    first_method = next(iter(index_sets))
    reference = index_sets[first_method]
    for method, indices in index_sets.items():
        if indices != reference:
            missing = sorted(reference - indices, key=natural_index_key)
            extra = sorted(indices - reference, key=natural_index_key)
            raise ValueError(
                f"method {method}: sample_index set is not aligned; missing={missing[:8]}, extra={extra[:8]}"
            )
    if len(profile_ids) != 1:
        raise ValueError(f"one grouping run must use one target profile; observed {sorted(profile_ids)}")
    return sorted(reference, key=natural_index_key)


def partition_indices(indices: Sequence[str], group_size: int, seed: int) -> list[list[str]]:
    shuffled = list(indices)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + group_size] for i in range(0, len(shuffled), group_size)]


def pair_set(groups: Iterable[Sequence[str]]) -> set[tuple[str, str]]:
    return {
        tuple(sorted(pair, key=natural_index_key))
        for group in groups
        for pair in combinations(group, 2)
    }


def optimized_second_partition(
    indices: Sequence[str],
    group_size: int,
    seed: int,
    first_groups: Sequence[Sequence[str]],
    candidates: int,
) -> tuple[list[list[str]], int, int]:
    first_pairs = pair_set(first_groups)
    best_groups: list[list[str]] | None = None
    best_overlap: int | None = None
    best_seed = seed
    for candidate in range(max(1, candidates)):
        candidate_seed = stable_seed(seed, candidate)
        groups = partition_indices(indices, group_size, candidate_seed)
        overlap = len(first_pairs & pair_set(groups))
        if best_overlap is None or overlap < best_overlap:
            best_groups, best_overlap, best_seed = groups, overlap, candidate_seed
            if overlap == 0:
                break
    assert best_groups is not None and best_overlap is not None
    return best_groups, best_seed, best_overlap


def build_partitions(
    indices: Sequence[str], group_size: int, seeds: Sequence[int], candidates: int
) -> list[tuple[str, int, int | None, list[list[str]]]]:
    if not seeds:
        raise ValueError("at least one grouping seed is required")
    first = partition_indices(indices, group_size, seeds[0])
    result: list[tuple[str, int, int | None, list[list[str]]]] = [("P1", seeds[0], None, first)]
    prior_pairs = pair_set(first)
    for position, seed in enumerate(seeds[1:], start=2):
        # For v1's canonical two-partition design, minimize overlap with P1. For
        # additional noncanonical partitions, minimize overlap with all earlier pairs.
        best: list[list[str]] | None = None
        best_seed = seed
        best_overlap: int | None = None
        for candidate in range(max(1, candidates)):
            candidate_seed = stable_seed(seed, candidate)
            groups = partition_indices(indices, group_size, candidate_seed)
            overlap = len(prior_pairs & pair_set(groups))
            if best_overlap is None or overlap < best_overlap:
                best, best_seed, best_overlap = groups, candidate_seed, overlap
                if overlap == 0:
                    break
        assert best is not None and best_overlap is not None
        result.append((f"P{position}", best_seed, best_overlap, best))
        prior_pairs |= pair_set(best)
    return result


def opaque_group_id(method: str, partition: str, group_index: int, members: Sequence[str]) -> str:
    payload = json.dumps(
        {"method": method, "partition": partition, "group": group_index, "members": list(members)},
        sort_keys=True,
    )
    return "GE-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_records(
    by_method: dict[str, list[Sample]],
    partitions: Sequence[tuple[str, int, int | None, list[list[str]]]],
    protocol_path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    lookup = {
        method: {sample.sample_index: sample for sample in samples}
        for method, samples in by_method.items()
    }
    for method in sorted(lookup):
        samples_by_index = lookup[method]
        for partition_id, grouping_seed, repeated_pairs, groups in partitions:
            for group_index, member_indices in enumerate(groups):
                order_seed = stable_seed(method, partition_id, group_index, grouping_seed)
                display_members = list(member_indices)
                random.Random(order_seed).shuffle(display_members)
                group_id = opaque_group_id(method, partition_id, group_index, member_indices)
                sample_map: list[dict[str, Any]] = []
                blinded_samples: list[dict[str, str]] = []
                for position, sample_index in enumerate(display_members, start=1):
                    sample = samples_by_index[sample_index]
                    display_id = f"S{position}"
                    sample_map.append(
                        {
                            "display_id": display_id,
                            "sample_id": sample.sample_id,
                            "sample_index": sample.sample_index,
                            "text_sha256": sample.text_sha256,
                        }
                    )
                    blinded_samples.append({"sample_id": display_id, "text": sample.text})
                first_sample = samples_by_index[member_indices[0]]
                metadata: dict[str, Any] = {
                    "protocol_version": PROTOCOL_VERSION,
                    "protocol_sha256": protocol_sha,
                    "method_code": method,
                    "partition_id": partition_id,
                    "group_index": group_index,
                    "group_id": group_id,
                    "grouping_seed": grouping_seed,
                    "order_seed": order_seed,
                    "repeated_pairs_with_earlier_partitions": repeated_pairs,
                    "target_profile_id": first_sample.target_profile_id,
                    "sample_map": sample_map,
                    "private_method_metadata": first_sample.metadata,
                }
                records.append(
                    {
                        "request_metadata": metadata,
                        "group_input": {
                            "protocol_version": PROTOCOL_VERSION,
                            "group_id": group_id,
                            "target_profile_id": first_sample.target_profile_id,
                            "samples": blinded_samples,
                        },
                    }
                )
    return records


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=128)
    parser.add_argument("--partition-seeds", type=parse_seeds, default=list(DEFAULT_SEEDS))
    parser.add_argument("--optimization-candidates", type=int, default=2000)
    parser.add_argument("--allow-noncanonical", action="store_true")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "protocol" / "protocol_v1.yaml",
    )
    args = parser.parse_args()

    if args.group_size < 2 or args.group_size > 16:
        raise ValueError("group size must be between 2 and 16")
    by_method = load_manifest(args.manifest)
    indices = validate_design(
        by_method,
        group_size=args.group_size,
        expected_samples=args.expected_samples,
        allow_noncanonical=args.allow_noncanonical,
    )
    partitions = build_partitions(
        indices,
        group_size=args.group_size,
        seeds=args.partition_seeds,
        candidates=args.optimization_candidates,
    )
    records = build_records(by_method, partitions, args.protocol)
    write_jsonl(args.output, records)

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "methods": sorted(by_method),
        "samples_per_method": len(indices),
        "group_size": args.group_size,
        "partitions": [
            {
                "partition_id": partition_id,
                "seed": seed,
                "repeated_pairs_with_earlier_partitions": overlap,
                "groups": len(groups),
            }
            for partition_id, seed, overlap, groups in partitions
        ],
        "judge_calls": len(records),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
