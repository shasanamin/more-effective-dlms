#!/usr/bin/env python3
"""Prepare the frozen two-operating-point MDLM-Small GroupEval pilot.

This script never generates model outputs and never contacts a judge API. It
verifies the ten archived result/configuration pairs used by the paper, writes
normalized sample text and a blinded manifest outside the repository, records
raw and normalized provenance, and builds the canonical GroupEval groups.

The archived runs have one global Lightning seed and deterministic list order,
but no per-sample seeds. Accordingly, this script leaves ``generation_seed``
blank and does not declare sample-level common-random-number pairing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import grouping


PREPARATION_VERSION = "1.0.0"
TARGET_PROFILE_ID = "owt1024-fullstream-v1"
EOS_LITERAL = "<|endoftext|>"
DOCUMENT_BOUNDARY = "\n\n[DOCUMENT BOUNDARY]\n\n"
EXPECTED_SAMPLE_COUNT = 128
EXPECTED_GLOBAL_SEED = 1
EXPECTED_CHECKPOINT_SHA256 = (
    "74bf9b853206e73b4811d8127e836c2cdcfb7a23f8e04ad7d1356a58f1138d27"
)
EXPECTED_CHECKPOINT_BYTES = 2_715_759_768
GROUP_SIZE = 8
PARTITION_SEEDS = (271828, 314159)
OPTIMIZATION_CANDIDATES = 2000


@dataclass(frozen=True)
class CellSpec:
    updates: int
    temperature: float
    precision: str
    result_sha256: str
    config_sha256: str

    @property
    def tag(self) -> str:
        if self.temperature == 1.0 and self.precision == "fp64":
            return "temp1p0_fp64"
        if self.temperature == 0.9 and self.precision == "fp32":
            return "temp0p9_fp32"
        raise AssertionError("unexpected frozen pilot cell")

    @property
    def result_name(self) -> str:
        return f"mdlm_s{self.updates}_{self.tag}.json"

    @property
    def hydra_dir_name(self) -> str:
        return f"hydra_mdlm_s{self.updates}_{self.tag}"

    @property
    def condition(self) -> str:
        return "unscaled-fp64-reference" if self.temperature == 1.0 else "selected"

    @property
    def method_code(self) -> str:
        payload = (
            f"GroupEval-MDLM-pilot-v1|{self.updates}|"
            f"{self.temperature:.1f}|{self.precision}"
        )
        return "GE-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


CELLS = (
    CellSpec(
        4,
        1.0,
        "fp64",
        "07f8d6b5f055603daa4993715f5d22926e1e0b4aff8357212e7fd24ad0693319",
        "5ce83609fb2f16387f9958b9c31ba9e3eef36d6954a87a7b044bebfffefc3408",
    ),
    CellSpec(
        4,
        0.9,
        "fp32",
        "9c32fb2a13b60e6b4a6fe2f6f31921ef30de56907cb299b3b4d4b96ce8c75502",
        "320ea2d51439747c2b3f08fbd70d827cec8a8e6f455f516757e3da4a56886def",
    ),
    CellSpec(
        16,
        1.0,
        "fp64",
        "863a4fdeb9ad294d9f30e10254c679f5f42f7e1d6507fbacb48fe0d96fb08921",
        "9757c9b799b0e6db14340243dab218c93fe57d9843ed635ec337388440a0be03",
    ),
    CellSpec(
        16,
        0.9,
        "fp32",
        "e04f6abbbebc433d542f67624438713514eefe3611db793ad202976ceaa076f6",
        "9dfeabd6408183129d6dfb6b2d1810e82a074af1cccbfe948bff38e0f2f7912b",
    ),
    CellSpec(
        64,
        1.0,
        "fp64",
        "4cf4a21e18a6b20f8880838c3ee293a142e1a8c0a3bff6a2a94beab16a148676",
        "9ce5460649bd009ff09bdd4be3e69fef24668c3604100c2da807db5316ebd813",
    ),
    CellSpec(
        64,
        0.9,
        "fp32",
        "63833807f9999f29e12f8174db7f05fd14afc6d9d51dc50b3016573f13fdbddb",
        "fa22925b650ea27a0e168f3d20e4bfc7ef8e8ec0b5ccb9467304c069284f6130",
    ),
    CellSpec(
        256,
        1.0,
        "fp64",
        "f1084c409490b553c550bee7a9ea974f38517dbd9001a88f25ec5e9aff66c565",
        "5efa40cc04dfb6146bc2329f429ede8a5723a9c6c7aec93c942b0f1879542d29",
    ),
    CellSpec(
        256,
        0.9,
        "fp32",
        "6b9f0152596db527fe4d2d15787924b379ebf1dad2e1ed9bb174c1567f3dceaa",
        "9aecf162c9152ced0d1775361bfba228e9720744be395436c10d84218d308b40",
    ),
    CellSpec(
        1024,
        1.0,
        "fp64",
        "d575b180506a60f8c3a765a323a4d2d4ffc83a395712a0a35e576833f87b0380",
        "0923f136fae20985d3bad3e375b5554fab3771d069160ed13d770493163f0819",
    ),
    CellSpec(
        1024,
        0.9,
        "fp32",
        "10d29760637eed8d038c30744fd2a8ffc54a6e0cb17d0f287dad6b8847441704",
        "83c4feea3f31164d56dc16a1751ffa0f1e62f589d4d11488b778dbb3994fc335",
    ),
)

EXPECTED_TOP_LEVEL_KEYS = {"metadata", "performance", "samples", "nfe"}
EXPECTED_METADATA_KEYS = {
    "timestamp",
    "model_path",
    "sampler",
    "sample_steps",
    "num_sample_batches",
    "batch_size",
    "total_samples",
    "prompt",
    "semi_ar",
    "stride_length",
    "num_strides",
    "use_float64",
    "gumbel_precision",
}
EXPECTED_PERFORMANCE_KEYS = {
    "generative_perplexity",
    "total_generation_time_seconds",
    "average_time_per_sample",
    "samples_per_second",
    "avg_nfe",
    "avg_entropy",
    "avg_entropy_per_sequence",
}
UNRESOLVED_MASKS = ("[MASK]", "<mask>", "<|mask|>")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_list(values: Iterable[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return sha256_text(payload)


def require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{label}: expected {expected!r}, observed {observed!r}")


def parse_flat_yaml(path: Path) -> dict[str, str]:
    """Read scalar paths needed from Hydra's emitted, non-flow YAML config."""
    paths: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    key_pattern = re.compile(r"^([A-Za-z0-9_]+):(.*)$")
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith(("#", "-")):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if "\t" in raw_line[:indent]:
            raise ValueError(f"{path}:{line_number}: tabs are unsupported")
        match = key_pattern.match(raw_line.strip())
        if not match:
            continue
        key, remainder = match.groups()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        prefix = [item[1] for item in stack]
        value = remainder.strip()
        if value:
            paths[".".join(prefix + [key])] = value
        else:
            stack.append((indent, key))
    return paths


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "yes"}:
        return True
    if normalized in {"false", "no"}:
        return False
    raise ValueError(f"expected YAML boolean, observed {value!r}")


def normalize_sample(raw_text: str) -> tuple[str, bool, int]:
    """Normalize decoded boundary syntax without deleting generated documents."""
    removed_leading_sentinel = raw_text.startswith(EOS_LITERAL)
    body = raw_text[len(EOS_LITERAL) :] if removed_leading_sentinel else raw_text
    later_boundaries = body.count(EOS_LITERAL)
    normalized = body.replace(EOS_LITERAL, DOCUMENT_BOUNDARY)
    return normalized, removed_leading_sentinel, later_boundaries


def verify_config(spec: CellSpec, config_path: Path) -> tuple[dict[str, str], str]:
    observed_hash = sha256_file(config_path)
    require_equal(f"{config_path} SHA256", observed_hash, spec.config_sha256)
    config = parse_flat_yaml(config_path)
    expected_scalars = {
        "mode": "sample_eval",
        "seed": str(EXPECTED_GLOBAL_SEED),
        "loader.batch_size": "4",
        "loader.eval_batch_size": "4",
        "sampling.predictor": "ancestral_cache",
        "sampling.steps": str(spec.updates),
        "sampling.noise_removal": "ancestral",
        "sampling.num_sample_batches": "32",
        "sampling.x0_temperature": f"{spec.temperature:.1f}",
        "sampling.x0_temperature_schedule": "constant",
        "sampling.ancestral_reveal_strategy": "independent",
        "sampling.gumbel_precision": "float64" if spec.precision == "fp64" else "float32",
        "model.length": "1024",
    }
    for key, expected in expected_scalars.items():
        require_equal(f"{config_path}:{key}", config.get(key), expected)
    require_equal(
        f"{config_path}:sampling.use_float64",
        parse_bool(config["sampling.use_float64"]),
        spec.precision == "fp64",
    )
    return config, observed_hash


def verify_result(
    spec: CellSpec, result_path: Path, config: dict[str, str]
) -> tuple[dict[str, Any], str]:
    observed_hash = sha256_file(result_path)
    require_equal(f"{result_path} SHA256", observed_hash, spec.result_sha256)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{result_path}: root must be an object")
    require_equal(f"{result_path}:top-level keys", set(data), EXPECTED_TOP_LEVEL_KEYS)
    metadata = data["metadata"]
    performance = data["performance"]
    samples = data["samples"]
    nfe = data["nfe"]
    require_equal(f"{result_path}:metadata keys", set(metadata), EXPECTED_METADATA_KEYS)
    require_equal(
        f"{result_path}:performance keys", set(performance), EXPECTED_PERFORMANCE_KEYS
    )
    require_equal(f"{result_path}:sample_steps", metadata["sample_steps"], spec.updates)
    require_equal(
        f"{result_path}:num_sample_batches", metadata["num_sample_batches"], 32
    )
    require_equal(f"{result_path}:batch_size", metadata["batch_size"], 4)
    require_equal(
        f"{result_path}:total_samples", metadata["total_samples"], EXPECTED_SAMPLE_COUNT
    )
    require_equal(f"{result_path}:sampler", metadata["sampler"], "ancestral_cache")
    require_equal(f"{result_path}:prompt", metadata["prompt"], None)
    require_equal(f"{result_path}:semi_ar", metadata["semi_ar"], False)
    require_equal(f"{result_path}:stride_length", metadata["stride_length"], None)
    require_equal(f"{result_path}:num_strides", metadata["num_strides"], None)
    require_equal(
        f"{result_path}:use_float64",
        metadata["use_float64"],
        spec.precision == "fp64",
    )
    require_equal(
        f"{result_path}:gumbel_precision",
        metadata["gumbel_precision"],
        "float64" if spec.precision == "fp64" else "float32",
    )
    require_equal(
        f"{result_path}:model path",
        metadata["model_path"],
        config["eval.checkpoint_path"],
    )
    require_equal(f"{result_path}:samples length", len(samples), EXPECTED_SAMPLE_COUNT)
    require_equal(f"{result_path}:nfe length", len(nfe), EXPECTED_SAMPLE_COUNT)
    if not all(isinstance(text, str) and text for text in samples):
        raise ValueError(f"{result_path}: every sample must be a nonempty string")
    if len(set(samples)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"{result_path}: exact duplicate samples detected")
    if any(token in text for text in samples for token in UNRESOLVED_MASKS):
        raise ValueError(f"{result_path}: unresolved mask token detected")
    if any(DOCUMENT_BOUNDARY.strip() in text for text in samples):
        raise ValueError(
            f"{result_path}: raw text collides with the presentation boundary marker"
        )
    if any(value != spec.updates for value in nfe):
        raise ValueError(f"{result_path}: NFE entries do not all equal {spec.updates}")
    require_equal(
        f"{result_path}:avg_nfe", float(performance["avg_nfe"]), float(spec.updates)
    )
    return data, observed_hash


def write_json(path: Path, value: Any, private: bool = False) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if private:
        path.chmod(0o600)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a fresh scratch path"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    samples_root = output_dir / "samples"
    samples_root.mkdir()
    samples_root.chmod(0o700)

    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "protocol" / "protocol_v1.yaml"
    profile_path = root / "protocol" / "target_profile_owt1024_fullstream.md"
    if not profile_path.is_file():
        raise FileNotFoundError(f"target profile not found: {profile_path}")

    manifest_rows: list[dict[str, str]] = []
    cell_provenance: list[dict[str, Any]] = []
    private_methods: dict[str, Any] = {}
    checkpoint_paths: set[str] = set()

    for spec in CELLS:
        result_path = source_root / spec.result_name
        config_path = (
            source_root / spec.hydra_dir_name / ".hydra" / "config.yaml"
        )
        if not result_path.is_file():
            raise FileNotFoundError(f"result file not found: {result_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Hydra config not found: {config_path}")
        config, config_hash = verify_config(spec, config_path)
        result, result_hash = verify_result(spec, result_path, config)
        checkpoint_paths.add(config["eval.checkpoint_path"])

        method_dir = samples_root / spec.method_code
        method_dir.mkdir()
        method_dir.chmod(0o700)
        raw_hashes: list[str] = []
        normalized_hashes: list[str] = []
        sample_records: list[dict[str, Any]] = []
        leading_sentinels_removed = 0
        document_boundaries = 0
        for index, raw_text in enumerate(result["samples"]):
            normalized, removed_sentinel, boundary_count = normalize_sample(raw_text)
            if not normalized:
                raise ValueError(
                    f"{result_path}: sample {index} became empty after sentinel removal"
                )
            raw_hash = sha256_text(raw_text)
            normalized_hash = sha256_text(normalized)
            raw_hashes.append(raw_hash)
            normalized_hashes.append(normalized_hash)
            leading_sentinels_removed += int(removed_sentinel)
            document_boundaries += boundary_count
            sample_id = f"{spec.method_code}-{index:03d}"
            relative_text_path = Path("samples") / spec.method_code / f"{index:03d}.txt"
            text_path = output_dir / relative_text_path
            text_path.write_text(normalized, encoding="utf-8")
            text_path.chmod(0o600)
            manifest_rows.append(
                {
                    "method_code": spec.method_code,
                    "sample_index": str(index),
                    "sample_id": sample_id,
                    "target_profile_id": TARGET_PROFILE_ID,
                    "text_path": relative_text_path.as_posix(),
                    "text_sha256": normalized_hash,
                    "generation_seed": "",
                }
            )
            sample_records.append(
                {
                    "sample_index": index,
                    "sample_id": sample_id,
                    "source_locator": f"{spec.result_name}#/samples/{index}",
                    "raw_text_sha256": raw_hash,
                    "normalized_text_sha256": normalized_hash,
                    "normalized_text_path": relative_text_path.as_posix(),
                    "leading_sentinel_removed": removed_sentinel,
                    "document_boundaries_rendered": boundary_count,
                }
            )

        private_methods[spec.method_code] = {
            "model": "public MDLM-Small",
            "configured_updates_M": spec.updates,
            "sampler": "ancestral_cache",
            "temperature": spec.temperature,
            "categorical_precision": spec.precision,
            "condition": spec.condition,
            "source_result": str(result_path),
            "source_config": str(config_path),
        }
        cell_provenance.append(
            {
                "method_code": spec.method_code,
                "configured_updates_M": spec.updates,
                "temperature": spec.temperature,
                "categorical_precision": spec.precision,
                "global_lightning_seed": EXPECTED_GLOBAL_SEED,
                "per_sample_seed_available": False,
                "sample_index_semantics": (
                    "deterministic decoded-list order 0..127; not a per-sample RNG seed"
                ),
                "result_file": {
                    "path": str(result_path),
                    "sha256": result_hash,
                    "bytes": result_path.stat().st_size,
                },
                "hydra_config": {
                    "path": str(config_path),
                    "sha256": config_hash,
                    "bytes": config_path.stat().st_size,
                },
                "source_timestamp": result["metadata"]["timestamp"],
                "sample_count": EXPECTED_SAMPLE_COUNT,
                "raw_sample_hash_list_sha256": hash_list(raw_hashes),
                "normalized_sample_hash_list_sha256": hash_list(normalized_hashes),
                "leading_sentinels_removed": leading_sentinels_removed,
                "document_boundaries_rendered": document_boundaries,
                "performance_for_cross_check_only": result["performance"],
                "samples": sample_records,
            }
        )

    if len(checkpoint_paths) != 1:
        raise ValueError(f"cells reference multiple checkpoints: {sorted(checkpoint_paths)}")
    configured_checkpoint = Path(next(iter(checkpoint_paths)))
    checkpoint_path = (
        args.checkpoint.resolve() if args.checkpoint is not None else configured_checkpoint
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    require_equal(
        f"{checkpoint_path}:bytes", checkpoint_path.stat().st_size, EXPECTED_CHECKPOINT_BYTES
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    require_equal(
        f"{checkpoint_path}:SHA256", checkpoint_hash, EXPECTED_CHECKPOINT_SHA256
    )

    manifest_path = output_dir / "paper_manifest.csv"
    fieldnames = [
        "method_code",
        "sample_index",
        "sample_id",
        "target_profile_id",
        "text_path",
        "text_sha256",
        "generation_seed",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_path.chmod(0o600)

    private_map_path = output_dir / "private_label_map.json"
    write_json(
        private_map_path,
        {
            "artifact": "GroupEval MDLM-Small pilot private method map",
            "preparation_version": PREPARATION_VERSION,
            "target_profile_id": TARGET_PROFILE_ID,
            "keep_private_until_judging_complete": True,
            "methods": private_methods,
        },
        private=True,
    )

    provenance_path = output_dir / "generation_provenance.json"
    provenance = {
        "artifact": "GroupEval MDLM-Small two-operating-point pilot",
        "preparation_version": PREPARATION_VERSION,
        "target_profile": {
            "id": TARGET_PROFILE_ID,
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
        },
        "source_root": str(source_root),
        "checkpoint": {
            "configured_path": str(configured_checkpoint),
            "verified_path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "bytes": checkpoint_path.stat().st_size,
        },
        "normalization": {
            "leading_rule": (
                "remove one leading <|endoftext|> only when it occurs at character 0"
            ),
            "finite_step_bos_failures": (
                "if the decoded output does not begin with the sentinel, preserve "
                "position-zero text without correction or postselection"
            ),
            "later_literal": EOS_LITERAL,
            "later_replacement": DOCUMENT_BOUNDARY,
            "truncation": False,
            "whitespace_stripping": False,
        },
        "seed_and_pairing": {
            "global_lightning_seed": EXPECTED_GLOBAL_SEED,
            "per_sample_seeds_available": False,
            "sample_indices": "decoded-list positions 0..127",
            "sample_level_common_random_numbers_declared": False,
            "allowed_bootstrap_pairing_mode": "partitions",
        },
        "archive_limitations": {
            "code_revision_recorded_by_run": False,
            "generated_token_ids_archived": False,
            "available_identity_evidence": (
                "exact result, Hydra-config, checkpoint, raw-text, and normalized-text "
                "SHA256 hashes"
            ),
            "evaluation_scope": (
                "GroupEval judges the complete decoded stream; the archived gPPL "
                "implementation masks tokens after its first non-leading EOS, so the "
                "two diagnostics are complementary rather than span-identical"
            ),
        },
        "design": {
            "methods": len(CELLS),
            "samples_per_method": EXPECTED_SAMPLE_COUNT,
            "total_samples": len(manifest_rows),
            "group_size": GROUP_SIZE,
            "partition_seeds": list(PARTITION_SEEDS),
        },
        "cells": cell_provenance,
    }
    write_json(provenance_path, provenance, private=True)

    by_method = grouping.load_manifest(manifest_path)
    indices = grouping.validate_design(
        by_method,
        group_size=GROUP_SIZE,
        expected_samples=EXPECTED_SAMPLE_COUNT,
        allow_noncanonical=False,
    )
    partitions = grouping.build_partitions(
        indices,
        group_size=GROUP_SIZE,
        seeds=PARTITION_SEEDS,
        candidates=OPTIMIZATION_CANDIDATES,
    )
    group_records = grouping.build_records(by_method, partitions, protocol_path)
    groups_path = output_dir / "paper_groups.jsonl"
    grouping.write_jsonl(groups_path, group_records)
    groups_path.chmod(0o600)

    grouping_summary = {
        "protocol_version": grouping.PROTOCOL_VERSION,
        "target_profile_id": TARGET_PROFILE_ID,
        "methods": sorted(by_method),
        "samples_per_method": len(indices),
        "group_size": GROUP_SIZE,
        "partitions": [
            {
                "partition_id": partition_id,
                "seed": seed,
                "repeated_pairs_with_earlier_partitions": overlap,
                "groups_per_method": len(groups),
            }
            for partition_id, seed, overlap, groups in partitions
        ],
        "judge_calls": len(group_records),
        "groups_path": str(groups_path),
    }
    summary_path = output_dir / "grouping_summary.json"
    write_json(summary_path, grouping_summary)

    top_level_hashes = {
        path.name: sha256_file(path)
        for path in (
            manifest_path,
            private_map_path,
            provenance_path,
            groups_path,
            summary_path,
        )
    }
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(top_level_hashes.items())),
        encoding="utf-8",
    )

    return {
        **grouping_summary,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "private_label_map": str(private_map_path),
        "provenance": str(provenance_path),
        "sha256sums": str(sums_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Directory containing the frozen sahoo_official_apr30 JSON/Hydra outputs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Fresh scratch directory for normalized private pilot artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional relocated mdlm.ckpt; otherwise use the path frozen in Hydra.",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
