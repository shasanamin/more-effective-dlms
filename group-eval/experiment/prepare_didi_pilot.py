#!/usr/bin/env python3
"""Prepare the frozen DiDi-Instruct GroupEval operating-point pilot.

This script performs no generation and makes no judge/API calls. It verifies
the exact ten reward-guided DiDi-Instruct artifacts used by the paper, their
full-grid manifest rows, run logs, and checkpoint before writing any output.
It then normalizes the complete decoded streams under
``owt1024-fullstream-v1`` and creates blinded GroupEval groups.

The archived DiDi grid contains 40 samples per cell rather than GroupEval's
canonical 128. The resulting design is therefore an explicitly noncanonical
exploratory extension: groups of eight, two partitions, five groups per
partition and cell, and 100 judge calls in total.

Raw generations, normalized samples, method labels, and provenance are
private. The output directory must be supplied by the caller and must be
outside this project tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import grouping


PREPARATION_VERSION = "1.0.0"
TARGET_PROFILE_ID = "owt1024-fullstream-v1"
EOS_LITERAL = "<|endoftext|>"
DOCUMENT_BOUNDARY = "\n\n[DOCUMENT BOUNDARY]\n\n"
EXPECTED_SAMPLE_COUNT = 40
CANONICAL_SAMPLE_COUNT = 128
EXPECTED_GLOBAL_SEED = 1
EXPECTED_BATCH_SIZE = 8
EXPECTED_NUM_BATCHES = 5
EXPECTED_SEQUENCE_LENGTH = 1024
EXPECTED_EVALUATOR = "gpt2-large"
EXPECTED_MANIFEST_ROWS = 308
EXPECTED_MANIFEST_SHA256 = (
    "7db8837b4a3d5ce7253ffc8ed00a49c9ae1e4b5b13cf0e882672121eaa697eeb"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "e7f41f80b3a1700dcc46f6920a0bec616a55753e216cf6c3bc8985335b51dd82"
)
EXPECTED_CHECKPOINT_BYTES = 4_168_811_487
GROUP_SIZE = 8
PARTITION_SEEDS = (271828, 314159)
OPTIMIZATION_CANDIDATES = 2000
EXPECTED_RESULT_KEYS = {"generative_ppl", "entropy", "generated_seqs"}
UNRESOLVED_MASKS = ("[MASK]", "<mask>", "<|mask|>")
EXPECTED_MANIFEST_COLUMNS = (
    "job_id",
    "gpu",
    "sampler",
    "guided_variant",
    "use_float64",
    "precision",
    "temperature",
    "steps",
    "eval_bs",
    "num_batches",
    "cost",
    "json",
    "log",
)


@dataclass(frozen=True)
class CellSpec:
    updates: int
    temperature: float
    precision: str
    result_sha256: str
    log_sha256: str
    generative_ppl: float
    entropy: float

    @property
    def condition(self) -> str:
        if self.temperature == 1.0 and self.precision == "fp64":
            return "unscaled-fp64-reference"
        if self.temperature == 0.9 and self.precision == "fp32":
            return "selected"
        raise AssertionError("unexpected frozen DiDi pilot cell")

    @property
    def use_float64(self) -> bool:
        return self.precision == "fp64"

    @property
    def temperature_tag(self) -> str:
        return "1p0" if self.temperature == 1.0 else "0p9"

    @property
    def job_id(self) -> str:
        return (
            "guided_current_ema_"
            f"t{self.temperature_tag}_{self.precision}_s{self.updates}"
        )

    @property
    def result_name(self) -> str:
        return f"{self.job_id}.json"

    @property
    def log_name(self) -> str:
        return f"{self.job_id}.log"

    @property
    def method_code(self) -> str:
        payload = (
            "GroupEval-DiDi-pilot-v1|guided-current-ema|"
            f"{self.updates}|{self.temperature:.1f}|{self.precision}"
        )
        return "GE-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


CELLS = (
    CellSpec(
        4,
        1.0,
        "fp64",
        "5e2cd5767a3ac46b7903a8481a8755aeef0002414837482053b51df1672186e3",
        "92d5c4b0899b3597b7d45096a1c61cad60d6b1381fc6efa2ac825a33e270bc30",
        523.5781860351562,
        5.563353061676025,
    ),
    CellSpec(
        4,
        0.9,
        "fp32",
        "0ec7b074d6570f1ee61e98a31c9b84e90c764f91714f9bfc371520be906052aa",
        "fbad25ac5ad1febaad97f23b6077a1265afde8c8cb65a78a30a2f2a38d0a85ec",
        220.06297302246094,
        5.117529392242432,
    ),
    CellSpec(
        16,
        1.0,
        "fp64",
        "1d2521f0690e7623a0b277f8018cefc233c3632879f2389a05e5fa09a6daac91",
        "86402de8dd39bc243410d7a60acfcdb90181c556909fe2971ee3302a445a8a5f",
        71.55656433105469,
        5.430166721343994,
    ),
    CellSpec(
        16,
        0.9,
        "fp32",
        "66ee3cb2f680d2e2d88d335378e15982f732d590927e4929c1cbf51ca6b21675",
        "40000fcc3dd1929bce3ef95db556ff96e98bd9e81501c92830532c6b4d996532",
        40.66252899169922,
        5.173752784729004,
    ),
    CellSpec(
        64,
        1.0,
        "fp64",
        "f5390c870b9fa13967ebacdf41343f2c23daa709a460a9986fc3900baa6e8027",
        "6bcd60201fb4b88d0c4e9dcea16d5ca70e16ce0710d9ccdd05b058d0d1f2e3cc",
        37.28147888183594,
        5.370088577270508,
    ),
    CellSpec(
        64,
        0.9,
        "fp32",
        "ea1fe8d6512c799f188ef12c4c4be48d48a6cc23ffd798262fc13d7afecc9177",
        "ec31c087c787de63e46ffafb0578800784f8596dcafc791ee07af913ec3146c9",
        20.74135398864746,
        5.067018985748291,
    ),
    CellSpec(
        256,
        1.0,
        "fp64",
        "b06583c645a4df0ed1a091892548c38250519a85f9d96032821ff69a791c7f8b",
        "45134f935489077290a715e4d065a99713eeaa50949e4d1c48473162a19048df",
        30.932842254638672,
        5.333018779754639,
    ),
    CellSpec(
        256,
        0.9,
        "fp32",
        "2a4ac2bb41efb0d811a37d3fa4304094630f4c176026ed331fc1809a50d2c178",
        "ab08c9709d838517a906970273e2fa1557004f12715782cd2b9d6d898e95a95e",
        17.0389461517334,
        5.031113624572754,
    ),
    CellSpec(
        1024,
        1.0,
        "fp64",
        "95eafe0f0ff0e03bba39ac3adda8fd1683318a7890c540a8c3e6bc967cda6ead",
        "7eea63d60cce7b66334b23882c743efa0ae724c731af010374e55f45fd72cfc6",
        30.05565643310547,
        5.307525634765625,
    ),
    CellSpec(
        1024,
        0.9,
        "fp32",
        "d93de198e71aed8ee4cbff2f633d84aede1b659d51495df9cfaa9285a3bbdb69",
        "ae913c4e08ce6fd6f39597d55fd647c0ed26d77c3bfa8d72ccd66d3c2e34cd26",
        15.075967788696289,
        4.948769569396973,
    ),
)


@dataclass(frozen=True)
class VerifiedCell:
    spec: CellSpec
    result_path: Path
    result_sha256: str
    result: Mapping[str, Any]
    log_path: Path
    log_sha256: str
    log_metadata: Mapping[str, Any]
    archived_manifest_row: Mapping[str, str]


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


def require_finite_number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{label}: expected a JSON number, observed {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number, observed {result!r}")
    return result


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def normalize_sample(raw_text: str) -> tuple[str, bool, int]:
    """Render decoded sentinel syntax without deleting generated documents."""
    removed_leading_sentinel = raw_text.startswith(EOS_LITERAL)
    body = raw_text[len(EOS_LITERAL) :] if removed_leading_sentinel else raw_text
    later_boundaries = body.count(EOS_LITERAL)
    normalized = body.replace(EOS_LITERAL, DOCUMENT_BOUNDARY)
    return normalized, removed_leading_sentinel, later_boundaries


def parse_rich_config(log_text: str, log_path: Path) -> dict[str, str]:
    """Parse the scalar fields needed from DiDi's Rich-rendered config tree."""
    values: dict[str, str] = {}
    section: str | None = None
    in_config = False
    top_level = re.compile(r"^[├└]── ([A-Za-z0-9_]+)\s*$")
    scalar_key = re.compile(r"^[A-Za-z0-9_]+$")

    for raw_line in log_text.splitlines():
        if raw_line == "CONFIG":
            in_config = True
            continue
        if not in_config:
            continue
        match = top_level.match(raw_line)
        if match:
            section = match.group(1)
            continue
        if section is None:
            continue

        cleaned = re.sub(r"^(?:│   |    )", "", raw_line)
        cleaned = re.sub(r"^└──\s*", "", cleaned).strip()
        if not cleaned:
            continue
        if ":" in cleaned:
            key, value = cleaned.split(":", 1)
            key = key.strip()
            value = value.strip()
            if scalar_key.fullmatch(key) and value:
                # The printed tree contains nested mappings with repeated leaf
                # names (for example algo.name and algo.lr_scheduler.name).
                # The first occurrence is the section's top-level scalar.
                values.setdefault(f"{section}.{key}", value)
        elif section in {"mode", "seed"}:
            values.setdefault(section, cleaned)

    if not in_config:
        raise ValueError(f"{log_path}: missing CONFIG tree")
    return values


def verify_result(spec: CellSpec, path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen result not found: {path}")
    observed_hash = sha256_file(path)
    require_equal(f"{path}:SHA256", observed_hash, spec.result_sha256)

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be an object")
    require_equal(f"{path}:top-level keys", set(data), EXPECTED_RESULT_KEYS)

    generative_ppl = require_finite_number(
        f"{path}:generative_ppl", data["generative_ppl"]
    )
    entropy = require_finite_number(f"{path}:entropy", data["entropy"])
    require_equal(f"{path}:generative_ppl", generative_ppl, spec.generative_ppl)
    require_equal(f"{path}:entropy", entropy, spec.entropy)

    samples = data["generated_seqs"]
    if not isinstance(samples, list):
        raise ValueError(f"{path}:generated_seqs must be an array")
    require_equal(f"{path}:sample count", len(samples), EXPECTED_SAMPLE_COUNT)
    if not all(isinstance(text, str) and text.strip() for text in samples):
        raise ValueError(f"{path}: every generated sequence must be a nonempty string")
    if len(set(samples)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"{path}: exact duplicate generated sequences detected")
    if any(token in text for text in samples for token in UNRESOLVED_MASKS):
        raise ValueError(f"{path}: unresolved mask token detected")
    if any(DOCUMENT_BOUNDARY.strip() in text for text in samples):
        raise ValueError(
            f"{path}: raw text collides with the presentation boundary marker"
        )
    return data, observed_hash


def verify_log(
    spec: CellSpec,
    path: Path,
    recorded_source_root: Path,
    recorded_checkpoint_path: Path,
) -> tuple[dict[str, Any], str, Path]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen run log not found: {path}")
    observed_hash = sha256_file(path)
    require_equal(f"{path}:SHA256", observed_hash, spec.log_sha256)
    text = path.read_text(encoding="utf-8")
    config = parse_rich_config(text, path)

    expected_config = {
        "mode": "sample_eval",
        "seed": str(EXPECTED_GLOBAL_SEED),
        "loader.batch_size": str(EXPECTED_BATCH_SIZE),
        "loader.eval_batch_size": str(EXPECTED_BATCH_SIZE),
        "sampling.predictor": "guided",
        "sampling.steps": str(spec.updates),
        "sampling.noise_removal": "ancestral",
        "sampling.use_float64": str(spec.use_float64).lower(),
        "sampling.temperature": f"{spec.temperature:.1f}",
        "sampling.use_student_ema": "true",
        "sampling.guided_variant": "current",
        "sampling.p_nucleus": "1.0",
        "sampling.num_sample_batches": str(EXPECTED_NUM_BATCHES),
        "eval.compute_generative_perplexity": "true",
        "eval.gen_ppl_eval_model_name_or_path": EXPECTED_EVALUATOR,
        "eval.generate_samples": "true",
        "model.length": str(EXPECTED_SEQUENCE_LENGTH),
        "algo.name": "didi_instruct",
    }
    for key, expected in expected_config.items():
        require_equal(f"{path}:{key}", config.get(key), expected)

    require_equal(
        f"{path}:`Seed set` count", text.count("Seed set to 1"), 1
    )
    require_equal(
        f"{path}:sample-eval start count", text.count("Starting Sample Eval."), 1
    )

    checkpoint_values = {
        item.strip()
        for item in re.findall(r"(?m)^  --> (.+?)\s*$", text)
    }
    require_equal(
        f"{path}:checkpoint banner paths",
        checkpoint_values,
        {str(recorded_checkpoint_path)},
    )

    saved_paths = re.findall(r"(?m)^Samples saved at: (.+?)\s*$", text)
    require_equal(
        f"{path}:saved-result paths",
        saved_paths,
        [str(recorded_source_root / spec.result_name)],
    )

    ppl_values = re.findall(r"(?m)^Generative perplexity: ([^\s]+)\s*$", text)
    entropy_values = re.findall(r"(?m)^Sample entropy: ([^\s]+)\s*$", text)
    require_equal(f"{path}:gPPL report count", len(ppl_values), 1)
    require_equal(f"{path}:entropy report count", len(entropy_values), 1)
    require_equal(
        f"{path}:reported gPPL", float(ppl_values[0]), spec.generative_ppl
    )
    require_equal(
        f"{path}:reported entropy", float(entropy_values[0]), spec.entropy
    )

    starts = re.findall(
        r"(?m)^\[([0-9-]+ [0-9:,]+)\]\[__main__\]\[INFO\] - "
        r"Starting Sample Eval\.$",
        text,
    )
    require_equal(f"{path}:timestamp count", len(starts), 1)

    verified = {
        "source_timestamp": starts[0],
        "config": {key: config[key] for key in expected_config},
        "reported_generative_ppl": float(ppl_values[0]),
        "reported_entropy": float(entropy_values[0]),
        "archived_checkpoint_path": str(recorded_checkpoint_path),
        "archived_result_path": str(recorded_source_root / spec.result_name),
    }
    return verified, observed_hash, recorded_checkpoint_path


def verify_full_grid_manifest(
    source_root: Path, recorded_source_root: Path
) -> tuple[Path, str, dict[str, dict[str, str]]]:
    path = source_root / "manifest.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"frozen full-grid manifest not found: {path}")
    observed_hash = sha256_file(path)
    require_equal(f"{path}:SHA256", observed_hash, EXPECTED_MANIFEST_SHA256)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_equal(
            f"{path}:columns", tuple(reader.fieldnames or ()), EXPECTED_MANIFEST_COLUMNS
        )
        rows = list(reader)
    require_equal(f"{path}:row count", len(rows), EXPECTED_MANIFEST_ROWS)

    job_ids = [row["job_id"] for row in rows]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError(f"{path}: duplicate job_id values")

    rows_by_job = {row["job_id"]: row for row in rows}
    target_rows: dict[str, dict[str, str]] = {}
    for spec in CELLS:
        if spec.job_id not in rows_by_job:
            raise ValueError(f"{path}: missing target row {spec.job_id}")
        row = rows_by_job[spec.job_id]
        expected = {
            "job_id": spec.job_id,
            "sampler": "guided",
            "guided_variant": "current",
            "use_float64": str(spec.use_float64).lower(),
            "precision": spec.precision,
            "temperature": f"{spec.temperature:.1f}",
            "steps": str(spec.updates),
            "eval_bs": str(EXPECTED_BATCH_SIZE),
            "num_batches": str(EXPECTED_NUM_BATCHES),
            "cost": f"{3.0 * spec.updates:.1f}",
            "json": str(recorded_source_root / spec.result_name),
            "log": str(recorded_source_root / "logs" / spec.log_name),
        }
        for key, value in expected.items():
            require_equal(f"{path}:{spec.job_id}:{key}", row.get(key), value)
        target_rows[spec.job_id] = dict(row)
    return path, observed_hash, target_rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def validate_inputs(
    source_root: Path,
    checkpoint_path: Path,
    recorded_source_root: Path,
    recorded_checkpoint_path: Path,
) -> tuple[
    list[VerifiedCell],
    dict[str, Any],
]:
    manifest_path, manifest_hash, target_rows = verify_full_grid_manifest(
        source_root, recorded_source_root
    )
    verified_cells: list[VerifiedCell] = []
    checkpoint_paths: set[Path] = set()

    for spec in CELLS:
        result_path = source_root / spec.result_name
        log_path = source_root / "logs" / spec.log_name
        result, result_hash = verify_result(spec, result_path)
        log_metadata, log_hash, recovered_checkpoint_path = verify_log(
            spec,
            log_path,
            recorded_source_root,
            recorded_checkpoint_path,
        )
        checkpoint_paths.add(recovered_checkpoint_path)
        require_equal(
            f"{spec.job_id}:result/log gPPL",
            float(result["generative_ppl"]),
            log_metadata["reported_generative_ppl"],
        )
        require_equal(
            f"{spec.job_id}:result/log entropy",
            float(result["entropy"]),
            log_metadata["reported_entropy"],
        )
        verified_cells.append(
            VerifiedCell(
                spec=spec,
                result_path=result_path,
                result_sha256=result_hash,
                result=result,
                log_path=log_path,
                log_sha256=log_hash,
                log_metadata=log_metadata,
                archived_manifest_row=target_rows[spec.job_id],
            )
        )

    require_equal(
        "checkpoint paths recovered from logs",
        checkpoint_paths,
        {recorded_checkpoint_path},
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}"
        )
    require_equal(
        f"{checkpoint_path}:bytes",
        checkpoint_path.stat().st_size,
        EXPECTED_CHECKPOINT_BYTES,
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    require_equal(
        f"{checkpoint_path}:SHA256",
        checkpoint_hash,
        EXPECTED_CHECKPOINT_SHA256,
    )

    shared = {
        "full_grid_manifest": {
            "verified_path": str(manifest_path),
            "archived_path": str(recorded_source_root / "manifest.tsv"),
            "sha256": manifest_hash,
            "bytes": manifest_path.stat().st_size,
            "rows": EXPECTED_MANIFEST_ROWS,
            "target_rows_verified": len(target_rows),
        },
        "checkpoint": {
            "archived_path": str(recorded_checkpoint_path),
            "verified_path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "bytes": checkpoint_path.stat().st_size,
        },
    }
    return verified_cells, shared


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    recorded_source_root = args.recorded_source_root
    recorded_checkpoint_path = args.recorded_checkpoint_path
    output_dir = args.output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if not recorded_source_root.is_absolute():
        raise ValueError(
            "--recorded-source-root must be the absolute path literal stored "
            "in the frozen manifest and logs"
        )
    if not recorded_checkpoint_path.is_absolute():
        raise ValueError(
            "--recorded-checkpoint-path must be the absolute path literal "
            "stored in the frozen logs"
        )

    groupeval_root = Path(__file__).resolve().parents[1]
    project_root = groupeval_root.parent
    if is_within(output_dir, project_root):
        raise ValueError(
            f"private output must be outside the project tree {project_root}: "
            f"{output_dir}"
        )
    if is_within(output_dir, source_root):
        raise ValueError(
            f"private output must not be inside the frozen source root: {output_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a fresh path"
        )

    protocol_path = groupeval_root / "protocol" / "protocol_v1.yaml"
    profile_path = (
        groupeval_root / "protocol" / "target_profile_owt1024_fullstream.md"
    )
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol not found: {protocol_path}")
    if not profile_path.is_file():
        raise FileNotFoundError(f"target profile not found: {profile_path}")

    # Validate every frozen input, including the 4.17 GB checkpoint, before
    # creating the private output tree.
    verified_cells, shared_provenance = validate_inputs(
        source_root,
        checkpoint_path,
        recorded_source_root,
        recorded_checkpoint_path,
    )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory became nonempty during validation: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    samples_root = output_dir / "samples"
    samples_root.mkdir()
    samples_root.chmod(0o700)

    manifest_rows: list[dict[str, str]] = []
    cell_provenance: list[dict[str, Any]] = []
    private_methods: dict[str, Any] = {}

    for verified in verified_cells:
        spec = verified.spec
        samples = verified.result["generated_seqs"]
        method_dir = samples_root / spec.method_code
        method_dir.mkdir()
        method_dir.chmod(0o700)

        raw_hashes: list[str] = []
        normalized_hashes: list[str] = []
        sample_records: list[dict[str, Any]] = []
        leading_sentinels_removed = 0
        document_boundaries = 0
        for index, raw_text in enumerate(samples):
            normalized, removed_sentinel, boundary_count = normalize_sample(raw_text)
            if not normalized.strip():
                raise ValueError(
                    f"{verified.result_path}: sample {index} became empty "
                    "after sentinel removal"
                )
            raw_hash = sha256_text(raw_text)
            normalized_hash = sha256_text(normalized)
            raw_hashes.append(raw_hash)
            normalized_hashes.append(normalized_hash)
            leading_sentinels_removed += int(removed_sentinel)
            document_boundaries += boundary_count

            sample_id = f"{spec.method_code}-{index:03d}"
            relative_text_path = (
                Path("samples") / spec.method_code / f"{index:03d}.txt"
            )
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
                    "source_locator": (
                        f"{spec.result_name}#/generated_seqs/{index}"
                    ),
                    "raw_text_sha256": raw_hash,
                    "normalized_text_sha256": normalized_hash,
                    "normalized_text_path": relative_text_path.as_posix(),
                    "leading_sentinel_removed": removed_sentinel,
                    "document_boundaries_rendered": boundary_count,
                }
            )

        private_methods[spec.method_code] = {
            "model": "DiDi-Instruct EMA student",
            "configured_updates_M": spec.updates,
            "sampler": "reward-guided ancestral",
            "predictor": "guided",
            "guided_variant": "current",
            "noise_removal": "ancestral",
            "use_student_ema": True,
            "temperature": spec.temperature,
            "categorical_precision": spec.precision,
            "condition": spec.condition,
            "verified_source_result": str(verified.result_path),
            "verified_source_log": str(verified.log_path),
            "archived_source_result": str(
                recorded_source_root / spec.result_name
            ),
            "archived_source_log": str(
                recorded_source_root / "logs" / spec.log_name
            ),
        }
        cell_provenance.append(
            {
                "method_code": spec.method_code,
                "configured_updates_M": spec.updates,
                "temperature": spec.temperature,
                "categorical_precision": spec.precision,
                "sampler": {
                    "predictor": "guided",
                    "guided_variant": "current",
                    "noise_removal": "ancestral",
                    "use_student_ema": True,
                    "p_nucleus": 1.0,
                },
                "global_lightning_seed": EXPECTED_GLOBAL_SEED,
                "per_sample_seed_available": False,
                "sample_index_semantics": (
                    "deterministic decoded-list order 0..39; not a per-sample RNG seed"
                ),
                "result_file": {
                    "verified_path": str(verified.result_path),
                    "archived_path": str(
                        recorded_source_root / spec.result_name
                    ),
                    "sha256": verified.result_sha256,
                    "bytes": verified.result_path.stat().st_size,
                },
                "run_log": {
                    "verified_path": str(verified.log_path),
                    "archived_path": str(
                        recorded_source_root / "logs" / spec.log_name
                    ),
                    "sha256": verified.log_sha256,
                    "bytes": verified.log_path.stat().st_size,
                    **verified.log_metadata,
                },
                "archived_full_grid_manifest_row": dict(
                    verified.archived_manifest_row
                ),
                "source_timestamp": verified.log_metadata["source_timestamp"],
                "sample_count": EXPECTED_SAMPLE_COUNT,
                "raw_sample_hash_list_sha256": hash_list(raw_hashes),
                "normalized_sample_hash_list_sha256": hash_list(
                    normalized_hashes
                ),
                "leading_sentinels_removed": leading_sentinels_removed,
                "document_boundaries_rendered": document_boundaries,
                "performance_for_cross_check_only": {
                    "generative_perplexity": spec.generative_ppl,
                    "entropy": spec.entropy,
                    "evaluator": EXPECTED_EVALUATOR,
                },
                "samples": sample_records,
            }
        )

    require_equal(
        "prepared manifest row count",
        len(manifest_rows),
        len(CELLS) * EXPECTED_SAMPLE_COUNT,
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
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    manifest_path.chmod(0o600)

    private_map_path = output_dir / "private_label_map.json"
    write_json(
        private_map_path,
        {
            "artifact": "GroupEval DiDi-Instruct pilot private method map",
            "preparation_version": PREPARATION_VERSION,
            "target_profile_id": TARGET_PROFILE_ID,
            "keep_private_until_judging_complete": True,
            "methods": private_methods,
        },
    )

    provenance_path = output_dir / "generation_provenance.json"
    provenance = {
        "artifact": (
            "GroupEval DiDi-Instruct reward-guided two-operating-point pilot"
        ),
        "preparation_version": PREPARATION_VERSION,
        "target_profile": {
            "id": TARGET_PROFILE_ID,
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
        },
        "protocol": {
            "version": grouping.PROTOCOL_VERSION,
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "verified_source_root": str(source_root),
        **shared_provenance,
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
            "quality_filtering": False,
        },
        "seed_and_pairing": {
            "global_lightning_seed": EXPECTED_GLOBAL_SEED,
            "per_sample_seeds_available": False,
            "sample_indices": "decoded-list positions 0..39",
            "sample_level_common_random_numbers_declared": False,
            "allowed_bootstrap_pairing_mode": "partitions",
        },
        "archive_limitations": {
            "generator_code_revision_recorded_by_run": False,
            "generated_token_ids_archived": False,
            "available_identity_evidence": (
                "exact full-grid-manifest, result, run-log, checkpoint, raw-text, "
                "and normalized-text SHA256 hashes"
            ),
            "evaluation_scope": (
                "GroupEval judges the complete decoded stream after sentinel "
                "presentation normalization. The archived gPPL evaluator "
                "retokenizes decoded strings and masks later EOS tokens, so the "
                "diagnostics are complementary rather than presentation-identical."
            ),
        },
        "design": {
            "noncanonical": True,
            "noncanonical_reason": (
                "the frozen paper artifacts contain 40 rather than 128 samples per cell"
            ),
            "methods": len(CELLS),
            "samples_per_method": EXPECTED_SAMPLE_COUNT,
            "canonical_samples_per_method": CANONICAL_SAMPLE_COUNT,
            "total_samples": len(manifest_rows),
            "group_size": GROUP_SIZE,
            "partition_seeds": list(PARTITION_SEEDS),
        },
        "cells": cell_provenance,
    }
    write_json(provenance_path, provenance)

    by_method = grouping.load_manifest(manifest_path)
    indices = grouping.validate_design(
        by_method,
        group_size=GROUP_SIZE,
        expected_samples=EXPECTED_SAMPLE_COUNT,
        allow_noncanonical=True,
    )
    partitions = grouping.build_partitions(
        indices,
        group_size=GROUP_SIZE,
        seeds=PARTITION_SEEDS,
        candidates=OPTIMIZATION_CANDIDATES,
    )
    group_records = grouping.build_records(
        by_method, partitions, protocol_path
    )
    expected_calls = (
        len(CELLS)
        * len(PARTITION_SEEDS)
        * EXPECTED_SAMPLE_COUNT
        // GROUP_SIZE
    )
    require_equal("blinded group count", len(group_records), expected_calls)
    group_ids = [
        str(record["group_input"]["group_id"]) for record in group_records
    ]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("duplicate blinded group IDs generated")
    if not all(
        len(record["group_input"]["samples"]) == GROUP_SIZE
        for record in group_records
    ):
        raise ValueError("generated group with an unexpected sample count")

    groups_path = output_dir / "paper_groups.jsonl"
    grouping.write_jsonl(groups_path, group_records)
    groups_path.chmod(0o600)

    grouping_summary = {
        "protocol_version": grouping.PROTOCOL_VERSION,
        "target_profile_id": TARGET_PROFILE_ID,
        "noncanonical": True,
        "canonical_samples_per_method": CANONICAL_SAMPLE_COUNT,
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
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(top_level_hashes.items())
        ),
        encoding="utf-8",
    )
    sums_path.chmod(0o600)

    return {
        **grouping_summary,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "private_label_map": str(private_map_path),
        "provenance": str(provenance_path),
        "sha256sums": str(sums_path),
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help=(
            "Actual directory to read, containing manifest.tsv, the ten frozen "
            "JSON results, and logs/. It may be an exact relocated copy."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help=(
            "Actual checkpoint file to read and hash. It may be an exact "
            "relocated copy."
        ),
    )
    parser.add_argument(
        "--recorded-source-root",
        required=True,
        type=Path,
        help=(
            "Audit-only absolute source-root path literal recorded in the "
            "frozen manifest and run logs. This path need not exist."
        ),
    )
    parser.add_argument(
        "--recorded-checkpoint-path",
        required=True,
        type=Path,
        help=(
            "Audit-only absolute checkpoint path literal recorded in the "
            "frozen run logs. This path need not exist."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Fresh private directory outside the project tree.",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
