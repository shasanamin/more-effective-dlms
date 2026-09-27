#!/usr/bin/env python3
"""Prepare the matched 40-output MDLM-Small/DiDi-Instruct GroupEval study.

This program performs no generation and makes no API calls.  It verifies the
exact twenty generation artifacts behind ``cross_model_gp_may05.csv`` (two
models, two operating points, and five update budgets), both checkpoints, and
the frozen paper manifests before writing private, blinded GroupEval inputs.

Each method cell has forty complete decoded streams.  The exploratory design
uses method-homogeneous groups of eight and two deterministic, overlap-reduced
partitions: ten judge calls per cell and 200 calls in total.  Requests are
ordered in ten rounds containing every one of the twenty cells once, with a
deterministically shuffled within-round order.  This balances every cell over
a judge endpoint that may drift during a long run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import grouping


PREPARATION_VERSION = "1.0.0"
ARTIFACT_NAME = "Matched MDLM-Small/DiDi-Instruct GroupEval pilot"
TARGET_PROFILE_ID = "owt1024-fullstream-v1"
EOS_LITERAL = "<|endoftext|>"
DOCUMENT_BOUNDARY = "\n\n[DOCUMENT BOUNDARY]\n\n"
EXPECTED_SAMPLE_COUNT = 40
CANONICAL_SAMPLE_COUNT = 128
EXPECTED_GLOBAL_SEED = 1
EXPECTED_SEQUENCE_LENGTH = 1024
EXPECTED_EVALUATOR = "gpt2-large"
EXPECTED_EVALUATOR_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
EXPECTED_MANIFEST_SHA256 = "163d1ffe8faebe455e0f75b472a424740312f3240f395060358d46431fa7dc8f"
EXPECTED_METRICS_SHA256 = "31d614df7694193c2f7fa4b9aef0acf4c1bc6a370e304db30720b1cb39b02309"
EXPECTED_CHECKPOINTS = {
    "MDLM-Small": {
        "filename": "mdlm.ckpt",
        "bytes": 2_715_759_768,
        "sha256": "74bf9b853206e73b4811d8127e836c2cdcfb7a23f8e04ad7d1356a58f1138d27",
    },
    "DiDi-Instruct": {
        "filename": "didi-instruct.sanitized.ckpt",
        "bytes": 4_168_811_487,
        "sha256": "e7f41f80b3a1700dcc46f6920a0bec616a55753e216cf6c3bc8985335b51dd82",
    },
}
GROUP_SIZE = 8
PARTITION_SEEDS = (271828, 314159)
OPTIMIZATION_CANDIDATES = 2000
CALLS_PER_METHOD = 10
INTERLEAVE_SEED = 20260731
INTERLEAVE_ALGORITHM = "complete-cell-rounds-v1"
SHARD_COUNT = 4
SHARD_ALGORITHM = "rotating-opaque-method-rank-v1"
UPDATES = (4, 16, 64, 256, 1024)
UNRESOLVED_MASKS = ("[MASK]", "<mask>", "<|mask|>")


@dataclass(frozen=True)
class CellSpec:
    model: str
    operating_point: str
    updates: int
    temperature: float
    precision: str
    result_name: str
    result_sha256: str
    generative_perplexity: float
    entropy: float

    @property
    def condition(self) -> str:
        return (
            "unscaled-fp64-reference"
            if self.operating_point == "reference"
            else "selected"
        )

    @property
    def method_code(self) -> str:
        payload = (
            "GroupEval-cross-model-pilot-v1|"
            f"{self.model}|{self.operating_point}|{self.updates}"
        )
        return "GE-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def metrics_model(self) -> str:
        if self.model == "MDLM-Small":
            return "MDLM" if self.operating_point == "reference" else "MDLM calibrated"
        return (
            "DiDi-Instruct"
            if self.operating_point == "reference"
            else "DiDi-Instruct calibrated"
        )

    @property
    def source_root_key(self) -> str:
        return "mdlm" if self.model == "MDLM-Small" else "didi"


def _cell(
    model: str,
    operating_point: str,
    updates: int,
    result_name: str,
    result_sha256: str,
    generative_perplexity: float,
    entropy: float,
) -> CellSpec:
    selected = operating_point == "selected"
    return CellSpec(
        model=model,
        operating_point=operating_point,
        updates=updates,
        temperature=0.9 if selected else 1.0,
        precision="fp32" if selected else "fp64",
        result_name=result_name,
        result_sha256=result_sha256,
        generative_perplexity=generative_perplexity,
        entropy=entropy,
    )


CELLS = (
    _cell("MDLM-Small", "reference", 4, "mdlm_default_tau1_fp64_s4.json", "a5e9af728e461319ec0f8400de90c1fb3f653281ef0920ef9671a89464227e20", 1948.8134765625, 5.950348854064941),
    _cell("MDLM-Small", "reference", 16, "mdlm_default_tau1_fp64_s16.json", "f96392471b488c09ba4050084edf09bb4443dfc5572bd4892584e1c85473eea5", 344.1566467285156, 5.787546634674072),
    _cell("MDLM-Small", "reference", 64, "mdlm_default_tau1_fp64_s64.json", "d897b7d0b9a6dd1ffaf54228b04930a214ea755552913f7d8e2fc270f53a6033", 145.38784790039062, 5.706974029541016),
    _cell("MDLM-Small", "reference", 256, "mdlm_default_tau1_fp64_s256.json", "4fb9a7b0bbfe8b29d300c6d51a42d282c09048bb9e90c0ff6e945cbf72ae4c37", 109.89892578125, 5.644200325012207),
    _cell("MDLM-Small", "reference", 1024, "mdlm_default_tau1_fp64_s1024.json", "3cf7967cbc25fbeaec63827f03e5c8ca9d5dd5a54561920846fa8d8b8cc6d402", 103.75968170166016, 5.636950492858887),
    _cell("MDLM-Small", "selected", 4, "mdlm_calibrated_tau0p9_fp32_s4.json", "56d1eb94d5024db95a0281bfe4651f745180d50f8c893c380486f21622381509", 519.4310913085938, 5.287627220153809),
    _cell("MDLM-Small", "selected", 16, "mdlm_calibrated_tau0p9_fp32_s16.json", "1603128eddf554d94beed7e6293d0399f5b8edbd7303d80fb5db7bae234e92d0", 103.61832427978516, 5.356069087982178),
    _cell("MDLM-Small", "selected", 64, "mdlm_calibrated_tau0p9_fp32_s64.json", "64db4ebbdbbaddbc8fb7be1a89eba8aeb9c9b453bcd065936fdd75daf6c0af76", 44.5380973815918, 5.302527904510498),
    _cell("MDLM-Small", "selected", 256, "mdlm_calibrated_tau0p9_fp32_s256.json", "16490be8f0d52948c1f26e10ad216e5589f737621ef616af80f3c4a07b0b076a", 30.131942749023438, 5.148928165435791),
    _cell("MDLM-Small", "selected", 1024, "mdlm_calibrated_tau0p9_fp32_s1024.json", "e56800674680d29d079c93b7ed909c5c4c8d309a9708ac6b78f5210596232489", 21.4289493560791, 4.982621192932129),
    _cell("DiDi-Instruct", "reference", 4, "guided_current_ema_t1p0_fp64_s4.json", "5e2cd5767a3ac46b7903a8481a8755aeef0002414837482053b51df1672186e3", 523.5781860351562, 5.563353061676025),
    _cell("DiDi-Instruct", "reference", 16, "guided_current_ema_t1p0_fp64_s16.json", "1d2521f0690e7623a0b277f8018cefc233c3632879f2389a05e5fa09a6daac91", 71.55656433105469, 5.430166721343994),
    _cell("DiDi-Instruct", "reference", 64, "guided_current_ema_t1p0_fp64_s64.json", "f5390c870b9fa13967ebacdf41343f2c23daa709a460a9986fc3900baa6e8027", 37.28147888183594, 5.370088577270508),
    _cell("DiDi-Instruct", "reference", 256, "guided_current_ema_t1p0_fp64_s256.json", "b06583c645a4df0ed1a091892548c38250519a85f9d96032821ff69a791c7f8b", 30.932842254638672, 5.333018779754639),
    _cell("DiDi-Instruct", "reference", 1024, "guided_current_ema_t1p0_fp64_s1024.json", "95eafe0f0ff0e03bba39ac3adda8fd1683318a7890c540a8c3e6bc967cda6ead", 30.05565643310547, 5.307525634765625),
    _cell("DiDi-Instruct", "selected", 4, "guided_current_ema_t0p9_fp32_s4.json", "0ec7b074d6570f1ee61e98a31c9b84e90c764f91714f9bfc371520be906052aa", 220.06297302246094, 5.117529392242432),
    _cell("DiDi-Instruct", "selected", 16, "guided_current_ema_t0p9_fp32_s16.json", "66ee3cb2f680d2e2d88d335378e15982f732d590927e4929c1cbf51ca6b21675", 40.66252899169922, 5.173752784729004),
    _cell("DiDi-Instruct", "selected", 64, "guided_current_ema_t0p9_fp32_s64.json", "ea1fe8d6512c799f188ef12c4c4be48d48a6cc23ffd798262fc13d7afecc9177", 20.74135398864746, 5.067018985748291),
    _cell("DiDi-Instruct", "selected", 256, "guided_current_ema_t0p9_fp32_s256.json", "2a4ac2bb41efb0d811a37d3fa4304094630f4c176026ed331fc1809a50d2c178", 17.0389461517334, 5.031113624572754),
    _cell("DiDi-Instruct", "selected", 1024, "guided_current_ema_t0p9_fp32_s1024.json", "d93de198e71aed8ee4cbff2f633d84aede1b659d51495df9cfaa9285a3bbdb69", 15.075967788696289, 4.948769569396973),
)


@dataclass(frozen=True)
class VerifiedCell:
    spec: CellSpec
    path: Path
    data: Mapping[str, Any]
    samples: Sequence[str]
    timestamp: str | None


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_list(values: Iterable[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return sha256_text(payload)


def require_equal(context: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{context}: expected {expected!r}, observed {observed!r}")


def finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: expected a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite number")
    return result


def normalize_sample(raw_text: str) -> tuple[str, bool, int]:
    """Apply the predeclared full-stream rendering used by existing GroupEval."""
    removed_leading_sentinel = raw_text.startswith(EOS_LITERAL)
    body = raw_text[len(EOS_LITERAL) :] if removed_leading_sentinel else raw_text
    later_boundaries = body.count(EOS_LITERAL)
    return (
        body.replace(EOS_LITERAL, DOCUMENT_BOUNDARY),
        removed_leading_sentinel,
        later_boundaries,
    )


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_paper_inputs(manifest_path: Path, metrics_path: Path) -> None:
    require_equal(f"{manifest_path}:SHA256", sha256_file(manifest_path), EXPECTED_MANIFEST_SHA256)
    require_equal(f"{metrics_path}:SHA256", sha256_file(metrics_path), EXPECTED_METRICS_SHA256)

    manifest_rows = _load_csv(manifest_path)
    expected_manifest = {
        "MDLM-Small": EXPECTED_CHECKPOINTS["MDLM-Small"],
        "DiDi-Instruct": EXPECTED_CHECKPOINTS["DiDi-Instruct"],
    }
    by_label = {row["paper_label"]: row for row in manifest_rows}
    for label, checkpoint in expected_manifest.items():
        if label not in by_label:
            raise ValueError(f"{manifest_path}: missing {label} row")
        row = by_label[label]
        require_equal(f"{manifest_path}:{label}:checkpoint_filename", row["checkpoint_filename"], checkpoint["filename"])
        require_equal(f"{manifest_path}:{label}:checkpoint_sha256", row["checkpoint_sha256"], checkpoint["sha256"])
        require_equal(f"{manifest_path}:{label}:evaluator_id", row["evaluator_id"], EXPECTED_EVALUATOR)
        require_equal(f"{manifest_path}:{label}:evaluator_revision", row["evaluator_revision"], EXPECTED_EVALUATOR_REVISION)
        require_equal(f"{manifest_path}:{label}:generation_seed", row["generation_seed"], str(EXPECTED_GLOBAL_SEED))
        require_equal(f"{manifest_path}:{label}:samples_per_point", row["samples_per_point"], str(EXPECTED_SAMPLE_COUNT))
        require_equal(f"{manifest_path}:{label}:sequence_length", row["sequence_length"], str(EXPECTED_SEQUENCE_LENGTH))

    wanted = {(cell.metrics_model, cell.updates): cell for cell in CELLS}
    observed: dict[tuple[str, int], Mapping[str, str]] = {}
    for row in _load_csv(metrics_path):
        key = (row["model"], int(row["steps"]))
        if key in wanted:
            if key in observed:
                raise ValueError(f"{metrics_path}: duplicate target row {key}")
            observed[key] = row
    require_equal(f"{metrics_path}:target rows", set(observed), set(wanted))
    for key, spec in wanted.items():
        row = observed[key]
        require_equal(f"{metrics_path}:{key}:setting", row["setting"], "calibrated" if spec.operating_point == "selected" else "default")
        require_equal(f"{metrics_path}:{key}:total_samples", int(row["total_samples"]), EXPECTED_SAMPLE_COUNT)
        require_equal(f"{metrics_path}:{key}:temperature", float(row["temperature"]), spec.temperature)
        require_equal(f"{metrics_path}:{key}:evaluator", row["evaluator"], EXPECTED_EVALUATOR)
        require_equal(f"{metrics_path}:{key}:source basename", Path(row["source"].split(" ", 1)[0]).name, spec.result_name)
        require_equal(f"{metrics_path}:{key}:gPPL", float(row["gen_ppl"]), spec.generative_perplexity)
        require_equal(f"{metrics_path}:{key}:entropy", float(row["entropy"]), spec.entropy)
        precision = row["precision"].lower().replace("float", "fp")
        require_equal(f"{metrics_path}:{key}:precision", precision, spec.precision)
        use_float64 = row["use_float64"].strip().lower()
        if use_float64 not in {"true", "false"}:
            raise ValueError(f"{metrics_path}:{key}: invalid use_float64 value")
        require_equal(
            f"{metrics_path}:{key}:use_float64",
            use_float64 == "true",
            spec.precision == "fp64",
        )
        expected_sampler = "ancestral_cache" if spec.model == "MDLM-Small" else "guided"
        require_equal(f"{metrics_path}:{key}:sampler", row["sampler"], expected_sampler)


def verify_result(spec: CellSpec, path: Path) -> VerifiedCell:
    if not path.is_file():
        raise FileNotFoundError(f"frozen generation artifact not found: {path}")
    require_equal(f"{path}:SHA256", sha256_file(path), spec.result_sha256)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be an object")

    if spec.model == "MDLM-Small":
        require_equal(f"{path}:top-level keys", set(data), {"metadata", "performance", "samples", "nfe"})
        metadata = data["metadata"]
        performance = data["performance"]
        samples = data["samples"]
        nfe = data["nfe"]
        expected_metadata = {
            "algo": "mdlm",
            "sampler": "ancestral_cache",
            "sample_steps": spec.updates,
            "num_sample_batches": 5,
            "batch_size": 8,
            "total_samples": EXPECTED_SAMPLE_COUNT,
            "use_float64": spec.precision == "fp64",
            "gumbel_precision": "float64" if spec.precision == "fp64" else "float32",
            "x0_temperature": spec.temperature,
            "x0_temperature_schedule": "constant",
            "gen_ppl_model": EXPECTED_EVALUATOR,
        }
        for key, value in expected_metadata.items():
            require_equal(f"{path}:metadata.{key}", metadata.get(key), value)
        require_equal(f"{path}:nfe count", len(nfe), EXPECTED_SAMPLE_COUNT)
        if any(value != spec.updates for value in nfe):
            raise ValueError(f"{path}: every NFE must equal M={spec.updates}")
        gppl = finite_float(performance.get("generative_perplexity"), f"{path}:gPPL")
        entropy = finite_float(performance.get("avg_entropy"), f"{path}:entropy")
        timestamp = str(metadata.get("timestamp"))
    else:
        require_equal(f"{path}:top-level keys", set(data), {"generative_ppl", "entropy", "generated_seqs"})
        samples = data["generated_seqs"]
        gppl = finite_float(data.get("generative_ppl"), f"{path}:gPPL")
        entropy = finite_float(data.get("entropy"), f"{path}:entropy")
        timestamp = None

    require_equal(f"{path}:gPPL", gppl, spec.generative_perplexity)
    require_equal(f"{path}:entropy", entropy, spec.entropy)
    if not isinstance(samples, list):
        raise ValueError(f"{path}: samples must be an array")
    require_equal(f"{path}:sample count", len(samples), EXPECTED_SAMPLE_COUNT)
    if not all(isinstance(text, str) and text.strip() for text in samples):
        raise ValueError(f"{path}: every decoded stream must be a nonempty string")
    if len(set(samples)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"{path}: exact duplicate decoded streams detected")
    if any(token in text for text in samples for token in UNRESOLVED_MASKS):
        raise ValueError(f"{path}: unresolved mask token detected")
    if any(DOCUMENT_BOUNDARY.strip() in text for text in samples):
        raise ValueError(f"{path}: raw stream collides with presentation marker")
    return VerifiedCell(spec=spec, path=path, data=data, samples=samples, timestamp=timestamp)


def verify_checkpoint(label: str, path: Path) -> dict[str, Any]:
    expected = EXPECTED_CHECKPOINTS[label]
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    require_equal(f"{path}:bytes", path.stat().st_size, expected["bytes"])
    observed = sha256_file(path)
    require_equal(f"{path}:SHA256", observed, expected["sha256"])
    return {"verified_path": str(path), "filename": expected["filename"], "bytes": expected["bytes"], "sha256": observed}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def interleave_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        metadata = record["request_metadata"]
        by_method.setdefault(str(metadata["method_code"]), []).append(record)
    methods = sorted(by_method)
    require_equal("interleave method count", len(methods), len(CELLS))
    for method, rows in by_method.items():
        rows.sort(key=lambda row: (str(row["request_metadata"]["partition_id"]), int(row["request_metadata"]["group_index"])))
        require_equal(f"{method}:judge calls", len(rows), 10)

    interleaved: list[dict[str, Any]] = []
    for round_index in range(10):
        order = list(methods)
        random.Random(grouping.stable_seed(INTERLEAVE_SEED, round_index)).shuffle(order)
        for position, method in enumerate(order):
            record = by_method[method][round_index]
            metadata = record["request_metadata"]
            metadata["request_sequence_index"] = len(interleaved)
            metadata["interleave_round"] = round_index
            metadata["interleave_position_in_round"] = position
            metadata["interleave_seed"] = INTERLEAVE_SEED
            metadata["interleave_algorithm"] = INTERLEAVE_ALGORITHM
            interleaved.append(record)
    return interleaved


def shard_records(
    records: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Assign complete-round requests to four balanced concurrent workers.

    Opaque method rank rotates across workers by round.  Thus every worker gets
    five calls from every complete-cell round, while every method contributes
    either two or three calls to each worker over the ten rounds.  Filtering
    the canonical order preserves the within-worker temporal schedule.
    """
    methods = sorted(
        {str(record["request_metadata"]["method_code"]) for record in records}
    )
    require_equal("sharding method count", len(methods), len(CELLS))
    method_rank = {method: rank for rank, method in enumerate(methods)}
    shards: list[list[dict[str, Any]]] = [[] for _ in range(SHARD_COUNT)]
    per_method_worker: dict[tuple[str, int], int] = {}
    for record in records:
        metadata = record["request_metadata"]
        method = str(metadata["method_code"])
        round_index = int(metadata["interleave_round"])
        shard_index = (method_rank[method] + round_index) % SHARD_COUNT
        metadata["shard_id"] = f"S{shard_index + 1}"
        metadata["shard_count"] = SHARD_COUNT
        metadata["shard_sequence_index"] = len(shards[shard_index])
        metadata["shard_algorithm"] = SHARD_ALGORITHM
        shards[shard_index].append(record)
        key = (method, shard_index)
        per_method_worker[key] = per_method_worker.get(key, 0) + 1

    expected_per_round = len(CELLS) // SHARD_COUNT
    for shard_index, shard in enumerate(shards):
        require_equal(
            f"shard {shard_index + 1}:call count",
            len(shard),
            len(records) // SHARD_COUNT,
        )
        for round_index in range(CALLS_PER_METHOD):
            round_calls = [
                record
                for record in shard
                if int(record["request_metadata"]["interleave_round"])
                == round_index
            ]
            require_equal(
                f"shard {shard_index + 1}:round {round_index}:calls",
                len(round_calls),
                expected_per_round,
            )
    for method in methods:
        counts = [per_method_worker.get((method, shard_index), 0) for shard_index in range(SHARD_COUNT)]
        require_equal(f"{method}:shard count range", (min(counts), max(counts)), (2, 3))
        require_equal(f"{method}:shard call total", sum(counts), CALLS_PER_METHOD)
    return shards


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    mdlm_root = args.mdlm_source_root.resolve()
    didi_root = args.didi_source_root.resolve()
    manifest_path = args.cross_model_manifest.resolve()
    metrics_path = args.cross_model_metrics.resolve()
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[2]
    if not mdlm_root.is_dir() or not didi_root.is_dir():
        raise FileNotFoundError("both generation source roots must exist")
    if is_within(output_dir, project_root):
        raise ValueError(f"private output must be outside project tree {project_root}")
    if is_within(output_dir, mdlm_root) or is_within(output_dir, didi_root):
        raise ValueError("private output must not be inside either frozen source root")
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    verify_paper_inputs(manifest_path, metrics_path)
    roots = {"mdlm": mdlm_root, "didi": didi_root}
    verified_cells = [verify_result(spec, roots[spec.source_root_key] / spec.result_name) for spec in CELLS]
    checkpoints = {
        "MDLM-Small": verify_checkpoint("MDLM-Small", args.mdlm_checkpoint.resolve()),
        "DiDi-Instruct": verify_checkpoint("DiDi-Instruct", args.didi_checkpoint.resolve()),
    }

    groupeval_root = Path(__file__).resolve().parents[1]
    protocol_path = groupeval_root / "protocol" / "protocol_v1.yaml"
    profile_path = groupeval_root / "protocol" / "target_profile_owt1024_fullstream.md"
    if not protocol_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("GroupEval protocol/profile artifact is missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    samples_root = output_dir / "samples"
    samples_root.mkdir()
    samples_root.chmod(0o700)

    manifest_rows: list[dict[str, str]] = []
    methods: dict[str, dict[str, Any]] = {}
    provenance_cells: list[dict[str, Any]] = []
    for verified in verified_cells:
        spec = verified.spec
        method_dir = samples_root / spec.method_code
        method_dir.mkdir()
        method_dir.chmod(0o700)
        raw_hashes: list[str] = []
        normalized_hashes: list[str] = []
        sample_records: list[dict[str, Any]] = []
        leading_removed = 0
        boundary_count = 0
        normalized_seen: set[str] = set()
        source_key = "samples" if spec.model == "MDLM-Small" else "generated_seqs"
        for index, raw_text in enumerate(verified.samples):
            normalized, removed, boundaries = normalize_sample(raw_text)
            if not normalized.strip():
                raise ValueError(f"{verified.path}: sample {index} became empty")
            if normalized in normalized_seen:
                raise ValueError(f"{verified.path}: normalization created duplicate sample {index}")
            normalized_seen.add(normalized)
            raw_hash = sha256_text(raw_text)
            normalized_hash = sha256_text(normalized)
            raw_hashes.append(raw_hash)
            normalized_hashes.append(normalized_hash)
            leading_removed += int(removed)
            boundary_count += boundaries
            sample_id = f"{spec.method_code}-{index:03d}"
            relative = Path("samples") / spec.method_code / f"{index:03d}.txt"
            destination = output_dir / relative
            destination.write_text(normalized, encoding="utf-8")
            destination.chmod(0o600)
            manifest_rows.append({
                "method_code": spec.method_code,
                "sample_index": str(index),
                "sample_id": sample_id,
                "target_profile_id": TARGET_PROFILE_ID,
                "text_path": relative.as_posix(),
                "text_sha256": normalized_hash,
                "generation_seed": "",
            })
            sample_records.append({
                "sample_index": index,
                "sample_id": sample_id,
                "source_locator": f"{spec.result_name}#/{source_key}/{index}",
                "raw_text_sha256": raw_hash,
                "normalized_text_sha256": normalized_hash,
                "normalized_text_path": relative.as_posix(),
                "leading_sentinel_removed": removed,
                "document_boundaries_rendered": boundaries,
            })

        sampler = "ancestral_cache" if spec.model == "MDLM-Small" else "reward-guided ancestral"
        methods[spec.method_code] = {
            "model": spec.model,
            "configured_updates_M": spec.updates,
            "operating_point": spec.operating_point,
            "condition": spec.condition,
            "temperature": spec.temperature,
            "categorical_precision": spec.precision,
            "sampler": sampler,
            "source_result": str(verified.path),
        }
        provenance_cells.append({
            "method_code": spec.method_code,
            "model": spec.model,
            "configured_updates_M": spec.updates,
            "operating_point": spec.operating_point,
            "condition": spec.condition,
            "temperature": spec.temperature,
            "categorical_precision": spec.precision,
            "sampler": sampler,
            "global_generation_seed": EXPECTED_GLOBAL_SEED,
            "per_sample_seed_available": False,
            "sample_index_semantics": "decoded-list position 0..39; not a per-sample RNG seed",
            "source_result": {"verified_path": str(verified.path), "sha256": spec.result_sha256, "bytes": verified.path.stat().st_size},
            "source_timestamp": verified.timestamp,
            "source_metrics": {"evaluator": EXPECTED_EVALUATOR, "evaluator_revision": EXPECTED_EVALUATOR_REVISION, "generative_perplexity": spec.generative_perplexity, "within_sequence_entropy": spec.entropy},
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "raw_sample_hash_list_sha256": hash_list(raw_hashes),
            "normalized_sample_hash_list_sha256": hash_list(normalized_hashes),
            "leading_sentinels_removed": leading_removed,
            "document_boundaries_rendered": boundary_count,
            "samples": sample_records,
        })

    paper_manifest = output_dir / "paper_manifest.csv"
    fieldnames = ("method_code", "sample_index", "sample_id", "target_profile_id", "text_path", "text_sha256", "generation_seed")
    with paper_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    paper_manifest.chmod(0o600)

    private_map_path = output_dir / "private_label_map.json"
    write_json(private_map_path, {
        "artifact": f"{ARTIFACT_NAME} private method map",
        "preparation_version": PREPARATION_VERSION,
        "target_profile_id": TARGET_PROFILE_ID,
        "keep_private_until_judging_complete": True,
        "methods": methods,
    })

    provenance_path = output_dir / "generation_provenance.json"
    provenance = {
        "artifact": ARTIFACT_NAME,
        "preparation_version": PREPARATION_VERSION,
        "paper_inputs": {
            "cross_model_manifest": {"path": str(manifest_path), "sha256": EXPECTED_MANIFEST_SHA256, "bytes": manifest_path.stat().st_size},
            "cross_model_metrics": {"path": str(metrics_path), "sha256": EXPECTED_METRICS_SHA256, "bytes": metrics_path.stat().st_size},
        },
        "checkpoints": checkpoints,
        "target_profile": {"id": TARGET_PROFILE_ID, "path": str(profile_path), "sha256": sha256_file(profile_path)},
        "protocol": {"version": grouping.PROTOCOL_VERSION, "path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "normalization": {"name": TARGET_PROFILE_ID, "complete_decoded_stream": True, "leading_eos_removed": True, "later_eos_rendered_as_document_boundary": True, "truncation_or_filtering": False},
        "source_metric_scope": {
            "role": "exact descriptive covariates imported from the paper table",
            "warning": (
                "The common evaluator ID/revision does not by itself establish "
                "identical archived EOS-span treatment across generation wrappers. "
                "Use a common rescoring pipeline for new cross-model perplexity claims."
            ),
        },
        "seed_and_pairing": {"global_generation_seed": EXPECTED_GLOBAL_SEED, "per_sample_seeds_available": False, "sample_level_common_random_numbers_declared": False, "allowed_bootstrap_pairing_mode": "partitions", "note": "Model outputs are independent; only frozen partition-context weights may be shared in matched-cell contrasts."},
        "design": {"models": 2, "operating_points": 2, "update_budgets": list(UPDATES), "methods": len(CELLS), "samples_per_method": EXPECTED_SAMPLE_COUNT, "total_samples": len(manifest_rows), "group_size": GROUP_SIZE, "partition_seeds": list(PARTITION_SEEDS), "groups_per_partition_per_method": 5, "calls_per_method": 10, "total_judge_calls": 200, "noncanonical": True, "canonical_samples_per_method": CANONICAL_SAMPLE_COUNT},
        "request_order": {
            "algorithm": INTERLEAVE_ALGORITHM,
            "seed": INTERLEAVE_SEED,
            "rounds": CALLS_PER_METHOD,
            "calls_per_round": len(CELLS),
            "invariant": "Each contiguous round contains every method cell exactly once.",
            "sharding": {
                "algorithm": SHARD_ALGORITHM,
                "shards": SHARD_COUNT,
                "calls_per_shard": 50,
                "calls_per_shard_per_round": 5,
                "per_method_calls_per_shard": "two or three; ten total across four shards",
                "merge_order": "paper_groups.jsonl canonical request sequence",
            },
        },
        "cells": provenance_cells,
    }
    write_json(provenance_path, provenance)

    by_method = grouping.load_manifest(paper_manifest)
    indices = grouping.validate_design(by_method, GROUP_SIZE, EXPECTED_SAMPLE_COUNT, allow_noncanonical=True)
    partitions = grouping.build_partitions(indices, GROUP_SIZE, PARTITION_SEEDS, OPTIMIZATION_CANDIDATES)
    records = interleave_records(grouping.build_records(by_method, partitions, protocol_path))
    require_equal("judge call count", len(records), 200)
    expected_methods = set(methods)
    for round_index in range(10):
        round_methods = {str(row["request_metadata"]["method_code"]) for row in records[round_index * 20 : (round_index + 1) * 20]}
        require_equal(f"interleave round {round_index}", round_methods, expected_methods)
    shards = shard_records(records)
    groups_path = output_dir / "paper_groups.jsonl"
    grouping.write_jsonl(groups_path, records)
    groups_path.chmod(0o600)

    shards_root = output_dir / "shards"
    shards_root.mkdir()
    shards_root.chmod(0o700)
    shard_descriptors: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards, 1):
        name = f"paper_groups_shard_{shard_index:02d}_of_{SHARD_COUNT:02d}.jsonl"
        path = shards_root / name
        grouping.write_jsonl(path, shard)
        path.chmod(0o600)
        method_counts: dict[str, int] = {}
        for record in shard:
            method = str(record["request_metadata"]["method_code"])
            method_counts[method] = method_counts.get(method, 0) + 1
        shard_descriptors.append({
            "shard_id": f"S{shard_index}",
            "path": (Path("shards") / name).as_posix(),
            "sha256": sha256_file(path),
            "judge_calls": len(shard),
            "rounds": CALLS_PER_METHOD,
            "calls_per_round": len(CELLS) // SHARD_COUNT,
            "method_call_counts": dict(sorted(method_counts.items())),
        })
    shard_manifest_path = output_dir / "shard_manifest.json"
    write_json(shard_manifest_path, {
        "artifact": f"{ARTIFACT_NAME} deterministic request shards",
        "algorithm": SHARD_ALGORITHM,
        "canonical_groups": "paper_groups.jsonl",
        "canonical_groups_sha256": sha256_file(groups_path),
        "shards": shard_descriptors,
        "merge": {
            "tool": "experiment/merge_judgment_shards.py",
            "canonical_order_source": "paper_groups.jsonl",
            "expected_provider": "institutional_genai",
            "expected_model": "gpt-oss:120b",
        },
    })

    summary = {
        "artifact": ARTIFACT_NAME,
        "protocol_version": grouping.PROTOCOL_VERSION,
        "target_profile_id": TARGET_PROFILE_ID,
        "noncanonical": True,
        "methods": sorted(methods),
        "samples_per_method": EXPECTED_SAMPLE_COUNT,
        "group_size": GROUP_SIZE,
        "partitions": [{"partition_id": partition_id, "seed": seed, "repeated_pairs_with_earlier_partitions": overlap, "groups_per_method": len(groups)} for partition_id, seed, overlap, groups in partitions],
        "calls_per_method": 10,
        "judge_calls": len(records),
        "request_order": provenance["request_order"],
        "request_shards": {
            "count": SHARD_COUNT,
            "calls_per_shard": [len(shard) for shard in shards],
            "manifest": "shard_manifest.json",
        },
    }
    summary_path = output_dir / "grouping_summary.json"
    write_json(summary_path, summary)

    hash_paths = sorted(
        (path for path in output_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(output_dir).as_posix(),
    )
    sums_path = output_dir / "SHA256SUMS"
    sums_path.write_text("".join(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n" for path in hash_paths), encoding="utf-8")
    sums_path.chmod(0o600)
    return {**summary, "output_dir": str(output_dir), "groups": str(groups_path), "generation_provenance": str(provenance_path), "private_label_map": str(private_map_path), "shard_manifest": str(shard_manifest_path), "sha256sums": str(sums_path)}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mdlm-source-root", required=True, type=Path, help="Directory containing exact cross_model_may05 MDLM JSON files")
    parser.add_argument("--didi-source-root", required=True, type=Path, help="Directory containing exact didi_full_grid_may01 JSON files")
    parser.add_argument("--mdlm-checkpoint", required=True, type=Path)
    parser.add_argument("--didi-checkpoint", required=True, type=Path)
    parser.add_argument("--cross-model-manifest", type=Path, default=root / "paper_data" / "cross_model_manifest.csv")
    parser.add_argument("--cross-model-metrics", type=Path, default=root / "paper_data" / "cross_model_gp_may05.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh private directory outside the project tree")
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
