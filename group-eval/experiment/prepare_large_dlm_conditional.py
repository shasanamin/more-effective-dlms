#!/usr/bin/env python3
"""Prepare frozen conditional GEQ inputs from the archived large-DLM runs.

The script reconstructs the exact deterministic GSM8K flexible-extract labels
and the exact sanitized HumanEval functional labels used by the paper. Hidden
answers, tests, and canonical solutions are used only in memory to obtain binary
labels and are never written to judge inputs or output artifacts.

HumanEval scoring executes model-generated programs in the guarded subprocess
path used by ``rescore_fast_dllm_humaneval.py``. It therefore requires the
explicit ``--allow-code-execution`` acknowledgement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from conditional_large_dlm_common import (
    EVALUATION_PROTOCOL,
    PROTOCOL_VERSION,
    TARGET_PROFILE_ID,
)


GROUP_EVAL_ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESCORE_SCRIPT = (
    SUPPLEMENT_ROOT / "code/scripts/research/rescore_fast_dllm_humaneval.py"
)
DEFAULT_SANITIZER = SUPPLEMENT_ROOT / "fast-dllm/llada/sanitize.py"
DEFAULT_INPUT_SCHEMA = GROUP_EVAL_ROOT / "schemas/conditional_group_input.schema.json"
DEFAULT_PROMPT = GROUP_EVAL_ROOT / "prompts/group_eval_conditional_task.md"
DEFAULT_OUTPUT_SCHEMA = GROUP_EVAL_ROOT / "schemas/conditional_geq_output.schema.json"
DEFAULT_PROFILE = GROUP_EVAL_ROOT / "protocol/target_profile_large_dlm_tasks.md"

MODELS = ("llada", "dream")
MODEL_CHECKPOINTS = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "dream": "Dream-org/Dream-v0-Base-7B",
}
MODEL_REVISIONS = {
    "llada": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
    "dream": "6572adb5535263e4d1a337b56942ba48b6dee2a9",
}
EXPECTED_PASS_COUNTS = {
    "gsm8k": {"llada": (207, 256), "dream": (193, 256)},
    "humaneval": {"llada": (66, 164), "dream": (84, 164)},
}
SOURCE_RELATIVE_PATHS = {
    "gsm8k_llada": (
        "llada_gsm8k_baseline_default_t0_seed0_len256_block32/"
        "generated/rank_0.jsonl"
    ),
    "gsm8k_dream": (
        "dream_gsm8k_baseline_default_t0_seed0_len256_block32/"
        "generated/rank_0.jsonl"
    ),
    "humaneval_llada": (
        "llada_humaneval_baseline_default_t0_seed0_len256_block32/"
        "lm_eval/GSAI-ML__LLaDA-8B-Instruct/"
        "samples_humaneval_2026-06-12T07-40-55.693862.jsonl"
    ),
    "humaneval_dream": (
        "dream_humaneval_baseline_default_t0_seed0_len256_block32/"
        "lm_eval/Dream-org__Dream-v0-Base-7B/"
        "samples_humaneval_2026-06-12T16-25-07.617040.jsonl"
    ),
}
EXPECTED_SOURCE_SHA256 = {
    "gsm8k_llada": "ad36c6af2760f8dce1284f3272ad7fa7108e681b3d8b23fd401a7ca628879721",
    "gsm8k_dream": "2cfbc1d0282a7e1a194b3810d74a3feb6be699bbd11898ae4a7c7060beb08897",
    "humaneval_llada": "4084d19b38bb28a6f832ed29a609e998a14e2a1a616319a06429ed09949f18ce",
    "humaneval_dream": "2b8dedb8092ea934cedcefc382c6928bf090c57f73425b1dcbe2911800fe625b",
    "gsm8k_arrow": "45965b000311d1550e5619b60b5bf31cf76edebfd8b8eddc62a876fbf8c9be95",
    "rescore_script": "37bc04528177660f6a4cf461c6c120142e257cbb2ace21ff9177baa1ca24c8a0",
    "sanitizer": "46146c1ac3b6d34adad89adee410a99b055b451d9b5e0e7514e8e8f50f30b9b7",
}
GSM8K_FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
GSM8K_IGNORE_PATTERNS = (",", r"\$", r"(?s).*#### ", r"\.$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_json_atomic(path: Path, value: Any, mode: int = 0o600) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def write_jsonl_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]], mode: int = 0o600
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def assert_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_path(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def gsm8k_extract(response: str) -> str:
    matches = GSM8K_FLEXIBLE_PATTERN.findall(response)
    if not matches:
        return "[invalid]"
    nonempty = [part for part in matches[-1] if part]
    return nonempty[0].strip() if nonempty else "[invalid]"


def gsm8k_normalize(value: str) -> str:
    normalized = value
    for pattern in GSM8K_IGNORE_PATTERNS:
        normalized = re.sub(pattern, "", normalized)
    return normalized.lower()


def gsm8k_pass(response: str, reference: str) -> bool:
    return gsm8k_normalize(gsm8k_extract(response)) == gsm8k_normalize(reference)


def load_gsm8k(path: Path) -> list[dict[str, str]]:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "prepare_large_dlm_conditional.py requires the existing Fast-dLLM "
            "environment, which contains datasets/pyarrow"
        ) from exc
    dataset = Dataset.from_file(str(path))
    rows = [
        {"question": str(dataset[i]["question"]), "answer": str(dataset[i]["answer"])}
        for i in range(256)
    ]
    if len(dataset) != 1319 or len(rows) != 256:
        raise ValueError("unexpected GSM8K test-set or evaluation-prefix size")
    return rows


def import_rescorer(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("large_dlm_humaneval_rescore", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import HumanEval rescorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_programs(
    evaluator: Callable[[str, str, float], bool],
    programs: list[str],
    references: list[str],
    workers: int,
    timeout: float,
) -> list[bool]:
    def one(pair: tuple[str, str]) -> bool:
        return bool(evaluator(pair[0], pair[1], timeout))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(one, zip(programs, references, strict=True)))


def assert_pass_counts(labels: Mapping[str, Mapping[str, list[bool]]]) -> None:
    for task, by_model in EXPECTED_PASS_COUNTS.items():
        for model, (expected_passed, expected_n) in by_model.items():
            observed = labels[task][model]
            if len(observed) != expected_n or sum(observed) != expected_passed:
                raise ValueError(
                    f"{task}/{model} deterministic score mismatch: expected "
                    f"{expected_passed}/{expected_n}, observed "
                    f"{sum(observed)}/{len(observed)}"
                )


def validate_prepared_group(group: Mapping[str, Any]) -> None:
    """Validate the small fixed input contract without adding an env dependency."""

    if group.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("prepared group has wrong protocol_version")
    if group.get("target_profile_id") != TARGET_PROFILE_ID:
        raise ValueError("prepared group has wrong target_profile_id")
    if group.get("task_name") not in {"gsm8k", "humaneval"}:
        raise ValueError("prepared group has unsupported task_name")
    condition = group.get("condition")
    if not isinstance(condition, Mapping) or not str(condition.get("text", "")).strip():
        raise ValueError("prepared group requires a nonempty condition")
    samples = group.get("samples")
    if not isinstance(samples, list) or len(samples) != 2:
        raise ValueError("prepared group requires exactly two responses")
    ids = [sample.get("sample_id") for sample in samples if isinstance(sample, Mapping)]
    if set(ids) != {"S1", "S2"} or len(ids) != 2:
        raise ValueError("prepared group requires unique S1/S2 display IDs")
    if any(not isinstance(sample.get("response"), str) for sample in samples):
        raise ValueError("prepared response must be a string")


def group_id(task: str, condition_id: str, condition_sha: str) -> str:
    payload = canonical_json_bytes(
        {
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "task": task,
            "condition_id": condition_id,
            "condition_sha256": condition_sha,
        }
    )
    return "CGE-" + sha256_bytes(payload)[:20]


def build_groups(
    task_rows: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []
    for task in ("gsm8k", "humaneval"):
        for row in task_rows[task]:
            condition_id = str(row["condition_id"])
            condition_text = str(row["condition"])
            condition_sha = sha256_bytes(condition_text.encode("utf-8"))
            gid = group_id(task, condition_id, condition_sha)
            order_seed = stable_seed(EVALUATION_PROTOCOL, task, condition_id)
            model_order = list(MODELS)
            random.Random(order_seed).shuffle(model_order)
            sample_map: list[dict[str, Any]] = []
            samples: list[dict[str, str]] = []
            labels: list[dict[str, Any]] = []
            for position, model in enumerate(model_order, 1):
                display_id = f"S{position}"
                response = str(row["responses"][model])
                sample_map.append(
                    {
                        "display_id": display_id,
                        "model_code": model,
                        "condition_id": condition_id,
                        "response_sha256": sha256_bytes(response.encode("utf-8")),
                    }
                )
                samples.append({"sample_id": display_id, "response": response})
                labels.append(
                    {
                        "display_id": display_id,
                        "model_code": model,
                        "deterministic_pass": bool(row["labels"][model]),
                    }
                )
            metadata = {
                "protocol_version": PROTOCOL_VERSION,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "group_id": gid,
                "task_name": task,
                "condition_id": condition_id,
                "condition_sha256": condition_sha,
                "target_profile_id": TARGET_PROFILE_ID,
                "order_seed": order_seed,
                "sample_map": sample_map,
                "response_representation": (
                    "raw_generation" if task == "gsm8k" else "sanitized_complete_program"
                ),
            }
            groups.append(
                {
                    "request_metadata": metadata,
                    "group_input": {
                        "protocol_version": PROTOCOL_VERSION,
                        "group_id": gid,
                        "target_profile_id": TARGET_PROFILE_ID,
                        "task_name": task,
                        "condition": {
                            "condition_id": condition_id,
                            "text": condition_text,
                        },
                        "samples": samples,
                    },
                }
            )
            private_records.append(
                {
                    "group_id": gid,
                    "task_name": task,
                    "condition_id": condition_id,
                    "labels": labels,
                }
            )
    return groups, {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "records": private_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gsm8k-arrow", type=Path, required=True)
    parser.add_argument("--rescore-script", type=Path, default=DEFAULT_RESCORE_SCRIPT)
    parser.add_argument("--sanitizer", type=Path, default=DEFAULT_SANITIZER)
    parser.add_argument("--input-schema", type=Path, default=DEFAULT_INPUT_SCHEMA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--execution-timeout", type=float, default=3.0)
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="Acknowledge guarded execution of HumanEval candidate programs",
    )
    args = parser.parse_args()
    if not args.allow_code_execution:
        raise RuntimeError(
            "HumanEval label recreation executes generated code; rerun with "
            "--allow-code-execution only in the existing isolated environment"
        )
    if args.workers < 1 or args.execution_timeout <= 0:
        raise ValueError("workers and execution-timeout must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = {
        name: args.run_root / relative
        for name, relative in SOURCE_RELATIVE_PATHS.items()
    }
    source_hashes = {
        name: assert_sha(path, EXPECTED_SOURCE_SHA256[name], name)
        for name, path in source_paths.items()
    }
    source_hashes["gsm8k_arrow"] = assert_sha(
        args.gsm8k_arrow, EXPECTED_SOURCE_SHA256["gsm8k_arrow"], "gsm8k_arrow"
    )
    source_hashes["rescore_script"] = assert_sha(
        args.rescore_script,
        EXPECTED_SOURCE_SHA256["rescore_script"],
        "rescore_script",
    )
    source_hashes["sanitizer"] = assert_sha(
        args.sanitizer, EXPECTED_SOURCE_SHA256["sanitizer"], "sanitizer"
    )

    gsm8k = load_gsm8k(args.gsm8k_arrow)
    gsm_responses: dict[str, list[str]] = {}
    gsm_labels: dict[str, list[bool]] = {}
    for model in MODELS:
        rows = read_jsonl(source_paths[f"gsm8k_{model}"])
        if len(rows) != 256 or any(not isinstance(value, str) for value in rows):
            raise ValueError(f"unexpected GSM8K response file for {model}")
        gsm_responses[model] = [str(value) for value in rows]
        gsm_labels[model] = [
            gsm8k_pass(response, gsm8k[index]["answer"])
            for index, response in enumerate(gsm_responses[model])
        ]

    rescorer = import_rescorer(args.rescore_script)
    imported_sanitizer_path = Path(rescorer.SANITIZER_PATH).resolve()
    if imported_sanitizer_path != args.sanitizer.resolve():
        raise ValueError(
            "the imported rescorer would use a different sanitizer: "
            f"{imported_sanitizer_path} != {args.sanitizer.resolve()}"
        )
    sanitizer = rescorer._load_sanitize()
    human_rows = {
        model: read_jsonl(source_paths[f"humaneval_{model}"]) for model in MODELS
    }
    for model, rows in human_rows.items():
        if len(rows) != 164 or any(not isinstance(value, dict) for value in rows):
            raise ValueError(f"unexpected HumanEval sample file for {model}")
    for index, (llada_row, dream_row) in enumerate(
        zip(human_rows["llada"], human_rows["dream"], strict=True)
    ):
        identity_fields = ("doc_id", "doc_hash", "prompt_hash")
        if any(llada_row[field] != dream_row[field] for field in identity_fields):
            raise ValueError(f"HumanEval model alignment mismatch at row {index}")
        if llada_row["doc"]["prompt"] != dream_row["doc"]["prompt"]:
            raise ValueError(f"HumanEval condition mismatch at row {index}")

    human_programs: dict[str, list[str]] = {}
    human_references: dict[str, list[str]] = {}
    human_labels: dict[str, list[bool]] = {}
    for model in MODELS:
        run_dir = args.run_root / (
            f"{model}_humaneval_baseline_default_t0_seed0_len256_block32"
        )
        selected_sample_path = rescorer._sample_path(run_dir)
        if selected_sample_path is None or (
            selected_sample_path.resolve()
            != source_paths[f"humaneval_{model}"].resolve()
        ):
            raise ValueError(
                f"{model}: rescorer-selected HumanEval log is not the hash-locked source"
            )
        programs, references = rescorer._load_run(run_dir, sanitizer)
        if len(programs) != 164 or len(references) != 164:
            raise ValueError(f"unexpected sanitized HumanEval count for {model}")
        human_programs[model] = programs
        human_references[model] = references
        human_labels[model] = score_programs(
            rescorer._evaluate_program,
            programs,
            references,
            args.workers,
            args.execution_timeout,
        )
    if human_references["llada"] != human_references["dream"]:
        raise ValueError("HumanEval functional tests differ across model sample logs")

    all_labels = {"gsm8k": gsm_labels, "humaneval": human_labels}
    assert_pass_counts(all_labels)

    task_rows: dict[str, list[dict[str, Any]]] = {"gsm8k": [], "humaneval": []}
    for index in range(256):
        task_rows["gsm8k"].append(
            {
                "condition_id": f"gsm8k:{index}",
                "condition": gsm8k[index]["question"],
                "responses": {model: gsm_responses[model][index] for model in MODELS},
                "labels": {model: gsm_labels[model][index] for model in MODELS},
            }
        )
    for index in range(164):
        prompt = str(human_rows["llada"][index]["doc"]["prompt"])
        task_rows["humaneval"].append(
            {
                "condition_id": f"humaneval:{index}",
                "condition": prompt,
                "responses": {model: human_programs[model][index] for model in MODELS},
                "labels": {model: human_labels[model][index] for model in MODELS},
            }
        )

    groups, private_labels = build_groups(task_rows)
    for index, record in enumerate(groups):
        try:
            validate_prepared_group(record["group_input"])
        except ValueError as exc:
            raise ValueError(f"prepared group {index} is invalid: {exc}") from exc

    groups_path = args.output_dir / "conditional_groups.jsonl"
    labels_path = args.output_dir / "private_label_map.json"
    provenance_path = args.output_dir / "generation_provenance.json"
    summary_path = args.output_dir / "preparation_summary.json"
    write_jsonl_atomic(groups_path, groups)
    write_json_atomic(labels_path, private_labels)

    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "model_checkpoints": MODEL_CHECKPOINTS,
        "cached_model_revisions": MODEL_REVISIONS,
        "generation_operating_point": {
            "temperature": 0.0,
            "generation_seed": 0,
            "maximum_new_tokens": 256,
            "baseline_diffusion_steps": 256,
            "block_length": 32,
        },
        "tasks": {
            "gsm8k": {
                "conditions": 256,
                "num_fewshot": 5,
                "response_representation": "raw generation",
                "deterministic_label": "lm-eval v3 flexible-extract exact match",
            },
            "humaneval": {
                "conditions": 164,
                "num_fewshot": 0,
                "response_representation": "Fast-dLLM sanitized complete program",
                "deterministic_label": "guarded functional pass@1",
            },
        },
        "source_sha256": source_hashes,
        "resource_sha256": {
            "prompt": sha256_path(DEFAULT_PROMPT),
            "input_schema": sha256_path(args.input_schema),
            "output_schema": sha256_path(DEFAULT_OUTPUT_SCHEMA),
            "target_profile": sha256_path(DEFAULT_PROFILE),
        },
        "privacy": {
            "judge_inputs_include_reference_answers": False,
            "judge_inputs_include_humaneval_tests": False,
            "judge_inputs_include_canonical_solutions": False,
            "binary_labels_stored_only_in_private_label_map": True,
        },
    }
    write_json_atomic(provenance_path, provenance)

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "groups": len(groups),
        "candidate_responses": 2 * len(groups),
        "tasks": {
            task: {
                "conditions": len(task_rows[task]),
                "deterministic_accuracy": {
                    model: {
                        "passed": sum(all_labels[task][model]),
                        "total": len(all_labels[task][model]),
                        "accuracy": sum(all_labels[task][model])
                        / len(all_labels[task][model]),
                    }
                    for model in MODELS
                },
            }
            for task in ("gsm8k", "humaneval")
        },
        "outputs": {
            "groups": groups_path.name,
            "private_labels": labels_path.name,
            "provenance": provenance_path.name,
        },
    }
    write_json_atomic(summary_path, summary, mode=0o644)

    manifest_paths = (groups_path, labels_path, provenance_path, summary_path)
    manifest = args.output_dir / "SHA256SUMS"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        for path in manifest_paths:
            handle.write(f"{sha256_path(path)}  {path.name}\n")
    os.chmod(manifest, 0o644)

    print(
        json.dumps(
            {
                "groups": len(groups),
                "responses": 2 * len(groups),
                "gsm8k_passed": {
                    model: sum(gsm_labels[model]) for model in MODELS
                },
                "humaneval_passed": {
                    model: sum(human_labels[model]) for model in MODELS
                },
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
