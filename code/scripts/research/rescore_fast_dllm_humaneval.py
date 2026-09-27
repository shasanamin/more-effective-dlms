#!/usr/bin/env python3
"""Rescore Fast-dLLM HumanEval generations with its required sanitizer.

The stock lm-eval HumanEval filter is not the scoring path documented by
Fast-dLLM. Fast-dLLM first combines the prompt and generated response fragment,
then sanitizes that program before functional evaluation. This script
reproduces that path for the saved temperature-sweep generations.

WARNING: HumanEval executes model-generated Python. Run this only inside an
appropriately isolated environment.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
SANITIZER_PATH = ROOT / "fast-dllm" / "llada" / "sanitize.py"
MODELS = ("llada", "dream")
TEMPERATURES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SEEDS = (0, 1, 2)


def _load_sanitize() -> Callable[[str, str | None], str]:
    spec = importlib.util.spec_from_file_location("fast_dllm_sanitize", SANITIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Fast-dLLM sanitizer: {SANITIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sanitize


def _expected_run_names() -> list[str]:
    names: list[str] = []
    for model in MODELS:
        for temperature in TEMPERATURES:
            seeds = (0,) if temperature == 0.0 else SEEDS
            temp_label = "0" if temperature == 0.0 else str(temperature).replace(".", "p")
            sampler = "default" if temperature == 0.0 else "g32"
            for seed in seeds:
                names.append(
                    f"{model}_humaneval_baseline_{sampler}_t{temp_label}_seed{seed}"
                    "_len256_block32"
                )
    return names


def _sample_path(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("lm_eval/*/samples_humaneval_*.jsonl"))
    return paths[-1] if paths else None


def _parse_run_name(run_name: str) -> tuple[str, str, str]:
    model = run_name.split("_humaneval_", 1)[0]
    temperature = run_name.split("_t", 1)[1].split("_seed", 1)[0].replace("p", ".")
    seed = run_name.split("_seed", 1)[1].split("_", 1)[0]
    return model, temperature, seed


def _load_run(
    run_dir: Path, sanitize: Callable[[str, str | None], str]
) -> tuple[list[str], list[str]]:
    sample_path = _sample_path(run_dir)
    if sample_path is None:
        raise FileNotFoundError(f"No completed HumanEval sample log in {run_dir}")

    predictions: list[str] = []
    references: list[str] = []
    with sample_path.open(encoding="utf-8") as handle:
        for line in handle:
            sample: dict[str, Any] = json.loads(line)
            response = sample["resps"][0][0]
            # This is the exact extraction used by Fast-dLLM postprocess_code.py.
            response_fragment = response.split("```python\n", 1)[-1].split("```")[0]
            candidate = sample["doc"]["prompt"] + "\n" + response_fragment
            predictions.append(sanitize(candidate, sample["doc"]["entry_point"]))
            references.append(sample["target"])
    return predictions, references


def _score_run(
    run_dir: Path,
    sanitize: Callable[[str, str | None], str],
    workers: int,
    timeout: float,
) -> dict[str, str | int | float]:
    predictions, references = _load_run(run_dir, sanitize)
    passed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_evaluate_program, program, reference, timeout)
            for program, reference in zip(predictions, references)
        ]
        for future in as_completed(futures):
            passed += int(future.result())

    model, temperature, seed = _parse_run_name(run_dir.name)
    return {
        "run": run_dir.name,
        "model": model,
        "task": "humaneval",
        "temperature": temperature,
        "generation_seed": seed,
        "effective_samples": len(predictions),
        "passed": passed,
        "pass_at_1": passed / len(predictions),
        "nonempty_sanitized_programs": sum(bool(program) for program in predictions),
    }


def _evaluate_program(program: str, reference: str, timeout: float) -> bool:
    if not program:
        return False
    payload = json.dumps({"program": program, "reference": reference})
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _worker_guard() -> None:
    """Disable APIs unnecessary for HumanEval before executing a candidate."""

    builtins.open = None  # type: ignore[assignment]
    builtins.exit = None
    builtins.quit = None
    builtins.help = None
    for name in (
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "chmod",
        "chown",
        "chroot",
        "lchmod",
        "lchown",
        "open",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    shutil.rmtree = None  # type: ignore[assignment]
    shutil.move = None  # type: ignore[assignment]
    shutil.chown = None  # type: ignore[assignment]
    socket.socket = None  # type: ignore[assignment]
    subprocess.Popen = None  # type: ignore[assignment]
    subprocess.run = None  # type: ignore[assignment]
    subprocess.call = None  # type: ignore[assignment]
    subprocess.check_call = None  # type: ignore[assignment]
    subprocess.check_output = None  # type: ignore[assignment]
    for module in ("ctypes", "multiprocessing", "resource", "tkinter"):
        sys.modules[module] = None


def _worker_main() -> int:
    payload = json.loads(sys.stdin.read())
    check_program = payload["program"] + "\n" + payload["reference"]
    try:
        os.chdir(tempfile.gettempdir())
        _worker_guard()
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                exec(check_program, {})
        return 0
    except BaseException:
        return 1


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())

    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--run", action="append", help="Score only this run name; repeatable.")
    args = parser.parse_args()

    sanitize = _load_sanitize()
    rows: list[dict[str, str | int | float]] = []
    for run_name in args.run or _expected_run_names():
        try:
            row = _score_run(args.run_root / run_name, sanitize, args.workers, args.timeout)
        except FileNotFoundError as exc:
            print(f"SKIP {run_name}: {exc}")
            continue
        rows.append(row)
        print(
            f"{run_name}: pass@1={row['pass_at_1']:.6f}, "
            f"passed={row['passed']}/{row['effective_samples']}, "
            f"nonempty={row['nonempty_sanitized_programs']}"
        )

    if not rows:
        raise RuntimeError("No completed HumanEval runs were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
