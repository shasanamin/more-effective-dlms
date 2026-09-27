#!/usr/bin/env python3
"""Rescore archived generations with one pinned causal-LM evaluator.

The input manifest, source generations, and per-output artifact are private.
The aggregate summary contains hashes and non-sensitive diagnostics only and
can be packaged for reviewers after inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
import torch.nn.functional as F


GENERATOR_SENTINEL = "<|endoftext|>"
OUTPUT_ROOT_PLACEHOLDER = "${OUTPUT_ROOT}"
SpanPolicy = Literal["first_document", "full_stream_segmented"]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def safe_exp(value: float) -> float:
    return math.exp(min(value, 700.0))


def optional_exp(value: float | None) -> float | None:
    return safe_exp(value) if value is not None else None


def optional_ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class SourceSegment:
    content: str
    boundary_after: bool


@dataclass(frozen=True)
class EncodedSegment:
    input_ids: list[int]
    content_sha256: str
    content_character_count: int
    content_utf8_bytes: int
    content_tokens_before_limit: int
    scored_content_token_count: int
    scored_content_character_count: int
    scored_content_utf8_bytes: int
    source_boundary_after: bool
    boundary_scored: bool
    truncated: bool


@dataclass(frozen=True)
class EncodedOutput:
    segments: tuple[EncodedSegment, ...]
    raw_text_sha256: str
    raw_character_count: int
    raw_utf8_bytes: int
    leading_sentinel_removed: bool
    source_segment_count: int
    included_segment_count: int
    source_boundary_count: int
    included_boundary_count: int
    source_content_character_count: int
    source_content_utf8_bytes: int
    included_content_character_count: int
    included_content_utf8_bytes: int


def split_generator_stream(text: str) -> tuple[bool, tuple[SourceSegment, ...]]:
    """Remove one leading primer sentinel and retain every generated boundary."""
    leading = text.startswith(GENERATOR_SENTINEL)
    body = text[len(GENERATOR_SENTINEL) :] if leading else text
    fragments = body.split(GENERATOR_SENTINEL)
    segments = [
        SourceSegment(content=fragment, boundary_after=index < len(fragments) - 1)
        for index, fragment in enumerate(fragments)
    ]
    # Preserve empty fragments, including the unscorable fragment after a
    # terminal boundary.  This keeps segment and boundary accounting faithful
    # without changing the scored targets: the boundary remains attached to
    # its preceding fragment.
    return leading, tuple(segments)


def _tokenize_with_offsets(
    content: str,
    tokenizer: Any,
) -> tuple[list[int], list[tuple[int, int]]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "causal-LM rescoring requires a fast tokenizer with offset mappings "
            "so bits-per-byte has an exact scored-text denominator"
        )
    encoded = tokenizer(
        content,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    if len(ids) != len(offsets):
        raise ValueError("tokenizer returned mismatched token and offset counts")
    previous_start = 0
    for start, end in offsets:
        if start < 0 or end < start or end > len(content) or start < previous_start:
            raise ValueError("tokenizer returned an invalid character offset mapping")
        previous_start = start
    return ids, offsets


def _avoid_partial_character_cut(
    token_count: int,
    offsets: list[tuple[int, int]],
) -> int:
    """Back off if a byte-level tokenizer cut through one Unicode character."""
    while (
        token_count > 0
        and token_count < len(offsets)
        and offsets[token_count - 1][1] > offsets[token_count][0]
    ):
        token_count -= 1
    return token_count


def encode_segment(
    source: SourceSegment,
    tokenizer: Any,
    max_length: int,
) -> EncodedSegment:
    content_ids, offsets = _tokenize_with_offsets(source.content, tokenizer)
    primer = tokenizer.bos_token_id
    if primer is None:
        primer = tokenizer.eos_token_id
    if primer is None:
        raise ValueError("evaluator tokenizer has neither BOS nor EOS token")
    eos = tokenizer.eos_token_id
    if source.boundary_after and eos is None:
        raise ValueError("evaluator tokenizer has no EOS token for a source boundary")

    target_capacity = max_length - 1
    full_target_count = len(content_ids) + int(source.boundary_after)
    if full_target_count <= target_capacity:
        scored_content_tokens = len(content_ids)
        boundary_scored = source.boundary_after
    else:
        scored_content_tokens = min(len(content_ids), target_capacity)
        scored_content_tokens = _avoid_partial_character_cut(
            scored_content_tokens, offsets
        )
        boundary_scored = False

    scored_ids = content_ids[:scored_content_tokens]
    if scored_content_tokens:
        scored_character_count = max(end for _, end in offsets[:scored_content_tokens])
        if scored_character_count == 0:
            raise ValueError(
                "tokenizer assigned only zero-width offsets to scored content tokens"
            )
    else:
        scored_character_count = 0
    scored_content = source.content[:scored_character_count]
    sequence = [int(primer), *scored_ids]
    if boundary_scored:
        sequence.append(int(eos))
    if len(sequence) > max_length:
        raise AssertionError("encoded segment exceeds the evaluator length")

    return EncodedSegment(
        input_ids=sequence,
        content_sha256=sha256_bytes(source.content.encode("utf-8")),
        content_character_count=len(source.content),
        content_utf8_bytes=len(source.content.encode("utf-8")),
        content_tokens_before_limit=len(content_ids),
        scored_content_token_count=scored_content_tokens,
        scored_content_character_count=scored_character_count,
        scored_content_utf8_bytes=len(scored_content.encode("utf-8")),
        source_boundary_after=source.boundary_after,
        boundary_scored=boundary_scored,
        truncated=(
            scored_content_tokens < len(content_ids)
            or (source.boundary_after and not boundary_scored)
        ),
    )


def encode_output(
    text: str,
    tokenizer: Any,
    max_length: int,
    span_policy: SpanPolicy,
) -> EncodedOutput:
    leading, source_segments = split_generator_stream(text)
    if span_policy == "first_document":
        included_sources = source_segments[:1]
    elif span_policy == "full_stream_segmented":
        included_sources = source_segments
    else:  # pragma: no cover - argparse and the type alias constrain callers.
        raise ValueError(f"unsupported span policy: {span_policy}")
    encoded = tuple(
        encode_segment(segment, tokenizer, max_length)
        for segment in included_sources
    )
    return EncodedOutput(
        segments=encoded,
        raw_text_sha256=sha256_bytes(text.encode("utf-8")),
        raw_character_count=len(text),
        raw_utf8_bytes=len(text.encode("utf-8")),
        leading_sentinel_removed=leading,
        source_segment_count=len(source_segments),
        included_segment_count=len(encoded),
        source_boundary_count=sum(segment.boundary_after for segment in source_segments),
        included_boundary_count=sum(
            segment.source_boundary_after for segment in encoded
        ),
        source_content_character_count=sum(
            len(segment.content) for segment in source_segments
        ),
        source_content_utf8_bytes=sum(
            len(segment.content.encode("utf-8")) for segment in source_segments
        ),
        included_content_character_count=sum(
            segment.content_character_count for segment in encoded
        ),
        included_content_utf8_bytes=sum(
            segment.content_utf8_bytes for segment in encoded
        ),
    )


def bootstrap_mean_interval(
    values: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    cells = value.get("cells") if isinstance(value, dict) else None
    if not isinstance(cells, list) or not cells:
        raise ValueError("manifest must contain a nonempty cells array")
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("every manifest cell must be an object")
        required = {"cell_id", "source_path", "samples_field", "expected_count"}
        missing = sorted(required - set(cell))
        if missing:
            raise ValueError(f"manifest cell is missing {missing}")
        cell_id = str(cell["cell_id"])
        if cell_id in seen:
            raise ValueError(f"duplicate cell_id: {cell_id}")
        seen.add(cell_id)
        if int(cell["expected_count"]) < 1:
            raise ValueError(f"manifest cell {cell_id} has a nonpositive expected_count")
        expected_sha = cell.get("source_sha256_expected")
        if expected_sha is not None and not re.fullmatch(
            r"[0-9a-fA-F]{64}", str(expected_sha)
        ):
            raise ValueError(
                f"manifest cell {cell_id} has an invalid source_sha256_expected"
            )
    return cells


def resolve_source_path(
    source_path: str,
    manifest_path: Path,
    output_root: Path | None,
) -> Path:
    """Resolve a literal path or the one allowed portable root placeholder."""
    if OUTPUT_ROOT_PLACEHOLDER in source_path:
        root = output_root
        if root is None:
            environment_root = os.environ.get("OUTPUT_ROOT")
            if environment_root:
                root = Path(environment_root)
        if root is None:
            raise ValueError(
                f"source path uses {OUTPUT_ROOT_PLACEHOLDER}, but neither "
                "--output-root nor the OUTPUT_ROOT environment variable is set"
            )
        source_path = source_path.replace(OUTPUT_ROOT_PLACEHOLDER, str(root))
    if "${" in source_path or "$" in source_path:
        raise ValueError(
            "unsupported environment placeholder in source_path; only "
            f"{OUTPUT_ROOT_PLACEHOLDER} is allowed"
        )
    resolved = Path(source_path)
    if not resolved.is_absolute():
        resolved = manifest_path.resolve().parent / resolved
    return resolved.resolve()


def verify_source_sha256(cell: dict[str, Any], source: Path, source_bytes: bytes) -> str:
    actual = sha256_bytes(source_bytes)
    expected = cell.get("source_sha256_expected")
    if expected is not None and actual != str(expected).lower():
        raise ValueError(
            f"{source}: SHA-256 mismatch for cell {cell['cell_id']}; "
            f"expected {expected}, found {actual}"
        )
    return actual


def _resolved_commit_hash(value: Any) -> str | None:
    candidate = getattr(value, "_commit_hash", None)
    if candidate is None and hasattr(value, "init_kwargs"):
        candidate = value.init_kwargs.get("_commit_hash")
    return str(candidate).lower() if candidate else None


def _assert_resolved_revision(value: Any, requested: str, component: str) -> str | None:
    resolved = _resolved_commit_hash(value)
    if resolved is not None and resolved != requested.lower():
        raise RuntimeError(
            f"{component} resolved revision {resolved}, not requested {requested}"
        )
    return resolved


def _score_segments(
    flattened: list[tuple[int, int, EncodedSegment]],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int,
) -> dict[tuple[int, int], dict[str, float | int | None]]:
    scores: dict[tuple[int, int], dict[str, float | int | None]] = {}
    scorable = [item for item in flattened if len(item[2].input_ids) >= 2]
    for start in range(0, len(scorable), batch_size):
        batch = scorable[start : start + batch_size]
        width = max(len(item[2].input_ids) for item in batch)
        input_ids = torch.full(
            (len(batch), width),
            int(tokenizer.pad_token_id),
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros(
            (len(batch), width), dtype=torch.long, device=device
        )
        for row_index, (_, _, item) in enumerate(batch):
            length = len(item.input_ids)
            input_ids[row_index, :length] = torch.tensor(
                item.input_ids, dtype=torch.long, device=device
            )
            attention[row_index, :length] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits
            losses = F.cross_entropy(
                logits[:, :-1, :].float().transpose(1, 2),
                input_ids[:, 1:],
                reduction="none",
            )
        for row_index, (output_index, segment_index, item) in enumerate(batch):
            token_count = len(item.input_ids) - 1
            loss_values = losses[row_index, :token_count]
            content_count = item.scored_content_token_count
            expected = content_count + int(item.boundary_scored)
            if expected != token_count:
                raise AssertionError("encoded target accounting is inconsistent")
            content_nll = float(loss_values[:content_count].sum().item())
            boundary_nll = (
                float(loss_values[content_count].item())
                if item.boundary_scored
                else 0.0
            )
            nll_sum = content_nll + boundary_nll
            mean_nll = nll_sum / token_count
            scores[(output_index, segment_index)] = {
                "token_count": token_count,
                "nll_sum": nll_sum,
                "content_nll_sum": content_nll,
                "boundary_nll_sum": boundary_nll,
                "mean_nll": mean_nll,
                "perplexity": safe_exp(mean_nll),
            }
        del logits, losses, input_ids, attention
    return scores


def _output_row(
    cell_id: str,
    output_index: int,
    encoded: EncodedOutput,
    scores: dict[tuple[int, int], dict[str, float | int | None]],
) -> dict[str, Any]:
    segment_rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(encoded.segments):
        score = scores.get(
            (output_index, segment_index),
            {
                "token_count": 0,
                "nll_sum": 0.0,
                "content_nll_sum": 0.0,
                "boundary_nll_sum": 0.0,
                "mean_nll": None,
                "perplexity": None,
            },
        )
        segment_rows.append({"segment_index": segment_index, **asdict(segment), **score})

    token_count = sum(int(row["token_count"]) for row in segment_rows)
    nll_sum = sum(float(row["nll_sum"]) for row in segment_rows)
    content_nll_sum = sum(float(row["content_nll_sum"]) for row in segment_rows)
    boundary_nll_sum = sum(float(row["boundary_nll_sum"]) for row in segment_rows)
    mean_nll = optional_ratio(nll_sum, token_count)
    scored_content_bytes = sum(
        segment.scored_content_utf8_bytes for segment in encoded.segments
    )
    return {
        "cell_id": cell_id,
        "output_index": output_index,
        "output_id": f"{cell_id}:{output_index:03d}",
        "raw_text_sha256": encoded.raw_text_sha256,
        "raw_character_count": encoded.raw_character_count,
        "raw_utf8_bytes": encoded.raw_utf8_bytes,
        "leading_sentinel_removed": encoded.leading_sentinel_removed,
        "source_segment_count": encoded.source_segment_count,
        "included_segment_count": encoded.included_segment_count,
        "scorable_segment_count": sum(
            int(row["token_count"] > 0) for row in segment_rows
        ),
        "source_boundary_count": encoded.source_boundary_count,
        "included_boundary_count": encoded.included_boundary_count,
        "scored_boundary_count": sum(segment.boundary_scored for segment in encoded.segments),
        "truncated_segment_count": sum(segment.truncated for segment in encoded.segments),
        "source_content_character_count": encoded.source_content_character_count,
        "source_content_utf8_bytes": encoded.source_content_utf8_bytes,
        "included_content_character_count": encoded.included_content_character_count,
        "included_content_utf8_bytes": encoded.included_content_utf8_bytes,
        "scored_content_character_count": sum(
            segment.scored_content_character_count for segment in encoded.segments
        ),
        "scored_content_utf8_bytes": scored_content_bytes,
        "token_count": token_count,
        "nll_sum": nll_sum,
        "content_nll_sum": content_nll_sum,
        "boundary_nll_sum": boundary_nll_sum,
        "mean_nll": mean_nll,
        "perplexity": optional_exp(mean_nll),
        "content_bits_per_utf8_byte": optional_ratio(
            content_nll_sum / math.log(2.0), scored_content_bytes
        ),
        "segments": segment_rows,
    }


def _summarize_rows(
    rows: list[dict[str, Any]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    scorable = [row for row in rows if row["mean_nll"] is not None]
    means = np.asarray([row["mean_nll"] for row in scorable], dtype=np.float64)
    token_count = sum(int(row["token_count"]) for row in rows)
    nll_sum = sum(float(row["nll_sum"]) for row in rows)
    content_nll_sum = sum(float(row["content_nll_sum"]) for row in rows)
    scored_content_bytes = sum(int(row["scored_content_utf8_bytes"]) for row in rows)
    token_micro_mean = optional_ratio(nll_sum, token_count)
    output_macro_mean = float(means.mean()) if means.size else None
    low, high = bootstrap_mean_interval(means, bootstrap_replicates, bootstrap_seed)
    return {
        "output_count": len(rows),
        "scorable_output_count": len(scorable),
        "source_segment_count": sum(int(row["source_segment_count"]) for row in rows),
        "included_segment_count": sum(int(row["included_segment_count"]) for row in rows),
        "scorable_segment_count": sum(int(row["scorable_segment_count"]) for row in rows),
        "source_boundary_count": sum(int(row["source_boundary_count"]) for row in rows),
        "included_boundary_count": sum(int(row["included_boundary_count"]) for row in rows),
        "scored_boundary_count": sum(int(row["scored_boundary_count"]) for row in rows),
        "truncated_segment_count": sum(int(row["truncated_segment_count"]) for row in rows),
        "leading_sentinel_removed_count": sum(
            bool(row["leading_sentinel_removed"]) for row in rows
        ),
        "raw_character_count": sum(int(row["raw_character_count"]) for row in rows),
        "raw_utf8_bytes": sum(int(row["raw_utf8_bytes"]) for row in rows),
        "source_content_character_count": sum(
            int(row["source_content_character_count"]) for row in rows
        ),
        "source_content_utf8_bytes": sum(
            int(row["source_content_utf8_bytes"]) for row in rows
        ),
        "included_content_character_count": sum(
            int(row["included_content_character_count"]) for row in rows
        ),
        "included_content_utf8_bytes": sum(
            int(row["included_content_utf8_bytes"]) for row in rows
        ),
        "scored_content_character_count": sum(
            int(row["scored_content_character_count"]) for row in rows
        ),
        "scored_content_utf8_bytes": scored_content_bytes,
        "scored_evaluator_token_count": token_count,
        "nll_sum": nll_sum,
        "content_nll_sum_for_bpb": content_nll_sum,
        "token_micro_mean_nll": token_micro_mean,
        "token_micro_perplexity": optional_exp(token_micro_mean),
        "output_macro_mean_nll": output_macro_mean,
        "output_macro_geometric_perplexity": optional_exp(output_macro_mean),
        "output_macro_mean_nll_bootstrap_95": [low, high],
        "output_macro_geometric_perplexity_bootstrap_95": [
            optional_exp(low),
            optional_exp(high),
        ],
        "corpus_bits_per_utf8_byte": optional_ratio(
            content_nll_sum / math.log(2.0), scored_content_bytes
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--span-policy",
        required=True,
        choices=("first_document", "full_stream_segmented"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="replacement for ${OUTPUT_ROOT}; overrides the environment variable",
    )
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.max_length < 2:
        raise ValueError("max length must be at least two")
    if args.bootstrap_replicates < 1000:
        raise ValueError("use at least 1,000 bootstrap replicates")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("--revision must be an exact 40-character commit hash")
    if args.private_output.resolve() == args.summary.resolve():
        raise ValueError("private output and summary paths must differ")
    if args.private_output.exists() or args.summary.exists():
        raise FileExistsError("refusing to overwrite an existing output artifact")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cells = load_manifest(args.manifest)

    # Resolve and validate every input before allocating a model or scoring any
    # output.  This prevents a late stale-artifact failure from wasting a run.
    prepared_cells: list[tuple[dict[str, Any], bytes, str, list[str]]] = []
    for cell in cells:
        source = resolve_source_path(
            str(cell["source_path"]), args.manifest, args.output_root
        )
        source_bytes = source.read_bytes()
        source_sha = verify_source_sha256(cell, source, source_bytes)
        source_value = json.loads(source_bytes)
        texts = source_value.get(str(cell["samples_field"]))
        if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
            raise ValueError(
                f"{source}: field {cell['samples_field']!r} is not a string list"
            )
        expected = int(cell["expected_count"])
        if len(texts) != expected:
            raise ValueError(f"{source}: expected {expected} outputs, found {len(texts)}")
        prepared_cells.append((cell, source_bytes, source_sha, texts))

    # Model loading is optional for CPU-only accounting tests and --help.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer_commit = _assert_resolved_revision(tokenizer, args.revision, "tokenizer")
    if not tokenizer.is_fast:
        raise ValueError("the selected evaluator did not load a fast tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation="sdpa",
    ).eval()
    model_commit = _assert_resolved_revision(model.config, args.revision, "model")
    model.to(args.device)
    model.config.use_cache = False

    private_rows: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []

    for cell_index, (cell, source_bytes, source_sha, texts) in enumerate(prepared_cells):
        encoded = [
            encode_output(text, tokenizer, args.max_length, args.span_policy)
            for text in texts
        ]
        flattened = [
            (output_index, segment_index, segment)
            for output_index, output in enumerate(encoded)
            for segment_index, segment in enumerate(output.segments)
        ]
        segment_scores = _score_segments(
            flattened, model, tokenizer, args.device, args.batch_size
        )
        cell_rows = [
            _output_row(str(cell["cell_id"]), output_index, output, segment_scores)
            for output_index, output in enumerate(encoded)
        ]
        private_rows.extend(cell_rows)

        public_metadata = {
            key: value
            for key, value in cell.items()
            if key not in {"source_path", "samples_field", "source_sha256_expected"}
        }
        metrics = _summarize_rows(
            cell_rows,
            args.bootstrap_replicates,
            args.seed + 1009 * (cell_index + 1),
        )
        cell_summaries.append(
            {
                **public_metadata,
                "source_sha256": source_sha,
                **metrics,
            }
        )
        print(
            f"[{cell_index + 1}/{len(cells)}] {cell['cell_id']}: "
            f"n={metrics['output_count']} tokens={metrics['scored_evaluator_token_count']} "
            f"token_micro_ppl={metrics['token_micro_perplexity']}",
            flush=True,
        )

    overall_metrics = _summarize_rows(
        private_rows, args.bootstrap_replicates, args.seed + 1_000_003
    )
    manifest_sha = sha256_bytes(args.manifest.read_bytes())
    private_header = {
        "artifact": "private per-output causal-LM gPPL rescoring",
        "protocol_version": "common-causal-gppl-v2",
        "manifest_sha256": manifest_sha,
        "model": args.model,
        "revision": args.revision.lower(),
        "span_policy": args.span_policy,
        "max_length_per_segment_including_primer": args.max_length,
        "seed": args.seed,
    }
    write_jsonl_atomic(
        args.private_output,
        [{"run_header": private_header}, *private_rows],
    )
    private_sha = sha256_bytes(args.private_output.read_bytes())
    summary = {
        "artifact": "reviewer-safe aggregate causal-LM gPPL rescoring",
        "protocol_version": "common-causal-gppl-v2",
        "model": {
            "identifier": args.model,
            "requested_revision": args.revision.lower(),
            "resolved_model_revision": model_commit,
            "resolved_tokenizer_revision": tokenizer_commit,
            "architecture": model.config.model_type,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "dtype": str(dtype).replace("torch.", ""),
            "tokenizer_class": tokenizer.__class__.__name__,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "scoring": {
            "span_policy": args.span_policy,
            "max_evaluator_tokens_per_segment_including_primer": args.max_length,
            "leading_generator_sentinel_policy": (
                "remove at most one literal leading GPT-2 EOS sentinel"
            ),
            "generated_boundary_policy": (
                "map each included literal GPT-2 EOS sentinel to evaluator EOS "
                "when its preceding content and EOS fit; reset causal context "
                "after every included boundary"
            ),
            "primer_policy": (
                "prepend evaluator BOS, falling back to evaluator EOS, to every "
                "segment and do not score the primer"
            ),
            "add_special_tokens": False,
            "token_micro_perplexity_weighting": "all scored evaluator targets, including mapped EOS boundaries",
            "output_macro_weighting": "equal weight per output with at least one scored target",
            "bits_per_byte_definition": (
                "content-token NLL divided by ln(2) and exact UTF-8 bytes of "
                "scored source-content prefixes; mapped EOS losses are excluded"
            ),
            "bootstrap_unit": "output",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.seed,
        },
        "input_manifest_sha256": manifest_sha,
        "private_per_output_artifact_sha256": private_sha,
        "overall": overall_metrics,
        "total_cells": len(cells),
        "cells": cell_summaries,
    }
    write_json_atomic(args.summary, summary)
    print(
        json.dumps(
            {
                "complete": True,
                "cells": len(cells),
                "outputs": overall_metrics["output_count"],
                "segments": overall_metrics["included_segment_count"],
                "scored_tokens": overall_metrics["scored_evaluator_token_count"],
                "private_output": str(args.private_output),
                "summary": str(args.summary),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
