#!/usr/bin/env python3
"""Render and run GroupEval absolute judge calls with strict structured output.

GenAI-Inst (Chat Completions) and OpenAI (Responses API) share the
same prompt rendering, schema/semantic validation, retry policy, and resumable
JSONL output. Credentials are read only from provider-specific environment
variables; they are never accepted on the command line or written to artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCORING = ROOT / "scoring"
EXPERIMENT = ROOT / "experiment"
if str(SCORING) not in sys.path:
    sys.path.insert(0, str(SCORING))
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
from validate_outputs import semantic_errors  # noqa: E402
from conditional_large_dlm_common import (  # noqa: E402
    conditional_semantic_errors,
    is_conditional_request,
)

DEFAULT_PROVIDER = "inst"
DEFAULT_INST_MODEL = "gpt-oss:120b"
DEFAULT_INST_URL = "https://genai.example.org/api/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-5.4-2026-03-05"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_INST_MIN_REQUEST_INTERVAL = 3.0
INST_OMITTED_SCHEMA_KEYWORDS = ("uniqueItems",)
OPENAI_OMITTED_SCHEMA_KEYWORDS = ("uniqueItems",)
OPENAI_SCHEMA_NORMALIZATIONS = ("infer_type_for_const_or_enum",)
OPENAI_REASONING_EFFORTS = ("default", "none", "low", "medium", "high", "xhigh", "max")
DEFAULT_PROMPT = ROOT / "prompts" / "group_eval_absolute.md"
DEFAULT_SCHEMA = ROOT / "schemas" / "absolute_output.schema.json"
DEFAULT_PROFILE = ROOT / "protocol" / "target_profile_owt1024.md"
RETRY_NOTICE = (
    "\n\nRETRY NOTICE: The previous response was invalid. Return a complete "
    "replacement JSON object following the schema. Do not explain the error."
)
MAX_RETRY_FEEDBACK_ITEMS = 8
MAX_RETRY_FEEDBACK_CHARS = 320
RETRY_INVARIANT_CHECKLIST = """
Mechanical consistency checklist for the complete replacement:
- Emit compact, complete JSON without Markdown or explanatory prose.
- Every valid sample must have invalid_reason=null, a non-null mode_id, and
  exactly one matching membership in the modes array.
- Every invalid sample must have mode_id=null, hybrid=false, four zero quality
  scores, and no membership in the modes array.
- For a valid sample, nondegeneration_structure=0 and
  severe_degeneration=true must occur together.
- List mode IDs contiguously as M1, M2, ... in their first-appearance order.
""".strip()
CONDITIONAL_RETRY_INVARIANT_CHECKLIST = """
Mechanical consistency checklist for the complete replacement:
- Emit compact, complete JSON without Markdown or explanatory prose.
- Return both supplied sample IDs exactly once; do not add modes or model names.
- Every valid sample must have invalid_reason=null.
- Every invalid sample must have an invalid_reason and four zero quality scores.
- For a valid sample, nondegeneration_structure=0 and
  severe_degeneration=true must occur together.
""".strip()


def retry_notice(errors: Iterable[str], conditional: bool = False) -> str:
    """Render bounded local-validation feedback for a replacement-only retry.

    The provider output is never repaired locally.  Giving the judge the failed
    invariants is especially important for schema-constrained models: JSON Schema
    cannot express all GroupEval cross-field rules (for example, mode membership
    or the severe-degeneration flag).  Feedback is compacted and bounded so a
    provider-controlled validation message cannot grow the prompt without limit.
    """
    compact: list[str] = []
    error_list = list(errors)
    for error in error_list[:MAX_RETRY_FEEDBACK_ITEMS]:
        one_line = " ".join(str(error).split())
        if len(one_line) > MAX_RETRY_FEEDBACK_CHARS:
            one_line = one_line[: MAX_RETRY_FEEDBACK_CHARS - 3] + "..."
        compact.append(f"- {one_line}")
    if len(error_list) > MAX_RETRY_FEEDBACK_ITEMS:
        compact.append(
            f"- {len(error_list) - MAX_RETRY_FEEDBACK_ITEMS} additional "
            "validation failure(s) omitted; recheck every record."
        )
    if not compact:
        compact.append("- The response failed local validation; recheck every field.")
    checklist = (
        CONDITIONAL_RETRY_INVARIANT_CHECKLIST
        if conditional
        else RETRY_INVARIANT_CHECKLIST
    )
    return (
        RETRY_NOTICE
        + "\n\nLocal validation failures to correct in the replacement:\n"
        + "\n".join(compact)
        + "\n\n"
        + checklist
        + "\nRecheck every schema and cross-field invariant before returning JSON."
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: each row must be an object")
            yield line_no, value


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically replace a JSONL checkpoint and fsync it before publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    system_marker = "## System message"
    user_marker = "## User message template"
    if system_marker not in text or user_marker not in text:
        raise ValueError(f"{path}: expected System message and User message template sections")
    after_system = text.split(system_marker, 1)[1]
    system, user = after_system.split(user_marker, 1)
    return system.strip(), user.strip()


def strip_profile_title(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def render_samples(samples: list[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if "response" in sample:
            text = str(sample["response"])
            heading = "Candidate response"
        else:
            text = str(sample["text"])
            heading = "Candidate"
        # Use a content-derived nonce and verify that neither boundary occurs inside
        # the candidate. Keep the opaque boundary independent of ``sample_id``:
        # some schema-constrained judges otherwise copy the entire delimiter suffix
        # (nonce plus ID) into the output instead of the required short ID.
        attempt = 0
        while True:
            nonce = hashlib.sha256(
                f"{sample_id}\x00{text}\x00{attempt}".encode("utf-8")
            ).hexdigest()[:20]
            begin = f"<<<GROUPEVAL_DATA_{nonce}_BEGIN>>>"
            end = f"<<<GROUPEVAL_DATA_{nonce}_END>>>"
            if begin not in text and end not in text:
                break
            attempt += 1
        blocks.append(
            f"### {heading} {sample_id}\n"
            f"Output `sample_id` (copy exactly): `{sample_id}`\n"
            f"{begin}\n{text}\n{end}"
        )
    return "\n\n".join(blocks)


def render_condition(condition: Mapping[str, Any]) -> str:
    """Render a task condition in a boundary distinct from candidate responses."""

    condition_id = str(condition["condition_id"])
    text = str(condition["text"])
    if not text.strip():
        raise ValueError("group_input.condition.text must be nonempty")
    attempt = 0
    while True:
        nonce = hashlib.sha256(
            f"{condition_id}\x00{text}\x00{attempt}".encode("utf-8")
        ).hexdigest()[:20]
        begin = f"<<<GROUPEVAL_CONDITION_{nonce}_BEGIN>>>"
        end = f"<<<GROUPEVAL_CONDITION_{nonce}_END>>>"
        if begin not in text and end not in text:
            break
        attempt += 1
    return f"{begin}\n{text}\n{end}"


def render_request(
    record: Mapping[str, Any],
    system_template: str,
    user_template: str,
    target_profile_text: str,
) -> tuple[str, str]:
    group = record.get("group_input")
    if not isinstance(group, Mapping):
        raise ValueError("record.group_input must be an object")
    samples = group.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("group_input.samples must be a nonempty list")
    needs_condition = any(
        token in system_template or token in user_template
        for token in ("{CONDITION_BLOCK}", "{CONDITION_ID}", "{TASK_NAME}")
    )
    condition = group.get("condition")
    if needs_condition and not isinstance(condition, Mapping):
        raise ValueError("conditional prompt requires group_input.condition")
    condition_block = render_condition(condition) if isinstance(condition, Mapping) else ""
    replacements = {
        "{GROUP_SIZE}": str(len(samples)),
        "{GROUP_ID}": str(group["group_id"]),
        "{TARGET_PROFILE_ID}": str(group["target_profile_id"]),
        "{TARGET_PROFILE_TEXT}": target_profile_text,
        "{SAMPLES_BLOCK}": render_samples(samples),
        "{TASK_NAME}": str(group.get("task_name", "")),
        "{CONDITION_ID}": str(condition.get("condition_id", ""))
        if isinstance(condition, Mapping)
        else "",
        "{CONDITION_BLOCK}": condition_block,
    }
    system = system_template
    user = user_template
    for key, value in replacements.items():
        system = system.replace(key, value)
        user = user.replace(key, value)
    unresolved = [token for token in replacements if token in system or token in user]
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders: {unresolved}")
    return system, user


def response_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, Mapping):
        return dict(usage)
    return {key: getattr(usage, key) for key in dir(usage) if key.endswith("tokens")}


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text
    # Defensive fallback for SDK variants.
    output = getattr(response, "output", None)
    if output:
        parts: list[str] = []
        for item in output:
            for content in getattr(item, "content", []) or []:
                candidate = getattr(content, "text", None)
                if candidate:
                    parts.append(str(candidate))
        if parts:
            return "".join(parts)
    raise ValueError("API response contained no output text")


def validate_output(
    output: dict[str, Any],
    request_metadata: Mapping[str, Any],
    validator: Draft202012Validator,
) -> list[str]:
    errors = [
        f"schema:{'/'.join(map(str, err.absolute_path)) or '<root>'}:{err.message}"
        for err in validator.iter_errors(output)
    ]
    wrapper = {"request_metadata": dict(request_metadata), "output": output}
    semantic = (
        conditional_semantic_errors(wrapper)
        if is_conditional_request(request_metadata)
        else semantic_errors(wrapper)
    )
    errors.extend(f"semantic:{message}" for message in semantic)
    return errors


def make_openai_body(
    model: str,
    system: str,
    user: str,
    schema: Mapping[str, Any],
    max_output_tokens: int,
    reasoning_effort: str | None = "none",
) -> dict[str, Any]:
    # Provider structured-output endpoints generally expect the schema body rather
    # than documentation-only identifiers. Cross-field invariants are enforced by
    # scoring/validate_outputs.py after schema validation.
    provider_schema_document = {
        key: value for key, value in schema.items() if key not in {"$schema", "$id"}
    }
    provider_schema = _openai_provider_schema_copy(
        provider_schema_document,
    )
    body = {
        "model": model,
        "instructions": system,
        "input": user,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "groupeval_absolute_output",
                "strict": True,
                "schema": provider_schema,
            }
        },
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def make_inst_body(
    model: str,
    system: str,
    user: str,
    schema: Mapping[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    """Build an OpenAI-compatible Chat Completions request for GenAI Studio."""
    provider_schema_document = {
        key: value for key, value in schema.items() if key not in {"$schema", "$id"}
    }
    provider_schema = _provider_schema_copy(
        provider_schema_document,
        omitted_keywords=set(INST_OMITTED_SCHEMA_KEYWORDS),
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "groupeval_absolute_output",
                "strict": True,
                "schema": provider_schema,
            },
        },
        "max_tokens": max_output_tokens,
        "stream": False,
        # GenAI Studio forwards sampling controls through the options object.
        # Minimum-randomness judging is part of the frozen GroupEval protocol.
        "options": {"temperature": 0.0, "top_p": 1.0},
    }


def _provider_schema_copy(value: Any, omitted_keywords: set[str]) -> Any:
    """Recursively copy a schema while omitting provider-unsupported keywords."""
    if isinstance(value, Mapping):
        return {
            key: _provider_schema_copy(item, omitted_keywords)
            for key, item in value.items()
            if key not in omitted_keywords
        }
    if isinstance(value, list):
        return [_provider_schema_copy(item, omitted_keywords) for item in value]
    return value


def _json_schema_primitive_type(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def _openai_provider_schema_copy(value: Any) -> Any:
    """Map the frozen schema to OpenAI's supported Structured Outputs subset.

    The canonical Draft 2020-12 schema remains unchanged and is always used for
    local acceptance. OpenAI rejects ``uniqueItems`` and requires explicit
    ``type`` on enum/const-only nodes, so the provider copy omits/adds only
    those mechanically equivalent constraints.
    """
    copied = _provider_schema_copy(
        value,
        omitted_keywords=set(OPENAI_OMITTED_SCHEMA_KEYWORDS),
    )

    def add_required_types(node: Any) -> Any:
        if isinstance(node, list):
            return [add_required_types(item) for item in node]
        if not isinstance(node, Mapping):
            return node
        normalized = {
            key: add_required_types(item) for key, item in node.items()
        }
        if "type" not in normalized and "const" in normalized:
            inferred = _json_schema_primitive_type(normalized["const"])
            if inferred is not None:
                normalized["type"] = inferred
        if "type" not in normalized and isinstance(normalized.get("enum"), list):
            inferred_types = {
                _json_schema_primitive_type(item) for item in normalized["enum"]
            }
            inferred_types.discard(None)
            if len(inferred_types) == 1:
                normalized["type"] = next(iter(inferred_types))
        return normalized

    return add_required_types(copied)


class SerialRateLimiter:
    """Conservative start-to-start limiter shared by all GenAI-Inst retries."""

    def __init__(self, minimum_interval: float) -> None:
        if minimum_interval < 0:
            raise ValueError("minimum request interval must be nonnegative")
        self.minimum_interval = minimum_interval
        self.last_start = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_start
        if elapsed < self.minimum_interval:
            time.sleep(self.minimum_interval - elapsed)
        self.last_start = time.monotonic()


class NonRetriableProviderError(RuntimeError):
    """A request/configuration failure that backoff cannot resolve."""


def _validated_output(
    raw: str,
    request_metadata: Mapping[str, Any],
    validator: Draft202012Validator,
) -> tuple[dict[str, Any], list[str]]:
    try:
        output = json.loads(raw)
        if not isinstance(output, dict):
            raise ValueError("top-level output is not an object")
        errors = validate_output(output, request_metadata, validator)
    except Exception as exc:
        return {}, [str(exc)]
    return output, errors


def _openai_response_dump(response: Any) -> Any:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if isinstance(response, Mapping):
        return dict(response)
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "output_text": getattr(response, "output_text", None),
        "usage": response_usage(response),
    }


def run_one_openai(
    client: Any,
    body: dict[str, Any],
    request_metadata: Mapping[str, Any],
    validator: Draft202012Validator,
    max_transport_retries: int,
    max_content_retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    transport_attempts = 0
    content_attempts = 0
    attempt_body_hashes: list[str] = []
    while True:
        request_started_utc = utc_now()
        attempt_body_hashes.append(sha256_bytes(canonical_json_bytes(body)))
        try:
            response = client.responses.create(**body)
        except Exception as exc:  # provider errors vary by SDK/version
            if transport_attempts >= max_transport_retries:
                raise RuntimeError(f"judge transport failure after retries: {exc}") from exc
            transport_attempts += 1
            time.sleep(min(8.0, 2.0**transport_attempts))
            continue

        response_received_utc = utc_now()
        raw = extract_output_text(response)
        output, errors = _validated_output(raw, request_metadata, validator)
        if not errors:
            response_dump = _openai_response_dump(response)
            return output, {
                "response_id": getattr(response, "id", None),
                "provider_response_model": getattr(response, "model", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "usage": response_usage(response),
                "finish_reason": None,
                "transport_retries": transport_attempts,
                "content_retries": content_attempts,
                "request_started_utc": request_started_utc,
                "response_received_utc": response_received_utc,
                "attempt_body_sha256s": attempt_body_hashes,
                "raw_response_sha256": sha256_bytes(
                    canonical_json_bytes(response_dump)
                ),
                "output_text_sha256": sha256_bytes(raw.encode("utf-8")),
            }
        last_error = " | ".join(errors[:8])
        if content_attempts >= max_content_retries:
            raise RuntimeError(f"judge output remained invalid after retries: {last_error}")
        content_attempts += 1
        body = dict(body)
        body["input"] = str(body["input"]) + retry_notice(
            errors, conditional=is_conditional_request(request_metadata)
        )


def run_one_inst(
    url: str,
    api_key: str,
    body: dict[str, Any],
    request_metadata: Mapping[str, Any],
    validator: Draft202012Validator,
    max_transport_retries: int,
    max_content_retries: int,
    timeout: float,
    rate_limiter: SerialRateLimiter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one strict-schema GenAI Studio call with bounded retries."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("install scoring/requirements.txt before live calls") from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    transport_attempts = 0
    content_attempts = 0
    attempt_body_hashes: list[str] = []

    while True:
        rate_limiter.wait()
        request_started_utc = utc_now()
        attempt_body_hashes.append(sha256_bytes(canonical_json_bytes(body)))
        response: Any = None
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=timeout,
            )
            response_received_utc = utc_now()
            detail = response.text[:300].replace("\n", " ")
            # The GenAI Studio gateway occasionally wraps upstream failures and
            # its own rate limiter in HTTP 400 rather than 5xx/429. Treat only
            # those two explicit provider-side messages as transient; ordinary
            # 4xx responses still fail immediately so malformed requests or
            # credentials are not retried.
            transient_gateway_400 = response.status_code == 400 and any(
                marker in detail.lower()
                for marker in ("server connection error", "rate limit exceeded")
            )
            if (
                response.status_code == 429
                or response.status_code >= 500
                or transient_gateway_400
            ):
                retry_after = response.headers.get("Retry-After")
                raise RuntimeError(
                    f"retriable HTTP {response.status_code}"
                    + (f" ({detail})" if detail else "")
                    + (f"; Retry-After={retry_after}" if retry_after else "")
                )
            if response.status_code != 200:
                detail = response.text[:500].replace("\n", " ")
                raise NonRetriableProviderError(
                    f"non-retriable GenAI Studio HTTP {response.status_code}: {detail}"
                )
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError("GenAI Studio returned a non-object JSON body")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise RuntimeError("GenAI Studio response contained no choices")
            message = choices[0].get("message")
            if not isinstance(message, Mapping):
                raise RuntimeError("GenAI Studio response contained no assistant message")
            raw = message.get("content")
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("GenAI Studio response contained no output text")
        except NonRetriableProviderError:
            # Authentication, permissions, malformed request, and other 4xx
            # failures should be corrected by the operator, not retried.
            raise
        except Exception as exc:
            if transport_attempts >= max_transport_retries:
                raise RuntimeError(
                    f"GenAI Studio transport/provider failure after retries: {exc}"
                ) from exc
            transport_attempts += 1
            retry_after_seconds = 0.0
            if response is not None:
                retry_after = response.headers.get("Retry-After")
                try:
                    retry_after_seconds = float(retry_after)
                except (TypeError, ValueError):
                    retry_after_seconds = 0.0
            time.sleep(max(retry_after_seconds, min(30.0, 2.0**transport_attempts)))
            continue

        output, errors = _validated_output(raw, request_metadata, validator)
        if not errors:
            choice = choices[0]
            return output, {
                "response_id": data.get("id"),
                "provider_response_model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
                "usage": data.get("usage") if isinstance(data.get("usage"), Mapping) else {},
                "finish_reason": choice.get("finish_reason"),
                "transport_retries": transport_attempts,
                "content_retries": content_attempts,
                "request_started_utc": request_started_utc,
                "response_received_utc": response_received_utc,
                "attempt_body_sha256s": attempt_body_hashes,
                "raw_response_sha256": sha256_bytes(canonical_json_bytes(data)),
                "output_text_sha256": sha256_bytes(raw.encode("utf-8")),
            }

        last_error = " | ".join(errors[:8])
        if content_attempts >= max_content_retries:
            raise RuntimeError(f"judge output remained invalid after retries: {last_error}")
        content_attempts += 1
        body = dict(body)
        messages = [dict(message) for message in body["messages"]]
        messages[-1]["content"] = (
            str(messages[-1]["content"])
            + retry_notice(
                errors, conditional=is_conditional_request(request_metadata)
            )
        )
        body["messages"] = messages


def batch_record(custom_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": dict(body),
    }


def metadata_group_id(metadata: Mapping[str, Any], context: str) -> str:
    value = metadata.get("group_id")
    if value is None or str(value).strip() == "":
        raise ValueError(f"{context}: request_metadata.group_id is required")
    return str(value)


def validate_unique_inputs(
    records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    ordered_ids: list[str] = []
    metadata_by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        metadata = record.get("request_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"input record {index}: request_metadata must be an object")
        group_id = metadata_group_id(metadata, f"input record {index}")
        if group_id in metadata_by_id:
            raise ValueError(f"duplicate input request_metadata.group_id: {group_id}")
        ordered_ids.append(group_id)
        metadata_by_id[group_id] = metadata
    return ordered_ids, metadata_by_id


def load_resume_rows(
    path: Path,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    expected_body_hashes: Mapping[str, str],
    validator: Draft202012Validator,
    provider_label: str,
    model: str,
    endpoint: str,
    prompt_sha: str,
    schema_sha: str,
    profile_sha: str,
    reasoning_effort: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Load and strictly validate a checkpoint before skipping any calls."""
    rows = [record for _, record in read_jsonl(path)]
    seen: set[str] = set()
    required_run_values = {
        "judge_provider": provider_label,
        "judge_model_snapshot": model,
        "judge_endpoint": endpoint,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
        "target_profile_sha256": profile_sha,
        "reasoning_effort": reasoning_effort,
        "provider_schema_omitted_keywords": list(
            INST_OMITTED_SCHEMA_KEYWORDS
            if provider_label == "institutional_genai"
            else OPENAI_OMITTED_SCHEMA_KEYWORDS
        ),
        "provider_schema_normalizations": (
            []
            if provider_label == "institutional_genai"
            else list(OPENAI_SCHEMA_NORMALIZATIONS)
        ),
        "local_validation_uses_unmodified_schema": True,
    }
    for index, row in enumerate(rows):
        metadata = row.get("request_metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{path}: row {index + 1} has no request_metadata object")
        group_id = metadata_group_id(metadata, f"{path}: row {index + 1}")
        if group_id in seen:
            raise ValueError(f"{path}: duplicate completed group_id {group_id}")
        if group_id not in metadata_by_id:
            raise ValueError(f"{path}: completed group_id {group_id} is not in this input")
        if canonical_json_bytes(metadata) != canonical_json_bytes(metadata_by_id[group_id]):
            raise ValueError(f"{path}: metadata mismatch for completed group_id {group_id}")

        run_metadata = row.get("run_metadata")
        if not isinstance(run_metadata, Mapping):
            raise ValueError(f"{path}: row {index + 1} has no run_metadata object")
        for field, expected in required_run_values.items():
            if run_metadata.get(field) != expected:
                raise ValueError(
                    f"{path}: {field} mismatch for group_id {group_id}: "
                    f"expected {expected!r}, found {run_metadata.get(field)!r}"
                )
        if run_metadata.get("request_body_sha256") != expected_body_hashes[group_id]:
            raise ValueError(f"{path}: request body hash mismatch for group_id {group_id}")
        for field in (
            "response_id",
            "provider_response_model",
            "usage",
            "request_started_utc",
            "response_received_utc",
            "raw_response_sha256",
            "output_text_sha256",
        ):
            if field not in run_metadata:
                raise ValueError(
                    f"{path}: missing {field} for completed group_id {group_id}"
                )

        output = row.get("output")
        if not isinstance(output, dict):
            raise ValueError(f"{path}: row {index + 1} has no output object")
        errors = validate_output(output, metadata, validator)
        if errors:
            raise ValueError(
                f"{path}: invalid completed output for group_id {group_id}: "
                + " | ".join(errors[:8])
            )
        seen.add(group_id)
    return rows, seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--provider",
        choices=("inst", "openai"),
        default=DEFAULT_PROVIDER,
        help="Judge provider (default: GenAI-Inst)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model identifier (default: gpt-oss:120b for GenAI-Inst)",
    )
    parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help=(
            "Environment variable holding the OpenAI API key; the variable name "
            "is not written to outputs (default: OPENAI_API_KEY)"
        ),
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=OPENAI_REASONING_EFFORTS,
        default="none",
        help=(
            "OpenAI Responses reasoning effort. 'default' omits the reasoning "
            "field; GPT-5.6 currently documents medium as its default."
        ),
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--target-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--max-transport-retries", type=int, default=2)
    parser.add_argument("--max-content-retries", type=int, default=2)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--min-request-interval",
        type=float,
        default=None,
        help="Minimum start-to-start interval; GenAI-Inst defaults conservatively to 3 s",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and resume a compatible partial output checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--request-preview", type=Path, default=None)
    parser.add_argument(
        "--write-batch-jsonl",
        type=Path,
        default=None,
        help="Write /v1/responses Batch input and do not contact the API",
    )
    args = parser.parse_args()

    model = args.model or (
        DEFAULT_INST_MODEL if args.provider == "inst" else DEFAULT_OPENAI_MODEL
    )
    endpoint = (
        os.environ.get("INST_GENAI_URL", DEFAULT_INST_URL)
        if args.provider == "inst"
        else OPENAI_RESPONSES_URL
    )
    provider_label = (
        "institutional_genai" if args.provider == "inst" else "openai"
    )
    openai_reasoning_request = (
        None
        if args.openai_reasoning_effort == "default"
        else args.openai_reasoning_effort
    )
    if args.provider == "inst":
        reasoning_effort_effective = "provider_default"
    elif openai_reasoning_request is not None:
        reasoning_effort_effective = openai_reasoning_request
    elif model.startswith("gpt-5.6"):
        reasoning_effort_effective = "medium"
    else:
        reasoning_effort_effective = "provider_default"
    minimum_interval = args.min_request_interval
    if minimum_interval is None:
        minimum_interval = (
            DEFAULT_INST_MIN_REQUEST_INTERVAL if args.provider == "inst" else 0.0
    )

    if args.max_output_tokens < 1000:
        raise ValueError(
            "max-output-tokens is implausibly small for eight structured sample records"
        )
    if args.max_transport_retries < 0 or args.max_content_retries < 0:
        raise ValueError("retry counts must be nonnegative")
    if args.request_timeout <= 0:
        raise ValueError("request-timeout must be positive")
    if minimum_interval < 0:
        raise ValueError("min-request-interval must be nonnegative")
    if args.provider == "inst" and not endpoint.startswith("https://"):
        raise ValueError("INST_GENAI_URL must use HTTPS so bearer credentials remain protected")
    if args.provider == "inst" and args.openai_reasoning_effort != "none":
        raise ValueError("--openai-reasoning-effort applies only to --provider openai")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.openai_api_key_env):
        raise ValueError("--openai-api-key-env must be a valid environment-variable name")
    if args.groups.resolve() == args.output.resolve():
        raise ValueError("--groups and --output must be different paths")
    if args.dry_run and args.write_batch_jsonl is not None:
        raise ValueError("--dry-run and --write-batch-jsonl are mutually exclusive")
    if args.resume and (args.dry_run or args.write_batch_jsonl is not None):
        raise ValueError("--resume applies only to live synchronous calls")

    system_template, user_template = parse_prompt(args.prompt)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    profile_raw = args.target_profile.read_text(encoding="utf-8")
    target_profile_text = strip_profile_title(profile_raw)

    records = [record for _, record in read_jsonl(args.groups)]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    if not records:
        raise ValueError("no input groups selected")
    ordered_group_ids, metadata_by_id = validate_unique_inputs(records)

    rendered: list[dict[str, Any]] = []
    request_bodies: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record["request_metadata"]
        system, user = render_request(record, system_template, user_template, target_profile_text)
        if args.provider == "inst":
            body = make_inst_body(model, system, user, schema, args.max_output_tokens)
        else:
            body = make_openai_body(
                model,
                system,
                user,
                schema,
                args.max_output_tokens,
                reasoning_effort=openai_reasoning_request,
            )
        body_sha = sha256_bytes(canonical_json_bytes(body))
        request_bodies.append(body)
        rendered.append({
            "group_id": metadata.get("group_id"),
            "provider": provider_label,
            "model": model,
            "endpoint": endpoint,
            "provider_schema_omitted_keywords": list(
                INST_OMITTED_SCHEMA_KEYWORDS
                if args.provider == "inst"
                else OPENAI_OMITTED_SCHEMA_KEYWORDS
            ),
            "provider_schema_normalizations": (
                []
                if args.provider == "inst"
                else list(OPENAI_SCHEMA_NORMALIZATIONS)
            ),
            "instructions": system,
            "input": user,
            "body_sha256": body_sha,
            "reasoning_effort": reasoning_effort_effective,
            "reasoning_effort_source": (
                "provider_default"
                if args.provider == "inst" or openai_reasoning_request is None
                else "explicit_request"
            ),
        })
        if args.provider == "openai":
            batch_rows.append(batch_record(str(metadata.get("group_id")), body))

    if args.request_preview is not None:
        if args.request_preview.resolve() == args.output.resolve():
            raise ValueError("--request-preview and --output must be different paths")
        write_jsonl_atomic(args.request_preview, rendered)
    if args.write_batch_jsonl is not None:
        if args.provider != "openai":
            raise ValueError("--write-batch-jsonl is supported only for OpenAI")
        write_jsonl_atomic(args.write_batch_jsonl, batch_rows)
        print(
            json.dumps(
                {
                    "batch_requests": len(batch_rows),
                    "path": str(args.write_batch_jsonl),
                },
                indent=2,
            )
        )
        return 0
    if args.dry_run:
        # Do not manufacture judge outputs. The requested output file is a manifest of
        # rendered calls, explicitly labeled as dry-run metadata.
        dry = [{"dry_run": True, **item} for item in rendered]
        write_jsonl_atomic(args.output, dry)
        print(json.dumps({
            "dry_run": True,
            "requests": len(dry),
            "output": str(args.output),
            "provider": provider_label,
            "model": model,
        }, indent=2))
        return 0

    prompt_sha = sha256_bytes(args.prompt.read_bytes())
    schema_sha = sha256_bytes(args.schema.read_bytes())
    profile_sha = sha256_bytes(profile_raw.encode("utf-8"))
    expected_body_hashes = {
        group_id: request["body_sha256"]
        for group_id, request in zip(ordered_group_ids, rendered, strict=True)
    }

    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output} already exists; pass --resume only if it is a compatible "
            "partial live checkpoint, or choose a new output path"
        )
    output_rows: list[dict[str, Any]]
    completed_ids: set[str]
    if args.output.exists():
        output_rows, completed_ids = load_resume_rows(
            args.output,
            metadata_by_id,
            expected_body_hashes,
            validator,
            provider_label,
            model,
            endpoint,
            prompt_sha,
            schema_sha,
            profile_sha,
            reasoning_effort_effective,
        )
    else:
        output_rows, completed_ids = [], set()

    initial_completed_count = len(completed_ids)
    pending_count = len(records) - initial_completed_count
    key_variable = (
        "INSTITUTIONAL_GENAI_API_KEY"
        if args.provider == "inst"
        else args.openai_api_key_env
    )
    api_key = os.environ.get(key_variable, "")
    if pending_count and not api_key:
        raise RuntimeError(f"{key_variable} is required for live calls")

    client: Any = None
    if pending_count and args.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install scoring/requirements.txt before live calls") from exc
        # Keep every retry visible in our audited retry counters and body hashes;
        # the SDK otherwise performs additional implicit retries.
        client = OpenAI(
            api_key=api_key,
            timeout=args.request_timeout,
            max_retries=0,
        )

    rate_limiter = SerialRateLimiter(minimum_interval)
    for input_index, (record, request, body) in enumerate(
        zip(records, rendered, request_bodies, strict=True), 1
    ):
        metadata = record["request_metadata"]
        group_id = metadata_group_id(metadata, "live input")
        if group_id in completed_ids:
            continue
        print(
            f"[{input_index}/{len(records)}] judging group_id={group_id} "
            f"with {provider_label}/{model}",
            file=sys.stderr,
            flush=True,
        )
        if args.provider == "inst":
            output, call_meta = run_one_inst(
                endpoint,
                api_key,
                body,
                metadata,
                validator,
                args.max_transport_retries,
                args.max_content_retries,
                args.request_timeout,
                rate_limiter,
            )
        else:
            output, call_meta = run_one_openai(
                client,
                body,
                metadata,
                validator,
                args.max_transport_retries,
                args.max_content_retries,
            )
        output_rows.append({
            "request_metadata": metadata,
            "run_metadata": {
                "judge_provider": provider_label,
                "judge_model_snapshot": model,
                "judge_endpoint": endpoint,
                "model_identifier_is_provider_alias": args.provider == "inst",
                "reasoning_effort": reasoning_effort_effective,
                "reasoning_effort_requested": (
                    openai_reasoning_request if args.provider == "openai" else None
                ),
                "reasoning_effort_source": (
                    "provider_default"
                    if args.provider == "inst" or openai_reasoning_request is None
                    else "explicit_request"
                ),
                "tools_enabled": False,
                "strict_json_schema": True,
                "provider_schema_omitted_keywords": list(
                    INST_OMITTED_SCHEMA_KEYWORDS
                    if args.provider == "inst"
                    else OPENAI_OMITTED_SCHEMA_KEYWORDS
                ),
                "provider_schema_normalizations": (
                    []
                    if args.provider == "inst"
                    else list(OPENAI_SCHEMA_NORMALIZATIONS)
                ),
                "local_validation_uses_unmodified_schema": True,
                "prompt_sha256": prompt_sha,
                "schema_sha256": schema_sha,
                "target_profile_sha256": profile_sha,
                "request_body_sha256": request["body_sha256"],
                "min_request_interval_seconds": minimum_interval,
                "timestamp_utc": utc_now(),
                **call_meta,
            },
            "output": output,
        })
        completed_ids.add(group_id)
        # Atomic per-group checkpointing survives interruption without partial
        # lines or duplicate append ambiguity.
        write_jsonl_atomic(args.output, output_rows)

    expected_ids = set(ordered_group_ids)
    if completed_ids != expected_ids:
        missing = sorted(expected_ids - completed_ids)
        extra = sorted(completed_ids - expected_ids)
        raise RuntimeError(
            f"incomplete output checkpoint: missing={missing[:20]}, extra={extra[:20]}"
        )
    if len(output_rows) != len(records):
        raise RuntimeError(
            f"output completeness failure: {len(output_rows)} rows for {len(records)} inputs"
        )

    print(json.dumps({
        "completed": len(output_rows),
        "newly_completed": len(completed_ids) - initial_completed_count,
        "output": str(args.output),
        "provider": provider_label,
        "model": model,
        "complete": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
