#!/usr/bin/env bash

# Source this file before running Fast-dLLM commands. It keeps large artifacts
# in scratch and points at the dedicated Fast-dLLM Python environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs}"
FAST_DLLM_ENV="${FAST_DLLM_ENV:-${OUTPUT_ROOT}/envs/fast-dllm}"
FAST_DLLM_SCRATCH="${FAST_DLLM_SCRATCH:-${OUTPUT_ROOT}/runs/fast_dllm}"
SHARED_CACHE="${SHARED_CACHE:-${OUTPUT_ROOT}/cache}"

export HF_HOME="${HF_HOME:-${SHARED_CACHE}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SHARED_CACHE}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SHARED_CACHE}/pip}"
export TMPDIR="${TMPDIR:-${FAST_DLLM_SCRATCH}/tmp}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# Required by lm-evaluation-harness for tasks such as HumanEval.
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${HF_XET_CACHE}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${FAST_DLLM_SCRATCH}/evals" \
  "${FAST_DLLM_SCRATCH}/logs" \
  "${FAST_DLLM_SCRATCH}/runs" \
  "${TMPDIR}"

if [[ ! -x "${FAST_DLLM_ENV}/bin/python" ]]; then
  echo "Fast-dLLM env not found at ${FAST_DLLM_ENV}" >&2
  echo "Expected Python at ${FAST_DLLM_ENV}/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${FAST_DLLM_ENV}/bin:${PATH}"
