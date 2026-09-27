#!/usr/bin/env bash

# Run model/task temperature sweeps sequentially on one GPU.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/fast_dllm_env.sh"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 MODEL:TASK[,MODEL:TASK,...]" >&2
  exit 2
fi

QUEUE_STATUS=0
IFS=',' read -r -a JOBS <<< "$1"

for job in "${JOBS[@]}"; do
  IFS=':' read -r MODEL TASK EXTRA <<< "${job}"
  if [[ -z "${MODEL:-}" || -z "${TASK:-}" || -n "${EXTRA:-}" ]]; then
    echo "Invalid queue entry: ${job}" >&2
    QUEUE_STATUS=1
    continue
  fi

  echo "[$(date -Is)] queue starting ${MODEL}:${TASK}"
  if bash "${SCRIPT_DIR}/fast_dllm_temperature_sweep.sh" "${MODEL}" "${TASK}"; then
    echo "[$(date -Is)] queue finished ${MODEL}:${TASK}"
  else
    STATUS="$?"
    echo "[$(date -Is)] queue FAILED ${MODEL}:${TASK} status=${STATUS}" >&2
    QUEUE_STATUS=1
  fi
done

exit "${QUEUE_STATUS}"
