#!/usr/bin/env bash

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RESULTS_ROOT_DEFAULT="${SCRIPT_DIR}/results/flux_sweep_defaultsteps_$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_ROOT=${FLUX_SWEEP_OUTPUT_ROOT:-${RESULTS_ROOT_DEFAULT}}
PYTHON_BIN=${FLUX_SWEEP_PYTHON:-python}
MODEL_PATH=${FLUX_MODEL:-black-forest-labs/FLUX.1-dev}
RUN_FLUX_PY="${SCRIPT_DIR}/run_flux.py"
PROMPT_VALUE=${FLUX_PROMPT:-A cat holding a sign that says hello world}

mkdir -p "${RESULTS_ROOT}"

IFS=, read -r -a DEVICES <<< "${FLUX_SWEEP_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ ${#DEVICES[@]} -eq 0 ]]; then
  echo "FLUX_SWEEP_DEVICES must provide at least one device id" >&2
  exit 1
fi

if [[ -n "${FLUX_SWEEP_CONFIGS:-}" ]]; then
  read -r -a CONFIGS <<< "${FLUX_SWEEP_CONFIGS}"
else
  CONFIGS=("dense" "1.0" "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")
fi
SUMMARY_CSV="${RESULTS_ROOT}/summary.csv"
COMMANDS_TXT="${RESULTS_ROOT}/commands.txt"

sanitize_topk_label() {
  local topk="$1"
  echo "topk${topk/./p}"
}

write_status_file() {
  local status_path="$1"
  shift
  : > "${status_path}"
  while (($#)); do
    local key="$1"
    local value="$2"
    printf '%s=%q\n' "${key}" "${value}" >> "${status_path}"
    shift 2
  done
}

active_jobs=0
launch_index=0

for config in "${CONFIGS[@]}"; do
  device="${DEVICES[$((launch_index % ${#DEVICES[@]}))]}"
  if [[ "${config}" == "dense" ]]; then
    label="dense"
    mode="dense"
    topk_value="dense"
    sparse_env=(
      FLUX_USE_SPARSE=0
    )
  else
    label=$(sanitize_topk_label "${config}")
    mode="sparse"
    topk_value="${config}"
    sparse_env=(
      FLUX_USE_SPARSE=1
      FLUX_SPARSE_TOPK="${config}"
      FLUX_SPARSE_Q_TILE_OVERRIDE=256
      FLUX_SPARSE_Q_BLOCK_TOKENS=256
      FLUX_SPARSE_K_BLOCK_TOKENS=64
    )
  fi

  png_path="${RESULTS_ROOT}/flux_${label}.png"
  log_path="${RESULTS_ROOT}/flux_${label}.log"
  status_path="${RESULTS_ROOT}/flux_${label}.status"
  cmd_path="${RESULTS_ROOT}/flux_${label}.cmd"

  cat > "${cmd_path}" <<EOF
ZE_AFFINITY_MASK=${device} \
FLUX_MODEL=${MODEL_PATH} \
FLUX_PROMPT=${PROMPT_VALUE@Q} \
${sparse_env[*]} \
FLUX_OUTPUT=${png_path} \
${PYTHON_BIN@Q} ${RUN_FLUX_PY@Q}
EOF

  {
    start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    start_epoch=$(date +%s)
    exit_code=0
    env \
      ZE_AFFINITY_MASK="${device}" \
      FLUX_MODEL="${MODEL_PATH}" \
      FLUX_PROMPT="${PROMPT_VALUE}" \
      FLUX_OUTPUT="${png_path}" \
      "${sparse_env[@]}" \
      "${PYTHON_BIN}" "${RUN_FLUX_PY}" > "${log_path}" 2>&1 || exit_code=$?
    end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    end_epoch=$(date +%s)
    elapsed_sec=$((end_epoch - start_epoch))
    write_status_file "${status_path}" \
      label "${label}" \
      mode "${mode}" \
      topk "${topk_value}" \
      device "${device}" \
      exit_code "${exit_code}" \
      start_iso "${start_iso}" \
      end_iso "${end_iso}" \
      elapsed_sec "${elapsed_sec}" \
      image_path "${png_path}" \
      log_path "${log_path}" \
      cmd_path "${cmd_path}"
    if [[ ${exit_code} -ne 0 ]]; then
      exit "${exit_code}"
    fi
  } &

  launch_index=$((launch_index + 1))
  active_jobs=$((active_jobs + 1))
  if [[ ${active_jobs} -ge ${#DEVICES[@]} ]]; then
    wait -n || true
    active_jobs=$((active_jobs - 1))
  fi
done

wait || true

{
  echo "# Flux sweep commands"
  echo "# Generated at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  for config in "${CONFIGS[@]}"; do
    if [[ "${config}" == "dense" ]]; then
      label="dense"
    else
      label=$(sanitize_topk_label "${config}")
    fi
    cat "${RESULTS_ROOT}/flux_${label}.cmd"
    echo
  done
} > "${COMMANDS_TXT}"

{
  echo "label,mode,topk,device,exit_code,elapsed_sec,image_path,log_path,cmd_path"
  for config in "${CONFIGS[@]}"; do
    if [[ "${config}" == "dense" ]]; then
      label="dense"
    else
      label=$(sanitize_topk_label "${config}")
    fi
    # shellcheck disable=SC1090
    source "${RESULTS_ROOT}/flux_${label}.status"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${label}" "${mode}" "${topk}" "${device}" "${exit_code}" "${elapsed_sec}" \
      "${image_path}" "${log_path}" "${cmd_path}"
  done
} > "${SUMMARY_CSV}"

echo "Saved FLUX sweep outputs to ${RESULTS_ROOT}"
echo "Summary: ${SUMMARY_CSV}"
echo "Commands: ${COMMANDS_TXT}"
