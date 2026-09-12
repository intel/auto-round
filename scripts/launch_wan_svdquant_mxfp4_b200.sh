#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
set -Eeuo pipefail

PROFILE="${1:-smoke}"
GPUS="${2:-${GPUS:-0}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
OUT="${OUT:-$PWD/wan-a14b-svdquant-$PROFILE}"
LOG="${LOG:-$OUT.log}"
if [[ "$GPUS" == *,* ]]; then
  echo 'This launcher uses one visible GPU with model CPU offload; choose one GPU index.' >&2
  exit 2
fi
ARGS=(--model "$MODEL" --output "$OUT" --profile "$PROFILE")
for name in HEIGHT WIDTH NUM_FRAMES RANK NSAMPLES; do
  if [[ -n "${!name:-}" ]]; then
    option="${name,,}"
    ARGS+=("--${option//_/-}" "${!name}")
  fi
done
[[ -z "${DATASET:-}" ]] || ARGS+=(--dataset "$DATASET")
[[ -z "${PROMPTS_FILE:-}" ]] || ARGS+=(--prompts-file "$PROMPTS_FILE")
[[ -z "${STEPS:-}" ]] || ARGS+=(--calib-steps "$STEPS")
[[ -z "${SIGNROUND_ITERS:-}" ]] || ARGS+=(--iters "$SIGNROUND_ITERS")
[[ -z "${RESIDUAL_ITERS:-}" ]] || ARGS+=(--residual-iters "$RESIDUAL_ITERS")
[[ -z "${SMOOTH_GRIDS:-}" ]] || ARGS+=(--smooth-grids "$SMOOTH_GRIDS")
[[ -z "${SMOOTH_MAX_CALLS:-}" ]] || ARGS+=(--smooth-calls "$SMOOTH_MAX_CALLS")
case "${LOW_GPU_MEM_USAGE:-1}" in
  1) ;;
  0) ARGS+=(--no-low-gpu-mem-usage) ;;
  *) echo 'LOW_GPU_MEM_USAGE must be 0 or 1' >&2; exit 2 ;;
esac
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 1; }
[[ ! -e "$LOG" ]] || { echo "Log already exists: $LOG" >&2; exit 1; }
mkdir -p "$(dirname "$LOG")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS"
"$PY" -u "$SCRIPT_DIR/quantize_wan_a14b_svdquant.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
