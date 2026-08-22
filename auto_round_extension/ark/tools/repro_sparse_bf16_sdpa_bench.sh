#!/usr/bin/env bash

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="/home/yiliu4/workspace/auto-round/auto_round_extension/ark"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
OUTPUT_CSV="${OUTPUT_CSV:-/tmp/bench_sparse_topk_bf16_dev6.csv}"
ZE_AFFINITY_MASK_VALUE="${ZE_AFFINITY_MASK_VALUE:-6}"
XPU_SO="${XPU_SO:-}"

cd "${REPO_ROOT}"

if [[ "${1:-}" == "--inner" ]]; then
  shift
elif command -v sg >/dev/null 2>&1; then
  current_groups="$(id -nG || true)"
  if [[ ! " ${current_groups} " =~ [[:space:]]render[[:space:]] ]] || [[ ! " ${current_groups} " =~ [[:space:]]video[[:space:]] ]]; then
    exec sg render -c "sg video -c '${SCRIPT_PATH} --inner'"
  fi
fi

# oneAPI's setvars.sh is not compatible with `set -u` in this environment.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

export MKLROOT=/opt/intel/oneapi/mkl/2026.1
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK_VALUE}"
export SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch

if [[ -z "${XPU_SO}" ]]; then
  XPU_SO="$(find "${REPO_ROOT}/auto_round_kernel/ark-xbuild" -maxdepth 1 -name 'auto_round_kernel_xpu*.so' | sort | tail -n 1)"
fi

echo "XPU process snapshot:"
xpu-smi ps || true
echo
echo "Running sparse BF16 SDPA benchmark on device ${ZE_AFFINITY_MASK}..."
echo "Using XPU extension: ${XPU_SO}"

exec "${PYTHON_BIN}" benchmarks/bench_sparse_topk.py \
  --dtype bf16 \
  --xpu-so "${XPU_SO}" \
  --output-csv "${OUTPUT_CSV}"
