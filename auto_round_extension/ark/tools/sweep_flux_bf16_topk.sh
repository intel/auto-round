#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# sweep_flux_bf16_topk.sh
# =======================
# One-GPU FLUX.1-dev BF16-sparse topk sweep. Runs examples/flux_gen_bf16_sweep.py
# with the perf-tuned qtile256 config (q_tile=256, q_block=256, k_block=64) and
# saves PNGs + a summary under a shared FLUX_OUTPUT_DIR.
#
# To use both XPU devices, launch two instances concurrently with disjoint
# topk subsets pointing at the SAME FLUX_OUTPUT_DIR, e.g.:
#   STAMP=$(date +%Y%m%d_%H%M%S)
#   OUT=benchmarks/results/flux_bf16_${STAMP}
#   ZE_AFFINITY_MASK_VALUE=5 FLUX_RUN_DENSE=1 FLUX_SPARSE_TOPKS="0.5 0.25" \
#     FLUX_OUTPUT_DIR=$OUT ./tools/sweep_flux_bf16_topk.sh &
#   ZE_AFFINITY_MASK_VALUE=6 FLUX_RUN_DENSE=0 FLUX_SPARSE_TOPKS="0.125" \
#     FLUX_OUTPUT_DIR=$OUT ./tools/sweep_flux_bf16_topk.sh &
#   wait
#
# ENV OVERRIDES (all optional)
#   ZE_AFFINITY_MASK_VALUE      XPU device index                          (default: 6)
#   FLUX_MODEL                  model dir (diffusers format)               (default: /mnt/disk3/models/black-forest-labs/FLUX.1-dev)
#   FLUX_SPARSE_TOPKS           space-separated topk list                  (default: "0.5 0.25 0.125")
#   FLUX_RUN_DENSE              1/0 include a dense baseline generation    (default: 1)
#   FLUX_OUTPUT_DIR             results dir                                (default: benchmarks/results/flux_bf16_<stamp>)
#   FLUX_HEIGHT / FLUX_WIDTH    generation size (keep 512)                 (default: 512)
#   FLUX_STEPS / FLUX_SEED / FLUX_PROMPT                                   (defaults: 50 / 0 / "A cat ...")
#   SPARGE_PREPROCESS_BACKEND   torch | triton_xpu | auto                 (default: triton_xpu)
#   XPU_SO                      explicit path to the built auto_round_kernel_xpu*.so
#                               (default: newest under auto_round_kernel/ark-xbuild)
#
# PREREQUISITES
#   - oneAPI 2026.1 runtime (sourced from /opt/intel/oneapi/setvars.sh)
#   - venv at .venv/bin/python with torch.xpu + diffusers
#   - rebuilt extension with BF16 sparse symbols under auto_round_kernel/ark-xbuild/
#   - a free GPU (pick via ZE_AFFINITY_MASK_VALUE; check `xpu-smi ps` first)

set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

ZE_AFFINITY_MASK_VALUE="${ZE_AFFINITY_MASK_VALUE:-6}"
FLUX_MODEL="${FLUX_MODEL:-/mnt/disk3/models/black-forest-labs/FLUX.1-dev}"
FLUX_SPARSE_TOPKS="${FLUX_SPARSE_TOPKS:-0.5 0.25 0.125}"
FLUX_RUN_DENSE="${FLUX_RUN_DENSE:-1}"
FLUX_HEIGHT="${FLUX_HEIGHT:-512}"
FLUX_WIDTH="${FLUX_WIDTH:-512}"
FLUX_STEPS="${FLUX_STEPS:-50}"
FLUX_SEED="${FLUX_SEED:-0}"
SPARGE_PREPROCESS_BACKEND="${SPARGE_PREPROCESS_BACKEND:-triton_xpu}"
XPU_SO="${XPU_SO:-}"

cd "${REPO_ROOT}"

# Re-exec under the render/video groups if this shell does not already have them.
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
export SAGE_ATTN_SPARSE_PREPROCESS_BACKEND="${SPARGE_PREPROCESS_BACKEND}"

# qtile256 perf config is fixed for this sweep; kernel is env-overridable (bf16 default, int8 for SAGE).
export FLUX_SPARSE_KERNEL="${FLUX_SPARSE_KERNEL:-bf16}"
export FLUX_SPARSE_Q_TILE_OVERRIDE=256
export FLUX_SPARSE_Q_BLOCK_TOKENS=256
export FLUX_SPARSE_K_BLOCK_TOKENS=64

if [[ -z "${XPU_SO}" ]]; then
  XPU_SO="$(find "${REPO_ROOT}/auto_round_kernel/ark-xbuild" -maxdepth 1 -name 'auto_round_kernel_xpu*.so' | sort | tail -n 1)"
fi
if [[ -z "${XPU_SO}" ]]; then
  echo "error: no auto_round_kernel_xpu*.so found under auto_round_kernel/ark-xbuild" >&2
  exit 1
fi

if [[ -z "${FLUX_OUTPUT_DIR:-}" ]]; then
  FLUX_OUTPUT_DIR="${REPO_ROOT}/benchmarks/results/flux_bf16_$(date +%Y%m%d_%H%M%S)"
fi
export FLUX_OUTPUT_DIR="${FLUX_OUTPUT_DIR}"
mkdir -p "${FLUX_OUTPUT_DIR}"

echo "XPU process snapshot:"
xpu-smi ps || true
echo "Using XPU extension: ${XPU_SO}"
echo "device=${ZE_AFFINITY_MASK_VALUE} topks='${FLUX_SPARSE_TOPKS}' run_dense=${FLUX_RUN_DENSE}"
echo "qtile256 config: FLUX_SPARSE_Q_TILE_OVERRIDE=256 FLUX_SPARSE_Q_BLOCK_TOKENS=256 FLUX_SPARSE_K_BLOCK_TOKENS=64"
echo "output dir: ${FLUX_OUTPUT_DIR}"
echo

"${PYTHON_BIN}" examples/flux_gen_bf16_sweep.py
