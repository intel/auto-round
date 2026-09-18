#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# bench_sparse_bf16_fp16_sweep.sh
# ===============================
# Run the sparse BF16 and FP16 kernel benchmark sweep (both HND and NHD layouts)
# against the rebuilt ARK XPU extension, and save CSVs under benchmarks/results/.
#
# USAGE
#   ./tools/bench_sparse_bf16_fp16_sweep.sh
#
# ENV OVERRIDES (all optional)
#   DTYPES         dtypes to sweep, space-separated   (default: "bf16 fp16")
#   SEQ_LENS       seq lengths, space-separated       (default: "75000")
#   TOPKS          topk values, space-separated       (default: "0.5 0.25 0.125")
#   LAYOUTS        tensor layouts, space-separated    (default: "HND NHD")
#   HEADS          num heads q (kv = q)               (default: 40)
#   HEAD_DIM       head dim                           (default: 128)
#   Q_TILE         q_tile_override                    (default: 256)
#   WARMUP         warmup iters                       (default: 2)
#   ITERS          measured iters                     (default: 3)
#   ZE_AFFINITY_MASK_VALUE  GPU device index          (default: 6)
#   XPU_SO         explicit path to the built .so (default: newest under ark-xbuild)
#   OUTPUT_DIR     where CSVs are written            (default: benchmarks/results)
#   SPARGE_PREPROCESS_BACKEND  sparse preprocess backend (default: triton_xpu); torch | triton_xpu | auto
#
# PREREQUISITES
#   - oneAPI 2026.1 runtime (sourced from /opt/intel/oneapi/setvars.sh)
#   - venv at .venv/bin/python with torch.xpu
#   - rebuilt extension under auto_round_kernel/ark-xbuild/
#   - a free GPU (pick via ZE_AFFINITY_MASK_VALUE; check `xpu-smi ps` first)

set -uo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
BENCH="${REPO_ROOT}/benchmarks/bench_sparse_topk.py"

DTYPES="${DTYPES:-bf16 fp16}"
SEQ_LENS="${SEQ_LENS:-75000}"
TOPKS="${TOPKS:-0.5 0.25 0.125}"
LAYOUTS="${LAYOUTS:-HND NHD}"
HEADS="${HEADS:-40}"
HEAD_DIM="${HEAD_DIM:-128}"
Q_TILE="${Q_TILE:-256}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-3}"
ZE_AFFINITY_MASK_VALUE="${ZE_AFFINITY_MASK_VALUE:-6}"
XPU_SO="${XPU_SO:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmarks/results}"

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

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

export MKLROOT=/opt/intel/oneapi/mkl/2026.1
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK_VALUE}"
export SAGE_ATTN_SPARSE_PREPROCESS_BACKEND="${SPARGE_PREPROCESS_BACKEND:-triton_xpu}"   # auto | torch | triton_xpu

if [[ -z "${XPU_SO}" ]]; then
  XPU_SO="$(find "${REPO_ROOT}/auto_round_kernel/ark-xbuild" -maxdepth 1 -name 'auto_round_kernel_xpu*.so' | sort | tail -n 1)"
fi
if [[ -z "${XPU_SO}" ]]; then
  echo "error: no auto_round_kernel_xpu*.so found under auto_round_kernel/ark-xbuild" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"

echo "XPU process snapshot:"
xpu-smi ps || true
echo "Using XPU extension: ${XPU_SO}"
echo "dtypes=${DTYPES} seq=${SEQ_LENS} topk=${TOPKS} layouts=${LAYOUTS} heads=${HEADS} head_dim=${HEAD_DIM} q_tile=${Q_TILE}"
echo

for dtype in ${DTYPES}; do
  out_csv="${OUTPUT_DIR}/bench_sparse_${dtype}_${stamp}.csv"
  echo "========== dtype=${dtype} -> ${out_csv} =========="
  "${PYTHON_BIN}" "${BENCH}" \
    --dtype "${dtype}" \
    --seq-len ${SEQ_LENS} \
    --topk ${TOPKS} \
    --tensor-layout ${LAYOUTS} \
    --head-dim "${HEAD_DIM}" \
    --num-heads-q "${HEADS}" --num-heads-kv "${HEADS}" \
    --q-tile-override "${Q_TILE}" \
    --sparse-q-block-tokens "${Q_TILE}" \
    --sparse-k-block-tokens 64 \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --xpu-so "${XPU_SO}" \
    --output-csv "${out_csv}"
  echo
done

echo "SWEEP_DONE: CSVs written under ${OUTPUT_DIR}"
