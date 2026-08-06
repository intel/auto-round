#!/usr/bin/env bash
#
# run_bf16_sparse_bench.sh
# ========================
# Documented reproduce command for the BF16 sparse SDPA benchmark on Intel XPU.
#
# WHAT IT DOES
#   Runs benchmarks/bench_sparse_topk.py --dtype bf16 against the rebuilt
#   XPU extension (auto_round_kernel/ark-xbuild), pinning a GPU device, and
#   saves the CSV + a readable log under benchmarks/results/.
#
# USAGE
#   ./tools/run_bf16_sparse_bench.sh
#
# ENV OVERRIDES
#   ZE_AFFINITY_MASK_VALUE  GPU device index to run on   (default: 6)
#   OUTPUT_DIR              where CSV + log are written (default: benchmarks/results)
#   XPU_SO                  explicit path to the built auto_round_kernel_xpu*.so
#                           (default: newest under auto_round_kernel/ark-xbuild)
#
# PREREQUISITES
#   - oneAPI 2026.1 runtime (sourced from /opt/intel/oneapi/setvars.sh)
#   - Python venv at .venv/bin/python (PyTorch 2.13.0+xpu)
#   - a rebuilt extension with BF16 sparse symbols under auto_round_kernel/ark-xbuild/
#   - a free GPU (pick via ZE_AFFINITY_MASK_VALUE; check `xpu-smi ps` first)
#
# BENCHMARK CONFIG (bench_sparse_topk.py defaults)
#   batch=1  num_heads_q/kv=40  head_dim=128  dtype=bf16  qtile=256
#   seq_len=[32768, 75600]  tensor_layout=[HND, NHD]  topk=[0.5, 0.25, 0.125]
#   warmup=2  iters=3
#
# NOTE
#   First call per process pays a ~30s SYCL JIT for the sparse kernels; the
#   full sweep above takes roughly 15-25 minutes on one GPU.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmarks/results}"
ZE_AFFINITY_MASK_VALUE="${ZE_AFFINITY_MASK_VALUE:-6}"
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
export SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch

if [[ -z "${XPU_SO}" ]]; then
  XPU_SO="$(find "${REPO_ROOT}/auto_round_kernel/ark-xbuild" -maxdepth 1 -name 'auto_round_kernel_xpu*.so' | sort | tail -n 1)"
fi
if [[ -z "${XPU_SO}" ]]; then
  echo "error: no auto_round_kernel_xpu*.so found under auto_round_kernel/ark-xbuild" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
out_csv="${OUTPUT_DIR}/bf16_sparse_${stamp}.csv"
out_log="${OUTPUT_DIR}/bf16_sparse_${stamp}.log"

echo "XPU process snapshot:"
xpu-smi ps || true
echo
echo "Running BF16 sparse SDPA benchmark on device ${ZE_AFFINITY_MASK}..."
echo "Using XPU extension: ${XPU_SO}"
echo "CSV:  ${out_csv}"
echo "Log:  ${out_log}"
echo

"${PYTHON_BIN}" benchmarks/bench_sparse_topk.py \
  --dtype bf16 \
  --xpu-so "${XPU_SO}" \
  --output-csv "${out_csv}" 2>&1 | tee "${out_log}"
