#!/usr/bin/env bash

set -euo pipefail

if [[ "${ARK_BENCH_LOGIN_SHELL_READY:-0}" != "1" ]]; then
    export ARK_BENCH_LOGIN_SHELL_READY=1
    quoted_args=()
    for arg in "$@"; do
        quoted_args+=("$(printf '%q' "${arg}")")
    done
    exec bash -lc "$(printf '%q' "$0") ${quoted_args[*]:-}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/ark-torch-212/bin/python}"
ONEAPI_ROOT_DIR="${ONEAPI_ROOT_DIR:-/home/yiliu4/intel/oneapi}"
ONEAPI_COMPILER_DIR="${ONEAPI_COMPILER_DIR:-${ONEAPI_ROOT_DIR}/compiler/2025.3}"
SYSTEM_ONEAPI_SETVARS="${SYSTEM_ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
XPU_MASK="${XPU_MASK:-6,7}"
OUTPUT_CSV="${OUTPUT_CSV:-bench_sparse_topk_results.csv}"

if [[ -f "${SYSTEM_ONEAPI_SETVARS}" ]]; then
    # XPU discovery depends on the wider oneAPI runtime environment, so seed
    # the shell from the system installation first, then override compiler
    # paths to the requested home-installed 2025.3 toolchain below.
    # shellcheck disable=SC1090
    source "${SYSTEM_ONEAPI_SETVARS}" >/dev/null 2>&1 || true
fi

export ONEAPI_ROOT="${ONEAPI_ROOT_DIR}"
export CMPLR_ROOT="${ONEAPI_COMPILER_DIR}"
export IntelSYCL_DIR="${ONEAPI_COMPILER_DIR}/lib/cmake/IntelSYCL"
export CMAKE_PREFIX_PATH="${IntelSYCL_DIR}"
export PATH="${ONEAPI_COMPILER_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ONEAPI_COMPILER_DIR}/lib:${ONEAPI_COMPILER_DIR}/opt/compiler/lib:${LD_LIBRARY_PATH:-}"
export ZE_AFFINITY_MASK="${XPU_MASK}"

cd "${REPO_ROOT}"

echo "Running bench_sparse_topk with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  ONEAPI_ROOT=${ONEAPI_ROOT}"
echo "  CMPLR_ROOT=${CMPLR_ROOT}"
echo "  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
echo "  OUTPUT_CSV=${OUTPUT_CSV}"

if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch

ok = torch.xpu.is_available()
print(f"torch={torch.__version__} xpu_available={ok} xpu_count={torch.xpu.device_count() if ok else 0}")
sys.exit(0 if ok else 1)
PY
then
    echo "XPU is not available in this environment." >&2
    echo "Checked with PYTHON_BIN=${PYTHON_BIN}" >&2
    echo "If this persists, verify the system driver/runtime is healthy and ${SYSTEM_ONEAPI_SETVARS} is usable." >&2
    exit 1
fi

"${PYTHON_BIN}" benchmarks/bench_sparse_topk.py --output-csv "${OUTPUT_CSV}" "$@"
