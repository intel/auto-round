#!/usr/bin/env bash

# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file instead of executing it:" >&2
    echo "  source benchmarks/source_env_xpu67_oneapi2025.sh" >&2
    exit 1
fi

ONEAPI_ROOT_DIR="${ONEAPI_ROOT_DIR:-/home/yiliu4/intel/oneapi}"
ONEAPI_COMPILER_DIR="${ONEAPI_COMPILER_DIR:-${ONEAPI_ROOT_DIR}/compiler/2025.3}"
SYSTEM_ONEAPI_SETVARS="${SYSTEM_ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
XPU_MASK="${XPU_MASK:-6,7}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/ark-torch-212/bin/python}"

if [[ -f "${SYSTEM_ONEAPI_SETVARS}" ]]; then
    # Seed the broader XPU runtime environment from the system oneAPI install.
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
export ARK_BENCH_PYTHON="${PYTHON_BIN}"

echo "Configured ARK benchmark environment:"
echo "  ONEAPI_ROOT=${ONEAPI_ROOT}"
echo "  CMPLR_ROOT=${CMPLR_ROOT}"
echo "  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
echo "  ARK_BENCH_PYTHON=${ARK_BENCH_PYTHON}"

"${ARK_BENCH_PYTHON}" - <<'PY'
import torch
print(f"torch={torch.__version__} xpu_available={torch.xpu.is_available()} xpu_count={torch.xpu.device_count() if torch.xpu.is_available() else 0}")
PY
