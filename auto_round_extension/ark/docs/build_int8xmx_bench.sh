#!/usr/bin/env bash
# Build + run the standalone int8-XMX GEMM benchmark (IKblockGemmDQCore).
# Mirrors the SYCL flags the bestla_benchmark target uses (bestla/CMakeLists.txt:225-234):
# JIT spir64 with the BMG device hint and the SPIR-V extensions joint_matrix needs.
#
# Usage:
#   source /opt/intel/oneapi/setvars.sh
#   export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"
#   ./build_int8xmx_bench.sh            # build, then run the default shape sweep
#   ./build_int8xmx_bench.sh --no-run   # build only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${HERE}/../auto_round_kernel" && pwd)"
BESTLA="${KERNEL_ROOT}/bestla"          # for  #include "bestla/bestla_utils.h"
BESTLA_INNER="${BESTLA}/bestla"         # for  #include "sycl/sycl_gemm.h"

SRC="${HERE}/int8xmx_gemm_bench.cpp"
OUT="${HERE}/int8xmx_gemm_bench"

CXX="${CXX:-icpx}"
DEVICE="${SYCL_DEVICE_NAME:-bmg_g21}"

command -v "${CXX}" >/dev/null || { echo "ERROR: ${CXX} not found. Run: source /opt/intel/oneapi/setvars.sh"; exit 1; }

echo ">>> Compiling ${SRC##*/}  (device=${DEVICE})"
"${CXX}" -std=c++17 -O2 -fsycl -w -fno-sycl-instrument-device-code \
  -DBTLA_SYCL \
  -I"${BESTLA_INNER}" -I"${BESTLA}" \
  "${SRC}" -o "${OUT}" \
  -fsycl-targets=spir64 \
  -Xs "-device ${DEVICE}" \
  -Xspirv-translator \
  "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"

echo ">>> Built: ${OUT}"

if [[ "${1:-}" == "--no-run" ]]; then
  echo ">>> Skipping run (--no-run). Invoke manually:"
  echo "    ONEAPI_DEVICE_SELECTOR=level_zero:gpu ${OUT}"
  exit 0
fi

echo ">>> Running on level_zero:gpu"
ONEAPI_DEVICE_SELECTOR=level_zero:gpu "${OUT}"
