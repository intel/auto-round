#!/usr/bin/env bash
#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Run the ARK XPU int3 (S3) end-to-end generation example on Battlemage.
#
# int3 supports both m==1 (GEMV/decode) and m>1 (GEMM/prefill), so this drives a normal
# multi-token prefill + greedy decode and cross-checks the ARK kernel against a plain-torch
# dequant of the same int3 weights.
#
# Usage:
#   ./run_int3_gen.sh                       # defaults: Qwen3-4B, prompt "1+1=?", 30 new tokens
#   ARK_GEN_MODEL=Qwen/Qwen3-0.6B \
#   ARK_GEN_EXPORT=/tmp/qwen3-w3g128-sym \
#   ARK_GEN_PROMPT="The capital of France is" ARK_GEN_NEW=20 ./run_int3_gen.sh
#
# Prereqs (see docs/ark_xpu_build_bkc.md): oneAPI installed, the build venv on PATH, and the
# compiled auto_round_kernel_xpu*.so sitting next to auto_round_kernel/qlinear.py.

set -euo pipefail

# --- paths (override via env if your layout differs) -------------------------------------------
ONEAPI_SETVARS="${ONEAPI_SETVARS:-/opt/intel/oneapi/setvars.sh}"
# venv that RUNS the example: torch-xpu + auto_round + transformers.
RUN_PY="${RUN_PY:-/home/yiliu7/workspace/venvs/ar/bin/python}"

# Repo dir holding the ARK extension (so `import auto_round_kernel` resolves the built .so).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- e2e knobs ---------------------------------------------------------------------------------
export ARK_GEN_MODEL="${ARK_GEN_MODEL:-Qwen/Qwen3-4B}"
export ARK_GEN_EXPORT="${ARK_GEN_EXPORT:-/tmp/qwen3-4b-w3g128-sym}"
export ARK_GEN_PROMPT="${ARK_GEN_PROMPT:-1+1=?}"
export ARK_GEN_NEW="${ARK_GEN_NEW:-30}"

# --- environment -------------------------------------------------------------------------------
# oneAPI SYCL runtime (icx/level-zero). setvars.sh references unset vars, so relax -u/-e around it.
if [[ -f "${ONEAPI_SETVARS}" ]]; then
  set +u +e
  # shellcheck disable=SC1090
  source "${ONEAPI_SETVARS}" >/dev/null 2>&1 || true
  set -u -e
fi

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export PYTHONPATH="${ARK_DIR}:${PYTHONPATH:-}"

echo "[run] python : ${RUN_PY}"
echo "[run] ark dir: ${ARK_DIR}"
echo "[run] model  : ${ARK_GEN_MODEL}  export=${ARK_GEN_EXPORT}"
echo "[run] prompt : '${ARK_GEN_PROMPT}'  new_tokens=${ARK_GEN_NEW}"
echo "[run] device : ${ONEAPI_DEVICE_SELECTOR}"

exec stdbuf -oL -eL "${RUN_PY}" "${SCRIPT_DIR}/ark_int3_generate.py"
