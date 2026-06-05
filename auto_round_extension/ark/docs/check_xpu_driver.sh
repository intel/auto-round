#!/usr/bin/env bash
#
# check_xpu_driver.sh
#
# Collects the Intel XPU GPU driver / runtime versions relevant to the ARK
# joint_matrix (m>1 WOQ GEMM) failure:
#
#   RuntimeError: no matrix hardware on the target device, joint_matrix is not supported
#
# Root cause (confirmed 2026-06-03): the SAME auto-round-lib wheel (TLA-OFF,
# md5 88c363a4...) PASSES on a node with Level-Zero driver 1.15.38308 and FAILS
# on a node with 1.15.37833. The difference is the GPU driver / compute-runtime
# version, NOT the kernel build. A new-enough driver can JIT-lower the
# joint_matrix SPIR-V; an older one cannot.
#
# Use this script on any node to capture the driver version and decide whether a
# driver upgrade is needed.
#
# Usage:
#   bash check_xpu_driver.sh                 # uses `python3` on PATH
#   PYTHON=/path/to/venv/bin/python bash check_xpu_driver.sh
#
# Known-good (PASS) reference on this fleet:
#   Level-Zero libze_intel_gpu : 1.15.38308  (Arc Pro B60, Battlemage)
# Known-bad (FAIL) reference:
#   Level-Zero libze_intel_gpu : 1.15.37833  (Graphics 0xe223, Battlemage)

set -uo pipefail

PYTHON="${PYTHON:-python3}"
GOOD_DRIVER_BUILD=38308   # minimum libze_intel_gpu build known to work

hr() { printf '%.0s=' {1..72}; echo; }

hr
echo "ARK XPU driver / runtime version report"
echo "host: $(hostname)    date: $(date -u '+%Y-%m-%d %H:%M:%SZ')"
hr

# ---------------------------------------------------------------------------
# 1. Level-Zero GPU driver (the decisive component)
# ---------------------------------------------------------------------------
echo "[1] Level-Zero GPU driver (libze_intel_gpu)"
ze_so="$(ldconfig -p 2>/dev/null | grep -m1 'libze_intel_gpu.so.1' | awk '{print $NF}')"
if [ -z "${ze_so}" ]; then
  for cand in /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1 \
              /usr/lib64/libze_intel_gpu.so.1; do
    [ -e "$cand" ] && ze_so="$cand" && break
  done
fi
if [ -n "${ze_so}" ] && [ -e "${ze_so}" ]; then
  real="$(readlink -f "${ze_so}")"
  echo "    path        : ${ze_so}"
  echo "    resolved    : ${real}"
  drv_ver="$(echo "${real}" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1)"
  echo "    driver ver  : ${drv_ver:-unknown}"
  drv_build="$(echo "${drv_ver}" | awk -F. '{print $3}')"
  if [ -n "${drv_build}" ]; then
    if [ "${drv_build}" -ge "${GOOD_DRIVER_BUILD}" ] 2>/dev/null; then
      echo "    verdict     : OK (>= ${GOOD_DRIVER_BUILD}, joint_matrix expected to work)"
    else
      echo "    verdict     : TOO OLD (< ${GOOD_DRIVER_BUILD}) -> joint_matrix m>1 GEMM will FAIL; upgrade driver"
    fi
  fi
else
  echo "    libze_intel_gpu.so.1 not found"
fi
echo

# ---------------------------------------------------------------------------
# 2. Compute-runtime / OpenCL driver packages (apt)
# ---------------------------------------------------------------------------
echo "[2] Installed GPU runtime packages (dpkg)"
if command -v dpkg >/dev/null 2>&1; then
  dpkg -l 2>/dev/null \
    | grep -iE 'intel-level-zero|level-zero|intel-opencl|intel-ocloc|compute-runtime|libze|intel-graphics' \
    | awk '{printf "    %-40s %s\n", $2, $3}' \
    || echo "    (none found)"
else
  echo "    dpkg not available"
fi
echo

# ---------------------------------------------------------------------------
# 3. Device + driver as seen by PyTorch XPU
# ---------------------------------------------------------------------------
echo "[3] torch.xpu device properties (PYTHON=${PYTHON})"
if command -v "${PYTHON}" >/dev/null 2>&1; then
  "${PYTHON}" - <<'PY' 2>&1 | sed 's/^/    /'
try:
    import torch
    print("torch       :", torch.__version__)
    if not torch.xpu.is_available():
        print("xpu         : NOT available")
    else:
        for i in range(torch.xpu.device_count()):
            p = torch.xpu.get_device_properties(i)
            name = getattr(p, "name", "?")
            drv  = getattr(p, "driver_version", "?")
            print(f"device[{i}]   : {name}")
            print(f"driver_ver  : {drv}")
except Exception as e:
    print("torch xpu probe failed:", type(e).__name__, e)
PY
else
  echo "    python '${PYTHON}' not found"
fi
echo

# ---------------------------------------------------------------------------
# 4. sycl-ls (if present) - shows aspects incl. matrix support
# ---------------------------------------------------------------------------
echo "[4] sycl-ls (optional)"
if command -v sycl-ls >/dev/null 2>&1; then
  sycl-ls 2>/dev/null | sed 's/^/    /'
else
  echo "    sycl-ls not on PATH (skip)"
fi
echo

# ---------------------------------------------------------------------------
# 5. clinfo driver version (optional)
# ---------------------------------------------------------------------------
echo "[5] clinfo (optional)"
if command -v clinfo >/dev/null 2>&1; then
  clinfo 2>/dev/null | grep -iE 'Device Name|Driver Version|Device Version' | sed 's/^/    /' | head
else
  echo "    clinfo not on PATH (skip)"
fi

hr
echo "Reference: PASS on driver build >= ${GOOD_DRIVER_BUILD} (e.g. 1.15.38308),"
echo "           FAIL on 1.15.37833. Same TLA-OFF wheel; driver is the variable."
hr
