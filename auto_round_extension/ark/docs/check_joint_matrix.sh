#!/usr/bin/env bash
#
# check_joint_matrix.sh  —  one-click joint_matrix driver capability check
#
# Builds a minimal standalone joint_matrix program (jm_probe.cpp, int8
# M8xN16xK32 DPAS — the SAME tile ARK's IGemmDQCore uses for the m>1 WOQ GEMM)
# to GENERIC SPIR-V (spir64, NO AOT), then runs it. Success/failure therefore
# depends purely on whether the GPU driver can JIT-lower the joint_matrix SPIR-V.
#
#   PASS -> "JOINT_MATRIX_OK <checksum>"  => driver supports joint_matrix; ARK m>1 GEMM will work
#   FAIL -> "no matrix hardware on the target device, joint_matrix is not supported"
#                                         => driver too old; upgrade GPU driver/compute-runtime
#
# Confirmed reference (2026-06-03), same byte-identical SPIR-V binary:
#   driver 1.15.38308 (compute-runtime 26.18) -> JOINT_MATRIX_OK
#   driver 1.15.37833 (compute-runtime 26.14) -> JOINT_MATRIX_FAIL
#
# Usage:
#   bash check_joint_matrix.sh
#   # optionally point at a specific oneAPI and/or runtime libs:
#   ICPX=/opt/intel/oneapi/compiler/2025.3/bin/icpx \
#   RUNTIME_LIB=/home/you/venv/lib \
#   bash check_joint_matrix.sh
#
# No compiler on the target node (e.g. driver lives on a bare host, oneAPI is
# only inside a container)? Build jm_probe once on any node WITH icpx, copy the
# resulting binary over, and run it directly:
#   icpx -fsycl -fsycl-targets=spir64 -O2 jm_probe.cpp -o jm_probe
#   scp jm_probe target:/tmp/ ; ssh target 'LD_LIBRARY_PATH=/venv/lib /tmp/jm_probe'
# The binary is generic SPIR-V, so it is portable across Intel GPUs/drivers.
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/jm_probe.cpp"
BIN="$(mktemp -u /tmp/jm_probe.XXXXXX)"

[ -f "${SRC}" ] || { echo "ERROR: ${SRC} not found"; exit 1; }

# ---- locate icpx --------------------------------------------------------------
ICPX="${ICPX:-}"
if [ -z "${ICPX}" ]; then
  if command -v icpx >/dev/null 2>&1; then
    ICPX="$(command -v icpx)"
  else
    for c in /opt/intel/oneapi/compiler/latest/bin/icpx \
             /opt/intel/oneapi/compiler/2025.3/bin/icpx \
             /opt/intel/oneapi/compiler/*/bin/icpx; do
      [ -x "$c" ] && ICPX="$c" && break
    done
  fi
fi
[ -n "${ICPX}" ] && [ -x "${ICPX}" ] || {
  echo "ERROR: icpx (oneAPI DPC++) not found. Source setvars.sh or set ICPX=..."
  exit 1
}

# Source the matching oneAPI env so the compiler's own libs are found at build time.
# (vendor scripts reference unset vars, so relax `set -u` while sourcing.)
ONEAPI_ROOT="$(echo "${ICPX}" | sed -E 's@/compiler/[^/]+/bin/icpx$@@')"
ICPX_VER="$(echo "${ICPX}" | grep -oE 'compiler/[^/]+' | cut -d/ -f2)"
set +u
if [ -f "/opt/intel/oneapi/compiler/${ICPX_VER}/env/vars.sh" ]; then
  # shellcheck disable=SC1090
  source "/opt/intel/oneapi/compiler/${ICPX_VER}/env/vars.sh" >/dev/null 2>&1 || true
elif [ -f "${ONEAPI_ROOT}/setvars.sh" ]; then
  # shellcheck disable=SC1090
  source "${ONEAPI_ROOT}/setvars.sh" >/dev/null 2>&1 || true
fi
set -u

echo "icpx   : ${ICPX}"

# ---- build to generic SPIR-V (no AOT) ----------------------------------------
echo "build  : ${ICPX} -fsycl -fsycl-targets=spir64 -O2"
if ! "${ICPX}" -fsycl -fsycl-targets=spir64 -O2 "${SRC}" -o "${BIN}" 2>/tmp/jm_build.err; then
  echo "ERROR: compile failed:"; cat /tmp/jm_build.err; exit 1
fi

# ---- run (optionally with a specific runtime LD_LIBRARY_PATH) -----------------
# RUNTIME_LIB lets you test against the exact libsycl/driver shim a given venv uses.
if [ -n "${RUNTIME_LIB:-}" ]; then
  export LD_LIBRARY_PATH="${RUNTIME_LIB}:${LD_LIBRARY_PATH:-}"
  echo "runtime: LD_LIBRARY_PATH=${RUNTIME_LIB}"
fi

echo "----------------------------------------------------------------------"
"${BIN}"
rc=$?
echo "----------------------------------------------------------------------"
rm -f "${BIN}"

if [ "${rc}" -eq 0 ]; then
  echo "RESULT : PASS — driver JIT-lowers joint_matrix (ARK m>1 WOQ GEMM will work)"
else
  echo "RESULT : FAIL — driver cannot lower joint_matrix; UPGRADE GPU driver (>= build 38308)"
fi
exit "${rc}"
