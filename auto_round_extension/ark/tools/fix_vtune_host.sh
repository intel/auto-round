#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

VTUNE_ROOT="${VTUNE_ROOT:-/opt/intel/oneapi/vtune/2026.2}"
SEP_SRC_DIR="${VTUNE_ROOT}/sepdk/src"
SYSCTL_CONF="${SYSCTL_CONF:-/etc/sysctl.d/10-vtune.conf}"
KERNEL_VERSION="$(uname -r)"

note() {
    printf '[fix_vtune_host] %s\n' "$*"
}

warn() {
    printf '[fix_vtune_host][warn] %s\n' "$*" >&2
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'missing required command: %s\n' "$1" >&2
        exit 1
    }
}

need_cmd sudo
need_cmd uname
need_cmd tee

note "installing host prerequisites used by VTune post-install and driver builds"
sudo apt-get update
sudo apt-get install -y pkg-config build-essential gcc-14 g++-14

if apt-cache show gcc-15 >/dev/null 2>&1; then
    note "installing gcc-15 toolchain for SEP driver compatibility with this kernel"
    sudo apt-get install -y gcc-15 g++-15
    SEP_CC=/usr/bin/gcc-15
else
    warn "gcc-15 is not available from configured APT repos"
    SEP_CC=
fi

note "writing VTune sysctl settings to ${SYSCTL_CONF}"
sudo tee "${SYSCTL_CONF}" >/dev/null <<'EOF'
kernel.yama.ptrace_scope = 0
kernel.perf_event_paranoid = 0
kernel.kptr_restrict = 0
EOF
sudo sysctl --system

if test -d "${SEP_SRC_DIR}"; then
    if test -n "${SEP_CC:-}" && test -x "${SEP_CC}"; then
        note "building SEP driver with ${SEP_CC} for kernel ${KERNEL_VERSION}"
        pushd "${SEP_SRC_DIR}" >/dev/null
        sudo ./build-driver --non-interactive --c-compiler="${SEP_CC}" --kernel-version="${KERNEL_VERSION}"
        note "loading SEP driver"
        sudo ./insmod-sep
        popd >/dev/null
    else
        warn "skipping SEP driver build because a gcc-15 compiler is not available"
        warn "VTune driverless analyses can still work after the sysctl change"
    fi
else
    warn "SEP source directory not found at ${SEP_SRC_DIR}"
fi

if test -x "${VTUNE_ROOT}/bin64/vtune-self-checker.sh"; then
    note "running VTune self-checker"
    "${VTUNE_ROOT}/bin64/vtune-self-checker.sh" || true
fi

note "done"
