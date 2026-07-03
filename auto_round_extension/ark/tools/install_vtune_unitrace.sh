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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITRACE_REPO_URL="${UNITRACE_REPO_URL:-https://github.com/intel/pti-gpu.git}"
UNITRACE_REF="${UNITRACE_REF:-master}"
UNITRACE_PREFIX_BASE="${UNITRACE_PREFIX_BASE:-/opt/intel/oneapi/unitrace}"
UNITRACE_INSTALL_VERSION="${UNITRACE_INSTALL_VERSION:-$(date -u +%Y.%m.%d)}"
UNITRACE_PREFIX="${UNITRACE_PREFIX_BASE}/${UNITRACE_INSTALL_VERSION}"
BUILD_DIR="${BUILD_DIR:-/tmp/unitrace-build}"
SRC_DIR="${SRC_DIR:-/tmp/pti-gpu}"

note() {
    printf '[install_vtune_unitrace] %s\n' "$*"
}

warn() {
    printf '[install_vtune_unitrace][warn] %s\n' "$*" >&2
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'missing required command: %s\n' "$1" >&2
        exit 1
    }
}

has_valid_keyring() {
    test -s /usr/share/keyrings/oneapi-archive-keyring.gpg || return 1
    gpg --batch --quiet --show-keys /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null 2>&1
}

install_oneapi_keyring() {
    local tmp_key
    tmp_key="$(mktemp)"

    note "fetching Intel oneAPI APT key"
    if curl -4fsSL --retry 5 --retry-delay 2 --connect-timeout 15 \
        https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        -o "${tmp_key}"; then
        :
    else
        rm -f "${tmp_key}"
        warn "failed to reach apt.repos.intel.com over IPv4"
        warn "download the key manually on a network that can reach Intel and place it at:"
        warn "  /usr/share/keyrings/oneapi-archive-keyring.gpg"
        warn "for example:"
        warn "  curl -fsSL <intel-key-url> | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg >/dev/null"
        exit 1
    fi

    if ! gpg --batch --yes --dearmor -o "${tmp_key}.gpg" "${tmp_key}" >/dev/null 2>&1; then
        rm -f "${tmp_key}" "${tmp_key}.gpg"
        warn "downloaded Intel key is not valid OpenPGP data"
        exit 1
    fi

    sudo install -D -m 0644 "${tmp_key}.gpg" /usr/share/keyrings/oneapi-archive-keyring.gpg
    rm -f "${tmp_key}" "${tmp_key}.gpg"
}

need_cmd sudo
need_cmd git
need_cmd cmake
need_cmd python3
need_cmd curl
need_cmd gpg

note "installing VTune prerequisites and package-manager metadata"
sudo apt-get update
sudo apt-get install -y gpg-agent wget ca-certificates build-essential pkg-config

if has_valid_keyring; then
    note "Intel oneAPI APT keyring already present"
else
    if test -e /usr/share/keyrings/oneapi-archive-keyring.gpg; then
        warn "removing invalid Intel oneAPI keyring"
        sudo rm -f /usr/share/keyrings/oneapi-archive-keyring.gpg
    fi
    install_oneapi_keyring
fi

if ! test -f /etc/apt/sources.list.d/oneAPI.list; then
    note "adding Intel oneAPI APT repository"
    printf 'deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main\n' \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list >/dev/null
fi

note "refreshing APT metadata"
sudo apt-get update

if apt-cache show intel-oneapi-vtune >/dev/null 2>&1; then
    note "installing Intel VTune Profiler from APT"
    sudo apt-get install -y intel-oneapi-vtune
else
    note "intel-oneapi-vtune is not visible in the configured APT metadata"
    note "leaving VTune uninstalled; rerun after fixing the Intel repo or use the standalone installer"
fi

note "bootstrapping oneAPI environment for unitrace build"
set +u
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -u

note "cloning unitrace source from ${UNITRACE_REPO_URL} (${UNITRACE_REF})"
rm -rf "${SRC_DIR}" "${BUILD_DIR}"
git clone --depth 1 --branch "${UNITRACE_REF}" "${UNITRACE_REPO_URL}" "${SRC_DIR}"

mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}" >/dev/null
note "configuring unitrace"
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_MPI=0 \
    -DCMAKE_INSTALL_PREFIX="${UNITRACE_PREFIX}" \
    "${SRC_DIR}/tools/unitrace"

note "building unitrace"
cmake --build . --parallel "$(nproc)"

note "installing unitrace into ${UNITRACE_PREFIX}"
sudo cmake --install .

note "refreshing latest symlink at ${UNITRACE_PREFIX_BASE}/latest"
sudo mkdir -p "${UNITRACE_PREFIX_BASE}"
sudo ln -sfn "${UNITRACE_PREFIX}" "${UNITRACE_PREFIX_BASE}/latest"
popd >/dev/null

note "installation probe"
if command -v vtune >/dev/null 2>&1; then
    vtune --version || true
else
    note "vtune is not yet on PATH in this shell; source tools/enable_vtune_unitrace_env.sh"
fi

if test -x "${UNITRACE_PREFIX_BASE}/latest/bin/unitrace"; then
    "${UNITRACE_PREFIX_BASE}/latest/bin/unitrace" --version || true
fi

note "done"
