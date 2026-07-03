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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if test -f /opt/intel/oneapi/setvars.sh; then
    # oneAPI setvars is not compatible with nounset in some releases.
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi

for candidate in \
    /opt/intel/oneapi/vtune/latest/bin64 \
    /opt/intel/oneapi/vtune/latest/bin \
    /opt/intel/oneapi/unitrace/latest/bin
do
    if test -d "${candidate}"; then
        case ":${PATH}:" in
            *":${candidate}:"*) ;;
            *) export PATH="${candidate}:${PATH}" ;;
        esac
    fi
done

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

printf 'oneAPI env ready\n'
printf 'repo=%s\n' "${ROOT_DIR}"
printf 'vtune=%s\n' "$(command -v vtune || echo missing)"
printf 'unitrace=%s\n' "$(command -v unitrace || echo missing)"
