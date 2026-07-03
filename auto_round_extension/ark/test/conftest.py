#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

"""Pytest configuration for ``auto_round_extension/ark/test``.

Registers CLI flags used by the MoE perf tests:

* ``--minimax-real-only`` -- restrict the MoE prefill perf sweep to the
  ``"minimax real"`` rows only (the heavy-tailed tokens-per-expert
  distribution). Without the flag the (default-restricted) shape matrix
  is used.

* ``--all-shapes`` -- opt in to the full shape matrix for the MoE prefill
  and decode perf tests. Without the flag the default is the smallest
  shape only (2K for prefill, bs1 for decode) so a CI run stays short;
  pass ``--all-shapes`` to reproduce the full performance sweep.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--minimax-real-only",
        action="store_true",
        default=False,
        help=(
            "In test_moe_prefill_perf.py, restrict the shape sweep to rows "
            "whose label contains 'minimax real' (the heavy-tailed "
            "tokens-per-expert distribution). Default: run all shapes "
            "(after the --all-shapes / default-smallest filter)."
        ),
    )
    parser.addoption(
        "--all-shapes",
        action="store_true",
        default=False,
        help=(
            "Run the full shape matrix for test_moe_prefill_perf.py and "
            "test_moe_decode_perf.py. Default (flag absent): run only the "
            "smallest shape group (2K for prefill, bs1 for decode) so the "
            "perf tests stay short in CI."
        ),
    )
