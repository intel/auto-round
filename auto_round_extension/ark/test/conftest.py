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

* ``--run-moe-prefill-perf`` -- opt in to running
  ``test_moe_prefill_perf.py``. The MoE prefill perf sweep is too
  expensive for a routine ``pytest`` invocation and is therefore
  **skipped by default**. Pass this flag (or select the file / its
  ``moe_prefill_perf`` marker explicitly) to run it.
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
    parser.addoption(
        "--run-moe-prefill-perf",
        action="store_true",
        default=False,
        help=(
            "Run test_moe_prefill_perf.py. The MoE prefill perf sweep is "
            "too expensive to run on every pytest invocation, so it is "
            "skipped by default. Pass this flag to opt in."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "moe_prefill_perf: MoE prefill perf sweep; skipped by default, " "opt in via --run-moe-prefill-perf.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip ``test_moe_prefill_perf.py`` unless explicitly opted in.

    The prefill perf sweep is too time-consuming for a default
    ``pytest`` run. It is skipped unless one of the following is true:

    * ``--run-moe-prefill-perf`` was passed on the CLI, or
    * the user selected the file / its ``moe_prefill_perf`` marker
      explicitly (e.g. ``pytest test_moe_prefill_perf.py`` or
      ``pytest -m moe_prefill_perf``).
    """
    if config.getoption("--run-moe-prefill-perf", default=False):
        return

    # If the user targeted the perf file / marker explicitly, honor it.
    args = [str(a) for a in (config.args or [])]
    if any("test_moe_prefill_perf" in a for a in args):
        return
    mark_expr = config.getoption("-m", default="") or ""
    if "moe_prefill_perf" in mark_expr:
        return

    import pytest

    skip_marker = pytest.mark.skip(
        reason=(
            "test_moe_prefill_perf.py is skipped by default because the "
            "MoE prefill perf sweep is too time-consuming. Pass "
            "--run-moe-prefill-perf (or select the file / -m "
            "moe_prefill_perf explicitly) to run it."
        )
    )
    for item in items:
        if "test_moe_prefill_perf.py" in str(item.fspath):
            item.add_marker(skip_marker)
