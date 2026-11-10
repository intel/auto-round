# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""CLI package for AutoRound.

Modules:
    main.py       - command routing, RECIPES, tune(), eval entry points
    parser.py     - argparse parser construction (all explicit flags here)
    algorithms.py - per-algorithm flag registration and config building
"""

from auto_round.cli.main import run, run_best, run_light, run_rtn, run_opt_rtn, run_mllm, run_eval  # noqa: F401
