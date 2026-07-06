#!/usr/bin/env python3
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
"""
Non-int8 CPU SDPA validation runbook (NS-parity final state).

This script is the authoritative reference for the non-int8 route status,
ISA requirements, test coverage, and deferred items.  Run it directly to
print the route summary and (optionally) run the available Python tests.

Usage:
    python validate_non_int8_cpu_sdpa.py          # print status table
    python validate_non_int8_cpu_sdpa.py --run     # print + run Python tests

The C++ unit tests must be run separately (see "C++ tests" section below).
"""

import argparse
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Route status table (NS-parity final state)
# ---------------------------------------------------------------------------

ROUTE_TABLE = """
Non-int8 CPU BestLA SDPA route summary (NS-parity final state)
==============================================================

 #  | Q/K/V/dst dtypes     | Launcher            | ISA required    | Tier   | Status
----+----------------------+---------------------+-----------------+--------+-------
 1  | f32 / f16 / f16 / f32| mha_stable_interface| AVX2            | Tier 1 | NS-parity (env-gated)
 2  | f32 / bf16/ bf16/ f32| mha_stable_interface| AVX512F/AMX-BF16| Tier 1 | NS-parity (env-gated)
 3  | f16 / f16 / f16 / f16| mha_stable_interface| AVX512-FP16     | Tier 2 | Internal-only (by design)
 4  | bf16/ bf16/ bf16/ bf16| mha_interface       | AMX-BF16        | Tier 2 | Internal-only (by design)

Exposure tiers:
  Tier 0: Scalar mha_dense_forward — default Python path, always active.
  Tier 1: BestLA mixed routes 1/2 — env-gated by ARK_UNSAFE_BESTLA_MIXED_SDPA=1.
           Python ABI: sdpa() and ark_cpu_bestla_sdpa_packed() / ark_cpu_packed_kv_alloc()
           / ark_cpu_update_packed_kv() (requires BestLA extension build).
  Tier 2: Homogeneous routes 3/4 — C++ only, NOT wired in ark.cpp/Python.
           Internal-only by design; route 3 needs a packed K/V layout bridge,
           route 4 is only justified for a dedicated AMX-BF16 bf16-compute use case
           (route 2 already covers bf16 K/V with full feature set).

Feature support matrix (S=supported, U=unsupported):

  Feature       | Route 1 | Route 2 | Route 3 | Route 4
  --------------+---------+---------+---------+--------
  causal        |    S    |    S    |    S    |    S
  GQA (MQA)     |    S    |    S    |    S    |    U
  padding-right |    S    |    S    |    U    |    U
  alibi (ALIBI8)|    S    |    S    |    U    |    U
  tanh (TANH30) |    S    |    S    |    U    |    U
  prefer_fp32   |    S    |    S    |    U    |    U

Packed/persistent KV cache path (NS-parity decode, Tier 1):
  bestla_sdpa_forward_packed + packed_kv_cache_shape + update_packed_k/v_cache
  — same feature set as routes 1/2 (full S matrix above).
  Python ABI: ark_cpu_packed_kv_alloc / ark_cpu_update_packed_kv / ark_cpu_bestla_sdpa_packed.
  Gate: ARK_UNSAFE_BESTLA_MIXED_SDPA=1.  Promote to default after per-ISA CI coverage.
"""

# ---------------------------------------------------------------------------
# Test coverage map
# ---------------------------------------------------------------------------

TEST_COVERAGE = """
Test coverage map
-----------------

C++ unit tests (build the CPU extension and run test_reorder_kv_main):
  TestReorderKV               — K/V reorder layout correctness (all dtypes/layouts)
  TestPersistentPackedKV      — incremental packed cache update vs one-shot reorder
  TestPackedForwardSetup      — logical capacity, zero-fill, forward arg validation
  TestHomogeneousForwardSetup — routes 3/4 pre-GEMM arg validation (any CPU)
  TestMixedPaddingRight       — routes 1/2 padding-right flag plumbing
  TestMixedAlibiTanh          — routes 1/2 alibi/tanh acceptance; routes 3/4 rejection
  TestMixedNumericalFeatures  — routes 1/2 numerical alibi/tanh/padding-right vs ref
    ISA skip conditions:
      Route 1 (f16 K/V): skip if cpu->AVX2() == false
      Route 2 (bf16 K/V): skip if cpu->AVX512F() == false

Python tests:
  test_ark_cpu_sdpa.py              — Tier 0 scalar path (HND/NHD, causal, GQA)
  test_ark_cpu_mixed_bestla_sdpa.py — Tier 1 mixed routes 1/2 features
    (prefer_fp32, padding-right, alibi, tanh, GQA, causal)
    Requires: ARK_UNSAFE_BESTLA_MIXED_SDPA=1, BestLA CPU extension build.
    ISA skip conditions (pytest.mark.skipif):
      Route 1 (F16): AVX2 required
      Route 2 (BF16): AVX512F required
"""

# ---------------------------------------------------------------------------
# Run commands
# ---------------------------------------------------------------------------

COMMANDS = {
    "Tier 0 scalar (Python)": [
        "pytest",
        "auto_round_extension/ark/test/test_ark_cpu_sdpa.py",
        "-v",
        "-x",
    ],
    "Tier 1 mixed BestLA (Python, requires AVX2/AVX512F)": [
        "env",
        "ARK_UNSAFE_BESTLA_MIXED_SDPA=1",
        "pytest",
        "auto_round_extension/ark/test/test_ark_cpu_mixed_bestla_sdpa.py",
        "-v",
        "-x",
    ],
}

# ---------------------------------------------------------------------------
# Deferred items (final delivery-stage pass only)
# ---------------------------------------------------------------------------

DEFERRED = """
Deferred to final delivery-stage pass
--------------------------------------
  1. CI hardening: per-ISA matrix CI jobs (AVX2, AVX512F, AMX-BF16, AVX512-FP16).
  2. Benchmark baselines: throughput/latency vs Neural Speed reference on each ISA.
  3. Broader hardware validation: SPR, EMR, GNR physical machines for routes 1/2.
  4. Optional future exposure expansion:
     - Promote Tier 1 routes 1/2 to default after CI coverage established.
     - Wire route 3 (fp16×4) in ark.cpp once packed K/V layout bridge is added.
     - Route 4 (bf16×4) only if a dedicated AMX-BF16 bf16-compute use case arises.
     - Remove ARK_UNSAFE_BESTLA_MIXED_SDPA gate once routes 1/2 are default.
  5. Cleanup: remove raw->packed reorder bridge in bestla_sdpa_forward once the
     packed path is the primary route (and per-forward allocation overhead is gone).
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Run Python test suites after printing status")
    args = parser.parse_args()

    print(ROUTE_TABLE)
    print(TEST_COVERAGE)
    print(DEFERRED)

    if not args.run:
        print("Pass --run to execute the Python test suites.")
        return 0

    print("\n" + "=" * 70)
    print("Running Python test suites")
    print("=" * 70 + "\n")

    # Find repo root (parent of the directory containing this script).
    import os

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

    overall = True
    for label, cmd in COMMANDS.items():
        print(f"--- {label} ---")
        # Expand env var prefix into real env when cmd starts with "env VAR=val"
        env = os.environ.copy()
        real_cmd = cmd
        if cmd[0] == "env":
            for item in cmd[1:]:
                if "=" in item:
                    k, v = item.split("=", 1)
                    env[k] = v
                else:
                    real_cmd = cmd[cmd.index(item) :]
                    break

        result = subprocess.run(real_cmd, cwd=repo_root, env=env)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})\n")
            overall = False
        else:
            print("  PASSED\n")

    if overall:
        print("All test suites passed.")
        return 0
    else:
        print("One or more test suites failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
