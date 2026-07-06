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
Non-int8 CPU SDPA validation runbook — delivery-stage final state.

This script is the authoritative reference for the non-int8 route status,
ISA requirements, test coverage, CI/readiness matrix, promotion decisions,
and known follow-up items.  Run it directly to print the full summary and
(optionally) execute the available Python tests.

Usage:
    python validate_non_int8_cpu_sdpa.py          # print full summary
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
    test_homogeneous_half_uses_tier0_not_internal_routes — Module C: asserts that
      homogeneous fp16/bf16 Q/K/V inputs do NOT enter routes 3/4 (internal-only)
      and produce correct output via Tier 0 scalar, regardless of env gate state.
  test_ark_cpu_mixed_bestla_sdpa.py — Tier 1 mixed routes 1/2 features
    (prefer_fp32, padding-right, alibi, tanh, GQA, causal)
    Requires: ARK_UNSAFE_BESTLA_MIXED_SDPA=1, BestLA CPU extension build.
    ISA skip conditions (pytest.mark.skipif):
      Route 1 (F16): AVX2 required
      Route 2 (BF16): AVX512F required
    test_bestla_packed_sdpa_numerical_parity — Module B: packed path (alloc +
      update + forward) vs PyTorch SDPA reference for fp16/bf16, causal on/off.
    test_bestla_raw_vs_packed_output_consistency — Module B: raw mixed path and
      packed path must agree on the same inputs within per-dtype tolerance.
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
    "Tier 1 packed path (Python, requires AVX2/AVX512F)": [
        "env",
        "ARK_UNSAFE_BESTLA_MIXED_SDPA=1",
        "pytest",
        "auto_round_extension/ark/test/test_ark_cpu_mixed_bestla_sdpa.py",
        "-v",
        "-k",
        "packed",
    ],
}

# ---------------------------------------------------------------------------
# CI / readiness matrix (delivery-stage final state)
# ---------------------------------------------------------------------------

CI_MATRIX = """
CI / readiness matrix
---------------------

ISA tier       | Runner class              | Tier 0 (scalar) | Tier 1 mixed R1/R2 | C++ UTs
---------------+---------------------------+-----------------+--------------------+--------
AVX2           | ubuntu-latest (x86_64)    | required        | required (R1 only) | required
AVX512F        | self-hosted SPR/EMR/GNR   | required        | required (R1+R2)   | required
AMX-BF16       | self-hosted SPR/EMR/GNR   | required        | required (R2 AMX)  | required
AVX512-FP16    | self-hosted GNR/SRF       | required        | skip (R3 internal) | required

Notes:
  * AVX2 / standard x86_64: GitHub Actions ubuntu-latest is sufficient.
    Route 1 (f16 K/V) runs; route 2 (bf16 K/V) ISA-skipped by both Python and C++ UTs.
  * AVX512F (no AMX): SPR/EMR without AMX-BF16 enabled. Route 2 fp32-score path.
  * AMX-BF16: SPR/EMR/GNR with AMX enabled. Route 2 AMX-BF16 compute path.
  * AVX512-FP16: GNR/SRF. Used only for C++ UT coverage of route 3 (internal-only).
  * Tier 1 packed KV cache path follows route 1/2 ISA requirements exactly.

CI workflow definition: .github/workflows/non_int8_cpu_sdpa.yml
  -- AVX2 job: runs on ubuntu-latest, exercises Tier 0 + Tier 1 R1 + C++ UT dispatch/reorder.
  -- SPR/EMR/GNR jobs: self-hosted, full ISA coverage for routes 1/2 and packed path.
  -- These jobs must pass before routes 1/2 can be promoted to default.

Benchmark commands (identify for regression tracking):
  # Tier 0 vs Tier 1 raw-path throughput comparison (all ISAs):
  python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py \\
      --dtype float32 --shape all

  # Tier 1 raw vs packed path comparison (routes 1/2, ISA-specific):
  python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py \\
      --dtype float16 --shape decode --mode both      # R1 raw vs packed
  python auto_round_extension/ark/test/bench_ark_cpu_sdpa.py \\
      --dtype bfloat16 --shape decode --mode both     # R2 raw vs packed

  Regression-sensitive behavior:
    - Tier 0 scalar: latency must not exceed the 1.3× tolerance vs PyTorch math SDPA.
    - Tier 1 mixed raw path: must show ≥1.0× speedup on decode shapes vs Tier 0.
    - Tier 1 packed vs raw: packed must match or beat raw (no per-forward reorder).
    - Numerical parity: max absolute error must remain within documented tolerances
      (fp16: 3e-2, bf16: 8e-2) against PyTorch SDPA on the same dtype-round-tripped inputs.
"""

# ---------------------------------------------------------------------------
# Promotion decision (delivery-stage final)
# ---------------------------------------------------------------------------

PROMOTION_DECISION = """
Promotion decision — delivery-stage final
------------------------------------------

Routes 1/2 (Tier 1): REMAIN ENV-GATED (ARK_UNSAFE_BESTLA_MIXED_SDPA=1).

Decision rationale:
  - Implementation is structurally complete and NS-parity validated in Python.
  - C++ unit tests cover layout correctness (TestReorderKV), packed cache
    updates (TestPersistentPackedKV), setup/dispatch (TestPackedForwardSetup,
    TestHomogeneousForwardSetup), and feature plumbing (TestMixedPaddingRight,
    TestMixedAlibiTanh, TestMixedNumericalFeatures).
  - Python ABI end-to-end parity confirmed for: causal, GQA, padding-right,
    alibi, tanh, prefer_fp32, and packed KV cache.
  - NO per-ISA CI coverage on physical SPR/EMR/GNR hardware yet.
  - NO benchmark baselines recorded against Neural Speed reference paths.

Blockers before routes 1/2 can be promoted to default:
  [B1] CI jobs passing on AVX2 runner (standard ubuntu-latest).
  [B2] CI jobs passing on AVX512F self-hosted runner (SPR or EMR).
  [B3] CI jobs passing on AMX-BF16 runner for route 2 AMX path.
  [B4] Benchmark baseline recorded: raw mixed vs packed mixed on at least one
       physical ISA target (SPR preferred).
  [B5] The ARK_UNSAFE_BESTLA_MIXED_SDPA gate removal reviewed and approved
       (default-on path must not regress Tier 0 scalar numerical parity).

Routes 3/4 (Tier 2): REMAIN INTERNAL-ONLY.

Decision rationale:
  - Route 3 (fp16×4, AVX512-FP16) requires a packed K/V layout bridge for PLAIN
    inputs that is not yet implemented. Wiring in ark.cpp before that bridge
    exists would expose an incomplete path.
  - Route 4 (bf16×4, AMX-BF16) provides no feature advantage over route 2 (which
    already covers bf16 K/V with full fp32-score feature set). A dedicated
    AMX-BF16 bf16-compute preference use case has not been identified.
  - No promotion path for routes 3/4 in this delivery pass.
"""

# ---------------------------------------------------------------------------
# Deferred / follow-up items (delivery-stage final)
# ---------------------------------------------------------------------------

DEFERRED = """
Known follow-up items after delivery-stage pass
------------------------------------------------
  [F1] Unblock B1–B5 above to promote routes 1/2 to default (remove gate).
  [F2] Per-ISA CI jobs: wire AVX2 (ubuntu-latest) job to pass in every PR;
       wire SPR/EMR self-hosted jobs once hardware is available.
  [F3] Benchmark baselines: record decode/prefill throughput on SPR for routes 1/2
       vs Tier 0 scalar and vs Neural Speed mha_dense reference.
  [F4] Route 3 promotion path: implement PLAIN->NTILE24_ROWPACK1 layout bridge
       for fp16 K/V in ark.cpp, then wire route 3 behind the same env gate.
  [F5] Route 4: no planned promotion unless a bf16-compute-preference use case arises.
  [F6] Packed path cleanup: remove raw->packed per-forward reorder bridge in
       bestla_sdpa_forward once the persistent packed path is the primary route.
  [F7] Remove ARK_UNSAFE_BESTLA_MIXED_SDPA gate after B1–B4 are resolved.
"""

# ---------------------------------------------------------------------------
# Final delivery summary
# ---------------------------------------------------------------------------

DELIVERY_SUMMARY = """
Final delivery summary — non-int8 CPU BestLA SDPA
==================================================

DONE (this delivery pass):
  * Route 1 (f32/f16/f16/f32): NS-parity, env-gated, fully tested in Python + C++ UT.
  * Route 2 (f32/bf16/bf16/f32): NS-parity, env-gated, fully tested in Python + C++ UT.
  * Routes 3/4: finalized as internal-only by design; C++ UT covers setup/rejection.
  * Packed/persistent KV cache path: Python-accessible under env gate; C++ UT validates
    layout correctness (TestReorderKV, TestPersistentPackedKV, TestPackedForwardSetup).
  * Feature coverage validated end-to-end (Python + C++): causal, GQA, padding-right,
    alibi (ALIBI8), tanh (TANH30), prefer_fp32.
  * Final dispatch rule enforced: first layer by Q/K/V/dst dtype tuple; second layer by
    ISA + layout + stride/shape within each dtype-specific route.
  * Python ABI complete: sdpa(), ark_cpu_packed_kv_alloc(), ark_cpu_update_packed_kv(),
    ark_cpu_bestla_sdpa_packed() — all documented and gated.

VALIDATED (this pass):
  * Python numerical tests: test_ark_cpu_sdpa.py (Tier 0), test_ark_cpu_mixed_bestla_sdpa.py
    (Tier 1) — both structured to ISA-skip cleanly without BestLA extension present.
  * C++ UTs: TestReorderKV, TestPersistentPackedKV, TestPackedForwardSetup,
    TestHomogeneousForwardSetup, TestMixedPaddingRight, TestMixedAlibiTanh,
    TestMixedNumericalFeatures — all runnable when extension is built.
  * Runbook (this file): authoritative reference for route status, ISA requirements,
    CI matrix, promotion decisions, and follow-up items.

BENCHMARKED:
  * bench_ark_cpu_sdpa.py: decode + prefill sweep, Tier 0 vs PyTorch math SDPA,
    raw-vs-packed comparison (--mode both), CSV output for regression tracking.
  * Physical hardware baselines (SPR/EMR/GNR): NOT YET RECORDED. Required for B4.

GATED (ARK_UNSAFE_BESTLA_MIXED_SDPA=1):
  * Routes 1/2 raw path (bestla_sdpa_forward).
  * Routes 1/2 packed KV cache path (bestla_sdpa_forward_packed + helpers).
  * All three Python-facing packed-cache functions.

INTERNAL-ONLY (NOT in Python ABI, NOT wired in ark.cpp):
  * Route 3: bestla_sdpa_forward_homogeneous with f16 dtype.
  * Route 4: bestla_sdpa_forward_homogeneous with bf16 dtype.

FOLLOW-UP REQUIRED (see [F1]–[F7] above):
  * Per-ISA CI coverage (B1–B3).
  * Benchmark baselines on physical hardware (B4).
  * Gate removal after B1–B5 resolved (F1, F7).
  * Route 3 promotion path (F4) — not in this delivery pass.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Run Python test suites after printing status")
    args = parser.parse_args()

    print(ROUTE_TABLE)
    print(TEST_COVERAGE)
    print(CI_MATRIX)
    print(PROMOTION_DECISION)
    print(DEFERRED)
    print(DELIVERY_SUMMARY)

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
