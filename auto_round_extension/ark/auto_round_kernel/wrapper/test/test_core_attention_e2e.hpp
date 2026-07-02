//  Copyright (c) 2026 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#pragma once

// Core E2E Milestone: first end-to-end validation of the four main migrated CPU
// BestLA attention dtype tuples, exercised through the *same* two-layer
// Neural-Speed-style dispatch the production entries use. This is deliberately
// NOT a numerical GEMM benchmark: like the other CPU wrapper tests it must be
// deterministic on any CPU (CI runs on hosts without AVX512-FP16 / AMX-BF16), so
// it drives each tuple through both dispatch layers up to the terminal pre-kernel
// gate and asserts the correct host-capability-conditioned outcome.
//
// The four target dtype tuples, and their DISTINCT routes (never collapsed into a
// single generic homogeneous/mixed branch), are:
//
//   dtype tuple (Q,K,V,dst) | entry / first-layer route            | 2nd-layer ISA
//   ------------------------+--------------------------------------+--------------
//   fp32,fp16,fp16,fp32     | bestla_sdpa_forward(.., F16)  [mixed] | AVX2
//   fp32,bf16,bf16,fp32     | bestla_sdpa_forward(.., BF16) [mixed] | AVX512F
//   fp16,fp16,fp16,fp16     | bestla_sdpa_forward_homogeneous(F16)  | AVX512-FP16
//   bf16,bf16,bf16,bf16     | bestla_sdpa_forward_homogeneous(BF16) | AMX-BF16
//
// Dispatch model mirrored here (see sdpa.cpp for the production copy):
//   1. First layer -- the full Q/K/V/dst dtype tuple selects the entry + launcher
//      family. The mixed tuples (fp32 Q/dst + low-precision K/V) go through
//      bestla_sdpa_forward; the homogeneous tuples (one shared element type) go
//      through bestla_sdpa_forward_homogeneous. These are separate C-ABI entries,
//      not one branch keyed on a "homogeneous vs mixed" flag.
//   2. Second layer -- inside the dtype-specific route, ISA + layout + stride
//      conditions select the concrete kernel. Each route ends in an explicit ISA
//      capability gate that raises a route-specific std::runtime_error when the
//      required extension is missing (instead of a release-mode-stripped assert or
//      a silent wrong result).
//
// What this test asserts per tuple, using the real bestla CpuDevice probe so the
// expectation matches the running host:
//   * required ISA ABSENT  -> the entry raises std::runtime_error whose message
//     names the required extension (the loud second-layer gate), and is NOT a
//     first-layer/route std::invalid_argument.
//   * required ISA PRESENT  -> the entry passes both dispatch layers and stops at
//     the shared pre-kernel gate (threading pool required), raising
//     std::invalid_argument("... threading pool must be provided"). Reaching this
//     point proves the full dtype-tuple dispatch resolved to a runnable kernel on
//     this host; true numerical parity for the runnable path is covered by the
//     Python e2e (test_ark_cpu_mixed_bestla_sdpa.py for the mixed tuples) and is a
//     follow-up for the homogeneous tuples once they are wired into the Python
//     C-ABI with packed operands.
//
// On the AVX2-class CI hosts this file runs on today, the mixed-fp16 tuple reaches
// the threading gate (AVX2 present) while the other three raise their explicit ISA
// errors -- so both outcome branches above are exercised in one run.

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestla/bestla_device.h"

#include "ark/cpu/mha_dense.h"
#include "ark/cpu/sdpa.h"

namespace ark::cpu {

struct TestCoreAttentionE2E {
  TestCoreAttentionE2E() { run_all(); }

  // First-layer dispatch family for a dtype tuple.
  enum class Route { Mixed, Homogeneous };

  struct Tuple {
    const char* name;    // human-readable Q/K/V/dst dtype tuple
    Route route;         // first-layer entry family
    BTLA_DTYPE dt;       // mixed: K/V dtype (Q/dst are fp32); homogeneous: shared dtype
    const char* isa;     // substring the second-layer ISA gate error must contain
  };

  static std::vector<Tuple> tuples() {
    return {
        {"(fp32,fp16,fp16,fp32) mixed", Route::Mixed, BTLA_DTYPE::F16, "AVX2"},
        {"(fp32,bf16,bf16,fp32) mixed", Route::Mixed, BTLA_DTYPE::BF16, "AVX512F"},
        {"(fp16,fp16,fp16,fp16) homogeneous", Route::Homogeneous, BTLA_DTYPE::F16, "AVX512-FP16"},
        {"(bf16,bf16,bf16,bf16) homogeneous", Route::Homogeneous, BTLA_DTYPE::BF16, "AMX-BF16"},
    };
  }

  // True when the running host provides the extension the tuple's second-layer
  // gate requires. Uses the exact bestla probe sdpa.cpp gates on.
  static bool host_has_isa(const Tuple& t) {
    auto* cpu = bestla::device::CpuDevice::getInstance();
    if (t.route == Route::Mixed) {
      return t.dt == BTLA_DTYPE::F16 ? cpu->AVX2() : cpu->AVX512F();
    }
    return t.dt == BTLA_DTYPE::F16 ? cpu->AVX512_FP16() : cpu->AMX_BF16();
  }

  // Build a route-valid, PLAIN arg bundle for `t` so first-layer dispatch and the
  // second-layer route/stride validation both pass and the call reaches the ISA
  // gate. threading is left null on purpose: the ISA gate runs before the
  // threading requirement, so a host WITHOUT the ISA stops at the gate, while a
  // host WITH the ISA falls through to the (shared) threading gate. Buffers are
  // sized generously; only their non-null-ness matters before the kernel runs.
  static attn_fwd_args_t make_args(const Tuple& t, std::vector<uint8_t>& q, std::vector<uint8_t>& k,
                                   std::vector<uint8_t>& v, std::vector<uint8_t>& dst) {
    attn_fwd_args_t a{};
    a.Q = q.data();
    a.K = k.data();
    a.V = v.data();
    a.dst = dst.data();
    a.batch_size = 1;
    a.head_size = 8;
    a.sl_q = 1;
    a.sl_kv = 4;
    a.Q_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.K_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.V_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.dst_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.threading = nullptr;
    if (t.route == Route::Homogeneous) {
      // Homogeneous routes additionally validate layout/stride/GQA before the ISA
      // gate; satisfy their accept contract (contiguous PLAIN K/V strides).
      a.step_v_head_size = 1;
      a.step_k_sl = 1;
      a.step_k_head_size = 1;
      if (t.dt == BTLA_DTYPE::F16) {
        // fp16 stable route supports GQA (head_num a positive multiple of heads_kv).
        a.head_num = 2;
        a.heads_kv = 1;
      } else {
        // bf16 non-stable route requires head_num == heads_kv.
        a.head_num = 1;
        a.heads_kv = 1;
      }
    } else {
      // Mixed route only validates PLAIN layouts + unsupported flags before the
      // ISA gate; head counts just need to be self-consistent.
      a.head_num = 1;
      a.heads_kv = 1;
    }
    return a;
  }

  // Dispatch `a` through the first-layer entry selected by `t.route`.
  static void dispatch(const Tuple& t, const attn_fwd_args_t& a) {
    if (t.route == Route::Mixed) {
      bestla_sdpa_forward(a, t.dt);
    } else {
      bestla_sdpa_forward_homogeneous(a, t.dt);
    }
  }

  static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
  }

  // Drive each tuple through both dispatch layers to the terminal pre-kernel gate
  // and assert the host-capability-conditioned outcome described in the header.
  static void check_dispatch_terminates_per_isa() {
    for (const auto& t : tuples()) {
      // fp16/bf16 payloads are 2 bytes; fp32 Q/dst are 4 bytes. Over-allocate.
      std::vector<uint8_t> q(4096, 0), k(4096, 0), v(4096, 0), dst(4096, 0);
      auto a = make_args(t, q, k, v, dst);
      const bool cap = host_has_isa(t);

      bool threw = false;
      try {
        dispatch(t, a);
      } catch (const std::runtime_error& e) {
        // std::runtime_error is the second-layer ISA gate (distinct hierarchy from
        // the std::invalid_argument used by the first-layer/route/threading gates).
        threw = true;
        const std::string msg = e.what();
        if (cap) {
          throw std::runtime_error(std::string("core-e2e ") + t.name +
                                   ": host has the required ISA but dispatch raised an ISA gate error: " + msg);
        }
        if (!contains(msg, t.isa)) {
          throw std::runtime_error(std::string("core-e2e ") + t.name +
                                   ": ISA gate message does not name the required extension: " + msg);
        }
        if (contains(msg, "route")) {
          throw std::runtime_error(std::string("core-e2e ") + t.name +
                                   ": expected the ISA gate, not a route-validation failure: " + msg);
        }
      } catch (const std::invalid_argument& e) {
        // invalid_argument here means both dispatch layers (incl. the ISA gate)
        // passed and the call reached the shared pre-kernel threading requirement.
        threw = true;
        const std::string msg = e.what();
        if (!cap) {
          throw std::runtime_error(std::string("core-e2e ") + t.name +
                                   ": host lacks the required ISA but dispatch passed the ISA gate: " + msg);
        }
        if (!contains(msg, "threading")) {
          throw std::runtime_error(std::string("core-e2e ") + t.name +
                                   ": expected the pre-kernel threading gate past the ISA gate: " + msg);
        }
      }
      if (!threw) {
        throw std::runtime_error(std::string("core-e2e ") + t.name +
                                 ": dispatch did not stop at a pre-kernel gate (threading was null)");
      }
      printf("[core_attention_e2e] %-38s -> %s\n", t.name,
             cap ? "ISA present: reached pre-kernel threading gate"
                 : "ISA absent: raised explicit second-layer ISA gate");
    }
  }

  // First-layer distinctness: the homogeneous entry is keyed on the shared operand
  // dtype and must reject fp32 (the mixed route's Q/dst type, never a homogeneous
  // operand tuple) up front -- proving the two families are dispatched separately
  // rather than folded into one generic branch.
  static void check_first_layer_distinct() {
    std::vector<uint8_t> q(256, 0), k(256, 0), v(256, 0), dst(256, 0);
    auto a = make_args(tuples()[2], q, k, v, dst);  // any homogeneous-shaped bundle
    bool rejected = false;
    try {
      bestla_sdpa_forward_homogeneous(a, BTLA_DTYPE::F32);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error("core-e2e: homogeneous entry did not reject the fp32 (mixed-only) operand dtype");
    }
    printf("[core_attention_e2e] first-layer dtype-tuple dispatch keeps mixed/homogeneous routes distinct\n");
  }

  static void run_all() {
    check_first_layer_distinct();
    check_dispatch_terminates_per_isa();
    printf("[core_attention_e2e] four core attention dtype tuples validated end-to-end through dispatch\n");
  }
};

}  // namespace ark::cpu
