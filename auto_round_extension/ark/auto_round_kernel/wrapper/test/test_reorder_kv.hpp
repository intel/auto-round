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

// Phase 4 Step 2: layout-correctness validation for the experimental raw->packed
// K/V reorder bridge (ark::cpu::reorder_k_to_packed / reorder_v_to_packed).
//
// These checks do not run any BestLA GEMM. Instead they independently recompute
// the byte address that the wired weight prologues read for each raw (seq,
// head_size) element and assert reorder_*_to_packed deposited that exact raw
// element there. Mirrors:
//   * fp16 K/V -> NTILE24_ROWPACK1: avx2::weight_cvt_fp16_fp32_n24
//       src = B + k_offset*24 + n_offset*ldb;  read[i*24 + j] = src[i*24 + j]
//   * bf16 K/V -> NTILE48_ROWPACK2: avx512f::weight_cvt_bf16_fp32_n48 (NTILE=48,
//       PACK_ROW=2): seq is the K dim packed ROWPACK over pairs, NTILE over the
//       N dim, ldb = padded-N stride.
// K is the QK weight (N=seq, K=head_size); V is the PV weight (N=head_size,
// K=seq). Both packed caches use these row-packed prologue addresses, so an
// index match here proves reorder feeds the kernel the values it consumes.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "ark/cpu/mha_dense.h"
#include "ark/cpu/mha_dense_wrapper.h"
#include "ark/cpu/sdpa.h"

namespace ark::cpu {

struct TestReorderKV {
  TestReorderKV() {
    run_all();
  }

  static int pad_up(int v, int p) { return ((v + p - 1) / p) * p; }

  // Expected packed index of K element (s, d) per the QK prologue addressing:
  //   tile=s/NTILE, sl_in=s%NTILE; kp=d/ROWPACK, rp_i=d%ROWPACK; hs_pad=pad(D,RP)
  //   idx = tile*hs_pad*NTILE + kp*NTILE*ROWPACK + sl_in*ROWPACK + rp_i
  static size_t expect_k_idx(int s, int d, int ntile, int rp, int head_dim) {
    const int hs_pad = pad_up(head_dim, rp);
    const int tile = s / ntile, sl_in = s % ntile;
    const int kp = d / rp, rp_i = d % rp;
    return size_t(tile) * hs_pad * ntile + size_t(kp) * ntile * rp + size_t(sl_in) * rp + rp_i;
  }

  // Expected packed index of V element (s, d) per the PV prologue addressing:
  //   NTILE over head_size, ROWPACK over seq; sl_pad=pad(S,RP)
  static size_t expect_v_idx(int s, int d, int ntile, int rp, int seq_len) {
    const int sl_pad = pad_up(seq_len, rp);
    const int tile = d / ntile, hs_in = d % ntile;
    const int kp = s / rp, rp_i = s % rp;
    return size_t(tile) * sl_pad * ntile + size_t(kp) * ntile * rp + size_t(hs_in) * rp + rp_i;
  }

  static void check_k(BTLA_DTYPE dt, int batch, int hkv, int sl, int hd, bool nhd) {
    auto sh = reorder_kv_shape(batch, hkv, sl, hd, dt);
    std::vector<uint16_t> raw(size_t(batch) * hkv * sl * hd);
    for (size_t i = 0; i < raw.size(); ++i) store_scalar(raw.data(), i, dt, float((i % 251) - 125) * 0.1f);
    // HND ([B,Hkv,S,D]) vs NHD ([B,S,Hkv,D]) strides over the raw plain layout.
    AttentionStrides st;
    st.dim = 1;
    st.seq = nhd ? hkv * hd : hd;
    st.head = nhd ? hd : sl * hd;
    st.batch = sl * hkv * hd;
    std::vector<uint16_t> packed(reorder_kv_cache_elems(sh, false));
    reorder_k_to_packed(packed.data(), raw.data(), sh, st, batch, hkv, sl, hd, dt);
    for (int b = 0; b < batch; ++b)
      for (int h = 0; h < hkv; ++h) {
        size_t base = (size_t(b) * hkv + h) * sh.k_head_elems;
        for (int s = 0; s < sl; ++s)
          for (int d = 0; d < hd; ++d) {
            float want = load_scalar(raw.data(), qko_offset(st, b, h, s, d), dt);
            float got = load_scalar(packed.data(), base + expect_k_idx(s, d, sh.ntile, sh.rowpack, hd), dt);
            if (got != want) throw std::runtime_error("K reorder mismatch");
          }
      }
  }

  static void check_v(BTLA_DTYPE dt, int batch, int hkv, int sl, int hd, bool nhd) {
    auto sh = reorder_kv_shape(batch, hkv, sl, hd, dt);
    std::vector<uint16_t> raw(size_t(batch) * hkv * sl * hd);
    for (size_t i = 0; i < raw.size(); ++i) store_scalar(raw.data(), i, dt, float((i % 241) - 120) * 0.1f);
    ValueStrides st;
    st.dim = 1;
    st.seq = nhd ? hkv * hd : hd;
    st.head = nhd ? hd : sl * hd;
    st.batch = sl * hkv * hd;
    std::vector<uint16_t> packed(reorder_kv_cache_elems(sh, true));
    reorder_v_to_packed(packed.data(), raw.data(), sh, st, batch, hkv, sl, hd, dt);
    for (int b = 0; b < batch; ++b)
      for (int h = 0; h < hkv; ++h) {
        size_t base = (size_t(b) * hkv + h) * sh.v_head_elems;
        for (int s = 0; s < sl; ++s)
          for (int d = 0; d < hd; ++d) {
            float want = load_scalar(raw.data(), value_offset(st, b, h, s, d), dt);
            float got = load_scalar(packed.data(), base + expect_v_idx(s, d, sh.ntile, sh.rowpack, sl), dt);
            if (got != want) throw std::runtime_error("V reorder mismatch");
          }
      }
  }

  static size_t qko_offset(const AttentionStrides& s, int b, int h, int sq, int d) {
    return size_t(b) * s.batch + size_t(h) * s.head + size_t(sq) * s.seq + size_t(d) * s.dim;
  }
  static size_t value_offset(const ValueStrides& s, int b, int h, int sq, int d) {
    return size_t(b) * s.batch + size_t(h) * s.head + size_t(sq) * s.seq + size_t(d) * s.dim;
  }

  void run_all() {
    int pass = 0;
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16})
      for (bool nhd : {false, true})
        // GQA: hkv smaller than heads_q; non-multiples of 24/48 and ROWPACK.
        for (auto sl : {24, 48, 50, 100}) {
          for (auto hd : {16, 17, 64}) {
            check_k(dt, 2, 2, sl, hd, nhd);
            check_v(dt, 2, 2, sl, hd, nhd);
            ++pass;
          }
        }
    printf("[reorder_kv] %d shape/dtype/layout cases passed\n", pass * 2);
  }
};

}  // namespace ark::cpu
