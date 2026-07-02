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

// Phase 4 Step 4: persistent packed K/V cache + in-place update validation.
// The persistent cache, sized for `capacity`, is filled incrementally
// ([0,start_pos) then [start_pos,append_len)) and must byte-match a one-shot
// reorder of the same final K/V sequence padded out to `capacity`.
struct TestPersistentPackedKV {
  TestPersistentPackedKV() { run_all(); }

  static size_t qko_offset(const AttentionStrides& s, int b, int h, int sq, int d) {
    return size_t(b) * s.batch + size_t(h) * s.head + size_t(sq) * s.seq + size_t(d) * s.dim;
  }
  static size_t value_offset(const ValueStrides& s, int b, int h, int sq, int d) {
    return size_t(b) * s.batch + size_t(h) * s.head + size_t(sq) * s.seq + size_t(d) * s.dim;
  }

  // Build HND/NHD raw strides over a [B,Hkv,cap,D] plain buffer.
  template <typename ST>
  static ST raw_strides(int hkv, int cap, int hd, bool nhd) {
    ST st;
    st.dim = 1;
    st.seq = nhd ? hkv * hd : hd;
    st.head = nhd ? hd : cap * hd;
    st.batch = cap * hkv * hd;
    return st;
  }

  static void check(BTLA_DTYPE dt, int batch, int hkv, int hd, int capacity, int start_pos, int append_len, bool nhd) {
    const int seq = start_pos + append_len;
    // Raw buffer sized for capacity; positions >= seq are zero (match reorder pad).
    std::vector<uint16_t> rawk(size_t(batch) * hkv * capacity * hd, 0);
    std::vector<uint16_t> rawv(size_t(batch) * hkv * capacity * hd, 0);
    auto ks = raw_strides<AttentionStrides>(hkv, capacity, hd, nhd);
    auto vs = raw_strides<ValueStrides>(hkv, capacity, hd, nhd);
    for (int b = 0; b < batch; ++b)
      for (int h = 0; h < hkv; ++h)
        for (int s = 0; s < seq; ++s)
          for (int d = 0; d < hd; ++d) {
            store_scalar(rawk.data(), qko_offset(ks, b, h, s, d), dt, float(((b + h + s + d) % 251) - 125) * 0.1f);
            store_scalar(rawv.data(), value_offset(vs, b, h, s, d), dt, float(((b * 3 + h + s + d) % 241) - 120) * 0.1f);
          }
    // One-shot reorder of the capacity-length sequence (zeros past seq).
    auto sh = packed_kv_cache_shape(batch, hkv, capacity, hd, dt);
    std::vector<uint16_t> ref_k(reorder_kv_cache_elems(sh, false));
    std::vector<uint16_t> ref_v(reorder_kv_cache_elems(sh, true));
    reorder_k_to_packed(ref_k.data(), rawk.data(), sh, ks, batch, hkv, capacity, hd, dt);
    reorder_v_to_packed(ref_v.data(), rawv.data(), sh, vs, batch, hkv, capacity, hd, dt);
    // Persistent: zero, append prefix [0,start_pos), then [start_pos,append_len).
    std::vector<uint16_t> cur_k(ref_k.size(), 0);
    std::vector<uint16_t> cur_v(ref_v.size(), 0);
    if (start_pos > 0) {
      update_packed_k_cache(cur_k.data(), rawk.data(), sh, ks, batch, hkv, start_pos, hd, 0, dt);
      update_packed_v_cache(cur_v.data(), rawv.data(), sh, vs, batch, hkv, start_pos, hd, 0, dt);
    }
    auto ks2 = raw_strides<AttentionStrides>(hkv, capacity, hd, nhd);  // append slice begins at row start_pos
    auto vs2 = raw_strides<ValueStrides>(hkv, capacity, hd, nhd);
    update_packed_k_cache(cur_k.data(), rawk.data() + size_t(start_pos) * ks2.seq, sh, ks2, batch, hkv, append_len, hd,
                          start_pos, dt);
    update_packed_v_cache(cur_v.data(), rawv.data() + size_t(start_pos) * vs2.seq, sh, vs2, batch, hkv, append_len, hd,
                          start_pos, dt);
    for (size_t i = 0; i < ref_k.size(); ++i)
      if (cur_k[i] != ref_k[i]) throw std::runtime_error("persistent K cache mismatch");
    for (size_t i = 0; i < ref_v.size(); ++i)
      if (cur_v[i] != ref_v[i]) throw std::runtime_error("persistent V cache mismatch");
  }

  void run_all() {
    int pass = 0;
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16})
      for (bool nhd : {false, true})
        for (int hd : {17, 64}) {
          check(dt, 2, 2, hd, 128, 0, 30, nhd);    // start_pos=0, append not tile-aligned
          check(dt, 2, 2, hd, 128, 24, 26, nhd);   // non-zero start_pos, capacity > seq
          check(dt, 2, 2, hd, 256, 48, 49, nhd);   // non-zero start_pos, odd append
          ++pass;
        }
    printf("[persistent_packed_kv] %d cases passed\n", pass);
  }
};

// Phase 4 Step 5: logical-vs-padded capacity, zero-fill, and packed-forward arg
// construction checks. Verifies update_packed_* reject writes past the logical
// capacity even when buffers are padded, that padded regions stay zero, and that
// bestla_sdpa_forward_packed validates dtype/layout/capacity before any GEMM.
struct TestPackedForwardSetup {
  TestPackedForwardSetup() { run_all(); }

  static void check_logical_capacity(BTLA_DTYPE dt, int cap, int hd) {
    auto sh = packed_kv_cache_shape(2, 2, cap, hd, dt);
    if (sh.logical_capacity != cap) throw std::runtime_error("logical_capacity not preserved");
    std::vector<uint16_t> k(reorder_kv_cache_elems(sh, false), 0), v(reorder_kv_cache_elems(sh, true), 0);
    AttentionStrides ks{hd, 1, cap * hd, cap * 2 * hd};
    ValueStrides vs{1, hd, cap * hd, cap * 2 * hd};
    std::vector<uint16_t> raw(size_t(2) * 2 * cap * hd, 0);
    // start_pos + append == capacity must be allowed.
    update_packed_k_cache(k.data(), raw.data(), sh, ks, 2, 2, cap, hd, 0, dt);
    update_packed_v_cache(v.data(), raw.data(), sh, vs, 2, 2, cap, hd, 0, dt);
    // start_pos + append > capacity must throw, even inside padded capacity.
    bool threw = false;
    try { update_packed_k_cache(k.data(), raw.data(), sh, ks, 2, 2, 1, hd, cap, dt); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) throw std::runtime_error("K overflow not rejected");
    threw = false;
    try { update_packed_v_cache(v.data(), raw.data(), sh, vs, 2, 2, 1, hd, cap, dt); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) throw std::runtime_error("V overflow not rejected");
  }

  static void check_padding_zero(BTLA_DTYPE dt, int cap, int hd) {
    auto sh = packed_kv_cache_shape(2, 2, cap, hd, dt);
    std::vector<uint16_t> k(reorder_kv_cache_elems(sh, false), 0xFFFF), v(reorder_kv_cache_elems(sh, true), 0xFFFF);
    clear_packed_k_cache(k.data(), sh, dt);
    clear_packed_v_cache(v.data(), sh, dt);
    std::vector<uint16_t> raw(size_t(2) * 2 * cap * hd, 0);
    AttentionStrides ks{hd, 1, cap * hd, cap * 2 * hd};
    ValueStrides vs{1, hd, cap * hd, cap * 2 * hd};
    for (size_t i = 0; i < raw.size(); ++i) store_scalar(raw.data(), i, dt, 1.0f);
    update_packed_k_cache(k.data(), raw.data(), sh, ks, 2, 2, 1, hd, 0, dt);  // append only 1 token
    update_packed_v_cache(v.data(), raw.data(), sh, vs, 2, 2, 1, hd, 0, dt);
    // Padded head_dim / tile / rowpack slots beyond the single token stay zero.
    int zeros = 0;
    for (size_t i = 0; i < k.size(); ++i) if (k[i] == 0) ++zeros;
    if (zeros == 0) throw std::runtime_error("padded K not zero");
    zeros = 0;
    for (size_t i = 0; i < v.size(); ++i) if (v[i] == 0) ++zeros;
    if (zeros == 0) throw std::runtime_error("padded V not zero");
  }

  static void check_forward_rejects() {
    auto sh = packed_kv_cache_shape(1, 1, 32, 64, BTLA_DTYPE::F16);
    std::vector<uint16_t> k(reorder_kv_cache_elems(sh, false), 0), v(reorder_kv_cache_elems(sh, true), 0);
    std::vector<float> q(64), dst(64);
    attn_fwd_args_t a{};
    a.Q = q.data(); a.K = k.data(); a.V = v.data(); a.dst = dst.data();
    a.batch_size = 1; a.head_num = 1; a.heads_kv = 1; a.head_size = 64; a.sl_q = 1; a.sl_kv = 16;
    a.Q_layout = ATTN_FWD_LAYOUT_PLAIN; a.dst_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.K_layout = sh.layout; a.V_layout = sh.layout;
    // Capacity overflow: sl_kv > logical_capacity must throw.
    a.sl_kv = 99;
    bool threw = false;
    try { bestla_sdpa_forward_packed(a, sh, BTLA_DTYPE::F16); } catch (const std::exception&) { threw = true; }
    if (!threw) throw std::runtime_error("forward capacity overflow not rejected");
    // Wrong dtype/layout pairing must throw.
    a.sl_kv = 16;
    threw = false;
    try { bestla_sdpa_forward_packed(a, sh, BTLA_DTYPE::BF16); } catch (const std::exception&) { threw = true; }
    if (!threw) throw std::runtime_error("forward dtype/layout mismatch not rejected");
  }

  void run_all() {
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16})
      for (int cap : {30, 50, 100}) {  // not divisible by NTILE/ROWPACK
        check_logical_capacity(dt, cap, 17);
        check_padding_zero(dt, cap, 17);
      }
    check_forward_rejects();
    printf("[packed_forward_setup] checks passed\n");
  }
};

// Phase 4.5 Step 5: pre-GEMM validation for the internal homogeneous SDPA
// dispatch (ark::cpu::bestla_sdpa_forward_homogeneous). Like TestPackedForwardSetup
// these checks never run a BestLA GEMM: they only exercise the argument-validation
// gates that fire before any ISA-specific kernel is reached, so they are
// deterministic on any CPU regardless of AVX512-FP16 / AMX-BF16 support.
struct TestHomogeneousForwardSetup {
  TestHomogeneousForwardSetup() { run_all(); }

  // Build a minimally-populated homogeneous arg bundle (Q==K==V==dst dtype).
  static attn_fwd_args_t make_args(std::vector<uint16_t>& q, std::vector<uint16_t>& k, std::vector<uint16_t>& v,
                                   std::vector<uint16_t>& dst, void* threading) {
    attn_fwd_args_t a{};
    a.Q = q.data();
    a.K = k.data();
    a.V = v.data();
    a.dst = dst.data();
    a.batch_size = 1;
    a.head_num = 1;
    a.heads_kv = 1;
    a.head_size = 8;
    a.sl_q = 1;
    a.sl_kv = 4;
    a.Q_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.K_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.V_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.dst_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.threading = threading;
    return a;
  }

  static void check_rejects() {
    std::vector<uint16_t> q(8, 0), k(32, 0), v(32, 0), dst(8, 0);
    // Null pointers must throw regardless of dtype.
    {
      auto a = make_args(q, k, v, dst, nullptr);
      a.Q = nullptr;
      bool threw = false;
      try { bestla_sdpa_forward_homogeneous(a, BTLA_DTYPE::F16); } catch (const std::exception&) { threw = true; }
      if (!threw) throw std::runtime_error("homogeneous null Q not rejected");
    }
    // Unsupported operand dtype (F32 is the mixed route's dst, never homogeneous
    // here) must throw before any ISA gate / GEMM.
    {
      auto a = make_args(q, k, v, dst, nullptr);
      bool threw = false;
      try { bestla_sdpa_forward_homogeneous(a, BTLA_DTYPE::F32); } catch (const std::exception&) { threw = true; }
      if (!threw) throw std::runtime_error("homogeneous unsupported dtype not rejected");
    }
    // Alibi/tanh/padding-right flags are rejected before the ISA gate / GEMM.
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      auto a = make_args(q, k, v, dst, nullptr);
      a.attn_flags = ATTN_FLAG_IS_ALIBI8;
      bool threw = false;
      try { bestla_sdpa_forward_homogeneous(a, dt); } catch (const std::exception&) { threw = true; }
      if (!threw) throw std::runtime_error("homogeneous unsupported flag not rejected");
    }
  }

  void run_all() {
    check_rejects();
    printf("[homogeneous_forward_setup] checks passed\n");
  }
};

}  // namespace ark::cpu
