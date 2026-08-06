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

// Layout-correctness validation for the NS-parity raw->packed K/V reorder
// bridge (ark::cpu::reorder_k_to_packed / reorder_v_to_packed).
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ark/cpu/mha_dense.h"
#include "ark/cpu/mha_dense_wrapper.h"
#include "ark/cpu/sdpa.h"

namespace ark::cpu {

struct TestReorderKV {
  TestReorderKV() {
    run_all();
  }

  // Expected packed index of K element (s, d) per the QK prologue addressing:
  //   tile=s/NTILE, sl_in=s%NTILE; kp=d/ROWPACK, rp_i=d%ROWPACK
  //   idx = tile*hs_pad*NTILE + kp*NTILE*ROWPACK + sl_in*ROWPACK + rp_i
  static size_t expect_k_idx(int s, int d, int ntile, int rp, int hs_pad) {
    const int tile = s / ntile, sl_in = s % ntile;
    const int kp = d / rp, rp_i = d % rp;
    return size_t(tile) * hs_pad * ntile + size_t(kp) * ntile * rp + size_t(sl_in) * rp + rp_i;
  }

  // Expected packed index of V element (s, d) per the PV prologue addressing:
  //   NTILE over head_size, ROWPACK over seq.
  static size_t expect_v_idx(int s, int d, int ntile, int rp, int sl_pad) {
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
    reorder_k_to_packed(packed.data(), raw.data(), sh, st);
    for (int b = 0; b < batch; ++b)
      for (int h = 0; h < hkv; ++h) {
        size_t base = (size_t(b) * hkv + h) * sh.k_head_elems;
        for (int s = 0; s < sl; ++s)
          for (int d = 0; d < hd; ++d) {
            float want = load_scalar(raw.data(), qko_offset(st, b, h, s, d), dt);
            float got = load_scalar(packed.data(),
                                    base + expect_k_idx(s, d, sh.ntile, sh.rowpack, sh.k_head_size_pad), dt);
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
    reorder_v_to_packed(packed.data(), raw.data(), sh, st);
    for (int b = 0; b < batch; ++b)
      for (int h = 0; h < hkv; ++h) {
        size_t base = (size_t(b) * hkv + h) * sh.v_head_elems;
        for (int s = 0; s < sl; ++s)
          for (int d = 0; d < hd; ++d) {
            float want = load_scalar(raw.data(), value_offset(st, b, h, s, d), dt);
            float got = load_scalar(packed.data(), base + expect_v_idx(s, d, sh.ntile, sh.rowpack, sh.v_seq_pad), dt);
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

// Persistent packed K/V cache + in-place update validation.
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
    reorder_k_to_packed(ref_k.data(), rawk.data(), sh, ks);
    reorder_v_to_packed(ref_v.data(), rawv.data(), sh, vs);
    // Persistent: zero, append prefix [0,start_pos), then [start_pos,append_len).
    std::vector<uint16_t> cur_k(ref_k.size(), 0);
    std::vector<uint16_t> cur_v(ref_v.size(), 0);
    if (start_pos > 0) {
      update_packed_k_cache(cur_k.data(), rawk.data(), sh, ks, start_pos, 0);
      update_packed_v_cache(cur_v.data(), rawv.data(), sh, vs, start_pos, 0);
    }
    auto ks2 = raw_strides<AttentionStrides>(hkv, capacity, hd, nhd);  // append slice begins at row start_pos
    auto vs2 = raw_strides<ValueStrides>(hkv, capacity, hd, nhd);
    update_packed_k_cache(cur_k.data(), rawk.data() + size_t(start_pos) * ks2.seq, sh, ks2, append_len, start_pos);
    update_packed_v_cache(cur_v.data(), rawv.data() + size_t(start_pos) * vs2.seq, sh, vs2, append_len, start_pos);
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

// Logical-vs-padded capacity, zero-fill, and packed-forward arg construction
// checks. Verifies update_packed_* reject writes past the logical capacity even
// when buffers are padded, that padded regions stay zero, and that
// bestla_sdpa_forward_packed validates dtype/layout/capacity before any GEMM.
struct TestPackedForwardSetup {
  TestPackedForwardSetup() { run_all(); }

  static void check_logical_capacity(BTLA_DTYPE dt, int cap, int hd) {
    auto sh = packed_kv_cache_shape(2, 2, cap, hd, dt);
    if (sh.logical_capacity != cap) throw std::runtime_error("logical_capacity not preserved");
    if (sh.batch_size != 2 || sh.heads_kv != 2 || sh.dtype != dt) {
      throw std::runtime_error("packed descriptor metadata not preserved");
    }
    std::vector<uint16_t> k(reorder_kv_cache_elems(sh, false), 0), v(reorder_kv_cache_elems(sh, true), 0);
    AttentionStrides ks{hd, 1, cap * hd, cap * 2 * hd};
    ValueStrides vs{1, hd, cap * hd, cap * 2 * hd};
    std::vector<uint16_t> raw(size_t(2) * 2 * cap * hd, 0);
    // start_pos + append == capacity must be allowed.
    update_packed_k_cache(k.data(), raw.data(), sh, ks, cap, 0);
    update_packed_v_cache(v.data(), raw.data(), sh, vs, cap, 0);
    // start_pos + append > capacity must throw, even inside padded capacity.
    bool threw = false;
    try { update_packed_k_cache(k.data(), raw.data(), sh, ks, 1, cap); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) throw std::runtime_error("K overflow not rejected");
    threw = false;
    try { update_packed_v_cache(v.data(), raw.data(), sh, vs, 1, cap); }
    catch (const std::invalid_argument&) { threw = true; }
    if (!threw) throw std::runtime_error("V overflow not rejected");
  }

  static void check_padding_zero(BTLA_DTYPE dt, int cap, int hd) {
    auto sh = packed_kv_cache_shape(2, 2, cap, hd, dt);
    std::vector<uint16_t> k(reorder_kv_cache_elems(sh, false), 0xFFFF), v(reorder_kv_cache_elems(sh, true), 0xFFFF);
    clear_packed_k_cache(k.data(), sh);
    clear_packed_v_cache(v.data(), sh);
    std::vector<uint16_t> raw(size_t(2) * 2 * cap * hd, 0);
    AttentionStrides ks{hd, 1, cap * hd, cap * 2 * hd};
    ValueStrides vs{1, hd, cap * hd, cap * 2 * hd};
    for (size_t i = 0; i < raw.size(); ++i) store_scalar(raw.data(), i, dt, 1.0f);
    update_packed_k_cache(k.data(), raw.data(), sh, ks, 1, 0);  // append only 1 token
    update_packed_v_cache(v.data(), raw.data(), sh, vs, 1, 0);
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
    a.K_layout = sh.k_layout; a.V_layout = sh.v_layout;
    // Capacity overflow: sl_kv > logical_capacity must throw.
    a.sl_kv = 99;
    bool threw = false;
    try { bestla_sdpa_forward_packed(a, sh); } catch (const std::exception&) { threw = true; }
    if (!threw) throw std::runtime_error("forward capacity overflow not rejected");
    // Wrong dtype/layout pairing must throw.
    a.sl_kv = 16;
    threw = false;
    auto bad = sh;
    bad.dtype = BTLA_DTYPE::BF16;
    try { bestla_sdpa_forward_packed(a, bad); } catch (const std::exception&) { threw = true; }
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

// Pre-GEMM validation for the internal homogeneous SDPA dispatch
// (ark::cpu::bestla_sdpa_forward_homogeneous, routes 3 fp16×4 and 4 bf16×4).
// Like TestPackedForwardSetup these checks never run a BestLA GEMM: they only
// exercise the argument-validation gates that fire before any ISA-specific
// kernel is reached, so they are deterministic on any CPU regardless of
// AVX512-FP16 / AMX-BF16 support.
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

  // Build an arg bundle that satisfies the full layout/stride/head-count contract
  // of the requested homogeneous route, so every std::invalid_argument route
  // guard passes and only the ISA/threading gates remain. threading stays null:
  // route validation runs before the threading/ISA gates, so these args exercise
  // the accept path of the route validators on any CPU.
  static attn_fwd_args_t make_route_valid_args(std::vector<uint16_t>& q, std::vector<uint16_t>& k,
                                               std::vector<uint16_t>& v, std::vector<uint16_t>& dst, BTLA_DTYPE dt) {
    auto a = make_args(q, k, v, dst, nullptr);
    // Both routes accept PLAIN K/V with contiguous V head-size and K seq strides.
    a.step_v_head_size = 1;
    a.step_k_sl = 1;
    a.step_k_head_size = 1;
    if (dt == BTLA_DTYPE::F16) {
      // fp16 stable route supports GQA (head_num a multiple of heads_kv).
      a.head_num = 2;
      a.heads_kv = 1;
    } else {
      // bf16 non-stable route requires head_num == heads_kv.
      a.head_num = 1;
      a.heads_kv = 1;
    }
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
    // alibi/tanh are U for BOTH homogeneous routes and rejected PER ROUTE.
    // Build route-valid args so the rejection comes from the route validator's
    // alibi/tanh guard (message contains "route"), not an earlier layout/stride check.
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      for (auto flag : {ATTN_FLAG_IS_ALIBI8, ATTN_FLAG_IS_TANH30}) {
        std::vector<uint16_t> rq(64, 0), rk(64, 0), rv(64, 0), rd(64, 0);
        auto a = make_route_valid_args(rq, rk, rv, rd, dt);
        a.attn_flags = flag;
        if (!route_validation_rejects(a, dt))
          throw std::runtime_error("homogeneous alibi/tanh not rejected by the route validator");
      }
    }
    // prefer_fp32 is unsupported for BOTH homogeneous routes and is rejected per route.
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      std::vector<uint16_t> rq(64, 0), rk(64, 0), rv(64, 0), rd(64, 0);
      auto a = make_route_valid_args(rq, rk, rv, rd, dt);
      a.attn_flags = ATTN_FLAG_PREFER_FP32;
      if (!route_validation_rejects(a, dt))
        throw std::runtime_error("homogeneous prefer_fp32 not rejected by the route validator");
    }
  }

  // True if calling the homogeneous entry with `a`/`dt` throws an exception whose
  // message names a route-validation failure (all route guards contain "route").
  static bool route_validation_rejects(const attn_fwd_args_t& a, BTLA_DTYPE dt) {
    try {
      bestla_sdpa_forward_homogeneous(a, dt);
    } catch (const std::exception& e) {
      return std::string(e.what()).find("route") != std::string::npos;
    }
    return false;
  }

  // fp16 stable route (mha_stable_interface_t) contract: PLAIN Q/dst, K/V PLAIN
  // or NTILE24_ROWPACK1, GQA head_num multiple of heads_kv, PLAIN K/V strides.
  static void check_fp16_route_rejects() {
    std::vector<uint16_t> q(64, 0), k(64, 0), v(64, 0), dst(64, 0);
    // Non-PLAIN Q layout is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::F16);
      a.Q_layout = ATTN_FWD_LAYOUT_NTILE24_ROWPACK1;
      if (!route_validation_rejects(a, BTLA_DTYPE::F16)) throw std::runtime_error("fp16 non-PLAIN Q not rejected");
    }
    // A K layout that belongs to the bf16 packing (NTILE48_ROWPACK2) is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::F16);
      a.K_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      if (!route_validation_rejects(a, BTLA_DTYPE::F16)) throw std::runtime_error("fp16 wrong K layout not rejected");
    }
    // head_num not a whole multiple of heads_kv is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::F16);
      a.head_num = 3;
      a.heads_kv = 2;
      if (!route_validation_rejects(a, BTLA_DTYPE::F16)) throw std::runtime_error("fp16 bad GQA not rejected");
    }
    // PLAIN K with a non-contiguous V head-size stride is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::F16);
      a.step_v_head_size = 4;
      if (!route_validation_rejects(a, BTLA_DTYPE::F16)) throw std::runtime_error("fp16 bad step_v not rejected");
    }
    // PLAIN V with a non-contiguous K seq stride is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::F16);
      a.step_k_sl = 8;
      if (!route_validation_rejects(a, BTLA_DTYPE::F16)) throw std::runtime_error("fp16 bad step_k not rejected");
    }
  }

  // bf16 non-stable route (mha_interface_t) contract: all-PLAIN, no GQA
  // (head_num == heads_kv), contiguous V head-size stride, contiguous K stride.
  static void check_bf16_route_rejects() {
    std::vector<uint16_t> q(64, 0), k(64, 0), v(64, 0), dst(64, 0);
    // Any non-PLAIN layout is rejected (the non-stable path takes no packed K/V).
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::BF16);
      a.K_layout = ATTN_FWD_LAYOUT_NTILE48_ROWPACK2;
      if (!route_validation_rejects(a, BTLA_DTYPE::BF16)) throw std::runtime_error("bf16 non-PLAIN K not rejected");
    }
    // GQA (head_num != heads_kv) is rejected -- the non-stable path has no GQA.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::BF16);
      a.head_num = 2;
      a.heads_kv = 1;
      if (!route_validation_rejects(a, BTLA_DTYPE::BF16)) throw std::runtime_error("bf16 GQA not rejected");
    }
    // Non-contiguous V head-size stride is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::BF16);
      a.step_v_head_size = 4;
      if (!route_validation_rejects(a, BTLA_DTYPE::BF16)) throw std::runtime_error("bf16 bad step_v not rejected");
    }
    // Neither K stride contiguous is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::BF16);
      a.step_k_head_size = 8;
      a.step_k_sl = 8;
      if (!route_validation_rejects(a, BTLA_DTYPE::BF16)) throw std::runtime_error("bf16 bad K stride not rejected");
    }
    // Causal mask with sl_q > sl_kv is rejected.
    {
      auto a = make_route_valid_args(q, k, v, dst, BTLA_DTYPE::BF16);
      a.attn_flags = ATTN_FLAG_IS_CAUSAL;
      a.sl_q = 8;
      a.sl_kv = 4;
      if (!route_validation_rejects(a, BTLA_DTYPE::BF16)) throw std::runtime_error("bf16 bad causal shape not rejected");
    }
  }

  // Route-valid args must pass every std::invalid_argument route guard; the only
  // failure left is the ISA capability gate (std::runtime_error) on a CPU that
  // lacks AVX512-FP16 / AMX-BF16, or the threading gate -- never a "route" error.
  static void check_valid_routes_pass_validation() {
    std::vector<uint16_t> q(64, 0), k(64, 0), v(64, 0), dst(64, 0);
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      auto a = make_route_valid_args(q, k, v, dst, dt);
      if (route_validation_rejects(a, dt)) throw std::runtime_error("valid homogeneous route wrongly rejected");
    }
  }

  void run_all() {
    check_rejects();
    check_fp16_route_rejects();
    check_bf16_route_rejects();
    check_valid_routes_pass_validation();
    printf("[homogeneous_forward_setup] checks passed\n");
  }
};

// Padding-right plumbing + validation for routes 1/2 (ark::cpu::bestla_sdpa_forward,
// f32/f16 and f32/bf16). Both mixed routes compose the fp32-score stable interface
// whose ScaleTrackMax epilogue implements padding_type==2 (see the AVX2/AVX512F
// scale_track_max_fp32_fp32 paths), so padding-right is S: the entry forwards
// n_padding and validates the boundary. Like the setups above, every case here is
// decided by the argument-validation gates that fire BEFORE the ISA/threading gates
// and the raw->packed reorder, so the rejection cases are deterministic on any CPU.
// The accept case asserts padding-right with a valid boundary is no longer treated as
// an unsupported/invalid flag — it passes the padding gate and stops at the same
// pre-kernel ISA/threading gate as a plain call (never a "padding-right" error).
struct TestMixedPaddingRight {
  TestMixedPaddingRight() { run_all(); }

  // Minimal PLAIN, GQA-consistent mixed arg bundle. threading stays null: the
  // padding/causal/GQA gates run before the ISA/threading gates, so this exercises
  // the accept path of the padding validator on any CPU. Buffers are over-sized;
  // only their non-null-ness matters before the kernel runs (fp32 Q/dst are 4B,
  // fp16/bf16 K/V are 2B).
  static attn_fwd_args_t make_args(std::vector<uint8_t>& q, std::vector<uint8_t>& k, std::vector<uint8_t>& v,
                                   std::vector<uint8_t>& dst) {
    attn_fwd_args_t a{};
    a.Q = q.data();
    a.K = k.data();
    a.V = v.data();
    a.dst = dst.data();
    a.batch_size = 1;
    a.head_num = 1;
    a.heads_kv = 1;
    a.head_size = 8;
    a.sl_q = 4;
    a.sl_kv = 8;
    a.Q_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.K_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.V_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.dst_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.threading = nullptr;
    return a;
  }

  // True iff bestla_sdpa_forward rejects `a` with a padding-right-specific
  // std::invalid_argument. An ISA gate (std::runtime_error) or the shared
  // threading gate (a non-padding std::invalid_argument) means the padding gate
  // PASSED, so both count as "not a padding rejection".
  static bool padding_rejected(const attn_fwd_args_t& a, BTLA_DTYPE dt) {
    try {
      bestla_sdpa_forward(a, dt);
    } catch (const std::invalid_argument& e) {
      return std::string(e.what()).find("padding-right") != std::string::npos;
    } catch (const std::exception&) {
      return false;  // ISA/threading gate reached: padding gate already passed
    }
    return false;
  }

  static void run_all() {
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      std::vector<uint8_t> q(4096, 0), k(4096, 0), v(4096, 0), dst(4096, 0);
      // Accept: a valid boundary (0 < n_padding <= sl_kv) is not rejected.
      {
        auto a = make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_PADDING_RIGHT;
        a.n_padding_scalar = a.sl_kv / 2;
        if (padding_rejected(a, dt))
          throw std::runtime_error("mixed padding-right with valid n_padding wrongly rejected");
      }
      // Reject: n_padding <= 0 (no valid K/V positions).
      {
        auto a = make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_PADDING_RIGHT;
        a.n_padding_scalar = 0;
        if (!padding_rejected(a, dt)) throw std::runtime_error("mixed padding-right n_padding<=0 not rejected");
      }
      // Reject: n_padding > sl_kv (boundary past the K/V sequence).
      {
        auto a = make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_PADDING_RIGHT;
        a.n_padding_scalar = a.sl_kv + 1;
        if (!padding_rejected(a, dt)) throw std::runtime_error("mixed padding-right n_padding>sl_kv not rejected");
      }
      // Reject: padding-right combined with causal (mutually exclusive -- the stable
      // epilogue applies a single padding_type per call).
      {
        auto a = make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_PADDING_RIGHT | ATTN_FLAG_IS_CAUSAL;
        a.n_padding_scalar = a.sl_kv / 2;
        if (!padding_rejected(a, dt)) throw std::runtime_error("mixed padding-right + causal not rejected");
      }
      // Per-batch pointer semantics: valid entries for every batch must pass.
      {
        auto a = make_args(q, k, v, dst);
        a.batch_size = 2;
        int per_batch[] = {a.sl_kv / 2, a.sl_kv - 1};
        a.attn_flags = ATTN_FLAG_PADDING_RIGHT;
        a.n_padding = per_batch;
        if (padding_rejected(a, dt))
          throw std::runtime_error("mixed padding-right per-batch n_padding wrongly rejected");
      }
      // Per-batch pointer semantics: one invalid entry must reject the whole call.
      {
        auto a = make_args(q, k, v, dst);
        a.batch_size = 2;
        int per_batch[] = {a.sl_kv / 2, a.sl_kv + 1};
        a.attn_flags = ATTN_FLAG_PADDING_RIGHT;
        a.n_padding = per_batch;
        if (!padding_rejected(a, dt))
          throw std::runtime_error("mixed padding-right invalid per-batch n_padding not rejected");
      }
    }
    // Homogeneous routes 3/4 stay U for padding-right (route 3 fp16-score
    // ScaleTrackMax asserts padding_type != 2; route 4 has no padding path). Use a
    // route-valid homogeneous bundle so the rejection comes from the flag gate, not
    // a layout/stride failure.
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      std::vector<uint16_t> hq(64, 0), hk(64, 0), hv(64, 0), hd(64, 0);
      auto a = TestHomogeneousForwardSetup::make_route_valid_args(hq, hk, hv, hd, dt);
      a.attn_flags = ATTN_FLAG_PADDING_RIGHT;
      a.n_padding_scalar = 2;
      bool threw = false;
      try {
        bestla_sdpa_forward_homogeneous(a, dt);
      } catch (const std::exception&) {
        threw = true;
      }
      if (!threw) throw std::runtime_error("homogeneous padding-right not rejected");
    }
    printf("[mixed_padding_right] checks passed\n");
  }
};

// ---------------------------------------------------------------------------
// Alibi/tanh wiring + per-route classification for routes 1/2 and 3/4.
// Both mixed routes (ark::cpu::bestla_sdpa_forward, route 1 f32/f16 and route 2
// f32/bf16) compose fp32-score cores whose ScaleTrackMax epilogue implements the
// alibi slope and the tanh scale (the templated scale_track_max_fp32_fp32<HAS_ALIBI,
// HAS_TANH> AVX2/AVX512F kernels), driven entirely by the ALIBI8/TANH30 flags that
// make_typed_attn_args already forwards, so both features are S: the entry no longer
// rejects them and they flow through to the kernel. Each case here is decided before
// the ISA/threading gates, so acceptance is deterministic on any CPU (a valid alibi/
// tanh call stops at the same pre-kernel ISA/threading gate as a plain call, never an
// "alibi"/"tanh" rejection). The two homogeneous routes (3 fp16-score, 4 non-stable)
// stay U and reject alibi/tanh in their per-route validators.
struct TestMixedAlibiTanh {
  TestMixedAlibiTanh() { run_all(); }

  // True iff bestla_sdpa_forward rejects `a`/`dt` with an alibi/tanh-specific
  // std::invalid_argument. Any other exception (the ISA std::runtime_error gate or
  // the shared threading std::invalid_argument gate) means the alibi/tanh flags were
  // ACCEPTED and the call simply stopped at a later pre-kernel gate.
  static bool alibi_tanh_rejected(const attn_fwd_args_t& a, BTLA_DTYPE dt) {
    try {
      bestla_sdpa_forward(a, dt);
    } catch (const std::invalid_argument& e) {
      const std::string msg = e.what();
      return msg.find("alibi") != std::string::npos || msg.find("tanh") != std::string::npos;
    } catch (const std::exception&) {
      return false;  // ISA/threading gate reached: alibi/tanh gate already passed
    }
    return false;
  }

  static void run_all() {
    // Mixed routes 1/2: alibi, tanh, both, and alibi+causal are all accepted (S).
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      std::vector<uint8_t> q(4096, 0), k(4096, 0), v(4096, 0), dst(4096, 0);
      {
        auto a = TestMixedPaddingRight::make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_IS_ALIBI8;
        if (alibi_tanh_rejected(a, dt)) throw std::runtime_error("mixed alibi wrongly rejected");
      }
      {
        auto a = TestMixedPaddingRight::make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_IS_TANH30;
        if (alibi_tanh_rejected(a, dt)) throw std::runtime_error("mixed tanh wrongly rejected");
      }
      {
        auto a = TestMixedPaddingRight::make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_IS_ALIBI8 | ATTN_FLAG_IS_TANH30;
        if (alibi_tanh_rejected(a, dt)) throw std::runtime_error("mixed alibi+tanh wrongly rejected");
      }
      {
        // alibi composes with causal (per-head slope + sl_q<=sl_kv mask).
        auto a = TestMixedPaddingRight::make_args(q, k, v, dst);
        a.attn_flags = ATTN_FLAG_IS_ALIBI8 | ATTN_FLAG_IS_CAUSAL;
        if (alibi_tanh_rejected(a, dt)) throw std::runtime_error("mixed alibi+causal wrongly rejected");
      }
    }
    // Homogeneous routes 3/4 stay U for alibi/tanh; the rejection is per route (its
    // message names the route). Use route-valid bundles so the reject comes from the
    // alibi/tanh guard, not a layout/stride failure.
    for (auto dt : {BTLA_DTYPE::F16, BTLA_DTYPE::BF16}) {
      for (auto flag : {ATTN_FLAG_IS_ALIBI8, ATTN_FLAG_IS_TANH30}) {
        std::vector<uint16_t> hq(64, 0), hk(64, 0), hv(64, 0), hd(64, 0);
        auto a = TestHomogeneousForwardSetup::make_route_valid_args(hq, hk, hv, hd, dt);
        a.attn_flags = flag;
        if (!TestHomogeneousForwardSetup::route_validation_rejects(a, dt))
          throw std::runtime_error("homogeneous alibi/tanh not rejected by the route validator");
      }
    }
    printf("[mixed_alibi_tanh] checks passed\n");
  }
};

// ---------------------------------------------------------------------------
// Numerical validation for mixed-route features (alibi, padding-right, tanh).
// Each test builds a small attention problem, runs it through
// bestla_sdpa_forward, and compares against a scalar fp32 reference.
//
// Test dimensions: B=1, Hq=4, Hkv=2 (GQA 2×), Sq=4, Sk=8, D=32.
//
// ISA gates (mirroring bestla_sdpa_forward's own gates):
//   F16 K/V  (route 1) -> AVX2:    skip if cpu->AVX2() == false
//   BF16 K/V (route 2) -> AVX512F: skip if cpu->AVX512F() == false
//
// Note on tanh: the AVX2 (F16) kernel instantiates HAS_TANH=true but does NOT
// apply the tanh nonlinearity (the if constexpr (HAS_TANH) epilogue block only
// exists in the AVX512F specialisation). Testing tanh on the F16 route would
// just validate the reduced QK scale (QK_scale/30), which is not meaningful.
// Therefore tanh is tested on BF16 (AVX512F) only, where the full
// 30*tanh(score * QK_scale/30) path is executed.
// ---------------------------------------------------------------------------
struct TestMixedNumericalFeatures {
  TestMixedNumericalFeatures() { run_all(); }

  // Compute the ALiBi slope for query head h in a model with head_num query
  // heads. Mirrors mha_dense_wrapper.h lines 1027-1066 exactly (k_offset=0).
  static float alibi_slope_for_head(int h, int head_num) {
    const int n_log2 = 1 << int(std::floor(std::log2f(float(head_num))));
    const float m0 = std::pow(2.f, -8.f / float(n_log2));
    const float m1 = std::pow(2.f, -4.f / float(n_log2));
    return (h < n_log2) ? std::pow(m0, float(h + 1)) : std::pow(m1, float(2 * (h - n_log2) + 1));
  }

  // Scalar fp32 attention reference. Layout: Q[Hq*Sq*D], K_rt[Hkv*Sk*D],
  // V_rt[Hkv*Sk*D] (K/V have been round-tripped through fp16/bf16 to match the
  // dtype error in the kernel path), dst[Hq*Sq*D].
  //
  // use_tanh  : scores = 30*tanh(dot * QK_scale/30) before alibi+softmax.
  // slopes    : non-null -> add alibi_slope[hq]*k to each score.
  // n_valid   : key positions [n_valid, Sk) are masked (-inf) before softmax.
  static void scalar_attn_ref(int Hq, int Hkv, int Sq, int Sk, int D, float qk_scale, bool use_tanh, int n_valid,
                               const float* Q, const float* K_rt, const float* V_rt, const float* slopes,
                               float* dst) {
    const float inner_scale = use_tanh ? qk_scale / 30.f : qk_scale;
    const int gqa_ratio = Hq / Hkv;
    for (int hq = 0; hq < Hq; ++hq) {
      const int hkv = hq / gqa_ratio;
      const float slope = slopes ? slopes[hq] : 0.f;
      for (int q = 0; q < Sq; ++q) {
        const float* Qrow = Q + (hq * Sq + q) * D;
        // Compute raw scores and track max over valid positions.
        std::vector<float> scores(Sk, -std::numeric_limits<float>::infinity());
        float max_s = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < n_valid; ++k) {
          const float* Krow = K_rt + (hkv * Sk + k) * D;
          float dot = 0.f;
          for (int d = 0; d < D; ++d) dot += Qrow[d] * Krow[d];
          float s = dot * inner_scale;
          if (use_tanh) s = 30.f * std::tanh(s);
          s += slope * float(k);
          scores[k] = s;
          max_s = std::max(max_s, s);
        }
        // Softmax over valid positions only.
        float sum_exp = 0.f;
        for (int k = 0; k < n_valid; ++k) {
          scores[k] = std::exp(scores[k] - max_s);
          sum_exp += scores[k];
        }
        for (int k = 0; k < n_valid; ++k) scores[k] /= sum_exp;
        // Weighted sum of V.
        float* dstrow = dst + (hq * Sq + q) * D;
        for (int d = 0; d < D; ++d) {
          float acc = 0.f;
          for (int k = 0; k < n_valid; ++k) acc += scores[k] * V_rt[(hkv * Sk + k) * D + d];
          dstrow[d] = acc;
        }
      }
    }
  }

  // Run one numerical check. extra_flags is OR-ed into attn_flags (caller
  // provides the feature flag, e.g. ATTN_FLAG_IS_ALIBI8). n_valid_kv is the
  // n_padding value when ATTN_FLAG_PADDING_RIGHT is set, else unused (pass Sk).
  // Returns true if the check ran; false if ISA was unavailable (skip).
  static bool run_check(BTLA_DTYPE kv_dtype, uint32_t extra_flags, int n_valid_kv, const char* tag) {
    auto* cpu = bestla::device::CpuDevice::getInstance();
    if (kv_dtype == BTLA_DTYPE::F16 && !cpu->AVX2()) {
      printf("[num_mixed] %s: SKIP (no AVX2)\n", tag);
      return false;
    }
    if (kv_dtype == BTLA_DTYPE::BF16 && !cpu->AVX512F()) {
      printf("[num_mixed] %s: SKIP (no AVX512F)\n", tag);
      return false;
    }

    static constexpr int B = 1, Hq = 4, Hkv = 2, Sq = 4, Sk = 8, D = 32;
    const float qk_scale = 1.f / std::sqrt(float(D));

    // Deterministic Q/K/V with values in [-1, 1].
    std::vector<float> Q_f32(Hq * Sq * D), K_f32(Hkv * Sk * D), V_f32(Hkv * Sk * D);
    for (int i = 0; i < static_cast<int>(Q_f32.size()); ++i) Q_f32[i] = 0.1f * float((i * 17 + 3) % 20 - 10);
    for (int i = 0; i < static_cast<int>(K_f32.size()); ++i) K_f32[i] = 0.1f * float((i * 13 + 7) % 20 - 10);
    for (int i = 0; i < static_cast<int>(V_f32.size()); ++i) V_f32[i] = 0.1f * float((i * 11 + 5) % 20 - 10);

    // Convert K/V to the target dtype and back to fp32 (dtype round-trip).
    std::vector<uint16_t> K_kv(Hkv * Sk * D), V_kv(Hkv * Sk * D);
    std::vector<float> K_rt(Hkv * Sk * D), V_rt(Hkv * Sk * D);
    for (int i = 0; i < Hkv * Sk * D; ++i) {
      if (kv_dtype == BTLA_DTYPE::F16) {
        const bestla::utils::fp16 kf(K_f32[i]), vf(V_f32[i]);
        K_kv[i] = kf.x;
        V_kv[i] = vf.x;
        K_rt[i] = float(kf);
        V_rt[i] = float(vf);
      } else {
        const auto kb = bestla::utils::cast<float, bestla::utils::bf16>(K_f32[i]);
        const auto vb = bestla::utils::cast<float, bestla::utils::bf16>(V_f32[i]);
        K_kv[i] = kb.x;
        V_kv[i] = vb.x;
        K_rt[i] = kb.tofloat();
        V_rt[i] = vb.tofloat();
      }
    }

    const bool use_alibi = (extra_flags & ATTN_FLAG_IS_ALIBI8) != 0;
    const bool use_tanh = (extra_flags & ATTN_FLAG_IS_TANH30) != 0;
    const bool use_padding = (extra_flags & ATTN_FLAG_PADDING_RIGHT) != 0;
    const int n_valid = use_padding ? n_valid_kv : Sk;

    // Per-head alibi slopes (matched exactly to mha_dense_wrapper.h formula).
    std::vector<float> slopes;
    if (use_alibi) {
      slopes.resize(Hq);
      for (int h = 0; h < Hq; ++h) slopes[h] = alibi_slope_for_head(h, Hq);
    }

    // Scalar reference output.
    std::vector<float> ref_dst(Hq * Sq * D);
    scalar_attn_ref(Hq, Hkv, Sq, Sk, D, qk_scale, use_tanh, n_valid, Q_f32.data(), K_rt.data(), V_rt.data(),
                    use_alibi ? slopes.data() : nullptr, ref_dst.data());

    // Kernel output via bestla_sdpa_forward (handles raw->packed reorder internally).
    std::vector<float> out_dst(Hq * Sq * D, 0.f);
    bestla::parallel::SingleThread sth;

    attn_fwd_args_t a{};
    a.Q = Q_f32.data();
    a.K = K_kv.data();
    a.V = V_kv.data();
    a.dst = out_dst.data();
    a.QK_scale = qk_scale;
    a.attn_flags = extra_flags;
    a.batch_size = B;
    a.head_num = Hq;
    a.heads_kv = Hkv;
    a.head_size = D;
    a.sl_q = Sq;
    a.sl_kv = Sk;
    a.n_padding_scalar = use_padding ? n_valid_kv : 0;
    a.Q_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.K_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.V_layout = ATTN_FWD_LAYOUT_PLAIN;
    a.dst_layout = ATTN_FWD_LAYOUT_PLAIN;
    // HND strides: [Head × Seq × Dim], contiguous.
    a.step_q_bs = Hq * Sq * D;
    a.step_q_head_num = Sq * D;
    a.step_q_sl = D;
    a.step_k_bs = Hkv * Sk * D;
    a.step_k_head_num = Sk * D;
    a.step_k_sl = D;
    a.step_k_head_size = 1;
    a.step_v_bs = Hkv * Sk * D;
    a.step_v_head_num = Sk * D;
    a.step_v_sl = D;
    a.step_v_head_size = 1;
    a.step_dst_bs = Hq * Sq * D;
    a.step_dst_head_num = Sq * D;
    a.step_dst_sl = D;
    a.tmp = nullptr;
    a.threading = &sth;

    bestla_sdpa_forward(a, kv_dtype);

    // Numerical comparison with per-dtype tolerances.
    const float tol = (kv_dtype == BTLA_DTYPE::F16) ? 3e-2f : 8e-2f;
    float max_diff = 0.f;
    for (int i = 0; i < Hq * Sq * D; ++i) max_diff = std::max(max_diff, std::abs(out_dst[i] - ref_dst[i]));
    if (max_diff > tol) {
      printf("[num_mixed] %s: FAIL max_abs_diff=%.5f tol=%.5f\n", tag, max_diff, tol);
      throw std::runtime_error(std::string("[num_mixed] ") + tag + ": max_abs_diff exceeds tolerance");
    }
    printf("[num_mixed] %s: PASS max_abs_diff=%.5f\n", tag, max_diff);
    return true;
  }

  static void run_all() {
    // Route 1 (F16 K/V, fp32-score, AVX2): alibi and padding-right.
    // tanh is omitted here: AVX2 scale_track_max_fp32_fp32<false,true> compiles
    // but has no if constexpr(HAS_TANH) epilogue block, so TANH30 on the F16
    // route only reduces the QK scale (no tanh function is applied). The
    // semantically correct tanh test lives on the BF16 route below.
    run_check(BTLA_DTYPE::F16, ATTN_FLAG_IS_ALIBI8, 8, "alibi/F16");
    run_check(BTLA_DTYPE::F16, ATTN_FLAG_PADDING_RIGHT, 5, "padding-right/F16");
    // Route 2 (BF16 K/V, fp32-score, AVX512F): alibi, padding-right, and tanh.
    // tanh uses the full 30*tanh(dot * QK_scale/30) path in the AVX512F kernel.
    run_check(BTLA_DTYPE::BF16, ATTN_FLAG_IS_ALIBI8, 8, "alibi/BF16");
    run_check(BTLA_DTYPE::BF16, ATTN_FLAG_PADDING_RIGHT, 5, "padding-right/BF16");
    run_check(BTLA_DTYPE::BF16, ATTN_FLAG_IS_TANH30, 8, "tanh/BF16");
    printf("[num_mixed] all numerical feature checks complete\n");
  }
};

}  // namespace ark::cpu
