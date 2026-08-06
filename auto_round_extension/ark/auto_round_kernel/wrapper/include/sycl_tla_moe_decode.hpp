// SYCL MoE Decode Kernel
//
// GEMV-style MoE kernel optimized for the decode phase, where each expert
// typically processes only 1-2 tokens (top-k routing with batch size 1).
//
// Layout convention (caller already sorted activations per expert,
// identical to the prefill `moe_gemm` interface):
//   - activations:           [total_tokens, K]            row-major
//   - weights (fp/bf16):     [num_experts, N, K]          row-major
//   - weights (int8):        [num_experts, N, K]          row-major, one
//                            int8 per byte (sym: signed -128..127;
//                            asym: unsigned 0..255 with zero-point)
//   - weights (int4 packed): [num_experts, N, K/2]        row-major, two
//                            4-bit values per byte (low nibble at lower K).
//                            The scalar-GEMV fallback repacks this on-device
//                            into an N-tiled [E, N/16, ceil(K/8), 16, 4]
//                            layout so that sub-group weight loads are
//                            coalesced *and* each lane loads 4 packed bytes at
//                            a time; the external [E, N, K/2] contract is
//                            unchanged.
//   - weights (int2 packed): [num_experts, N, K/4]        row-major, four
//                            2-bit values per byte (field j at K index
//                            4*i+j is bits [2j+1:2j])
//   - weights (fp8):         [num_experts, N, K]          row-major, one
//                            FP8 byte per weight (E4M3 / E5M2); scales
//                            applied per-group, no zero-points
//   - scales:                [num_experts, N, K/group_size]
//   - zeros (asym only):     [num_experts, N, K/group_size]
//   - num_tokens_per_expert: [num_experts]                int32
//   - outputs:               [total_tokens, N]
//
// Target: Intel BMG (Xe2), sub_group_size = 16. One sub-group per (token, N-tile)
// with N_TILE == SG_SIZE: each lane independently computes one output element,
// so no cross-lane reduction is needed and activation reads are coalesced across
// the sub-group through the L1 cache.
//
// The FP8 scalar path additionally offers a K-split mapping (one sub-group per
// output element, lanes splitting K, `ARK_MOE_DECODE_FP8_KSPLIT`, default ON):
// it trades a sub-group reduction for fully coalesced weight loads and 16x the
// thread count, which is what the memory-bound decode GEMV is short of. See the
// block comment above `launch_fp8_ksplit`.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

#include "bestla/bestla.h"
#include "sycl_tla_moe_dequant.hpp"
// S4-sym per-group DPAS grouped-GEMM (shared with the prefill path). The
// header self-guards on `ARK_XPU && ARK_SYCL_TLA`, so including it here is a
// no-op when the DPAS backend is disabled. Decode routes small-M int4-sym
// GEMV through this kernel; see `moe_gemm_decode` below.
#include "sycl_tla_moe_prefill_s4_dpas.hpp"

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

// ----------------------------------------------------------------------------
// FP8 decode implementation switch (runtime)
//
// FP8 weight bytes can be dequantized three ways, all mathematically equivalent
// for the values a real checkpoint contains:
//   - word : convert four bytes of a 32-bit weight word straight into four fp16
//            bit patterns with native DWORD field moves, folding E4M3's
//            residual 2^-8 into the per-K-group scale. No memory traffic, no
//            8-bit ALU ops.
//   - lut  : the 128-entry magnitude LUT in `bestla/sycl/fp8_lut.h` (sign
//            applied separately).
//   - bits : self-contained inline bit manipulation.
//
// Selection is done at runtime through `ARK_FP8_DECODE_MODE` ("word" / "lut" /
// "bits", case-insensitive), defaulting to "word". The legacy
// `ARK_FP8_DECODE_USE_LUT` variable still works when set explicitly and keeps
// its old meaning (truthy -> lut, falsy -> bits).
//
// The env var is read on the host and passed as a template parameter into the
// SYCL kernel, so there is no per-element runtime branch. The actual primitives
// live in `sycl_tla_moe_dequant.hpp` (shared with the mixed-input prefill
// path); this file just re-exports them via `using`.
// ----------------------------------------------------------------------------

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

namespace ark {
namespace moe_decode_detail {

constexpr int SG_SIZE = 16;
constexpr int N_TILE = SG_SIZE;  // one output element per sub-group lane

// ----------------------------------------------------------------------------
// Allocation-free boolean env-var lookup.
//
// The int4 decode dispatch consults up to three of these on *every* call -- they
// are deliberately re-read rather than cached so tests and benchmarks can toggle
// a path in-process -- and decode issues one call per generated token. Building
// a `std::string` per lookup put a heap allocation on that hot path for nothing,
// so the comparison is done in place instead. The accepted spellings are
// unchanged: "0" / "false" / "off" / "no" (case-insensitive) mean off, any other
// value means on, and an unset variable falls back to `default_value`.
// ----------------------------------------------------------------------------
inline bool env_flag_enabled(const char* name, bool default_value) {
  const char* env = std::getenv(name);
  if (env == nullptr) return default_value;
  auto iequals = [](const char* value, const char* lowercase_literal) {
    const char* a = value;
    const char* b = lowercase_literal;
    for (; *a != '\0' && *b != '\0'; ++a, ++b) {
      if (static_cast<char>(std::tolower(static_cast<unsigned char>(*a))) != *b) return false;
    }
    return *a == '\0' && *b == '\0';
  };
  return !(iequals(env, "0") || iequals(env, "false") || iequals(env, "off") || iequals(env, "no"));
}

// Token-blocking factor for the coalesced int4 decode GEMV. A work-item that
// owns one (n_tile, lane) output column processes up to TOKEN_BLOCK consecutive
// tokens, loading each packed weight byte from the (expert, n_tile) tile once
// and applying it to every token in the block that routes to the same expert.
// When decode routing is bursty (runs of tokens hitting the same expert), this
// amortizes the dominant weight traffic across the block instead of re-reading
// the tile once per token, moving the problem from pure GEMV toward GEMM. A
// value of 1 reproduces the one-token-per-work-item behaviour exactly.
constexpr int TOKEN_BLOCK = 4;

// ----------------------------------------------------------------------------
// Kernel name tags (one per specialization, required for SYCL kernel naming)
// ----------------------------------------------------------------------------
template <typename ScalarT>
class MoEDecodeKernelFP;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt4;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt4Coalesced;

template <typename ScalarT, bool Asym>
class MoEDecodeRepackInt4;

template <typename ScalarT>
class MoEDecodeActGroupSum;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt8;

template <typename ScalarT, bool Asym>
class MoEDecodeKernelInt2;

template <typename ScalarT, bool IsE4M3, moe_dequant::Fp8DecodeMode Mode>
class MoEDecodeKernelFP8;

template <typename ScalarT, bool IsE4M3, moe_dequant::Fp8DecodeMode Mode>
class MoEDecodeKernelFP8KSplit;

// ----------------------------------------------------------------------------
// FP8 weight dequantization primitives + host-side env-var reader live in
// `sycl_tla_moe_dequant.hpp` so the prefill (mixed-input Grouped GEMM) and
// decode (GEMV) paths share one definition. The `using` declarations below
// keep the in-kernel call sites (`decode_fp8<...>(byte)`) and the host-side
// `fp8_decode_use_lut()` lookup inside `moe_decode_detail` working unchanged.
// ----------------------------------------------------------------------------
using moe_dequant::Fp8DecodeMode;
using moe_dequant::decode_fp8;
using moe_dequant::decode_fp8_e4m3_bits;
using moe_dequant::decode_fp8_e4m3_lut;
using moe_dequant::decode_fp8_e5m2_bits;
using moe_dequant::decode_fp8_e5m2_lut;
using moe_dequant::decode_fp8_half_bits;
using moe_dequant::decode_fp8_quad_half_bits;
using moe_dequant::decode_int2_quad;
using moe_dequant::decode_int4_octet;
using moe_dequant::decode_int4_pair;
using moe_dequant::decode_int8;
using moe_dequant::fp8_decode_mode;
using moe_dequant::fp8_decode_use_lut;
using moe_dequant::fp8_word_scale_bias;

// ----------------------------------------------------------------------------
// Build a [total_tokens] -> expert_id mapping from num_tokens_per_expert.
// Runs on host (num_experts is small, total_tokens is small in decode).
// Caller-managed buffer (USM device allocation) keeps host noise out of the
// hot path; here we just fill it via a tiny SYCL kernel for simplicity.
// ----------------------------------------------------------------------------
inline void fill_expert_id_per_token(sycl::queue* q, int* expert_id_per_token,
                                     const int* num_tokens_per_expert, int num_experts,
                                     int total_tokens) {
  // Parallel fill: each work-item independently scans the small
  // num_tokens_per_expert array (typ. <= 256) to find its expert id. This
  // removes the single-task serialization point and avoids an explicit
  // host-device sync; the in-order queue chains this with the GEMV launch.
  if (total_tokens == 0) return;
  q->parallel_for(sycl::range<1>(static_cast<size_t>(total_tokens)), [=](sycl::id<1> idx) {
    const int i = static_cast<int>(idx[0]);
    int offset = 0;
    int expert = num_experts - 1;
    for (int e = 0; e < num_experts; ++e) {
      const int n = num_tokens_per_expert[e];
      if (i < offset + n) {
        expert = e;
        break;
      }
      offset += n;
    }
    expert_id_per_token[i] = expert;
  });
}

// ----------------------------------------------------------------------------
// FP16 / BF16 baseline GEMV (no quantization).
// ----------------------------------------------------------------------------
template <typename ScalarT>
void launch_fp(sycl::queue* q, const ScalarT* activations, const ScalarT* weights, ScalarT* outputs,
               const int* expert_id_per_token, int total_tokens, int N, int K) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode: N must be a multiple of 16");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelFP<ScalarT>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;
         const ScalarT* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;

         float acc = 0.0f;
         // Unroll by 8 with a 16-byte vector load for both activations and
         // weights. Activations are sub-group-uniform so they coalesce via
         // L1; each lane's weight load is an independent 16-byte transaction.
         // We load through a uint16_t vector to stay portable across SYCL
         // implementations that may not provide sycl::vec<bfloat16, N>.
         int k = 0;
         constexpr int VEC = 8;
         using LoadVec = sycl::vec<uint16_t, VEC>;
         static_assert(sizeof(ScalarT) == sizeof(uint16_t),
                       "ScalarT must be a 16-bit floating type");
         const int k_vec_end = (K / VEC) * VEC;
         for (; k < k_vec_end; k += VEC) {
           const LoadVec av = *reinterpret_cast<const LoadVec*>(act_row + k);
           const LoadVec wv = *reinterpret_cast<const LoadVec*>(w_row + k);
#pragma unroll
           for (int u = 0; u < VEC; ++u) {
             const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
             const ScalarT w = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(wv[u]));
             acc += static_cast<float>(a) * static_cast<float>(w);
           }
         }
         for (; k < K; ++k) {
           acc += static_cast<float>(act_row[k]) * static_cast<float>(w_row[k]);
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// Persistent per-queue device scratch pool.
//
// The int4 decode fallbacks need two device-side scratch buffers: the N-tiled
// weight repack and the per-(token, K-group) activation-sum table. Allocating
// them with `sycl::malloc_device` on every call is not viable on the decode hot
// path -- decode issues one call per generated token, and a USM allocation
// (plus the `queue::wait()` that has to precede the matching `sycl::free`)
// costs on the order of the GEMV itself. Instead each buffer is served from a
// slab that is allocated once per queue and grown on demand, so steady-state
// decode performs no allocation and needs no host-side synchronization: the
// in-order queue already serializes the producer kernel before the consumer,
// which is the same ordering guarantee `fill_expert_id_per_token` relies on.
//
// A slab additionally carries an optional *tag* -- the address of the source
// buffer it was derived from plus a caller-supplied key that must fold in
// everything else the derived contents depend on (shape, layout parameters).
// `acquire` reports whether the slab already holds the result for that exact
// tag, which lets the caller skip regenerating it. This is only consulted when
// the caller opts in (see `moe_decode_int4_repack_cache_enabled`), because the
// address half of a tag is a pointer identity and a freed-then-reallocated
// buffer can land on the same address.
//
// Slabs are intentionally never freed from a static destructor: the SYCL
// context may already be torn down at that point. `release_all` provides
// explicit teardown for callers that need it (exposed to Python as
// `moe_decode_release_scratch`).
// ----------------------------------------------------------------------------
class DeviceScratchPool {
 public:
  uint8_t* acquire(sycl::queue* q, size_t bytes, const void* tag_ptr, size_t tag_key, bool use_tag,
                   bool* tag_hit) {
    std::lock_guard<std::mutex> lock(mu_);
    Slab& slab = slabs_[q];
    if (slab.ptr == nullptr || slab.bytes < bytes) {
      if (slab.ptr != nullptr) {
        // The old slab may still be referenced by in-flight kernels.
        q->wait();
        sycl::free(slab.ptr, *q);
        slab = Slab{};
      }
      uint8_t* p = sycl::malloc_device<uint8_t>(bytes, *q);
      if (p == nullptr) {
        throw std::runtime_error("moe_gemm_decode: failed to allocate device scratch buffer");
      }
      slab.ptr = p;
      slab.bytes = bytes;
    }
    const bool hit = use_tag && slab.tagged && slab.tag_ptr == tag_ptr && slab.tag_key == tag_key;
    if (tag_hit != nullptr) *tag_hit = hit;
    if (!hit) {
      slab.tagged = use_tag;
      slab.tag_ptr = tag_ptr;
      slab.tag_key = tag_key;
    }
    return slab.ptr;
  }

  uint8_t* acquire(sycl::queue* q, size_t bytes) {
    return acquire(q, bytes, nullptr, 0, false, nullptr);
  }

  void release_all() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& kv : slabs_) {
      if (kv.second.ptr != nullptr) {
        kv.first->wait();
        sycl::free(kv.second.ptr, *kv.first);
      }
    }
    slabs_.clear();
  }

 private:
  struct Slab {
    uint8_t* ptr = nullptr;
    size_t bytes = 0;
    bool tagged = false;
    const void* tag_ptr = nullptr;
    size_t tag_key = 0;
  };
  std::mutex mu_;
  std::map<sycl::queue*, Slab> slabs_;
};

inline DeviceScratchPool& int4_repack_pool() {
  static DeviceScratchPool pool;
  return pool;
}

inline DeviceScratchPool& act_group_sum_pool() {
  static DeviceScratchPool pool;
  return pool;
}

// ----------------------------------------------------------------------------
// Per-(token, K-group) activation sums (asym int4 only).
//
// The asym int4 GEMVs fold their per-group scale/zero as
// `scale * (Σ a·q - zero · Σ a)`, where `Σ a` runs over the group's K range.
// `Σ a` depends only on the activation row and the group, *not* on the output
// column, yet the GEMVs used to recompute it inside the inner loop -- once per
// sub-group lane (16x redundant) and again for every N-tile work-group (N/16x
// redundant). That cost one extra float add per K element on the hot path.
//
// This pass computes the `[total_tokens, K/group_size]` table once, so the
// GEMVs only accumulate `Σ a·q` and read one float per group. The table is
// tiny (tokens x groups floats) and comes from the scratch pool, so no
// allocation happens in steady state.
//
// Sym does *not* use this at all: it decodes true signed nibbles, so its fold
// carries no zero-point term. That keeps this extra kernel launch -- a
// first-order cost when the GEMV itself is only tens of microseconds -- off the
// sym decode timeline entirely.
//
// The summation order differs from the previous in-loop accumulation, so
// results move by a few float ULPs -- far inside the kernel's quantization
// tolerance.
// ----------------------------------------------------------------------------
template <typename ScalarT>
void launch_act_group_sums(sycl::queue* q, const ScalarT* activations, float* a_sums, int total_tokens, int K,
                           int group_size) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  const int num_groups_k = K / group_size;
  q->parallel_for<MoEDecodeActGroupSum<ScalarT>>(
      sycl::range<2>(static_cast<size_t>(total_tokens), static_cast<size_t>(num_groups_k)),
      [=](sycl::id<2> id) {
        const int token = static_cast<int>(id[0]);
        const int g = static_cast<int>(id[1]);
        const ScalarT* row = activations + static_cast<size_t>(token) * K + static_cast<size_t>(g) * group_size;
        // Split accumulators + a 16-wide vector load, mirroring the GEMV's own
        // activation access pattern.
        float s0 = 0.0f;
        float s1 = 0.0f;
        constexpr int SUB = 16;
        using ActVec = sycl::vec<uint16_t, SUB>;
        int k = 0;
        const int end = (group_size / SUB) * SUB;
        for (; k < end; k += SUB) {
          const ActVec av = *reinterpret_cast<const ActVec*>(row + k);
#pragma unroll
          for (int u = 0; u < SUB; u += 2) {
            s0 += static_cast<float>(sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u])));
            s1 += static_cast<float>(sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u + 1])));
          }
        }
        for (; k < group_size; ++k) {
          s0 += static_cast<float>(row[k]);
        }
        a_sums[static_cast<size_t>(token) * num_groups_k + g] = s0 + s1;
      });
}

// Convenience wrapper: fetch the activation-sum table from the scratch pool and
// (re)compute it for this call's activations.
template <typename ScalarT>
float* compute_act_group_sums(sycl::queue* q, const ScalarT* activations, int total_tokens, int K,
                              int group_size) {
  const int num_groups_k = K / group_size;
  const size_t bytes = static_cast<size_t>(total_tokens) * static_cast<size_t>(num_groups_k) * sizeof(float);
  float* a_sums = reinterpret_cast<float*>(act_group_sum_pool().acquire(q, bytes));
  launch_act_group_sums<ScalarT>(q, activations, a_sums, total_tokens, K, group_size);
  return a_sums;
}

// ----------------------------------------------------------------------------
// INT4 (S4_CLIP) GEMV with group-wise dequantization.
//
// Asym=false: signed nibble in [-8, 7], dequant = q * scale
// Asym=true : unsigned nibble in [0, 15], dequant = (q - zero) * scale
//
// Packing: two 4-bit values per byte; the value at k = 2*i is the LOW nibble
// of byte i, the value at k = 2*i+1 is the HIGH nibble. This matches the
// existing CPU/XPU `packq` layout for S4_CLIP weights.
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK/2
// packed weight bytes + a vec<ScalarT,CHUNK> activation block). Templated on
// CHUNK so the caller can run a wide (32) stage first and a narrower (16)
// stage for the remainder, which keeps the fast path active for group sizes
// that are a multiple of 32 (32/64/128/256 -- the shipped quant configs)
// without regressing group_size == 16 (which drops straight to the 16-wide
// stage).
//
// The packed weights are consumed as 32-bit *words* (four packed bytes, eight
// K elements) through the shared `decode_int4_octet` primitive rather than as
// a `sycl::vec<uint8_t, N>` byte vector. On Xe the ALU is 32-bit-lane based
// and byte-typed vector operations lower to restricted byte regioning that IGC
// often has to expand, so every per-byte step in the hot loop -- the element
// extraction *and*, for sym, the sign handling -- paid that expansion. In
// word form both modes issue exactly two native DWORD operations per nibble:
//
//     asym: (word >> 4j) & 0xF
//     sym : (int)(word << (28 - 4j)) >> 28
//
// so sym's sign extension is no longer a serial byte-typed shift/narrow/shift
// chain and costs the same as asym's mask+shift. That removes the reason the
// previous revision biased sym with a `^0x88` vector XOR and folded a constant
// zero-point of 8: sym now accumulates *true signed* nibbles, which means it
// no longer needs the `Σ a` term at all (see `launch_int4`) -- one fewer fp32
// add per K element, one fewer table read per K-group, and one fewer device
// kernel launch per decode call than asym. The decoded integers are
// bit-identical to `decode_int4_pair` for every input word, so decode/prefill
// parity is unchanged.
//
// The per-group scale and zero-point are NOT applied here: this accumulates
// the raw integer-weighted dot product into `acc_q0`/`acc_q1`. The caller
// folds the group's scale (and, for asym, its zero-point against the
// precomputed `Σ a`) in exactly once per group:
//   sym : acc += scale * (acc_q0 + acc_q1)
//   asym: acc += scale * ((acc_q0 + acc_q1) - zero * Σ a)
// Hoisting the scale removes one float multiply per K element on the decode
// hot path, and because the fold is exact-once per group the result stays well
// within the kernel's existing quantization tolerance.
//
// Two independent partial accumulators (``acc_q0``/``acc_q1``) break the
// single fp32 dependency chain so the FMA pipeline is not latency-bound; the
// caller reduces the pair.
template <typename ScalarT, bool Asym, int CHUNK>
static inline void int4_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float& acc_q0,
                                     float& acc_q1) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  // sycl::vec only supports widths of 1, 2, 3, 4, 8 or 16, so a single
  // vec<uint16_t, 32> load is illegal. Process the chunk in 16-wide sub-blocks
  // (16 activations + 8 packed weight bytes each), which keeps CHUNK == 32
  // valid while reusing the same code path for CHUNK == 16. The 8 packed bytes
  // are loaded as two 32-bit words in one 8-byte transaction -- the same
  // access width (and the same 8-byte alignment requirement) as the byte
  // vector it replaces.
  constexpr int SUB = 16;
  constexpr int WORDS = SUB / 8;  // one 32-bit word per 8 K elements
  using ActVec = sycl::vec<uint16_t, SUB>;
  using WordVec = sycl::vec<uint32_t, WORDS>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    const WordVec wv = *reinterpret_cast<const WordVec*>(w_ptr + s * (SUB / 2));
#pragma unroll
    for (int w = 0; w < WORDS; ++w) {
      int q[8];
      decode_int4_octet<Asym>(wv[w], q);
#pragma unroll
      for (int u = 0; u < 8; u += 2) {
        const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[8 * w + u]));
        const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[8 * w + u + 1]));
        acc_q0 += static_cast<float>(a0) * static_cast<float>(q[u]);
        acc_q1 += static_cast<float>(a1) * static_cast<float>(q[u + 1]);
      }
    }
  }
}

template <typename ScalarT, bool Asym>
void launch_int4(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                 const ScalarT* zeros, ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N,
                 int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): N must be a multiple of 16");
  }
  if (K % group_size != 0 || (group_size & 1) != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): K must be a multiple of group_size and group_size must be even");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int4): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;
  const int k_packed = K / 2;  // bytes of packed weight per (expert, n)

  // Per-(token, K-group) activation sums, shared by every lane and every
  // N-tile instead of being recomputed inside the inner loop. Only the *asym*
  // fold needs them (`Σ a·(q - z) == Σ a·q - z·Σ a`): sym decodes true signed
  // nibbles, so its fold is a plain per-group scale multiply with no
  // zero-point term. Skipping the pre-pass keeps a whole extra kernel launch
  // off the sym decode timeline -- on decode-sized batches the GEMV itself is
  // only tens of microseconds, so an extra dispatch is a first-order cost.
  [[maybe_unused]] const float* a_sums = nullptr;
  if constexpr (Asym) {
    a_sums = compute_act_group_sums<ScalarT>(q, activations, total_tokens, K, group_size);
  }

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt4<ScalarT, Asym>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * k_packed;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
         [[maybe_unused]] const ScalarT* z_row = nullptr;
         [[maybe_unused]] const float* a_sum_row = nullptr;
         if constexpr (Asym) {
           z_row = zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
           a_sum_row = a_sums + static_cast<size_t>(token) * num_groups_k;
         }

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]);
           const int k_base = g * group_size;
           // Vectorized ladder: process 32 K-elements at a time (16 packed
           // weight bytes + vec<ScalarT,32> activation block), then a 16-wide
           // stage for the remainder, then a scalar tail. Widening the first
           // stage to 32 amortizes the per-group scale load and loop overhead
           // across twice as many multiply-adds for the shipped group sizes
           // (32/64/128/256), while the 16-wide stage keeps group_size == 16
           // on the fast path.
           //
           // The scale and zero are constant across the group, so the wide
           // stages accumulate only the raw integer-weighted dot product
           // ``Σ a·q`` (split across two partial accumulators to break the
           // fp32 dependency chain). The fold below applies the scale exactly
           // once per group; asym additionally subtracts its per-group
           // zero-point against the precomputed ``Σ a``:
           //   sym : acc += scale * (acc_q0 + acc_q1)
           //   asym: acc += scale * ((acc_q0 + acc_q1) - zero * a_sum)
           float acc_q0 = 0.0f;
           float acc_q1 = 0.0f;
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             int4_decode_chunk<ScalarT, Asym, CHUNK32>(act_row + k_base + kk, w_row + (k_base + kk) / 2, acc_q0,
                                                       acc_q1);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             int4_decode_chunk<ScalarT, Asym, CHUNK16>(act_row + k_base + kk, w_row + (k_base + kk) / 2, acc_q0,
                                                       acc_q1);
           }
           // Scalar tail for group_size not divisible by 16. Uses the same
           // raw-accumulation convention as the wide stages so the single
           // scale/zero fold below stays valid.
           for (; kk < group_size; kk += 2) {
             const uint8_t packed = w_row[(k_base + kk) / 2];
             int q0, q1;
             decode_int4_pair<Asym>(packed, q0, q1);
             const float fa0 = static_cast<float>(act_row[k_base + kk]);
             const float fa1 = static_cast<float>(act_row[k_base + kk + 1]);
             acc_q0 += fa0 * static_cast<float>(q0);
             acc_q1 += fa1 * static_cast<float>(q1);
           }
           if constexpr (Asym) {
             acc += scale * ((acc_q0 + acc_q1) - static_cast<float>(z_row[g]) * a_sum_row[g]);
           } else {
             acc += scale * (acc_q0 + acc_q1);
           }
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// Opt-in reuse of the int4 weight repack across calls.
//
// The repack output depends only on the weight buffer, which does not change
// between decode steps of a real inference loop, so in principle it can be
// built once and reused. The pool tag is a *pointer identity*, though, and a
// freed-then-reallocated weight tensor can land on the address of the previous
// one (torch's caching allocator makes this common in test loops that build a
// fresh packed tensor of the same shape per iteration). Reusing a stale repack
// would then silently produce wrong results, so this is off by default and must
// be enabled explicitly by a caller that owns the weight lifetime:
//
//   ARK_MOE_DECODE_INT4_REPACK_CACHE=1
//
// `ark::moe_decode_release_scratch()` (exposed to Python as
// `moe_decode_release_scratch`) drops the cached buffers.
// ----------------------------------------------------------------------------
inline bool moe_decode_int4_repack_cache_enabled() {
  return env_flag_enabled("ARK_MOE_DECODE_INT4_REPACK_CACHE", false);  // default OFF -- see comment above
}

// ----------------------------------------------------------------------------
// INT4 (S4_CLIP) coalesced-load GEMV.
//
// The scalar `launch_int4` above is memory-bandwidth-bound: for a single decode
// token it just streams the whole packed weight matrix once with ~1 MAC per
// byte, so the arithmetic tweaks (split accumulators, hoisted scale) cannot
// help. Its real cost is that weight loads are *not coalesced across the
// sub-group*: with the `[E, N, K/2]` (K-contiguous) layout, lane `l` and lane
// `l+1` of a sub-group read packed bytes `K/2` apart at a fixed `k`, so each
// step issues 16 scattered transactions instead of one contiguous cache line.
//
// This path fixes that by first repacking the weights on-device into an
// N-tiled, 4-byte-blocked layout `[E, N/16, ceil(K/8), 16, 4]`: a *chunk* holds
// four consecutive packed bytes for each of the 16 columns owned by a sub-group
// tile, lane-major. Lane `l` therefore reads its four bytes at chunk offset
// `l*4`, and the 16 lanes of a sub-group together cover 64 contiguous bytes ->
// still a single coalesced transaction, but now each lane issues one 32-bit
// word load instead of four separate byte loads. That word is decoded with the
// shared `decode_int4_octet` primitive, so all eight nibbles are extracted with
// native DWORD shift/mask pairs and neither mode touches the 8-bit ALU (see
// `int4_decode_chunk`). The dequant math is otherwise identical to
// `launch_int4` (bit-identical nibbles, same per-group scale/zero fold, sym
// accumulating true signed nibbles with no `Σ a` term); only the weight memory
// layout changes, and the caller's `[E, N, K/2]` weight contract is unchanged.
//
// Group sizes that are a multiple of 8 (16/32/64/128/256 -- the shipped quant
// configs) start every K-group on a chunk boundary, so the vectorized stage
// covers the whole group. Other even group sizes are handled by a scalar
// prologue/epilogue around the vector stage, which reads the same layout one
// byte at a time.
//
// The repack buffer comes from the persistent per-queue scratch pool
// (`DeviceScratchPool`), so decode steady state performs no USM allocation and
// -- unlike the previous transient allocation, which had to be freed behind a
// blocking `queue::wait()` on every call -- introduces no host-side
// synchronization. The repack kernel itself still runs per call unless the
// caller opts into `ARK_MOE_DECODE_INT4_REPACK_CACHE`.
//
// On top of coalescing, this path blocks tokens: each work-item owns one
// output column but processes up to `TOKEN_BLOCK` consecutive tokens. For each
// distinct expert appearing in the block it makes a single weight-streaming
// pass and reuses every loaded (coalesced) byte across all tokens in the block
// routed to that expert. When decode routing is bursty -- runs of tokens
// hitting the same expert -- this amortizes the dominant weight traffic across
// the block (GEMV -> small GEMM). Fully-scattered routing degrades gracefully
// to one pass per token with the same per-pass weight reads as before, so the
// result is independent of routing.
// ----------------------------------------------------------------------------
template <typename ScalarT, bool Asym>
void launch_int4_coalesced(sycl::queue* q, const ScalarT* activations, const uint8_t* weights,
                           const ScalarT* scales, const ScalarT* zeros, ScalarT* outputs,
                           const int* expert_id_per_token, int total_tokens, int N, int K, int group_size,
                           int num_experts) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): N must be a multiple of 16");
  }
  if (K % group_size != 0 || (group_size & 1) != 0) {
    throw std::invalid_argument("moe_gemm_decode(int4): K must be a multiple of group_size and group_size must be even");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int4): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;
  const int k_packed = K / 2;  // bytes of packed weight per (expert, n)
  // Packed bytes are blocked by 4 along K so each lane can issue one 4-byte
  // load. The last chunk is zero-padded when k_packed is not a multiple of 4.
  constexpr int PACK_VEC = 4;
  const int k_chunks = (k_packed + PACK_VEC - 1) / PACK_VEC;
  const int chunk_stride = N_TILE * PACK_VEC;  // bytes per (chunk) across the tile

  const size_t repacked_bytes = static_cast<size_t>(num_experts) * static_cast<size_t>(n_tiles) *
                                static_cast<size_t>(k_chunks) * static_cast<size_t>(chunk_stride);
  // Reuse the repack across calls only when the caller opted in; the tag key
  // folds in the full shape so a tensor of different dimensions cannot alias a
  // cached repack that happens to sit at the same address.
  const size_t repack_key = (static_cast<size_t>(num_experts) * 1000003u + static_cast<size_t>(N)) * 1000003u +
                            static_cast<size_t>(k_packed);
  bool repack_cached = false;
  uint8_t* repacked = int4_repack_pool().acquire(q, repacked_bytes, weights, repack_key,
                                                 moe_decode_int4_repack_cache_enabled(), &repack_cached);

  // Repack kernel: one work-item per (expert, column, packed-byte slot). The
  // write index places the 16 columns of a tile contiguously in chunks of 4
  // bytes, lane-major. Slots past `k_packed` are zero-filled so the padded tail
  // of the last chunk is always initialized.
  if (!repack_cached) {
    sycl::range<3> rp_global{static_cast<size_t>(num_experts), static_cast<size_t>(N),
                             static_cast<size_t>(k_chunks * PACK_VEC)};
    q->parallel_for<MoEDecodeRepackInt4<ScalarT, Asym>>(rp_global, [=](sycl::id<3> id) {
      const int e = static_cast<int>(id[0]);
      const int n = static_cast<int>(id[1]);
      const int kb = static_cast<int>(id[2]);
      const int t = n / N_TILE;
      const int l = n % N_TILE;
      const int c = kb / PACK_VEC;
      const int r = kb % PACK_VEC;
      const size_t dst = ((static_cast<size_t>(e) * n_tiles + t) * k_chunks + c) * chunk_stride +
                         static_cast<size_t>(l) * PACK_VEC + r;
      if (kb < k_packed) {
        repacked[dst] = weights[(static_cast<size_t>(e) * N + static_cast<size_t>(n)) * k_packed + kb];
      } else {
        repacked[dst] = 0;
      }
    });
  }

  // Per-(token, K-group) activation sums, hoisted out of the inner loop. Only
  // asym needs them -- sym decodes true signed nibbles and folds a plain scale
  // -- so the sym path skips this kernel launch entirely (see `launch_int4`).
  [[maybe_unused]] const float* a_sums = nullptr;
  if constexpr (Asym) {
    a_sums = compute_act_group_sums<ScalarT>(q, activations, total_tokens, K, group_size);
  }

  sycl::range<2> global{static_cast<size_t>((total_tokens + TOKEN_BLOCK - 1) / TOKEN_BLOCK),
                        static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt4Coalesced<ScalarT, Asym>>(
      sycl::nd_range<2>(global, local),
      [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
        const int token_base = static_cast<int>(it.get_global_id(0)) * TOKEN_BLOCK;
        const int n_tile = static_cast<int>(it.get_group(1));
        const int lane = static_cast<int>(it.get_local_id(1));
        const int n_global = n_tile * N_TILE + lane;

        // Number of tokens this work-item owns (last block may be short).
        int block = TOKEN_BLOCK;
        if (token_base + block > total_tokens) {
          block = total_tokens - token_base;
        }

        // Experts routed by each token in the block. The tile weight byte is
        // loaded once per k-step and reused only for tokens whose expert
        // matches the byte's owning expert, so blocking tokens that share an
        // expert amortizes the dominant weight traffic; tokens with a
        // different expert contribute nothing from this pass and are handled
        // by the pass whose leader expert matches theirs.
        int experts[TOKEN_BLOCK];
        for (int b = 0; b < block; ++b) {
          experts[b] = expert_id_per_token[token_base + b];
        }

        // Which distinct experts appear in this block. For each we make one
        // weight-streaming pass, reusing every loaded byte across all tokens
        // in the block routed to that expert. Bursty routing collapses to a
        // single pass; fully-scattered routing degrades to one pass per token
        // (i.e. the previous behaviour) with no extra weight reads per pass.
        for (int lead = 0; lead < block; ++lead) {
          const int expert = experts[lead];
          // Skip experts already streamed by an earlier token in this block.
          bool seen = false;
          for (int p = 0; p < lead; ++p) {
            if (experts[p] == expert) {
              seen = true;
              break;
            }
          }
          if (seen) continue;

          // Base of this (expert, n_tile) weight tile in the repacked buffer.
          // Layout [E, N/16, ceil(K/8), 16, 4]; this lane reads packed byte
          // `b_abs` at w_tile[(b_abs/4)*64 + lane*4 + b_abs%4], so the 16 lanes
          // of the sub-group span 64 contiguous bytes per chunk.
          const uint8_t* w_tile =
              repacked + (static_cast<size_t>(expert) * n_tiles + n_tile) * k_chunks * chunk_stride;
          const ScalarT* s_row =
              scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
          [[maybe_unused]] const ScalarT* z_row = nullptr;
          if constexpr (Asym) {
            z_row = zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
          }

          // Compact the tokens routed to `expert` into a dense member list
          // once per pass. Hoisting the routing filter out of the hot k-loop
          // removes a per-(kb, token) branch and lets the compiler keep the
          // per-member activation base pointers in registers; the numerics
          // are identical to the previous per-kb `experts[b] != expert`
          // filter.
          int members[TOKEN_BLOCK];
          const ScalarT* act_rows[TOKEN_BLOCK];
          [[maybe_unused]] const float* a_sum_rows[TOKEN_BLOCK];
          int nmembers = 0;
          for (int b = 0; b < block; ++b) {
            if (experts[b] != expert) continue;
            members[nmembers] = b;
            act_rows[nmembers] = activations + static_cast<size_t>(token_base + b) * K;
            if constexpr (Asym) {
              a_sum_rows[nmembers] = a_sums + static_cast<size_t>(token_base + b) * num_groups_k;
            }
            ++nmembers;
          }

          float acc[TOKEN_BLOCK];
          for (int m = 0; m < nmembers; ++m) acc[m] = 0.0f;

          for (int g = 0; g < num_groups_k; ++g) {
            const float scale = static_cast<float>(s_row[g]);
            const int k_base = g * group_size;
            // Per-token split accumulators for the raw integer-weighted dot
            // product; the per-group scale (and, for asym, the zero-point
            // against the precomputed activation sum) are folded once after
            // the K-loop, exactly as in the scalar path.
            float acc_q0[TOKEN_BLOCK];
            float acc_q1[TOKEN_BLOCK];
            for (int m = 0; m < nmembers; ++m) {
              acc_q0[m] = 0.0f;
              acc_q1[m] = 0.0f;
            }

            // Accumulate one decoded nibble pair (two K elements) into every
            // token of this pass.
            auto accumulate_pair = [&](int q0, int q1, int k0) {
              const float fq0 = static_cast<float>(q0);
              const float fq1 = static_cast<float>(q1);
              for (int m = 0; m < nmembers; ++m) {
                const ScalarT* act_row = act_rows[m];
                acc_q0[m] += static_cast<float>(act_row[k0]) * fq0;
                acc_q1[m] += static_cast<float>(act_row[k0 + 1]) * fq1;
              }
            };
            // Load and decode a single packed byte through the chunked layout.
            auto accumulate_byte = [&](int b_abs, int k0) {
              const uint8_t packed = w_tile[static_cast<size_t>(b_abs / PACK_VEC) * chunk_stride +
                                            static_cast<size_t>(lane) * PACK_VEC + (b_abs % PACK_VEC)];
              int q0, q1;
              decode_int4_pair<Asym>(packed, q0, q1);
              accumulate_pair(q0, q1, k0);
            };

            const int kb_base = k_base / 2;
            const int kb_count = group_size / 2;
            int kb = 0;
            // Prologue to the next 4-byte chunk boundary. Empty whenever
            // group_size % 8 == 0, i.e. for every shipped quant config.
            for (; kb < kb_count && ((kb_base + kb) % PACK_VEC) != 0; ++kb) {
              accumulate_byte(kb_base + kb, k_base + 2 * kb);
            }
            // A lane's PACK_VEC == 4 bytes inside a chunk are contiguous, so
            // they are exactly one little-endian 32-bit word: load it as such
            // and decode all 8 nibbles with native DWORD ops (no 8-bit ALU,
            // no sign-bias XOR for sym) via the shared octet primitive.
            for (; kb + PACK_VEC <= kb_count; kb += PACK_VEC) {
              const int b_abs = kb_base + kb;  // 4-byte aligned here
              const uint32_t word = *reinterpret_cast<const uint32_t*>(
                  w_tile + static_cast<size_t>(b_abs / PACK_VEC) * chunk_stride +
                  static_cast<size_t>(lane) * PACK_VEC);
              int qv[8];
              decode_int4_octet<Asym>(word, qv);
#pragma unroll
              for (int u = 0; u < PACK_VEC; ++u) {
                accumulate_pair(qv[2 * u], qv[2 * u + 1], k_base + 2 * (kb + u));
              }
            }
            // Scalar tail for group sizes that are not a multiple of 8.
            for (; kb < kb_count; ++kb) {
              accumulate_byte(kb_base + kb, k_base + 2 * kb);
            }

            if constexpr (Asym) {
              const float zero = static_cast<float>(z_row[g]);
              for (int m = 0; m < nmembers; ++m) {
                acc[m] += scale * ((acc_q0[m] + acc_q1[m]) - zero * a_sum_rows[m][g]);
              }
            } else {
              for (int m = 0; m < nmembers; ++m) {
                acc[m] += scale * (acc_q0[m] + acc_q1[m]);
              }
            }
          }

          for (int m = 0; m < nmembers; ++m) {
            const int b = members[m];
            outputs[static_cast<size_t>(token_base + b) * N + n_global] = static_cast<ScalarT>(acc[m]);
          }
        }
      });
}

// ----------------------------------------------------------------------------
// INT8 (S8) GEMV with group-wise dequantization.
//
// Asym=false: signed byte in [-128, 127], dequant = q * scale
// Asym=true : unsigned byte in [0, 255], dequant = (q - zero) * scale
//
// Weights are stored as raw uint8 bytes (1 byte per weight). The same buffer
// type is used for sym and asym; the only difference is the sign interpretation
// performed at decode time.
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK weight
// bytes + a vec<ScalarT,CHUNK> activation block). Templated on CHUNK so the
// caller can run a wide (32) stage first and a narrower (16) stage for the
// remainder, mirroring the int4 path: widening the first stage amortizes the
// per-group scale load and loop overhead across twice as many multiply-adds for
// the shipped group sizes (32/64/128/256) without regressing group_size == 16.
// sycl::vec only supports widths of 1, 2, 3, 4, 8 or 16, so CHUNK is processed
// in 16-wide sub-blocks. The math is identical to the scalar path.
template <typename ScalarT, bool Asym, int CHUNK>
static inline void int8_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float scale, float zero,
                                     float& acc) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  constexpr int SUB = 16;
  using ActVec = sycl::vec<uint16_t, SUB>;
  using ByteVec = sycl::vec<uint8_t, SUB>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    const ByteVec wv = *reinterpret_cast<const ByteVec*>(w_ptr + s * SUB);
#pragma unroll
    for (int u = 0; u < SUB; ++u) {
      const int qv = decode_int8<Asym>(wv[u]);
      float w;
      if constexpr (Asym) {
        w = (static_cast<float>(qv) - zero) * scale;
      } else {
        w = static_cast<float>(qv) * scale;
      }
      const ScalarT a = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
      acc += static_cast<float>(a) * w;
    }
  }
}

template <typename ScalarT, bool Asym>
void launch_int8(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                 const ScalarT* zeros, ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N,
                 int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int8): N must be a multiple of 16");
  }
  if (K % group_size != 0) {
    throw std::invalid_argument("moe_gemm_decode(int8): K must be a multiple of group_size");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int8): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt8<ScalarT, Asym>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
         const ScalarT* z_row = Asym
             ? zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k
             : nullptr;

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]);
           float zero = 0.0f;
           if constexpr (Asym) {
             zero = static_cast<float>(z_row[g]);
           }
           const int k_base = g * group_size;
           // Vectorized ladder mirroring the int4 path: process 32 K-elements
           // (32 weight bytes + vec<ScalarT,32> activations) at a time, then a
           // 16-wide stage for the remainder, then a scalar tail. Widening the
           // first stage amortizes the per-group scale load and loop overhead
           // for the shipped group sizes (32/64/128/256), while the 16-wide
           // stage keeps group_size == 16 on the fast path.
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             int8_decode_chunk<ScalarT, Asym, CHUNK32>(act_row + k_base + kk, w_row + k_base + kk, scale, zero,
                                                       acc);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             int8_decode_chunk<ScalarT, Asym, CHUNK16>(act_row + k_base + kk, w_row + k_base + kk, scale, zero,
                                                       acc);
           }
           for (; kk < group_size; ++kk) {
             const int qv = decode_int8<Asym>(w_row[k_base + kk]);
             float w;
             if constexpr (Asym) {
               w = (static_cast<float>(qv) - zero) * scale;
             } else {
               w = static_cast<float>(qv) * scale;
             }
             acc += static_cast<float>(act_row[k_base + kk]) * w;
           }
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// INT2 (S2_CLIP) GEMV with group-wise dequantization.
//
// Packing: 4 values per byte. The value at K index 4*i + j is stored in
// bits [2j+1 : 2j] of byte i (i.e. byte = q0 | (q1<<2) | (q2<<4) | (q3<<6)).
//
// Asym=false: signed 2-bit value in [-2, 1]; dequant = q * scale
// Asym=true : unsigned 2-bit value in [0, 3]; dequant = (q - zero) * scale
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK/4
// packed weight bytes + a vec<ScalarT,CHUNK> activation block). Templated on
// CHUNK so the caller can run a wide (32) stage first and a narrower (16) stage
// for the remainder, mirroring the int4/int8 paths. sycl::vec only supports
// widths of 1, 2, 3, 4, 8 or 16, so CHUNK is processed in 16-wide sub-blocks
// (16 activations + 4 packed bytes each). The math is identical to the scalar
// path.
template <typename ScalarT, bool Asym, int CHUNK>
static inline void int2_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float scale, float zero,
                                     float& acc) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  constexpr int SUB = 16;
  using ActVec = sycl::vec<uint16_t, SUB>;
  using PackVec = sycl::vec<uint8_t, SUB / 4>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    const PackVec pv = *reinterpret_cast<const PackVec*>(w_ptr + s * (SUB / 4));
#pragma unroll
    for (int b = 0; b < SUB / 4; ++b) {
      int qq[4];
      decode_int2_quad<Asym>(pv[b], qq);
      float w0, w1, w2, w3;
      if constexpr (Asym) {
        w0 = (static_cast<float>(qq[0]) - zero) * scale;
        w1 = (static_cast<float>(qq[1]) - zero) * scale;
        w2 = (static_cast<float>(qq[2]) - zero) * scale;
        w3 = (static_cast<float>(qq[3]) - zero) * scale;
      } else {
        w0 = static_cast<float>(qq[0]) * scale;
        w1 = static_cast<float>(qq[1]) * scale;
        w2 = static_cast<float>(qq[2]) * scale;
        w3 = static_cast<float>(qq[3]) * scale;
      }
      const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 0]));
      const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 1]));
      const ScalarT a2 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 2]));
      const ScalarT a3 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * b + 3]));
      acc += static_cast<float>(a0) * w0;
      acc += static_cast<float>(a1) * w1;
      acc += static_cast<float>(a2) * w2;
      acc += static_cast<float>(a3) * w3;
    }
  }
}

template <typename ScalarT, bool Asym>
void launch_int2(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                 const ScalarT* zeros, ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N,
                 int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(int2): N must be a multiple of 16");
  }
  if ((K & 0x3) != 0) {
    throw std::invalid_argument("moe_gemm_decode(int2): K must be a multiple of 4");
  }
  if (K % group_size != 0 || (group_size & 0x3) != 0) {
    throw std::invalid_argument(
        "moe_gemm_decode(int2): K must be a multiple of group_size and group_size must be a multiple of 4");
  }
  if (Asym && zeros == nullptr) {
    throw std::invalid_argument("moe_gemm_decode(int2): zeros pointer required when asym=true");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;
  const int k_packed = K / 4;  // bytes of packed weight per (expert, n)

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelInt2<ScalarT, Asym>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * k_packed;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;
         const ScalarT* z_row = Asym
             ? zeros + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k
             : nullptr;

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]);
           float zero = 0.0f;
           if constexpr (Asym) {
             zero = static_cast<float>(z_row[g]);
           }
           const int k_base = g * group_size;
           // Vectorized ladder mirroring the int4/int8 paths: process 32
           // K-elements (8 packed bytes + vec<ScalarT,32> activations) at a
           // time, then a 16-wide stage for the remainder, then a scalar tail.
           // group_size is a multiple of 4; the wide stage amortizes the
           // per-group scale load for the shipped group sizes (32/64/128/256).
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             int2_decode_chunk<ScalarT, Asym, CHUNK32>(act_row + k_base + kk, w_row + (k_base + kk) / 4, scale,
                                                       zero, acc);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             int2_decode_chunk<ScalarT, Asym, CHUNK16>(act_row + k_base + kk, w_row + (k_base + kk) / 4, scale,
                                                       zero, acc);
           }
           // Scalar tail (4 values per byte).
           for (; kk < group_size; kk += 4) {
             const uint8_t packed = w_row[(k_base + kk) / 4];
             int q[4];
             decode_int2_quad<Asym>(packed, q);
             float w[4];
             if constexpr (Asym) {
               w[0] = (static_cast<float>(q[0]) - zero) * scale;
               w[1] = (static_cast<float>(q[1]) - zero) * scale;
               w[2] = (static_cast<float>(q[2]) - zero) * scale;
               w[3] = (static_cast<float>(q[3]) - zero) * scale;
             } else {
               w[0] = static_cast<float>(q[0]) * scale;
               w[1] = static_cast<float>(q[1]) * scale;
               w[2] = static_cast<float>(q[2]) * scale;
               w[3] = static_cast<float>(q[3]) * scale;
             }
             acc += static_cast<float>(act_row[k_base + kk + 0]) * w[0];
             acc += static_cast<float>(act_row[k_base + kk + 1]) * w[1];
             acc += static_cast<float>(act_row[k_base + kk + 2]) * w[2];
             acc += static_cast<float>(act_row[k_base + kk + 3]) * w[3];
           }
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// FP8 (E4M3 / E5M2) GEMV with group-wise scale (no zero-point).
//
// Weights are 1 FP8 byte per element [E, N, K]. How a byte becomes a float is
// chosen at launch time by `fp8_decode_mode()` and passed in as the `Mode`
// template parameter, so the hot path stays branch-free:
//
//   * `kWord` (default) -- the four bytes of a 32-bit weight word are turned
//     into four fp16 bit patterns by `decode_fp8_quad_half_bits`, i.e. a couple
//     of native DWORD ops and no memory traffic at all. This mirrors the
//     word-native `decode_int4_octet` treatment that made int4 decode fast: Xe
//     ALU lanes are 32-bit, so the previous `sycl::vec<uint8_t, 16>` weight
//     vector plus per-byte decode paid narrow-type regioning on *every* weight
//     element, and the LUT variant additionally issued one load per element in
//     a loop that already does only ~1 MAC per byte.
//     For E4M3 the field move leaves a constant 2^-8 factor, which is folded
//     into the per-K-group scale below (`fp8_word_scale_bias`), so it costs
//     nothing per element.
//
//   * `kLut` / `kBits` -- the original per-byte `decode_fp8<IsE4M3, UseLut>`
//     decoders, kept for A/B measurement and regression escape.
//
// Two lane mappings share those decoders: the legacy per-work-item GEMV
// (`launch_fp8`) and the K-split GEMV (`launch_fp8_ksplit`, default, see its
// block comment). `launch_fp8_by_mode` picks between them.
// ----------------------------------------------------------------------------

// Vectorized inner accumulation over CHUNK consecutive K elements (CHUNK weight
// bytes + a vec<ScalarT,CHUNK> activation block). Templated on CHUNK so the
// caller can run a wide (32) stage first and a narrower (16) stage for the
// remainder, mirroring the int4/int8 paths. sycl::vec only supports widths of
// 1, 2, 3, 4, 8 or 16, so CHUNK is processed in 16-wide sub-blocks.
//
// In `kWord` mode the 16 weight bytes of a sub-block are read as a
// `sycl::vec<uint32_t, 4>` -- the same 16-byte transaction (and the same
// 16-byte alignment requirement) as the byte vector it replaces, but 32-bit
// typed, so the decode never leaves the native datapath.
//
// The per-group scale is constant across the whole group, so it is NOT applied
// here: this accumulates the raw dot product (sum of act * decoded_fp8) and the
// caller multiplies the group total by the scale once (Σ a·(w·s) == s·Σ a·w).
// For the per-expert / per-tensor scale case (group_size == K, one scale per
// output row) this collapses the whole K reduction to a single scale multiply,
// removing one multiply per K element on the decode hot path.
//
// Two independent partial accumulators break the single fp32 dependency chain
// so the FMA pipeline is not latency-bound (same trick as `int4_decode_chunk`);
// the caller reduces the pair.
template <typename ScalarT, bool IsE4M3, Fp8DecodeMode Mode, int CHUNK>
static inline void fp8_decode_chunk(const ScalarT* act_ptr, const uint8_t* w_ptr, float& acc0,
                                    float& acc1) {
  static_assert(sizeof(ScalarT) == sizeof(uint16_t), "ScalarT must be a 16-bit floating type");
  static_assert(CHUNK % 16 == 0, "CHUNK must be a multiple of 16");
  constexpr int SUB = 16;
  using ActVec = sycl::vec<uint16_t, SUB>;
#pragma unroll
  for (int s = 0; s < CHUNK / SUB; ++s) {
    const ActVec av = *reinterpret_cast<const ActVec*>(act_ptr + s * SUB);
    if constexpr (Mode == Fp8DecodeMode::kWord) {
      constexpr int WORDS = SUB / 4;  // one 32-bit word per 4 FP8 bytes
      using WordVec = sycl::vec<uint32_t, WORDS>;
      const WordVec wv = *reinterpret_cast<const WordVec*>(w_ptr + s * SUB);
#pragma unroll
      for (int w = 0; w < WORDS; ++w) {
        uint32_t lo2, hi2;
        decode_fp8_quad_half_bits<IsE4M3>(wv[w], lo2, hi2);
        const uint16_t hb[4] = {static_cast<uint16_t>(lo2), static_cast<uint16_t>(lo2 >> 16),
                                static_cast<uint16_t>(hi2), static_cast<uint16_t>(hi2 >> 16)};
#pragma unroll
        for (int u = 0; u < 4; u += 2) {
          const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * w + u]));
          const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[4 * w + u + 1]));
          acc0 += static_cast<float>(a0) * static_cast<float>(sycl::bit_cast<sycl::half>(hb[u]));
          acc1 += static_cast<float>(a1) * static_cast<float>(sycl::bit_cast<sycl::half>(hb[u + 1]));
        }
      }
    } else {
      constexpr bool kUseLut = (Mode == Fp8DecodeMode::kLut);
      using ByteVec = sycl::vec<uint8_t, SUB>;
      const ByteVec wv = *reinterpret_cast<const ByteVec*>(w_ptr + s * SUB);
#pragma unroll
      for (int u = 0; u < SUB; u += 2) {
        const float w0 = decode_fp8<IsE4M3, kUseLut>(wv[u]);
        const float w1 = decode_fp8<IsE4M3, kUseLut>(wv[u + 1]);
        const ScalarT a0 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u]));
        const ScalarT a1 = sycl::bit_cast<ScalarT>(static_cast<uint16_t>(av[u + 1]));
        acc0 += static_cast<float>(a0) * w0;
        acc1 += static_cast<float>(a1) * w1;
      }
    }
  }
}

// Single-byte decode matching `fp8_decode_chunk`'s convention: in `kWord` mode
// the returned value carries the same folded 2^-8 bias as the vector stage, so
// the scalar tail can share the group accumulator.
template <bool IsE4M3, Fp8DecodeMode Mode>
static inline float fp8_decode_scalar(uint8_t raw) {
  if constexpr (Mode == Fp8DecodeMode::kWord) {
    return static_cast<float>(sycl::bit_cast<sycl::half>(decode_fp8_half_bits<IsE4M3>(raw)));
  } else {
    return decode_fp8<IsE4M3, Mode == Fp8DecodeMode::kLut>(raw);
  }
}

template <typename ScalarT, bool IsE4M3, Fp8DecodeMode Mode>
void launch_fp8(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N, int K, int group_size) {
  if (N % N_TILE != 0) {
    throw std::invalid_argument("moe_gemm_decode(fp8): N must be a multiple of 16");
  }
  if (K % group_size != 0) {
    throw std::invalid_argument("moe_gemm_decode(fp8): K must be a multiple of group_size");
  }
  if (total_tokens == 0) return;

  const int n_tiles = N / N_TILE;
  const int num_groups_k = K / group_size;
  // Undoes the exponent re-bias the word-native decode leaves behind (1.0f for
  // every other mode). Exact power of two, applied once per K-group.
  constexpr float kScaleBias =
      (Mode == Fp8DecodeMode::kWord) ? fp8_word_scale_bias<IsE4M3>() : 1.0f;

  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(n_tiles * SG_SIZE)};
  sycl::range<2> local{1, static_cast<size_t>(SG_SIZE)};

  q->parallel_for<MoEDecodeKernelFP8<ScalarT, IsE4M3, Mode>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const int token = static_cast<int>(it.get_global_id(0));
         const int n_tile = static_cast<int>(it.get_group(1));
         const int lane = static_cast<int>(it.get_local_id(1));
         const int n_global = n_tile * N_TILE + lane;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;

         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;

         float acc = 0.0f;
         for (int g = 0; g < num_groups_k; ++g) {
           const float scale = static_cast<float>(s_row[g]) * kScaleBias;
           const int k_base = g * group_size;
           // Vectorized ladder mirroring the int4/int8 paths: process 32
           // K-elements (32 weight bytes + vec<ScalarT,32> activations) at a
           // time, then a 16-wide stage for the remainder, then a scalar tail.
           // The per-group scale is constant across the group, so accumulate the
           // raw dot product here and apply the scale once below (Σ a·(w·s) ==
           // s·Σ a·w). Widening the first stage amortizes the per-group scale
           // load for the shipped group sizes (32/64/128/256); hoisting the
           // scale removes one multiply per K element, which is the dominant
           // cost for the per-expert / per-tensor scale case (group_size == K).
           // Two partial accumulators break the fp32 dependency chain.
           float group_acc0 = 0.0f;
           float group_acc1 = 0.0f;
           int kk = 0;
           constexpr int CHUNK32 = 32;
           const int end32 = (group_size / CHUNK32) * CHUNK32;
           for (; kk < end32; kk += CHUNK32) {
             fp8_decode_chunk<ScalarT, IsE4M3, Mode, CHUNK32>(act_row + k_base + kk, w_row + k_base + kk,
                                                              group_acc0, group_acc1);
           }
           constexpr int CHUNK16 = 16;
           const int end16 = kk + ((group_size - kk) / CHUNK16) * CHUNK16;
           for (; kk < end16; kk += CHUNK16) {
             fp8_decode_chunk<ScalarT, IsE4M3, Mode, CHUNK16>(act_row + k_base + kk, w_row + k_base + kk,
                                                              group_acc0, group_acc1);
           }
           for (; kk < group_size; ++kk) {
             const float w = fp8_decode_scalar<IsE4M3, Mode>(w_row[k_base + kk]);
             group_acc0 += static_cast<float>(act_row[k_base + kk]) * w;
           }
           acc += (group_acc0 + group_acc1) * scale;
         }

         outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(acc);
       });
}

// ----------------------------------------------------------------------------
// FP8 K-split (lane-parallel) decode GEMV.
//
// `launch_fp8` above maps one *work-item* to one output element, so a lane
// walks a whole `[n_global, K]` weight row on its own. Two things follow from
// that mapping, and both cost real bandwidth on a kernel that does ~1 MAC per
// weight byte:
//
//   1. Weight loads are not coalesced. Lane `l` and lane `l+1` of a sub-group
//      read bytes that are `K` apart, so every 16-byte load instruction turns
//      into 16 scattered cache-line requests. The lines are eventually fully
//      consumed (each lane walks its own row sequentially), so no DRAM byte is
//      wasted -- but the memory controller sees `16 x resident sub-groups`
//      independent streams instead of one per thread, which is exactly the
//      access pattern DRAM row buffers handle worst.
//   2. The grid is small. Decode runs `total_tokens * N / 16` sub-groups; for
//      a batch-1 MiniMax-M2 step (8 tokens, N=1536) that is 768 SIMD16
//      threads, below the thread slots of a BMG-class GPU, so there are not
//      enough outstanding loads in flight to cover DRAM latency.
//
// This kernel transposes the lane mapping: a whole *sub-group* cooperates on
// one output element and the lanes split K. Lane `l` owns the `KSPLIT_CH`
// consecutive K elements at `l * KSPLIT_CH` inside each `KSPLIT_STEP`-wide
// K-tile, so per instruction the sub-group covers `KSPLIT_STEP` *contiguous*
// weight bytes (256 B -- four full cache lines) and `2 * KSPLIT_STEP`
// contiguous activation bytes. Each thread now walks a single sequential
// stream, and the grid grows by `SG_SIZE` (12288 sub-groups for the batch-1
// step above), which is what puts enough requests in flight. The per-lane
// partial sums are reduced once at the end with `reduce_over_group` -- a
// handful of shuffles per output element against `K` multiply-adds.
//
// The int4 fallback solves the same coalescing problem by repacking the packed
// weights into an N-tiled layout (`launch_int4_coalesced`), which costs a full
// pass over the weight tensor and is therefore gated on a token-count
// amortization heuristic. FP8 weights are one byte per element and already
// K-contiguous, so K-splitting the lane mapping gets the same coalescing with
// no repack, no scratch buffer and no extra kernel launch.
//
// Scale handling: a lane's chunk is `KSPLIT_CH` consecutive K elements
// starting at a multiple of `KSPLIT_CH`, so with `group_size` a power of two
// that is >= `KSPLIT_CH` (the shape gate below) the chunk always sits inside a
// single K-group and its scale index is `k0 >> log2(group_size)` -- one shift,
// no integer division in the hot loop. The scale is applied per chunk instead
// of once per group; that is one extra multiply per `KSPLIT_CH` elements and
// keeps the `Sigma a * (w * s) == s * Sigma a * w` fold exact-per-group,
// including the folded `2^-8` E4M3 word-decode bias.
// ----------------------------------------------------------------------------

// K elements a lane owns per step. 16 FP8 bytes = one 16-byte weight load and
// one `vec<ScalarT, 16>` (32-byte) activation load per lane, i.e. exactly the
// transactions `fp8_decode_chunk` already issues, so the alignment contract is
// unchanged.
constexpr int KSPLIT_CH = 16;
// K elements a sub-group covers per step: the contiguous span its 16 lanes
// read in one instruction.
constexpr int KSPLIT_STEP = SG_SIZE * KSPLIT_CH;
// Sub-groups per work-group. Each owns one output column, so a work-group
// covers `N_TILE` consecutive columns and `N % N_TILE == 0` (already required
// by every decode path) is enough to tile N exactly.
constexpr int KSPLIT_WG_SGS = N_TILE;

// ----------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_DECODE_FP8_KSPLIT` (default ON). When ON, the FP8
// scalar decode GEMV uses the K-split kernel above; setting the var to "0" /
// "false" / "off" / "no" (case-insensitive) forces the legacy per-lane-strided
// `launch_fp8`, for A/B comparison and regression escape. Re-read on every call
// so tests and benchmarks can toggle the path in-process.
// ----------------------------------------------------------------------------
inline bool moe_decode_fp8_ksplit_enabled() {
  return env_flag_enabled("ARK_MOE_DECODE_FP8_KSPLIT", true);  // default ON
}

// Shape gate for the K-split kernel. `group_size` must be a power of two of at
// least `KSPLIT_CH` so that (a) a lane's chunk never straddles a K-group
// boundary and (b) the group index is a shift rather than an integer division
// on the hot path. Every shipped FP8 quant config (32 / 64 / 128 / 256) passes;
// anything else keeps the legacy GEMV, which handles arbitrary group sizes.
inline bool moe_decode_fp8_ksplit_shape_ok(int N, int K, int group_size) {
  if (N % N_TILE != 0) return false;
  if (group_size < KSPLIT_CH) return false;
  if ((group_size & (group_size - 1)) != 0) return false;  // not a power of two
  if (K % group_size != 0) return false;
  return true;
}

template <typename ScalarT, bool IsE4M3, Fp8DecodeMode Mode>
void launch_fp8_ksplit(sycl::queue* q, const ScalarT* activations, const uint8_t* weights, const ScalarT* scales,
                       ScalarT* outputs, const int* expert_id_per_token, int total_tokens, int N, int K,
                       int group_size) {
  if (!moe_decode_fp8_ksplit_shape_ok(N, K, group_size)) {
    throw std::invalid_argument("moe_gemm_decode(fp8): K-split GEMV called on an unsupported shape");
  }
  if (total_tokens == 0) return;

  const int num_groups_k = K / group_size;
  int log2_group = 0;
  while ((1 << log2_group) < group_size) ++log2_group;
  // Undoes the exponent re-bias the word-native decode leaves behind (1.0f for
  // every other mode). Exact power of two, applied once per lane chunk.
  constexpr float kScaleBias = (Mode == Fp8DecodeMode::kWord) ? fp8_word_scale_bias<IsE4M3>() : 1.0f;

  // One sub-group per (token, output column); `KSPLIT_WG_SGS` of them per
  // work-group so the dispatcher sees `N / N_TILE` work-groups per token
  // instead of `N` single-sub-group ones.
  sycl::range<2> global{static_cast<size_t>(total_tokens), static_cast<size_t>(N) * SG_SIZE};
  sycl::range<2> local{1, static_cast<size_t>(KSPLIT_WG_SGS * SG_SIZE)};

  q->parallel_for<MoEDecodeKernelFP8KSplit<ScalarT, IsE4M3, Mode>>(
       sycl::nd_range<2>(global, local),
       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SG_SIZE)]] {
         const auto sg = it.get_sub_group();
         const int token = static_cast<int>(it.get_global_id(0));
         const int local_id = static_cast<int>(it.get_local_id(1));
         // The work-group is one row of `KSPLIT_WG_SGS * SG_SIZE` work-items, so
         // sub-group index and lane index are just the halves of the local id.
         const int lane = local_id % SG_SIZE;
         const int n_global = static_cast<int>(it.get_group(1)) * KSPLIT_WG_SGS + local_id / SG_SIZE;

         const int expert = expert_id_per_token[token];
         const ScalarT* act_row = activations + static_cast<size_t>(token) * K;
         const uint8_t* w_row =
             weights + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * K;
         const ScalarT* s_row =
             scales + (static_cast<size_t>(expert) * N + static_cast<size_t>(n_global)) * num_groups_k;

         // Each lane accumulates the scaled partial dot product of the chunks
         // it owns; `fp8_decode_chunk` keeps two partial accumulators per chunk
         // so the fp32 dependency chain stays broken.
         float acc = 0.0f;
         int k0 = lane * KSPLIT_CH;
         // Two chunks per iteration: their loads are independent, so the pair
         // doubles the number of weight requests a thread keeps in flight.
         for (; k0 + KSPLIT_STEP + KSPLIT_CH <= K; k0 += 2 * KSPLIT_STEP) {
           float a0 = 0.0f, a1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
           fp8_decode_chunk<ScalarT, IsE4M3, Mode, KSPLIT_CH>(act_row + k0, w_row + k0, a0, a1);
           fp8_decode_chunk<ScalarT, IsE4M3, Mode, KSPLIT_CH>(act_row + k0 + KSPLIT_STEP,
                                                              w_row + k0 + KSPLIT_STEP, b0, b1);
           const float s0 = static_cast<float>(s_row[k0 >> log2_group]) * kScaleBias;
           const float s1 = static_cast<float>(s_row[(k0 + KSPLIT_STEP) >> log2_group]) * kScaleBias;
           acc += (a0 + a1) * s0 + (b0 + b1) * s1;
         }
         // Remainder (at most one chunk per lane, plus the lanes whose chunk
         // falls past K when K < KSPLIT_STEP -- those simply contribute 0).
         for (; k0 < K; k0 += KSPLIT_STEP) {
           float p0 = 0.0f, p1 = 0.0f;
           fp8_decode_chunk<ScalarT, IsE4M3, Mode, KSPLIT_CH>(act_row + k0, w_row + k0, p0, p1);
           acc += (p0 + p1) * (static_cast<float>(s_row[k0 >> log2_group]) * kScaleBias);
         }

         const float total = sycl::reduce_over_group(sg, acc, sycl::plus<float>{});
         if (lane == 0) {
           outputs[static_cast<size_t>(token) * N + n_global] = static_cast<ScalarT>(total);
         }
       });
}

// Runtime -> compile-time bridge for the decode-mode selector. Keeps the
// `moe_gemm_decode` dispatch to one branch per (act dtype, format) instead of
// re-nesting the mode selection at every call site. The K-split vs legacy
// choice is made here as well, so all three decode modes run the same kernel
// structure and `word` / `lut` / `bits` stay comparable to one another.
template <typename ScalarT, bool IsE4M3, Fp8DecodeMode Mode>
void launch_fp8_dispatch(sycl::queue* q, const ScalarT* activations, const uint8_t* weights,
                         const ScalarT* scales, ScalarT* outputs, const int* expert_id_per_token,
                         int total_tokens, int N, int K, int group_size, bool ksplit) {
  if (ksplit) {
    launch_fp8_ksplit<ScalarT, IsE4M3, Mode>(q, activations, weights, scales, outputs, expert_id_per_token,
                                             total_tokens, N, K, group_size);
  } else {
    launch_fp8<ScalarT, IsE4M3, Mode>(q, activations, weights, scales, outputs, expert_id_per_token,
                                      total_tokens, N, K, group_size);
  }
}

template <typename ScalarT, bool IsE4M3>
void launch_fp8_by_mode(sycl::queue* q, const ScalarT* activations, const uint8_t* weights,
                        const ScalarT* scales, ScalarT* outputs, const int* expert_id_per_token,
                        int total_tokens, int N, int K, int group_size) {
  const bool ksplit = moe_decode_fp8_ksplit_enabled() && moe_decode_fp8_ksplit_shape_ok(N, K, group_size);
  switch (fp8_decode_mode()) {
    case Fp8DecodeMode::kLut:
      launch_fp8_dispatch<ScalarT, IsE4M3, Fp8DecodeMode::kLut>(
          q, activations, weights, scales, outputs, expert_id_per_token, total_tokens, N, K, group_size, ksplit);
      return;
    case Fp8DecodeMode::kBits:
      launch_fp8_dispatch<ScalarT, IsE4M3, Fp8DecodeMode::kBits>(
          q, activations, weights, scales, outputs, expert_id_per_token, total_tokens, N, K, group_size, ksplit);
      return;
    case Fp8DecodeMode::kWord:
    default:
      launch_fp8_dispatch<ScalarT, IsE4M3, Fp8DecodeMode::kWord>(
          q, activations, weights, scales, outputs, expert_id_per_token, total_tokens, N, K, group_size, ksplit);
      return;
  }
}

}  // namespace moe_decode_detail

// ----------------------------------------------------------------------------
// Release every device scratch buffer the int4 decode fallbacks hold (the
// N-tiled weight repack and the activation-sum table). Both are served from
// grow-on-demand per-queue slabs that are normally kept for the lifetime of the
// process; call this to hand the memory back, or to drop a repack cached under
// `ARK_MOE_DECODE_INT4_REPACK_CACHE` before the underlying weight buffer is
// freed. Safe to call at any time -- the next decode simply reallocates.
// ----------------------------------------------------------------------------
inline void moe_decode_release_scratch() {
  moe_decode_detail::int4_repack_pool().release_all();
  moe_decode_detail::act_group_sum_pool().release_all();
}

// ----------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_DECODE_DPAS_S4` (default ON). When ON, int4-sym
// (S4_CLIP, !asym) decode is routed to the dedicated decode-phase S4 DPAS
// grouped GEMM (`moe_dpas_s4::moe_decode_s4_dpas_per_group_dispatch`) instead
// of the scalar FMA GEMV (`launch_int4`). Mirroring vLLM-xpu-kernels'
// `w4a16` decode dispatch, this path selects the DPAS tile from the average
// tokens-per-expert (`A_avg_M`) ladder (`_m_8` -> `_m_16` -> `_m_32` -> wide),
// reusing the shared per-group mainloop's 2D VNNI block load + register-resident
// per-N scale. It reads the same `[E, N, K/2]` packed weights + `[E, N, K/group]`
// scales, so no repack is needed. (`ARK_MOE_DECODE_S4_DPAS_M8=1` forces the
// legacy hard-pinned 8-row tile for A/B comparison; the two are numerically
// identical.)
//
// Setting the var to "0" / "false" / "off" / "no" (case-insensitive) forces
// the legacy scalar GEMV, for A/B comparison and regression escape. Asym
// weights, shapes that fail the DPAS shape gate, and batches that fail the
// tokens-per-expert occupancy gate (`moe_decode_dpas_s4_occupancy_ok`, see
// below -- this is what keeps real decode batches on the fast scalar GEMV)
// always fall back to the scalar path regardless of this flag. Re-read on every call so tests /
// benchmarks can toggle the path in-process.
// ----------------------------------------------------------------------------
inline bool moe_decode_dpas_s4_enabled() {
  return moe_decode_detail::env_flag_enabled("ARK_MOE_DECODE_DPAS_S4", true);  // default ON
}

// ----------------------------------------------------------------------------
// Occupancy gate for the int4-sym S4 DPAS decode path.
//
// The DPAS grouped GEMM pays off only when its M tile is actually filled: the
// smallest tile (`dpas_w4a16_policy_m_8`) processes 8 token rows per expert, so
// with fewer than 8 tokens routed to an expert on average the tile is mostly
// padding and the (bandwidth-bound) packed weights are streamed for rows that
// contribute nothing. Real decode batches are exactly that regime -- e.g.
// MiniMax-M2 decode is 8 tokens (bs1) or 256 tokens (bs32) spread over 192
// experts, i.e. 0.04-1.3 tokens per expert -- and there the shared scalar GEMV
// (`launch_int4`, the very kernel the *asym* path uses, where sym is just
// `Asym=false`) is up to ~3x faster because it reads each weight byte exactly
// once per active token with no tile padding.
//
// So route int4-sym decode through the same scalar GEMV as int4-asym unless the
// batch has at least one full 8-row tile of tokens per expert on average.
// `ARK_MOE_DECODE_DPAS_S4_MIN_TPE` overrides the tokens-per-expert threshold;
// "0" disables the gate (always take DPAS when the shape gate allows), which is
// what the accuracy tests use to exercise the DPAS kernel on tiny shapes.
// ----------------------------------------------------------------------------
inline bool moe_decode_dpas_s4_occupancy_ok(int total_tokens, int num_experts) {
  if (num_experts <= 0) return true;
  long long min_tokens_per_expert = 8;  // rows in `dpas_w4a16_policy_m_8`
  const char* env = std::getenv("ARK_MOE_DECODE_DPAS_S4_MIN_TPE");
  if (env != nullptr) {
    char* end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 0) min_tokens_per_expert = v;
  }
  if (min_tokens_per_expert == 0) return true;
  return static_cast<long long>(total_tokens) >= min_tokens_per_expert * static_cast<long long>(num_experts);
}

// ----------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_DECODE_DPAS_FP8` (default ON). When ON, FP8
// (E4M3 / E5M2, sym) decode is routed to the decode-phase FP8 DPAS grouped
// GEMM (`moe_dpas_fp8::moe_decode_fp8_dpas_per_group_dispatch`) instead of the
// scalar FMA GEMV (`launch_fp8`). This is the FP8 twin of
// `ARK_MOE_DECODE_DPAS_S4`: same `[E, N, K]` FP8 bytes and `[E, N, K/group]`
// scales, no repack, tile picked from the `A_avg_M` ladder.
//
// Setting the var to "0" / "false" / "off" / "no" (case-insensitive) forces the
// scalar GEMV, for A/B comparison and regression escape. Shapes that fail the
// DPAS shape gate and batches that fail the tokens-per-expert occupancy gate
// (`moe_decode_dpas_fp8_occupancy_ok`, below -- this is what keeps real decode
// batches on the fast scalar GEMV) always fall back to the scalar path
// regardless of this flag. Re-read on every call so tests / benchmarks can
// toggle the path in-process.
// ----------------------------------------------------------------------------
inline bool moe_decode_dpas_fp8_enabled() {
  return moe_decode_detail::env_flag_enabled("ARK_MOE_DECODE_DPAS_FP8", true);  // default ON
}

// ----------------------------------------------------------------------------
// Occupancy gate for the FP8 DPAS decode path. Identical reasoning to
// `moe_decode_dpas_s4_occupancy_ok`: the smallest DPAS tile the decode
// dispatch can pick (`dpas_w4a16_policy_m_8`) processes 8 token rows per
// expert, so below 8 tokens per expert on average the tile is mostly padding
// and the bandwidth-bound FP8 weights get streamed for rows that contribute
// nothing -- exactly the regime real decode batches live in. Above that the
// DPAS pipeline wins, so the threshold is where the two cross.
//
// `ARK_MOE_DECODE_DPAS_FP8_MIN_TPE` overrides the tokens-per-expert threshold;
// "0" disables the gate (always take DPAS when the shape gate allows), which is
// what the accuracy tests use to exercise the DPAS kernel on tiny shapes.
// ----------------------------------------------------------------------------
inline bool moe_decode_dpas_fp8_occupancy_ok(int total_tokens, int num_experts) {
  if (num_experts <= 0) return true;
  long long min_tokens_per_expert = 8;  // rows in `dpas_w4a16_policy_m_8`
  const char* env = std::getenv("ARK_MOE_DECODE_DPAS_FP8_MIN_TPE");
  if (env != nullptr) {
    char* end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 0) min_tokens_per_expert = v;
  }
  if (min_tokens_per_expert == 0) return true;
  return static_cast<long long>(total_tokens) >= min_tokens_per_expert * static_cast<long long>(num_experts);
}

// ----------------------------------------------------------------------------
// Env-flag helper -- `ARK_MOE_DECODE_COALESCE_INT4` (default ON). When ON, the
// int4 scalar-GEMV fallback (asym, or sym with the DPAS path disabled / shape
// or occupancy gate miss) uses `launch_int4_coalesced`, which repacks the weights on-device
// into an N-tiled layout so sub-group weight loads are coalesced. Setting the
// var to "0" / "false" / "off" / "no" (case-insensitive) forces the legacy
// per-lane-strided `launch_int4`, for A/B comparison and regression escape.
// Re-read on every call so tests / benchmarks can toggle it in-process.
// ----------------------------------------------------------------------------
inline bool moe_decode_coalesce_int4_enabled() {
  return moe_decode_detail::env_flag_enabled("ARK_MOE_DECODE_COALESCE_INT4", true);  // default ON
}

// ----------------------------------------------------------------------------
// Amortization gate for the coalesced int4 decode kernel. The coalesced path
// repacks the *entire* weight tensor for all `num_experts` on every call
// (cost proportional to num_experts * N * K/2) before running the GEMV. That
// one-time repack only pays off when a work-group reuses each repacked weight
// tile across many tokens -- i.e. when there are enough active tokens relative
// to the number of experts. For tiny decode batches (e.g. 8 tokens spread
// across 192 experts) the repack dominates and the coalesced kernel is far
// slower than the per-lane-strided `launch_int4`, which reads the weights in
// place with no repack. Require at least one full TOKEN_BLOCK worth of tokens
// per expert on average before coalescing; otherwise fall back to `launch_int4`.
// `ARK_MOE_DECODE_COALESCE_MIN_TOKENS` overrides the threshold (tokens per
// expert scaled by TOKEN_BLOCK); "0" disables the gate (always coalesce).
// ----------------------------------------------------------------------------
inline bool moe_decode_coalesce_int4_amortized(int total_tokens, int num_experts) {
  if (num_experts <= 0) return true;
  long long min_tokens = static_cast<long long>(num_experts) * moe_decode_detail::TOKEN_BLOCK;
  const char* env = std::getenv("ARK_MOE_DECODE_COALESCE_MIN_TOKENS");
  if (env != nullptr) {
    char* end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end != env && v >= 0) min_tokens = v;
  }
  return static_cast<long long>(total_tokens) >= min_tokens;
}

// ----------------------------------------------------------------------------
//
// weight_dtype:
//   BTLA_DTYPE::F16  / BF16       : weights stored as [E, N, K] in matching
//                                   floating dtype, no scales/zeros needed
//   BTLA_DTYPE::S8                : int8 weights [E, N, K] (uint8 buffer,
//                                   interpreted as signed when asym==false,
//                                   unsigned with zero-points when asym==true)
//   BTLA_DTYPE::S4_CLIP           : packed int4 weights [E, N, K/2] (uint8),
//                                   scales [E, N, K/group_size] in act dtype,
//                                   zeros optional (asym==true requires it).
//                                   Sym weights are routed to the shared
//                                   per-group S4 DPAS grouped GEMM only when
//                                   the batch fills its M tile (>= 8 tokens per
//                                   expert on average, `ARK_MOE_DECODE_DPAS_S4`
//                                   default ON); asym, a disabled flag, a
//                                   shape-gate miss, or a decode-sized batch
//                                   uses the shared scalar GEMV.
//   BTLA_DTYPE::S2_CLIP           : packed int2 weights [E, N, K/4] (uint8),
//                                   4 values per byte, sym/asym like int4
//   BTLA_DTYPE::F8_E4M3 / F8_E5M2 : FP8 weights [E, N, K] (uint8 buffer),
//                                   group-wise scales, no zero-points. Routed
//                                   to the per-group FP8 DPAS grouped GEMM only
//                                   when the batch fills its M tile (>= 8
//                                   tokens per expert on average,
//                                   `ARK_MOE_DECODE_DPAS_FP8` default ON); a
//                                   disabled flag, a shape-gate miss, or a
//                                   decode-sized batch uses the scalar GEMV.
// act_dtype: F16 or BF16 (must match scales/outputs dtype)
// ----------------------------------------------------------------------------
inline void moe_gemm_decode(sycl::queue* q, void* activations, void* weights, void* scales, void* zeros,
                            void* outputs, int* expert_id_per_token_buf, BTLA_DTYPE act_dtype,
                            BTLA_DTYPE weight_dtype, int N, int K, int group_size, int* num_tokens_per_expert,
                            int num_experts, int total_tokens, bool asym) {
  // The S4-sym and FP8 DPAS fast paths consume `num_tokens_per_expert` directly
  // and never read `expert_id_per_token_buf`. Skipping the fill on those paths
  // removes an extra device-timeline kernel launch from the decode hot path;
  // every other path (fp, int8, int2, and the scalar int4 / fp8 fallbacks)
  // still needs the per-token expert mapping.
  const bool s4_dpas_fastpath = weight_dtype == BTLA_DTYPE::S4_CLIP && !asym && moe_decode_dpas_s4_enabled() &&
                                moe_decode_dpas_s4_occupancy_ok(total_tokens, num_experts) &&
                                moe_dpas_s4::moe_prefill_dpas_s4_pergroup_shape_ok(N, K, group_size);
  const bool fp8_dpas_fastpath = (weight_dtype == BTLA_DTYPE::F8_E4M3 || weight_dtype == BTLA_DTYPE::F8_E5M2) &&
                                 !asym && moe_decode_dpas_fp8_enabled() &&
                                 moe_decode_dpas_fp8_occupancy_ok(total_tokens, num_experts) &&
                                 moe_dpas_fp8::moe_prefill_dpas_fp8_pergroup_shape_ok(N, K, group_size);
  if (!s4_dpas_fastpath && !fp8_dpas_fastpath) {
    moe_decode_detail::fill_expert_id_per_token(q, expert_id_per_token_buf, num_tokens_per_expert, num_experts,
                                                total_tokens);
  }

  if (weight_dtype == BTLA_DTYPE::F16 || weight_dtype == BTLA_DTYPE::BF16) {
    if (weight_dtype != act_dtype) {
      throw std::invalid_argument("moe_gemm_decode: unquantized weight_dtype must match act_dtype");
    }
    if (act_dtype == BTLA_DTYPE::F16) {
      moe_decode_detail::launch_fp<sycl::half>(q, static_cast<const sycl::half*>(activations),
                                               static_cast<const sycl::half*>(weights),
                                               static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
                                               total_tokens, N, K);
    } else {
      moe_decode_detail::launch_fp<sycl::ext::oneapi::bfloat16>(
          q, static_cast<const sycl::ext::oneapi::bfloat16*>(activations),
          static_cast<const sycl::ext::oneapi::bfloat16*>(weights),
          static_cast<sycl::ext::oneapi::bfloat16*>(outputs), expert_id_per_token_buf, total_tokens, N, K);
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::S4_CLIP) {
    if (act_dtype != BTLA_DTYPE::F16 && act_dtype != BTLA_DTYPE::BF16) {
      throw std::invalid_argument("moe_gemm_decode(int4): act_dtype must be FP16 or BF16");
    }
    // Fast path: sym int4 through the shared per-group S4 DPAS grouped GEMM.
    // Falls back to the scalar GEMV for asym weights (DPAS S4 is sym-only),
    // when the env flag is off, when the batch is too small to fill the DPAS M
    // tile (the usual decode case -- sym then runs the exact same
    // `launch_int4*` kernel as asym, with `Asym=false`), or when the shape gate
    // rejects the tile geometry (e.g. N%64!=0, K%32!=0, unsupported
    // group_size). Reuses the
    // `s4_dpas_fastpath` predicate computed above (which also gated the
    // `fill_expert_id_per_token` skip) so the two decisions cannot diverge.
    if (s4_dpas_fastpath) {
      if (act_dtype == BTLA_DTYPE::F16) {
        moe_dpas_s4::moe_decode_s4_dpas_per_group_dispatch<sycl::half>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), num_tokens_per_expert,
            num_experts, N, K, group_size, total_tokens);
      } else {
        using BF = sycl::ext::oneapi::bfloat16;
        moe_dpas_s4::moe_decode_s4_dpas_per_group_dispatch<BF>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<BF*>(outputs), num_tokens_per_expert, num_experts, N, K,
            group_size, total_tokens);
      }
      return;
    }
    // Scalar FMA GEMV fallback (asym, flag off, or shape gate miss). By
    // default this uses the coalesced-load variant, which repacks the weights
    // on-device so sub-group loads are contiguous; `ARK_MOE_DECODE_COALESCE_INT4=0`
    // forces the legacy per-lane-strided kernel.
    const bool coalesce = moe_decode_coalesce_int4_enabled() &&
                          moe_decode_coalesce_int4_amortized(total_tokens, num_experts);
    if (act_dtype == BTLA_DTYPE::F16) {
      if (asym) {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<sycl::half, true>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size,
              num_experts);
        } else {
          moe_decode_detail::launch_int4<sycl::half, true>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      } else {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<sycl::half, false>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size,
              num_experts);
        } else {
          moe_decode_detail::launch_int4<sycl::half, false>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
              static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      }
    } else {
      using BF = sycl::ext::oneapi::bfloat16;
      if (asym) {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<BF, true>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size, num_experts);
        } else {
          moe_decode_detail::launch_int4<BF, true>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      } else {
        if (coalesce) {
          moe_decode_detail::launch_int4_coalesced<BF, false>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size, num_experts);
        } else {
          moe_decode_detail::launch_int4<BF, false>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
              expert_id_per_token_buf, total_tokens, N, K, group_size);
        }
      }
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::S8) {
    if (act_dtype == BTLA_DTYPE::F16) {
      if (asym) {
        moe_decode_detail::launch_int8<sycl::half, true>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int8<sycl::half, false>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else if (act_dtype == BTLA_DTYPE::BF16) {
      using BF = sycl::ext::oneapi::bfloat16;
      if (asym) {
        moe_decode_detail::launch_int8<BF, true>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int8<BF, false>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else {
      throw std::invalid_argument("moe_gemm_decode(int8): act_dtype must be FP16 or BF16");
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::S2_CLIP) {
    if (act_dtype == BTLA_DTYPE::F16) {
      if (asym) {
        moe_decode_detail::launch_int2<sycl::half, true>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int2<sycl::half, false>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<const sycl::half*>(zeros),
            static_cast<sycl::half*>(outputs), expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else if (act_dtype == BTLA_DTYPE::BF16) {
      using BF = sycl::ext::oneapi::bfloat16;
      if (asym) {
        moe_decode_detail::launch_int2<BF, true>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_int2<BF, false>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<const BF*>(zeros), static_cast<BF*>(outputs),
            expert_id_per_token_buf, total_tokens, N, K, group_size);
      }
    } else {
      throw std::invalid_argument("moe_gemm_decode(int2): act_dtype must be FP16 or BF16");
    }
    return;
  }

  if (weight_dtype == BTLA_DTYPE::F8_E4M3 || weight_dtype == BTLA_DTYPE::F8_E5M2) {
    if (asym) {
      throw std::invalid_argument("moe_gemm_decode(fp8): asym mode is not supported");
    }
    if (act_dtype != BTLA_DTYPE::F16 && act_dtype != BTLA_DTYPE::BF16) {
      throw std::invalid_argument("moe_gemm_decode(fp8): act_dtype must be FP16 or BF16");
    }
    const bool is_e4m3 = (weight_dtype == BTLA_DTYPE::F8_E4M3);
    // Fast path: FP8 through the decode-phase per-group DPAS grouped GEMM.
    // Falls back to the scalar GEMV when the env flag is off, when the batch is
    // too small to fill the DPAS M tile (the usual decode case), or when the
    // shape gate rejects the tile geometry (e.g. N%64!=0, K%32!=0, unsupported
    // group_size). Reuses the `fp8_dpas_fastpath` predicate computed above
    // (which also gated the `fill_expert_id_per_token` skip) so the two
    // decisions cannot diverge.
    if (fp8_dpas_fastpath) {
      if (act_dtype == BTLA_DTYPE::F16) {
        if (is_e4m3) {
          moe_dpas_fp8::moe_decode_fp8_dpas_per_group_dispatch<sycl::half, true>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), num_tokens_per_expert,
              num_experts, N, K, group_size, total_tokens);
        } else {
          moe_dpas_fp8::moe_decode_fp8_dpas_per_group_dispatch<sycl::half, false>(
              q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), num_tokens_per_expert,
              num_experts, N, K, group_size, total_tokens);
        }
      } else {
        using BF = sycl::ext::oneapi::bfloat16;
        if (is_e4m3) {
          moe_dpas_fp8::moe_decode_fp8_dpas_per_group_dispatch<BF, true>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<BF*>(outputs), num_tokens_per_expert, num_experts, N, K,
              group_size, total_tokens);
        } else {
          moe_dpas_fp8::moe_decode_fp8_dpas_per_group_dispatch<BF, false>(
              q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
              static_cast<const BF*>(scales), static_cast<BF*>(outputs), num_tokens_per_expert, num_experts, N, K,
              group_size, total_tokens);
        }
      }
      return;
    }
    if (act_dtype == BTLA_DTYPE::F16) {
      if (is_e4m3) {
        moe_decode_detail::launch_fp8_by_mode<sycl::half, true>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
            total_tokens, N, K, group_size);
      } else {
        moe_decode_detail::launch_fp8_by_mode<sycl::half, false>(
            q, static_cast<const sycl::half*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const sycl::half*>(scales), static_cast<sycl::half*>(outputs), expert_id_per_token_buf,
            total_tokens, N, K, group_size);
      }
    } else {
      using BF = sycl::ext::oneapi::bfloat16;
      if (is_e4m3) {
        moe_decode_detail::launch_fp8_by_mode<BF, true>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
            group_size);
      } else {
        moe_decode_detail::launch_fp8_by_mode<BF, false>(
            q, static_cast<const BF*>(activations), static_cast<const uint8_t*>(weights),
            static_cast<const BF*>(scales), static_cast<BF*>(outputs), expert_id_per_token_buf, total_tokens, N, K,
            group_size);
      }
    }
    return;
  }

  throw std::invalid_argument(
      "moe_gemm_decode: unsupported weight_dtype (supported: F16, BF16, S8, S4_CLIP, S2_CLIP, F8_E4M3, F8_E5M2)");
}

}  // namespace ark

#endif  // ARK_XPU && ARK_SYCL_TLA
