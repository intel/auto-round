//
// Copyright (c) 2026 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Activation fused kernel: FP16/BF16 activation -> 32-point normalized
// Hadamard transform -> MXFP4 quantization (packed FP4 codes + E8M0 scales).
//
// MVP contract (see xpu_mxfp4_hadamard_design_revised.md):
//   * hadamard_dim == group_size == 32, K % 32 == 0
//   * H is a *normalized* 32x32 Hadamard matrix (already contains 1/sqrt(32))
//   * y      = reshape(x, [-1, 32]) @ H            (FP32 accumulation)
//   * amax   = max(|y_g|) per 32-element group
//   * e8m0   = clamp(floor(log2(amax)) - 2 + 127, 0, 254)
//   * q      = y * 2^-(e8m0 - 127)
//   * code   = signbit(q) << 3 | e2m1_magnitude_index(|q|)
//   * canonical zero: magnitude index 0 always encodes as 0x0, never 0x8
//   * two codes per byte, even element in the low nibble
//   * all-zero group -> e8m0 = 0 and all codes = 0
//
// FP32 transform contract: two paths exist and each is bit-exact against its own
// PyTorch reference.
//
//   FWHT (default): 5 butterfly stages over lane ^ (1 << stage), followed by a
//   single multiply by H[0][0] == 1/sqrt(32). Only adds and subtracts occur, so
//   nothing can contract into an FMA and the order is fixed by the stage index.
//   This is the performance path -- the O(D^2) dot product below is compute
//   bound well below streaming-copy bandwidth on this device, the butterfly is
//   memory bound.
//
//   Path A (non-Sylvester matrix only): sums over j in increasing order with a
//   separate FP32 rounding after each multiply and each add (no FMA, no
//   reassociation), enforced by #pragma clang fp contract(off).
//
// The two paths round differently and are deliberately *not* bit-exact against
// each other; the wrapper picks one and the reference mirrors that choice.

#pragma once

#include <cstdint>

#if defined(ARK_XPU)
#include <sycl/sycl.hpp>

namespace ark {

class XpuMxfp4Hadamard {
 public:
  static constexpr int kHadamardDim = 32;
  static constexpr int kGroupSize = 32;
  static constexpr int kSubGroupSize = 32;
  static constexpr int kWorkGroupSize = 256;
  // log2(32): number of FWHT butterfly stages.
  static constexpr int kNumFwhtStages = 5;

  // FP4 (E2M1) magnitude levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
  // Thresholds and boundary comparison operators are taken verbatim from the
  // PyTorch reference (auto_round_extension/vllm_ext/fp4_utils.py::cast_to_fp4)
  // so that the kernel is bit-exact with it.
  static inline int e2m1_magnitude_index(float a) {
    if (a <= 0.25f) return 0;
    if (a < 0.75f) return 1;
    if (a <= 1.25f) return 2;
    if (a < 1.75f) return 3;
    if (a <= 2.5f) return 4;
    if (a < 3.5f) return 5;
    if (a <= 5.0f) return 6;
    return 7;
  }

  // Fast path: one work-item owns one full 32-element group.
  //
  // The original design gave each *lane* one element, so a 32-lane sub-group
  // handled a single group: 134M work-items each moving 2 bytes, and 16 separate
  // 1-byte stores per group. Byte-granularity stores force read-modify-write on
  // cache lines and the per-work-item overhead dwarfs the actual work, which
  // pinned the kernel near 33 GB/s regardless of how cheap the transform was --
  // replacing the O(D^2) dot product with the butterfly changed nothing.
  //
  // Here a work-item loads its whole group as 16-byte vectors, runs the
  // butterflies entirely in registers (no sub-group shuffles at all), and emits
  // the 16 packed code bytes as a single aligned 16-byte store. Because the
  // activation is contiguous, group g occupies exactly x[g*32 .. g*32+31],
  // codes bytes [g*16 .. g*16+15] and scale byte g.
  //
  // The group is read straight from global memory with two 16-byte vector loads
  // per work-item. An earlier revision staged the work-group's whole slab
  // through SLM first, on the theory that a work-item reading 64 contiguous
  // bytes makes neighbouring work-items stride 64B apart and therefore
  // uncoalesced. Measured on Arc Pro B60 that staging *cost* 2.3x: the transpose
  // read tile[lt * 32 + i] has every lane hitting the same SLM bank for a given
  // i (stride 64B = 16 dwords = the bank count), so it serializes 16-32 ways;
  // padding the slot to 34 elements removes the conflict but the extra SLM
  // round trip plus the barrier still leaves it slower than not staging at all.
  // The direct path is fine because a 64B-per-work-item stride is exactly one
  // cache line per work-item: the loads are already at full line granularity, so
  // there is nothing for a staging buffer to coalesce. Direct + vector loads
  // measures 1.01-1.03x the streaming-copy baseline versus 0.31-0.43x staged.
  //
  // Bit-exactness is unaffected: the butterfly order is identical to the SLM
  // version, to the sub-group version and to fwht_transform_reference.
  template <typename T>
  static void fwht_quant_per_item(sycl::queue* q, const T* x, const float* hadamard, uint8_t* out_codes,
                                  uint8_t* out_scale, int64_t total_groups) {
    const int64_t num_wg = (total_groups + kWorkGroupSize - 1) / kWorkGroupSize;
    const size_t global_size = static_cast<size_t>(num_wg) * kWorkGroupSize;
    // 16-byte vector loads: 8 halves per chunk, 4 chunks per 32-element group.
    constexpr int kLoadVecElems = 8;
    constexpr int kLoadVecCount = kGroupSize / kLoadVecElems;

    q->parallel_for(sycl::nd_range<1>(global_size, kWorkGroupSize), [=](sycl::nd_item<1> item) {
      const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
      if (gid >= total_groups) {
        return;
      }

      // hadamard[0] == H[0][0] == 1/sqrt(32); applied before the butterflies so
      // intermediates stay bounded by sqrt(32)*max|x| (see the header comment).
      const float norm = hadamard[0];

      // x + gid * 32 is 64-byte aligned for T = half/bfloat16, so each chunk
      // load is an aligned 16-byte access.
      const auto* src = reinterpret_cast<const sycl::vec<T, kLoadVecElems>*>(x + gid * kGroupSize);

      float v[kGroupSize];
#pragma unroll
      for (int c = 0; c < kLoadVecCount; ++c) {
        const sycl::vec<T, kLoadVecElems> chunk = src[c];
#pragma unroll
        for (int i = 0; i < kLoadVecElems; ++i) {
          v[c * kLoadVecElems + i] = static_cast<float>(chunk[i]) * norm;
        }
      }

#pragma unroll
      for (int stage = 0; stage < kNumFwhtStages; ++stage) {
        const int h = 1 << stage;
#pragma unroll
        for (int i = 0; i < kGroupSize; ++i) {
          if ((i & h) == 0) {
            const float a = v[i];
            const float b = v[i ^ h];
            v[i] = a + b;
            v[i ^ h] = a - b;
          }
        }
      }

      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kGroupSize; ++i) {
        amax = sycl::fmax(amax, sycl::fabs(v[i]));
      }

      uint8_t e8m0 = 0;
      int exp_shift = 0;
      if (amax > 0.0f) {
        int biased = sycl::ilogb(amax) - 2 + 127;
        biased = biased < 0 ? 0 : (biased > 254 ? 254 : biased);
        e8m0 = static_cast<uint8_t>(biased);
        exp_shift = biased - 127;
      }

      // Pack 32 codes into 16 bytes, emitted as four 32-bit words. gid*16 is
      // 16-byte aligned, so this is a single aligned vector store.
      sycl::vec<uint32_t, 4> packed(0u);
      if (amax > 0.0f) {
#pragma unroll
        for (int i = 0; i < kGroupSize; ++i) {
          const float qv = sycl::ldexp(v[i], -exp_shift);
          const int idx = e2m1_magnitude_index(sycl::fabs(qv));
          // Canonical zero: never emit 0x8 (negative zero). A value that rounds
          // to magnitude 0 may carry either sign depending on FP32 rounding
          // residue and on flush-to-zero, so the sign is dropped.
          const int code = (idx == 0) ? 0 : ((sycl::signbit(qv) ? 8 : 0) | idx);
          // Even element -> low nibble of its byte.
          packed[i >> 3] |= static_cast<uint32_t>(code & 0xF) << ((i & 7) * 4);
        }
      }

      auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(out_codes + gid * (kGroupSize / 2));
      *dst = packed;
      out_scale[gid] = e8m0;
    });
  }

  // in:  x        [num_rows, k]        (T = sycl::half or bfloat16)
  //      hadamard [32, 32]             (FP32, row major, normalized)
  // out: codes    [num_rows, k / 2]    (uint8, two FP4 codes per byte)
  //      scale    [num_rows, k / 32]   (uint8, one E8M0 exponent per group)
  //
  // Path A fallback for a caller-supplied non-Sylvester matrix. Bit-exact
  // against hadamard_transform_reference; deliberately *not* bit-exact against
  // the FWHT path above, because a butterfly network and a 32-term dot product
  // round differently.
  template <typename T>
  static void mxfp4_hadamard_quant_impl(sycl::queue* q, const T* x, const float* hadamard, uint8_t* out_codes,
                                        uint8_t* out_scale, int64_t num_rows, int64_t k) {
    constexpr int groups_per_wg = kWorkGroupSize / kSubGroupSize;
    const int64_t groups_per_row = k / kGroupSize;
    const int64_t total_groups = num_rows * groups_per_row;
    if (total_groups <= 0) {
      return;
    }
    const int64_t num_wg = (total_groups + groups_per_wg - 1) / groups_per_wg;
    const size_t global_size = static_cast<size_t>(num_wg) * kWorkGroupSize;

    q->parallel_for(sycl::nd_range<1>(global_size, kWorkGroupSize),
                    [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(kSubGroupSize)]] {
                      auto sg = item.get_sub_group();
                      const int lane = static_cast<int>(sg.get_local_id()[0]);
                      const int64_t group_id = static_cast<int64_t>(item.get_group(0)) * groups_per_wg +
                                               static_cast<int64_t>(sg.get_group_id()[0]);
                      // Tail work-groups: the whole sub-group exits together, so the
                      // sub-group collectives below stay converged.
                      if (group_id >= total_groups) {
                        return;
                      }

                      const int64_t row = group_id / groups_per_row;
                      const int64_t group_in_row = group_id % groups_per_row;
                      const int64_t base = row * k + group_in_row * kGroupSize;

                      // Path A: generic 32x32 matrix multiply. Lane i owns column i of
                      // H, values of x are broadcast one by one inside the sub-group.
                      //
                      // The Path A accumulation contract (see mxfp4_hadamard.py)
                      // requires increasing j, no reassociation and a separate FP32
                      // rounding after the multiply and after the add. Contracting
                      // into an FMA would change near-threshold elements and break
                      // bit-exactness with the reference, so it is disabled here.
                      const float xv = static_cast<float>(x[base + lane]);
                      float acc = 0.0f;
#pragma unroll
                      for (int j = 0; j < kHadamardDim; ++j) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
                        const float xj = sycl::select_from_group(sg, xv, j);
                        acc += xj * hadamard[j * kHadamardDim + lane];
                      }

                      const float amax = sycl::reduce_over_group(sg, sycl::fabs(acc), sycl::maximum<float>{});

                      uint8_t e8m0 = 0;
                      int code = 0;
                      if (amax > 0.0f) {
                        // floor(log2(amax)) is exact through ilogb, including for
                        // exact powers of two and subnormal inputs.
                        int biased = sycl::ilogb(amax) - 2 + 127;
                        biased = biased < 0 ? 0 : (biased > 254 ? 254 : biased);
                        e8m0 = static_cast<uint8_t>(biased);
                        const float qv = sycl::ldexp(acc, -(biased - 127));
                        const int idx = e2m1_magnitude_index(sycl::fabs(qv));
                        const int sign = sycl::signbit(qv) ? 1 : 0;
                        // Canonical zero: never emit 0x8 (negative zero). A value
                        // that rounds to magnitude 0 may carry either sign
                        // depending on FP32 accumulation residue and on whether
                        // the device flushes subnormals, so the sign is dropped.
                        code = (idx == 0) ? 0 : ((sign << 3) | idx);
                      }

                      // Even lane keeps the low nibble, its odd neighbour the high one.
                      const int partner = ((lane & 1) == 0) ? (lane + 1) : lane;
                      const int hi_code = sycl::select_from_group(sg, code, partner);
                      if ((lane & 1) == 0) {
                        const int64_t byte_idx = (base + lane) >> 1;
                        out_codes[byte_idx] = static_cast<uint8_t>((code & 0xF) | ((hi_code & 0xF) << 4));
                      }
                      if (lane == 0) {
                        out_scale[row * groups_per_row + group_in_row] = e8m0;
                      }
                    });
  }

  template <typename T>
  static void mxfp4_hadamard_quant(sycl::queue* q, const T* x, const float* hadamard, uint8_t* out_codes,
                                   uint8_t* out_scale, int64_t num_rows, int64_t k, bool use_fwht) {
    if (use_fwht) {
      const int64_t total_groups = num_rows * (k / kGroupSize);
      if (total_groups > 0) {
        fwht_quant_per_item<T>(q, x, hadamard, out_codes, out_scale, total_groups);
      }
    } else {
      mxfp4_hadamard_quant_impl<T>(q, x, hadamard, out_codes, out_scale, num_rows, k);
    }
  }
};

}  // namespace ark

#endif  // ARK_XPU
