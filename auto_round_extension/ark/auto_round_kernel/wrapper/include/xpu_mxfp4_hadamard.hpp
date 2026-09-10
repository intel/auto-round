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
// Activation fused kernel: FP16/BF16 activation -> normalized Hadamard
// transform (D = 32, 64, 128, 256 or 512) -> MXFP4 quantization (packed FP4
// codes + E8M0 scales).
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
//   FWHT (default): log2(D) butterfly stages over lane ^ (1 << stage), plus a
//   single multiply by H[0][0] == 1/sqrt(D). The butterflies are pure adds and
//   subtracts, so their order is fixed by the stage index; the only contractable
//   operation is that normalization multiply, and *where* it sits decides
//   whether the kernel is reproducible. See the normalization note above
//   fwht_quant_cooperative. This is the performance path: the O(D^2) dot product
//   below is compute bound well below streaming-copy bandwidth on this device,
//   the butterfly is memory bound.
//
//   Path A (non-Sylvester matrix only): sums over j in increasing order with a
//   separate FP32 rounding after each multiply and each add (no FMA, no
//   reassociation), enforced by #pragma clang fp contract(off).
//
// The two paths round differently and are deliberately *not* bit-exact against
// each other; the wrapper picks one and the reference mirrors that choice.
//
// D > 32 (e.g. 128, the attention head dim). Only the FWHT path exists: H is
// required to be the normalized Sylvester matrix. The quantization group stays
// at 32, so a D-element row produces D/32 independent MXFP4 groups. See
// fwht_quant_cooperative below for the two-level (lane-local + cross-lane)
// decomposition and why a single work-item per row is not viable.
//
// Baseline modes. Two controlled ablations share the fused kernel's exact
// memory traffic and item mapping, so their bandwidths are directly
// comparable (see test/README_HMT_QUANT_ONLY_BASELINE.md):
//   * quant-only  -- drops the Hadamard transform, keeps the quantization.
//   * stream-only -- drops the transform *and* the quantization math, keeping
//                    only the loads, the packing shape and the stores. This is
//                    the traffic-matched roofline: the highest bandwidth any
//                    kernel with this read:write ratio and this access pattern
//                    can reach on the device, and therefore the only
//                    self-calibrating denominator for the fused kernel.

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

  // Cooperative FWHT for D > 32: L lanes per row, each owning one 32-element
  // quantization group, so D = 32 * L. The sub-group is pinned so that the L
  // lanes of a row always land in one sub-group, which is what makes the
  // cross-lane butterfly shuffles legal: sub-groups start at multiples of their
  // size and rows are L-lane aligned, so any size divisible by L is correct.
  //
  // 32 rather than 16. Both are legal, but requesting 16 forces the kernel to
  // be compiled SIMD16, which needs twice as many instructions to retire the
  // same work as SIMD32 and roughly halves ALU throughput. This kernel has
  // little slack, so that is not free: measured 207 GB/s at SIMD16 against
  // 357 GB/s at SIMD32 on Arc Pro B60, same code otherwise.
  static constexpr int kSubGroupSizeCoop = 32;
  // log2(32): butterfly stages that stay inside a lane's own 32 elements. The
  // remaining log2(L) stages cross lanes.
  static constexpr int kNumFwhtStagesLocal = 5;

  // Largest supported cooperative fan-out. L must divide the sub-group size,
  // so L <= 32; L = 32 would put one whole sub-group on a single row and is not
  // instantiated because no current model needs D = 1024.
  static constexpr int kMaxLanesPerRow = 16;
  static constexpr int kMaxHadamardDim = kGroupSize * kMaxLanesPerRow;

  static constexpr int ct_log2(int n) { return n <= 1 ? 0 : 1 + ct_log2(n / 2); }

  // A transform size is supported iff it is 32 * L for a power-of-two L that
  // divides the sub-group size and is within the instantiated range.
  static constexpr bool is_supported_hadamard_dim(int64_t dim) {
    return dim >= kGroupSize && dim <= kMaxHadamardDim && dim % kGroupSize == 0 &&
           (dim / kGroupSize) * kGroupSize == dim && ((dim / kGroupSize) & (dim / kGroupSize - 1)) == 0;
  }

  // FP4 (E2M1) magnitude levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
  // Thresholds and boundary comparison operators are taken verbatim from the
  // PyTorch reference (auto_round_extension/vllm_ext/fp4_utils.py::cast_to_fp4)
  // so that the kernel is bit-exact with it.
  //
  // Written branchless, as a descending count rather than the reference's
  // early-out if-ladder, because this runs 32 times per work-item on data whose
  // magnitudes are unrelated across neighbouring SIMD lanes. An if-ladder there
  // diverges on essentially every group and the hardware has to execute all
  // eight arms under mask, which is enough to turn this memory-bound kernel
  // ALU-bound: measured 386 -> 397 GB/s at D = 32 and 357 -> 397 GB/s at
  // D = 128 on Arc Pro B60.
  //
  // The descending form is required, not merely convenient. Counting *up* with
  // the complementary predicates (a > 0.25f, a >= 0.75f, ...) agrees with the
  // ladder on every ordered input but maps NaN to 0, whereas the ladder falls
  // through to 7. Counting down with the ladder's own predicates reproduces
  // that: for NaN every comparison is false, giving 7 - 0. Verified exhaustively
  // against the if-ladder over all 2^32 float bit patterns: zero mismatches.
  static inline int e2m1_magnitude_index(float a) {
    return 7 - (static_cast<int>(a <= 5.0f) + static_cast<int>(a < 3.5f) + static_cast<int>(a <= 2.5f) +
                static_cast<int>(a < 1.75f) + static_cast<int>(a <= 1.25f) + static_cast<int>(a < 0.75f) +
                static_cast<int>(a <= 0.25f));
  }

  // Quantize one 32-element group held in registers: E8M0 exponent + the 16
  // packed code bytes as four 32-bit words.
  //
  // Shared by every path that owns a whole group in a single work-item (the
  // D = 32 per-item kernel and each lane of the D = 128 cooperative kernel), so
  // that all of them are bit-exact against the same PyTorch reference. The
  // operation order is part of the frozen contract: ilogb for the exponent,
  // a per-element ldexp for the scaling, and the branchy threshold ladder
  // above -- not the cheaper bit manipulations, which round NaN/Inf amax
  // differently.
  static inline void quant_group32(const float* v, sycl::vec<uint32_t, 4>& packed, uint8_t& e8m0) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kGroupSize; ++i) {
      amax = sycl::fmax(amax, sycl::fabs(v[i]));
    }

    e8m0 = 0;
    int exp_shift = 0;
    if (amax > 0.0f) {
      int biased = sycl::ilogb(amax) - 2 + 127;
      biased = biased < 0 ? 0 : (biased > 254 ? 254 : biased);
      e8m0 = static_cast<uint8_t>(biased);
      exp_shift = biased - 127;
    }

    packed = sycl::vec<uint32_t, 4>(0u);
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
  //
  // Quant-only mode (``quant_only = true``) is the controlled baseline used to
  // attribute bandwidth: it strips the Hadamard transform (forces ``norm`` to
  // 1 and skips the butterfly stages) while keeping the exact same loads,
  // packing and stores. The raw activation is then quantized directly, so the
  // memory traffic -- and therefore the bandwidth -- is directly comparable to
  // the fused path (see test/README_HMT_QUANT_ONLY_BASELINE.md).
  template <typename T>
  static void fwht_quant_per_item(sycl::queue* q, const T* x, const float* hadamard, uint8_t* out_codes,
                                  uint8_t* out_scale, int64_t total_groups, bool quant_only = false) {
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
      // Quant-only mode forces norm to 1 (hadamard is *not* dereferenced: the
      // ternary only evaluates the taken branch) and skips the butterflies
      // below, so the raw activation is quantized directly.
      const float norm = quant_only ? 1.0f : hadamard[0];

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

      if (!quant_only) {
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
      }

      // Pack 32 codes into 16 bytes, emitted as four 32-bit words. gid*16 is
      // 16-byte aligned, so this is a single aligned vector store.
      sycl::vec<uint32_t, 4> packed;
      uint8_t e8m0;
      quant_group32(v, packed, e8m0);

      auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(out_codes + gid * (kGroupSize / 2));
      *dst = packed;
      out_scale[gid] = e8m0;
    });
  }

  // Stream-only baseline: the traffic-matched roofline for every fused path.
  //
  // Keeps the fused kernel's item mapping (one work-item = 32 contiguous
  // elements = 16 code bytes + 1 scale byte), its two 16-byte vector loads and
  // its single 16-byte vector store, but replaces the transform *and* the
  // quantization math with the cheapest possible data-dependent function: the
  // low nibble of each element's raw bit pattern, packed with the same
  // even-element-low-nibble rule, and an XOR fold for the scale byte.
  //
  // Why this and not ``dst.copy_(src)``. A streaming copy has a 1:1 read:write
  // ratio; the fused kernel reads 64 B and writes 17 B per group, i.e. W/R =
  // 0.266. Measured on Arc Pro B60 the device's cost per 64 B read is strongly
  // non-linear in W/R (0.144 ns at W/R = 0, 0.187 at 0.25, 0.414 at 1.0), and a
  // hand-written 1:1 kernel reaches only 310 GB/s where ``memcpy`` reaches 407
  // -- so a copy differs from the fused kernel in *both* the traffic ratio and
  // the instruction mix, and its bandwidth is not an attainable target. This
  // baseline differs in neither: it is the same kernel with the arithmetic
  // removed, so ``BW_fused / BW_stream`` is a true utilization figure and stays
  // meaningful when the dtype, the shape or the Hadamard dimension changes.
  //
  // The result is deterministic and has a PyTorch reference
  // (``mxfp4_stream_reference``) precisely so that a test can prove the loads
  // were not optimized away -- a dead-code-eliminated baseline would report an
  // absurd bandwidth and silently invalidate every ratio computed against it.
  template <typename T>
  static void stream_baseline_per_item(sycl::queue* q, const T* x, uint8_t* out_codes, uint8_t* out_scale,
                                       int64_t total_groups) {
    const int64_t num_wg = (total_groups + kWorkGroupSize - 1) / kWorkGroupSize;
    const size_t global_size = static_cast<size_t>(num_wg) * kWorkGroupSize;
    constexpr int kLoadVecElems = 8;
    constexpr int kLoadVecCount = kGroupSize / kLoadVecElems;

    q->parallel_for(sycl::nd_range<1>(global_size, kWorkGroupSize), [=](sycl::nd_item<1> item) {
      const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
      if (gid >= total_groups) {
        return;
      }

      const auto* src = reinterpret_cast<const sycl::vec<T, kLoadVecElems>*>(x + gid * kGroupSize);

      sycl::vec<uint32_t, 4> packed(0u);
      uint32_t fold = 0u;
#pragma unroll
      for (int c = 0; c < kLoadVecCount; ++c) {
        const sycl::vec<T, kLoadVecElems> chunk = src[c];
        uint32_t word = 0u;
#pragma unroll
        for (int i = 0; i < kLoadVecElems; ++i) {
          const uint32_t bits = static_cast<uint32_t>(sycl::bit_cast<uint16_t>(chunk[i]));
          word |= (bits & 0xFu) << (i * 4);
        }
        packed[c] = word;
        // Every loaded element feeds the stored result, so nothing can be
        // eliminated; the fold also produces the scale byte.
        fold ^= word;
      }

      auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(out_codes + gid * (kGroupSize / 2));
      *dst = packed;
      out_scale[gid] = static_cast<uint8_t>(fold & 0xFFu);
    });
  }

  // D = 128 FWHT + MXFP4, four cooperating lanes per row.
  //
  // A 128-point transform does not fit the one-work-item-per-row shape that
  // works so well at D = 32: 128 live FP32 values exceed the default per-thread
  // register budget and the kernel spills catastrophically (measured 140 GB/s,
  // a third of the roofline; raising the GRF to 256 recovers only 317 GB/s
  // because occupancy drops).
  //
  // Instead the transform is factored as H_D = H32 (x) H_L, with L = D / 32
  // lanes per row. Lane l of a group of L owns the contiguous chunk
  // [l*32, l*32+32):
  //
  //   * stages 0..4 (h = 1, 2, 4, 8, 16) are entirely inside the lane's own 32
  //     values -- bit-for-bit the same butterfly as the D = 32 kernel;
  //   * stages 5..(5 + log2(L) - 1) (h = 32, 64, ...) become cross-lane
  //     butterflies against lane ^1, ^2, ^4, ..., i.e. log2(L) * 32 sub-group
  //     shuffles.
  //
  // The 32-element quantization group is exactly one lane's chunk, so absmax,
  // E8M0, encoding, packing and the 16-byte store are all lane-local: no SLM,
  // no barriers, and no cross-lane reduction anywhere in the quantizer. This is
  // the property that makes the fan-out cheap, and it holds for every L: the
  // per-lane register footprint is a constant 32 floats, so unlike a
  // one-work-item-per-row formulation nothing spills as D grows. Only the
  // shuffle count grows, and it grows as log2(L).
  //
  // Measured on Arc Pro B60 at the attention QKV shape ([1, 75600, 40, 128],
  // bf16): 395 GB/s against a 391 GB/s stream-only roofline, versus 140 GB/s
  // for one work-item per row (register spill) and 318 GB/s for two lanes per
  // row. Note that raising the GRF size *hurts* here (276 GB/s), the opposite
  // of the single-work-item variant.
  //
  // Convergence. The cross-lane stages are sub-group collectives, so every lane
  // of the sub-group must reach them. Out-of-range lanes in the tail therefore
  // do *not* return early: they clamp their row index to 0, participate in the
  // shuffles with valid data, and are masked off only at the store. Because
  // rows are L-lane aligned and the sub-group size is a multiple of L, the L
  // lanes of a row always live in the same sub-group.
  //
  // L = 1 instantiates correctly (zero cross-lane stages, degenerating to the
  // D = 32 butterfly) but is not what the dispatcher routes D = 32 to: that
  // case goes to fwht_quant_per_item, which also carries the quant-only and
  // stream-only ablations and must keep the exact item mapping they compare
  // against.
  //
  // Normalization placement. Unlike the D = 32 kernel, which scales by
  // 1/sqrt(D) on load, this kernel scales *after* the butterflies. Scaling on
  // load would leave a multiply feeding directly into the first stage's add,
  // and the Intel GPU compiler fuses that pair into an FMA. The fused form
  // rounds differently from the reference's separate multiply and add, and the
  // resulting one-ULP error propagates through every later stage: measured on
  // Arc Pro B60, 60% of the transform outputs differed at D = 128 and D = 512,
  // which showed up downstream as 17 and 25 wrong MXFP4 codes per 67 M -- only
  // groups whose scaled magnitude landed within an ULP of an E2M1 threshold
  // flipped, which is exactly the kind of rate that small unit-test shapes miss.
  //
  // Neither -ffp-contract=off nor a #pragma clang fp contract(off) suppresses
  // it (both were measured to change nothing: the fusion happens in the GPU
  // backend, below the point those act on), and a bit_cast round trip through
  // uint32_t does not either. Moving the multiply past the butterflies removes
  // the mul/add pair altogether: what remains multiplies into fabs, fmax and
  // the comparison ladder of the quantizer, none of which is an add. That is a
  // structural fix rather than a request the compiler may ignore, and it costs
  // nothing measurable (235 vs 237 GB/s).
  //
  // Note this is *why* D = 64 and D = 256 never failed: their 1/sqrt(D) is
  // exactly 1/8 and 1/16, so the load-time product was exact and fusing it
  // changed nothing. Only the irrational norms exposed the bug.
  //
  // The cost is the intermediate range: values now grow to D*max|x| before
  // being scaled down, rather than staying at sqrt(D)*max|x|. In FP32 that is
  // immaterial here -- FP16 activations cap the intermediate at 512*65504 =
  // 3.4e7 -- but it does mean this path and the D = 32 path are not bit-exact
  // against each other, and the torch reference mirrors the split.
  template <typename T, int kLanes>
  static void fwht_quant_cooperative(sycl::queue* q, const T* x, const float* hadamard, uint8_t* out_codes,
                                     uint8_t* out_scale, int64_t num_rows) {
    static_assert(kLanes >= 1 && kLanes <= kMaxLanesPerRow, "lanes per row out of range");
    static_assert((kLanes & (kLanes - 1)) == 0, "lanes per row must be a power of two");
    static_assert(kSubGroupSizeCoop % kLanes == 0,
                  "lanes per row must divide the sub-group size, otherwise a row's lanes can straddle two "
                  "sub-groups and the cross-lane butterfly reads the wrong partner");

    constexpr int kDim = kGroupSize * kLanes;
    constexpr int kCrossStages = ct_log2(kLanes);

    const int64_t total_items = num_rows * kLanes;
    const int64_t num_wg = (total_items + kWorkGroupSize - 1) / kWorkGroupSize;
    const size_t global_size = static_cast<size_t>(num_wg) * kWorkGroupSize;
    constexpr int kLoadVecElems = 8;
    constexpr int kLoadVecCount = kGroupSize / kLoadVecElems;

    q->parallel_for(
        sycl::nd_range<1>(global_size, kWorkGroupSize),
        [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(kSubGroupSizeCoop)]] {
          auto sg = item.get_sub_group();
          const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
          const int64_t row = gid / kLanes;
          const int lane = static_cast<int>(gid % kLanes);
          const int sg_lane = static_cast<int>(sg.get_local_id()[0]);

          // Tail lanes stay converged for the sub-group shuffles below by
          // re-reading row 0; only their store is suppressed.
          const bool active = row < num_rows;
          const int64_t safe_row = active ? row : 0;

          // hadamard[0] == H[0][0] == 1/sqrt(D), applied *after* the butterflies
          // -- see the normalization-placement note above the function.
          const float norm = hadamard[0];

          const auto* src =
              reinterpret_cast<const sycl::vec<T, kLoadVecElems>*>(x + safe_row * kDim + lane * kGroupSize);

          float v[kGroupSize];
#pragma unroll
          for (int c = 0; c < kLoadVecCount; ++c) {
            const sycl::vec<T, kLoadVecElems> chunk = src[c];
#pragma unroll
            for (int i = 0; i < kLoadVecElems; ++i) {
              v[c * kLoadVecElems + i] = static_cast<float>(chunk[i]);
            }
          }

          // Stages 0..4: h < 32, both partners live in this lane.
#pragma unroll
          for (int stage = 0; stage < kNumFwhtStagesLocal; ++stage) {
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

          // Remaining stages: h = 32, 64, ... The partner element lives in lane
          // ^1, ^2, ^4, ... at the same offset within the chunk. The lane
          // holding the high half of the pair computes the difference, matching
          // the D = 32 butterfly's orientation.
#pragma unroll
          for (int stage = 0; stage < kCrossStages; ++stage) {
            const int lane_h = 1 << stage;
            const int partner = sg_lane ^ lane_h;
            const bool is_low = (lane & lane_h) == 0;
#pragma unroll
            for (int i = 0; i < kGroupSize; ++i) {
              const float other = sycl::select_from_group(sg, v[i], partner);
              v[i] = is_low ? (v[i] + other) : (other - v[i]);
            }
          }

          // Normalization, applied last (see the note above the function). The
          // product feeds fabs/fmax and the comparison ladder inside the
          // quantizer, never an add, so there is nothing here to contract.
#pragma unroll
          for (int i = 0; i < kGroupSize; ++i) {
            v[i] *= norm;
          }

          sycl::vec<uint32_t, 4> packed;
          uint8_t e8m0;
          quant_group32(v, packed, e8m0);

          if (active) {
            // Group index within the flattened [total_groups, 32] view: the
            // row's L groups are contiguous, so this matches the D = 32 layout.
            const int64_t group_id = row * kLanes + lane;
            auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(out_codes + group_id * (kGroupSize / 2));
            *dst = packed;
            out_scale[group_id] = e8m0;
          }
        });
  }

  // Runtime dim -> compile-time lane count. Every supported D gets its own
  // fully unrolled instantiation; an unsupported one is rejected here as a
  // backstop, the wrapper and the binding having already validated it.
  template <typename T>
  static bool dispatch_cooperative(sycl::queue* q, const T* x, const float* hadamard, uint8_t* out_codes,
                                   uint8_t* out_scale, int64_t num_rows, int64_t hadamard_dim) {
    switch (hadamard_dim) {
      case kGroupSize * 2:
        fwht_quant_cooperative<T, 2>(q, x, hadamard, out_codes, out_scale, num_rows);
        return true;
      case kGroupSize * 4:
        fwht_quant_cooperative<T, 4>(q, x, hadamard, out_codes, out_scale, num_rows);
        return true;
      case kGroupSize * 8:
        fwht_quant_cooperative<T, 8>(q, x, hadamard, out_codes, out_scale, num_rows);
        return true;
      case kGroupSize * 16:
        fwht_quant_cooperative<T, 16>(q, x, hadamard, out_codes, out_scale, num_rows);
        return true;
      default:
        return false;
    }
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

  // Dispatcher.
  //
  // ``hadamard_dim`` selects the transform size: 32 (the GEMM activation case,
  // one work-item per group) or 32 * L for L in {2, 4, 8, 16} -- i.e. 64, 128,
  // 256, 512 -- which fan a row out over L cooperating lanes. Every size
  // quantizes in groups of 32 and shares the output layout, so the two baseline
  // modes below are dimension-agnostic: they only ever see a flat sequence of
  // 32-element groups.
  //
  // ``stream_only`` and ``quant_only`` are mutually exclusive; the caller is
  // expected to have rejected the combination already.
  template <typename T>
  static void mxfp4_hadamard_quant(sycl::queue* q, const T* x, const float* hadamard, uint8_t* out_codes,
                                   uint8_t* out_scale, int64_t num_rows, int64_t k, bool use_fwht,
                                   bool quant_only = false, int64_t hadamard_dim = kHadamardDim,
                                   bool stream_only = false) {
    const int64_t total_groups = num_rows * (k / kGroupSize);
    if (total_groups <= 0) {
      return;
    }

    if (stream_only) {
      // Traffic-matched roofline: same loads, same packing shape, same stores,
      // no transform and no quantization math. hadamard is not read.
      stream_baseline_per_item<T>(q, x, out_codes, out_scale, total_groups);
      return;
    }
    if (quant_only) {
      // Quant-only baseline: strip the Hadamard transform entirely. The
      // per-item layout and memory traffic are identical to the fused FWHT
      // path, so bandwidths are directly comparable; hadamard is not read.
      fwht_quant_per_item<T>(q, x, nullptr, out_codes, out_scale, total_groups, /*quant_only=*/true);
      return;
    }

    if (hadamard_dim > kGroupSize) {
      // D > 32 supports the Sylvester matrix only; the caller enforces this.
      dispatch_cooperative<T>(q, x, hadamard, out_codes, out_scale, num_rows * (k / hadamard_dim), hadamard_dim);
      return;
    }

    if (use_fwht) {
      fwht_quant_per_item<T>(q, x, hadamard, out_codes, out_scale, total_groups);
    } else {
      mxfp4_hadamard_quant_impl<T>(q, x, hadamard, out_codes, out_scale, num_rows, k);
    }
  }
};

}  // namespace ark

#endif  // ARK_XPU
