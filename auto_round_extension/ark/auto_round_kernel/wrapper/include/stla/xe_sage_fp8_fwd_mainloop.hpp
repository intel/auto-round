/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
 * @file xe_sage_fp8_fwd_mainloop.hpp
 * @brief SageAttention FP8 forward mainloop for Intel Xe2 GPUs.
 *
 * Key differences from SAGEV1FwdMainloop (INT8 QK path):
 *
 * 1. Q and K are FP8 E4M3 (pre-quantized on host) with per-block FP32 scales.
 *    GEMM1: FP8 x FP8 -> FP32 via DPAS with block scaling.
 *
 * 2. K may be smoothed (K = K - mean(K)) on host before quantization.
 *
 * 3. After softmax: P (FP32) can be quantized to FP8 E4M3 for GEMM2 when
 *    UseFP8PV is true. When false (default), P remains FP32 for higher precision.
 *    If UseFP8PV, P quantization uses static scale = 1/448.
 *
 * 4. V is FP8 E4M3 (pre-quantized on host) with a per-tensor FP32 scale.
 *    GEMM2: P(FP32 or FP8) x V(FP8) -> FP32 via DPAS with V scaling.
 *
 * 5. Two-layer accumulation for GEMM2 (Layer 1: MMA -> RO_inst_buf,
 *    Layer 2: RO_inst_buf -> RO). This compensates for FP22 truncation
 *    in Xe2 DPAS accumulation.
 *
 * 6. The scalar V scale is applied once to each GEMM2 contribution.
 *
 * 7. SIMD32 exp2 optimization using vISA asm for better performance.
 *
 * Reference: SageAttention (ICLR 2025) / SageAttention2 (ICML 2025).
 *
 * Implementation note: Q, K, V FP8 tensors are expected to be in E4M3 format
 * (IEEE float8_e4m3fn). The scale tensors are FP32 per-block (Q, K) or
 * scalar (V) quantization scales computed on the host.
 */

#pragma once

#include "cute/util/print_tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include <cmath>
#include <type_traits>
#include <cute/util/xe_split_barrier.hpp>

namespace cutlass::sage {

template <int Stages, bool UseFP8PV_ = false>
class XeSageFP8 {};  // SageAttention FP8 mainloop, P in registers.

}  // namespace cutlass::sage

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_,
          class TiledMMAQK_,  // Tiling for Q*K GEMM (FP8 MMA)
          class TiledMMAPV_,  // Tiling for P*V GEMM (FP8 MMA)
          int VTiles_,        // # of tiles in V dimension
          class TensorQ_,     // Global Q/K/V tensors (FP8 E4M3)
          class TensorK_, class TensorV_,
          class TensorScaleQ_,  // Per-block scale for Q (FP32)
          class TensorScaleK_,  // Per-block scale for K (FP32)
          class TensorScaleV_,  // Scalar scale for V (FP32)
          class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_ = void,
          class TiledCopyK_ = void,
          class TiledCopyV_ = void,
          class TiledCopyK_cache_ = void,
          class TiledCopyV_cache_ = void>
struct SageFP8FwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>,
                "Could not find a SageFP8FwdMainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for Xe2 FP8 DPAS mainloop
template <int Stages, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_,
          bool UseFP8PV_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TensorScaleQ_, class TensorScaleK_, class TensorScaleV_,
          class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_,
          class TiledCopyK_cache_, class TiledCopyV_cache_>
struct SageFP8FwdMainloop<
    sage::XeSageFP8<Stages>, CausalMask_, FullMask_, CachedKV_, PagedKV_,
    UseFP8PV_,
    TiledMMAQK_, TiledMMAPV_, VTiles_,
    TensorQ_, TensorK_, TensorV_,
    TensorScaleQ_, TensorScaleK_, TensorScaleV_,
    TensorK_cache_, TensorV_cache_,
    TiledCopyQ_, TiledCopyK_, TiledCopyV_,
    TiledCopyK_cache_, TiledCopyV_cache_> {

  // =========================================================================
  // Type Aliases
  // =========================================================================

  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SubgroupLayoutPV = decltype(TiledMMAPV{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;
  using ElementQ = typename TensorQ::element_type;   // fp8_e4m3_t (E4M3)
  using ElementK = typename TensorK::element_type;
  using ElementV = typename TensorV::element_type;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TensorScaleQ = TensorScaleQ_;
  using TensorScaleK = TensorScaleK_;
  using TensorScaleV = TensorScaleV_;
  using ElementScale = typename TensorScaleQ::element_type;  // float (FP32)

  using TiledCopyQ = conditional_t<is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK = conditional_t<is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV = conditional_t<is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;

  using TensorK_cache2D = decltype(TensorK_cache_{}(append<rank_v<TensorK_cache_>>(make_coord(_, _), 0)));
  using TensorV_cache2D = decltype(TensorV_cache_{}(append<rank_v<TensorV_cache_>>(make_coord(_, _), 0)));
  using TiledCopyK_cache = conditional_t<is_void_v<TiledCopyK_cache_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK_cache2D{})), TiledCopyK_cache_>;
  using TiledCopyV_cache = conditional_t<is_void_v<TiledCopyV_cache_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV_cache2D{})), TiledCopyV_cache_>;

  // Accumulator types
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;      // QK result (FP32)
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSPartialRow = decltype(reduce<1, ReduceMode::Vertical>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;  // float
  using ElementM = typename TiledMMAQK::ValTypeA;
  using SingleFragPV = FragC<TiledMMAPV>;
  using FragA = expand_sg_fragment_t<SingleFragPV, 1, VTiles>;
  using FragARow = FragSRow;
  using ElementA = typename SingleFragPV::value_type;  // float

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool FullMask = FullMask_;
  static constexpr bool PerTensorScale = true;
  static constexpr bool BlockScale = false;
  static constexpr bool UseFP8PV = UseFP8PV_;

  // P quantization constants
  static constexpr float kFP8Max = 448.0f;        // FP8 E4M3 max value
  static constexpr float kInvFP8Max = 1.0f / kFP8Max;  // Static scale for P quantization

  // =========================================================================
  // User-facing arguments
  // =========================================================================

  struct Arguments {
    float const scale;       // sm_scale * log2(e)
    float const* qscale = nullptr;   // Per-block Q scale (FP32)
    float const* kscale = nullptr;   // Per-block K scale (FP32)
    float const* vscale = nullptr;   // Scalar V scale (FP32)
    float const* mask = nullptr;      // Full mask (if FullMask_)
    int scale_block_size = 0;         // Block size for scales
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int const* num_pages_per_seq = nullptr;
  };

  using Params = Arguments;
  struct SharedStorage {};
  Params params;

  SageFP8FwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void*) {
    constexpr double kLog2e = 1.4426950408889634074;
    return Params{args.scale * static_cast<float>(kLog2e),
                 args.qscale, args.kscale, args.vscale,
                 args.mask, args.scale_block_size,
                 args.ptr_page_table, args.page_size, args.num_pages_per_seq};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) { return true; }

  // DPAS kernels run as SIMD16, so pack two SIMD16 vectors into one temporary SIMD32 vector
  // before issuing a single vISA SIMD32 instruction.
  CUTLASS_DEVICE static void exp2_pair_simd32_asm(ElementA x0, ElementA x1, ElementA& y0, ElementA& y1) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
    asm(
        "{\n"
        ".decl OUT0 v_type=G type=F num_elts=16 alias=<%0,0>\n"
        ".decl OUT1 v_type=G type=F num_elts=16 alias=<%1,0>\n"
        ".decl IN0  v_type=G type=F num_elts=16 alias=<%2,0>\n"
        ".decl IN1  v_type=G type=F num_elts=16 alias=<%3,0>\n"
        ".decl TMP_IN  v_type=G type=F num_elts=32 align=64\n"
        ".decl TMP_OUT v_type=G type=F num_elts=32 align=64\n"
        "mov (M1_NM, 16) TMP_IN(0,0)<1> IN0(0,0)<1;1,0>\n"
        "mov (M1_NM, 16) TMP_IN(1,0)<1> IN1(0,0)<1;1,0>\n"
        "exp (M1_NM, 32) TMP_OUT(0,0)<1> TMP_IN(0,0)<1;1,0>\n"
        "mov (M1_NM, 16) OUT0(0,0)<1> TMP_OUT(0,0)<1;1,0>\n"
        "mov (M1_NM, 16) OUT1(0,0)<1> TMP_OUT(1,0)<1;1,0>\n"
        "}\n"
        : "=rw"(y0), "=rw"(y1)
        : "rw"(x0), "rw"(x1));
#else
    y0 = sycl::native::exp2(x0);
    y1 = sycl::native::exp2(x1);
#endif
  }

  // =========================================================================
  // Main operator()
  // =========================================================================

  CUTLASS_DEVICE
  int get_physical_k_tile(int K, int l_coord, int seq_len_kv_cache) {
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int batch_offset = params.num_pages_per_seq
        ? params.num_pages_per_seq[l_coord]
        : l_coord * (seq_len_kv_cache / params.page_size);
    return params.ptr_page_table[batch_offset + K / tiles_per_page] * tiles_per_page
         + K % tiles_per_page;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void
  operator()(TensorQ2D const& Q_2D,
             TensorK2D const& K_2D,
             TensorV2D const& V_2D,
             FragA& tArA,
             FragARow& tA_max,
             FragSPartialRow& tA_sum,
             QVCoord blk_qv,
             int blk_k0,
             int blk_k1,
             int total_blk,
             int thr_id,
             int seq_len,
             int seq_len_kv_cache,
             int l_coord,
             int full_tile_offset,
             int discard_seq_coord,
             TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
             TensorV_cache2D const& V_cache_2D = TensorV_cache2D{},
             float = 1.0f,
             float = 1.0f,
             float = 1.0f) {
    using namespace sycl::ext::oneapi::this_work_item;

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor cK = make_identity_tensor(K_2D.shape());
    Tensor cV = make_identity_tensor(V_2D.shape());
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape());
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape());
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));
    Tensor cPV = make_identity_tensor(select<0, 1>(TileShapePV{}));

    Tensor gQ = local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});
    Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});

    Tensor gK_cache = local_tile(cK_cache, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor gV_cache = local_tile(cV_cache, tile_shape_v, make_coord(get<1>(blk_qv), _));
    Tensor gV_cache_split = local_tile(gV_cache, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});

    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};
    TiledCopyK_cache copy_k_cache{K_cache_2D};
    TiledCopyV_cache copy_v_cache{V_cache_2D};

    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_copy_k_cache = copy_k_cache.get_slice(thr_id);
    auto thr_copy_v_cache = copy_v_cache.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tKgK = thr_copy_k.partition_S(gK);
    auto tVgV = thr_copy_v.partition_S(gV_split);
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);

    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto prefetch_k_cache = make_block_2d_prefetch(copy_k_cache);
    auto prefetch_v_cache = make_block_2d_prefetch(copy_v_cache);

    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pKgK_cache = prefetch_k_cache.get_slice(thr_id).partition_S(gK_cache);
    auto pVgV_cache = prefetch_v_cache.get_slice(thr_id).partition_S(gV_cache_split);

    int kblocks_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
      CUTLASS_PRAGMA_UNROLL
      for (int K = 0; K < Stages; K++) {
        if (K < kblocks_cache) {
          if constexpr (PagedKV) {
            int physical_K_tile = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
            prefetch(prefetch_k_cache, pKgK_cache(_, _, _, physical_K_tile, D));
          } else {
            prefetch(prefetch_k_cache, pKgK_cache(_, _, _, K, D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_, _, _, K - kblocks_cache, D));
        }
      }
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    // =========================================================================
    // Main loop body
    // =========================================================================
    auto mainloop_body = [&](auto cached_k, int K,
                             auto& copy_k_cur, auto& copy_v_cur,
                             auto& prefetch_v_cur,
                             auto& tKgK_cur, auto& tVgV_cur, auto& pVgV_cur) {
      barrier_arrive(ScopeWorkgroup);
      constexpr bool is_cache = decltype(cached_k)::value;

      int k_idx;
      if constexpr (is_cache) {
        k_idx = K;
        if constexpr (PagedKV) {
          k_idx = get_physical_k_tile(K, l_coord, seq_len_kv_cache);
        }
      } else {
        k_idx = K - kblocks_cache;
      }

      // =====================================================================
      // GEMM1: S = Q(FP8) * K(FP8)^T -> FP32
      // Q and K are FP8 E4M3. Scale is applied after MMA.
      // =====================================================================
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK_cur); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        if constexpr (is_cache) {
          copy(copy_k_cur, tKgK_cur(_, _, _, k_idx, D), tKrK);
        } else {
          copy(copy_k_cur, tKgK_cur(_, _, _, k_idx, D), tKrK);
        }
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      // =====================================================================
      // Apply Q/K scales (block-wise)
      // =====================================================================
      if (params.scale_block_size) {
        int q_base = get<0>(blk_qv) * get<0>(TileShapeQK{});
        int k_base = k_idx * get<1>(TileShapeQK{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          int row = get<0>(tSrS.tv_layout()(i));
          int col = get<1>(tSrS.tv_layout()(i));
          int q_idx = q_base + row;
          int k_idx_local = k_base + col;
          int q_scale_idx = q_idx / params.scale_block_size;
          int k_scale_idx = k_idx_local / params.scale_block_size;
          float scale = params.qscale[q_scale_idx] * params.kscale[k_scale_idx];
          tSrS(i) *= scale * params.scale;
        }
      } else {
        const float qk_scale = (params.qscale ? params.qscale[0] : 1.0f) *
                   (params.kscale ? params.kscale[0] : 1.0f) * params.scale;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) *= qk_scale;
        }
      }

      // =====================================================================
      // V prefetch for GEMM2
      // =====================================================================
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_, _, _, VV, k_idx));
      }

      // =====================================================================
      // Causal masking
      // =====================================================================
      if constexpr (!is_cache && CausalMask) {
        if (K == total_blk - 1) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - seq_len_kv_cache - full_tile_offset > row_idx - discard_seq_coord) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }

      // =====================================================================
      // Full mask
      // =====================================================================
      if constexpr (!is_cache && FullMask) {
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        int row_idx_begin = get<0>(cS_thread(0));
        int row_idx_end = row_idx_begin + q_sg_tile;
        int col_idx_begin = get<1>(cS_thread(0));
        int col_idx_end = col_idx_begin + get<1>(TileShapeQK{});
        if (row_idx_end <= seq_len && col_idx_end <= seq_len) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            tSrS(i) += ElementS(params.mask[col_idx + row_idx * seq_len]);
          }
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            tSrS(i) += (row_idx < seq_len && col_idx < seq_len)
                           ? ElementS(params.mask[col_idx + row_idx * seq_len])
                           : ElementS(-INFINITY);
          }
        }
      }

      // =====================================================================
      // Remainder k masking
      // =====================================================================
      if constexpr (!is_cache) {
        if (check_remainder_k && K == total_blk - 1) {
          FragSRow k_rem_mask;
          int k_val = get<0>(tKgK_cur(0, 0, 0, k_idx, 0)) + kblocks_cache * get<1>(TileShapeQK{});
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); i++) {
            tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
          }
        }
      }

      // =====================================================================
      // Softmax
      // =====================================================================
      auto [rescale, tS_partial_sum] = softmax(tSrS, tA_max);
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      constexpr int kSumSize = decltype(tA_sum.size())::value;
      constexpr bool kSumDivVT = (kSumSize % VTiles == 0);
      constexpr int kSumPerVT = kSumDivVT ? (kSumSize / VTiles) : 0;

      if constexpr (!kSumDivVT) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSumSize; i++) {
          tA_sum(i) = tA_sum(i) * group_broadcast(sg, rescale(0), i) + tS_partial_sum(i);
        }
      }

      reorder(tSrS, tArP);

      // Get V scale for this iteration
      const ElementS v_scale = (params.vscale != nullptr) ? params.vscale[0] : 1.0f;

      // Optional P quantization to FP8 for GEMM2 (when UseFP8PV is enabled)
      if constexpr (UseFP8PV) {
        quantize_p_to_fp8(tSrS, v_scale);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        auto tArA_v = tArA(_, _, _, VV);
        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA_v.size(); i++) {
            tArA_v(i) *= broadcast<0>(rescale, tArA, i);
          }
        }

        if constexpr (kSumDivVT) {
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < kSumPerVT; j++) {
            int const i = VV * kSumPerVT + j;
            tA_sum(i) = tA_sum(i) * group_broadcast(sg, rescale(0), i) + tS_partial_sum(i);
          }
        }

        copy(copy_v_cur, tVgV_cur(_, _, _, VV, k_idx), tVrV);
        reorder(tVrV, tArV);

        SingleFragPV tArAcc;
        clear(tArAcc);
        cute::gemm(mma_pv, tArP, tArV, tArAcc);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tArA_v.size(); i++) {
          // Apply V scale after GEMM2 (unless already applied in quantize_p_to_fp8)
          if constexpr (!UseFP8PV) {
            tArA_v(i) += ElementA(tArAcc(i)) * v_scale;
          } else {
            tArA_v(i) += ElementA(tArAcc(i));
          }
        }
      }

      // =====================================================================
      // K prefetch
      // =====================================================================
      int K_next = K + Stages;
      for (int D = 0; D < size<4>(pKgK); D++) {
        if constexpr (is_cache) {
          bool is_cache_next = K_next < kblocks_cache;
          int physical_K_next = K_next;
          if constexpr (PagedKV) {
            if (is_cache_next) {
              physical_K_next = get_physical_k_tile(K_next, l_coord, seq_len_kv_cache);
            }
          }
          if (is_cache_next) {
            prefetch(prefetch_k_cache, pKgK_cache(_, _, _, physical_K_next, D));
          } else {
            prefetch(prefetch_k, pKgK(_, _, _, K_next - kblocks_cache, D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_, _, _, K_next - kblocks_cache, D));
        }
      }

      barrier_wait(ScopeWorkgroup);
    };

    // Execute: first KV-cache tiles, then new KV tiles
    if constexpr (CachedKV) {
      for (int K = blk_k0; K < kblocks_cache; K++) {
        mainloop_body(std::bool_constant<true>{}, K,
                      copy_k_cache, copy_v_cache, prefetch_v_cache,
                      tKgK_cache, tVgV_cache, pVgV_cache);
      }
    }
    for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
      mainloop_body(std::bool_constant<false>{}, K,
                    copy_k, copy_v, prefetch_v,
                    tKgK, tVgV, pVgV);
    }
  }

  // =========================================================================
  // Softmax (FP32 online softmax)
  // =========================================================================
  CUTLASS_DEVICE
  auto softmax(FragS& tS, FragARow& tS_max) {
    auto tS_bmax = reduce<1, ReduceMode::Full, /*EnableFast64Rows=*/!CausalMask>(tS, sycl::maximum<void>{});
    FragARow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementA new_max = sycl::max(tS_max(i), ElementA(tS_bmax(i)));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      tS(i) -= broadcast<0>(tS_max, tS, i);
    }

    // Use SIMD32 asm exp2 for better performance (matching BF16 v1 implementation)
    static_assert(FragS{}.size() % 2 == 0, "FragS size must be even for pairwise SIMD32 exp.");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i += 2) {
      exp2_pair_simd32_asm(tS(i), tS(i + 1), tS(i), tS(i + 1));
    }

    auto tS_partial_sum = reduce<1, ReduceMode::Vertical>(tS, sycl::plus<void>{});
    return cute::make_tuple(rescale, tS_partial_sum);
  }

  // =========================================================================
  // P quantization to FP8 (when UseFP8PV is enabled)
  // =========================================================================
  CUTLASS_DEVICE
  void quantize_p_to_fp8(FragS& tS, ElementS v_scale) {
    // Quantize P from FP32 to FP8 E4M3 using static scale = 1/448
    // Then dequantize back to FP32 for GEMM2 (FP8 MMA requires FP8 input)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      // Scale P: multiply by inv_fp8_max to normalize to [-1, 1]
      float p_scaled = tS(i) * kInvFP8Max;
      // Clamp to FP8 E4M3 range [-448, 448]
      p_scaled = sycl::clamp(p_scaled, -kFP8Max, kFP8Max);
      // Convert to FP8 E4M3 and back to float
      ElementQ p_fp8(p_scaled);
      // Dequantize: multiply by fp8_max to get proper value for GEMM2
      tS(i) = static_cast<ElementS>(p_fp8) * kFP8Max;
      // Apply V scale during dequantization
      tS(i) *= v_scale;
    }
  }
};

}  // namespace cutlass::fmha::collective
