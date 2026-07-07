/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

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

template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.

};  // namespace cutlass::sage

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_, bool UseInt8PV_,
          bool WriteBackInt8PV_, bool ExecuteInt8PV_,
          class TiledMMAQK_,  // Tiling for Q*K GEMM
          class TiledMMAPV_,  // Tiling for P*V GEMM
          int VTiles_,        // # of tiles in V dimension
          class TensorQ_,     // Global Q/K/V tensors
          class TensorK_, class TensorV_, class TensorK_cache_, class TensorV_cache_,
          class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
          class TiledCopyK_ = void,  // Optional TiledCopy for loading K
          class TiledCopyV_ = void,  // Optional TiledCopy for loading V
          class TiledCopyK_cache_ = void,
          class TiledCopyV_cache_ = void>  // Optional TiledCopy for loading V_cache
struct SAGEV1FwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_, bool UseInt8PV_,
          bool WriteBackInt8PV_, bool ExecuteInt8PV_, class TiledMMAQK_,
          class TiledMMAPV_, int VTiles_, class TensorQ_, class TensorK_, class TensorV_, class TensorK_cache_,
          class TensorV_cache_, class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_, class TiledCopyK_cache_,
          class TiledCopyV_cache_>
struct SAGEV1FwdMainloop<sage::XeDefault<Stages>, CausalMask_, FullMask_, CachedKV_, PagedKV_, UseInt8PV_,
                         WriteBackInt8PV_, ExecuteInt8PV_, TiledMMAQK_, TiledMMAPV_, VTiles_, TensorQ_, TensorK_,
                         TensorV_, TensorK_cache_, TensorV_cache_, TiledCopyQ_, TiledCopyK_, TiledCopyV_,
                         TiledCopyK_cache_, TiledCopyV_cache_> {
  //
  // Type Aliases
  //
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

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ =
      conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK =
      conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV =
      conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;
  using TensorK_cache = TensorK_cache_;
  using TensorV_cache = TensorV_cache_;
  using TensorK_cache2D = decltype(TensorK_cache_{}(append<rank_v<TensorK_cache_>>(make_coord(_, _), 0)));
  using TensorV_cache2D = decltype(TensorV_cache_{}(append<rank_v<TensorV_cache_>>(make_coord(_, _), 0)));
  using TiledCopyK_cache =
      conditional_t<is_void_v<TiledCopyK_cache_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK_cache2D{})),
                    TiledCopyK_cache_>;
  using TiledCopyV_cache =
      conditional_t<is_void_v<TiledCopyV_cache_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV_cache2D{})),
                    TiledCopyV_cache_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;
  using ElementM = typename TiledMMAQK::ValTypeA;
  using ElementP = float;
  using FragP = decltype(make_subgroup_tensor(make_fragment_like<ElementP>(typename FragS::Base{}),
                                              decltype(FragS{}.tv_layout()){}));

  using FragPRow = decltype(reduce<1>(FragP{}, sycl::plus<void>{}));
  using FragPCol = decltype(reduce<0>(FragP{}, sycl::plus<void>{}));
    using SingleFragPV = FragC<TiledMMAPV>;                      // (atom val,q',v')
    using SingleFragAFloat = std::remove_reference_t<decltype(make_subgroup_tensor(
      make_fragment_like<float>(typename SingleFragPV::Base{}), decltype(SingleFragPV{}.tv_layout()){}))>;
    using SingleFragA = conditional_t<UseInt8PV_, SingleFragAFloat, SingleFragPV>;
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
    using ElementA = typename SingleFragA::value_type;

    static constexpr int SGTilePVQ = get<0>(shape_div(TileShapePV{}, shape(SubgroupLayoutPV{})))();
    using QuantizedPVOperation = XE_DPAS_TT<cute::gcd(SGTilePVQ, 8), int32_t, int8_t, int8_t>;
    using QuantizedTiledMMAPV =
      typename TiledMMAHelper<MMA_Atom<QuantizedPVOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;
    using QuantizedSingleFragA = FragC<QuantizedTiledMMAPV>;

    static_assert(size(SingleFragA{}.shape()) == size(QuantizedSingleFragA{}.shape()),
          "Quantized PV accumulator must match float PV tile size");

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;
    static constexpr bool UseInt8PV = UseInt8PV_;
    static constexpr bool WriteBackInt8PV = WriteBackInt8PV_;
    static constexpr bool ExecuteInt8PV = ExecuteInt8PV_;

  // User-facing arguments
  struct Arguments {
    float const scale;
    float const* mask = nullptr;
    int scale_block_size = 0;  // if non-zero, apply scaling in blocks of this size (for block-sparse attention)
    float const* qscale = nullptr;
    float const* kscale = nullptr;
    float const* vscale = nullptr;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int const* num_pages_per_seq = nullptr;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  SAGEV1FwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    float val = args.scale * static_cast<float>(kLog2e);
    return Params{val, args.mask, args.scale_block_size, args.qscale, args.kscale, args.vscale,
            args.ptr_page_table, args.page_size, args.num_pages_per_seq};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) { return true; }

  // DPAS kernels run as SIMD16, so pack two SIMD16 vectors into one temporary SIMD32 vector
  // before issuing a single vISA SIMD32 instruction.
  CUTLASS_DEVICE static void exp2_pair_simd32_asm(ElementP x0, ElementP x1, ElementP& y0, ElementP& y1) {
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

  CUTLASS_DEVICE
  int get_physical_k_tile(int K, int l_coord, int seq_len_kv_cache) {
    int next_page_logical_idx = K * get<1>(TileShapeQK{}) / params.page_size;
    // get<1>(TileShapeQK{}) usually smaller than page_size.
    // assuming page_size is multiple of get<1>(TileShapeQK{})
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int batch_offset =
        params.num_pages_per_seq ? params.num_pages_per_seq[l_coord] : l_coord * (seq_len_kv_cache / params.page_size);

    return params.ptr_page_table[batch_offset + next_page_logical_idx] * tiles_per_page + K % tiles_per_page;
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(TensorQ2D const& Q_2D,  // (q,d)
                                 TensorK2D const& K_2D,  // (k,d)
                                 TensorV2D const& V_2D,  // (d,k)
                                 FragA& tArA,            // Output accumulator (q,v)
                                 FragARow& tA_max,       // Softmax row-wise max accumulator
                                 FragARow& tA_sum,       // Softmax row-wise sum accumulator
                                 QVCoord blk_qv,         // WG tile indices: (Q,V)
                                 int blk_k0,             // K block range: [K0,K1)
                                 int blk_k1,
                                 int total_blk,  // Total # of K blocks
                                 int thr_id, int seq_len, int seq_len_kv_cache, int l_coord, float* scaleQ,
                                 float* scaleK, float* scaleV, int full_tile_offset, int discard_seq_coord,
                                 TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
                                 TensorV_cache2D const& V_cache_2D = TensorV_cache2D{}) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape());   // (k,d)
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape());   // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)
    Tensor cPV = make_identity_tensor(select<0, 1>(TileShapePV{}));

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});          // (q,d,D)
    Tensor gK = local_tile(cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV = local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    Tensor gK_cache = local_tile(cK_cache, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});        // (k,d,K,D)
    Tensor gV_cache = local_tile(cV_cache, tile_shape_v, make_coord(get<1>(blk_qv), _));                  // (v,k,K)
    Tensor gV_cache_split = local_tile(gV_cache, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};
    TiledCopyK_cache copy_k_cache{K_cache_2D};
    TiledCopyV_cache copy_v_cache{V_cache_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
    QuantizedTiledMMAPV mma_pv_i8{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_copy_k_cache = copy_k_cache.get_slice(thr_id);
    auto thr_copy_v_cache = copy_v_cache.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);
    auto thr_mma_pv_i8 = mma_pv_i8.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);

    FragP tPrS = make_subgroup_tensor(make_fragment_like<ElementP>(tSrS.tensor()), tSrS.tv_layout());

    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tArP_i8 = thr_mma_pv_i8.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));
    auto tCrA = thr_mma_pv_i8.partition_C(cPV);

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto prefetch_k_cache = make_block_2d_prefetch(copy_k_cache);
    auto prefetch_v_cache = make_block_2d_prefetch(copy_v_cache);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pKgK_cache = prefetch_k_cache.get_slice(thr_id).partition_S(gK_cache);
    auto pVgV_cache = prefetch_v_cache.get_slice(thr_id).partition_S(gV_cache_split);

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
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

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    float dq_scale = params.scale_block_size
                         ? scaleQ[(get<0>(blk_qv) * get<0>(TileShapeQK{}) + get_sub_group_id() * q_sg_tile) /
                                  params.scale_block_size] *
                               params.scale
                         : 1.f;
    /* Main loop body */
    auto mainloop_body = [&](auto cached_k, int K, auto& copy_k_cur, auto& copy_v_cur, auto& prefetch_v_cur,
                             auto& tKgK_cur, auto& tVgV_cur, auto& pVgV_cur) {
      /* Split barrier to keep threads together */
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
      int scalek_idx = params.scale_block_size ? K * get<1>(TileShapeQK{}) / params.scale_block_size : 0;
      /* GEMM 1: S = K * Q */
      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_, _, _, k_idx, D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_, _, _, VV, k_idx));
      }
      reorder(tSrS, tPrS);
      if constexpr (!CausalMask) {
        if (params.scale_block_size) {
          if (params.scale_block_size == 1) {
            // Need to get global col and row indices to mask the elements
            Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
            Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
            auto cS_thread = thr_mma_qk.partition_C(gP);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tPrS.size(); ++i) {
              int row_idx = get<0>(cS_thread(i));
              int col_idx = get<1>(cS_thread(i));
              tPrS(i) *= ElementP(scaleQ[row_idx] * scaleK[col_idx]);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tPrS.size(); ++i) {
              tPrS(i) *= params.scale;
            }
          } else {
            float _scale = dq_scale * scaleK[scalek_idx];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tPrS.size(); i++) tPrS(i) *= _scale;
          }
        }
      } else {
        float _scale = dq_scale * scaleK[scalek_idx];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tPrS.size(); i++) tPrS(i) *= _scale;
      }

      /* Causal masking - only in non-cache mode */
      if constexpr (!is_cache && CausalMask) {
        if (K == total_blk - 1) {
          // Need to get global col and row indices to mask the elements
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tPrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - seq_len_kv_cache - full_tile_offset > row_idx - discard_seq_coord) {
              tPrS(i) = ElementP(-INFINITY);
            }
          }
        }
      } else {
        if constexpr (FullMask_) {
          // Need to get global col and row indices to mask the elements
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          int row_idx_begin = get<0>(cS_thread(0));
          int row_idx_end = row_idx_begin + q_sg_tile;
          int col_idx_begin = get<1>(cS_thread(0));
          int col_idx_end = col_idx_begin + get<1>(TileShapeQK{});
          if (row_idx_end <= seq_len && col_idx_end <= seq_len) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tPrS.size(); ++i) {
              int row_idx = get<0>(cS_thread(i));
              int col_idx = get<1>(cS_thread(i));
              tPrS(i) += ElementP(params.mask[col_idx + row_idx * seq_len]);
            }
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tPrS.size(); ++i) {
              int row_idx = get<0>(cS_thread(i));
              int col_idx = get<1>(cS_thread(i));
              tPrS(i) += (row_idx < seq_len && col_idx < seq_len) ? ElementP(params.mask[col_idx + row_idx * seq_len])
                                                                  : ElementP(-INFINITY);
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if constexpr (!is_cache) {
        if (check_remainder_k && K == total_blk - 1) {
          FragPCol k_rem_mask;
          int k_val = get<0>(tKgK_cur(0, 0, 0, k_idx, 0)) + kblocks_cache * get<1>(TileShapeQK{});
          int k = k_val + get_sub_group().get_local_id()[0];
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
            k_rem_mask(i) = (k < seq_len) ? ElementP(sycl::nan(0u)) : ElementP(-INFINITY);
          }
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tPrS.size(); i++) {
            tPrS(i) = sycl::fmin(tPrS(i), broadcast<1>(k_rem_mask, tPrS, i));
          }
        }
      }

      /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop) */
      auto rescale = softmax(K == blk_k0, tPrS, tA_max, tA_sum);
      if constexpr (!UseInt8PV) {
        reorder(tPrS, tArP);
      } else {
        reorder(tPrS, tArP_i8);
      }

      /* GEMM 2: A += P * V, split in v dimension.
        tArA rescaling is fused to per-VTile */
      if constexpr (!UseInt8PV) {
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v_cur, tVgV_cur(_, _, _, VV, k_idx), tVrV);
          reorder(tVrV, tArV);
          if (K != blk_k0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++) tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
          }

          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        }
      } else {
        constexpr ElementA kInvVQuantScale = ElementA(1.0f / 127.0f);
        int scalev_head_dim = int(size<0>(V_2D));
        int v_block_base = int(get<1>(blk_qv)) * int(get<1>(TileShapePV{})) * VTiles;
        int scalev_block_base = scalek_idx * scalev_head_dim;
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          auto tArA_v = tArA(_, _, _, VV);
          if (K != blk_k0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++) tArA_v(i) *= broadcast<0>(rescale, tArA, i);
          }

          auto tArAcc = thr_mma_pv_i8.partition_sg_fragment_C(cPV);
          auto tVrV_i8 = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
          auto tArV_i8 = thr_mma_pv_i8.partition_sg_fragment_B(gV_split(_, _, 0, 0));
          copy(copy_v_cur, tVgV_cur(_, _, _, VV, k_idx), tVrV_i8);
          reorder(tVrV_i8, tArV_i8);
          clear(tArAcc);

          if constexpr (ExecuteInt8PV) {
            cute::gemm(mma_pv_i8, tArP_i8, tArV_i8, tArAcc);
          }

          if constexpr (WriteBackInt8PV) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArAcc.size(); ++i) {
              int local_v = int(get<1>(tCrA(i)));
              int scalev_idx = scalev_block_base + v_block_base + VV * int(get<1>(TileShapePV{})) + local_v;
              tArA_v(i) += ElementA(tArAcc(i)) * ElementA(scaleV[scalev_idx]) * kInvVQuantScale;
            }
          }
        }
      }

      /* K prefetch */
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

    /* Main loop, blocked in k. */
    if constexpr (CachedKV) {
      for (int K = blk_k0; K < kblocks_cache; K++) {
        mainloop_body(std::bool_constant<true>{}, K, copy_k_cache, copy_v_cache, prefetch_v_cache, tKgK_cache,
                      tVgV_cache, pVgV_cache);
      }
    }

    for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
      mainloop_body(std::bool_constant<false>{}, K, copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragARow softmax(bool first_block,    // First softmax block?
                   FragP& tS,           // Softmax src/dst block
                   FragARow& tS_max,    // Softmax row-wise max accumulator
                   FragARow& tS_sum) {   // Softmax row-wise sum accumulator
                                         /* Compute row-wise maxima for this block */

    auto tS_bmax = reduce<1>(tS, sycl::maximum{});
    FragARow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementA new_max = sycl::max(tS_max(i), ElementA(tS_bmax(i)));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    static_assert(FragP{}.size() % 2 == 0, "FragP size must be even for pairwise SIMD32 exp.");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i += 2) {
      ElementP x0 = tS(i) - broadcast<0>(tS_max, tS, i);
      ElementP x1 = tS(i + 1) - broadcast<0>(tS_max, tS, i + 1);
      exp2_pair_simd32_asm(x0, x1, tS(i), tS(i + 1));
    }

    if constexpr (UseInt8PV) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS.size(); i++) {
        tS(i) *= ElementP(127.0f);
      }
    }

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    if constexpr (UseInt8PV) {
      constexpr ElementA kInvPQuantScale = ElementA(1.0f / 127.0f);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) tS_sum(i) += ElementA(tS_bsum(i)) * kInvPQuantScale;
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) tS_sum(i) += tS_bsum(i);
    }

    return rescale;
  }
};
}  // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////