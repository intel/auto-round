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

#ifndef ARK_CUTLASS_SAGE_XEDEFAULT_DEFINED
#define ARK_CUTLASS_SAGE_XEDEFAULT_DEFINED
template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.
#endif

enum class SparseProfileMode : int {
  Full = 0,
  QkOnly = 1,
  QkSoftmaxOnly = 2,
  PvOnlySynthetic = 3,
  SoftmaxOnlySynth = 4,
  PvOnlyRealish = 5,
  QkPlusPvNoSoftmax = 6,
  PvReorderOnly = 7,
  PvLoadVOnly = 8,
  PvMmaOnly = 9,
  PvReorderPlusMma = 10,
};

};  // namespace cutlass::sage

namespace cutlass::fmha::collective {

using namespace cute;

#ifndef ARK_SPARSE_SAGE_ENABLE_K_PREFETCH
#define ARK_SPARSE_SAGE_ENABLE_K_PREFETCH 0
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_, bool UseInt8PV_,
          bool WriteBackInt8PV_, bool ExecuteInt8PV_,
          cutlass::sage::SparseProfileMode SparseProfileMode_,
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
struct SPARSESAGEV1FwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_, bool UseInt8PV_,
          bool WriteBackInt8PV_, bool ExecuteInt8PV_, cutlass::sage::SparseProfileMode SparseProfileMode_,
          class TiledMMAQK_,
          class TiledMMAPV_, int VTiles_, class TensorQ_, class TensorK_, class TensorV_, class TensorK_cache_,
          class TensorV_cache_, class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_, class TiledCopyK_cache_,
          class TiledCopyV_cache_>
struct SPARSESAGEV1FwdMainloop<sage::XeDefault<Stages>, CausalMask_, FullMask_, CachedKV_, PagedKV_, UseInt8PV_,
                         WriteBackInt8PV_, ExecuteInt8PV_, SparseProfileMode_, TiledMMAQK_, TiledMMAPV_, VTiles_, TensorQ_, TensorK_,
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
  static constexpr bool EnableSparseKPrefetch = ARK_SPARSE_SAGE_ENABLE_K_PREFETCH != 0;
  static constexpr cutlass::sage::SparseProfileMode ProfileMode = SparseProfileMode_;
  static constexpr bool ProfileRunsRealQk =
      ProfileMode == cutlass::sage::SparseProfileMode::Full ||
      ProfileMode == cutlass::sage::SparseProfileMode::QkOnly ||
      ProfileMode == cutlass::sage::SparseProfileMode::QkSoftmaxOnly ||
      ProfileMode == cutlass::sage::SparseProfileMode::QkPlusPvNoSoftmax;
  static constexpr bool ProfileRequiresPV =
      ProfileMode == cutlass::sage::SparseProfileMode::Full ||
      ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
      ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
      ProfileMode == cutlass::sage::SparseProfileMode::QkPlusPvNoSoftmax ||
      ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
      ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
      ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
      ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma;
  // The qtile64 Q-staging experiment stays disabled until the Xe SLM copy layout
  // matches the fragment layout expected by make_A_slm_copies().
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
    int const* lut = nullptr;
    int const* valid_block_num = nullptr;
    int num_q_blocks = 0;
    int num_k_blocks = 0;
    int sparse_q_block_size = 0;
    bool canonical_nhd_k = false;
    int const* ptr_page_table = nullptr;
    int page_size = 0;
    int const* num_pages_per_seq = nullptr;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct EmptySharedStorage {};
  using SharedStorage = EmptySharedStorage;

  Params params;
  SharedStorage& shared_storage;

  //
  // Methods
  //

  SPARSESAGEV1FwdMainloop(Params const& params_, SharedStorage& shared_storage_)
      : params(params_), shared_storage(shared_storage_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    float val = args.scale * static_cast<float>(kLog2e);
    return Params{val, args.mask, args.scale_block_size, args.qscale, args.kscale, args.vscale,
            args.lut, args.valid_block_num, args.num_q_blocks, args.num_k_blocks, args.sparse_q_block_size,
            args.canonical_nhd_k,
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

  CUTLASS_DEVICE
  static int logical_block_from_delta_row(int const* row, int valid_blocks, int idx) {
    int logical_block = 0;
    for (int i = 0; i <= idx && i < valid_blocks; ++i) {
      logical_block += row[i];
    }
    return logical_block;
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
                                 int const* lut_rows_base = nullptr, int const* valid_blocks_base = nullptr,
                                 int sparse_q_rows_in_tile = 1,
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
    TiledCopyK_cache copy_k_cache{K_cache_2D};
    [[maybe_unused]] TiledCopyV copy_v{V_2D};
    [[maybe_unused]] TiledCopyV_cache copy_v_cache{V_cache_2D};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    [[maybe_unused]] TiledMMAPV mma_pv{};
    [[maybe_unused]] QuantizedTiledMMAPV mma_pv_i8{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    [[maybe_unused]] auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_copy_k_cache = copy_k_cache.get_slice(thr_id);
    [[maybe_unused]] auto thr_copy_v_cache = copy_v_cache.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    [[maybe_unused]] auto thr_mma_pv = mma_pv.get_slice(thr_id);
    [[maybe_unused]] auto thr_mma_pv_i8 = mma_pv_i8.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    [[maybe_unused]] auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)
    auto tKgK_cache = thr_copy_k_cache.partition_S(gK_cache);
    [[maybe_unused]] auto tVgV_cache = thr_copy_v_cache.partition_S(gV_cache_split);

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);

    FragP tPrS = make_subgroup_tensor(make_fragment_like<ElementP>(tSrS.tensor()), tSrS.tv_layout());

    [[maybe_unused]] auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    [[maybe_unused]] auto tArP_i8 = thr_mma_pv_i8.partition_sg_fragment_A(cP);

    [[maybe_unused]] auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    [[maybe_unused]] auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));
    [[maybe_unused]] auto tCrA = thr_mma_pv_i8.partition_C(cPV);

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    [[maybe_unused]] auto prefetch_v = make_block_2d_prefetch(copy_v);
    auto prefetch_k_cache = make_block_2d_prefetch(copy_k_cache);
    [[maybe_unused]] auto prefetch_v_cache = make_block_2d_prefetch(copy_v_cache);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    [[maybe_unused]] auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_split);
    auto pKgK_cache = prefetch_k_cache.get_slice(thr_id).partition_S(gK_cache);
    [[maybe_unused]] auto pVgV_cache = prefetch_v_cache.get_slice(thr_id).partition_S(gV_cache_split);

    // ------
    // Kernel
    // ------

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    int kblocks_cache = ceil_div(seq_len_kv_cache, get<1>(TileShapeQK{}));
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }
    if (lut_rows_base == nullptr) {
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
    }
    if (blk_k0 == 0) {
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));
    int sparse_q_block_size = params.sparse_q_block_size > 0 ? params.sparse_q_block_size : params.scale_block_size;
    int q_blocks_per_wg_tile =
        sparse_q_block_size > 0 ? cute::max(1, int(get<0>(TileShapeQK{})) / sparse_q_block_size) : 1;
    int sg_rows_per_sparse_q_block =
        sparse_q_block_size > 0 ? cute::max(1, sparse_q_block_size / q_sg_tile) : 1;
    int subgroup_q_row_in_tile = get_sub_group_id() / sg_rows_per_sparse_q_block;
    subgroup_q_row_in_tile = cute::min(subgroup_q_row_in_tile, q_blocks_per_wg_tile - 1);

    float dq_scale = params.scale_block_size
                         ? scaleQ[(get<0>(blk_qv) * get<0>(TileShapeQK{}) + get_sub_group_id() * q_sg_tile) /
                                  params.scale_block_size] *
                               params.scale
                         : 1.f;
    auto prefetch_sparse_k_block = [&](int logical_block) {
      if (logical_block < 0 || logical_block >= total_blk) return;
      for (int D = 0; D < size<4>(pKgK); D++) {
        if constexpr (CachedKV) {
          if (logical_block < kblocks_cache) {
            int physical_block = logical_block;
            if constexpr (PagedKV) {
              physical_block = get_physical_k_tile(logical_block, l_coord, seq_len_kv_cache);
            }
            prefetch(prefetch_k_cache, pKgK_cache(_, _, _, physical_block, D));
          } else {
            prefetch(prefetch_k, pKgK(_, _, _, logical_block - kblocks_cache, D));
          }
        } else {
          prefetch(prefetch_k, pKgK(_, _, _, logical_block - kblocks_cache, D));
        }
      }
    };
    auto prefetch_next_k_block = [&](auto cached_k, int K) {
      constexpr bool is_cache = decltype(cached_k)::value;
      if (lut_rows_base == nullptr) {
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
      }
    };
    auto write_qk_profile_proxy = [&](bool first_block) {
      if (first_block) {
        clear(tArA);
        fill(tA_max, ElementA(0));
        fill(tA_sum, ElementA(1));
      }
      ElementA proxy = ElementA(0);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tPrS.size(); ++i) {
        proxy += ElementA(tPrS(i));
      }
      tArA(0) += proxy;
    };

    auto init_profile_pv_state = [&](bool first_block) {
      if (first_block) {
        clear(tArA);
        fill(tA_max, ElementA(0));
        fill(tA_sum, ElementA(1));
      }
    };

    auto fill_softmax_only_synth_scores = [&]() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tPrS.size(); ++i) {
        switch (i & 3) {
          case 0:
            tPrS(i) = ElementP(0.75f);
            break;
          case 1:
            tPrS(i) = ElementP(-0.25f);
            break;
          case 2:
            tPrS(i) = ElementP(0.125f);
            break;
          default:
            tPrS(i) = ElementP(-0.5f);
            break;
        }
      }
    };

    auto fill_pv_only_realish_probs = [&]() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tPrS.size(); ++i) {
        switch (i % 5) {
          case 0:
            tPrS(i) = ElementP(0.3125f);
            break;
          case 1:
            tPrS(i) = ElementP(0.1875f);
            break;
          case 2:
            tPrS(i) = ElementP(0.1250f);
            break;
          case 3:
            tPrS(i) = ElementP(0.0625f);
            break;
          default:
            tPrS(i) = ElementP(0.03125f);
            break;
        }
      }
    };

    auto prepare_qk_scores_for_pv = [&]() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tPrS.size(); ++i) {
        ElementP shifted = tPrS(i) * ElementP(0.0625f) + ElementP(0.5f);
        tPrS(i) = sycl::fmin(ElementP(1.0f), sycl::fmax(ElementP(0.0f), shifted));
      }
    };

    auto fill_pv_mma_only_probs = [&]() {
      if constexpr (!UseInt8PV) {
        using ElementPV = std::remove_reference_t<decltype(tArP(0))>;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tArP.size(); ++i) {
          tArP(i) = ElementPV((i & 3) == 0 ? 0.3125f : (i & 3) == 1 ? 0.1875f : (i & 3) == 2 ? 0.125f : 0.0625f);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tArP_i8.size(); ++i) {
          tArP_i8(i) = int8_t((i & 3) == 0 ? 40 : (i & 3) == 1 ? 24 : (i & 3) == 2 ? 16 : 8);
        }
      }
    };

    auto sink_reordered_p = [&]() {
      ElementA proxy = ElementA(0);
      if constexpr (!UseInt8PV) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tArP.size(); ++i) {
          proxy += ElementA(tArP(i));
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tArP_i8.size(); ++i) {
          proxy += ElementA(tArP_i8(i));
        }
      }
      tArA(0) += proxy;
    };

    auto sink_loaded_v = [&]() {
      ElementA proxy = ElementA(0);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArV.size(); ++i) {
        proxy += ElementA(tArV(i));
      }
      tArA(0) += proxy;
    };

    auto run_qk_math = [&](auto cached_k, int K, bool first_block, bool subgroup_selected, int sparse_prefetch_block,
                           auto& copy_k_cur, auto& tKgK_cur) {
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

      if constexpr (ProfileMode != cutlass::sage::SparseProfileMode::SoftmaxOnlySynth) {
        clear(tSrS);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(tKgK); D++) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          copy(copy_k_cur, tKgK_cur(_, _, _, k_idx, D), tKrK);
          reorder(tQrQ, tSrQ);
          reorder(tKrK, tSrK);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
      }

      if (subgroup_selected) {
        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::SoftmaxOnlySynth) {
          fill_softmax_only_synth_scores();
        } else {
          reorder(tSrS, tPrS);
          if constexpr (!CausalMask) {
            if (params.scale_block_size) {
              if (params.scale_block_size == 1) {
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

          if constexpr (!is_cache && CausalMask) {
            if (K == total_blk - 1) {
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
                  tPrS(i) += (row_idx < seq_len && col_idx < seq_len)
                                 ? ElementP(params.mask[col_idx + row_idx * seq_len])
                                 : ElementP(-INFINITY);
                }
              }
            }
          }

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
        }

        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::QkOnly) {
          write_qk_profile_proxy(first_block);
        } else {
          auto rescale = softmax(first_block, tPrS, tA_max, tA_sum);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tPrS.size(); ++i) {
            tPrS(i) *= broadcast<0>(rescale, tPrS, i);
          }
          write_qk_profile_proxy(first_block);
        }
      }

      if (lut_rows_base == nullptr) {
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
      } else if constexpr (EnableSparseKPrefetch) {
        if (sparse_prefetch_block < total_blk) {
          prefetch_sparse_k_block(sparse_prefetch_block);
        }
      }

      barrier_wait(ScopeWorkgroup);
    };

    auto run_pv_only_math = [&](auto cached_k, int K, bool first_block, bool subgroup_selected, int sparse_prefetch_block,
                                auto& copy_v_cur, auto& prefetch_v_cur, auto& tVgV_cur, auto& pVgV_cur) {
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

      /* V prefetch for GEMM 2 stays workgroup-cooperative in the PV-only path. */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_, _, _, VV, k_idx));
      }

      if (subgroup_selected) {
        init_profile_pv_state(first_block);
        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                      ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                      ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tPrS.size(); ++i) {
            tPrS(i) = ElementP(((i & 1) == 0) ? 0.25f : 0.125f);
          }
        } else {
          fill_pv_only_realish_probs();
        }

        if constexpr (!UseInt8PV) {
          reorder(tPrS, tArP);
        } else {
          reorder(tPrS, tArP_i8);
        }

        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                      ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                      ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
          if constexpr (UseInt8PV) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArP_i8.size(); ++i) {
              tArP_i8(i) = int8_t((i & 1) == 0 ? 32 : 16);
            }
          }
        }

        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly) {
          sink_reordered_p();
          goto finish_pv_only_profile;
        }

        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly) {
          CUTLASS_PRAGMA_UNROLL
          for (int VV = 0; VV < VTiles; VV++) {
            copy(copy_v_cur, tVgV_cur(_, _, _, VV, k_idx), tVrV);
            reorder(tVrV, tArV);
            sink_loaded_v();
          }
          goto finish_pv_only_profile;
        }

        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly) {
          fill_pv_mma_only_probs();
        }

        /* GEMM 2: A += P * V, split in v dimension. */
        if constexpr (!UseInt8PV) {
          CUTLASS_PRAGMA_UNROLL
          for (int VV = 0; VV < VTiles; VV++) {
            copy(copy_v_cur, tVgV_cur(_, _, _, VV, k_idx), tVrV);
            reorder(tVrV, tArV);
            cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
          }
        } else {
          constexpr ElementA kInvVQuantScale = ElementA(1.0f / 127.0f);
          int scalev_head_dim = int(size<0>(V_2D));
          int v_block_base = int(get<1>(blk_qv)) * int(get<1>(TileShapePV{})) * VTiles;
          int scalev_block_base = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int VV = 0; VV < VTiles; VV++) {
            auto tArA_v = tArA(_, _, _, VV);
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
      }

finish_pv_only_profile:

      if (lut_rows_base == nullptr) {
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
      } else if constexpr (EnableSparseKPrefetch) {
        if (sparse_prefetch_block < total_blk) {
          prefetch_sparse_k_block(sparse_prefetch_block);
        }
      }

      barrier_wait(ScopeWorkgroup);
    };

    /* Main loop body */
      auto mainloop_body = [&](auto cached_k, int K, bool first_block, bool subgroup_selected,
                               int sparse_prefetch_block, auto& copy_k_cur, auto& copy_v_cur, auto& prefetch_v_cur,
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

        if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::Full) {
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
          if (subgroup_selected) {
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
                    tPrS(i) += (row_idx < seq_len && col_idx < seq_len)
                                   ? ElementP(params.mask[col_idx + row_idx * seq_len])
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
            auto rescale = softmax(first_block, tPrS, tA_max, tA_sum);
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
                if (!first_block) {
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
                if (!first_block) {
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
          }
        } else {
          if constexpr (ProfileRunsRealQk) {
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
          }

          if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                        ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                        ProfileMode == cutlass::sage::SparseProfileMode::QkPlusPvNoSoftmax) {
            /* V prefetch for GEMM 2 stays workgroup-cooperative in the PV-only path. */
            CUTLASS_PRAGMA_UNROLL
            for (int VV = 0; VV < VTiles; VV++) {
              prefetch(prefetch_v_cur, pVgV_cur(_, _, _, VV, k_idx));
            }
          }

          auto write_profile_proxy = [&]() {
            if (first_block) {
              clear(tArA);
              fill(tA_max, ElementA(0));
              fill(tA_sum, ElementA(1));
            }
            ElementA proxy = ElementA(0);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tPrS.size(); ++i) {
              proxy += ElementA(tPrS(i));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size(); ++i) {
              tArA(i) += proxy;
            }
          };

          if (subgroup_selected) {
            if constexpr (ProfileRunsRealQk) {
              reorder(tSrS, tPrS);
              if constexpr (!CausalMask) {
                if (params.scale_block_size) {
                  if (params.scale_block_size == 1) {
                    // Need to get global col and row indices to mask the elements.
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
            } else if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic) {
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < tPrS.size(); ++i) {
                tPrS(i) = ElementP(((i & 1) == 0) ? 0.25f : 0.125f);
              }
            } else {
              fill_pv_only_realish_probs();
            }

            if constexpr (ProfileRunsRealQk) {
              /* Causal masking - only in non-cache mode */
              if constexpr (!is_cache && CausalMask) {
                if (K == total_blk - 1) {
                  // Need to get global col and row indices to mask the elements.
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
                  // Need to get global col and row indices to mask the elements.
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
                      tPrS(i) += (row_idx < seq_len && col_idx < seq_len)
                                     ? ElementP(params.mask[col_idx + row_idx * seq_len])
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
            }

            if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::QkOnly) {
              write_profile_proxy();
              prefetch_next_k_block(cached_k, K);
              barrier_wait(ScopeWorkgroup);
              return;
            }

            if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                          ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                          ProfileMode == cutlass::sage::SparseProfileMode::QkPlusPvNoSoftmax) {
              init_profile_pv_state(first_block);
              if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::QkPlusPvNoSoftmax) {
                prepare_qk_scores_for_pv();
              }
              if constexpr (!UseInt8PV) {
                reorder(tPrS, tArP);
              } else {
                reorder(tPrS, tArP_i8);
              }
              if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic) {
                if constexpr (UseInt8PV) {
                  CUTLASS_PRAGMA_UNROLL
                  for (int i = 0; i < tArP_i8.size(); ++i) {
                    tArP_i8(i) = int8_t((i & 1) == 0 ? 32 : 16);
                  }
                }
              }
            } else {
              /* Apply softmax and scaling (tA rescaling fused into GEMM2 VTile loop). */
              auto rescale = softmax(first_block, tPrS, tA_max, tA_sum);
              if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::QkSoftmaxOnly) {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < tPrS.size(); ++i) {
                  tPrS(i) *= broadcast<0>(rescale, tPrS, i);
                }
                write_profile_proxy();
                prefetch_next_k_block(cached_k, K);
                barrier_wait(ScopeWorkgroup);
                return;
              }

              if constexpr (!UseInt8PV) {
                reorder(tPrS, tArP);
              } else {
                reorder(tPrS, tArP_i8);
              }
            }

            /* GEMM 2: A += P * V, split in v dimension.
              tArA rescaling is fused to per-VTile. */
            if constexpr (!UseInt8PV) {
              CUTLASS_PRAGMA_UNROLL
              for (int VV = 0; VV < VTiles; VV++) {
                copy(copy_v_cur, tVgV_cur(_, _, _, VV, k_idx), tVrV);
                reorder(tVrV, tArV);
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
          }
        }

        /* K prefetch */
        if (lut_rows_base == nullptr) {
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
        } else if constexpr (EnableSparseKPrefetch) {
          if (sparse_prefetch_block < total_blk) {
            prefetch_sparse_k_block(sparse_prefetch_block);
          }
        }

        barrier_wait(ScopeWorkgroup);
      };

      /* Main loop, blocked in k. */
      if (lut_rows_base != nullptr && valid_blocks_base != nullptr) {
        bool subgroup_started = false;
        if (sparse_q_rows_in_tile == 1) {
          int const* row_ptr = lut_rows_base;
          int row_valid = valid_blocks_base[0];
          int row_pos = 0;
          int row_cur_block = row_valid > 0 ? logical_block_from_delta_row(row_ptr, row_valid, 0) : total_blk;
          int prefetch_pos = row_pos;
          int prefetch_cur_block = row_cur_block;

          auto advance_single_sparse_block = [&](int& frontier_pos, int& frontier_cur_block) {
            if (frontier_cur_block >= total_blk) return;
            frontier_pos += 1;
            if (frontier_pos < row_valid) {
              frontier_cur_block += row_ptr[frontier_pos];
            } else {
              frontier_cur_block = total_blk;
            }
          };

          auto pop_single_sparse_block = [&](int& frontier_pos, int& frontier_cur_block) {
            int block = frontier_cur_block;
            advance_single_sparse_block(frontier_pos, frontier_cur_block);
            return block;
          };

          if constexpr (EnableSparseKPrefetch) {
            for (int stage = 0; stage < Stages; ++stage) {
              int sparse_prefetch_block = pop_single_sparse_block(prefetch_pos, prefetch_cur_block);
              if (sparse_prefetch_block >= total_blk) break;
              prefetch_sparse_k_block(sparse_prefetch_block);
            }
          }

          while (row_cur_block < total_blk) {
            int next_block = row_cur_block;
            bool subgroup_selected = (subgroup_q_row_in_tile == 0);
            bool first_selected_block = !subgroup_started;
            int sparse_prefetch_block = total_blk;
            if constexpr (EnableSparseKPrefetch) {
              sparse_prefetch_block = pop_single_sparse_block(prefetch_pos, prefetch_cur_block);
            }
            int K = next_block;
            if constexpr (CachedKV) {
              if (K < kblocks_cache) {
                if (K >= blk_k0 && K < blk_k1) {
                  if constexpr (ProfileRequiresPV) {
                    if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
                      run_pv_only_math(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                                       sparse_prefetch_block, copy_v_cache, prefetch_v_cache, tVgV_cache, pVgV_cache);
                    } else {
                      mainloop_body(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                                    sparse_prefetch_block, copy_k_cache, copy_v_cache, prefetch_v_cache, tKgK_cache,
                                    tVgV_cache, pVgV_cache);
                    }
                  } else {
                    run_qk_math(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                                sparse_prefetch_block, copy_k_cache, tKgK_cache);
                  }
                  subgroup_started = true;
                }
              } else {
                K += kblocks_cache;
                if (K >= (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache) && K < blk_k1) {
                  if constexpr (ProfileRequiresPV) {
                    if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
                      run_pv_only_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                       sparse_prefetch_block, copy_v, prefetch_v, tVgV, pVgV);
                    } else {
                      mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                    sparse_prefetch_block, copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
                    }
                  } else {
                    run_qk_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                sparse_prefetch_block, copy_k, tKgK);
                  }
                  subgroup_started = true;
                }
              }
            } else {
              if (K >= blk_k0 && K < blk_k1) {
                if constexpr (ProfileRequiresPV) {
                  if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
                    run_pv_only_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                     sparse_prefetch_block, copy_v, prefetch_v, tVgV, pVgV);
                  } else {
                    mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                  sparse_prefetch_block, copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
                  }
                } else {
                  run_qk_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                              sparse_prefetch_block, copy_k, tKgK);
                }
                subgroup_started = true;
              }
            }

            advance_single_sparse_block(row_pos, row_cur_block);
          }
        } else {
          static constexpr int kMaxSparseRowsPerTile = cute::max(1, int(get<0>(TileShapeQK{})) / 64);
          int const* row_ptrs[kMaxSparseRowsPerTile];
          int active_rows[kMaxSparseRowsPerTile];
          int active_row_count = 0;
          int row_valid[kMaxSparseRowsPerTile];
          int row_pos[kMaxSparseRowsPerTile];
          int row_cur_block[kMaxSparseRowsPerTile];

          for (int row = 0; row < kMaxSparseRowsPerTile; ++row) {
            row_ptrs[row] = lut_rows_base + row * params.num_k_blocks;
            if (row < sparse_q_rows_in_tile) {
              row_valid[row] = valid_blocks_base[row];
              row_pos[row] = 0;
              row_cur_block[row] =
                  row_valid[row] > 0 ? logical_block_from_delta_row(row_ptrs[row], row_valid[row], 0) : total_blk;
              if (row_valid[row] > 0) {
                active_rows[active_row_count++] = row;
              }
            } else {
              row_valid[row] = 0;
              row_pos[row] = 0;
              row_cur_block[row] = total_blk;
            }
          }

          auto find_sparse_block = [&](int* frontier_pos, int* frontier_cur_block) {
            int block = total_blk;
            for (int active = 0; active < active_row_count; ++active) {
              int row = active_rows[active];
              if (frontier_pos[row] < row_valid[row]) {
                block = cute::min(block, frontier_cur_block[row]);
              }
            }
            return block;
          };

          auto advance_sparse_block = [&](int block, int* frontier_pos, int* frontier_cur_block) {
            if (block >= total_blk) return;
            for (int active = 0; active < active_row_count; ++active) {
              int row = active_rows[active];
              if (frontier_pos[row] < row_valid[row] && frontier_cur_block[row] == block) {
                frontier_pos[row] += 1;
                if (frontier_pos[row] < row_valid[row]) {
                  frontier_cur_block[row] += row_ptrs[row][frontier_pos[row]];
                } else {
                  frontier_cur_block[row] = total_blk;
                }
              }
            }
          };

          auto pop_sparse_block = [&](int* frontier_pos, int* frontier_cur_block) {
            int block = find_sparse_block(frontier_pos, frontier_cur_block);
            advance_sparse_block(block, frontier_pos, frontier_cur_block);
            return block;
          };

          int prefetch_pos[kMaxSparseRowsPerTile];
          int prefetch_cur_block[kMaxSparseRowsPerTile];
          if constexpr (EnableSparseKPrefetch) {
            for (int row = 0; row < kMaxSparseRowsPerTile; ++row) {
              prefetch_pos[row] = row_pos[row];
              prefetch_cur_block[row] = row_cur_block[row];
            }

            for (int stage = 0; stage < Stages; ++stage) {
              int sparse_prefetch_block = pop_sparse_block(prefetch_pos, prefetch_cur_block);
              if (sparse_prefetch_block >= total_blk) break;
              prefetch_sparse_k_block(sparse_prefetch_block);
            }
          }

          int next_block = find_sparse_block(row_pos, row_cur_block);
          while (next_block < total_blk) {
            bool subgroup_selected = subgroup_q_row_in_tile < sparse_q_rows_in_tile &&
                                     row_pos[subgroup_q_row_in_tile] < row_valid[subgroup_q_row_in_tile] &&
                                     row_cur_block[subgroup_q_row_in_tile] == next_block;
            bool first_selected_block = !subgroup_started;
            int sparse_prefetch_block = total_blk;
            if constexpr (EnableSparseKPrefetch) {
              sparse_prefetch_block = pop_sparse_block(prefetch_pos, prefetch_cur_block);
            }
            int K = next_block;
            if constexpr (CachedKV) {
              if (K < kblocks_cache) {
                if (K >= blk_k0 && K < blk_k1) {
                  if constexpr (ProfileRequiresPV) {
                    if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
                      run_pv_only_math(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                                       sparse_prefetch_block, copy_v_cache, prefetch_v_cache, tVgV_cache, pVgV_cache);
                    } else {
                      mainloop_body(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                                    sparse_prefetch_block, copy_k_cache, copy_v_cache, prefetch_v_cache, tKgK_cache,
                                    tVgV_cache, pVgV_cache);
                    }
                  } else {
                    run_qk_math(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                                sparse_prefetch_block, copy_k_cache, tKgK_cache);
                  }
                  subgroup_started = subgroup_started || subgroup_selected;
                }
              } else {
                K += kblocks_cache;
                if (K >= (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache) && K < blk_k1) {
                  if constexpr (ProfileRequiresPV) {
                    if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                                  ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
                      run_pv_only_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                       sparse_prefetch_block, copy_v, prefetch_v, tVgV, pVgV);
                    } else {
                      mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                    sparse_prefetch_block, copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
                    }
                  } else {
                    run_qk_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                sparse_prefetch_block, copy_k, tKgK);
                  }
                  subgroup_started = subgroup_started || subgroup_selected;
                }
              }
            } else {
              if (K >= blk_k0 && K < blk_k1) {
                if constexpr (ProfileRequiresPV) {
                  if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                                ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
                    run_pv_only_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                     sparse_prefetch_block, copy_v, prefetch_v, tVgV, pVgV);
                  } else {
                    mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                                  sparse_prefetch_block, copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
                  }
                } else {
                  run_qk_math(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                              sparse_prefetch_block, copy_k, tKgK);
                }
                subgroup_started = subgroup_started || subgroup_selected;
              }
            }

            advance_sparse_block(next_block, row_pos, row_cur_block);
            next_block = find_sparse_block(row_pos, row_cur_block);
          }
        }
      } else {
        if constexpr (CachedKV) {
          for (int K = blk_k0; K < kblocks_cache; K++) {
            if constexpr (ProfileRequiresPV) {
              if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                            ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                            ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                            ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                            ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                            ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
                run_pv_only_math(std::bool_constant<true>{}, K, K == blk_k0, true, total_blk, copy_v_cache,
                                 prefetch_v_cache, tVgV_cache, pVgV_cache);
              } else {
                mainloop_body(std::bool_constant<true>{}, K, K == blk_k0, true, total_blk, copy_k_cache, copy_v_cache,
                              prefetch_v_cache, tKgK_cache, tVgV_cache, pVgV_cache);
              }
            } else {
              run_qk_math(std::bool_constant<true>{}, K, K == blk_k0, true, total_blk, copy_k_cache, tKgK_cache);
            }
          }
        }
        for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
          if constexpr (ProfileRequiresPV) {
            if constexpr (ProfileMode == cutlass::sage::SparseProfileMode::PvOnlySynthetic ||
                          ProfileMode == cutlass::sage::SparseProfileMode::PvOnlyRealish ||
                          ProfileMode == cutlass::sage::SparseProfileMode::PvReorderOnly ||
                          ProfileMode == cutlass::sage::SparseProfileMode::PvLoadVOnly ||
                          ProfileMode == cutlass::sage::SparseProfileMode::PvMmaOnly ||
                          ProfileMode == cutlass::sage::SparseProfileMode::PvReorderPlusMma) {
              run_pv_only_math(std::bool_constant<false>{}, K,
                               K == (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache), true, total_blk, copy_v,
                               prefetch_v, tVgV, pVgV);
            } else {
              mainloop_body(std::bool_constant<false>{}, K,
                            K == (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache), true, total_blk, copy_k, copy_v,
                            prefetch_v, tKgK, tVgV, pVgV);
            }
          } else {
            run_qk_math(std::bool_constant<false>{}, K,
                        K == (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache), true, total_blk, copy_k, tKgK);
          }
        }
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
