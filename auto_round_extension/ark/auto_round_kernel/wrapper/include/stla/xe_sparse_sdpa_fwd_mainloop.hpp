/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
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
#include <cute/util/xe_split_barrier.hpp>

namespace cutlass::sdpa {

template <int Stages>
class XeDefault {};

}  // namespace cutlass::sdpa

namespace cutlass::fmha::collective {

using namespace cute;

template <class DispatchPolicy_, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_, class TensorQ_, class TensorK_, class TensorV_,
          class TensorK_cache_, class TensorV_cache_, class TiledCopyQ_ = void, class TiledCopyK_ = void,
          class TiledCopyV_ = void, class TiledCopyK_cache_ = void, class TiledCopyV_cache_ = void>
struct SPARSESDPAFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

template <int Stages, bool CausalMask_, bool FullMask_, bool CachedKV_, bool PagedKV_, class TiledMMAQK_,
          class TiledMMAPV_, int VTiles_, class TensorQ_, class TensorK_, class TensorV_, class TensorK_cache_,
          class TensorV_cache_, class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_, class TiledCopyK_cache_,
          class TiledCopyV_cache_>
struct SPARSESDPAFwdMainloop<cutlass::sdpa::XeDefault<Stages>, CausalMask_, FullMask_, CachedKV_, PagedKV_,
                             TiledMMAQK_, TiledMMAPV_, VTiles_, TensorQ_, TensorK_, TensorV_, TensorK_cache_,
                             TensorV_cache_, TiledCopyQ_, TiledCopyK_, TiledCopyV_, TiledCopyK_cache_,
                             TiledCopyV_cache_> {
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
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

  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;
  using ElementM = typename TiledMMAQK::ValTypeA;

  using SingleFragA = FragC<TiledMMAPV>;
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool CachedKV = CachedKV_;
  static constexpr bool PagedKV = PagedKV_;

  struct Arguments {
    float const scale;
    float const* mask = nullptr;
    int scale_block_size = 0;
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

  using Params = Arguments;
  struct EmptySharedStorage {};
  using SharedStorage = EmptySharedStorage;

  Params params;
  SharedStorage& shared_storage;

  SPARSESDPAFwdMainloop(Params const& params_, SharedStorage& shared_storage_)
      : params(params_), shared_storage(shared_storage_) {}

  static constexpr Params to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;
    float val = args.scale * static_cast<float>(kLog2e);
    return Params{val, args.mask, args.scale_block_size, args.qscale, args.kscale, args.vscale, args.lut,
                  args.valid_block_num, args.num_q_blocks, args.num_k_blocks, args.sparse_q_block_size,
                  args.canonical_nhd_k, args.ptr_page_table, args.page_size, args.num_pages_per_seq};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) { return true; }

  CUTLASS_DEVICE
  int get_physical_k_tile(int K, int l_coord, int seq_len_kv_cache) {
    int next_page_logical_idx = K * get<1>(TileShapeQK{}) / params.page_size;
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
  CUTLASS_DEVICE void operator()(TensorQ2D const& Q_2D, TensorK2D const& K_2D, TensorV2D const& V_2D, FragA& tArA,
                                 FragARow& tA_max, FragARow& tA_sum, QVCoord blk_qv, int blk_k0, int blk_k1,
                                 int total_blk, int thr_id, int seq_len, int seq_len_kv_cache, int l_coord,
                                 [[maybe_unused]] float* scaleQ, [[maybe_unused]] float* scaleK,
                                 [[maybe_unused]] float* scaleV, int full_tile_offset, int discard_seq_coord,
                                 int const* lut_rows_base = nullptr, int const* valid_blocks_base = nullptr,
                                 int sparse_q_rows_in_tile = 1,
                                 TensorK_cache2D const& K_cache_2D = TensorK_cache2D{},
                                 TensorV_cache2D const& V_cache_2D = TensorV_cache2D{}) {
    using namespace sycl::ext::oneapi::this_work_item;

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor cK = make_identity_tensor(K_2D.shape());
    Tensor cV = make_identity_tensor(V_2D.shape());
    Tensor cK_cache = make_identity_tensor(K_cache_2D.shape());
    Tensor cV_cache = make_identity_tensor(V_cache_2D.shape());
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));

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

    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));
    int sparse_q_block_size = params.sparse_q_block_size > 0 ? params.sparse_q_block_size : params.scale_block_size;
    int q_blocks_per_wg_tile =
        sparse_q_block_size > 0 ? cute::max(1, int(get<0>(TileShapeQK{})) / sparse_q_block_size) : 1;
    int sg_rows_per_sparse_q_block =
        sparse_q_block_size > 0 ? cute::max(1, sparse_q_block_size / q_sg_tile) : 1;
    int subgroup_q_row_in_tile = get_sub_group_id() / sg_rows_per_sparse_q_block;
    subgroup_q_row_in_tile = cute::min(subgroup_q_row_in_tile, q_blocks_per_wg_tile - 1);

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

    auto mainloop_body = [&](auto cached_k, int K, bool first_block, bool subgroup_selected, int sparse_prefetch_block,
                             auto& copy_k_cur, auto& copy_v_cur, auto& prefetch_v_cur, auto& tKgK_cur,
                             auto& tVgV_cur, auto& pVgV_cur) {
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

      clear(tSrS);
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k_cur, tKgK_cur(_, _, _, k_idx, D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        prefetch(prefetch_v_cur, pVgV_cur(_, _, _, VV, k_idx));
      }

      if (subgroup_selected) {
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
        } else if constexpr (FullMask_) {
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
              tSrS(i) +=
                  (row_idx < seq_len && col_idx < seq_len) ? ElementS(params.mask[col_idx + row_idx * seq_len])
                                                            : ElementS(-INFINITY);
            }
          }
        }

        if constexpr (!is_cache) {
          if (check_remainder_k && K == total_blk - 1) {
            FragSCol k_rem_mask;
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

        auto rescale = softmax(first_block, tSrS, tA_max, tA_sum);
        reorder(tSrS, tArP);

        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v_cur, tVgV_cur(_, _, _, VV, k_idx), tVrV);
          reorder(tVrV, tArV);
          if (!first_block) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tArA.size() / VTiles; i++) {
              tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
            }
          }
          cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
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
      } else if (sparse_prefetch_block < total_blk) {
        prefetch_sparse_k_block(sparse_prefetch_block);
      }

      barrier_wait(ScopeWorkgroup);
    };

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

        for (int stage = 0; stage < Stages; ++stage) {
          int sparse_prefetch_block = pop_single_sparse_block(prefetch_pos, prefetch_cur_block);
          if (sparse_prefetch_block >= total_blk) break;
          prefetch_sparse_k_block(sparse_prefetch_block);
        }

        while (row_cur_block < total_blk) {
          int next_block = row_cur_block;
          bool subgroup_selected = (subgroup_q_row_in_tile == 0);
          bool first_selected_block = !subgroup_started;
          int sparse_prefetch_block = pop_single_sparse_block(prefetch_pos, prefetch_cur_block);
          int K = next_block;
          if constexpr (CachedKV) {
            if (K < kblocks_cache) {
              if (K >= blk_k0 && K < blk_k1) {
                mainloop_body(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                              sparse_prefetch_block, copy_k_cache, copy_v_cache, prefetch_v_cache, tKgK_cache,
                              tVgV_cache, pVgV_cache);
                subgroup_started = true;
              }
            } else {
              K += kblocks_cache;
              if (K >= (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache) && K < blk_k1) {
                mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                              sparse_prefetch_block, copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
                subgroup_started = true;
              }
            }
          } else if (K >= blk_k0 && K < blk_k1) {
            mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected, sparse_prefetch_block,
                          copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
            subgroup_started = true;
          }
          advance_single_sparse_block(row_pos, row_cur_block);
        }
      } else {
        static constexpr int kMaxSparseRowsPerTile = cute::max(1, int(get<0>(TileShapeQK{})) / 64);
        int const* row_ptrs[kMaxSparseRowsPerTile];
        int active_rows[kMaxSparseRowsPerTile];
        int active_row_count = 0;
        bool subgroup_started_rows[kMaxSparseRowsPerTile];
        int row_valid[kMaxSparseRowsPerTile];
        int row_pos[kMaxSparseRowsPerTile];
        int row_cur_block[kMaxSparseRowsPerTile];

        for (int row = 0; row < kMaxSparseRowsPerTile; ++row) {
          row_ptrs[row] = lut_rows_base + row * params.num_k_blocks;
          subgroup_started_rows[row] = false;
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
        for (int row = 0; row < kMaxSparseRowsPerTile; ++row) {
          prefetch_pos[row] = row_pos[row];
          prefetch_cur_block[row] = row_cur_block[row];
        }

        for (int stage = 0; stage < Stages; ++stage) {
          int sparse_prefetch_block = pop_sparse_block(prefetch_pos, prefetch_cur_block);
          if (sparse_prefetch_block >= total_blk) break;
          prefetch_sparse_k_block(sparse_prefetch_block);
        }

        int next_block = find_sparse_block(row_pos, row_cur_block);
        while (next_block < total_blk) {
          int selected_row = subgroup_q_row_in_tile;
          bool subgroup_selected = subgroup_q_row_in_tile < sparse_q_rows_in_tile &&
                                   row_pos[selected_row] < row_valid[selected_row] &&
                                   row_cur_block[selected_row] == next_block;
          bool first_selected_block = subgroup_selected ? !subgroup_started_rows[selected_row] : false;
          int sparse_prefetch_block = pop_sparse_block(prefetch_pos, prefetch_cur_block);
          int K = next_block;
          if constexpr (CachedKV) {
            if (K < kblocks_cache) {
              if (K >= blk_k0 && K < blk_k1) {
                mainloop_body(std::bool_constant<true>{}, K, first_selected_block, subgroup_selected,
                              sparse_prefetch_block, copy_k_cache, copy_v_cache, prefetch_v_cache, tKgK_cache,
                              tVgV_cache, pVgV_cache);
                if (subgroup_selected) subgroup_started_rows[selected_row] = true;
              }
            } else {
              K += kblocks_cache;
              if (K >= (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache) && K < blk_k1) {
                mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected,
                              sparse_prefetch_block, copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
                if (subgroup_selected) subgroup_started_rows[selected_row] = true;
              }
            }
          } else if (K >= blk_k0 && K < blk_k1) {
            mainloop_body(std::bool_constant<false>{}, K, first_selected_block, subgroup_selected, sparse_prefetch_block,
                          copy_k, copy_v, prefetch_v, tKgK, tVgV, pVgV);
            if (subgroup_selected) subgroup_started_rows[selected_row] = true;
          }

          advance_sparse_block(next_block, row_pos, row_cur_block);
          next_block = find_sparse_block(row_pos, row_cur_block);
        }
      }
    } else {
      if constexpr (CachedKV) {
        for (int K = blk_k0; K < kblocks_cache; K++) {
          mainloop_body(std::bool_constant<true>{}, K, K == blk_k0, true, total_blk, copy_k_cache, copy_v_cache,
                        prefetch_v_cache, tKgK_cache, tVgV_cache, pVgV_cache);
        }
      }
      for (int K = (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache); K < blk_k1; K++) {
        mainloop_body(std::bool_constant<false>{}, K,
                      K == (blk_k0 > kblocks_cache ? blk_k0 : kblocks_cache), true, total_blk, copy_k, copy_v,
                      prefetch_v, tKgK, tVgV, pVgV);
      }
    }
  }

  CUTLASS_DEVICE
  FragSRow softmax(bool first_block, FragS& tS, FragSRow& tS_max, FragSRow& tS_sum) {
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});
    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), params.scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));
    }

    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++) {
      tS_sum(i) += tS_bsum(i);
    }
    return rescale;
  }
};

}  // namespace cutlass::fmha::collective
