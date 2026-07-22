/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

#include <cmath>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "xe_fmha_fwd_epilogue_compat.hpp"
#include "xe_sparse_sagev1_fwd_mainloop.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include <type_traits>

namespace cutlass::fmha::kernel {

using namespace cute;

template <class T, class = void>
struct has_canonical_nhd_k : std::false_type {};

template <class T>
struct has_canonical_nhd_k<T, std::void_t<decltype(std::declval<T&>().canonical_nhd_k)>> : std::true_type {};

template <class T, class = void>
struct has_sparse_q_block_size : std::false_type {};

template <class T>
struct has_sparse_q_block_size<T, std::void_t<decltype(std::declval<T&>().sparse_q_block_size)>> : std::true_type {};

template <bool IsVarLen_ = false>
struct SparseSageProblemShape {
  using SeqLenType = cute::conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv, seq_len_kv_cache;
  int head_size_qk, head_size_vo;
};

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeSparseSageFwdKernel {
 public:
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len = cutlass::fmha::collective::is_variable_length_v<typename ProblemShape::SeqLenType>;
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;
  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using SGPerWG = typename CollectiveMainloop::SGPerWG;
  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K;
    StrideK dK;
    const ElementV* V;
    StrideV dV;
    ElementO* O;
    StrideO dO;
    const ElementK* K_cache;
    StrideK dK_cache{};
    const ElementV* V_cache;
    StrideV dV_cache{};
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {args.kernel, CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }

  CUTLASS_DEVICE
  Shape<int, int, int> get_sequence_length_shape(ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength, VariableLength, VariableLength>{problem_shape.seq_len_qo, problem_shape.seq_len_kv,
                                                                problem_shape.seq_len_kv_cache},
          batch);
    } else {
      return Shape<int, int, int>{problem_shape.seq_len_qo, problem_shape.seq_len_kv, problem_shape.seq_len_kv_cache};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);
    constexpr int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] = tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = head_q / head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv, seq_len_kv_cache] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo) continue;

      auto offset = cute::min(seq_len_qo, seq_len_kv);
      auto discard_seq_coord = seq_len_qo - offset;
      auto full_tile_offset = seq_len_kv - offset;
      int seq_coord = cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      if (CollectiveMainloop::CausalMask && seq_coord < discard_seq_coord) continue;
      const int seq_len_new = CollectiveMainloop::CausalMask
                                  ? full_tile_offset + cute::min(seq_len_kv, seq_coord - discard_seq_coord) + q_sg_tile
                                  : seq_len_kv;
      const int seq_len = seq_len_new + seq_len_kv_cache;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_k_cache = 0, offset_v_cache = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q = s.num_heads_q * s.head_size_qk * qo_cumulative[idx_b];
        offset_k = s.num_heads_kv * s.head_size_qk * kv_cumulative[idx_b];
        offset_v = s.num_heads_kv * s.head_size_vo * kv_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
        if (s.seq_len_kv_cache.cumulative_length) {
          auto kv_cumulative_cache = s.seq_len_kv_cache.cumulative_length;
          offset_k_cache = s.num_heads_kv * s.head_size_qk * kv_cumulative_cache[idx_b];
          offset_v_cache = s.num_heads_kv * s.head_size_vo * kv_cumulative_cache[idx_b];
        }
      }

      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q = make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K = make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O = make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim);
      auto shape_K_cache = make_shape(seq_len_kv_cache, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V_cache = make_shape(s.head_size_vo, seq_len_kv_cache, s.num_heads_kv, batch_dim);

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K + offset_k);
      auto dcV = const_cast<ElementV*>(p.V + offset_v);
      auto dcK_cache = const_cast<ElementK*>(p.K_cache + offset_k_cache);
      auto dcV_cache = const_cast<ElementV*>(p.V_cache + offset_v_cache);
      int seq_q_pad = (seq_len_qo + params.mainloop.scale_block_size - 1) / params.mainloop.scale_block_size;
      int seq_kv_total = seq_len_kv + seq_len_kv_cache;
      int seq_kv_pad = (seq_kv_total + params.mainloop.scale_block_size - 1) / params.mainloop.scale_block_size;
      auto scaleQ = params.mainloop.scale_block_size
                        ? (float*)params.mainloop.qscale + (idx_b * s.num_heads_q * seq_q_pad + head_q * seq_q_pad)
                        : nullptr;
      auto scaleK = params.mainloop.scale_block_size
                        ? (float*)params.mainloop.kscale + (idx_b * s.num_heads_kv * seq_kv_pad + head * seq_kv_pad)
                        : nullptr;
      auto scaleV = params.mainloop.scale_block_size && params.mainloop.vscale
                        ? (float*)params.mainloop.vscale +
                              ((idx_b * s.num_heads_kv * seq_kv_pad + head * seq_kv_pad) * s.head_size_vo)
                        : nullptr;
      int sparse_q_block = blk_q;
      int sparse_q_rows_in_tile = 1;
      // Dense FMHA treats blk_q as "the" Q tile handled by this workgroup.
      // Sparse FMHA may further subdivide or coarsen that tile into routing rows.
      // sparse_q_block_size tells us how many Q tokens share one LUT row.
      int sparse_q_block_size = params.mainloop.scale_block_size;
      if constexpr (has_sparse_q_block_size<MainloopParams>::value) {
        if (params.mainloop.sparse_q_block_size > 0) {
          sparse_q_block_size = params.mainloop.sparse_q_block_size;
        }
      }
      if (sparse_q_block_size > 0) {
        // q_blocks_per_tile is the sparse-only notion that differs from the dense
        // path: one dense workgroup tile may correspond to multiple sparse routing
        // rows. For q_tile=256 + sparse_q_block_size=64 we get 4 sparse rows;
        // for q_tile=256 + sparse_q_block_size=256 we get 1 sparse row.
        int q_blocks_per_tile = cute::max(1, int(get<0>(TileShapeQK{})) / sparse_q_block_size);
        sparse_q_block = blk_q * q_blocks_per_tile;
        sparse_q_rows_in_tile = q_blocks_per_tile;
        if (params.mainloop.num_q_blocks > 0) {
          // Tail tiles can expose fewer sparse rows than the nominal tile shape.
          // Clamp both the starting row and the row count so the sparse mainloop
          // never reads past the logical routing table.
          sparse_q_rows_in_tile = cute::min(sparse_q_rows_in_tile, params.mainloop.num_q_blocks - sparse_q_block);
          sparse_q_block = cute::min(sparse_q_block, params.mainloop.num_q_blocks - 1);
        }
      }
      // Each workgroup gets a contiguous slice of valid_block_num / lut rows for
      // the sparse Q rows covered by this tile. Dense FMHA has no equivalent
      // per-tile routing metadata.
      auto valid_blocks_base = params.mainloop.valid_block_num
                                   ? params.mainloop.valid_block_num +
                                         (idx_b * s.num_heads_q + head_q) * params.mainloop.num_q_blocks +
                                         sparse_q_block
                                   : nullptr;
      auto lut_rows_base = params.mainloop.lut
                         ? params.mainloop.lut +
                               (((idx_b * s.num_heads_q + head_q) * params.mainloop.num_q_blocks + sparse_q_block) *
                                params.mainloop.num_k_blocks)
                         : nullptr;
      auto ptrO = p.O + offset_o;

      auto stride_q = is_var_len ? cutlass::make_cute_packed_stride(StrideQ{}, shape_Q) : p.dQ;
      auto stride_k = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K) : p.dK;
      auto stride_v = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V) : p.dV;
      auto stride_o = is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_O) : p.dO;
      auto stride_k_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache) : p.dK_cache;
      auto stride_v_cache = is_var_len ? cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache) : p.dV_cache;

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, stride_q));
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, stride_k));
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, stride_v));
      Tensor K_cache = make_tensor(make_gmem_ptr(dcK_cache), make_layout(shape_K_cache, stride_k_cache));
      Tensor V_cache = make_tensor(make_gmem_ptr(dcV_cache), make_layout(shape_V_cache, stride_v_cache));
      Tensor O = make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      FragA tArA;
      FragARow tA_max, tA_sum;

      int l_coord = is_var_len ? 0 : idx_b;
      auto mainloop_params = params.mainloop;
      if constexpr (has_canonical_nhd_k<MainloopParams>::value) {
        mainloop_params.canonical_nhd_k =
            !is_var_len &&
            int(get<1>(stride_k)) == 1 &&
            int(get<2>(stride_k)) == s.head_size_qk &&
            int(get<0>(stride_k)) == s.num_heads_kv * s.head_size_qk;
      }
      CollectiveMainloop mainloop(mainloop_params, shared_storage.mainloop);
      // sparse_q_rows_in_tile is the key extra argument versus dense mainloop:
      // it tells the sparse kernel whether this workgroup should walk a single
      // LUT row linearly or merge multiple LUT rows inside one Q tile.
      mainloop(Q(_, _, head_q, l_coord), K(_, _, head, l_coord), V(_, _, head, l_coord), tArA, tA_max, tA_sum, blk_qv,
               0, k_blocks, k_blocks, thr_id, seq_len, seq_len_kv_cache, idx_b, scaleQ, scaleK, scaleV,
               full_tile_offset, discard_seq_coord, lut_rows_base, valid_blocks_base, sparse_q_rows_in_tile,
               K_cache(_, _, head, l_coord),
               V_cache(_, _, head, l_coord));

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      detail::run_fmha_fwd_epilogue(epilogue, O(_, _, head_q, l_coord), tArA, tA_max, tA_sum, blk_qv, thr_id, head_q,
                                    idx_b, 0);
    }
  }
};

}  // namespace cutlass::fmha::kernel
