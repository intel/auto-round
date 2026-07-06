// SYCL-TLA Flash Attention Wrapper (Prefill + Decode)
// Provides both prefill and decode entry points while accounting for convert_type ODR concerns in sycl-tla
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <cute/util/compat/memory.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sycl/aliases.hpp>
#include "bestla/bestla.h"
#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_SYCL_TLA)
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "stla/xe_sdpa_fwd_mainloop.hpp"
#include "stla/xe_sagev1_fwd_mainloop.hpp"
#include "stla/xe_sage_fwd_kernel.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "helper.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "sycl_common.hpp"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#endif  // ARK_SYCL_TLA

namespace ark {

/// Flash Attention data type codes (matches Python side)
enum class FlashAttnDtype : int {
  FP16 = 0,
  BF16 = 1,
  FP32 = 2,
  FP8_E4M3 = 3,
  FP8_E5M2 = 4,
};

#if defined(ARK_SYCL_TLA)

namespace detail {

using namespace cute;

// Command line options parsing
struct Options {
   const void *q = nullptr, *k = nullptr, *v = nullptr;
   void* mask = nullptr;
   void* o = nullptr;
  int scale_block_size = 0;
  const void *qscale = nullptr, *kscale = nullptr, *vscale = nullptr;
  const void *block_K = nullptr, *block_V = nullptr;
  const int *page_table = nullptr, *num_pages_per_seq = nullptr;
  bool is_causal = false;
  bool varlen = false;
  bool use_paged_kv = false;
  bool use_tensor_strides = false;
  int batch = 0, num_heads_q = 0, num_heads_kv = 0, seq_len_qo = 0, seq_len_kv = 0, seq_len_kv_cache = 0, page_size = 0,
      head_size_qk = 0, head_size_vo = 0;
  int total_seqlen_q = 0, total_seqlen_kv = 0, total_seqlen_kv_cache = 0;
  int max_seqlen_q = 0, max_seqlen_kv = 0, max_seqlen_kv_cache = 0;
  const int* cu_seqlens_q = nullptr;
  const int* cu_seqlens_k = nullptr;
  const int* cu_seqlens_kv_cache = nullptr;
  int q_stride_s = 0, q_stride_d = 1, q_stride_h = 0, q_stride_b = 0;
  int k_stride_s = 0, k_stride_d = 1, k_stride_h = 0, k_stride_b = 0;
  int v_stride_d = 1, v_stride_s = 0, v_stride_h = 0, v_stride_b = 0;
  int o_stride_s = 0, o_stride_d = 1, o_stride_h = 0, o_stride_b = 0;
  float softmax_scale = 0.0f;
  float* lse = nullptr;  // LSE output buffer (null = skip)
  bool persistent = false;

  void print(std::ostream& os = std::cout) const {
    os << std::boolalpha << "Options {\n"
       << "  q: " << q << "\n"
       << "  k: " << k << "\n"
       << "  v: " << v << "\n"
       << "  o: " << o << "\n"
       << "  mask: " << mask << "\n"
       << "  block_K: " << block_K << "\n"
       << "  block_V: " << block_V << "\n"
       << "  page_table: " << page_table << "\n"
       << "  num_pages_per_seq: " << num_pages_per_seq << "\n"
       << "  is_causal: " << is_causal << "\n"
       << "  varlen: " << varlen << "\n"
       << "  use_paged_kv: " << use_paged_kv << "\n"
      << "  use_tensor_strides: " << use_tensor_strides << "\n"
       << "  batch: " << batch << "\n"
       << "  num_heads_q: " << num_heads_q << "\n"
       << "  num_heads_kv: " << num_heads_kv << "\n"
       << "  seq_len_qo: " << seq_len_qo << "\n"
       << "  seq_len_kv: " << seq_len_kv << "\n"
       << "  seq_len_kv_cache: " << seq_len_kv_cache << "\n"
       << "  page_size: " << page_size << "\n"
       << "  head_size_qk: " << head_size_qk << "\n"
       << "  head_size_vo: " << head_size_vo << "\n"
       << "  total_seqlen_q: " << total_seqlen_q << "\n"
       << "  total_seqlen_kv: " << total_seqlen_kv << "\n"
       << "  total_seqlen_kv_cache: " << total_seqlen_kv_cache << "\n"
       << "  max_seqlen_q: " << max_seqlen_q << "\n"
       << "  max_seqlen_kv: " << max_seqlen_kv << "\n"
       << "  max_seqlen_kv_cache: " << max_seqlen_kv_cache << "\n"
      << "  q_stride: (" << q_stride_s << ", " << q_stride_d << ", " << q_stride_h << ", " << q_stride_b << ")\n"
      << "  k_stride: (" << k_stride_s << ", " << k_stride_d << ", " << k_stride_h << ", " << k_stride_b << ")\n"
      << "  v_stride: (" << v_stride_d << ", " << v_stride_s << ", " << v_stride_h << ", " << v_stride_b << ")\n"
      << "  o_stride: (" << o_stride_s << ", " << o_stride_d << ", " << o_stride_h << ", " << o_stride_b << ")\n"
       << "  softmax_scale: " << softmax_scale << "\n"
       << "  persistent: " << persistent << "\n"
       << "}\n";
  }
};

    template <typename StrideQ, typename StrideK, typename StrideV, typename StrideO>
    inline void set_tensor_strides_from_options(const Options& options, StrideQ& stride_Q, StrideK& stride_K,
                  StrideV& stride_V, StrideO& stride_O) {
      stride_Q = cute::make_stride(options.q_stride_s, cute::_1{}, options.q_stride_h, options.q_stride_b);
      stride_K = cute::make_stride(options.k_stride_s, cute::_1{}, options.k_stride_h, options.k_stride_b);
      stride_V = cute::make_stride(cute::_1{}, options.v_stride_s, options.v_stride_h, options.v_stride_b);
      stride_O = cute::make_stride(options.o_stride_s, cute::_1{}, options.o_stride_h, options.o_stride_b);
    }

// 3 input matrices: (K)eys, (Q)ueries and (V)alues.
using LayoutQ = cutlass::layout::RowMajor;
using LayoutK = cutlass::layout::ColumnMajor;
using LayoutV = cutlass::layout::RowMajor;
using LayoutO = cutlass::layout::RowMajor;

template <class FMHAKernel, bool isVarLen = false>
struct KernelRunner {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideK stride_K_cache;
  StrideV stride_V_cache;
  StrideO stride_O;
  uint64_t seed = 0;
  /// Scratch buffer for zero-filled kv_cache cumulative_length.
  cutlass::device_memory::allocation<int> zero_cu_cache_;

  //
  // Methods
  //

  // Note that the GemmUniversalAdapter currently doesn't support flash attention, which is why this
  // secondary `run` function is required to launch the kernel.
  static void run(typename FMHAKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{syclex::sub_group_size<cute::intel::sg_size>,
                                                         intelex::grf_size<256>};
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);

    EventManager::getInstance().addEvent(event);
  }

  template <class ProblemShape>
  auto initialize_varlen(const ProblemShape& problem_size, const Options& options) {
    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;
    get<3>(problem_size_for_init) = options.total_seqlen_q;
    get<4>(problem_size_for_init) = options.total_seqlen_kv;
    get<5>(problem_size_for_init) = options.total_seqlen_kv_cache;

    ProblemShapeType problem_size_for_launch;
    problem_size_for_launch.batch = get<0>(problem_size);
    problem_size_for_launch.num_heads_q = get<1>(problem_size);
    problem_size_for_launch.num_heads_kv = get<2>(problem_size);
    problem_size_for_launch.seq_len_qo = cutlass::fmha::collective::VariableLength{options.max_seqlen_q};
    problem_size_for_launch.seq_len_kv = cutlass::fmha::collective::VariableLength{options.max_seqlen_kv};
    problem_size_for_launch.seq_len_kv_cache = cutlass::fmha::collective::VariableLength{options.max_seqlen_kv_cache};
    problem_size_for_launch.seq_len_qo.cumulative_length = const_cast<int*>(options.cu_seqlens_q);
    problem_size_for_launch.seq_len_kv.cumulative_length = const_cast<int*>(options.cu_seqlens_k);
    if (options.cu_seqlens_kv_cache) {
      problem_size_for_launch.seq_len_kv_cache.cumulative_length = const_cast<int*>(options.cu_seqlens_kv_cache);
    } else {
      zero_cu_cache_.reset(options.batch + 1);
      std::fill_n(zero_cu_cache_.get(), options.batch + 1, 0);
      problem_size_for_launch.seq_len_kv_cache.cumulative_length = zero_cu_cache_.get();
    }
    problem_size_for_launch.head_size_qk = get<6>(problem_size);
    problem_size_for_launch.head_size_vo = get<7>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  ProblemShapeType initialize(const Options& options) {
    auto problem_shape_in =
        cute::make_tuple(options.batch, options.num_heads_q, options.num_heads_kv, options.seq_len_qo,
                         options.seq_len_kv, options.seq_len_kv_cache, options.head_size_qk, options.head_size_vo);
    ProblemShapeType shape;

    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(problem_shape_in, options);
      problem_size = problem_shape_init;
      shape = problem_shape_launch;
    } else {
      problem_size = problem_shape_in;
      shape.batch = options.batch;
      shape.num_heads_q = options.num_heads_q;
      shape.num_heads_kv = options.num_heads_kv;
      shape.seq_len_qo = options.seq_len_qo;
      shape.seq_len_kv = options.seq_len_kv;
      shape.seq_len_kv_cache = options.seq_len_kv_cache;
      shape.head_size_qk = options.head_size_qk;
      shape.head_size_vo = options.head_size_vo;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] =
        problem_size;
    auto shape_Q = cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch);
    auto shape_K = cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch);
    auto shape_V = cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch);
    auto shape_K_cache = cute::make_shape(seq_len_kv_cache, head_size_qk, num_heads_kv, batch);
    auto shape_V_cache = cute::make_shape(head_size_vo, seq_len_kv_cache, num_heads_kv, batch);
    auto shape_O = cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch);

    if (options.use_tensor_strides) {
      set_tensor_strides_from_options(options, stride_Q, stride_K, stride_V, stride_O);
    } else {
      stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
      stride_K = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
      stride_V = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
      stride_O = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
    }
    stride_K_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache);
    stride_V_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache);

    if constexpr (isVarLen) {
      shape.seq_len_qo.cumulative_length = const_cast<int*>(options.cu_seqlens_q);
      shape.seq_len_kv.cumulative_length = const_cast<int*>(options.cu_seqlens_k);
      if (options.cu_seqlens_kv_cache) {
        shape.seq_len_kv_cache.cumulative_length = const_cast<int*>(options.cu_seqlens_kv_cache);
      } else {
        zero_cu_cache_.reset(options.batch + 1);
        std::fill_n(zero_cu_cache_.get(), options.batch + 1, 0);
        shape.seq_len_kv_cache.cumulative_length = zero_cu_cache_.get();
      }
    }
    return shape;
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(options);

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            static_cast<const FMHAKernel::ElementQ*>(options.q),
            stride_Q,
            static_cast<const FMHAKernel::ElementK*>(options.k),
            stride_K,
            static_cast<const FMHAKernel::ElementV*>(options.v),
            stride_V,
            static_cast<FMHAKernel::ElementO*>(options.o),
            stride_O,
            static_cast<const FMHAKernel::ElementK*>(options.block_K),
            stride_K_cache,
            static_cast<const FMHAKernel::ElementV*>(options.block_V),
            stride_V_cache,
            static_cast<float*>(options.lse),
        },
        {options.softmax_scale, static_cast<FMHAKernel::ElementQ*>(options.mask),
         options.use_paged_kv ? options.page_table : nullptr, options.use_paged_kv ? options.page_size : 0,
         options.use_paged_kv ? options.num_pages_per_seq : nullptr},
        {},
        hw_info};
    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << options.batch << 'x' << options.num_heads_q << 'x' << options.seq_len_qo
                << 'x' << options.seq_len_kv << 'x' << options.head_size_qk << 'x' << options.head_size_vo
                << (options.is_causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    run(params);

    return cutlass::Status::kSuccess;
  }
};

template <bool Causal, typename TileShapeQK, typename TileShapePV, typename TileShapeOutput, typename SubgroupLayoutQK,
          typename SubgroupLayoutPV_, /* void -> default */
          int PipelineStages, bool persistent, typename ElementQ = bfloat16_t, typename ElementK = bfloat16_t,
          typename ElementV = bfloat16_t, typename ElementO = ElementQ,
          typename MMAOperation_ = void, /* void -> default */
          typename StrideQ = Stride<int, _1, int, int>, typename StrideK = Stride<int, _1, int, int>,
          typename StrideV = Stride<_1, int, int, int>, typename StrideO = Stride<int, _1, int, int>,
          typename GmemTiledCopyQ = void, /* void -> default block 2D */
          typename GmemTiledCopyK = void, typename GmemTiledCopyV = void, typename GmemTiledCopyO = void>
struct FMHAConfig {
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      typename cute::conditional_t<
          cute::is_same_v<ElementQ, cutlass::float_e5m2_t> || cute::is_same_v<ElementQ, cutlass::float_e4m3_t>,
          XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, half_t>, XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>>,
      MMAOperation_>;
  using SubgroupLayoutPV =
      cute::conditional_t<is_void_v<SubgroupLayoutPV_>,
                          decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})), SubgroupLayoutPV_>;

  template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
  static int run(const Options& options) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
                  "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorK_cache = TensorK;
    using TensorV_cache = TensorV;
    using GmemTiledCopyK_cache = GmemTiledCopyK;
    using GmemTiledCopyV_cache = GmemTiledCopyV;
    using MainloopDispatchPolicy = cutlass::sdpa::XeDefault<PipelineStages>;
    if constexpr (Causal) {
      // Mainloop
      using CollectiveMainloop =
          cutlass::fmha::collective::SDPAFwdMainloop<MainloopDispatchPolicy, Causal, false, CachedKV, PagedKV,
                                                     TiledMMAQK, TiledMMAPV, VTiles, TensorQ, TensorK, TensorV,
                                                     TensorK_cache, TensorV_cache, GmemTiledCopyQ, GmemTiledCopyK,
                                                     GmemTiledCopyV, GmemTiledCopyK_cache, GmemTiledCopyV_cache>;

      // Epilogue
      using CollectiveEpilogue =
          cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

      static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
      using FMHAKernel = conditional_t<
          is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
          cutlass::fmha::kernel::XeFMHAFwdDynamicSplitKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue,
                                                             Scheduler>,
          cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

      KernelRunner<FMHAKernel, isVarLen> runner;

      CUTLASS_CHECK(runner.run(options, hw_info));
    } else {
      if (options.mask) {
        // Mainloop
        using CollectiveMainloop =
            cutlass::fmha::collective::SDPAFwdMainloop<MainloopDispatchPolicy, Causal, true, CachedKV, PagedKV,
                                                       TiledMMAQK, TiledMMAPV, VTiles, TensorQ, TensorK, TensorV,
                                                       TensorK_cache, TensorV_cache, GmemTiledCopyQ, GmemTiledCopyK,
                                                       GmemTiledCopyV, GmemTiledCopyK_cache, GmemTiledCopyV_cache>;

        // Epilogue
        using CollectiveEpilogue =
            cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

        static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
        using FMHAKernel =
            conditional_t<is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
                          cutlass::fmha::kernel::XeFMHAFwdDynamicSplitKernel<ProblemShapeType, CollectiveMainloop,
                                                                             CollectiveEpilogue, Scheduler>,
                          cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShapeType, CollectiveMainloop,
                                                                 CollectiveEpilogue, Scheduler>>;

        KernelRunner<FMHAKernel, isVarLen> runner;

        CUTLASS_CHECK(runner.run(options, hw_info));
      } else {
        // Mainloop
        using CollectiveMainloop =
            cutlass::fmha::collective::SDPAFwdMainloop<MainloopDispatchPolicy, Causal, false, CachedKV, PagedKV,
                                                       TiledMMAQK, TiledMMAPV, VTiles, TensorQ, TensorK, TensorV,
                                                       TensorK_cache, TensorV_cache, GmemTiledCopyQ, GmemTiledCopyK,
                                                       GmemTiledCopyV, GmemTiledCopyK_cache, GmemTiledCopyV_cache>;

        // Epilogue
        using CollectiveEpilogue =
            cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

        static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
        using FMHAKernel =
            conditional_t<is_same_v<Scheduler, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>,
                          cutlass::fmha::kernel::XeFMHAFwdDynamicSplitKernel<ProblemShapeType, CollectiveMainloop,
                                                                             CollectiveEpilogue, Scheduler>,
                          cutlass::fmha::kernel::XeFMHAFwdKernel<ProblemShapeType, CollectiveMainloop,
                                                                 CollectiveEpilogue, Scheduler>>;

        KernelRunner<FMHAKernel, isVarLen> runner;

        CUTLASS_CHECK(runner.run(options, hw_info));
      }
    }

    return 0;
  }

  static int run(const Options& options) {
    bool cached_kv = options.seq_len_kv_cache > 0;
    if constexpr (persistent) {
      if (options.use_paged_kv || options.seq_len_kv_cache > 0) {
        std::cerr
            << "Error: Persistent kernel does not support paged/cached KV cache (use_paged_kv or seq_len_kv_cache > 0)."
            << std::endl;
        return -1;
      }
      return run<false, false, false, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>(options);
    } else if (options.use_paged_kv && !options.varlen) {
      throw std::runtime_error("Paged KV without varlen is not supported yet");
      // return run<false, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && options.varlen && !cached_kv) {
      return run<true, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && !options.varlen && !cached_kv) {
      return run<false, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && options.varlen && cached_kv) {
      throw std::runtime_error("Varlen with cached KV but without paged KV is not supported yet");
      // return run<true, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && !options.varlen && cached_kv) {
      throw std::runtime_error("Cached KV without varlen is not supported yet");
      // return run<false, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else {
      throw std::runtime_error("The combination of options is not supported yet");
      // return run<true, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    }
  }
};

template <class FMHAKernel, bool isVarLen = false>
struct SageKernelRunner {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::SageProblemShape<isVarLen>;

  //
  // Data members
  //

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideK stride_K_cache;
  StrideV stride_V_cache;
  StrideO stride_O;
  uint64_t seed = 0;
  /// Scratch buffer for zero-filled kv_cache cumulative_length.
  cutlass::device_memory::allocation<int> zero_cu_cache_;

  //
  // Methods
  //

  // Note that the GemmUniversalAdapter currently doesn't support flash attention, which is why this
  // secondary `run` function is required to launch the kernel.
  static void run(typename FMHAKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{syclex::sub_group_size<cute::intel::sg_size>,
                                                         intelex::grf_size<256>};
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, params);

    EventManager::getInstance().addEvent(event);
  }

  template <class ProblemShape>
  auto initialize_varlen(const ProblemShape& problem_size, const Options& options) {
    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = 1;
    get<3>(problem_size_for_init) = options.total_seqlen_q;
    get<4>(problem_size_for_init) = options.total_seqlen_kv;
    get<5>(problem_size_for_init) = options.total_seqlen_kv_cache;

    ProblemShapeType problem_size_for_launch;
    problem_size_for_launch.batch = get<0>(problem_size);
    problem_size_for_launch.num_heads_q = get<1>(problem_size);
    problem_size_for_launch.num_heads_kv = get<2>(problem_size);
    problem_size_for_launch.seq_len_qo = cutlass::fmha::collective::VariableLength{options.max_seqlen_q};
    problem_size_for_launch.seq_len_kv = cutlass::fmha::collective::VariableLength{options.max_seqlen_kv};
    problem_size_for_launch.seq_len_kv_cache = cutlass::fmha::collective::VariableLength{options.max_seqlen_kv_cache};
    problem_size_for_launch.seq_len_qo.cumulative_length = const_cast<int*>(options.cu_seqlens_q);
    problem_size_for_launch.seq_len_kv.cumulative_length = const_cast<int*>(options.cu_seqlens_k);
    if (options.cu_seqlens_kv_cache) {
      problem_size_for_launch.seq_len_kv_cache.cumulative_length = const_cast<int*>(options.cu_seqlens_kv_cache);
    } else {
      zero_cu_cache_.reset(options.batch + 1);
      std::fill_n(zero_cu_cache_.get(), options.batch + 1, 0);
      problem_size_for_launch.seq_len_kv_cache.cumulative_length = zero_cu_cache_.get();
    }
    problem_size_for_launch.head_size_qk = get<6>(problem_size);
    problem_size_for_launch.head_size_vo = get<7>(problem_size);

    return cute::make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  ProblemShapeType initialize(const Options& options) {
    auto problem_shape_in =
        cute::make_tuple(options.batch, options.num_heads_q, options.num_heads_kv, options.seq_len_qo,
                         options.seq_len_kv, options.seq_len_kv_cache, options.head_size_qk, options.head_size_vo);
    ProblemShapeType shape;

    decltype(problem_shape_in) problem_size;

    if constexpr (isVarLen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(problem_shape_in, options);
      problem_size = problem_shape_init;
      shape = problem_shape_launch;
    } else {
      problem_size = problem_shape_in;
      shape.batch = options.batch;
      shape.num_heads_q = options.num_heads_q;
      shape.num_heads_kv = options.num_heads_kv;
      shape.seq_len_qo = options.seq_len_qo;
      shape.seq_len_kv = options.seq_len_kv;
      shape.seq_len_kv_cache = options.seq_len_kv_cache;
      shape.head_size_qk = options.head_size_qk;
      shape.head_size_vo = options.head_size_vo;
    }

    auto [batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, seq_len_kv_cache, head_size_qk, head_size_vo] =
        problem_size;
    auto shape_Q = cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch);
    auto shape_K = cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch);
    auto shape_V = cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch);
    auto shape_K_cache = cute::make_shape(seq_len_kv_cache, head_size_qk, num_heads_kv, batch);
    auto shape_V_cache = cute::make_shape(head_size_vo, seq_len_kv_cache, num_heads_kv, batch);
    auto shape_O = cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch);

    if (options.use_tensor_strides) {
      set_tensor_strides_from_options(options, stride_Q, stride_K, stride_V, stride_O);
    } else {
      stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, shape_Q);
      stride_K = cutlass::make_cute_packed_stride(StrideK{}, shape_K);
      stride_V = cutlass::make_cute_packed_stride(StrideV{}, shape_V);
      stride_O = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
    }
    stride_K_cache = cutlass::make_cute_packed_stride(StrideK{}, shape_K_cache);
    stride_V_cache = cutlass::make_cute_packed_stride(StrideV{}, shape_V_cache);

    if constexpr (isVarLen) {
      shape.seq_len_qo.cumulative_length = const_cast<int*>(options.cu_seqlens_q);
      shape.seq_len_kv.cumulative_length = const_cast<int*>(options.cu_seqlens_k);
      if (options.cu_seqlens_kv_cache) {
        shape.seq_len_kv_cache.cumulative_length = const_cast<int*>(options.cu_seqlens_kv_cache);
      } else {
        zero_cu_cache_.reset(options.batch + 1);
        std::fill_n(zero_cu_cache_.get(), options.batch + 1, 0);
        shape.seq_len_kv_cache.cumulative_length = zero_cu_cache_.get();
      }
    }
    return shape;
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(options);

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            static_cast<const FMHAKernel::ElementQ*>(options.q),
            stride_Q,
            static_cast<const FMHAKernel::ElementK*>(options.k),
            stride_K,
            static_cast<const FMHAKernel::ElementV*>(options.v),
            stride_V,
            static_cast<FMHAKernel::ElementO*>(options.o),
            stride_O,
            static_cast<const FMHAKernel::ElementK*>(options.block_K),
            stride_K_cache,
            static_cast<const FMHAKernel::ElementV*>(options.block_V),
            stride_V_cache,
            static_cast<float*>(options.lse),
        },
        {options.softmax_scale, static_cast<float*>(options.mask), options.scale_block_size,
         static_cast<const float*>(options.qscale), static_cast<const float*>(options.kscale),
         static_cast<const float*>(options.vscale),
         options.use_paged_kv ? options.page_table : nullptr, options.use_paged_kv ? options.page_size : 0,
         options.use_paged_kv ? options.num_pages_per_seq : nullptr},
        {},
        hw_info};
    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << options.batch << 'x' << options.num_heads_q << 'x' << options.seq_len_qo
                << 'x' << options.seq_len_kv << 'x' << options.head_size_qk << 'x' << options.head_size_vo
                << (options.is_causal ? "xCausal" : "xNonCausal") << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));

    // Convert host-side arguments to device-side arguments to be passed to the kernel
    auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

    run(params);

    return cutlass::Status::kSuccess;
  }
};

template <bool Causal, bool UseInt8PV, bool WriteBackInt8PV, bool ExecuteInt8PV, typename TileShapeQK,
          typename TileShapePV, typename TileShapeOutput, typename SubgroupLayoutQK,
          typename SubgroupLayoutPV_, /* void -> default */
          int PipelineStages, bool persistent, typename ElementQ = bfloat16_t, typename ElementK = bfloat16_t,
          typename ElementV = bfloat16_t, typename ElementO = ElementQ,
          typename MMAOperation_ = void, /* void -> default */
          typename StrideQ = Stride<int, _1, int, int>, typename StrideK = Stride<int, _1, int, int>,
          typename StrideV = Stride<_1, int, int, int>, typename StrideO = Stride<int, _1, int, int>,
          typename GmemTiledCopyQ = void, /* void -> default block 2D */
          typename GmemTiledCopyK = void, typename GmemTiledCopyV = void, typename GmemTiledCopyO = void>
struct SageConfig {
  static constexpr int SGTileQ = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation =
      cute::conditional_t<is_void_v<MMAOperation_>, XE_DPAS_TT<cute::gcd(SGTileQ, 8), int32_t, int8_t>, MMAOperation_>;
  // The PV "float" tiled MMA operates on the output element type. For UseInt8PV the kernel also
  // constructs a separate int8 quantized PV MMA internally, while this float MMA only needs to
  // describe the tile shape used for the accumulator and dequantized path. For the non-int8-PV
  // path, V is consumed directly so we use ElementO (== ElementV) which supports half_t / bfloat16_t.
  using MMAOperationPV = cute::conditional_t<is_void_v<MMAOperation_>,
                                             XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementO>, MMAOperation_>;
  using SubgroupLayoutPV =
      cute::conditional_t<is_void_v<SubgroupLayoutPV_>,
                          decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})), SubgroupLayoutPV_>;

  template <bool isVarLen, bool CachedKV, bool PagedKV, class Scheduler>
  static int run(const Options& options) {
    //
    // Run examples
    //

    // The KernelHardwareInfo struct holds the number of EUs on the GPU with a given device ID. This
    // information is used by the underlying kernel.
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::SageProblemShape<isVarLen>;

    using TiledMMAQK = typename TiledMMAHelper<MMA_Atom<MMAOperation>, Layout<TileShapeQK>, SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV =
        typename TiledMMAHelper<MMA_Atom<MMAOperationPV>, Layout<TileShapePV>, SubgroupLayoutPV>::TiledMMA;

    static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
                  "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(make_gmem_ptr(&val), make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorK_cache = TensorK;
    using TensorV_cache = TensorV;
    using GmemTiledCopyK_cache = GmemTiledCopyK;
    using GmemTiledCopyV_cache = GmemTiledCopyV;

    // Mainloop
    using MainloopDispatchPolicy = cutlass::sage::XeDefault<PipelineStages>;
    if constexpr (Causal) {
      using CollectiveMainloop =
          cutlass::fmha::collective::SAGEV1FwdMainloop<MainloopDispatchPolicy, Causal, false, CachedKV, PagedKV,
                                 UseInt8PV, WriteBackInt8PV, ExecuteInt8PV, TiledMMAQK,
                                 TiledMMAPV, VTiles, TensorQ, TensorK, TensorV,
                                                       TensorK_cache, TensorV_cache, GmemTiledCopyQ, GmemTiledCopyK,
                                                       GmemTiledCopyV, GmemTiledCopyK_cache, GmemTiledCopyV_cache>;

      // Epilogue
      using CollectiveEpilogue =
          cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

      static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
      using FMHAKernel =
          cutlass::fmha::kernel::XeSageFwdKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

      SageKernelRunner<FMHAKernel, isVarLen> runner;

      CUTLASS_CHECK(runner.run(options, hw_info));
    } else {
      if (options.mask) {
        using CollectiveMainloop =
            cutlass::fmha::collective::SAGEV1FwdMainloop<MainloopDispatchPolicy, Causal, true, CachedKV, PagedKV,
                                 UseInt8PV, WriteBackInt8PV, ExecuteInt8PV, TiledMMAQK,
                                 TiledMMAPV, VTiles, TensorQ, TensorK, TensorV,
                                                         TensorK_cache, TensorV_cache, GmemTiledCopyQ, GmemTiledCopyK,
                                                         GmemTiledCopyV, GmemTiledCopyK_cache, GmemTiledCopyV_cache>;

        // Epilogue
        using CollectiveEpilogue =
            cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

        static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
        using FMHAKernel =
            cutlass::fmha::kernel::XeSageFwdKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

        SageKernelRunner<FMHAKernel, isVarLen> runner;

        CUTLASS_CHECK(runner.run(options, hw_info));
      } else {
        using CollectiveMainloop =
            cutlass::fmha::collective::SAGEV1FwdMainloop<MainloopDispatchPolicy, Causal, false, CachedKV, PagedKV,
                                 UseInt8PV, WriteBackInt8PV, ExecuteInt8PV, TiledMMAQK,
                                 TiledMMAPV, VTiles, TensorQ, TensorK, TensorV,
                                                         TensorK_cache, TensorV_cache, GmemTiledCopyQ, GmemTiledCopyK,
                                                         GmemTiledCopyV, GmemTiledCopyK_cache, GmemTiledCopyV_cache>;

        // Epilogue
        using CollectiveEpilogue =
            cutlass::fmha::collective::FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutput, TensorO, GmemTiledCopyO>;

        static_assert(!(persistent & Causal), "persistent SDPA kernel not support Causal yet");
        using FMHAKernel =
            cutlass::fmha::kernel::XeSageFwdKernel<ProblemShapeType, CollectiveMainloop, CollectiveEpilogue, Scheduler>;

        SageKernelRunner<FMHAKernel, isVarLen> runner;

        CUTLASS_CHECK(runner.run(options, hw_info));
      }
    }
    return 0;
  }

  static int run(const Options& options) {
    bool cached_kv = options.seq_len_kv_cache > 0;
    if constexpr (persistent) {
      if (options.use_paged_kv || options.seq_len_kv_cache > 0) {
        std::cerr
            << "Error: Persistent kernel does not support paged/cached KV cache (use_paged_kv or seq_len_kv_cache > 0)."
            << std::endl;
        return -1;
      }
      return run<false, false, false, cutlass::fmha::kernel::XeFHMAIndividualPersistentTileScheduler>(options);
    } else if (options.use_paged_kv && !options.varlen) {
      throw std::runtime_error("Paged KV without varlen is not supported yet");
      // return run<false, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && options.varlen && !cached_kv) {
      return run<true, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && !options.varlen && !cached_kv) {
      return run<false, false, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && options.varlen && cached_kv) {
      throw std::runtime_error("Varlen with cached KV but without paged KV is not supported yet");
      // return run<true, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else if (!options.use_paged_kv && !options.varlen && cached_kv) {
      throw std::runtime_error("Cached KV without varlen is not supported yet");
      // return run<false, true, false, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    } else {
      throw std::runtime_error("The combination of options is not supported yet");
      // return run<true, true, true, cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(options);
    }
  }
};

// ========================================================================
// Prefill Kernel Launch
// ========================================================================

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_prefill_kernel_128(Options const& options) {
  constexpr int PipelineStages = 2;
  constexpr int PipelineStages1 = 2;
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
  using ShapeQK1 = Shape<_256, _32, _32>;
  using ShapePV1 = Shape<_256, _32, _32>;
  using ShapeOut1 = Shape<_256, _128>;
  using SubgroupLayoutQK1 = Layout<Shape<_16, _1, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK1, ShapePV1, ShapeOut1, SubgroupLayoutQK1, void, PipelineStages1,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, typename ElementO = ElementV, bool UseInt8PV = false,
          bool WriteBackInt8PV = true, bool ExecuteInt8PV = true>
inline int launch_sage_prefill_kernel_128(Options const& options) {
  constexpr int PipelineStages = 2;
  constexpr int PipelineStages1 = 2;
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
  using ShapeQK1 = Shape<_256, _64, _32>;
  using ShapePV1 = Shape<_256, _32, _64>;
  using ShapeOut1 = Shape<_256, _128>;
  using SubgroupLayoutQK1 = Layout<Shape<_16, _1, _1>>;
  return options.is_causal ? SageConfig<true, UseInt8PV, WriteBackInt8PV, ExecuteInt8PV, ShapeQK, ShapePV, ShapeOut,
                                        SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/false, ElementQ, ElementK, ElementV, ElementO>::run(options)
                           : SageConfig<false, UseInt8PV, WriteBackInt8PV, ExecuteInt8PV, ShapeQK1, ShapePV1,
                                        ShapeOut1, SubgroupLayoutQK1, void, PipelineStages1,
                                        /*persistent=*/false, ElementQ, ElementK, ElementV, ElementO>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, typename ElementO = ElementV, bool UseInt8PV = false,
          bool WriteBackInt8PV = true, bool ExecuteInt8PV = true>
inline int launch_sage_prefill_kernel_64(Options const& options) {
  constexpr int PipelineStages = 2;
  constexpr int PipelineStages1 = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
  using SubgroupLayoutPV = void;
  return options.is_causal
             ? SageConfig<true, UseInt8PV, WriteBackInt8PV, ExecuteInt8PV, ShapeQK, ShapePV, ShapeOut,
                          SubgroupLayoutQK, SubgroupLayoutPV, PipelineStages,
                          /*persistent=*/false, ElementQ, ElementK, ElementV, ElementO>::run(options)
             : SageConfig<false, UseInt8PV, WriteBackInt8PV, ExecuteInt8PV, ShapeQK, ShapePV, ShapeOut,
                          SubgroupLayoutQK, SubgroupLayoutPV, PipelineStages1,
                          /*persistent=*/false, ElementQ, ElementK, ElementV, ElementO>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_prefill_kernel_64(Options const& options) {
  constexpr int PipelineStages = 1;
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _64>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
  using ShapeQK1 = Shape<_128, _64, _32>;
  using ShapePV1 = Shape<_128, _32, _64>;
  using ShapeOut1 = Shape<_128, _64>;
  using SubgroupLayoutQK1 = Layout<Shape<_8, _1, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK1, ShapePV1, ShapeOut1, SubgroupLayoutQK1, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_prefill_kernel_192(Options const& options) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_prefill_kernel_96(Options const& options) {
  constexpr int PipelineStages = 2;
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_decode_kernel_128(Options const& options) {
  constexpr int PipelineStages = 1;
  using NUM_SG = std::conditional_t<persistent, _16, _8>;
  using KV_TILE_SIZE = std::conditional_t<persistent, _256, _512>;
  using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
  using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
  using ShapeOut = Shape<_1, _128>;
  using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_decode_kernel_64(Options const& options) {
  constexpr int PipelineStages = 1;
  using NUM_SG = std::conditional_t<persistent, _16, _8>;
  using KV_TILE_SIZE = std::conditional_t<persistent, _256, _512>;
  using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
  using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
  using ShapeOut = Shape<_1, _64>;
  using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_decode_kernel_96(Options const& options) {
  constexpr int PipelineStages = 1;
  using NUM_SG = std::conditional_t<persistent, _16, _8>;
  using KV_TILE_SIZE = std::conditional_t<persistent, _256, _512>;
  using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
  using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
  using ShapeOut = Shape<_1, _96>;
  using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

template <typename ElementQ, typename ElementK, typename ElementV, bool persistent = false>
inline int launch_decode_kernel_192(Options const& options) {
  constexpr int PipelineStages = 1;
  using NUM_SG = std::conditional_t<persistent, _16, _8>;
  using KV_TILE_SIZE = std::conditional_t<persistent, _256, _512>;
  using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
  using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
  using ShapeOut = Shape<_1, _192>;
  using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options)
                           : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,
                                        /*persistent=*/persistent, ElementQ, ElementK, ElementV>::run(options);
}

}  // namespace detail
#endif  // ARK_SYCL_TLA

}  // namespace ark
