// Flash Attention Decode Benchmarks using sycl-tla
// Separated from sycl_benchmark.cpp to avoid header conflicts with prefill kernels.
//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include <type_traits>
#include "bestla_ut.h"
#include "bestla_utils.h"
#include "sycl_ut.h"
#include "sycl/sycl_utils.h"

#if defined(ARK_SYCL_TLA)

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

// FlashAttention-v2 (Xe) building blocks (from sycl-tla).
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/tile_scheduler.hpp"

// FlashAttention-v2 Decode kernels only
#include "flash_attention_v2/kernel/xe_flash_attn_decode.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_decode_epilogue.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_decode_softmax_epilogue.hpp"

// CUTLASS-SYCL/CuTe compat layer
#include "cute/util/compat.hpp"

// sycl-tla example utilities
#include "helper.h"
#include "sycl_common.hpp"

#include "cute/tensor.hpp"

namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
namespace sycl_ut {

int constexpr TestMs = 1000;

template <typename Kernel, typename = void>
struct BestlaBenchSubgroupSizeDecode {
  static constexpr int value = Kernel::DispatchPolicy::SubgroupSize;
};

template <typename Kernel>
struct BestlaBenchSubgroupSizeDecode<Kernel, std::void_t<decltype(Kernel::SubgroupSize)>> {
  static constexpr int value = Kernel::SubgroupSize;
};

// ========================================================================
// Flash Attention Decode Benchmarks (FP16)
// ========================================================================

class Benchmark_SyclTla_FlashAttnDecode {
 public:
  Benchmark_SyclTla_FlashAttnDecode() {
    UT_START();
    // Baseline
    run_all_fp16(/*batch=*/1, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*head_dim=*/128, /*cache_len=*/4096,
                 /*is_causal=*/true);

    // Heavier-cache cases to expose bandwidth effects (use fewer heads to keep memory reasonable).
    // KV cache elems = cache_len * head_dim * batch * num_heads_kv.
    run_all_fp16(/*batch=*/4, /*num_heads_q=*/16, /*num_heads_kv=*/16, /*head_dim=*/128, /*cache_len=*/8192,
                 /*is_causal=*/true);
    run_all_fp16(/*batch=*/8, /*num_heads_q=*/8, /*num_heads_kv=*/8, /*head_dim=*/128, /*cache_len=*/8192,
                 /*is_causal=*/true);
  }

  template <typename ElementInput>
  static const char* dtype_str() {
    if constexpr (std::is_same_v<ElementInput, cutlass::bfloat16_t>) {
      return "bf16";
    } else if constexpr (std::is_same_v<ElementInput, cutlass::half_t>) {
      return "fp16";
    } else if constexpr (std::is_same_v<ElementInput, cutlass::float_e4m3_t>) {
      return "f8_e4m3";
    } else if constexpr (std::is_same_v<ElementInput, cutlass::float_e5m2_t>) {
      return "f8_e5m2";
    } else {
      return "unknown";
    }
  }

 private:
  template <typename KernelT>
  static sycl::event launch_decode(typename KernelT::Params const& params) {
    compat::dim3 const block = KernelT::get_block_shape();
    compat::dim3 const grid = KernelT::get_grid_shape(params);
    int smem_size = KernelT::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    return launch<cutlass::device_kernel<KernelT>>(
        launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                      kernel_properties{sycl_exp::sub_group_size<BestlaBenchSubgroupSizeDecode<KernelT>::value>}},
        params);
#else
    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<BestlaBenchSubgroupSizeDecode<KernelT>::value>};
    compat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    return compat::experimental::launch<cutlass::device_kernel<KernelT>, KernelT>(policy, params);
#endif
  }

  template <int KVTile, int NumSGs>
  struct DecodeTileConfigH64 {
    using ShapeQK = cute::Shape<cute::_1, cute::Int<KVTile>, cute::_64>;
    using ShapePV = cute::Shape<cute::_1, cute::_32, cute::Int<KVTile>>;
    using ShapeOutput = cute::Shape<cute::_1, cute::_64, cute::Int<KVTile>>;
    // Match sycl-tla examples/06_bmg_flash_attention subgroup layout
    using SubgroupLayout =
        cute::Layout<cute::Shape<cute::Int<NumSGs>, cute::_1, cute::_1>, cute::Stride<cute::_1, cute::_1, cute::_1>>;
  };

  template <int KVTile, int NumSGs>
  struct DecodeTileConfigH128 {
    using ShapeQK = cute::Shape<cute::_1, cute::Int<KVTile>, cute::_64>;
    using ShapePV = cute::Shape<cute::_1, cute::_32, cute::Int<KVTile>>;
    using ShapeOutput = cute::Shape<cute::_1, cute::_128, cute::Int<KVTile>>;
    // Match sycl-tla examples/06_bmg_flash_attention subgroup layout
    using SubgroupLayout =
        cute::Layout<cute::Shape<cute::Int<NumSGs>, cute::_1, cute::_1>, cute::Stride<cute::_1, cute::_1, cute::_1>>;
  };

  template <int HeadDim>
  using DecodeTileConfig = std::conditional_t<HeadDim == 64, DecodeTileConfigH64</*KVTile=*/512, /*NumSGs=*/8>,
                                              DecodeTileConfigH128</*KVTile=*/512, /*NumSGs=*/8>>;

  template <typename ElementInput, bool Causal, int HeadDim, typename = void>
  struct DecodeKernelBuilder {
    static constexpr bool supported = false;
  };

  // FP16 decode kernel for BMG
  template <bool Causal, int HeadDim>
  struct DecodeKernelBuilder<cutlass::half_t, Causal, HeadDim> {
    static constexpr bool supported = true;

    using ElementOutput = float;
    using ElementAccumulator = float;
    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16</*Stages=*/2>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    using MMAOperation = cute::XE_1x16x16_F32F16F16F32_TT;

    using Tile = DecodeTileConfig<HeadDim>;

    using GmemTiledCopyQ = cute::XE_2D_U16x1x16_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U16x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U16x32x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U32x1x16_ST_N;

    using CollectiveEpilogue = cutlass::flash_attention::collective::FlashDecodeEpilogue<
        EpilogueDispatchPolicy, MMAOperation, typename Tile::ShapeOutput, typename Tile::SubgroupLayout,
        ElementAccumulator, ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput, GmemTiledCopyO>;

    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashDecodeSoftmaxEpilogue<Causal, EpilogueDispatchPolicy,
                                                                         ElementAccumulator>;

    using ProblemShapeType = cute::tuple<int, int, int, int, int, int, int, int>;

    using CollectiveMainloop = cutlass::flash_attention::collective::FlashDecodeMma<
        GEMMDispatchPolicy, ProblemShapeType, cutlass::half_t, cutlass::gemm::TagToStrideA_t<LayoutQ>, cutlass::half_t,
        cutlass::gemm::TagToStrideB_t<LayoutK>, cutlass::half_t, cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation,
        typename Tile::ShapeQK, typename Tile::ShapePV, typename Tile::SubgroupLayout, GmemTiledCopyQ, GmemTiledCopyK,
        GmemTiledCopyV, Causal,
        /*PagedKV=*/false>;

    using Kernel = cutlass::flash_attention::kernel::FMHADecode<ProblemShapeType, CollectiveMainloop,
                                                                CollectiveSoftmaxEpilogue, CollectiveEpilogue>;
  };

  // FP8 (E5M2) decode kernel for BMG - only HeadDim=128 is supported
  // HeadDim=64 fails FP8->FP16 conversion constraint (num_elements must be multiple of fragment_size)
  template <bool Causal>
  struct DecodeKernelBuilder<cutlass::float_e5m2_t, Causal, 128> {
    static constexpr bool supported = true;
    static constexpr int HeadDim = 128;

    using ElementInput = cutlass::float_e5m2_t;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16</*Stages=*/2>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    using MMAOperation = cute::XE_1x16x16_F32F16F16F32_TT;

    using Tile = DecodeTileConfig<HeadDim>;

    // FP8 uses U8 copy operations
    using GmemTiledCopyQ = cute::XE_2D_U8x1x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U8x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U8x32x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U32x1x16_ST_N;

    using CollectiveEpilogue = cutlass::flash_attention::collective::FlashDecodeEpilogue<
        EpilogueDispatchPolicy, MMAOperation, typename Tile::ShapeOutput, typename Tile::SubgroupLayout,
        ElementAccumulator, ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput, GmemTiledCopyO>;

    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashDecodeSoftmaxEpilogue<Causal, EpilogueDispatchPolicy,
                                                                         ElementAccumulator>;

    using ProblemShapeType = cute::tuple<int, int, int, int, int, int, int, int>;

    using CollectiveMainloop = cutlass::flash_attention::collective::FlashDecodeMma<
        GEMMDispatchPolicy, ProblemShapeType, ElementInput, cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInput,
        cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInput, cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation,
        typename Tile::ShapeQK, typename Tile::ShapePV, typename Tile::SubgroupLayout, GmemTiledCopyQ, GmemTiledCopyK,
        GmemTiledCopyV, Causal,
        /*PagedKV=*/false>;

    using Kernel = cutlass::flash_attention::kernel::FMHADecode<ProblemShapeType, CollectiveMainloop,
                                                                CollectiveSoftmaxEpilogue, CollectiveEpilogue>;
  };

  // FP8 (E4M3) decode kernel for BMG - only HeadDim=128 is supported
  // HeadDim=64 fails FP8->FP16 conversion constraint (num_elements must be multiple of fragment_size)
  template <bool Causal>
  struct DecodeKernelBuilder<cutlass::float_e4m3_t, Causal, 128> {
    static constexpr bool supported = true;
    static constexpr int HeadDim = 128;

    using ElementInput = cutlass::float_e4m3_t;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using LayoutQ = cutlass::layout::RowMajor;
    using LayoutK = cutlass::layout::ColumnMajor;
    using LayoutV = cutlass::layout::RowMajor;
    using LayoutO = cutlass::layout::RowMajor;

    using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16</*Stages=*/2>;
    using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
    using MMAOperation = cute::XE_1x16x16_F32F16F16F32_TT;

    using Tile = DecodeTileConfig<HeadDim>;

    // FP8 uses U8 copy operations
    using GmemTiledCopyQ = cute::XE_2D_U8x1x32_LD_N;
    using GmemTiledCopyK = cute::XE_2D_U8x16x16_LD_T;
    using GmemTiledCopyV = cute::XE_2D_U8x32x32_LD_V;
    using GmemTiledCopyO = cute::XE_2D_U32x1x16_ST_N;

    using CollectiveEpilogue = cutlass::flash_attention::collective::FlashDecodeEpilogue<
        EpilogueDispatchPolicy, MMAOperation, typename Tile::ShapeOutput, typename Tile::SubgroupLayout,
        ElementAccumulator, ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutO>, ElementOutput, GmemTiledCopyO>;

    using CollectiveSoftmaxEpilogue =
        cutlass::flash_attention::collective::FlashDecodeSoftmaxEpilogue<Causal, EpilogueDispatchPolicy,
                                                                         ElementAccumulator>;

    using ProblemShapeType = cute::tuple<int, int, int, int, int, int, int, int>;

    using CollectiveMainloop = cutlass::flash_attention::collective::FlashDecodeMma<
        GEMMDispatchPolicy, ProblemShapeType, ElementInput, cutlass::gemm::TagToStrideA_t<LayoutQ>, ElementInput,
        cutlass::gemm::TagToStrideB_t<LayoutK>, ElementInput, cutlass::gemm::TagToStrideB_t<LayoutV>, MMAOperation,
        typename Tile::ShapeQK, typename Tile::ShapePV, typename Tile::SubgroupLayout, GmemTiledCopyQ, GmemTiledCopyK,
        GmemTiledCopyV, Causal,
        /*PagedKV=*/false>;

    using Kernel = cutlass::flash_attention::kernel::FMHADecode<ProblemShapeType, CollectiveMainloop,
                                                                CollectiveSoftmaxEpilogue, CollectiveEpilogue>;
  };

 public:
  template <typename ElementInput, int HeadDim, bool Causal>
  static void bench_one(int batch, int num_heads_q, int num_heads_kv, int cache_len, float softmax_scale,
                        float timems) {
    if constexpr (!DecodeKernelBuilder<ElementInput, Causal, HeadDim>::supported) {
      printf("[sycl-tla flash_attn decode] Skip: unsupported dtype/head (only fp16 kernels wired for BMG)\n");
      return;
    } else {
      using Kernel = typename DecodeKernelBuilder<ElementInput, Causal, HeadDim>::Kernel;

      auto dev = UT_Device::get();
      auto q = dev->getQueue();
      compat::set_default_queue(*q);

      const int seq_qo = 1;
      const int seq_kv_new = 1;

      const size_t q_elems = size_t(seq_qo) * size_t(HeadDim) * size_t(batch) * size_t(num_heads_q);
      const size_t kv_new_elems = size_t(seq_kv_new) * size_t(HeadDim) * size_t(batch) * size_t(num_heads_kv);
      const size_t kv_cache_elems = size_t(cache_len) * size_t(HeadDim) * size_t(batch) * size_t(num_heads_kv);
      const size_t o_elems = q_elems;

      cutlass::DeviceAllocation<ElementInput> dQ(q_elems);
      cutlass::DeviceAllocation<ElementInput> dK_new(kv_new_elems);
      cutlass::DeviceAllocation<ElementInput> dV_new(kv_new_elems);
      cutlass::DeviceAllocation<ElementInput> dK_cache(kv_cache_elems);
      cutlass::DeviceAllocation<ElementInput> dV_cache(kv_cache_elems);
      cutlass::DeviceAllocation<float> dO(o_elems);

      q->memset(dQ.get(), 0, q_elems * sizeof(ElementInput)).wait();
      q->memset(dK_new.get(), 0, kv_new_elems * sizeof(ElementInput)).wait();
      q->memset(dV_new.get(), 0, kv_new_elems * sizeof(ElementInput)).wait();
      q->memset(dK_cache.get(), 0, kv_cache_elems * sizeof(ElementInput)).wait();
      q->memset(dV_cache.get(), 0, kv_cache_elems * sizeof(ElementInput)).wait();
      q->memset(dO.get(), 0, o_elems * sizeof(float)).wait();

      typename Kernel::StrideQ stride_Q = cutlass::make_cute_packed_stride(
          typename Kernel::StrideQ{}, cute::make_shape(seq_qo, HeadDim, batch * num_heads_q));
      typename Kernel::StrideK stride_K = cutlass::make_cute_packed_stride(
          typename Kernel::StrideK{}, cute::make_shape(seq_kv_new, HeadDim, batch * num_heads_kv));
      typename Kernel::StrideV stride_V = cutlass::make_cute_packed_stride(
          typename Kernel::StrideV{}, cute::make_shape(HeadDim, seq_kv_new, batch * num_heads_kv));
      typename Kernel::StrideK stride_K_cache = cutlass::make_cute_packed_stride(
          typename Kernel::StrideK{}, cute::make_shape(cache_len, HeadDim, batch * num_heads_kv));
      typename Kernel::StrideV stride_V_cache = cutlass::make_cute_packed_stride(
          typename Kernel::StrideV{}, cute::make_shape(HeadDim, cache_len, batch * num_heads_kv));
      typename Kernel::StrideO stride_O = cutlass::make_cute_packed_stride(
          typename Kernel::StrideO{}, cute::make_shape(seq_qo, HeadDim, batch * num_heads_q));

      typename Kernel::ProblemShape problem_size =
          cute::make_tuple(batch, num_heads_q, num_heads_kv, seq_qo, seq_kv_new, cache_len, HeadDim, HeadDim);

      cutlass::KernelHardwareInfo hw_info;
      typename Kernel::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                           problem_size,
                                           {
                                               dQ.get(),
                                               stride_Q,
                                               dK_new.get(),
                                               stride_K,
                                               dV_new.get(),
                                               stride_V,
                                               dK_cache.get(),
                                               stride_K_cache,
                                               dV_cache.get(),
                                               stride_V_cache,
                                               /*ptr_page_table=*/nullptr,
                                               /*page_size=*/0,
                                               /*num_pages_per_seq=*/nullptr,
                                           },
                                           {softmax_scale},
                                           {dO.get(), stride_O},
                                           hw_info};

      if (!Kernel::can_implement(arguments)) {
        printf("[sycl-tla flash_attn decode] Skip: can_implement failed\n");
        return;
      }

      size_t workspace_size = Kernel::get_workspace_size(arguments);
      cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
      auto status = Kernel::initialize_workspace(arguments, workspace.get());
      if (status != cutlass::Status::kSuccess) {
        printf("[sycl-tla flash_attn decode] Skip: initialize_workspace failed\n");
        return;
      }

      auto params = Kernel::to_underlying_arguments(arguments, workspace.get());
      (void)launch_decode<Kernel>(params).wait();

      using LOG = timer_statistics_logger<TestMs * 2>;
      LOG log;
      utils::timer<std::chrono::milliseconds> wall;
      wall.start();
      while (wall.stop() < timems) {
        auto ev = launch_decode<Kernel>(params);
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (wall.stop() >= timems) break;
      }
      log.record();

      // Print sycl-tla runner style metrics for easy comparison.
      // For decode, Sq=1 and Skv=cache_len; FLOPs convention matches the prefill runner:
      //   flops = 4 * B * Hq * Sq * Skv * D
      const double time_ms = static_cast<double>(log.avg_val);
      const double time_s = time_ms * 1e-3;
      const double flops = 4.0 * static_cast<double>(batch) * static_cast<double>(num_heads_q) * 1.0 *
                           static_cast<double>(cache_len) * static_cast<double>(HeadDim);

      // Count K/V bytes against Hq (logical per-q-head work), matching the runner's intent.
      const double kv_read_elems_equiv = static_cast<double>(batch) * static_cast<double>(num_heads_q) *
                                         static_cast<double>(cache_len) * static_cast<double>(HeadDim);
      const double bytes = static_cast<double>(q_elems) * sizeof(ElementInput) +
                           2.0 * kv_read_elems_equiv * sizeof(ElementInput) +
                           static_cast<double>(o_elems) * sizeof(float);
      const double tflops = (time_s > 0.0) ? (flops / time_s / 1.0e12) : 0.0;
      const double gbs = (time_s > 0.0) ? (bytes / time_s / 1.0e9) : 0.0;
      printf("Performance:   %.3f  GB/s,    %.3f  TFlop/s,   %.4f  ms (dtype=%s)\n", gbs, tflops, time_ms,
             dtype_str<ElementInput>());

      printf("[sycl-tla flash_attn decode] dtype=%s B=%d Hq=%d Hkv=%d cache=%d D=%d causal=%d  %s\n",
             dtype_str<ElementInput>(), batch, num_heads_q, num_heads_kv, cache_len, HeadDim, int(Causal),
             log.get_log_str());
    }
  }

  static void run_all_fp16(int batch, int num_heads_q, int num_heads_kv, int head_dim, int cache_len, bool is_causal) {
    printf("\n[sycl-tla flash_attn decode fp16] case: B=%d Hq=%d Hkv=%d cache=%d D=%d causal=%d\n", batch, num_heads_q,
           num_heads_kv, cache_len, head_dim, int(is_causal));
    if (head_dim == 64) {
      if (is_causal)
        bench_one<cutlass::half_t, 64, true>(batch, num_heads_q, num_heads_kv, cache_len, 1.0f, float(TestMs));
      else
        bench_one<cutlass::half_t, 64, false>(batch, num_heads_q, num_heads_kv, cache_len, 1.0f, float(TestMs));
    } else if (head_dim == 128) {
      if (is_causal)
        bench_one<cutlass::half_t, 128, true>(batch, num_heads_q, num_heads_kv, cache_len, 1.0f, float(TestMs));
      else
        bench_one<cutlass::half_t, 128, false>(batch, num_heads_q, num_heads_kv, cache_len, 1.0f, float(TestMs));
    }
  }
};
#if 0
static Benchmark_SyclTla_FlashAttnDecode sBenchmark_SyclTla_FlashAttnDecode;
#endif
// ========================================================================
// Flash Attention Decode FP8
// ========================================================================

class Benchmark_SyclTla_FlashAttnDecodeFP8 {
 public:
  Benchmark_SyclTla_FlashAttnDecodeFP8() {
    UT_START();
    // Baseline
    run_all_fp8(/*batch=*/1, /*num_heads_q=*/32, /*num_heads_kv=*/32, /*head_dim=*/128, /*cache_len=*/4096,
                /*is_causal=*/true);

    // Heavier-cache cases to expose bandwidth effects (use fewer heads to keep memory reasonable).
    run_all_fp8(/*batch=*/4, /*num_heads_q=*/16, /*num_heads_kv=*/16, /*head_dim=*/128, /*cache_len=*/8192,
                /*is_causal=*/true);
    run_all_fp8(/*batch=*/8, /*num_heads_q=*/8, /*num_heads_kv=*/8, /*head_dim=*/128, /*cache_len=*/8192,
                /*is_causal=*/true);
  }

 private:
  static void run_all_fp8(int batch, int num_heads_q, int num_heads_kv, int head_dim, int cache_len, bool is_causal) {
    printf("\n[sycl-tla flash_attn decode fp8] case: B=%d Hq=%d Hkv=%d cache=%d D=%d causal=%d\n", batch, num_heads_q,
           num_heads_kv, cache_len, head_dim, int(is_causal));

    // NOTE:
    // sycl-tla BMG FP8 decode example (examples/06_bmg_flash_attention/06_bmg_decode_attention_fp8.cpp)
    // is wired for FP8 E5M2. Instantiating E4M3 currently triggers a static_assert in fp8_to_fp16.h
    // (work-item must convert a multiple of fragment_size).

    if (head_dim == 64) {
      if (is_causal) {
        Benchmark_SyclTla_FlashAttnDecode::bench_one<cutlass::float_e5m2_t, 64, true>(batch, num_heads_q, num_heads_kv,
                                                                                      cache_len, 1.0f, float(TestMs));
      } else {
        Benchmark_SyclTla_FlashAttnDecode::bench_one<cutlass::float_e5m2_t, 64, false>(batch, num_heads_q, num_heads_kv,
                                                                                       cache_len, 1.0f, float(TestMs));
      }
    } else if (head_dim == 128) {
      if (is_causal) {
        Benchmark_SyclTla_FlashAttnDecode::bench_one<cutlass::float_e5m2_t, 128, true>(batch, num_heads_q, num_heads_kv,
                                                                                       cache_len, 1.0f, float(TestMs));
      } else {
        Benchmark_SyclTla_FlashAttnDecode::bench_one<cutlass::float_e5m2_t, 128, false>(
            batch, num_heads_q, num_heads_kv, cache_len, 1.0f, float(TestMs));
      }
    }
  }
};

#if 0
static Benchmark_SyclTla_FlashAttnDecodeFP8 sBenchmark_SyclTla_FlashAttnDecodeFP8;
#endif

#endif  // ARK_SYCL_TLA
}  // namespace sycl_ut
}  // namespace bestla
