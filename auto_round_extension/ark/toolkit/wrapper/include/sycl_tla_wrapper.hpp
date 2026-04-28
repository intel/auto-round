// SYCL-TLA Flash Attention Wrapper (Header-Only)
// This file wraps sycl-tla flash attention kernels for use in ark.cpp
//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef ARK_XPU
#include <sycl/sycl.hpp>
#endif

#if defined(ARK_SYCL_TLA)

#include <type_traits>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

// FlashAttention-v2 (Xe) building blocks (from sycl-tla).
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/tile_scheduler.hpp"

// FlashAttention-v2 Prefill kernels
#include "flash_attention_v2/kernel/xe_flash_attn_prefill.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_epilogue.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_softmax_epilogue.hpp"

// FlashAttention-v2 Decode kernels
// NOTE: Decode is temporarily disabled due to convert_type ODR violation in sycl-tla
// Both prefill_mma.hpp and decode_mma.hpp define the same convert_type function
// TODO: Enable when upstream sycl-tla fixes this issue
// #include "flash_attention_v2/kernel/xe_flash_attn_decode.hpp"
// #include "flash_attention_v2/collective/xe_flash_attn_decode_epilogue.hpp"
// #include "flash_attention_v2/collective/xe_flash_attn_decode_softmax_epilogue.hpp"
#define ARK_FLASH_ATTN_DECODE_DISABLED 1

// CUTLASS-SYCL/CuTe compat layer
#include "cute/util/compat.hpp"
#include "cute/tensor.hpp"

// sycl-tla example utilities
#include "helper.h"
#include "sycl_common.hpp"

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

// Import cute namespace for convenience in kernel builders
using namespace cute;

// Helper to get subgroup size from kernel
template <typename Kernel, typename = void>
struct KernelSubgroupSize {
  static constexpr int value = Kernel::DispatchPolicy::SubgroupSize;
};

template <typename Kernel>
struct KernelSubgroupSize<Kernel, std::void_t<decltype(Kernel::SubgroupSize)>> {
  static constexpr int value = Kernel::SubgroupSize;
};

// ========================================================================
// Prefill Kernel Launch
// ========================================================================

template <typename KernelT>
inline sycl::event launch_prefill_kernel(typename KernelT::Params const& kernel_params) {
  compat::dim3 const block = KernelT::get_block_shape();
  compat::dim3 const grid = KernelT::get_grid_shape(kernel_params);
  int smem_size = KernelT::SharedStorageSize;

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
  using namespace compat::experimental;
  return launch<cutlass::device_kernel<KernelT>>(
      launch_policy{grid,
                    block,
                    local_mem_size{static_cast<std::size_t>(smem_size)},
                    kernel_properties{sycl_exp::sub_group_size<KernelSubgroupSize<KernelT>::value>}},
      kernel_params);
#else
  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<KernelSubgroupSize<KernelT>::value>,
  };
  compat::experimental::launch_policy policy{grid, block, launch_props, kernel_props};
  return compat::experimental::launch<cutlass::device_kernel<KernelT>, KernelT>(policy, kernel_params);
#endif
}

// ========================================================================
// Prefill Kernel Builder Templates
// ========================================================================

// FP16 Prefill Kernel
template <int HeadDim, bool Causal>
struct PrefillKernelBuilderFP16 {
  using ElementInput = cutlass::half_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  static constexpr int PipelineStages = 2;

  using MMAOperation = XE_8x16x16_F32F16F16F32_TT;
  using GmemTiledCopyQ = XE_2D_U16x8x16_LD_N;
  using GmemTiledCopyK = XE_2D_U16x16x16_LD_T;
  using GmemTiledCopyV = XE_2D_U16x32x32_LD_V;
  using GmemTiledCopyO = XE_2D_U32x8x16_ST_N;

  // Shape format: [CTA_M, CTA_N, CTA_K] - matches sycl-tla examples
  using ShapeQK = std::conditional_t<HeadDim == 64, Shape<_128, _64, _64>, Shape<_128, _64, _64>>;
  using ShapePV = std::conditional_t<HeadDim == 64, Shape<_128, _32, _64>, Shape<_128, _32, _64>>;
  // ShapeOutput must be rank-3: [CTA_M_QO, CTA_N_VO (HeadDim), CTA_K_PV]
  using ShapeOutput = std::conditional_t<HeadDim == 64, Shape<_128, _64, _64>, Shape<_128, _128, _64>>;
  using SubgroupLayout = std::conditional_t<HeadDim == 64, Layout<Shape<_8, _1, _1>>, Layout<Shape<_16, _1, _1>>>;

  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;

  using CollectiveEpilogue = cutlass::flash_attention::collective::FlashPrefillEpilogue<
      EpilogueDispatchPolicy,
      MMAOperation,
      ShapeOutput,
      SubgroupLayout,
      ElementComputeEpilogue,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutO>,
      ElementOutput,
      GmemTiledCopyO>;

  using CollectiveSoftmaxEpilogue =
      cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<Causal, EpilogueDispatchPolicy, ElementAccumulator>;

  using ProblemShape = cute::tuple<int, int, int, int, int, int, int>;

  using CollectiveMainloop = cutlass::flash_attention::collective::FlashPrefillMma<
      GEMMDispatchPolicy,
      ProblemShape,
      ElementInput,
      cutlass::gemm::TagToStrideA_t<LayoutQ>,
      ElementInput,
      cutlass::gemm::TagToStrideB_t<LayoutK>,
      ElementInput,
      cutlass::gemm::TagToStrideB_t<LayoutV>,
      MMAOperation,
      ShapeQK,
      ShapePV,
      SubgroupLayout,
      GmemTiledCopyQ,
      GmemTiledCopyK,
      GmemTiledCopyV,
      Causal>;

  using Kernel = cutlass::flash_attention::kernel::FMHAPrefill<
      ProblemShape,
      CollectiveMainloop,
      CollectiveSoftmaxEpilogue,
      CollectiveEpilogue,
      cutlass::flash_attention::IndividualScheduler>;
};

#ifndef ARK_FLASH_ATTN_DECODE_DISABLED
// ========================================================================
// Decode Kernel Builder Templates
// ========================================================================

template <int KVTile, int NumSGs>
struct DecodeTileConfigH64 {
  using ShapeQK = cute::Shape<cute::_1, cute::Int<KVTile>, cute::_64>;
  using ShapePV = cute::Shape<cute::_1, cute::_32, cute::Int<KVTile>>;
  using ShapeOutput = cute::Shape<cute::_1, cute::_64, cute::Int<KVTile>>;
  using SubgroupLayout = cute::Layout<cute::Shape<cute::Int<NumSGs>, cute::_1, cute::_1>,
                                      cute::Stride<cute::_1, cute::_1, cute::_1>>;
};

template <int KVTile, int NumSGs>
struct DecodeTileConfigH128 {
  using ShapeQK = cute::Shape<cute::_1, cute::Int<KVTile>, cute::_64>;
  using ShapePV = cute::Shape<cute::_1, cute::_32, cute::Int<KVTile>>;
  using ShapeOutput = cute::Shape<cute::_1, cute::_128, cute::Int<KVTile>>;
  using SubgroupLayout = cute::Layout<cute::Shape<cute::Int<NumSGs>, cute::_1, cute::_1>,
                                      cute::Stride<cute::_1, cute::_1, cute::_1>>;
};

template <int HeadDim>
using DecodeTileConfig = std::conditional_t<HeadDim == 64,
                                           DecodeTileConfigH64<512, 8>,
                                           DecodeTileConfigH128<512, 8>>;

// FP16 Decode Kernel
template <int HeadDim, bool Causal>
struct DecodeKernelBuilderFP16 {
  using ElementInput = cutlass::half_t;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<2>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
  using MMAOperation = cute::XE_1x16x16_F32F16F16F32_TT;

  using Tile = DecodeTileConfig<HeadDim>;

  using GmemTiledCopyQ = cute::XE_2D_U16x1x16_LD_N;
  using GmemTiledCopyK = cute::XE_2D_U16x16x16_LD_T;
  using GmemTiledCopyV = cute::XE_2D_U16x32x32_LD_V;
  using GmemTiledCopyO = cute::XE_2D_U32x1x16_ST_N;

  using CollectiveEpilogue = cutlass::flash_attention::collective::FlashDecodeEpilogue<
      EpilogueDispatchPolicy,
      MMAOperation,
      typename Tile::ShapeOutput,
      typename Tile::SubgroupLayout,
      ElementAccumulator,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutO>,
      ElementOutput,
      GmemTiledCopyO>;

  using CollectiveSoftmaxEpilogue =
      cutlass::flash_attention::collective::FlashDecodeSoftmaxEpilogue<Causal, EpilogueDispatchPolicy, ElementAccumulator>;

  using ProblemShapeType = cute::tuple<int, int, int, int, int, int, int, int>;

  using CollectiveMainloop = cutlass::flash_attention::collective::FlashDecodeMma<
      GEMMDispatchPolicy,
      ProblemShapeType,
      ElementInput,
      cutlass::gemm::TagToStrideA_t<LayoutQ>,
      ElementInput,
      cutlass::gemm::TagToStrideB_t<LayoutK>,
      ElementInput,
      cutlass::gemm::TagToStrideB_t<LayoutV>,
      MMAOperation,
      typename Tile::ShapeQK,
      typename Tile::ShapePV,
      typename Tile::SubgroupLayout,
      GmemTiledCopyQ,
      GmemTiledCopyK,
      GmemTiledCopyV,
      Causal,
      false>;

  using Kernel = cutlass::flash_attention::kernel::FMHADecode<
      ProblemShapeType,
      CollectiveMainloop,
      CollectiveSoftmaxEpilogue,
      CollectiveEpilogue>;
};

// ========================================================================
// Decode Kernel Launch
// ========================================================================

template <typename KernelT>
inline sycl::event launch_decode_kernel(typename KernelT::Params const& params) {
  compat::dim3 const block = KernelT::get_block_shape();
  compat::dim3 const grid = KernelT::get_grid_shape(params);
  int smem_size = KernelT::SharedStorageSize;

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
  using namespace compat::experimental;
  return launch<cutlass::device_kernel<KernelT>>(
      launch_policy{grid,
                    block,
                    local_mem_size{static_cast<std::size_t>(smem_size)},
                    kernel_properties{sycl_exp::sub_group_size<KernelSubgroupSize<KernelT>::value>}},
      params);
#else
  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<KernelSubgroupSize<KernelT>::value>,
  };
  compat::experimental::launch_policy policy{grid, block, launch_props, kernel_props};
  return compat::experimental::launch<cutlass::device_kernel<KernelT>, KernelT>(policy, params);
#endif
}
#endif  // ARK_FLASH_ATTN_DECODE_DISABLED

// ========================================================================
// Prefill Implementation
// ========================================================================

template <typename KernelBuilder>
inline void run_prefill_impl(
    sycl::queue* q,
    void* Q_ptr,
    void* K_ptr,
    void* V_ptr,
    void* O_ptr,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float softmax_scale) {
  using namespace cute;
  using Kernel = typename KernelBuilder::Kernel;
  using ElementInput = typename KernelBuilder::ElementInput;
  using ElementOutput = typename KernelBuilder::ElementOutput;
  using ProblemShape = typename Kernel::ProblemShape;

  compat::set_default_queue(*q);

  typename Kernel::StrideQ stride_Q = cutlass::make_cute_packed_stride(
      typename Kernel::StrideQ{}, make_shape(seq_len_q, head_dim, batch * num_heads_q));
  typename Kernel::StrideK stride_K = cutlass::make_cute_packed_stride(
      typename Kernel::StrideK{}, make_shape(seq_len_kv, head_dim, batch * num_heads_kv));
  typename Kernel::StrideV stride_V = cutlass::make_cute_packed_stride(
      typename Kernel::StrideV{}, make_shape(head_dim, seq_len_kv, batch * num_heads_kv));
  typename Kernel::StrideO stride_O = cutlass::make_cute_packed_stride(
      typename Kernel::StrideO{}, make_shape(seq_len_q, head_dim, batch * num_heads_q));

  // Problem shape: (batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, head_dim_v)
  ProblemShape problem_shape = make_tuple(batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, head_dim);

  // Hardware info
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = 64;  // Will be auto-detected if needed

  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {(ElementInput*)Q_ptr, stride_Q,
       (ElementInput*)K_ptr, stride_K,
       (ElementInput*)V_ptr, stride_V},
      {softmax_scale},
      {(ElementOutput*)O_ptr, stride_O},
      hw_info};

  // Get workspace size and allocate
  size_t workspace_size = Kernel::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Initialize workspace if needed
  Kernel::initialize_workspace(args, workspace.get());

  typename Kernel::Params params = Kernel::to_underlying_arguments(args, workspace.get());
  launch_prefill_kernel<Kernel>(params).wait();
}

template <int HeadDim, bool Causal>
inline void dispatch_prefill_dtype(
    sycl::queue* q,
    void* Q_ptr,
    void* K_ptr,
    void* V_ptr,
    void* O_ptr,
    FlashAttnDtype dtype,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_q,
    int seq_len_kv,
    float softmax_scale) {
  switch (dtype) {
    case FlashAttnDtype::FP16:
      run_prefill_impl<PrefillKernelBuilderFP16<HeadDim, Causal>>(
          q, Q_ptr, K_ptr, V_ptr, O_ptr, batch, num_heads_q, num_heads_kv,
          seq_len_q, seq_len_kv, HeadDim, softmax_scale);
      break;
    default:
      throw std::runtime_error("Unsupported dtype for flash_attn_prefill (only FP16 supported)");
  }
}

template <int HeadDim>
inline void dispatch_prefill_causal(
    sycl::queue* q,
    void* Q_ptr,
    void* K_ptr,
    void* V_ptr,
    void* O_ptr,
    FlashAttnDtype dtype,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_q,
    int seq_len_kv,
    float softmax_scale,
    bool is_causal) {
  if (is_causal) {
    dispatch_prefill_dtype<HeadDim, true>(
        q, Q_ptr, K_ptr, V_ptr, O_ptr, dtype, batch, num_heads_q, num_heads_kv,
        seq_len_q, seq_len_kv, softmax_scale);
  } else {
    dispatch_prefill_dtype<HeadDim, false>(
        q, Q_ptr, K_ptr, V_ptr, O_ptr, dtype, batch, num_heads_q, num_heads_kv,
        seq_len_q, seq_len_kv, softmax_scale);
  }
}

#ifndef ARK_FLASH_ATTN_DECODE_DISABLED
// ========================================================================
// Decode Implementation
// ========================================================================

template <typename KernelBuilder>
inline void run_decode_impl(
    sycl::queue* q,
    void* Q_ptr,
    void* K_new_ptr,
    void* V_new_ptr,
    void* K_cache_ptr,
    void* V_cache_ptr,
    void* O_ptr,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_kv_cache,
    int head_dim,
    float softmax_scale) {
  using namespace cute;
  using Kernel = typename KernelBuilder::Kernel;
  using ElementInput = typename KernelBuilder::ElementInput;

  compat::set_default_queue(*q);

  const int seq_qo = 1;
  const int seq_kv_new = 1;

  typename Kernel::StrideQ stride_Q = cutlass::make_cute_packed_stride(
      typename Kernel::StrideQ{}, make_shape(seq_qo, head_dim, batch * num_heads_q));
  typename Kernel::StrideK stride_K = cutlass::make_cute_packed_stride(
      typename Kernel::StrideK{}, make_shape(seq_kv_new, head_dim, batch * num_heads_kv));
  typename Kernel::StrideV stride_V = cutlass::make_cute_packed_stride(
      typename Kernel::StrideV{}, make_shape(head_dim, seq_kv_new, batch * num_heads_kv));
  typename Kernel::StrideK stride_K_cache = cutlass::make_cute_packed_stride(
      typename Kernel::StrideK{}, make_shape(seq_len_kv_cache, head_dim, batch * num_heads_kv));
  typename Kernel::StrideV stride_V_cache = cutlass::make_cute_packed_stride(
      typename Kernel::StrideV{}, make_shape(head_dim, seq_len_kv_cache, batch * num_heads_kv));
  typename Kernel::StrideO stride_O = cutlass::make_cute_packed_stride(
      typename Kernel::StrideO{}, make_shape(seq_qo, head_dim, batch * num_heads_q));

  typename Kernel::Arguments args{
      make_tuple(batch, num_heads_q, num_heads_kv, seq_qo, seq_kv_new, head_dim, seq_len_kv_cache, /*page_size=*/0),
      {(ElementInput*)Q_ptr, stride_Q,
       (ElementInput*)K_new_ptr, stride_K,
       (ElementInput*)V_new_ptr, stride_V,
       (ElementInput*)K_cache_ptr, stride_K_cache,
       (ElementInput*)V_cache_ptr, stride_V_cache,
       softmax_scale,
       nullptr,  // page_table
       0},       // page_table_batch
      {},
      {(float*)O_ptr, stride_O}};

  typename Kernel::Params params = Kernel::to_underlying_arguments(args);
  launch_decode_kernel<Kernel>(params).wait();
}

template <int HeadDim, bool Causal>
inline void dispatch_decode_dtype(
    sycl::queue* q,
    void* Q_ptr,
    void* K_new_ptr,
    void* V_new_ptr,
    void* K_cache_ptr,
    void* V_cache_ptr,
    void* O_ptr,
    FlashAttnDtype dtype,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_kv_cache,
    float softmax_scale) {
  switch (dtype) {
    case FlashAttnDtype::FP16:
      run_decode_impl<DecodeKernelBuilderFP16<HeadDim, Causal>>(
          q, Q_ptr, K_new_ptr, V_new_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
          batch, num_heads_q, num_heads_kv, seq_len_kv_cache, HeadDim, softmax_scale);
      break;
    default:
      throw std::runtime_error("Unsupported dtype for flash_attn_decode (only FP16 supported)");
  }
}

template <int HeadDim>
inline void dispatch_decode_causal(
    sycl::queue* q,
    void* Q_ptr,
    void* K_new_ptr,
    void* V_new_ptr,
    void* K_cache_ptr,
    void* V_cache_ptr,
    void* O_ptr,
    FlashAttnDtype dtype,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_kv_cache,
    float softmax_scale,
    bool is_causal) {
  if (is_causal) {
    dispatch_decode_dtype<HeadDim, true>(
        q, Q_ptr, K_new_ptr, V_new_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
        dtype, batch, num_heads_q, num_heads_kv, seq_len_kv_cache, softmax_scale);
  } else {
    dispatch_decode_dtype<HeadDim, false>(
        q, Q_ptr, K_new_ptr, V_new_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
        dtype, batch, num_heads_q, num_heads_kv, seq_len_kv_cache, softmax_scale);
  }
}
#endif  // ARK_FLASH_ATTN_DECODE_DISABLED

}  // namespace detail

// ========================================================================
// Public API
// ========================================================================

/**
 * @brief Flash Attention Prefill (FP16)
 *
 * @param q SYCL queue
 * @param Q_ptr  Pointer to Q tensor [B, Hq, Sq, D]
 * @param K_ptr  Pointer to K tensor [B, Hkv, Skv, D]
 * @param V_ptr  Pointer to V tensor [B, Hkv, Skv, D]
 * @param O_ptr  Pointer to output tensor [B, Hq, Sq, D], fp32
 * @param q_dtype  Q/K/V data type (FP16)
 * @param batch  Batch size
 * @param num_heads_q  Number of query heads
 * @param num_heads_kv Number of KV heads
 * @param seq_len_q  Query sequence length
 * @param seq_len_kv  KV sequence length
 * @param head_dim  Head dimension (64 or 128)
 * @param softmax_scale  Softmax scale factor
 * @param is_causal  Whether to apply causal mask
 */
inline void flash_attn_prefill(
    sycl::queue* q,
    void* Q_ptr,
    void* K_ptr,
    void* V_ptr,
    void* O_ptr,
    FlashAttnDtype q_dtype,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    float softmax_scale,
    bool is_causal) {
  if (head_dim == 64) {
    detail::dispatch_prefill_causal<64>(
        q, Q_ptr, K_ptr, V_ptr, O_ptr, q_dtype, batch, num_heads_q, num_heads_kv,
        seq_len_q, seq_len_kv, softmax_scale, is_causal);
  } else if (head_dim == 128) {
    detail::dispatch_prefill_causal<128>(
        q, Q_ptr, K_ptr, V_ptr, O_ptr, q_dtype, batch, num_heads_q, num_heads_kv,
        seq_len_q, seq_len_kv, softmax_scale, is_causal);
  } else {
    throw std::runtime_error("flash_attn_prefill: unsupported head_dim (must be 64 or 128)");
  }
}

#ifndef ARK_FLASH_ATTN_DECODE_DISABLED
/**
 * @brief Flash Attention Decode (FP16)
 *
 * @param q SYCL queue
 * @param Q_ptr  Pointer to Q tensor [B, Hq, 1, D]
 * @param K_new_ptr  Pointer to new K tensor [B, Hkv, 1, D]
 * @param V_new_ptr  Pointer to new V tensor [B, Hkv, 1, D]
 * @param K_cache_ptr  Pointer to K cache [B, Hkv, Scache, D] (nullptr if no cache)
 * @param V_cache_ptr  Pointer to V cache [B, Hkv, Scache, D] (nullptr if no cache)
 * @param O_ptr  Pointer to output tensor [B, Hq, 1, D], fp32
 * @param q_dtype  Q/K/V data type (FP16)
 * @param batch  Batch size
 * @param num_heads_q  Number of query heads
 * @param num_heads_kv Number of KV heads
 * @param seq_len_kv_cache  KV cache sequence length (0 if no cache)
 * @param head_dim  Head dimension (64 or 128)
 * @param softmax_scale  Softmax scale factor
 * @param is_causal  Whether to apply causal mask
 */
inline void flash_attn_decode(
    sycl::queue* q,
    void* Q_ptr,
    void* K_new_ptr,
    void* V_new_ptr,
    void* K_cache_ptr,
    void* V_cache_ptr,
    void* O_ptr,
    FlashAttnDtype q_dtype,
    int batch,
    int num_heads_q,
    int num_heads_kv,
    int seq_len_kv_cache,
    int head_dim,
    float softmax_scale,
    bool is_causal) {
  if (head_dim == 64) {
    detail::dispatch_decode_causal<64>(
        q, Q_ptr, K_new_ptr, V_new_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
        q_dtype, batch, num_heads_q, num_heads_kv, seq_len_kv_cache, softmax_scale, is_causal);
  } else if (head_dim == 128) {
    detail::dispatch_decode_causal<128>(
        q, Q_ptr, K_new_ptr, V_new_ptr, K_cache_ptr, V_cache_ptr, O_ptr,
        q_dtype, batch, num_heads_q, num_heads_kv, seq_len_kv_cache, softmax_scale, is_causal);
  } else {
    throw std::runtime_error("flash_attn_decode: unsupported head_dim (must be 64 or 128)");
  }
}
#endif  // ARK_FLASH_ATTN_DECODE_DISABLED

#endif  // ARK_SYCL_TLA

}  // namespace ark
