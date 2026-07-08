//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <pybind11/pybind11.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "bestla/bestla/bestla.h"
typedef uintptr_t torch_ptr;
#if ARK_XPU
#include <sycl/sycl.hpp>
#include "xpu_wrapper.hpp"
#include "sycl_s8_wrapper.hpp"
#if ARK_SYCL_TLA
// Only include declarations, implementations are in separate .cpp files
#include "sycl_tla_common.hpp"
#include "sycl_tla_moe.hpp"
#include "sycl_tla_moe_decode.hpp"
#include "sycl_tla_moe_mixed.hpp"
#include "sycl_tla_sdpa.hpp"
#include "sycl_tla_dense_gemm.hpp"
#endif
#else
#include "ark/cpu/sdpa.h"
#include "cpu_wrapper.hpp"
#endif

#if ARK_DNNL
#include "dnnl_wrapper.hpp"
#endif

namespace ark {

static void matmul(torch_ptr stream, int m, int n, int k, torch_ptr A, int Adt, torch_ptr B, int Bdt, torch_ptr C,
                   int Cdt, torch_ptr bias, bool BT) {
#ifdef ARK_XPU
#if ARK_DNNL
  auto dt = ark::to_dt((BTLA_DTYPE)Adt);
  auto cdt = dt;
  if (Adt == (int)BTLA_DTYPE::S8) cdt = dnnl::memory::data_type::s32;
  ark::DnnlWrapper::gemm((sycl::queue*)stream, m, n, k, (void*)A, dt, (void*)B, dt, BT, (void*)C, cdt, (void*)bias);
#elif ARK_SYCL_TLA
  ark::sycl_tla_dense_gemm((sycl::queue*)stream, m, n, k, (void*)A, (BTLA_DTYPE)Adt, (void*)B, (BTLA_DTYPE)Bdt,
                           (void*)C, (BTLA_DTYPE)Cdt, (void*)bias, BT);
#else
  throw std::runtime_error("ark::matmul on XPU requires ARK_DNNL=ON or ARK_SYCL_TLA=ON");
#endif
#else
  CpuWrapper::gemm(m, n, k, (void*)A, (BTLA_DTYPE)Adt, (void*)B, BT, (float*)C, (const float*)bias);
#endif
}

static void woqgemm_s8(torch_ptr stream, int m, int n, int k, torch_ptr A, int ACdt, torch_ptr B, torch_ptr C,
                       torch_ptr bias, bool BT, torch_ptr scaleb) {
#if ARK_XPU
  ark::SyclS8Wrapper::woq_s8((sycl::queue*)stream, m, n, k, (void*)A, (void*)B, BT, (void*)C,
                             (BTLA_DTYPE)ACdt, (void*)scaleb, (void*)bias, k);
#elif ARK_DNNL
  auto dt = ark::to_dt((BTLA_DTYPE)ACdt);
  ark::DnnlWrapper::woq_s8((sycl::queue*)stream, m, n, k, (void*)A, (void*)B, BT, (void*)C, dt, (void*)scaleb,
                           (void*)bias, k);
#else
  throw std::runtime_error("ark::woqgemm_s8 requires ARK_XPU or ARK_DNNL");
#endif
}

static void woqgemm(torch_ptr stream, int m, int n, int k, torch_ptr A, int ACdt, torch_ptr BlobB, torch_ptr C,
                    torch_ptr bias, int blocksize, int compute_type, int weight_type, int scale_type, bool asym,
                    int blob_numel = 0) {
  QuantParam param{n, k, blocksize, compute_type, weight_type, scale_type, asym};
  size_t bc = static_cast<size_t>(blob_numel);
#ifdef ARK_XPU
  XpuWrapper::woq_gemm(m, (void*)A, (void*)BlobB, (void*)C, (void*)bias, (BTLA_DTYPE)ACdt, &param,
                       (sycl::queue*)stream, bc);
#else
  CpuWrapper::woq_gemm(m, (void*)A, (void*)BlobB, (void*)C, (void*)bias, (BTLA_DTYPE)ACdt, &param, bc);
#endif
}

static void repack_quantized_weight(torch_ptr stream, torch_ptr raws8, torch_ptr zp, torch_ptr scale, torch_ptr blob,
                                    int n, int k, int blocksize, int compute_type, int weight_type, int scale_type,
                                    bool asym) {
  QuantParam param{n, k, blocksize, compute_type, weight_type, scale_type, asym};
#ifdef ARK_XPU
  XpuWrapper::packq((int8_t*)raws8, (void*)scale, (int8_t*)zp, (int8_t*)blob, &param, (sycl::queue*)stream);
#else
  CpuWrapper::packq((int8_t*)raws8, (float*)scale, (int8_t*)zp, (int8_t*)blob, &param);
#endif
}

static void unpack_weight(torch_ptr stream, torch_ptr blob, torch_ptr output, int out_type, int n, int k, int blocksize,
                          int compute_type, int weight_type, int scale_type, bool asym, int blob_numel = 0) {
  QuantParam param{n, k, blocksize, compute_type, weight_type, scale_type, asym};
  size_t bc = static_cast<size_t>(blob_numel);
#ifdef ARK_XPU
  XpuWrapper::unpackq((BTLA_DTYPE)out_type, (int8_t*)blob, (void*)output, &param, (sycl::queue*)stream, bc);
#else
  CpuWrapper::unpackq((BTLA_DTYPE)out_type, (int8_t*)blob, (void*)output, &param, bc);
#endif
}

static size_t packed_weight_size(torch_ptr stream, int n, int k, int blocksize, int compute_type, int weight_type,
                                 int scale_type, bool asym) {
  QuantParam param{n, k, blocksize, compute_type, weight_type, scale_type, asym};
#ifdef ARK_XPU
  return XpuWrapper::get_packw_size(&param);
#else
  return CpuWrapper::get_packw_size(&param);
#endif
}

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)

static void matmul_sycl_tla(torch_ptr stream, int m, int n, int k, torch_ptr A, int Adt, torch_ptr B,
                                       int Bdt, torch_ptr C, int Cdt, torch_ptr bias, bool BT) {
  ark::sycl_tla_dense_gemm((sycl::queue*)stream, m, n, k, (void*)A, (BTLA_DTYPE)Adt, (void*)B,
                                      (BTLA_DTYPE)Bdt, (void*)C, (BTLA_DTYPE)Cdt, (void*)bias, BT);
}

// Tensor layout codes passed from Python (tensor_layout argument).
constexpr int TENSOR_LAYOUT_HND = 0;  // [B, H, S, D]
constexpr int TENSOR_LAYOUT_NHD = 1;  // [B, S, H, D]

static void sdpa(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                 int q_dtype, int k_dtype, int o_dtype,
                 int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                 float softmax_scale, bool is_causal,
                 int tensor_layout, torch_ptr lse = 0) {
  if (k_dtype != q_dtype || o_dtype != q_dtype) {
    throw std::invalid_argument("ark::sdpa: k_dtype and o_dtype must match q_dtype");
  }
  if(q_dtype != (int)BTLA_DTYPE::F16 && q_dtype != (int)BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::sdpa: only FP16 and BF16 are supported");
  }
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sdpa: mask and is_causal cannot both be set");
  }
  int q_stride_s, q_stride_d, q_stride_h, q_stride_b;
  int k_stride_s, k_stride_d, k_stride_h, k_stride_b;
  int v_stride_d, v_stride_s, v_stride_h, v_stride_b;
  int o_stride_s, o_stride_d, o_stride_h, o_stride_b;
  if (tensor_layout == TENSOR_LAYOUT_HND) {  // [B, H, S, D] -> (D, 1, S*D, H*S*D)
    int q_sh = seq_len_q * head_dim;
    int k_sh = seq_len_kv * head_dim;
    q_stride_s = head_dim;        q_stride_d = 1;    q_stride_h = q_sh;    q_stride_b = num_heads_q * q_sh;
    k_stride_s = head_dim;        k_stride_d = 1;    k_stride_h = k_sh;    k_stride_b = num_heads_kv * k_sh;
    v_stride_d = 1;               v_stride_s = head_dim;  v_stride_h = k_sh;    v_stride_b = num_heads_kv * k_sh;
    o_stride_s = head_dim;        o_stride_d = 1;    o_stride_h = q_sh;    o_stride_b = num_heads_q * q_sh;
  } else {  // NHD: [B, S, H, D] -> (H*D, 1, D, S*H*D)
    int q_hd = num_heads_q * head_dim;
    int k_hd = num_heads_kv * head_dim;
    q_stride_s = q_hd;            q_stride_d = 1;    q_stride_h = head_dim;  q_stride_b = seq_len_q * q_hd;
    k_stride_s = k_hd;            k_stride_d = 1;    k_stride_h = head_dim;  k_stride_b = seq_len_kv * k_hd;
    v_stride_d = 1;               v_stride_s = k_hd; v_stride_h = head_dim;  v_stride_b = seq_len_kv * k_hd;
    o_stride_s = q_hd;            o_stride_d = 1;    o_stride_h = head_dim;  o_stride_b = seq_len_q * q_hd;
  }
  ark::sdpa_impl((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (BTLA_DTYPE)(q_dtype),
                 q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b,
                 v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b,
                 batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal,
                 (float*)lse);
}

static void sdpa_varlen(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                        int q_dtype, int k_dtype, int o_dtype,
                        int batch, int num_heads_q, int num_heads_kv,
                        int total_seqlen_q, int total_seqlen_kv,
                        int max_seqlen_q, int max_seqlen_kv,
                        int head_dim, float softmax_scale, bool is_causal,
                        torch_ptr cu_seqlens_q, torch_ptr cu_seqlens_k,
                        int tensor_layout, torch_ptr lse = 0) {
  if (k_dtype != q_dtype || o_dtype != q_dtype) {
    throw std::invalid_argument("ark::sdpa_varlen: k_dtype and o_dtype must match q_dtype");
  }
  if (q_dtype != (int)BTLA_DTYPE::F16 && q_dtype != (int)BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::sdpa_varlen: only FP16 and BF16 are supported");
  }
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sdpa_varlen: mask and is_causal cannot both be set");
  }
  if (tensor_layout != TENSOR_LAYOUT_HND && tensor_layout != TENSOR_LAYOUT_NHD) {
    throw std::invalid_argument("ark::sdpa_varlen: tensor_layout must be TENSOR_LAYOUT_HND or TENSOR_LAYOUT_NHD");
  }

  // Strides for flat 3-D layout [total, num_heads, head_dim].
  // Shape order expected by the kernel: (seq, head-dim, num_heads, batch=1).
  //   For a contiguous tensor [total, H, D]:
  //     stride(seq) = H*D,  stride(dim) = 1,  stride(head) = D,
  //     stride(batch) = total * H*D
  //   V uses transposed order: (dim, seq, head, batch).
  int hd = num_heads_q * head_dim;
  int k_hd = num_heads_kv * head_dim;
  int q_stride_s = hd,            q_stride_d = 1,          q_stride_h = head_dim, q_stride_b = hd * total_seqlen_q;
  int k_stride_s = k_hd,          k_stride_d = 1,          k_stride_h = head_dim, k_stride_b = k_hd * total_seqlen_kv;
  int v_stride_d = 1,             v_stride_s = k_hd,       v_stride_h = head_dim, v_stride_b = k_hd * total_seqlen_kv;
  int o_stride_s = hd,            o_stride_d = 1,          o_stride_h = head_dim, o_stride_b = hd * total_seqlen_q;

  ark::sdpa_varlen_impl(
      (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (BTLA_DTYPE)(q_dtype),
      q_stride_s, q_stride_d, q_stride_h, q_stride_b,
      k_stride_s, k_stride_d, k_stride_h, k_stride_b,
      v_stride_d, v_stride_s, v_stride_h, v_stride_b,
      o_stride_s, o_stride_d, o_stride_h, o_stride_b,
      batch, num_heads_q, num_heads_kv,
      total_seqlen_q, total_seqlen_kv,
      max_seqlen_q, max_seqlen_kv,
      head_dim, softmax_scale, is_causal,
      (const int*)cu_seqlens_q, (const int*)cu_seqlens_k,
      (float*)lse);
}

// Varlen SageV1 bridge: quantizes Q/K to INT8, then dispatches with varlen=true.
static void sagev1_varlen(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                          int scale_block_size, int q_dtype, int k_dtype, int v_dtype, int o_dtype,
                          int batch, int num_heads_q, int num_heads_kv,
                          int total_seqlen_q, int total_seqlen_kv, int max_seqlen_q, int max_seqlen_kv,
                          int head_dim, float softmax_scale, bool is_causal,
                          torch_ptr cu_seqlens_q, torch_ptr cu_seqlens_k,
                          int use_int8_pv, bool use_mean_bias, torch_ptr lse = 0) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sagev1_varlen: mask and is_causal cannot both be set");
  }
  if (q_dtype != (int)BTLA_DTYPE::F16 && q_dtype != (int)BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::sagev1_varlen: only FP16 and BF16 are supported for Q");
  }
  if (batch <= 0 || total_seqlen_q <= 0 || total_seqlen_kv <= 0) {
    throw std::invalid_argument("ark::sagev1_varlen: batch, total_seqlen_q, total_seqlen_kv must be > 0");
  }
  if (!cu_seqlens_q || !cu_seqlens_k) {
    throw std::invalid_argument("ark::sagev1_varlen: cu_seqlens_q and cu_seqlens_k must not be null");
  }

  // Flat 3-D [total, H, D] strides, kernel order: (seq, dim, head, batch).
  int q_hd = num_heads_q * head_dim;
  int k_hd = num_heads_kv * head_dim;
  int q_stride_s = q_hd,  q_stride_d = 1,  q_stride_h = head_dim,  q_stride_b = q_hd * total_seqlen_q;
  int k_stride_s = k_hd,  k_stride_d = 1,  k_stride_h = head_dim,  k_stride_b = k_hd * total_seqlen_kv;
  int v_stride_d = 1,     v_stride_s = k_hd,  v_stride_h = head_dim,  v_stride_b = k_hd * total_seqlen_kv;
  int o_stride_s = q_hd,  o_stride_d = 1,  o_stride_h = head_dim,  o_stride_b = q_hd * total_seqlen_q;

  BTLA_DTYPE dtype = static_cast<BTLA_DTYPE>(q_dtype);
  if (dtype == BTLA_DTYPE::BF16) {
    XpuWrapper::sagev1_varlen_impl<sycl::ext::oneapi::bfloat16>(
        (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask,
        scale_block_size, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
        k_stride_s, k_stride_d, k_stride_h, k_stride_b,
        v_stride_d, v_stride_s, v_stride_h, v_stride_b,
        o_stride_s, o_stride_d, o_stride_h, o_stride_b,
        batch, num_heads_q, num_heads_kv,
        total_seqlen_q, total_seqlen_kv, max_seqlen_q, max_seqlen_kv,
        head_dim, softmax_scale, is_causal, bool(use_int8_pv),
        bool(use_mean_bias),
        (const int*)cu_seqlens_q, (const int*)cu_seqlens_k,
        (float*)lse);
  } else {
    XpuWrapper::sagev1_varlen_impl<sycl::half>(
        (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask,
        scale_block_size, q_stride_s, q_stride_d, q_stride_h, q_stride_b,
        k_stride_s, k_stride_d, k_stride_h, k_stride_b,
        v_stride_d, v_stride_s, v_stride_h, v_stride_b,
        o_stride_s, o_stride_d, o_stride_h, o_stride_b,
        batch, num_heads_q, num_heads_kv,
        total_seqlen_q, total_seqlen_kv, max_seqlen_q, max_seqlen_kv,
        head_dim, softmax_scale, is_causal, bool(use_int8_pv),
        bool(use_mean_bias),
        (const int*)cu_seqlens_q, (const int*)cu_seqlens_k,
        (float*)lse);
  }
}

static void sagev1_impl(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                        int scale_block_size, int q_dtype, int k_dtype, int v_dtype, int o_dtype,
                        int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                        float softmax_scale, bool is_causal, bool use_int8_pv,
                        int tensor_layout, bool use_mean_bias, torch_ptr lse = 0) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sagev1: mask and is_causal cannot both be set");
  }
  if (q_dtype != k_dtype || o_dtype != q_dtype || v_dtype != q_dtype) {
    throw std::invalid_argument("ark::sagev1: k_dtype and o_dtype must match q_dtype");
  }
  if (q_dtype != (int)BTLA_DTYPE::F16 && q_dtype != (int)BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::sagev1: only F16 and BF16 are supported for q_dtype");
  }
  int q_stride_s, q_stride_d, q_stride_h, q_stride_b;
  int k_stride_s, k_stride_d, k_stride_h, k_stride_b;
  int v_stride_d, v_stride_s, v_stride_h, v_stride_b;
  int o_stride_s, o_stride_d, o_stride_h, o_stride_b;
  if (tensor_layout == TENSOR_LAYOUT_HND) {
    int q_sh = seq_len_q * head_dim;
    int k_sh = seq_len_kv * head_dim;
    q_stride_s = head_dim;        q_stride_d = 1;    q_stride_h = q_sh;    q_stride_b = num_heads_q * q_sh;
    k_stride_s = head_dim;        k_stride_d = 1;    k_stride_h = k_sh;    k_stride_b = num_heads_kv * k_sh;
    v_stride_d = 1;               v_stride_s = head_dim;  v_stride_h = k_sh;    v_stride_b = num_heads_kv * k_sh;
    o_stride_s = head_dim;        o_stride_d = 1;    o_stride_h = q_sh;    o_stride_b = num_heads_q * q_sh;
  } else {  // NHD
    int q_hd = num_heads_q * head_dim;
    int k_hd = num_heads_kv * head_dim;
    q_stride_s = q_hd;            q_stride_d = 1;    q_stride_h = head_dim;  q_stride_b = seq_len_q * q_hd;
    k_stride_s = k_hd;            k_stride_d = 1;    k_stride_h = head_dim;  k_stride_b = seq_len_kv * k_hd;
    v_stride_d = 1;               v_stride_s = k_hd; v_stride_h = head_dim;  v_stride_b = seq_len_kv * k_hd;
    o_stride_s = q_hd;            o_stride_d = 1;    o_stride_h = head_dim;  o_stride_b = seq_len_q * q_hd;
  }
#ifdef ARK_XPU
  if (use_int8_pv) {
    XpuWrapper::sagev1_pvi8((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask,
                            scale_block_size, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s,
                            k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b,
                            o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                            seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, (BTLA_DTYPE)q_dtype,
                            (float*)lse);
  } else {
    XpuWrapper::sagev1((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, scale_block_size,
                       q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                       k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d,
                       o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                       softmax_scale, is_causal, (BTLA_DTYPE)q_dtype, (float*)lse, use_mean_bias);
  }
#else
  throw std::runtime_error("ark::sagev1 is only supported on XPU");
#endif
}

static void sagev1(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                   int scale_block_size,
                   int q_dtype, int k_dtype, int v_dtype, int o_dtype, int batch, int num_heads_q, int num_heads_kv,
                   int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
                   int tensor_layout, bool use_mean_bias, torch_ptr lse = 0) {
  sagev1_impl(stream, Q, K, V, O, mask, scale_block_size, q_dtype, k_dtype, v_dtype, o_dtype, batch,
              num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, false,
              tensor_layout, use_mean_bias, lse);
}

static void sagev1_pvi8(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                        int scale_block_size,
                        int q_dtype, int k_dtype, int v_dtype, int o_dtype, int batch, int num_heads_q, int num_heads_kv,
                        int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
                        int tensor_layout, bool use_mean_bias, torch_ptr lse = 0) {
  sagev1_impl(stream, Q, K, V, O, mask, scale_block_size, q_dtype, k_dtype, v_dtype, o_dtype, batch,
              num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, true,
              tensor_layout, use_mean_bias, lse);
}

static void sage(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                 int scale_block_size, torch_ptr qscale, torch_ptr kscale,
                 int q_dtype, int k_dtype, int o_dtype, int batch, int num_heads_q,
                 int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
                 bool is_causal, int tensor_layout, torch_ptr lse = 0) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sage: mask and is_causal cannot both be set");
  }
  int q_stride_s, q_stride_d, q_stride_h, q_stride_b;
  int k_stride_s, k_stride_d, k_stride_h, k_stride_b;
  int v_stride_d, v_stride_s, v_stride_h, v_stride_b;
  int o_stride_s, o_stride_d, o_stride_h, o_stride_b;
  if (tensor_layout == TENSOR_LAYOUT_HND) {
    int q_sh = seq_len_q * head_dim;
    int k_sh = seq_len_kv * head_dim;
    q_stride_s = head_dim;        q_stride_d = 1;    q_stride_h = q_sh;    q_stride_b = num_heads_q * q_sh;
    k_stride_s = head_dim;        k_stride_d = 1;    k_stride_h = k_sh;    k_stride_b = num_heads_kv * k_sh;
    v_stride_d = 1;               v_stride_s = head_dim;  v_stride_h = k_sh;    v_stride_b = num_heads_kv * k_sh;
    o_stride_s = head_dim;        o_stride_d = 1;    o_stride_h = q_sh;    o_stride_b = num_heads_q * q_sh;
  } else {  // NHD
    int q_hd = num_heads_q * head_dim;
    int k_hd = num_heads_kv * head_dim;
    q_stride_s = q_hd;            q_stride_d = 1;    q_stride_h = head_dim;  q_stride_b = seq_len_q * q_hd;
    k_stride_s = k_hd;            k_stride_d = 1;    k_stride_h = head_dim;  k_stride_b = seq_len_kv * k_hd;
    v_stride_d = 1;               v_stride_s = k_hd; v_stride_h = head_dim;  v_stride_b = seq_len_kv * k_hd;
    o_stride_s = q_hd;            o_stride_d = 1;    o_stride_h = head_dim;  o_stride_b = seq_len_q * q_hd;
  }
  ark::sdpa_impl_qks8_pvhalf((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask,
                             scale_block_size, (void*)qscale, (void*)kscale, q_stride_s, q_stride_d, q_stride_h,
                             q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
                             v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch,
                             num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal,
                             (BTLA_DTYPE)o_dtype, (float*)lse);
}

static void sage_pvi8(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                      int scale_block_size, torch_ptr qscale, torch_ptr kscale, torch_ptr vscale,
                      int q_dtype, int k_dtype, int o_dtype, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                      int seq_len_kv, int head_dim, float softmax_scale, bool is_causal,
                 int tensor_layout, torch_ptr lse = 0) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sage_pvi8: mask and is_causal cannot both be set");
  }
  int q_stride_s, q_stride_d, q_stride_h, q_stride_b;
  int k_stride_s, k_stride_d, k_stride_h, k_stride_b;
  int v_stride_d, v_stride_s, v_stride_h, v_stride_b;
  int o_stride_s, o_stride_d, o_stride_h, o_stride_b;
  if (tensor_layout == TENSOR_LAYOUT_HND) {
    int q_sh = seq_len_q * head_dim;
    int k_sh = seq_len_kv * head_dim;
    q_stride_s = head_dim;        q_stride_d = 1;    q_stride_h = q_sh;    q_stride_b = num_heads_q * q_sh;
    k_stride_s = head_dim;        k_stride_d = 1;    k_stride_h = k_sh;    k_stride_b = num_heads_kv * k_sh;
    v_stride_d = 1;               v_stride_s = head_dim;  v_stride_h = k_sh;    v_stride_b = num_heads_kv * k_sh;
    o_stride_s = head_dim;        o_stride_d = 1;    o_stride_h = q_sh;    o_stride_b = num_heads_q * q_sh;
  } else {  // NHD
    int q_hd = num_heads_q * head_dim;
    int k_hd = num_heads_kv * head_dim;
    q_stride_s = q_hd;            q_stride_d = 1;    q_stride_h = head_dim;  q_stride_b = seq_len_q * q_hd;
    k_stride_s = k_hd;            k_stride_d = 1;    k_stride_h = head_dim;  k_stride_b = seq_len_kv * k_hd;
    v_stride_d = 1;               v_stride_s = k_hd; v_stride_h = head_dim;  v_stride_b = seq_len_kv * k_hd;
    o_stride_s = q_hd;            o_stride_d = 1;    o_stride_h = head_dim;  o_stride_b = seq_len_q * q_hd;
  }
  ark::sdpa_impl_qks8_pvi8((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask,
                           scale_block_size, (void*)qscale, (void*)kscale, (void*)vscale, q_stride_s, q_stride_d,
                           q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d,
                           v_stride_s, v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b,
                           batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale,
                           is_causal, (BTLA_DTYPE)o_dtype, (float*)lse);
}

static void sage_sparse(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                        int scale_block_size, torch_ptr qscale, torch_ptr kscale, torch_ptr lut,
                        torch_ptr valid_block_num, int num_q_blocks, int num_k_blocks, int q_tile_override, int q_stride_s, int q_stride_d,
                        int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b,
                        int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s,
                        int o_stride_d, int o_stride_h, int o_stride_b, int q_dtype, int k_dtype, int o_dtype,
                        int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                        float softmax_scale, bool is_causal) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sage_sparse: mask and is_causal cannot both be set");
  }
  if (!lut || !valid_block_num) {
    throw std::invalid_argument("ark::sage_sparse: lut and valid_block_num must be provided");
  }
  auto matches_block_size = [](int seq_len, int num_blocks, int block_size) {
    return block_size > 0 && num_blocks == ((seq_len + block_size - 1) / block_size);
  };
  const bool key_block_is_64 = matches_block_size(seq_len_kv, num_k_blocks, 64);
  if (!key_block_is_64) {
    throw std::invalid_argument("ark::sage_sparse: only key block size 64 is supported");
  }
  if (head_dim == 64) {
    if (!matches_block_size(seq_len_q, num_q_blocks, 64)) {
      throw std::invalid_argument("ark::sage_sparse: head_dim=64 requires query block size 64");
    }
    ark::sdpa_impl_qks8_sparse_d64_pvhalf(
        (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, scale_block_size, (void*)qscale,
        (void*)kscale, (void*)lut, (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
        q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
        v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
        seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, (BTLA_DTYPE)o_dtype);
    return;
  }
  if (head_dim == 128) {
    if (matches_block_size(seq_len_q, num_q_blocks, 256)) {
      ark::sdpa_impl_qks8_sparse_qtile256_row64k_pvhalf(
          (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, scale_block_size, (void*)qscale,
          (void*)kscale, (void*)lut, (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
          q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
          v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, (BTLA_DTYPE)o_dtype);
      return;
    }
    if (matches_block_size(seq_len_q, num_q_blocks, 64)) {
      ark::sdpa_impl_qks8_sparse_row_linear_pvhalf(
          (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, scale_block_size, (void*)qscale,
          (void*)kscale, (void*)lut, (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s,
          q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s,
          v_stride_h, v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
          seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal, (BTLA_DTYPE)o_dtype);
      return;
    }
    throw std::invalid_argument("ark::sage_sparse: head_dim=128 supports query block sizes 64 and 256 only");
  }
  throw std::invalid_argument("ark::sage_sparse: unsupported head_dim; supported values are 64 and 128");
}

static void moe_gemm_wrapper(torch_ptr stream, torch_ptr activations, torch_ptr weights, torch_ptr scales,
                             torch_ptr outputs, int dtype, int N, int K, torch_ptr num_tokens_per_expert,
                             int num_experts) {
  ark::moe_gemm((sycl::queue*)stream, (void*)activations, (void*)weights, scales ? (void*)scales : nullptr,
                (void*)outputs, (BTLA_DTYPE)(dtype), N, K, (int*)num_tokens_per_expert, num_experts);
}

static void moe_gemm_decode_wrapper(torch_ptr stream, torch_ptr activations, torch_ptr weights, torch_ptr scales,
                                    torch_ptr zeros, torch_ptr outputs, torch_ptr expert_id_per_token_buf,
                                    int act_dtype, int weight_dtype, int N, int K, int group_size,
                                    torch_ptr num_tokens_per_expert, int num_experts, int total_tokens, bool asym) {
  ark::moe_gemm_decode((sycl::queue*)stream, (void*)activations, (void*)weights, scales ? (void*)scales : nullptr,
                       zeros ? (void*)zeros : nullptr, (void*)outputs, (int*)expert_id_per_token_buf,
                       (BTLA_DTYPE)(act_dtype), (BTLA_DTYPE)(weight_dtype), N, K, group_size,
                       (int*)num_tokens_per_expert, num_experts, total_tokens, asym);
}

static void moe_gemm_prefill_wrapper(torch_ptr stream, torch_ptr activations, torch_ptr weights, torch_ptr scales,
                                     torch_ptr zeros, torch_ptr outputs, torch_ptr dequant_workspace, int act_dtype,
                                     int weight_dtype, int N, int K, int group_size, torch_ptr num_tokens_per_expert,
                                     int num_experts, int total_tokens, bool asym) {
  ark::moe_gemm_prefill((sycl::queue*)stream, (void*)activations, (void*)weights, scales ? (void*)scales : nullptr,
                        zeros ? (void*)zeros : nullptr, (void*)outputs,
                        dequant_workspace ? (void*)dequant_workspace : nullptr, (BTLA_DTYPE)(act_dtype),
                        (BTLA_DTYPE)(weight_dtype), N, K, group_size, (int*)num_tokens_per_expert, num_experts,
                        total_tokens, asym);
}

// Variant A: FP8 per-tensor DPAS grouped GEMM (mirrors vllm-xpu-kernels'
// `cutlass_grouped_gemm_xe2_impl` FP8 branch). `scales` is [E] FP32.
// Weights are [E, K, N] row-major uint8. STATUS: NEEDS-HARDWARE-VALIDATION.
static void moe_gemm_prefill_fp8_dpas_wrapper(torch_ptr stream, torch_ptr activations, torch_ptr weights,
                                              torch_ptr scales, torch_ptr outputs, int act_dtype, int weight_dtype,
                                              int N, int K, torch_ptr num_tokens_per_expert, int num_experts,
                                              int total_tokens) {
  ark::moe_gemm_prefill_fp8_dpas((sycl::queue*)stream, (void*)activations, (void*)weights, (void*)scales,
                                 (void*)outputs, (BTLA_DTYPE)(act_dtype), (BTLA_DTYPE)(weight_dtype), N, K,
                                 (int*)num_tokens_per_expert, num_experts, total_tokens);
}

// INT8 sibling of `moe_gemm_prefill_fp8_dpas`: `scales` is [E] FP32,
// weights are [E, K, N] row-major int8. Storage-only INT8 (DPAS still
// runs on activation dtype after in-register upcast). STATUS:
// NEEDS-HARDWARE-VALIDATION.
static void moe_gemm_prefill_int_dpas_wrapper(torch_ptr stream, torch_ptr activations, torch_ptr weights,
                                              torch_ptr scales, torch_ptr outputs, int act_dtype, int weight_dtype,
                                              int N, int K, torch_ptr num_tokens_per_expert, int num_experts,
                                              int total_tokens) {
  ark::moe_gemm_prefill_int_dpas((sycl::queue*)stream, (void*)activations, (void*)weights, (void*)scales,
                                 (void*)outputs, (BTLA_DTYPE)(act_dtype), (BTLA_DTYPE)(weight_dtype), N, K,
                                 (int*)num_tokens_per_expert, num_experts, total_tokens);
}

static void sage_dynamic_quant(torch_ptr stream, torch_ptr input, torch_ptr bias, torch_ptr output, torch_ptr scale_out,
                               int num_rows, int head_dim, int block_size) {
  auto* q = (sycl::queue*)stream;
  auto* in_ptr = (sycl::half*)input;
  auto* bias_ptr = bias ? (sycl::half*)bias : nullptr;
  auto* out_ptr = (int8_t*)output;
  auto* scale_ptr = (float*)scale_out;

  int num_blocks = num_rows / block_size;
  int elems_per_block = block_size * head_dim;

  // Work-group size: use 256 threads (16 sub-groups × 16 lanes).
  // For block_size=1, head_dim=128: 128 elements / 256 threads < 1, so we use smaller WG.
  // For block_size=64, head_dim=128: 8192 elements / 256 threads = 32 elements per thread.
  constexpr int SG_SIZE = 32;
  constexpr int MAX_Reg = 64;
  constexpr int MAX_WG_SIZE = 512;
  constexpr int Unroll = 8;
  if (elems_per_block > MAX_Reg * MAX_WG_SIZE) {
    int wg_size = (elems_per_block <= 256) ? SG_SIZE : 256;
    // Ensure wg_size is a multiple of SG_SIZE
    wg_size = ((wg_size + SG_SIZE - 1) / SG_SIZE) * SG_SIZE;

    q->parallel_for(sycl::nd_range<1>(num_blocks * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                      int block_id = item.get_group(0);
                      int tid = item.get_local_id(0);
                      auto wg = item.get_group();
                      auto* block_in = in_ptr + (size_t)block_id * elems_per_block;
                      auto* block_bias = bias_ptr ? bias_ptr + (size_t)block_id * elems_per_block : nullptr;
                      auto* block_out = out_ptr + (size_t)block_id * elems_per_block;

                      // Phase 1: compute absmax across entire block
                      float local_max = 0.0f;
                      sycl::vec<sycl::half, Unroll> local_data, local_max_vec;
                      local_max_vec = sycl::vec<sycl::half, Unroll>(0.0f);
                      for (int i = tid * Unroll; i < elems_per_block; i += wg_size * Unroll) {
                        local_data = *(sycl::vec<sycl::half, Unroll>*)(&block_in[i]);
                        if (block_bias) {
                          local_data = local_data - *(sycl::vec<sycl::half, Unroll>*)(&block_bias[i]);
                        }
                        local_max_vec = sycl::fmax(local_max_vec, sycl::fabs(local_data));
                      }
                      for (int i = 0; i < Unroll; ++i) {
                        local_max = sycl::fmax(local_max, static_cast<float>(local_max_vec[i]));
                      }
                      float absmax = sycl::reduce_over_group(wg, local_max, sycl::maximum<float>{});

                      // Compute scale
                      float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

                      // Store scale (one thread writes)
                      if (tid == 0) {
                        scale_ptr[block_id] = absmax / 127.0f;
                      }

                      // Phase 2: fused quantize
                      for (int i = tid * Unroll; i < elems_per_block; i += wg_size * Unroll) {
#pragma unroll
                        for (int j = 0; j < Unroll; ++j) {
                          float val = static_cast<float>(block_in[i + j]);
                          if (block_bias) {
                            val -= static_cast<float>(block_bias[i + j]);
                          }
                          val *= inv_scale;
                          int iv = static_cast<int>(val + (val >= 0.0f ? 0.5f : -0.5f));
                          iv = sycl::clamp(iv, -127, 127);
                          block_out[i + j] = static_cast<int8_t>(iv);
                        }
                      }
                    });
  } else {
    int wg_size = MAX_WG_SIZE;
    q->parallel_for(sycl::nd_range<1>(num_blocks * wg_size, wg_size),
                    [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                      int block_id = item.get_group(0);
                      int tid = item.get_local_id(0);
                      auto wg = item.get_group();
                      auto* block_in = in_ptr + (size_t)block_id * elems_per_block;
                      auto* block_bias = bias_ptr ? bias_ptr + (size_t)block_id * elems_per_block : nullptr;
                      auto* block_out = out_ptr + (size_t)block_id * elems_per_block;

                      // Phase 1: compute absmax across entire block
                      float local_max = 0.0f;
                      sycl::vec<sycl::half, Unroll> local_data[MAX_Reg / Unroll], local_max_vec;
                      local_max_vec = sycl::vec<sycl::half, Unroll>(0.0f);
                      int local_i = 0;
                      for (int i = tid * Unroll; i < elems_per_block; i += wg_size * Unroll, local_i++) {
                        local_data[local_i] = *(sycl::vec<sycl::half, Unroll>*)&block_in[i];
                        if (block_bias) {
                          local_data[local_i] = local_data[local_i] - *(sycl::vec<sycl::half, Unroll>*)(&block_bias[i]);
                        }
                        local_max_vec = sycl::fmax(local_max_vec, sycl::fabs(local_data[local_i]));
                      }
#pragma unroll
                      for (int i = 0; i < Unroll; ++i) {
                        local_max = sycl::fmax(local_max, static_cast<float>(local_max_vec[i]));
                      }
                      float absmax = sycl::reduce_over_group(wg, local_max, sycl::maximum<float>{});

                      // Compute scale
                      float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

                      // Store scale (one thread writes)
                      if (tid == 0) {
                        scale_ptr[block_id] = absmax / 127.0f;
                      }

                      // Phase 2: fused quantize
                      local_i = 0;
                      for (int i = tid * Unroll; i < elems_per_block; i += wg_size * Unroll, local_i++) {
                        sycl::vec<float, Unroll> val =
                            local_data[local_i].template convert<float, sycl::rounding_mode::automatic>();
                        val = val * inv_scale;
                        val = sycl::round(val);
                        val = sycl::clamp(val, -127, 127);
                        sycl::vec<int8_t, Unroll> qv = val.template convert<int8_t, sycl::rounding_mode::automatic>();
                        *(sycl::vec<int8_t, Unroll>*)(&block_out[i]) = qv;
                      }
                    });
  }
}

static void sage_compute_seq_mean_bias_layout(torch_ptr stream, torch_ptr input, torch_ptr output, int batch,
                                              int num_heads, int seq, int head_dim, int stride_seq, int stride_dim,
                                              int stride_head, int stride_batch) {
  auto* q = (sycl::queue*)stream;
  auto* in_ptr = (sycl::half*)input;
  auto* out_ptr = (sycl::half*)output;
  if (stride_dim != 1) {
    throw std::invalid_argument("ark::sage_compute_seq_mean_bias_layout: head-dim stride must be 1");
  }
  if (ark::XpuWrapper::is_packed_hnd(stride_seq, stride_dim, stride_head, stride_batch, num_heads, seq, head_dim)) {
    ark::XpuWrapper::compute_seq_mean_bias<sycl::half>(q, in_ptr, out_ptr, batch * num_heads, seq, head_dim);
  } else {
    ark::XpuWrapper::compute_seq_mean_bias_strided<sycl::half>(q, in_ptr, out_ptr, batch, num_heads, seq, head_dim,
                                                               stride_seq, stride_dim, stride_head, stride_batch);
  }
}

static void sage_dynamic_quant_layout(torch_ptr stream, torch_ptr input, torch_ptr bias, torch_ptr output,
                                      torch_ptr scale_out, int batch, int num_heads, int seq, int head_dim,
                                      int block_size, int stride_seq, int stride_dim, int stride_head,
                                      int stride_batch) {
  auto* q = (sycl::queue*)stream;
  auto* in_ptr = (sycl::half*)input;
  auto* bias_ptr = bias ? (sycl::half*)bias : nullptr;
  auto* out_ptr = (int8_t*)output;
  auto* scale_ptr = (float*)scale_out;
  if (block_size <= 0) {
    throw std::invalid_argument("ark::sage_dynamic_quant_layout: block_size must be > 0");
  }
  if (stride_dim != 1) {
    throw std::invalid_argument("ark::sage_dynamic_quant_layout: head-dim stride must be 1");
  }
  int n_seq_blk = (seq + block_size - 1) / block_size;
  bool force_strided = ark::env_params::Instance()->sage_disable_packed_hnd_fast != 0;
  if (!force_strided &&
      ark::XpuWrapper::is_packed_hnd(stride_seq, stride_dim, stride_head, stride_batch, num_heads, seq, head_dim)) {
    ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, in_ptr, out_ptr, scale_ptr, batch * num_heads, seq, n_seq_blk,
                                                    head_dim, block_size, bias_ptr);
  } else {
    ark::XpuWrapper::sage_dynamic_quant_strided<sycl::half>(q, in_ptr, out_ptr, scale_ptr, batch, num_heads, seq,
                                                            n_seq_blk, head_dim, block_size, stride_seq, stride_dim,
                                                            stride_head, stride_batch, bias_ptr);
  }
}

static void sage_dynamic_quant_v_layout(torch_ptr stream, torch_ptr input, torch_ptr output, torch_ptr scale_out,
                                        int batch, int num_heads, int seq, int head_dim, int block_size,
                                        int stride_dim, int stride_seq, int stride_head, int stride_batch) {
  auto* q = (sycl::queue*)stream;
  auto* in_ptr = (sycl::half*)input;
  auto* out_ptr = (int8_t*)output;
  auto* scale_ptr = (float*)scale_out;
  if (block_size <= 0) {
    throw std::invalid_argument("ark::sage_dynamic_quant_v_layout: block_size must be > 0");
  }
  if (stride_dim != 1) {
    throw std::invalid_argument("ark::sage_dynamic_quant_v_layout: head-dim stride must be 1");
  }
  int n_seq_blk = (seq + block_size - 1) / block_size;
  bool force_strided = ark::env_params::Instance()->sage_disable_packed_hnd_fast != 0;
  if (!force_strided &&
      ark::XpuWrapper::is_packed_hnd(stride_seq, stride_dim, stride_head, stride_batch, num_heads, seq, head_dim)) {
    ark::XpuWrapper::sage_dynamic_quant_v<sycl::half>(q, in_ptr, out_ptr, scale_ptr, batch * num_heads, seq,
                                                      n_seq_blk, head_dim, block_size);
  } else {
    ark::XpuWrapper::sage_dynamic_quant_v_strided<sycl::half>(q, in_ptr, out_ptr, scale_ptr, batch, num_heads, seq,
                                                              n_seq_blk, head_dim, block_size, stride_dim, stride_seq,
                                                              stride_head, stride_batch);
  }
}

template <typename T>
static void xpu_copy_into_kv_cache_qk(sycl::queue* q, T* cache_ptr, const T* src_ptr, int stride_s, int stride_d,
                                    int stride_h, int stride_b, int batch, int num_heads_kv, int append_len,
                                    int head_dim, int capacity, int start_pos) {
  const size_t total = static_cast<size_t>(batch) * num_heads_kv * append_len * head_dim;
  q->parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    size_t linear = idx[0];
    const int d = linear % head_dim;
    linear /= head_dim;
    const int s = linear % append_len;
    linear /= append_len;
    const int h = linear % num_heads_kv;
    const int b = linear / num_heads_kv;
    const size_t src_offset =
      static_cast<size_t>(b) * stride_b + static_cast<size_t>(h) * stride_h + static_cast<size_t>(s) * stride_s + d;
    const size_t dst_offset =
      ((static_cast<size_t>(b) * num_heads_kv + h) * capacity + (start_pos + s)) * head_dim + d;
    cache_ptr[dst_offset] = src_ptr[src_offset];
  });
  q->wait();
}

template <typename T>
static void xpu_copy_into_kv_cache_v(sycl::queue* q, T* cache_ptr, const T* src_ptr, int stride_d, int stride_s,
                                   int stride_h, int stride_b, int batch, int num_heads_kv, int append_len,
                                   int head_dim, int capacity, int start_pos) {
  const size_t total = static_cast<size_t>(batch) * num_heads_kv * append_len * head_dim;
  q->parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    size_t linear = idx[0];
    const int d = linear % head_dim;
    linear /= head_dim;
    const int s = linear % append_len;
    linear /= append_len;
    const int h = linear % num_heads_kv;
    const int b = linear / num_heads_kv;
    const size_t src_offset =
      static_cast<size_t>(b) * stride_b + static_cast<size_t>(h) * stride_h + static_cast<size_t>(s) * stride_s +
      static_cast<size_t>(d) * stride_d;
    const size_t dst_offset =
      ((static_cast<size_t>(b) * num_heads_kv + h) * capacity + (start_pos + s)) * head_dim + d;
    cache_ptr[dst_offset] = src_ptr[src_offset];
  });
  q->wait();
}

static void ark_xpu_kv_update(torch_ptr stream, torch_ptr KCache, torch_ptr VCache, torch_ptr K, torch_ptr V,
                            int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
                            int v_stride_s, int v_stride_h, int v_stride_b, int kv_dtype, int batch,
                            int num_heads_kv, int append_len, int head_dim, int capacity, int start_pos) {
  auto* q = (sycl::queue*)stream;
  if (append_len <= 0 || start_pos < 0 || start_pos + append_len > capacity) {
    throw std::invalid_argument("ark::ark_xpu_kv_update: invalid append range for cache capacity");
  }
  switch (static_cast<BTLA_DTYPE>(kv_dtype)) {
    case BTLA_DTYPE::F16:
    xpu_copy_into_kv_cache_qk<sycl::half>(q, (sycl::half*)KCache, (const sycl::half*)K, k_stride_s, k_stride_d,
                                          k_stride_h, k_stride_b, batch, num_heads_kv, append_len, head_dim,
                                          capacity, start_pos);
    xpu_copy_into_kv_cache_v<sycl::half>(q, (sycl::half*)VCache, (const sycl::half*)V, v_stride_d, v_stride_s,
                                         v_stride_h, v_stride_b, batch, num_heads_kv, append_len, head_dim,
                                         capacity, start_pos);
    return;
    case BTLA_DTYPE::BF16:
    xpu_copy_into_kv_cache_qk<sycl::ext::oneapi::bfloat16>(
        q, (sycl::ext::oneapi::bfloat16*)KCache, (const sycl::ext::oneapi::bfloat16*)K, k_stride_s, k_stride_d,
        k_stride_h, k_stride_b, batch, num_heads_kv, append_len, head_dim, capacity, start_pos);
    xpu_copy_into_kv_cache_v<sycl::ext::oneapi::bfloat16>(
        q, (sycl::ext::oneapi::bfloat16*)VCache, (const sycl::ext::oneapi::bfloat16*)V, v_stride_d, v_stride_s,
        v_stride_h, v_stride_b, batch, num_heads_kv, append_len, head_dim, capacity, start_pos);
    return;
    default:
    throw std::invalid_argument("ark::ark_xpu_kv_update: only FP16 and BF16 caches are supported");
  }
}

static void sdpa_with_kv_cache(torch_ptr stream, torch_ptr Q, torch_ptr KCache, torch_ptr VCache, torch_ptr O,
                             torch_ptr mask, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                             int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
                             int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
                             int o_stride_h, int o_stride_b, int q_dtype, int batch, int num_heads_q,
                             int num_heads_kv, int seq_len_q, int seq_len_kv, int capacity, int head_dim,
                             float softmax_scale, bool is_causal) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sdpa_with_kv_cache: mask and is_causal cannot both be set");
  }
  if (seq_len_q <= 0 || seq_len_kv <= 0 || seq_len_kv > capacity) {
    throw std::invalid_argument("ark::sdpa_with_kv_cache: invalid query/KV lengths for cache capacity");
  }
  if (q_dtype != (int)BTLA_DTYPE::F16 && q_dtype != (int)BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::sdpa_with_kv_cache: only FP16 and BF16 are supported");
  }
  if (is_causal && seq_len_q != 1) {
    throw std::invalid_argument(
      "ark::sdpa_with_kv_cache: causal cache decode currently supports only seq_len_q == 1");
  }
  ark::flash_attn_prefill((sycl::queue*)stream, (void*)Q, (void*)KCache, (void*)VCache, (void*)O, (void*)mask,
                        (BTLA_DTYPE)(q_dtype), q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s,
                        k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b,
                        o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                        seq_len_kv, head_dim, softmax_scale, is_causal);
}

#elif !defined(ARK_XPU)

// Finalization note (non-int8 closure pass):
// Routes 1/2 (mixed BestLA, Tier 1) now accept the full feature set from the
// Python ABI: `use_alibi`, `use_tanh`, `prefer_fp32`, and `n_padding` are
// forwarded to `bargs.attn_flags` / `bargs.n_padding` in the mixed_bestla
// block below. The scalar Tier-0 path (MhaDenseArgs) only supports causal
// masking; alibi, tanh, and padding-right are BestLA-specific and are rejected
// on that path with a clear error pointing to ARK_UNSAFE_BESTLA_MIXED_SDPA.
// `prefer_fp32` on the Tier-0 path is silently accepted (pure fp32 computations
// are always fp32-compute; the flag is a no-op). Routes 3/4 (homogeneous, Tier 2)
// remain internal-only (not wired here).
static void sdpa(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                 int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                 int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                 int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int q_dtype, int k_dtype, int o_dtype,
                 int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                 float softmax_scale, bool is_causal,
                 bool use_alibi, bool use_tanh, bool prefer_fp32_flag, int n_padding_arg) {
  (void)stream;
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sdpa: mask and is_causal cannot both be set");
  }

  // Mixed-precision BestLA route (Phase 3): F32 Q + (F16|BF16) K/V -> F32 O.
  // K and V share `k_dtype` in this ABI, so a single check covers both.
  // Homogeneous fp16/bf16 and int8 are intentionally NOT routed here yet.
  //
  // Phase 6 exposure-tier note: this block is TIER 1 (experimental/env-gated).
  // Routes 1 (F16, AVX2) and 2 (BF16, AVX512F/AMX-BF16) have a full S feature
  // matrix (causal, GQA, padding-right, alibi, tanh, prefer_fp32) validated at
  // the C++ level. They are not the default path because:
  //   (a) the raw->packed reorder bridge adds per-forward allocation overhead;
  //   (b) persistent packed KV cache is deferred to a future cleanup pass.
  // The Python ABI (use_alibi/use_tanh/prefer_fp32/n_padding) is now wired;
  // see the finalization note above. The homogeneous routes (Tier 2, not wired
  // here) and the scalar fallback (Tier 0) are in sdpa.cpp's Phase 6 block.
  //
  // IMPORTANT (Phase 3 safety gate): the BestLA specializations wired today
  // (`bestla_fusion_attn_forward<float, fp16, ...>` /  `<float, bf16, ...>`)
  // expect NTILE24/NTILE48 row-packed (reordered) K/V, NOT the raw PLAIN
  // (HND/NHD-strided) K/V this entry point receives. This route is DISABLED BY
  // DEFAULT and only reachable via `ARK_UNSAFE_BESTLA_MIXED_SDPA=1`.
  const bool mixed_dtype =
      static_cast<BTLA_DTYPE>(q_dtype) == BTLA_DTYPE::F32 && static_cast<BTLA_DTYPE>(o_dtype) == BTLA_DTYPE::F32 &&
      (static_cast<BTLA_DTYPE>(k_dtype) == BTLA_DTYPE::F16 || static_cast<BTLA_DTYPE>(k_dtype) == BTLA_DTYPE::BF16);
  const char* const unsafe_mixed_env = std::getenv("ARK_UNSAFE_BESTLA_MIXED_SDPA");
  const bool mixed_bestla =
      mixed_dtype && unsafe_mixed_env != nullptr && std::strcmp(unsafe_mixed_env, "0") != 0;
  if (mixed_bestla) {
    if (mask) {
      throw std::invalid_argument("ark::sdpa: attn_mask is not supported on the BestLA mixed-precision path yet");
    }
    if (n_padding_arg > 0 && is_causal) {
      throw std::invalid_argument(
          "ark::sdpa: n_padding and is_causal are mutually exclusive on the BestLA mixed-precision path");
    }
    ark::cpu::attn_fwd_args_t bargs;
    bargs.Q = (void*)Q;
    bargs.K = (void*)K;
    bargs.V = (void*)V;
    bargs.dst = (void*)O;
    bargs.QK_scale = softmax_scale;
    // Build attn_flags from the individual Python kwargs (is_causal was the
    // only flag before the Python ABI was extended; now all four are wired).
    bargs.attn_flags = ark::cpu::ATTN_FLAG_NONE;
    if (is_causal) bargs.attn_flags |= ark::cpu::ATTN_FLAG_IS_CAUSAL;
    if (use_alibi) bargs.attn_flags |= ark::cpu::ATTN_FLAG_IS_ALIBI8;
    if (use_tanh) bargs.attn_flags |= ark::cpu::ATTN_FLAG_IS_TANH30;
    if (prefer_fp32_flag) bargs.attn_flags |= ark::cpu::ATTN_FLAG_PREFER_FP32;
    if (n_padding_arg > 0) bargs.attn_flags |= ark::cpu::ATTN_FLAG_PADDING_RIGHT;
    bargs.n_padding = n_padding_arg;
    bargs.batch_size = batch;
    bargs.head_num = num_heads_q;
    bargs.heads_kv = num_heads_kv;
    bargs.head_size = head_dim;
    bargs.sl_q = seq_len_q;
    bargs.sl_kv = seq_len_kv;
    // Strides describe an HND/NHD-friendly PLAIN interface; the wired BestLA
    // mixed kernels require packed/reordered (NTILE24/NTILE48) K/V, so this
    // raw PLAIN path is gated behind ARK_UNSAFE_BESTLA_MIXED_SDPA above.
    bargs.Q_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
    bargs.K_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
    bargs.V_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
    bargs.dst_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
    // Q/dst head-dim stride is assumed contiguous (== 1); batch/head/seq come
    // straight from the incoming stride arguments.
    bargs.step_q_bs = q_stride_b;
    bargs.step_q_head_num = q_stride_h;
    bargs.step_q_sl = q_stride_s;
    bargs.step_k_bs = k_stride_b;
    bargs.step_k_head_num = k_stride_h;
    bargs.step_k_sl = k_stride_s;
    bargs.step_k_head_size = k_stride_d;
    bargs.step_v_bs = v_stride_b;
    bargs.step_v_head_num = v_stride_h;
    bargs.step_v_sl = v_stride_s;
    bargs.step_v_head_size = v_stride_d;
    bargs.step_dst_bs = o_stride_b;
    bargs.step_dst_head_num = o_stride_h;
    bargs.step_dst_sl = o_stride_s;
    bargs.tmp = nullptr;  // scratch allocated inside bestla_sdpa_forward
    // Reuse ARK's shared CPU thread pool rather than a dedicated attention pool.
    bargs.threading = ark::CpuWrapper::get_threading();
    ark::cpu::bestla_sdpa_forward(bargs, static_cast<BTLA_DTYPE>(k_dtype));
    return;
  }

  // Tier 0 scalar fallback: alibi, tanh, and padding-right are BestLA-specific
  // features not implemented in the scalar MhaDenseArgs kernel. Reject them here
  // so callers get a clear message instead of a silently ignored flag. prefer_fp32
  // is accepted as a no-op (the scalar path is always fp32-compute).
  if (use_alibi || use_tanh || n_padding_arg > 0) {
    throw std::invalid_argument(
        "ark::sdpa: use_alibi, use_tanh, and n_padding are only supported on the BestLA mixed-precision path "
        "(Q=float32, K/V=float16|bfloat16). Set ARK_UNSAFE_BESTLA_MIXED_SDPA=1 to enable it.");
  }

  if (k_dtype != q_dtype || o_dtype != q_dtype) {
    throw std::invalid_argument("ark::sdpa: k_dtype and o_dtype must match q_dtype");
  }
  ark::cpu::MhaDenseArgs args;
  args.query = (const void*)Q;
  args.key = (const void*)K;
  args.value = (const void*)V;
  args.output = (void*)O;
  args.attn_mask = mask ? (const float*)mask : nullptr;
  args.q_strides = {q_stride_s, q_stride_d, q_stride_h, q_stride_b};
  args.k_strides = {k_stride_s, k_stride_d, k_stride_h, k_stride_b};
  args.v_strides = {v_stride_d, v_stride_s, v_stride_h, v_stride_b};
  args.o_strides = {o_stride_s, o_stride_d, o_stride_h, o_stride_b};
  args.dtype = (BTLA_DTYPE)q_dtype;
  args.batch = batch;
  args.num_heads_q = num_heads_q;
  args.num_heads_kv = num_heads_kv;
  args.seq_len_q = seq_len_q;
  args.seq_len_kv = seq_len_kv;
  args.head_dim = head_dim;
  args.softmax_scale = softmax_scale;
  args.is_causal = is_causal;
  ark::cpu::sdpa_forward(args);
}

static void ark_cpu_kv_update(torch_ptr KCache, torch_ptr VCache, torch_ptr K, torch_ptr V, int k_stride_s,
                              int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s,
                              int v_stride_h, int v_stride_b, int dtype, int batch, int num_heads_kv, int append_len,
                              int head_dim, int capacity, int start_pos) {
  ark::cpu::kv_cache_update((void*)KCache, (void*)VCache, (const void*)K, (const void*)V,
                            {k_stride_s, k_stride_d, k_stride_h, k_stride_b},
                            {v_stride_d, v_stride_s, v_stride_h, v_stride_b}, (BTLA_DTYPE)dtype, batch, num_heads_kv,
                            append_len, head_dim, capacity, start_pos);
}

// ---------------------------------------------------------------------------
// NS-parity persistent packed KV cache Python ABI (Tier 1 / internal).
//
// These four functions expose the packed-cache path for Python consumers:
//   ark_cpu_packed_kv_elems  — query the element counts for a given cache shape
//   ark_cpu_update_packed_k  — append raw K into the persistent packed K cache
//   ark_cpu_update_packed_v  — append raw V into the persistent packed V cache
//   ark_cpu_bestla_sdpa_packed — forward attention over a packed K/V cache
//
// All four are gated by ARK_UNSAFE_BESTLA_MIXED_SDPA (same gate as routes 1/2).
// kv_dtype must be F16 (15) or BF16 (14) — matching BTLA_DTYPE values.
// ---------------------------------------------------------------------------

// Returns (k_elems, v_elems): element counts for 1D allocation of the packed cache.
static std::pair<int64_t, int64_t> ark_cpu_packed_kv_elems(int batch, int num_heads_kv, int capacity, int head_dim,
                                                            int kv_dtype_int) {
  auto shape = ark::cpu::packed_kv_cache_shape(batch, num_heads_kv, capacity, head_dim,
                                               static_cast<BTLA_DTYPE>(kv_dtype_int));
  // Each packed head occupies k_head_elems / v_head_elems elements; total over all
  // batch×head slots gives the required 1D buffer size (in kv_dtype elements).
  int64_t k_elems = static_cast<int64_t>(shape.k_head_elems) * batch * num_heads_kv;
  int64_t v_elems = static_cast<int64_t>(shape.v_head_elems) * batch * num_heads_kv;
  return {k_elems, v_elems};
}

// Append raw K tokens at [start_pos, start_pos+append_len) into the packed K cache.
static void ark_cpu_update_packed_k(torch_ptr cache_k, torch_ptr key, int k_stride_s, int k_stride_d, int k_stride_h,
                                    int k_stride_b, int kv_dtype_int, int batch, int num_heads_kv, int append_len,
                                    int head_dim, int capacity, int start_pos) {
  auto shape = ark::cpu::packed_kv_cache_shape(batch, num_heads_kv, capacity, head_dim,
                                               static_cast<BTLA_DTYPE>(kv_dtype_int));
  ark::cpu::update_packed_k_cache((void*)cache_k, (const void*)key, shape,
                                  {k_stride_s, k_stride_d, k_stride_h, k_stride_b}, batch, num_heads_kv, append_len,
                                  head_dim, start_pos, static_cast<BTLA_DTYPE>(kv_dtype_int));
}

// Append raw V tokens at [start_pos, start_pos+append_len) into the packed V cache.
static void ark_cpu_update_packed_v(torch_ptr cache_v, torch_ptr value, int v_stride_d, int v_stride_s, int v_stride_h,
                                    int v_stride_b, int kv_dtype_int, int batch, int num_heads_kv, int append_len,
                                    int head_dim, int capacity, int start_pos) {
  auto shape = ark::cpu::packed_kv_cache_shape(batch, num_heads_kv, capacity, head_dim,
                                               static_cast<BTLA_DTYPE>(kv_dtype_int));
  ark::cpu::update_packed_v_cache((void*)cache_v, (const void*)value, shape,
                                  {v_stride_d, v_stride_s, v_stride_h, v_stride_b}, batch, num_heads_kv, append_len,
                                  head_dim, start_pos, static_cast<BTLA_DTYPE>(kv_dtype_int));
}

// Forward attention over a pre-packed K/V cache.  Requires ARK_UNSAFE_BESTLA_MIXED_SDPA=1.
// q_dtype must be F32 (10); kv_dtype must be F16 (15) or BF16 (14).
// sl_kv is the current valid sequence length (must be <= capacity).
static void ark_cpu_bestla_sdpa_packed(torch_ptr Q, torch_ptr K_packed, torch_ptr V_packed, torch_ptr O,
                                       int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                                       int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int q_dtype,
                                       int kv_dtype_int, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                                       int seq_len_kv, int capacity, int head_dim, float softmax_scale, bool is_causal,
                                       bool use_alibi, bool use_tanh, bool prefer_fp32, int n_padding) {
  const char* const unsafe_env = std::getenv("ARK_UNSAFE_BESTLA_MIXED_SDPA");
  if (unsafe_env == nullptr || std::strcmp(unsafe_env, "0") == 0) {
    throw std::runtime_error(
        "ark_cpu_bestla_sdpa_packed: requires ARK_UNSAFE_BESTLA_MIXED_SDPA=1 "
        "(packed BestLA mixed-precision path is experimental)");
  }
  if (static_cast<BTLA_DTYPE>(q_dtype) != BTLA_DTYPE::F32) {
    throw std::invalid_argument("ark_cpu_bestla_sdpa_packed: q_dtype must be F32 (10)");
  }
  auto shape = ark::cpu::packed_kv_cache_shape(batch, num_heads_kv, capacity, head_dim,
                                               static_cast<BTLA_DTYPE>(kv_dtype_int));
  ark::cpu::attn_fwd_args_t bargs;
  bargs.Q = (void*)Q;
  bargs.K = (void*)K_packed;
  bargs.V = (void*)V_packed;
  bargs.dst = (void*)O;
  bargs.QK_scale = softmax_scale;
  bargs.attn_flags = ark::cpu::ATTN_FLAG_NONE;
  if (is_causal) bargs.attn_flags |= ark::cpu::ATTN_FLAG_IS_CAUSAL;
  if (use_alibi) bargs.attn_flags |= ark::cpu::ATTN_FLAG_IS_ALIBI8;
  if (use_tanh) bargs.attn_flags |= ark::cpu::ATTN_FLAG_IS_TANH30;
  if (prefer_fp32) bargs.attn_flags |= ark::cpu::ATTN_FLAG_PREFER_FP32;
  if (n_padding > 0) bargs.attn_flags |= ark::cpu::ATTN_FLAG_PADDING_RIGHT;
  bargs.n_padding = n_padding;
  bargs.batch_size = batch;
  bargs.head_num = num_heads_q;
  bargs.heads_kv = num_heads_kv;
  bargs.head_size = head_dim;
  bargs.sl_q = seq_len_q;
  bargs.sl_kv = seq_len_kv;
  bargs.Q_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  bargs.dst_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  // K/V layouts are set by bestla_sdpa_forward_packed from shape; leave PLAIN here.
  bargs.K_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  bargs.V_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  bargs.step_q_bs = q_stride_b;
  bargs.step_q_head_num = q_stride_h;
  bargs.step_q_sl = q_stride_s;
  bargs.step_dst_bs = o_stride_b;
  bargs.step_dst_head_num = o_stride_h;
  bargs.step_dst_sl = o_stride_s;
  // K/V strides are taken from shape inside bestla_sdpa_forward_packed.
  bargs.step_k_bs = 0;
  bargs.step_k_head_num = 0;
  bargs.step_k_sl = 0;
  bargs.step_k_head_size = 0;
  bargs.step_v_bs = 0;
  bargs.step_v_head_num = 0;
  bargs.step_v_sl = 0;
  bargs.step_v_head_size = 0;
  bargs.tmp = nullptr;
  bargs.threading = ark::CpuWrapper::get_threading();
  ark::cpu::bestla_sdpa_forward_packed(bargs, shape, static_cast<BTLA_DTYPE>(kv_dtype_int));
}

#endif  // ARK_XPU && ARK_SYCL_TLA

}  // namespace ark

PYBIND11_MODULE(PY_NAME, m) {
  m.def("matmul", &ark::matmul);
  m.def("woqgemm_s8", &ark::woqgemm_s8);
  m.def("woqgemm", &ark::woqgemm);
  m.def("packed_weight_size", &ark::packed_weight_size);
  m.def("repack_quantized_weight", &ark::repack_quantized_weight);
  m.def("unpack_weight", &ark::unpack_weight);
#if (defined(ARK_XPU) && defined(ARK_SYCL_TLA)) || !defined(ARK_XPU)
  m.def("sdpa", &ark::sdpa);
#endif
#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
  m.def("sdpa_varlen", &ark::sdpa_varlen, pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"),
        pybind11::arg("V"), pybind11::arg("O"), pybind11::arg("mask"),
        pybind11::arg("q_dtype"), pybind11::arg("k_dtype"), pybind11::arg("o_dtype"),
        pybind11::arg("batch"), pybind11::arg("num_heads_q"), pybind11::arg("num_heads_kv"),
        pybind11::arg("total_seqlen_q"), pybind11::arg("total_seqlen_kv"),
        pybind11::arg("max_seqlen_q"), pybind11::arg("max_seqlen_kv"),
        pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
        pybind11::arg("cu_seqlens_q"), pybind11::arg("cu_seqlens_k"),
        pybind11::arg("tensor_layout"), pybind11::arg("lse") = 0);
  // Varlen SAGEV1: flat 3-D Q/K/V + cu_seqlens (use_int8_pv=0) or pvi8 (use_int8_pv=1).
  m.def("sagev1_varlen", &ark::sagev1_varlen, pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"),
        pybind11::arg("V"), pybind11::arg("O"), pybind11::arg("mask"),
        pybind11::arg("scale_block_size"),
        pybind11::arg("q_dtype"), pybind11::arg("k_dtype"), pybind11::arg("v_dtype"), pybind11::arg("o_dtype"),
        pybind11::arg("batch"), pybind11::arg("num_heads_q"), pybind11::arg("num_heads_kv"),
        pybind11::arg("total_seqlen_q"), pybind11::arg("total_seqlen_kv"),
        pybind11::arg("max_seqlen_q"), pybind11::arg("max_seqlen_kv"),
        pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
        pybind11::arg("cu_seqlens_q"), pybind11::arg("cu_seqlens_k"),
        pybind11::arg("use_int8_pv"), pybind11::arg("use_mean_bias"), pybind11::arg("lse") = 0);
  m.def("sagev1", &ark::sagev1, pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"),
        pybind11::arg("V"), pybind11::arg("O"), pybind11::arg("mask"),
        pybind11::arg("scale_block_size"),
        pybind11::arg("q_dtype"), pybind11::arg("k_dtype"), pybind11::arg("v_dtype"), pybind11::arg("o_dtype"),
        pybind11::arg("batch"), pybind11::arg("num_heads_q"), pybind11::arg("num_heads_kv"),
        pybind11::arg("seq_len_q"), pybind11::arg("seq_len_kv"),
        pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
        pybind11::arg("tensor_layout"), pybind11::arg("use_mean_bias"), pybind11::arg("lse") = 0);
  // High-level SAGEV1 PVi8 API: input Q/K/V are FP16 and quantized internally.
  m.def("sagev1_pvi8", &ark::sagev1_pvi8, pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"),
        pybind11::arg("V"), pybind11::arg("O"), pybind11::arg("mask"),
        pybind11::arg("scale_block_size"),
        pybind11::arg("q_dtype"), pybind11::arg("k_dtype"), pybind11::arg("v_dtype"), pybind11::arg("o_dtype"),
        pybind11::arg("batch"), pybind11::arg("num_heads_q"), pybind11::arg("num_heads_kv"),
        pybind11::arg("seq_len_q"), pybind11::arg("seq_len_kv"),
        pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
        pybind11::arg("tensor_layout"), pybind11::arg("use_mean_bias"), pybind11::arg("lse") = 0);
  m.def("sage", &ark::sage, pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"),
        pybind11::arg("V"), pybind11::arg("O"), pybind11::arg("mask"),
        pybind11::arg("scale_block_size"),
        pybind11::arg("qscale"), pybind11::arg("kscale"),
        pybind11::arg("q_dtype"), pybind11::arg("k_dtype"), pybind11::arg("o_dtype"),
        pybind11::arg("batch"), pybind11::arg("num_heads_q"), pybind11::arg("num_heads_kv"),
        pybind11::arg("seq_len_q"), pybind11::arg("seq_len_kv"),
        pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
        pybind11::arg("tensor_layout"), pybind11::arg("lse") = 0);
  m.def("sage_sparse", &ark::sage_sparse);
  // Low-level SAGE PVi8 API: input Q/K/V are pre-quantized int8 with qscale/kscale/vscale.
  m.def("sage_pvi8", &ark::sage_pvi8, pybind11::arg("stream"), pybind11::arg("Q"), pybind11::arg("K"),
        pybind11::arg("V"), pybind11::arg("O"), pybind11::arg("mask"),
        pybind11::arg("scale_block_size"),
        pybind11::arg("qscale"), pybind11::arg("kscale"), pybind11::arg("vscale"),
        pybind11::arg("q_dtype"), pybind11::arg("k_dtype"), pybind11::arg("o_dtype"),
        pybind11::arg("batch"), pybind11::arg("num_heads_q"), pybind11::arg("num_heads_kv"),
        pybind11::arg("seq_len_q"), pybind11::arg("seq_len_kv"),
        pybind11::arg("head_dim"), pybind11::arg("softmax_scale"), pybind11::arg("is_causal"),
        pybind11::arg("tensor_layout"), pybind11::arg("lse") = 0);
  m.def("sage_dynamic_quant", &ark::sage_dynamic_quant);
  m.def("sage_compute_seq_mean_bias_layout", &ark::sage_compute_seq_mean_bias_layout);
  m.def("sage_dynamic_quant_layout", &ark::sage_dynamic_quant_layout);
  m.def("sage_dynamic_quant_v_layout", &ark::sage_dynamic_quant_v_layout);
  m.def("ark_xpu_kv_update", &ark::ark_xpu_kv_update);
  m.def("sdpa_with_kv_cache", &ark::sdpa_with_kv_cache);
  m.def("moe_gemm", &ark::moe_gemm_wrapper);
  m.def("moe_gemm_decode", &ark::moe_gemm_decode_wrapper);
  m.def("moe_gemm_prefill", &ark::moe_gemm_prefill_wrapper);
  m.def("moe_gemm_prefill_fp8_dpas", &ark::moe_gemm_prefill_fp8_dpas_wrapper);
  m.def("moe_gemm_prefill_int_dpas", &ark::moe_gemm_prefill_int_dpas_wrapper);
  m.def("matmul_sycl_tla", &ark::matmul_sycl_tla);
#endif  // ARK_SYCL_TLA
#elif !defined(ARK_XPU)
  m.def("ark_cpu_kv_update", &ark::ark_cpu_kv_update);
  m.def("ark_cpu_packed_kv_elems", &ark::ark_cpu_packed_kv_elems);
  m.def("ark_cpu_update_packed_k", &ark::ark_cpu_update_packed_k);
  m.def("ark_cpu_update_packed_v", &ark::ark_cpu_update_packed_v);
  m.def("ark_cpu_bestla_sdpa_packed", &ark::ark_cpu_bestla_sdpa_packed);
#endif
}
