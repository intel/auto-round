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
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "bestla/bestla/bestla.h"
typedef uintptr_t torch_ptr;
#if ARK_XPU
#include <sycl/sycl.hpp>
#include "xpu_wrapper.hpp"
#include "sycl_s8_wrapper.hpp"
#if ARK_SYCL_TLA
#include "sycl_tla_common.hpp"
#endif
#else
#include "ark/cpu/sdpa.h"
#include "cpu_wrapper.hpp"
#endif

#if ARK_DNNL
#include "dnnl_wrapper.hpp"
#endif

namespace ark {
namespace py = pybind11;

constexpr int TENSOR_LAYOUT_HND = 0;  // [B, H, S, D]
constexpr int TENSOR_LAYOUT_NHD = 1;  // [B, S, H, D]

static std::vector<uint16_t> transpose_plain_half_k_for_homogeneous_fp16(torch_ptr K, int k_stride_s, int k_stride_d,
                                                                          int k_stride_h, int k_stride_b, int batch,
                                                                          int num_heads_kv, int seq_len_kv,
                                                                          int head_dim) {
  const auto* src = reinterpret_cast<const uint16_t*>(K);
  std::vector<uint16_t> transposed(static_cast<size_t>(batch) * num_heads_kv * seq_len_kv * head_dim);
  for (int ib = 0; ib < batch; ++ib) {
    for (int ih = 0; ih < num_heads_kv; ++ih) {
      const size_t head_base = (static_cast<size_t>(ib) * num_heads_kv + ih) * head_dim * seq_len_kv;
      for (int is = 0; is < seq_len_kv; ++is) {
        const size_t src_row = static_cast<size_t>(ib) * k_stride_b + static_cast<size_t>(ih) * k_stride_h +
                               static_cast<size_t>(is) * k_stride_s;
        for (int id = 0; id < head_dim; ++id) {
          transposed[head_base + static_cast<size_t>(id) * seq_len_kv + is] =
              src[src_row + static_cast<size_t>(id) * k_stride_d];
        }
      }
    }
  }
  return transposed;
}

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

static void block_sparse_sdpa(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                              torch_ptr lut, torch_ptr valid_block_num, int num_q_blocks, int num_k_blocks,
                              int q_tile_override, int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                              int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b, int v_stride_d,
                              int v_stride_s, int v_stride_h, int v_stride_b, int o_stride_s, int o_stride_d,
                              int o_stride_h, int o_stride_b, int q_dtype, int batch, int num_heads_q,
                              int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
                              bool is_causal) {
  if (q_dtype != (int)FlashAttnDtype::FP16 && q_dtype != (int)FlashAttnDtype::BF16) {
    throw std::invalid_argument("ark::block_sparse_sdpa: q_dtype must be FP16 or BF16");
  }
  if (mask && is_causal) {
    throw std::invalid_argument("ark::block_sparse_sdpa: mask and is_causal cannot both be set");
  }
  if (!lut || !valid_block_num) {
    throw std::invalid_argument("ark::block_sparse_sdpa: lut and valid_block_num must be provided");
  }
  auto matches_block_size = [](int seq_len, int num_blocks, int block_size) {
    return block_size > 0 && num_blocks == ((seq_len + block_size - 1) / block_size);
  };
  const bool key_block_is_64 = matches_block_size(seq_len_kv, num_k_blocks, 64);
  if (!key_block_is_64) {
    throw std::invalid_argument("ark::block_sparse_sdpa: only key block size 64 is supported");
  }
  const bool is_bf16 = (q_dtype == (int)FlashAttnDtype::BF16);
  if (head_dim == 64) {
    if (!matches_block_size(seq_len_q, num_q_blocks, 64)) {
      throw std::invalid_argument("ark::block_sparse_sdpa: head_dim=64 requires query block size 64");
    }
    if (is_bf16) {
      ark::sdpa_impl_bf16_sparse_sdpa_d64(
          (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (void*)lut,
          (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h,
          q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
          seq_len_kv, head_dim, softmax_scale, is_causal);
    } else {
      ark::sdpa_impl_fp16_sparse_sdpa_d64(
          (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (void*)lut,
          (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h,
          q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
          v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
          seq_len_kv, head_dim, softmax_scale, is_causal);
    }
    return;
  }
  if (head_dim == 128) {
    if (matches_block_size(seq_len_q, num_q_blocks, 256)) {
      if (q_tile_override != 256) {
        throw std::invalid_argument(
            "ark::block_sparse_sdpa: head_dim=128 query block size 256 requires q_tile_override=256");
      }
      if (is_bf16) {
        ark::sdpa_impl_bf16_sparse_sdpa_qtile256_row64k(
            (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (void*)lut,
            (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h,
            q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
            v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
            seq_len_kv, head_dim, softmax_scale, is_causal);
      } else {
        ark::sdpa_impl_fp16_sparse_sdpa_qtile256_row64k(
            (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (void*)lut,
            (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h,
            q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
            v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
            seq_len_kv, head_dim, softmax_scale, is_causal);
      }
      return;
    }
    if (matches_block_size(seq_len_q, num_q_blocks, 64)) {
      if (is_bf16) {
        ark::sdpa_impl_bf16_sparse_sdpa_row_linear(
            (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (void*)lut,
            (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h,
            q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
            v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
            seq_len_kv, head_dim, softmax_scale, is_causal);
      } else {
        ark::sdpa_impl_fp16_sparse_sdpa_row_linear(
            (sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (void*)lut,
            (void*)valid_block_num, num_q_blocks, num_k_blocks, q_tile_override, q_stride_s, q_stride_d, q_stride_h,
            q_stride_b, k_stride_s, k_stride_d, k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h,
            v_stride_b, o_stride_s, o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
            seq_len_kv, head_dim, softmax_scale, is_causal);
      }
      return;
    }
    throw std::invalid_argument("ark::block_sparse_sdpa: head_dim=128 supports query block sizes 64 and 256 only");
  }
  throw std::invalid_argument("ark::block_sparse_sdpa: unsupported head_dim; supported values are 64 and 128");
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

#elif !defined(ARK_XPU)

enum class CpuSdpaRoute {
  Scalar = 0,
  MixedRaw = 1,
  HomogeneousFp16 = 2,
  HomogeneousBf16 = 3,
};

struct CpuSdpaRequest {
  torch_ptr Q;
  torch_ptr K;
  torch_ptr V;
  torch_ptr O;
  torch_ptr mask;
  int q_stride_s;
  int q_stride_d;
  int q_stride_h;
  int q_stride_b;
  int k_stride_s;
  int k_stride_d;
  int k_stride_h;
  int k_stride_b;
  int v_stride_d;
  int v_stride_s;
  int v_stride_h;
  int v_stride_b;
  int o_stride_s;
  int o_stride_d;
  int o_stride_h;
  int o_stride_b;
  BTLA_DTYPE q_dtype;
  BTLA_DTYPE k_dtype;
  BTLA_DTYPE o_dtype;
  int batch;
  int num_heads_q;
  int num_heads_kv;
  int seq_len_q;
  int seq_len_kv;
  int head_dim;
  float softmax_scale;
  bool is_causal;

  bool mixed_dtype() const {
    return q_dtype == BTLA_DTYPE::F32 && o_dtype == BTLA_DTYPE::F32 &&
           (k_dtype == BTLA_DTYPE::F16 || k_dtype == BTLA_DTYPE::BF16);
  }

  bool homogeneous_fp16_dtype() const {
    return q_dtype == BTLA_DTYPE::F16 && k_dtype == BTLA_DTYPE::F16 && o_dtype == BTLA_DTYPE::F16;
  }

  bool homogeneous_bf16_dtype() const {
    return q_dtype == BTLA_DTYPE::BF16 && k_dtype == BTLA_DTYPE::BF16 && o_dtype == BTLA_DTYPE::BF16;
  }
};

static ark::cpu::attn_fwd_args_t make_bestla_attn_args(const CpuSdpaRequest& req) {
  ark::cpu::attn_fwd_args_t args;
  args.Q = reinterpret_cast<void*>(req.Q);
  args.K = reinterpret_cast<void*>(req.K);
  args.V = reinterpret_cast<void*>(req.V);
  args.dst = reinterpret_cast<void*>(req.O);
  args.QK_scale = req.softmax_scale;
  args.attn_flags = ark::cpu::ATTN_FLAG_NONE;
  if (req.is_causal) args.attn_flags |= ark::cpu::ATTN_FLAG_IS_CAUSAL;
  args.batch_size = req.batch;
  args.head_num = req.num_heads_q;
  args.heads_kv = req.num_heads_kv;
  args.head_size = req.head_dim;
  args.sl_q = req.seq_len_q;
  args.sl_kv = req.seq_len_kv;
  args.Q_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  args.K_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  args.V_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  args.dst_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  args.step_q_bs = req.q_stride_b;
  args.step_q_head_num = req.q_stride_h;
  args.step_q_sl = req.q_stride_s;
  args.step_k_bs = req.k_stride_b;
  args.step_k_head_num = req.k_stride_h;
  args.step_k_sl = req.k_stride_s;
  args.step_k_head_size = req.k_stride_d;
  args.step_v_bs = req.v_stride_b;
  args.step_v_head_num = req.v_stride_h;
  args.step_v_sl = req.v_stride_s;
  args.step_v_head_size = req.v_stride_d;
  args.step_dst_bs = req.o_stride_b;
  args.step_dst_head_num = req.o_stride_h;
  args.step_dst_sl = req.o_stride_s;
  args.tmp = nullptr;
  args.threading = ark::CpuWrapper::get_threading();
  return args;
}

static CpuSdpaRoute select_cpu_sdpa_route(const CpuSdpaRequest& req) {
  if (req.mixed_dtype()) {
    return CpuSdpaRoute::MixedRaw;
  }
  if (req.homogeneous_fp16_dtype()) {
    return CpuSdpaRoute::HomogeneousFp16;
  }
  if (req.homogeneous_bf16_dtype()) {
    return CpuSdpaRoute::HomogeneousBf16;
  }
  return CpuSdpaRoute::Scalar;
}

static bool can_dispatch_mixed_raw(const CpuSdpaRequest& req) {
  if (!req.mixed_dtype() || req.mask) {
    return false;
  }
  return ark::CpuWrapper::get_threading() != nullptr;
}

static bool can_dispatch_homogeneous_fp16(const CpuSdpaRequest& req) {
#if !CompileFP16()
  (void)req;
  return false;
#else
  auto* cpu = bestla::device::CpuDevice::getInstance();
  const bool gqa_ok = req.num_heads_kv > 0 && req.num_heads_q > 0 && (req.num_heads_q % req.num_heads_kv) == 0;
  const bool causal_shape_ok = !req.is_causal || req.seq_len_q <= req.seq_len_kv;
  const bool v_plain_ok = req.v_stride_d == 1;
  return cpu->AVX512_FP16() && gqa_ok && causal_shape_ok && v_plain_ok && !req.mask &&
         ark::CpuWrapper::get_threading() != nullptr;
#endif
}

static void dispatch_homogeneous_fp16(const CpuSdpaRequest& req) {
  std::vector<uint16_t> transposed_k = transpose_plain_half_k_for_homogeneous_fp16(
      req.K, req.k_stride_s, req.k_stride_d, req.k_stride_h, req.k_stride_b, req.batch, req.num_heads_kv,
      req.seq_len_kv, req.head_dim);
  auto hargs = make_bestla_attn_args(req);
  hargs.K = transposed_k.data();
  hargs.step_k_bs = req.num_heads_kv * req.head_dim * req.seq_len_kv;
  hargs.step_k_head_num = req.head_dim * req.seq_len_kv;
  hargs.step_k_sl = 1;
  hargs.step_k_head_size = req.seq_len_kv;
  if (hargs.threading == nullptr) {
    throw std::runtime_error("ark::sdpa: CPU threading handle is unavailable for the homogeneous fp16 route");
  }
  ark::cpu::bestla_sdpa_forward_homogeneous(hargs, BTLA_DTYPE::F16);
}

static bool can_dispatch_homogeneous_bf16(const CpuSdpaRequest& req) {
 #if !CompileBF16()
  (void)req;
  return false;
 #else
  auto* cpu = bestla::device::CpuDevice::getInstance();
  const bool no_gqa = req.num_heads_q > 0 && req.num_heads_q == req.num_heads_kv;
  const bool causal_shape_ok = !req.is_causal || req.seq_len_q <= req.seq_len_kv;
  const bool k_plain_ok = req.k_stride_d == 1;
  const bool v_plain_ok = req.v_stride_d == 1;
  return cpu->AMX_BF16() && no_gqa && causal_shape_ok && k_plain_ok && v_plain_ok && !req.mask &&
         ark::CpuWrapper::get_threading() != nullptr;
 #endif
}

static void dispatch_homogeneous_bf16(const CpuSdpaRequest& req) {
  auto hargs = make_bestla_attn_args(req);
  if (hargs.threading == nullptr) {
    throw std::runtime_error("ark::sdpa: CPU threading handle is unavailable for the homogeneous bf16 route");
  }
  ark::cpu::bestla_sdpa_forward_homogeneous(hargs, BTLA_DTYPE::BF16);
}

static CpuSdpaRoute resolve_cpu_sdpa_route(const CpuSdpaRequest& req) {
  switch (select_cpu_sdpa_route(req)) {
    case CpuSdpaRoute::MixedRaw:
      return can_dispatch_mixed_raw(req) ? CpuSdpaRoute::MixedRaw : CpuSdpaRoute::Scalar;
    case CpuSdpaRoute::HomogeneousFp16:
      return can_dispatch_homogeneous_fp16(req) ? CpuSdpaRoute::HomogeneousFp16 : CpuSdpaRoute::Scalar;
    case CpuSdpaRoute::HomogeneousBf16:
      return can_dispatch_homogeneous_bf16(req) ? CpuSdpaRoute::HomogeneousBf16 : CpuSdpaRoute::Scalar;
    case CpuSdpaRoute::Scalar:
      return CpuSdpaRoute::Scalar;
  }
  return CpuSdpaRoute::Scalar;
}

static void dispatch_mixed_raw(const CpuSdpaRequest& req) {
  auto bargs = make_bestla_attn_args(req);
  ark::cpu::bestla_sdpa_forward(bargs, req.k_dtype);
}

static void dispatch_scalar(const CpuSdpaRequest& req) {
  if (req.mixed_dtype()) {
    ark::cpu::MhaReferenceArgs args;
    args.query = reinterpret_cast<const void*>(req.Q);
    args.key = reinterpret_cast<const void*>(req.K);
    args.value = reinterpret_cast<const void*>(req.V);
    args.output = reinterpret_cast<void*>(req.O);
    args.attn_mask = req.mask ? reinterpret_cast<const float*>(req.mask) : nullptr;
    args.q_strides = {req.q_stride_s, req.q_stride_d, req.q_stride_h, req.q_stride_b};
    args.k_strides = {req.k_stride_s, req.k_stride_d, req.k_stride_h, req.k_stride_b};
    args.v_strides = {req.v_stride_d, req.v_stride_s, req.v_stride_h, req.v_stride_b};
    args.o_strides = {req.o_stride_s, req.o_stride_d, req.o_stride_h, req.o_stride_b};
    args.q_dtype = req.q_dtype;
    args.kv_dtype = req.k_dtype;
    args.o_dtype = req.o_dtype;
    args.batch = req.batch;
    args.num_heads_q = req.num_heads_q;
    args.num_heads_kv = req.num_heads_kv;
    args.seq_len_q = req.seq_len_q;
    args.seq_len_kv = req.seq_len_kv;
    args.head_dim = req.head_dim;
    args.softmax_scale = req.softmax_scale;
    args.is_causal = req.is_causal;
    ark::cpu::mha_reference_forward(args);
    return;
  }
  if (req.k_dtype != req.q_dtype || req.o_dtype != req.q_dtype) {
    throw std::invalid_argument("ark::sdpa: k_dtype and o_dtype must match q_dtype for homogeneous scalar dispatch");
  }
  ark::cpu::MhaDenseArgs args;
  args.query = reinterpret_cast<const void*>(req.Q);
  args.key = reinterpret_cast<const void*>(req.K);
  args.value = reinterpret_cast<const void*>(req.V);
  args.output = reinterpret_cast<void*>(req.O);
  args.attn_mask = req.mask ? reinterpret_cast<const float*>(req.mask) : nullptr;
  args.q_strides = {req.q_stride_s, req.q_stride_d, req.q_stride_h, req.q_stride_b};
  args.k_strides = {req.k_stride_s, req.k_stride_d, req.k_stride_h, req.k_stride_b};
  args.v_strides = {req.v_stride_d, req.v_stride_s, req.v_stride_h, req.v_stride_b};
  args.o_strides = {req.o_stride_s, req.o_stride_d, req.o_stride_h, req.o_stride_b};
  args.dtype = req.q_dtype;
  args.batch = req.batch;
  args.num_heads_q = req.num_heads_q;
  args.num_heads_kv = req.num_heads_kv;
  args.seq_len_q = req.seq_len_q;
  args.seq_len_kv = req.seq_len_kv;
  args.head_dim = req.head_dim;
  args.softmax_scale = req.softmax_scale;
  args.is_causal = req.is_causal;
  ark::cpu::sdpa_forward(args);
}

// Route selection is intentionally organized in two stages:
//   1. select_cpu_sdpa_route(req) picks the candidate backend from the standard
//      SDPA contract only (dtype tuple first, then the homogeneous families).
//   2. resolve_cpu_sdpa_route(req) folds in actual dispatchability for that
//      candidate (masking mode, decode/prefill shape, GQA constraints, ISA,
//      stride/layout requirements, and env-gated mixed route availability).
// Execution must always switch on the final resolved route, never a raw
// candidate, so debug resolution and actual dispatch stay identical.
static void sdpa(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                 int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b, int k_stride_s, int k_stride_d,
                 int k_stride_h, int k_stride_b, int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                 int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int q_dtype, int k_dtype, int o_dtype,
                 int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                 float softmax_scale, bool is_causal) {
  (void)stream;
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sdpa: mask and is_causal cannot both be set");
  }
  const CpuSdpaRequest req{
      Q,
      K,
      V,
      O,
      mask,
      q_stride_s,
      q_stride_d,
      q_stride_h,
      q_stride_b,
      k_stride_s,
      k_stride_d,
      k_stride_h,
      k_stride_b,
      v_stride_d,
      v_stride_s,
      v_stride_h,
      v_stride_b,
      o_stride_s,
      o_stride_d,
      o_stride_h,
      o_stride_b,
      static_cast<BTLA_DTYPE>(q_dtype),
      static_cast<BTLA_DTYPE>(k_dtype),
      static_cast<BTLA_DTYPE>(o_dtype),
      batch,
      num_heads_q,
      num_heads_kv,
      seq_len_q,
      seq_len_kv,
      head_dim,
      softmax_scale,
      is_causal,
  };

  switch (resolve_cpu_sdpa_route(req)) {
   case CpuSdpaRoute::MixedRaw:
     dispatch_mixed_raw(req);
     return;
   case CpuSdpaRoute::HomogeneousFp16:
     dispatch_homogeneous_fp16(req);
     return;
   case CpuSdpaRoute::HomogeneousBf16:
     dispatch_homogeneous_bf16(req);
     return;
   case CpuSdpaRoute::Scalar:
     dispatch_scalar(req);
     return;
  }
}

static int ark_cpu_debug_resolve_sdpa_route(torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                                            int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                                            int k_stride_s, int k_stride_d, int k_stride_h, int k_stride_b,
                                            int v_stride_d, int v_stride_s, int v_stride_h, int v_stride_b,
                                            int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int q_dtype,
                                            int k_dtype, int o_dtype, int batch, int num_heads_q, int num_heads_kv,
                                            int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale,
                                            bool is_causal) {
  const CpuSdpaRequest req{
      Q,
      K,
      V,
      O,
      mask,
      q_stride_s,
      q_stride_d,
      q_stride_h,
      q_stride_b,
      k_stride_s,
      k_stride_d,
      k_stride_h,
      k_stride_b,
      v_stride_d,
      v_stride_s,
      v_stride_h,
      v_stride_b,
      o_stride_s,
      o_stride_d,
      o_stride_h,
      o_stride_b,
      static_cast<BTLA_DTYPE>(q_dtype),
      static_cast<BTLA_DTYPE>(k_dtype),
      static_cast<BTLA_DTYPE>(o_dtype),
      batch,
      num_heads_q,
      num_heads_kv,
      seq_len_q,
      seq_len_kv,
      head_dim,
      softmax_scale,
      is_causal,
  };
  return static_cast<int>(resolve_cpu_sdpa_route(req));
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
// NS-parity persistent packed KV cache Python helpers (Tier 1 / internal).
//
// These four functions expose the packed-cache path for Python consumers:
//   ark_cpu_packed_kv_elems  — query the element counts for a given cache shape
//   ark_cpu_update_packed_k  — append raw K into the persistent packed K cache
//   ark_cpu_update_packed_v  — append raw V into the persistent packed V cache
//   ark_cpu_bestla_sdpa_packed — forward attention over a packed K/V cache
//
// kv_dtype must be F16 (15) or BF16 (14) — matching BTLA_DTYPE values.
// ---------------------------------------------------------------------------

static ark::cpu::ReorderKVShape ark_cpu_packed_kv_descriptor(int batch, int num_heads_kv, int capacity, int head_dim,
                                                              int kv_dtype_int) {
  return ark::cpu::packed_kv_cache_info(batch, num_heads_kv, capacity, head_dim, static_cast<BTLA_DTYPE>(kv_dtype_int));
}

static py::dict ark_cpu_packed_kv_info_desc(const ark::cpu::ReorderKVShape& shape) {
  py::dict out;
  out["dtype"] = static_cast<int>(shape.dtype);
  out["layout"] = static_cast<int>(shape.layout);
  out["k_layout"] = static_cast<int>(shape.k_layout);
  out["v_layout"] = static_cast<int>(shape.v_layout);
  out["ntile"] = shape.ntile;
  out["rowpack"] = shape.rowpack;
  out["batch_size"] = shape.batch_size;
  out["heads_kv"] = shape.heads_kv;
  out["head_dim"] = shape.head_dim;
  out["logical_capacity"] = shape.logical_capacity;
  out["num_heads"] = shape.num_heads;
  out["k_seq_pad"] = shape.k_seq_pad;
  out["k_head_size_pad"] = shape.k_head_size_pad;
  out["v_seq_pad"] = shape.v_seq_pad;
  out["v_head_size_pad"] = shape.v_head_size_pad;
  out["elem_bytes"] = static_cast<int64_t>(shape.elem_bytes);
  out["k_head_elems"] = static_cast<int64_t>(shape.k_head_elems);
  out["v_head_elems"] = static_cast<int64_t>(shape.v_head_elems);
  out["k_total_elems"] = static_cast<int64_t>(shape.k_total_elems);
  out["v_total_elems"] = static_cast<int64_t>(shape.v_total_elems);
  out["k_bytes"] = static_cast<int64_t>(shape.k_bytes);
  out["v_bytes"] = static_cast<int64_t>(shape.v_bytes);
  out["step_k_bs"] = shape.step_k_bs;
  out["step_k_head_num"] = shape.step_k_head_num;
  out["step_k_sl"] = shape.step_k_sl;
  out["step_k_head_size"] = shape.step_k_head_size;
  out["step_v_bs"] = shape.step_v_bs;
  out["step_v_head_num"] = shape.step_v_head_num;
  out["step_v_sl"] = shape.step_v_sl;
  out["step_v_head_size"] = shape.step_v_head_size;
  return out;
}

// Returns (k_elems, v_elems): element counts for 1D allocation of the packed cache.
static std::pair<int64_t, int64_t> ark_cpu_packed_kv_elems_desc(const ark::cpu::ReorderKVShape& shape) {
  // Each packed head occupies k_head_elems / v_head_elems elements; total over all
  // batch×head slots gives the required 1D buffer size (in kv_dtype elements).
  return {static_cast<int64_t>(shape.k_total_elems), static_cast<int64_t>(shape.v_total_elems)};
}

static std::pair<int64_t, int64_t> ark_cpu_packed_kv_elems(int batch, int num_heads_kv, int capacity, int head_dim,
                                                            int kv_dtype_int) {
  return ark_cpu_packed_kv_elems_desc(
      ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int));
}

static py::dict ark_cpu_packed_kv_info(int batch, int num_heads_kv, int capacity, int head_dim, int kv_dtype_int) {
  return ark_cpu_packed_kv_info_desc(
      ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int));
}

// Append raw K tokens at [start_pos, start_pos+append_len) into the packed K cache.
static void ark_cpu_update_packed_k_desc(torch_ptr cache_k, torch_ptr key, int k_stride_s, int k_stride_d, int k_stride_h,
                                         int k_stride_b, const ark::cpu::ReorderKVShape& shape, int append_len,
                                         int start_pos, bool no_zeroing) {
  ark::cpu::update_packed_k_cache((void*)cache_k, (const void*)key, shape,
                                  {k_stride_s, k_stride_d, k_stride_h, k_stride_b}, append_len, start_pos, no_zeroing);
}

static void ark_cpu_update_packed_k(torch_ptr cache_k, torch_ptr key, int k_stride_s, int k_stride_d, int k_stride_h,
                                    int k_stride_b, int kv_dtype_int, int batch, int num_heads_kv, int append_len,
                                    int head_dim, int capacity, int start_pos, bool no_zeroing) {
  ark_cpu_update_packed_k_desc(cache_k, key, k_stride_s, k_stride_d, k_stride_h, k_stride_b,
                               ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int),
                               append_len, start_pos, no_zeroing);
}

// Append raw V tokens at [start_pos, start_pos+append_len) into the packed V cache.
static void ark_cpu_update_packed_v_desc(torch_ptr cache_v, torch_ptr value, int v_stride_d, int v_stride_s, int v_stride_h,
                                         int v_stride_b, const ark::cpu::ReorderKVShape& shape, int append_len,
                                         int start_pos, bool no_zeroing) {
  ark::cpu::update_packed_v_cache((void*)cache_v, (const void*)value, shape,
                                  {v_stride_d, v_stride_s, v_stride_h, v_stride_b}, append_len, start_pos, no_zeroing);
}

static void ark_cpu_update_packed_v(torch_ptr cache_v, torch_ptr value, int v_stride_d, int v_stride_s, int v_stride_h,
                                    int v_stride_b, int kv_dtype_int, int batch, int num_heads_kv, int append_len,
                                    int head_dim, int capacity, int start_pos, bool no_zeroing) {
  ark_cpu_update_packed_v_desc(cache_v, value, v_stride_d, v_stride_s, v_stride_h, v_stride_b,
                               ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int),
                               append_len, start_pos, no_zeroing);
}

static void ark_cpu_copy_packed_k_desc(torch_ptr dst_cache_k, torch_ptr src_cache_k, const ark::cpu::ReorderKVShape& shape,
                                       int seq_off, int seq_size, bool no_zeroing) {
  ark::cpu::copy_packed_k_cache((void*)dst_cache_k, (const void*)src_cache_k, shape, seq_off, seq_size, no_zeroing);
}

static void ark_cpu_copy_packed_k(torch_ptr dst_cache_k, torch_ptr src_cache_k, int kv_dtype_int, int batch,
                                  int num_heads_kv, int capacity, int head_dim, int seq_off, int seq_size,
                                  bool no_zeroing) {
  ark_cpu_copy_packed_k_desc(dst_cache_k, src_cache_k,
                             ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int), seq_off,
                             seq_size, no_zeroing);
}

static void ark_cpu_copy_packed_v_desc(torch_ptr dst_cache_v, torch_ptr src_cache_v, const ark::cpu::ReorderKVShape& shape,
                                       int seq_off, int seq_size, bool no_zeroing) {
  ark::cpu::copy_packed_v_cache((void*)dst_cache_v, (const void*)src_cache_v, shape, seq_off, seq_size, no_zeroing);
}

static void ark_cpu_copy_packed_v(torch_ptr dst_cache_v, torch_ptr src_cache_v, int kv_dtype_int, int batch,
                                  int num_heads_kv, int capacity, int head_dim, int seq_off, int seq_size,
                                  bool no_zeroing) {
  ark_cpu_copy_packed_v_desc(dst_cache_v, src_cache_v,
                             ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int), seq_off,
                             seq_size, no_zeroing);
}

static void ark_cpu_shift_packed_k_desc(torch_ptr cache_k, torch_ptr cossin, const ark::cpu::ReorderKVShape& shape,
                                        int seq_keep) {
  ark::cpu::shift_packed_k_cache_rope((void*)cache_k, (const bestla::utils::fp16*)cossin, shape, seq_keep);
}

static void ark_cpu_shift_packed_k(torch_ptr cache_k, torch_ptr cossin, int kv_dtype_int, int batch, int num_heads_kv,
                                   int capacity, int head_dim, int seq_keep) {
  ark_cpu_shift_packed_k_desc(cache_k, cossin,
                              ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int), seq_keep);
}

// Forward attention over a pre-packed K/V cache.
// q_dtype must be F32 (10); kv_dtype must be F16 (15) or BF16 (14).
// sl_kv is the current valid sequence length (must be <= capacity).
static void ark_cpu_bestla_sdpa_packed_desc(torch_ptr Q, torch_ptr K_packed, torch_ptr V_packed, torch_ptr O,
                                            int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                                            int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int q_dtype,
                                            const ark::cpu::ReorderKVShape& shape, int num_heads_q, int seq_len_q,
                                            int seq_len_kv, float softmax_scale, bool is_causal) {
  if (static_cast<BTLA_DTYPE>(q_dtype) != BTLA_DTYPE::F32) {
    throw std::invalid_argument("ark_cpu_bestla_sdpa_packed: q_dtype must be F32 (10)");
  }
  ark::cpu::attn_fwd_args_t bargs;
  bargs.Q = (void*)Q;
  bargs.K = (void*)K_packed;
  bargs.V = (void*)V_packed;
  bargs.dst = (void*)O;
  bargs.QK_scale = softmax_scale;
  bargs.attn_flags = ark::cpu::ATTN_FLAG_NONE;
  if (is_causal) bargs.attn_flags |= ark::cpu::ATTN_FLAG_IS_CAUSAL;
  bargs.batch_size = shape.batch_size;
  bargs.head_num = num_heads_q;
  bargs.heads_kv = shape.heads_kv;
  bargs.head_size = shape.head_dim;
  bargs.sl_q = seq_len_q;
  bargs.sl_kv = seq_len_kv;
  bargs.Q_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  bargs.dst_layout = ark::cpu::ATTN_FWD_LAYOUT_PLAIN;
  // Packed forward consumes an already-reordered persistent cache, so K/V must
  // be tagged with the packed layout derived from `shape`.
  bargs.K_layout = shape.k_layout;
  bargs.V_layout = shape.v_layout;
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
  ark::cpu::bestla_sdpa_forward_packed(bargs, shape);
}

static void ark_cpu_bestla_sdpa_packed(torch_ptr Q, torch_ptr K_packed, torch_ptr V_packed, torch_ptr O,
                                       int q_stride_s, int q_stride_d, int q_stride_h, int q_stride_b,
                                       int o_stride_s, int o_stride_d, int o_stride_h, int o_stride_b, int q_dtype,
                                       int kv_dtype_int, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                                       int seq_len_kv, int capacity, int head_dim, float softmax_scale, bool is_causal) {
  ark_cpu_bestla_sdpa_packed_desc(
      Q, K_packed, V_packed, O, q_stride_s, q_stride_d, q_stride_h, q_stride_b, o_stride_s, o_stride_d, o_stride_h,
      o_stride_b, q_dtype, ark_cpu_packed_kv_descriptor(batch, num_heads_kv, capacity, head_dim, kv_dtype_int),
      num_heads_q, seq_len_q, seq_len_kv, softmax_scale, is_causal);
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
  m.def("block_sparse_sdpa", &ark::block_sparse_sdpa);
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
  m.def("moe_gemm", &ark::moe_gemm_wrapper);
  m.def("moe_gemm_decode", &ark::moe_gemm_decode_wrapper);
  m.def("moe_gemm_prefill", &ark::moe_gemm_prefill_wrapper);
  m.def("moe_gemm_prefill_fp8_dpas", &ark::moe_gemm_prefill_fp8_dpas_wrapper);
  m.def("moe_gemm_prefill_int_dpas", &ark::moe_gemm_prefill_int_dpas_wrapper);
  m.def("matmul_sycl_tla", &ark::matmul_sycl_tla);
#endif  // ARK_SYCL_TLA
#if !defined(ARK_XPU)
  pybind11::class_<ark::cpu::ReorderKVShape>(m, "ArkCpuPackedKVDescriptor")
      .def(pybind11::init<>())
      .def_readonly("dtype", &ark::cpu::ReorderKVShape::dtype)
      .def_readonly("layout", &ark::cpu::ReorderKVShape::layout)
      .def_readonly("k_layout", &ark::cpu::ReorderKVShape::k_layout)
      .def_readonly("v_layout", &ark::cpu::ReorderKVShape::v_layout)
      .def_readonly("ntile", &ark::cpu::ReorderKVShape::ntile)
      .def_readonly("rowpack", &ark::cpu::ReorderKVShape::rowpack)
      .def_readonly("batch_size", &ark::cpu::ReorderKVShape::batch_size)
      .def_readonly("heads_kv", &ark::cpu::ReorderKVShape::heads_kv)
      .def_readonly("head_dim", &ark::cpu::ReorderKVShape::head_dim)
      .def_readonly("logical_capacity", &ark::cpu::ReorderKVShape::logical_capacity)
      .def_readonly("num_heads", &ark::cpu::ReorderKVShape::num_heads)
      .def_readonly("k_seq_pad", &ark::cpu::ReorderKVShape::k_seq_pad)
      .def_readonly("k_head_size_pad", &ark::cpu::ReorderKVShape::k_head_size_pad)
      .def_readonly("v_seq_pad", &ark::cpu::ReorderKVShape::v_seq_pad)
      .def_readonly("v_head_size_pad", &ark::cpu::ReorderKVShape::v_head_size_pad)
      .def_readonly("elem_bytes", &ark::cpu::ReorderKVShape::elem_bytes)
      .def_readonly("k_head_elems", &ark::cpu::ReorderKVShape::k_head_elems)
      .def_readonly("v_head_elems", &ark::cpu::ReorderKVShape::v_head_elems)
      .def_readonly("k_total_elems", &ark::cpu::ReorderKVShape::k_total_elems)
      .def_readonly("v_total_elems", &ark::cpu::ReorderKVShape::v_total_elems)
      .def_readonly("k_bytes", &ark::cpu::ReorderKVShape::k_bytes)
      .def_readonly("v_bytes", &ark::cpu::ReorderKVShape::v_bytes)
      .def_readonly("step_k_bs", &ark::cpu::ReorderKVShape::step_k_bs)
      .def_readonly("step_k_head_num", &ark::cpu::ReorderKVShape::step_k_head_num)
      .def_readonly("step_k_sl", &ark::cpu::ReorderKVShape::step_k_sl)
      .def_readonly("step_k_head_size", &ark::cpu::ReorderKVShape::step_k_head_size)
      .def_readonly("step_v_bs", &ark::cpu::ReorderKVShape::step_v_bs)
      .def_readonly("step_v_head_num", &ark::cpu::ReorderKVShape::step_v_head_num)
      .def_readonly("step_v_sl", &ark::cpu::ReorderKVShape::step_v_sl)
      .def_readonly("step_v_head_size", &ark::cpu::ReorderKVShape::step_v_head_size);
  m.attr("ARK_CPU_SDPA_ROUTE_SCALAR") = pybind11::int_(static_cast<int>(ark::CpuSdpaRoute::Scalar));
  m.attr("ARK_CPU_SDPA_ROUTE_MIXED_RAW") = pybind11::int_(static_cast<int>(ark::CpuSdpaRoute::MixedRaw));
  m.attr("ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_FP16") = pybind11::int_(static_cast<int>(ark::CpuSdpaRoute::HomogeneousFp16));
  m.attr("ARK_CPU_SDPA_ROUTE_HOMOGENEOUS_BF16") = pybind11::int_(static_cast<int>(ark::CpuSdpaRoute::HomogeneousBf16));
  m.attr("ARK_CPU_SDPA_BUILD_HAS_FP16_ROUTE") = pybind11::bool_(CompileFP16());
  m.attr("ARK_CPU_SDPA_BUILD_HAS_BF16_ROUTE") = pybind11::bool_(CompileBF16());
  m.def("ark_cpu_debug_resolve_sdpa_route", &ark::ark_cpu_debug_resolve_sdpa_route);
  m.def("ark_cpu_kv_update", &ark::ark_cpu_kv_update);
  m.def("ark_cpu_packed_kv_descriptor", &ark::ark_cpu_packed_kv_descriptor);
  m.def("ark_cpu_packed_kv_elems", &ark::ark_cpu_packed_kv_elems);
  m.def("ark_cpu_packed_kv_info", &ark::ark_cpu_packed_kv_info);
  m.def("ark_cpu_packed_kv_elems_desc", &ark::ark_cpu_packed_kv_elems_desc);
  m.def("ark_cpu_packed_kv_info_desc", &ark::ark_cpu_packed_kv_info_desc);
  m.def("ark_cpu_update_packed_k", &ark::ark_cpu_update_packed_k);
  m.def("ark_cpu_update_packed_v", &ark::ark_cpu_update_packed_v);
  m.def("ark_cpu_update_packed_k_desc", &ark::ark_cpu_update_packed_k_desc);
  m.def("ark_cpu_update_packed_v_desc", &ark::ark_cpu_update_packed_v_desc);
  m.def("ark_cpu_copy_packed_k", &ark::ark_cpu_copy_packed_k);
  m.def("ark_cpu_copy_packed_v", &ark::ark_cpu_copy_packed_v);
  m.def("ark_cpu_copy_packed_k_desc", &ark::ark_cpu_copy_packed_k_desc);
  m.def("ark_cpu_copy_packed_v_desc", &ark::ark_cpu_copy_packed_v_desc);
  m.def("ark_cpu_shift_packed_k", &ark::ark_cpu_shift_packed_k);
  m.def("ark_cpu_shift_packed_k_desc", &ark::ark_cpu_shift_packed_k_desc);
  m.def("ark_cpu_bestla_sdpa_packed_desc", &ark::ark_cpu_bestla_sdpa_packed_desc);
  m.def("ark_cpu_bestla_sdpa_packed", &ark::ark_cpu_bestla_sdpa_packed);
#endif
}
