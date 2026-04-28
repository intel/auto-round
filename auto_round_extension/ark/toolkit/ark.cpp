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

#include <iostream>
#include <stdexcept>
#include "bestla/bestla/bestla.h"
typedef uintptr_t torch_ptr;
#if ARK_XPU
#include <sycl/sycl.hpp>
#include "xpu_wrapper.hpp"
#if ARK_SYCL_TLA
// Only include declarations, implementations are in separate .cpp files
#include "sycl_tla_common.hpp"
#include "sycl_tla_moe.hpp"
#include "sycl_tla_sdpa.hpp"
#endif
#else
#include "cpu_wrapper.hpp"
#endif

#include "dnnl_wrapper.hpp"

namespace ark {

static void matmul(torch_ptr stream, int m, int n, int k, torch_ptr A, int Adt, torch_ptr B, int Bdt, torch_ptr C,
                   int Cdt, torch_ptr bias, bool BT) {
  auto dt = ark::to_dt((BTLA_DTYPE)Adt);
  auto cdt = dt;
  if (Adt == (int)BTLA_DTYPE::S8) cdt = dnnl::memory::data_type::s32;
#ifdef ARK_XPU
  ark::DnnlWrapper::gemm((sycl::queue*)stream, m, n, k, (void*)A, dt, (void*)B, dt, BT, (void*)C, cdt, (void*)bias);
#else
  CpuWrapper::gemm(m, n, k, (void*)A, (BTLA_DTYPE)Adt, (void*)B, BT, (float*)C, (const float*)bias);
#endif
}

static void woqgemm_s8(torch_ptr stream, int m, int n, int k, torch_ptr A, int ACdt, torch_ptr B, torch_ptr C,
                       torch_ptr bias, bool BT, torch_ptr scaleb) {
  auto dt = ark::to_dt((BTLA_DTYPE)ACdt);
  ark::DnnlWrapper::woq_s8((sycl::queue*)stream, m, n, k, (void*)A, (void*)B, BT, (void*)C, dt, (void*)scaleb,
                           (void*)bias, k);
}

static void woqgemm(torch_ptr stream, int m, int n, int k, torch_ptr A, int ACdt, torch_ptr BlobB, torch_ptr C,
                    torch_ptr bias, int blocksize, int compute_type, int weight_type, int scale_type, bool asym) {
  QuantParam param{n, k, blocksize, compute_type, weight_type, scale_type, asym};
#ifdef ARK_XPU
  XpuWrapper::woq_gemm(m, (void*)A, (void*)BlobB, (void*)C, (void*)bias, (BTLA_DTYPE)ACdt, &param,
                       (sycl::queue*)stream);
#else
  CpuWrapper::woq_gemm(m, (void*)A, (void*)BlobB, (void*)C, (void*)bias, (BTLA_DTYPE)ACdt, &param);
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
                          int compute_type, int weight_type, int scale_type, bool asym) {
  QuantParam param{n, k, blocksize, compute_type, weight_type, scale_type, asym};
#ifdef ARK_XPU
  XpuWrapper::unpackq((BTLA_DTYPE)out_type, (int8_t*)blob, (void*)output, &param, (sycl::queue*)stream);
#else
  CpuWrapper::unpackq((BTLA_DTYPE)out_type, (int8_t*)blob, (void*)output, &param);
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

static void sdpa(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask, int q_dtype,
                 int k_dtype, int o_dtype, int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                 int head_dim, float softmax_scale, bool is_causal) {
  if (k_dtype != q_dtype || o_dtype != q_dtype) {
    throw std::invalid_argument("ark::sdpa: k_dtype and o_dtype must match q_dtype");
  }
  if(q_dtype != (int)BTLA_DTYPE::F16 && q_dtype != (int)BTLA_DTYPE::BF16) {
    throw std::invalid_argument("ark::sdpa: only FP16 and BF16 are supported");
  }
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sdpa: mask and is_causal cannot both be set");
  }
  ark::sdpa_impl((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, (BTLA_DTYPE)(q_dtype),
                 batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
}

static void sagev1(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                   int scale_block_size, int q_dtype, int k_dtype, int v_dtype, int o_dtype, int batch, int num_heads_q,
                   int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim, float softmax_scale, bool is_causal) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sagev1: mask and is_causal cannot both be set");
  }
  if (q_dtype != k_dtype || o_dtype != q_dtype || v_dtype != q_dtype) {
    throw std::invalid_argument("ark::sagev1: k_dtype and o_dtype must match q_dtype");
  }
  if (q_dtype != (int)BTLA_DTYPE::F16) {
    throw std::invalid_argument("ark::sagev1: only F16 is supported for q_dtype");
  }
#ifdef ARK_XPU
  XpuWrapper::sagev1((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask, scale_block_size, batch,
                     num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
#else
  throw std::runtime_error("ark::sagev1 is only supported on XPU");
#endif
}

static void sage(torch_ptr stream, torch_ptr Q, torch_ptr K, torch_ptr V, torch_ptr O, torch_ptr mask,
                 int scale_block_size, torch_ptr qscale, torch_ptr kscale, int q_dtype, int k_dtype, int o_dtype,
                 int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv, int head_dim,
                 float softmax_scale, bool is_causal) {
  if (mask && is_causal) {
    throw std::invalid_argument("ark::sagev1: mask and is_causal cannot both be set");
  }
  ark::sdpa_impl_qks8_pvhalf((sycl::queue*)stream, (void*)Q, (void*)K, (void*)V, (void*)O, (void*)mask,
                             scale_block_size, (void*)qscale, (void*)kscale, batch, num_heads_q, num_heads_kv,
                             seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
}

static void moe_gemm_wrapper(torch_ptr stream, torch_ptr activations, torch_ptr weights, torch_ptr scales,
                             torch_ptr outputs, int dtype, int N, int K, torch_ptr num_tokens_per_expert,
                             int num_experts) {
  ark::moe_gemm((sycl::queue*)stream, (void*)activations, (void*)weights, scales ? (void*)scales : nullptr,
                (void*)outputs, (BTLA_DTYPE)(dtype), N, K, (int*)num_tokens_per_expert, num_experts);
}

static void sage_dynamic_quant(torch_ptr stream, torch_ptr input, torch_ptr output, torch_ptr scale_out, int num_rows,
                               int head_dim, int block_size) {
  auto* q = (sycl::queue*)stream;
  auto* in_ptr = (sycl::half*)input;
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

    q->parallel_for(sycl::and_range<1>(num_blocks * wg_size, wg_size),
                    [=](sycl::and_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                      int block_id = item.get_group(0);
                      int tid = item.get_local_id(0);
                      auto wg = item.get_group();
                      auto* block_in = in_ptr + (size_t)block_id * elems_per_block;
                      auto* block_out = out_ptr + (size_t)block_id * elems_per_block;

                      // Phase 1: compute absmax across entire block
                      float local_max = 0.0f;
                      sycl::vec<sycl::half, Unroll> local_data, local_max_vec;
                      local_max_vec = sycl::vec<sycl::half, Unroll>(0.0f);
                      for (int i = tid * Unroll; i < elems_per_block; i += wg_size * Unroll) {
                        local_data = *(sycl::vec<sycl::half, Unroll>*)(&block_in[i]);
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
                          float val = static_cast<float>(block_in[i + j]) * inv_scale;
                          int iv = static_cast<int>(val + (val >= 0.0f ? 0.5f : -0.5f));
                          iv = sycl::clamp(iv, -127, 127);
                          block_out[i + j] = static_cast<int8_t>(iv);
                        }
                      }
                    });
  } else {
    int wg_size = MAX_WG_SIZE;
    q->parallel_for(sycl::and_range<1>(num_blocks * wg_size, wg_size),
                    [=](sycl::and_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                      int block_id = item.get_group(0);
                      int tid = item.get_local_id(0);
                      auto wg = item.get_group();
                      auto* block_in = in_ptr + (size_t)block_id * elems_per_block;
                      auto* block_out = out_ptr + (size_t)block_id * elems_per_block;

                      // Phase 1: compute absmax across entire block
                      float local_max = 0.0f;
                      sycl::vec<sycl::half, Unroll> local_data[MAX_Reg / Unroll], local_max_vec;
                      local_max_vec = sycl::vec<sycl::half, Unroll>(0.0f);
                      int local_i = 0;
                      for (int i = tid * Unroll; i < elems_per_block; i += wg_size * Unroll, local_i++) {
                        local_data[local_i] = *(sycl::vec<sycl::half, Unroll>*)&block_in[i];
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
#endif  // ARK_XPU && ARK_SYCL_TLA

}  // namespace ark

PYBIND11_MODULE(PY_NAME, m) {
  m.def("matmul", &ark::matmul);
  m.def("woqgemm_s8", &ark::woqgemm_s8);
  m.def("woqgemm", &ark::woqgemm);
  m.def("packed_weight_size", &ark::packed_weight_size);
  m.def("repack_quantized_weight", &ark::repack_quantized_weight);
  m.def("unpack_weight", &ark::unpack_weight);
#if defined(ARK_XPU) && defined(ARK_SYCL_TLA)
  m.def("sdpa", &ark::sdpa);
  m.def("sagev1", &ark::sagev1);
  m.def("sage", &ark::sage);
  m.def("sage_dynamic_quant", &ark::sage_dynamic_quant);
  m.def("moe_gemm", &ark::moe_gemm_wrapper);
#endif
}