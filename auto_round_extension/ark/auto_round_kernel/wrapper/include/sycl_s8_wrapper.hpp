//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once

#include "utils.hpp"

#if ARK_XPU

#if ARK_SYCL_TLA
#include "sycl_tla_common.hpp"
#endif

namespace ark {

class SyclS8Wrapper {
 public:
  template <typename T>
  static sycl::event dyn_quant_s8_kblock(sycl::queue* q, int m, int k, const T* a, int8_t* qa, T* scalea,
                                         int blocksize, int blks) {
    constexpr int SgSize = 16;
    constexpr int WGSize = 256;
    constexpr int SGNum = WGSize / SgSize;

    sycl::range<1> group{WGSize};
    int groups_per_row = (blks + SGNum - 1) / SGNum;
    sycl::range<1> problem{static_cast<size_t>(m) * groups_per_row * WGSize};
    return q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SgSize)]] {
            int group_idx = int(it.get_group(0));
            int row = group_idx / groups_per_row;
            auto sg = it.get_sub_group();
            int sg_id = int(sg.get_local_id()[0]);
            int sg_group_id = int(sg.get_group_id()[0]);
            int block_idx = (group_idx % groups_per_row) * SGNum + sg_group_id;
            if (block_idx >= blks) return;
            int base = row * k + block_idx * blocksize;

            float maxabs = 0.0f;
            for (int idx = sg_id; idx < blocksize; idx += SgSize) {
              maxabs = sycl::max(maxabs, sycl::fabs(static_cast<float>(a[base + idx])));
            }
            maxabs = sycl::reduce_over_group(sg, maxabs, sycl::maximum<float>());
            float scale = maxabs / 127.0f;
            float ratio = scale > 0.0f ? 1.0f / scale : 0.0f;
            for (int idx = sg_id; idx < blocksize; idx += SgSize) {
              int value = static_cast<int>(sycl::round(static_cast<float>(a[base + idx]) * ratio));
              value = value < -128 ? -128 : value;
              value = value > 127 ? 127 : value;
              qa[base + idx] = static_cast<int8_t>(value);
            }
            if (sg_id == 0) {
              scalea[block_idx * m + row] = static_cast<T>(scale);
            }
          });
    });
  }

   static inline void prepare_qa_and_quantize(sycl::queue* q, int m, int k, const void* a, BTLA_DTYPE act, 
                                             int blocksize, int8_t*& qa_ptr, int8_t*& scalea_ptr) {
    bool use_blockwise_a = false;
#if ARK_SYCL_TLA
    use_blockwise_a = blocksize > 0 && blocksize < k;
    if (use_blockwise_a && k % blocksize != 0) {
      throw std::invalid_argument("SyclS8Wrapper::prepare_qa_and_quantize: blocksize must divide k");
    }
#endif
    int blks = use_blockwise_a ? k / blocksize : 1;
    size_t qa_size = size_t(m) * size_t(k);
    size_t scalea_offset = (qa_size + alignof(float) - 1) & ~(size_t(alignof(float)) - 1);
    size_t tmp_size = scalea_offset + size_t(m) * size_t(blks) * sizeof(float);

    auto tmp_ptr = static_cast<int8_t*>(DeviceMemoryPool::Instance()->get_scratch_mem(tmp_size, 1, q));
    qa_ptr = tmp_ptr;
    scalea_ptr = tmp_ptr + scalea_offset;

#if ARK_SYCL_TLA
    if (use_blockwise_a) {
      switch (act) {
        case BTLA_DTYPE::F32:
          dyn_quant_s8_kblock(q, m, k, static_cast<const float*>(a), qa_ptr, reinterpret_cast<float*>(scalea_ptr),
                              blocksize, blks);
          break;
        case BTLA_DTYPE::F16:
          dyn_quant_s8_kblock(q, m, k, static_cast<const sycl::half*>(a), qa_ptr,
                              reinterpret_cast<sycl::half*>(scalea_ptr), blocksize, blks);
          break;
        case BTLA_DTYPE::BF16:
          dyn_quant_s8_kblock(q, m, k, static_cast<const sycl::ext::oneapi::bfloat16*>(a), qa_ptr,
                              reinterpret_cast<sycl::ext::oneapi::bfloat16*>(scalea_ptr), blocksize, blks);
          break;
        default:
          throw std::invalid_argument("SyclS8Wrapper::prepare_qa_and_quantize: unsupported activation dtype");
      }
      return;
    }
#endif
    dyn_quant_s8(q, m, k, a, act, qa_ptr, scalea_ptr, 0);
  }
  
  static void dyn_quant_s8(sycl::queue* q, int m, int k, const void* a, BTLA_DTYPE adt, int8_t* qa, void* scalea,
                           int mask) {
    if (adt == BTLA_DTYPE::F32) {
      using T = float;
      using Pro = bestla::sycl_prologue_a::ActivationBase<T>;
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(T*)a, k}, qa, (T*)scalea, q);
    } else if (adt == BTLA_DTYPE::F16) {
      using T = sycl::half;
      using Pro = bestla::sycl_prologue_a::ActivationBase<T>;
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(T*)a, k}, qa, (T*)scalea, q);
    } else if (adt == BTLA_DTYPE::BF16) {
      using T = sycl::ext::oneapi::bfloat16;
      using Pro = bestla::sycl_prologue_a::ActivationBase<T>;
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(T*)a, k}, qa, (T*)scalea, q);
    } else {
      throw std::invalid_argument("SyclS8Wrapper::dyn_quant_s8: unsupported activation dtype");
    }
  }

  static void igemm_s8s8(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c,
                         BTLA_DTYPE ct, void* scale_a, void* scale_b, void* bias, int blocksize) {
    if (!BT) {
      throw std::invalid_argument("SyclS8Wrapper::igemm_s8s8: only B as n x k is supported");
    }

#if ARK_JOINT_MATRIX
    igemm_s8s8_joint_matrix(q, m, n, k, a, b, BT, c, ct, scale_a, scale_b, bias, blocksize);
#elif ARK_SYCL_TLA
    igemm_s8s8_sycl_tla(q, m, n, k, a, b, BT, c, ct, scale_a, scale_b, bias, blocksize);
#else
    throw std::runtime_error("SyclS8Wrapper::igemm_s8s8 requires ARK_SYCL_TLA=ON or ARK_JOINT_MATRIX=ON");
#endif
  }

#if ARK_SYCL_TLA
  static void igemm_s8s8_sycl_tla(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c,
                         BTLA_DTYPE ct, void* scale_a, void* scale_b, void* bias, int blocksize) {
    if (!BT) {
      throw std::invalid_argument("SyclS8Wrapper::igemm_s8s8: only B as n x k is supported");
    }

    ark::sycl_tla_igemm_s8s8_dequant(q, m, n, k, a, b, c, ct, scale_a, scale_b, bias, blocksize);
  }
#endif  // ARK_SYCL_TLA


#if ARK_JOINT_MATRIX
  static void igemm_s8s8_joint_matrix(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c,
                         BTLA_DTYPE ct, void* scale_a, void* scale_b, void* bias, int blocksize) {
    if (!BT) {
      throw std::invalid_argument("SyclS8Wrapper::igemm_s8s8: only B as n x k is supported");
    }

    using namespace bestla::sycl_gemm;

    if (blocksize == k || blocksize == -1) {
      if (ct == BTLA_DTYPE::F32) {
        using T = float;
        Launcher<xmx::IGemmDQCfg<T>, xmx::IGemmDQCore>::run(
            q, {(void*)a, (void*)b, c, m, n, k, k, k, n, bias, scale_a, scale_b});
      } else if (ct == BTLA_DTYPE::F16) {
        using T = sycl::half;
        Launcher<xmx::IGemmDQCfg<T>, xmx::IGemmDQCore>::run(
            q, {(void*)a, (void*)b, c, m, n, k, k, k, n, bias, scale_a, scale_b});
      } else if (ct == BTLA_DTYPE::BF16) {
        using T = sycl::ext::oneapi::bfloat16;
        Launcher<xmx::IGemmDQCfg<T>, xmx::IGemmDQCore>::run(
            q, {(void*)a, (void*)b, c, m, n, k, k, k, n, bias, scale_a, scale_b});
      } else {
        throw std::invalid_argument("SyclS8Wrapper::igemm_s8s8: unsupported output dtype");
      }
      return;
    }

    if (ct == BTLA_DTYPE::F32) {
      using T = float;
      Launcher<xmx::IKblockGemmDQCfg<T>, xmx::IKblockGemmDQCore>::run(
          q, {(void*)a, (void*)b, c, m, n, k, k, k, n, bias, scale_a, scale_b, blocksize});
    } else if (ct == BTLA_DTYPE::F16) {
      using T = sycl::half;
      Launcher<xmx::IKblockGemmDQCfg<T>, xmx::IKblockGemmDQCore>::run(
          q, {(void*)a, (void*)b, c, m, n, k, k, k, n, bias, scale_a, scale_b, blocksize});
    } else {
      throw std::invalid_argument("SyclS8Wrapper::igemm_s8s8: k-block path supports only F32/F16 output");
    }
  }
#endif  // ARK_JOINT_MATRIX

  static void woq_s8(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c,
                     BTLA_DTYPE act, void* scale_b, void* bias, int blocksize) {

    int8_t *qa_ptr, *scalea_ptr;
    prepare_qa_and_quantize(q, m, k, a, act, blocksize, qa_ptr, scalea_ptr);
    igemm_s8s8(q, m, n, k, qa_ptr, b, BT, c, act, scalea_ptr, scale_b, bias, blocksize);
  }

};

}  // namespace ark

#endif  // ARK_XPU