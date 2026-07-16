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
#include "sycl_tla_s8_gemm.hpp"
#endif

namespace ark {

class SyclS8Wrapper {
 public:
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

#if ARK_SYCL_TLA
    ark::sycl_tla_igemm_s8s8_dequant(q, m, n, k, a, b, c, ct, scale_a, scale_b, bias, blocksize);
#else
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
#endif
  }

  static void woq_s8(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c,
                     BTLA_DTYPE act, void* scale_b, void* bias, int blocksize) {
    size_t qa_size = size_t(m) * size_t(k);
    size_t scalea_offset = (qa_size + alignof(float) - 1) & ~(size_t(alignof(float)) - 1);
    size_t tmp_size = scalea_offset + size_t(m) * sizeof(float);

    auto tmp_ptr = static_cast<int8_t*>(DeviceMemoryPool::Instance()->get_scratch_mem(tmp_size, 1, q));
    auto qa_ptr = tmp_ptr;
    auto scalea_ptr = tmp_ptr + scalea_offset;

    dyn_quant_s8(q, m, k, a, act, qa_ptr, scalea_ptr, 0);
    igemm_s8s8(q, m, n, k, qa_ptr, b, BT, c, act, scalea_ptr, scale_b, bias, blocksize);
  }
};

}  // namespace ark

#endif  // ARK_XPU