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
  static inline void profile_woq_s8_record(int m, int n, int k, BTLA_DTYPE act, int blocksize,
                                           int64_t quant_ns, int64_t igemm_ns, int64_t total_ns) {
    static std::atomic<int64_t> calls{0};
    static std::atomic<int64_t> quant_total_ns{0};
    static std::atomic<int64_t> igemm_total_ns{0};
    static std::atomic<int64_t> total_total_ns{0};

    int64_t call = calls.fetch_add(1, std::memory_order_relaxed) + 1;
    int64_t quant_total = quant_total_ns.fetch_add(quant_ns, std::memory_order_relaxed) + quant_ns;
    int64_t igemm_total = igemm_total_ns.fetch_add(igemm_ns, std::memory_order_relaxed) + igemm_ns;
    int64_t total = total_total_ns.fetch_add(total_ns, std::memory_order_relaxed) + total_ns;

    int interval = env_params::Instance()->profile_woq_interval;
    if (interval <= 0) interval = 1000;
    if (call % interval != 0) return;

    int64_t other_ns = total_ns - quant_ns - igemm_ns;
    if (other_ns < 0) other_ns = 0;
    double quant_ms = static_cast<double>(quant_total) / 1.0e6;
    double igemm_ms = static_cast<double>(igemm_total) / 1.0e6;
    double total_ms = static_cast<double>(total) / 1.0e6;
    double quant_pct = total > 0 ? static_cast<double>(quant_total) * 100.0 / static_cast<double>(total) : 0.0;
    double igemm_pct = total > 0 ? static_cast<double>(igemm_total) * 100.0 / static_cast<double>(total) : 0.0;
    double avg_quant_us = static_cast<double>(quant_total) / call / 1.0e3;
    double avg_igemm_us = static_cast<double>(igemm_total) / call / 1.0e3;
    double avg_total_us = static_cast<double>(total) / call / 1.0e3;
    double last_quant_us = static_cast<double>(quant_ns) / 1.0e3;
    double last_igemm_us = static_cast<double>(igemm_ns) / 1.0e3;
    double last_other_us = static_cast<double>(other_ns) / 1.0e3;
    double last_total_us = static_cast<double>(total_ns) / 1.0e3;
    double last_quant_pct = total_ns > 0 ? static_cast<double>(quant_ns) * 100.0 / static_cast<double>(total_ns) : 0.0;
    double last_igemm_pct = total_ns > 0 ? static_cast<double>(igemm_ns) * 100.0 / static_cast<double>(total_ns) : 0.0;
    double ops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    double last_igemm_tflops = igemm_ns > 0 ? ops / (static_cast<double>(igemm_ns) * 1.0e3) : 0.0;
    double last_total_tflops = total_ns > 0 ? ops / (static_cast<double>(total_ns) * 1.0e3) : 0.0;

    std::fprintf(stderr,
                 "[ARK_PROFILE_WOQ_S8] calls=%ld last_mnk=(%d,%d,%d) act=%s(%d) blocksize=%d "
                 "quant_total=%.3fms igemm_total=%.3fms total=%.3fms quant_pct=%.2f igemm_pct=%.2f "
                 "avg_quant=%.3fus avg_igemm=%.3fus avg_total=%.3fus "
                 "last_quant=%.3fus last_igemm=%.3fus last_other=%.3fus last_total=%.3fus "
                 "last_quant_pct=%.2f last_igemm_pct=%.2f last_igemm_tflops=%.3f last_total_tflops=%.3f\n",
                 call, m, n, k, bestla::utils::bestla_dtype_str(act), static_cast<int>(act), blocksize,
                 quant_ms, igemm_ms, total_ms, quant_pct, igemm_pct, avg_quant_us, avg_igemm_us, avg_total_us,
                 last_quant_us, last_igemm_us, last_other_us, last_total_us, last_quant_pct, last_igemm_pct,
                 last_igemm_tflops, last_total_tflops);
  }

  static inline void prepare_qa_and_quantize(sycl::queue* q, int m, int k, const void* a, BTLA_DTYPE act,
                                             int8_t*& qa_ptr, int8_t*& scalea_ptr) {
    size_t qa_size = size_t(m) * size_t(k);
    size_t scalea_offset = (qa_size + alignof(float) - 1) & ~(size_t(alignof(float)) - 1);
    size_t tmp_size = scalea_offset + size_t(m) * sizeof(float);

    auto tmp_ptr = static_cast<int8_t*>(DeviceMemoryPool::Instance()->get_scratch_mem(tmp_size, 1, q));
    qa_ptr = tmp_ptr;
    scalea_ptr = tmp_ptr + scalea_offset;

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

    bool profile_woq = env_params::Instance()->profile_woq != 0;
    int64_t profile_start_ns = 0;
    int64_t profile_quant_ns = 0;
    int64_t profile_igemm_ns = 0;
    if (profile_woq && q != nullptr) {
      q->wait();
      profile_start_ns = DeviceMemoryPool::profile_now_ns();
    }

    int8_t *qa_ptr, *scalea_ptr;
    int64_t profile_quant_start_ns = 0;
    if (profile_woq && q != nullptr) {
      profile_quant_start_ns = DeviceMemoryPool::profile_now_ns();
    }
    prepare_qa_and_quantize(q, m, k, a, act, qa_ptr, scalea_ptr);
    if (profile_woq && q != nullptr) {
      q->wait();
      profile_quant_ns = DeviceMemoryPool::profile_now_ns() - profile_quant_start_ns;
    }

    int64_t profile_igemm_start_ns = 0;
    if (profile_woq && q != nullptr) {
      profile_igemm_start_ns = DeviceMemoryPool::profile_now_ns();
    }
    igemm_s8s8(q, m, n, k, qa_ptr, b, BT, c, act, scalea_ptr, scale_b, bias, blocksize);
    if (profile_woq && q != nullptr) {
      q->wait();
      profile_igemm_ns = DeviceMemoryPool::profile_now_ns() - profile_igemm_start_ns;
      profile_woq_s8_record(m, n, k, act, blocksize, profile_quant_ns, profile_igemm_ns,
                            DeviceMemoryPool::profile_now_ns() - profile_start_ns);
    }
  }

};

}  // namespace ark

#endif  // ARK_XPU