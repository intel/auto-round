#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "../include/xpu_wrapper.hpp"

struct TestSDPA {
  TestSDPA() {
#if defined(ARK_XPU) && defined(ARK_SYCL_TLA) && ARK_XPU && ARK_SYCL_TLA
    // test_sagev1_accuracy("prefill_gqa_128_bias", 1, 32, 8, 256, 512, 128, 128, false, true);
    // test_sagev1_accuracy("prefill_gqa_128_no_bias", 1, 32, 8, 256, 512, 128, 128, false, false);
    benchmark_sagev1("bench_prefill_gqa_128", 1, 32, 8, 4096, 4096, 128, 128, false, 5, 20);
#else
    std::cout << "[sagev1] skipped: requires ARK_XPU and ARK_SYCL_TLA\n";
#endif
  }

#if defined(ARK_XPU) && defined(ARK_SYCL_TLA) && ARK_XPU && ARK_SYCL_TLA
  static std::vector<float> make_random_vector(size_t count, float low, float high, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> values(count);
    for (auto& value : values) {
      value = dist(rng);
    }
    return values;
  }

  static std::vector<sycl::half> to_fp16_vector(const std::vector<float>& src) {
    std::vector<sycl::half> dst(src.size());
    for (size_t index = 0; index < src.size(); ++index) {
      dst[index] = sycl::half(src[index]);
    }
    return dst;
  }

  static double compute_sdpa_flops(int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                                   int head_dim) {
    int group = num_heads_q / num_heads_kv;
    double qk = double(group) * num_heads_kv * seq_len_q * head_dim * seq_len_kv * 2.0;
    double softmax = double(group) * num_heads_kv * seq_len_q * seq_len_kv * 2.0;
    double pv = double(group) * num_heads_kv * seq_len_q * seq_len_kv * head_dim * 2.0;
    return double(batch) * (qk + softmax + pv);
  }

  static double compute_sdpa_io_bytes(int batch, int num_heads_q, int num_heads_kv, int seq_len_q, int seq_len_kv,
                                      int head_dim) {
    double q_bytes = double(batch) * num_heads_q * seq_len_q * head_dim * sizeof(sycl::half);
    double k_bytes = double(batch) * num_heads_kv * seq_len_kv * head_dim * sizeof(sycl::half);
    double v_bytes = double(batch) * num_heads_kv * seq_len_kv * head_dim * sizeof(sycl::half);
    double o_bytes = double(batch) * num_heads_q * seq_len_q * head_dim * sizeof(sycl::half);
    return q_bytes + k_bytes + v_bytes + o_bytes;
  }

  static double run_bench(const std::function<void()>& fn, sycl::queue* q, int warmup, int iters) {
    for (int iter = 0; iter < warmup; ++iter) {
      fn();
    }
    q->wait();
    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
      fn();
    }
    q->wait();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / double(iters);
  }

  void test_sagev1_accuracy(const std::string& name, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                            int seq_len_kv, int head_dim, int scale_block_size, bool is_causal,
                            bool use_mean_bias) {
    GETQ();
    LOG_LINE();
    size_t q_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
    size_t k_count = size_t(batch) * num_heads_kv * seq_len_kv * head_dim;
    size_t o_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
    size_t q_scale_count = size_t(batch) * num_heads_q * ((seq_len_q + scale_block_size - 1) / scale_block_size);
    size_t k_scale_count = size_t(batch) * num_heads_kv * ((seq_len_kv + scale_block_size - 1) / scale_block_size);
    size_t k_bias_count = size_t(batch) * num_heads_kv * head_dim;
    float softmax_scale = 1.0f / std::sqrt(float(head_dim));
    int previous_use_mean_bias = ark::env_params::Instance()->sage_use_mean_bias;
    ark::env_params::Instance()->sage_use_mean_bias = use_mean_bias ? 1 : 0;

    auto host_q = to_fp16_vector(make_random_vector(q_count, -1.0f, 1.0f, 601u + uint32_t(seq_len_q)));
    auto host_k = to_fp16_vector(make_random_vector(k_count, -1.0f, 1.0f, 701u + uint32_t(seq_len_kv)));
    auto host_v = to_fp16_vector(make_random_vector(k_count, -1.0f, 1.0f, 801u + uint32_t(head_dim)));

    auto* dev_q = reinterpret_cast<sycl::half*>(ctx->allocate(q_count * sizeof(sycl::half)));
    auto* dev_k = reinterpret_cast<sycl::half*>(ctx->allocate(k_count * sizeof(sycl::half)));
    auto* dev_v = reinterpret_cast<sycl::half*>(ctx->allocate(k_count * sizeof(sycl::half)));
    auto* dev_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_ref = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_qi8 = reinterpret_cast<int8_t*>(ctx->allocate(q_count * sizeof(int8_t)));
    auto* dev_ki8 = reinterpret_cast<int8_t*>(ctx->allocate(k_count * sizeof(int8_t)));
    auto* dev_qscale = reinterpret_cast<float*>(ctx->allocate(q_scale_count * sizeof(float)));
    auto* dev_kscale = reinterpret_cast<float*>(ctx->allocate(k_scale_count * sizeof(float)));
    auto* dev_kbias = use_mean_bias ? reinterpret_cast<sycl::half*>(ctx->allocate(k_bias_count * sizeof(sycl::half))) : nullptr;

    try {
      q->memcpy(dev_q, host_q.data(), q_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_k, host_k.data(), k_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_v, host_v.data(), k_count * sizeof(sycl::half)).wait();

      ark::XpuWrapper::sagev1(q, dev_q, dev_k, dev_v, dev_out, nullptr, scale_block_size, batch, num_heads_q,
                              num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);

      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_q, dev_qi8, dev_qscale, batch * num_heads_q, seq_len_q,
                                                      (seq_len_q + scale_block_size - 1) / scale_block_size, head_dim,
                                                      scale_block_size);
      if (use_mean_bias) {
        ark::XpuWrapper::compute_seq_mean_bias<sycl::half>(q, dev_k, dev_kbias, batch * num_heads_kv, seq_len_kv,
                                                           head_dim);
      }
      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_k, dev_ki8, dev_kscale, batch * num_heads_kv, seq_len_kv,
                                                      (seq_len_kv + scale_block_size - 1) / scale_block_size, head_dim,
                                                      scale_block_size, dev_kbias);
      ark::sdpa_impl_qks8_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_ref, nullptr, scale_block_size, dev_qscale,
                                 dev_kscale, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim,
                                 softmax_scale, is_causal);
      q->wait();

      std::vector<sycl::half> host_out(o_count);
      std::vector<sycl::half> host_ref(o_count);
      q->memcpy(host_out.data(), dev_out, o_count * sizeof(sycl::half)).wait();
      q->memcpy(host_ref.data(), dev_ref, o_count * sizeof(sycl::half)).wait();

      float max_diff = 0.0f;
      double mean_diff = 0.0;
      for (size_t index = 0; index < o_count; ++index) {
        float out_value = float(host_out[index]);
        float ref_value = float(host_ref[index]);
        if (!std::isfinite(out_value) || !std::isfinite(ref_value)) {
          throw std::runtime_error("sagev1 accuracy test produced non-finite output: " + name);
        }
        float diff = std::fabs(out_value - ref_value);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
      }
      mean_diff /= double(o_count);

      std::cout << std::fixed << std::setprecision(6) << "[sagev1][accuracy] " << name << " max_diff=" << max_diff
                << " mean_diff=" << mean_diff << " use_mean_bias=" << (use_mean_bias ? 1 : 0) << "\n";

      if (max_diff > 5e-3f || mean_diff > 5e-4f) {
        throw std::runtime_error("sagev1 accuracy mismatch: " + name);
      }
    } catch (...) {
      ark::env_params::Instance()->sage_use_mean_bias = previous_use_mean_bias;
      ctx->deallocate(dev_q);
      ctx->deallocate(dev_k);
      ctx->deallocate(dev_v);
      ctx->deallocate(dev_out);
      ctx->deallocate(dev_ref);
      ctx->deallocate(dev_qi8);
      ctx->deallocate(dev_ki8);
      ctx->deallocate(dev_qscale);
      ctx->deallocate(dev_kscale);
      ctx->deallocate(dev_kbias);
      throw;
    }

    ark::env_params::Instance()->sage_use_mean_bias = previous_use_mean_bias;

    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_out);
    ctx->deallocate(dev_ref);
    ctx->deallocate(dev_qi8);
    ctx->deallocate(dev_ki8);
    ctx->deallocate(dev_qscale);
    ctx->deallocate(dev_kscale);
    ctx->deallocate(dev_kbias);
  }

  void benchmark_sagev1(const std::string& name, int batch, int num_heads_q, int num_heads_kv, int seq_len_q,
                        int seq_len_kv, int head_dim, int scale_block_size, bool is_causal, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    size_t q_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
    size_t k_count = size_t(batch) * num_heads_kv * seq_len_kv * head_dim;
    size_t o_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
    float softmax_scale = 1.0f / std::sqrt(float(head_dim));

    auto host_q = to_fp16_vector(make_random_vector(q_count, -1.0f, 1.0f, 901u + uint32_t(seq_len_q)));
    auto host_k = to_fp16_vector(make_random_vector(k_count, -1.0f, 1.0f, 1001u + uint32_t(seq_len_kv)));
    auto host_v = to_fp16_vector(make_random_vector(k_count, -1.0f, 1.0f, 1101u + uint32_t(head_dim)));

    auto* dev_q = reinterpret_cast<sycl::half*>(ctx->allocate(q_count * sizeof(sycl::half)));
    auto* dev_k = reinterpret_cast<sycl::half*>(ctx->allocate(k_count * sizeof(sycl::half)));
    auto* dev_v = reinterpret_cast<sycl::half*>(ctx->allocate(k_count * sizeof(sycl::half)));
    auto* dev_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    q->memcpy(dev_q, host_q.data(), q_count * sizeof(sycl::half)).wait();
    q->memcpy(dev_k, host_k.data(), k_count * sizeof(sycl::half)).wait();
    q->memcpy(dev_v, host_v.data(), k_count * sizeof(sycl::half)).wait();

    double sage_ms = run_bench(
        [&]() {
          ark::XpuWrapper::sagev1(q, dev_q, dev_k, dev_v, dev_out, nullptr, scale_block_size, batch, num_heads_q,
                                  num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
        },
        q, warmup, iters);

    double flops = compute_sdpa_flops(batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim);
    double io_bytes = compute_sdpa_io_bytes(batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim);
    double sage_tflops = flops / (sage_ms * 1e-3) / 1e12;
    double sage_gbps = io_bytes / (sage_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(3) << "[sagev1][bench] " << name << " sage_ms=" << sage_ms
              << " sage_TFLOPS=" << sage_tflops << " sage_GBps=" << sage_gbps << "\n";

    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_out);
  }
#endif
};