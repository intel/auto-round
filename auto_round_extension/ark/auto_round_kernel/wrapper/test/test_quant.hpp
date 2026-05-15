#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "../include/xpu_wrapper.hpp"

struct TestQuant {
  TestQuant() {
#ifdef ARK_XPU
    test_compute_seq_mean_bias_accuracy("k_mean_bias", 3, 111, 128);
    test_compute_seq_mean_bias_perf("k_mean_bias_bw", 16, 2048, 128, 20, 100);
    test_sage_dynamic_quant_accuracy("no_bias_small_block", 4, 128, 128, 64, false);
    test_sage_dynamic_quant_accuracy("no_bias_unaligned_seq", 5, 96, 128, 64, false);
    test_sage_dynamic_quant_accuracy("bias_large_block", 3, 512, 128, 256, true);
    test_sage_dynamic_quant_accuracy("bias_unaligned_seq", 3, 112, 128, 64, true);
    test_sage_dynamic_quant_perf("bw_no_bias", 16, 2048, 128, 64, false, 20, 100);
    test_sage_dynamic_quant_perf("bw_with_bias", 16, 2048, 128, 64, true, 20, 100);
#else
    std::cout << "[sage_dynamic_quant] skipped: ARK_XPU is disabled\n";
#endif
  }

#ifdef ARK_XPU
  static int ceil_div(int value, int divisor) { return (value + divisor - 1) / divisor; }

  static sycl::half f32_to_f16(float value) { return sycl::half(value); }

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
      dst[index] = f32_to_f16(src[index]);
    }
    return dst;
  }

  static std::vector<float> to_f32_vector(const std::vector<sycl::half>& src) {
    std::vector<float> dst(src.size());
    for (size_t index = 0; index < src.size(); ++index) {
      dst[index] = static_cast<float>(src[index]);
    }
    return dst;
  }

  static std::vector<float> reference_seq_mean_bias(const std::vector<float>& input, int num_rows, int seq,
                                                    int head_dim) {
    std::vector<float> bias(size_t(num_rows) * size_t(head_dim), 0.0f);
    for (int row_id = 0; row_id < num_rows; ++row_id) {
      for (int dim = 0; dim < head_dim; ++dim) {
        float sum = 0.0f;
        for (int token = 0; token < seq; ++token) {
          size_t offset = (size_t(row_id) * seq + size_t(token)) * head_dim + size_t(dim);
          sum += input[offset];
        }
        bias[size_t(row_id) * head_dim + size_t(dim)] = sum / float(seq);
      }
    }
    return bias;
  }

  void test_compute_seq_mean_bias_accuracy(const std::string& name, int num_rows, int seq, int head_dim) {
    GETQ();
    LOG_LINE();
    size_t input_count = size_t(num_rows) * size_t(seq) * size_t(head_dim);
    size_t bias_count = size_t(num_rows) * size_t(head_dim);

    auto host_input_f32 = make_random_vector(input_count, -2.0f, 2.0f, 2025u + uint32_t(seq));
    auto host_input = to_fp16_vector(host_input_f32);
    auto ref_input_f32 = to_f32_vector(host_input);
    auto ref_bias = reference_seq_mean_bias(ref_input_f32, num_rows, seq, head_dim);

    auto* dev_input = reinterpret_cast<sycl::half*>(ctx->allocate(input_count * sizeof(sycl::half)));
    auto* dev_bias = reinterpret_cast<sycl::half*>(ctx->allocate(bias_count * sizeof(sycl::half)));

    q->memcpy(dev_input, host_input.data(), input_count * sizeof(sycl::half)).wait();
    ark::XpuWrapper::compute_seq_mean_bias<sycl::half>(q, dev_input, dev_bias, num_rows, seq, head_dim);
    q->wait();

    std::vector<sycl::half> host_bias(bias_count);
    q->memcpy(host_bias.data(), dev_bias, bias_count * sizeof(sycl::half)).wait();

    float max_diff = 0.0f;
    for (size_t index = 0; index < bias_count; ++index) {
      max_diff = std::max(max_diff, std::fabs(static_cast<float>(host_bias[index]) - ref_bias[index]));
    }

    std::cout << "[compute_seq_mean_bias][accuracy] " << name << " max_diff=" << max_diff << "\n";
    if (max_diff > 1e-3f) {
      throw std::runtime_error("compute_seq_mean_bias accuracy check failed: " + name);
    }

    ctx->deallocate(dev_input);
    ctx->deallocate(dev_bias);
  }

  void test_compute_seq_mean_bias_perf(const std::string& name, int num_rows, int seq, int head_dim, int warmup,
                                       int iters) {
    GETQ();
    LOG_LINE();
    size_t input_count = size_t(num_rows) * size_t(seq) * size_t(head_dim);
    size_t bias_count = size_t(num_rows) * size_t(head_dim);

    auto host_input = to_fp16_vector(make_random_vector(input_count, -2.0f, 2.0f, 5051u));

    auto* dev_input = reinterpret_cast<sycl::half*>(ctx->allocate(input_count * sizeof(sycl::half)));
    auto* dev_bias = reinterpret_cast<sycl::half*>(ctx->allocate(bias_count * sizeof(sycl::half)));

    q->memcpy(dev_input, host_input.data(), input_count * sizeof(sycl::half)).wait();

    for (int iter = 0; iter < warmup; ++iter) {
      ark::XpuWrapper::compute_seq_mean_bias<sycl::half>(q, dev_input, dev_bias, num_rows, seq, head_dim);
    }
    q->wait();

    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
      ark::XpuWrapper::compute_seq_mean_bias<sycl::half>(q, dev_input, dev_bias, num_rows, seq, head_dim);
    }
    q->wait();
    auto end = std::chrono::steady_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / double(iters);
    double input_bytes = double(input_count * sizeof(sycl::half));
    double output_bytes = double(bias_count * sizeof(sycl::half));
    double logical_io_bytes = input_bytes + output_bytes;
    double estimated_kernel_bytes = input_bytes + output_bytes;
    double logical_io_gbps = logical_io_bytes / (avg_ms * 1e-3) / 1e9;
    double estimated_kernel_gbps = estimated_kernel_bytes / (avg_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(3) << "[compute_seq_mean_bias][perf] " << name
              << " avg_ms=" << avg_ms << " input_MB=" << (input_bytes / (1024.0 * 1024.0))
              << " output_MB=" << (output_bytes / (1024.0 * 1024.0))
          << " logical_io_GBps=" << logical_io_gbps << " est_kernel_GBps=" << estimated_kernel_gbps << "\n";

    ctx->deallocate(dev_input);
    ctx->deallocate(dev_bias);
  }

  static void reference_quant(const std::vector<float>& input, const std::vector<float>* bias, int num_rows, int seq,
                              int head_dim, int block_size, std::vector<int8_t>& output, std::vector<float>& scales) {
    int n_seq_blk = ceil_div(seq, block_size);
    output.assign(input.size(), int8_t(0));
    scales.assign(size_t(num_rows) * size_t(n_seq_blk), 0.0f);
    for (int row_id = 0; row_id < num_rows; ++row_id) {
      for (int seq_blk = 0; seq_blk < n_seq_blk; ++seq_blk) {
        int seq_begin = seq_blk * block_size;
        int seq_end = std::min(seq_begin + block_size, seq);
        float absmax = 0.0f;
        for (int token = seq_begin; token < seq_end; ++token) {
          for (int dim = 0; dim < head_dim; ++dim) {
            size_t offset = (size_t(row_id) * seq + size_t(token)) * head_dim + size_t(dim);
            float value = input[offset];
            if (bias) {
              value -= (*bias)[size_t(row_id) * head_dim + size_t(dim)];
            }
            absmax = std::max(absmax, std::fabs(value));
          }
        }
        scales[size_t(row_id) * n_seq_blk + size_t(seq_blk)] = absmax / 127.0f;
        float inv_scale = absmax > 0.0f ? 127.0f / absmax : 0.0f;
        for (int token = seq_begin; token < seq_end; ++token) {
          for (int dim = 0; dim < head_dim; ++dim) {
            size_t offset = (size_t(row_id) * seq + size_t(token)) * head_dim + size_t(dim);
            float value = input[offset];
            if (bias) {
              value -= (*bias)[size_t(row_id) * head_dim + size_t(dim)];
            }
            value *= inv_scale;
            int quantized = static_cast<int>(std::round(value));
            quantized = std::max(-127, std::min(127, quantized));
            output[offset] = static_cast<int8_t>(quantized);
          }
        }
      }
    }
  }

  static void validate_results(const std::string& name, const std::vector<int8_t>& actual_q,
                               const std::vector<int8_t>& ref_q, const std::vector<float>& actual_scales,
                               const std::vector<float>& ref_scales) {
    size_t q_mismatches = 0;
    int max_q_diff = 0;
    for (size_t index = 0; index < actual_q.size(); ++index) {
      int diff = std::abs(int(actual_q[index]) - int(ref_q[index]));
      max_q_diff = std::max(max_q_diff, diff);
      if (diff != 0) {
        ++q_mismatches;
      }
    }

    float max_scale_diff = 0.0f;
    for (size_t index = 0; index < actual_scales.size(); ++index) {
      max_scale_diff = std::max(max_scale_diff, std::fabs(actual_scales[index] - ref_scales[index]));
    }

    std::cout << "[sage_dynamic_quant][accuracy] " << name << " q_mismatches=" << q_mismatches
              << " max_q_diff=" << max_q_diff << " max_scale_diff=" << max_scale_diff << "\n";

    if (max_q_diff > 1 || max_scale_diff > 1e-4f) {
      throw std::runtime_error("sage_dynamic_quant accuracy check failed: " + name);
    }
  }

  void test_sage_dynamic_quant_accuracy(const std::string& name, int num_rows, int seq, int head_dim, int block_size,
                                        bool use_bias) {
    GETQ();
    LOG_LINE();
    int n_seq_blk = ceil_div(seq, block_size);
    size_t input_count = size_t(num_rows) * size_t(seq) * size_t(head_dim);
    size_t bias_count = size_t(num_rows) * size_t(head_dim);
    size_t scale_count = size_t(num_rows) * size_t(n_seq_blk);

    auto host_input_f32 = make_random_vector(input_count, -2.0f, 2.0f, 2026u + uint32_t(block_size));
    auto host_input = to_fp16_vector(host_input_f32);
    auto ref_input_f32 = to_f32_vector(host_input);
    std::vector<float> host_bias_f32;
    std::vector<sycl::half> host_bias;
    std::vector<float> ref_bias_f32;
    if (use_bias) {
      host_bias_f32 = make_random_vector(bias_count, -0.5f, 0.5f, 3039u + uint32_t(num_rows));
      host_bias = to_fp16_vector(host_bias_f32);
      ref_bias_f32 = to_f32_vector(host_bias);
    }

    auto* dev_input = reinterpret_cast<sycl::half*>(ctx->allocate(input_count * sizeof(sycl::half)));
    auto* dev_bias = use_bias ? reinterpret_cast<sycl::half*>(ctx->allocate(bias_count * sizeof(sycl::half))) : nullptr;
    auto* dev_output = reinterpret_cast<int8_t*>(ctx->allocate(input_count * sizeof(int8_t)));
    auto* dev_scales = reinterpret_cast<float*>(ctx->allocate(scale_count * sizeof(float)));

    q->memcpy(dev_input, host_input.data(), input_count * sizeof(sycl::half)).wait();
    if (use_bias) {
      q->memcpy(dev_bias, host_bias.data(), bias_count * sizeof(sycl::half)).wait();
    }

    ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_input, dev_output, dev_scales, num_rows, seq, n_seq_blk,
                                                    head_dim, block_size, dev_bias);
    q->wait();

    std::vector<int8_t> host_output(input_count);
    std::vector<float> host_scales(scale_count);
    q->memcpy(host_output.data(), dev_output, input_count * sizeof(int8_t)).wait();
    q->memcpy(host_scales.data(), dev_scales, scale_count * sizeof(float)).wait();

    std::vector<int8_t> ref_output;
    std::vector<float> ref_scales;
    reference_quant(ref_input_f32, use_bias ? &ref_bias_f32 : nullptr, num_rows, seq, head_dim, block_size,
                    ref_output, ref_scales);
    validate_results(name, host_output, ref_output, host_scales, ref_scales);

    ctx->deallocate(dev_input);
    ctx->deallocate(dev_bias);
    ctx->deallocate(dev_output);
    ctx->deallocate(dev_scales);
  }

  void test_sage_dynamic_quant_perf(const std::string& name, int num_rows, int seq, int head_dim, int block_size,
                                    bool use_bias, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    int n_seq_blk = ceil_div(seq, block_size);
    size_t input_count = size_t(num_rows) * size_t(seq) * size_t(head_dim);
    size_t bias_count = size_t(num_rows) * size_t(head_dim);
    size_t scale_count = size_t(num_rows) * size_t(n_seq_blk);

    auto host_input = to_fp16_vector(make_random_vector(input_count, -2.0f, 2.0f, 4041u));
    std::vector<sycl::half> host_bias;
    if (use_bias) {
      host_bias = to_fp16_vector(make_random_vector(bias_count, -0.25f, 0.25f, 4042u));
    }

    auto* dev_input = reinterpret_cast<sycl::half*>(ctx->allocate(input_count * sizeof(sycl::half)));
    auto* dev_bias = use_bias ? reinterpret_cast<sycl::half*>(ctx->allocate(bias_count * sizeof(sycl::half))) : nullptr;
    auto* dev_output = reinterpret_cast<int8_t*>(ctx->allocate(input_count * sizeof(int8_t)));
    auto* dev_scales = reinterpret_cast<float*>(ctx->allocate(scale_count * sizeof(float)));

    q->memcpy(dev_input, host_input.data(), input_count * sizeof(sycl::half)).wait();
    if (use_bias) {
      q->memcpy(dev_bias, host_bias.data(), bias_count * sizeof(sycl::half)).wait();
    }

    for (int iter = 0; iter < warmup; ++iter) {
      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_input, dev_output, dev_scales, num_rows, seq, n_seq_blk,
                                                      head_dim, block_size, dev_bias);
    }
    q->wait();

    auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_input, dev_output, dev_scales, num_rows, seq, n_seq_blk,
                                                      head_dim, block_size, dev_bias);
    }
    q->wait();
    auto end = std::chrono::steady_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / double(iters);
    double input_bytes = double(input_count * sizeof(sycl::half)) + double(use_bias ? bias_count * sizeof(sycl::half) : 0);
    double output_bytes = double(input_count * sizeof(int8_t)) + double(scale_count * sizeof(float));
    double logical_io_bytes = input_bytes + output_bytes;
    double estimated_kernel_bytes =
        double(input_count * sizeof(sycl::half) * 2) +
        double(use_bias ? bias_count * sizeof(sycl::half) * 2 : 0) + output_bytes;
    double logical_io_gbps = logical_io_bytes / (avg_ms * 1e-3) / 1e9;
    double estimated_kernel_gbps = estimated_kernel_bytes / (avg_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(3)
              << "[sage_dynamic_quant][perf] " << name << " avg_ms=" << avg_ms << " input_MB="
              << (input_bytes / (1024.0 * 1024.0)) << " output_MB=" << (output_bytes / (1024.0 * 1024.0))
              << " logical_io_GBps=" << logical_io_gbps << " est_kernel_GBps=" << estimated_kernel_gbps << "\n";

    ctx->deallocate(dev_input);
    ctx->deallocate(dev_bias);
    ctx->deallocate(dev_output);
    ctx->deallocate(dev_scales);
  }
#endif
};