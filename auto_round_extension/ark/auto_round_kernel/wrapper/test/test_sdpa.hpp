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
    test_sparse_sage_matches_dense("sparse_all_selected_h64", 1, 4, 4, 128, 128, 64, 64, false);
    test_sparse_sage_matches_dense("sparse_all_selected_h128", 1, 4, 4, 128, 128, 128, 64, false);
    test_sparse_sage_partial_prefill("sparse_partial_prefill_h64", 1, 4, 4, 256, 64, 64);
    test_sparse_sage_partial_prefill("sparse_partial_prefill_h128", 1, 4, 4, 256, 128, 64);
    test_sparse_sage_partial_prefill_causal("sparse_partial_prefill_causal_h64", 1, 4, 4, 256, 64, 64);
    test_sparse_sage_partial_prefill_causal("sparse_partial_prefill_causal_h128", 1, 4, 4, 256, 128, 64);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_035_h64", 1, 4, 4, 384, 64, 64, {0, 3, 5}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_02_h64", 1, 4, 4, 384, 64, 64, {0, 2}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_135_h64", 1, 4, 4, 384, 64, 64, {1, 3, 5}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_014_h64", 1, 4, 4, 384, 64, 64, {0, 1, 4}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_035_h128", 1, 4, 4, 384, 128, 64, {0, 3, 5}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_02_h128", 1, 4, 4, 384, 128, 64, {0, 2}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_135_h128", 1, 4, 4, 384, 128, 64, {1, 3, 5}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_014_h128", 1, 4, 4, 384, 128, 64, {0, 1, 4}, false);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_causal_035_h64", 1, 4, 4, 384, 64, 64, {0, 3, 5}, true);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_causal_02_h64", 1, 4, 4, 384, 64, 64, {0, 2}, true);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_causal_014_h64", 1, 4, 4, 384, 64, 64, {0, 1, 4}, true);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_causal_035_h128", 1, 4, 4, 384, 128, 64, {0, 3, 5}, true);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_causal_02_h128", 1, 4, 4, 384, 128, 64, {0, 2}, true);
    test_sparse_sage_noncontiguous_prefill("sparse_noncontig_causal_014_h128", 1, 4, 4, 384, 128, 64, {0, 1, 4}, true);
    // test_sagev1_accuracy("prefill_gqa_128_bias", 1, 32, 8, 256, 512, 128, 128, false, true);
    // test_sagev1_accuracy("prefill_gqa_128_no_bias", 1, 32, 8, 256, 512, 128, 128, false, false);
    // test_sagev1_accuracy(\"prefill_gqa_128_causal_no_bias\", 1, 32, 8, 256, 512, 128, 128, true, false);
    benchmark_dynamic_quant("dq_b1_h32_s4096_d128", 1, 32, 4096, 128, 128, 5, 50);
    benchmark_dynamic_quant("dq_b1_h96_s8192_d128", 1, 96, 8192, 128, 128, 5, 50);
    // benchmark_sagev1("bench_prefill_gqa_128_s4096", 1, 32, 8, 4096, 4096, 128, 128, false, 5, 20);
    // benchmark_sagev1("bench_bmg_hq96_d128_k8192", 1, 96, 8, 4096, 8192, 128, 128, false, 5, 20);
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

  static void build_all_selected_lut(int batch, int num_heads_q, int q_blocks, int kv_blocks,
                                     std::vector<int>& lut, std::vector<int>& valid_block_num) {
    lut.assign(size_t(batch) * num_heads_q * q_blocks * kv_blocks, 0);
    valid_block_num.assign(size_t(batch) * num_heads_q * q_blocks, kv_blocks);
    for (int b = 0; b < batch; ++b) {
      for (int h = 0; h < num_heads_q; ++h) {
        for (int qblk = 0; qblk < q_blocks; ++qblk) {
          size_t row_offset = (size_t(b) * num_heads_q * q_blocks + size_t(h) * q_blocks + size_t(qblk)) * kv_blocks;
          if (kv_blocks > 0) {
            lut[row_offset] = 0;
            for (int kblk = 1; kblk < kv_blocks; ++kblk) {
              lut[row_offset + size_t(kblk)] = 1;
            }
          }
        }
      }
    }
  }

  static void build_prefill_sparse_lut_and_mask(int batch, int num_heads_q, int seq_len_q, int seq_len_kv,
                                                int q_blocks, int kv_blocks, int scale_block_size,
                                                int query_tile_tokens,
                                                const std::vector<std::vector<int>>& per_query_tile_selection,
                                                std::vector<int>& lut, std::vector<int>& valid_block_num,
                                                std::vector<float>& mask, bool is_causal = false) {
    int active_query_tiles = (seq_len_q + query_tile_tokens - 1) / query_tile_tokens;
    if (int(per_query_tile_selection.size()) != active_query_tiles) {
      throw std::invalid_argument("per_query_tile_selection size must match active query tiles");
    }
    lut.assign(size_t(batch) * num_heads_q * q_blocks * kv_blocks, 0);
    valid_block_num.assign(size_t(batch) * num_heads_q * q_blocks, 0);
    mask.assign(size_t(batch) * seq_len_q * seq_len_kv, -1.0e9f);

    for (int qtile = 0; qtile < active_query_tiles; ++qtile) {
      int previous = 0;
      bool first = true;
      for (int selected : per_query_tile_selection[qtile]) {
        if (selected < 0 || selected >= kv_blocks) {
          throw std::invalid_argument("selected sparse block index is out of range");
        }
        if (!first && selected <= previous) {
          throw std::invalid_argument("selected sparse block indices must be strictly increasing");
        }
        previous = selected;
        first = false;
      }
    }

    for (int b = 0; b < batch; ++b) {
      for (int h = 0; h < num_heads_q; ++h) {
        for (int qblk = 0; qblk < q_blocks; ++qblk) {
          int qtile = std::min(qblk, active_query_tiles - 1);
          size_t row_index = size_t(b) * num_heads_q * q_blocks + size_t(h) * q_blocks + size_t(qblk);
          size_t row_offset = row_index * kv_blocks;
          int previous = 0;
          bool first = true;
          int valid = int(per_query_tile_selection[qtile].size());
          valid_block_num[row_index] = valid;
          for (int i = 0; i < valid; ++i) {
            int selected = per_query_tile_selection[qtile][i];
            lut[row_offset + size_t(i)] = first ? selected : (selected - previous);
            previous = selected;
            first = false;
          }
        }

        for (int qtile = 0; qtile < active_query_tiles; ++qtile) {
          int q_start = qtile * query_tile_tokens;
          int q_end = std::min(q_start + query_tile_tokens, seq_len_q);
          for (int qt = q_start; qt < q_end; ++qt) {
            size_t mask_row = (size_t(b) * seq_len_q + size_t(qt)) * seq_len_kv;
            for (int selected : per_query_tile_selection[qtile]) {
              int k_start = selected * scale_block_size;
              int k_end = std::min(k_start + scale_block_size, seq_len_kv);
              if (!is_causal) {
                std::fill(mask.begin() + mask_row + k_start, mask.begin() + mask_row + k_end, 0.0f);
              } else {
                int visible_end = std::min(k_end, qt + 1);
                if (visible_end > k_start) {
                  std::fill(mask.begin() + mask_row + k_start, mask.begin() + mask_row + visible_end, 0.0f);
                }
              }
            }
          }
        }
      }
    }
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

  static void test_sparse_sage_matches_dense(const std::string& name, int batch, int num_heads_q, int num_heads_kv,
                                             int seq_len_q, int seq_len_kv, int head_dim, int scale_block_size,
                                             bool is_causal) {
    GETQ();
    LOG_LINE();
    size_t q_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
    size_t kv_count = size_t(batch) * num_heads_kv * seq_len_kv * head_dim;
    size_t o_count = size_t(batch) * num_heads_q * seq_len_q * head_dim;
    int q_blocks = (seq_len_q + scale_block_size - 1) / scale_block_size;
    int kv_blocks = (seq_len_kv + scale_block_size - 1) / scale_block_size;
    size_t q_scale_count = size_t(batch) * num_heads_q * q_blocks;
    size_t k_scale_count = size_t(batch) * num_heads_kv * kv_blocks;
    size_t lut_count = size_t(batch) * num_heads_q * q_blocks * kv_blocks;
    size_t valid_count = size_t(batch) * num_heads_q * q_blocks;
    float softmax_scale = 1.0f / std::sqrt(float(head_dim));

    auto host_q = to_fp16_vector(make_random_vector(q_count, -1.0f, 1.0f, 1201u + uint32_t(head_dim)));
    auto host_k = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 1301u + uint32_t(seq_len_kv)));
    auto host_v = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 1401u + uint32_t(seq_len_q)));
    std::vector<int> host_lut;
    std::vector<int> host_valid;
    build_all_selected_lut(batch, num_heads_q, q_blocks, kv_blocks, host_lut, host_valid);

    auto* dev_q = reinterpret_cast<sycl::half*>(ctx->allocate(q_count * sizeof(sycl::half)));
    auto* dev_k = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_v = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_qi8 = reinterpret_cast<int8_t*>(ctx->allocate(q_count * sizeof(int8_t)));
    auto* dev_ki8 = reinterpret_cast<int8_t*>(ctx->allocate(kv_count * sizeof(int8_t)));
    auto* dev_qscale = reinterpret_cast<float*>(ctx->allocate(q_scale_count * sizeof(float)));
    auto* dev_kscale = reinterpret_cast<float*>(ctx->allocate(k_scale_count * sizeof(float)));
    auto* dev_dense_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_sparse_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_lut = reinterpret_cast<int*>(ctx->allocate(lut_count * sizeof(int)));
    auto* dev_valid = reinterpret_cast<int*>(ctx->allocate(valid_count * sizeof(int)));

    int q_stride_s = head_dim;
    int q_stride_d = 1;
    int q_stride_h = seq_len_q * head_dim;
    int q_stride_b = num_heads_q * seq_len_q * head_dim;
    int k_stride_s = head_dim;
    int k_stride_d = 1;
    int k_stride_h = seq_len_kv * head_dim;
    int k_stride_b = num_heads_kv * seq_len_kv * head_dim;
    int v_stride_d = 1;
    int v_stride_s = head_dim;
    int v_stride_h = seq_len_kv * head_dim;
    int v_stride_b = num_heads_kv * seq_len_kv * head_dim;
    int o_stride_s = head_dim;
    int o_stride_d = 1;
    int o_stride_h = seq_len_q * head_dim;
    int o_stride_b = num_heads_q * seq_len_q * head_dim;

    try {
      q->memcpy(dev_q, host_q.data(), q_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_k, host_k.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_v, host_v.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_lut, host_lut.data(), lut_count * sizeof(int)).wait();
      q->memcpy(dev_valid, host_valid.data(), valid_count * sizeof(int)).wait();

      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_q, dev_qi8, dev_qscale, batch * num_heads_q, seq_len_q,
                                                      q_blocks, head_dim, scale_block_size);
      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_k, dev_ki8, dev_kscale, batch * num_heads_kv, seq_len_kv,
                                                      kv_blocks, head_dim, scale_block_size);

      ark::sdpa_impl_qks8_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_dense_out, nullptr, scale_block_size, dev_qscale,
                                 dev_kscale, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                                 k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                 o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len_q,
                                 seq_len_kv, head_dim, softmax_scale, is_causal);

      ark::sdpa_impl_qks8_sparse_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, scale_block_size,
                                        dev_qscale, dev_kscale, dev_lut, dev_valid, q_blocks, kv_blocks, q_stride_s,
                                        q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                                        k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                        o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv,
                                        seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
      q->wait();

      std::vector<sycl::half> host_dense(o_count);
      std::vector<sycl::half> host_sparse(o_count);
      q->memcpy(host_dense.data(), dev_dense_out, o_count * sizeof(sycl::half)).wait();
      q->memcpy(host_sparse.data(), dev_sparse_out, o_count * sizeof(sycl::half)).wait();

      float max_diff = 0.0f;
      double mean_diff = 0.0;
      for (size_t index = 0; index < o_count; ++index) {
        float dense_value = float(host_dense[index]);
        float sparse_value = float(host_sparse[index]);
        if (!std::isfinite(dense_value) || !std::isfinite(sparse_value)) {
          throw std::runtime_error("sparse sage test produced non-finite output: " + name);
        }
        float diff = std::fabs(dense_value - sparse_value);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
      }
      mean_diff /= double(o_count);
      std::cout << std::fixed << std::setprecision(6) << "[sage_sparse][accuracy] " << name
                << " max_diff=" << max_diff << " mean_diff=" << mean_diff << "\n";
      if (max_diff > 5e-3f || mean_diff > 5e-4f) {
        throw std::runtime_error("sparse sage mismatch: " + name);
      }
    } catch (...) {
      ctx->deallocate(dev_q);
      ctx->deallocate(dev_k);
      ctx->deallocate(dev_v);
      ctx->deallocate(dev_qi8);
      ctx->deallocate(dev_ki8);
      ctx->deallocate(dev_qscale);
      ctx->deallocate(dev_kscale);
      ctx->deallocate(dev_dense_out);
      ctx->deallocate(dev_sparse_out);
      ctx->deallocate(dev_lut);
      ctx->deallocate(dev_valid);
      throw;
    }

    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_qi8);
    ctx->deallocate(dev_ki8);
    ctx->deallocate(dev_qscale);
    ctx->deallocate(dev_kscale);
    ctx->deallocate(dev_dense_out);
    ctx->deallocate(dev_sparse_out);
    ctx->deallocate(dev_lut);
    ctx->deallocate(dev_valid);
  }

  static void test_sparse_sage_partial_prefill(const std::string& name, int batch, int num_heads_q, int num_heads_kv,
                                               int seq_len, int head_dim, int scale_block_size) {
    GETQ();
    LOG_LINE();
    size_t q_count = size_t(batch) * num_heads_q * seq_len * head_dim;
    size_t kv_count = size_t(batch) * num_heads_kv * seq_len * head_dim;
    size_t o_count = size_t(batch) * num_heads_q * seq_len * head_dim;
    int q_blocks = (seq_len + scale_block_size - 1) / scale_block_size;
    int kv_blocks = (seq_len + scale_block_size - 1) / scale_block_size;
    size_t q_scale_count = size_t(batch) * num_heads_q * q_blocks;
    size_t k_scale_count = size_t(batch) * num_heads_kv * kv_blocks;
    size_t lut_count = size_t(batch) * num_heads_q * q_blocks * kv_blocks;
    size_t valid_count = size_t(batch) * num_heads_q * q_blocks;
    size_t mask_count = size_t(batch) * seq_len * seq_len;
    float softmax_scale = 1.0f / std::sqrt(float(head_dim));

    auto host_q = to_fp16_vector(make_random_vector(q_count, -1.0f, 1.0f, 2201u + uint32_t(head_dim)));
    auto host_k = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 2301u + uint32_t(seq_len)));
    auto host_v = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 2401u + uint32_t(num_heads_q)));
    std::vector<int> host_lut;
    std::vector<int> host_valid;
    std::vector<float> host_mask;
    int query_tile_tokens = head_dim == 64 ? 128 : 256;
    std::vector<std::vector<int>> per_query_tile_selection =
        head_dim == 64 ? std::vector<std::vector<int>>{{0, 1}, {1, 3}} : std::vector<std::vector<int>>{{0, 2}};
    build_prefill_sparse_lut_and_mask(batch, num_heads_q, seq_len, seq_len, q_blocks, kv_blocks, scale_block_size,
                                      query_tile_tokens, per_query_tile_selection, host_lut, host_valid, host_mask);

    auto* dev_q = reinterpret_cast<sycl::half*>(ctx->allocate(q_count * sizeof(sycl::half)));
    auto* dev_k = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_v = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_qi8 = reinterpret_cast<int8_t*>(ctx->allocate(q_count * sizeof(int8_t)));
    auto* dev_ki8 = reinterpret_cast<int8_t*>(ctx->allocate(kv_count * sizeof(int8_t)));
    auto* dev_qscale = reinterpret_cast<float*>(ctx->allocate(q_scale_count * sizeof(float)));
    auto* dev_kscale = reinterpret_cast<float*>(ctx->allocate(k_scale_count * sizeof(float)));
    auto* dev_dense_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_sparse_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_lut = reinterpret_cast<int*>(ctx->allocate(lut_count * sizeof(int)));
    auto* dev_valid = reinterpret_cast<int*>(ctx->allocate(valid_count * sizeof(int)));
    auto* dev_mask = reinterpret_cast<float*>(ctx->allocate(mask_count * sizeof(float)));

    int q_stride_s = head_dim;
    int q_stride_d = 1;
    int q_stride_h = seq_len * head_dim;
    int q_stride_b = num_heads_q * seq_len * head_dim;
    int k_stride_s = head_dim;
    int k_stride_d = 1;
    int k_stride_h = seq_len * head_dim;
    int k_stride_b = num_heads_kv * seq_len * head_dim;
    int v_stride_d = 1;
    int v_stride_s = head_dim;
    int v_stride_h = seq_len * head_dim;
    int v_stride_b = num_heads_kv * seq_len * head_dim;
    int o_stride_s = head_dim;
    int o_stride_d = 1;
    int o_stride_h = seq_len * head_dim;
    int o_stride_b = num_heads_q * seq_len * head_dim;

    try {
      q->memcpy(dev_q, host_q.data(), q_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_k, host_k.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_v, host_v.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_lut, host_lut.data(), lut_count * sizeof(int)).wait();
      q->memcpy(dev_valid, host_valid.data(), valid_count * sizeof(int)).wait();
      q->memcpy(dev_mask, host_mask.data(), mask_count * sizeof(float)).wait();

      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_q, dev_qi8, dev_qscale, batch * num_heads_q, seq_len,
                                                      q_blocks, head_dim, scale_block_size);
      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_k, dev_ki8, dev_kscale, batch * num_heads_kv, seq_len,
                                                      kv_blocks, head_dim, scale_block_size);

      ark::sdpa_impl_qks8_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_dense_out, dev_mask, scale_block_size, dev_qscale,
                                 dev_kscale, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                                 k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                 o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len,
                                 seq_len, head_dim, softmax_scale, false);

      ark::sdpa_impl_qks8_sparse_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, scale_block_size,
                                        dev_qscale, dev_kscale, dev_lut, dev_valid, q_blocks, kv_blocks, q_stride_s,
                                        q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                                        k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                        o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len,
                                        seq_len, head_dim, softmax_scale, false);
      q->wait();

      std::vector<sycl::half> host_dense(o_count);
      std::vector<sycl::half> host_sparse(o_count);
      q->memcpy(host_dense.data(), dev_dense_out, o_count * sizeof(sycl::half)).wait();
      q->memcpy(host_sparse.data(), dev_sparse_out, o_count * sizeof(sycl::half)).wait();

      float max_diff = 0.0f;
      double mean_diff = 0.0;
      for (size_t index = 0; index < o_count; ++index) {
        float dense_value = float(host_dense[index]);
        float sparse_value = float(host_sparse[index]);
        if (!std::isfinite(dense_value) || !std::isfinite(sparse_value)) {
          throw std::runtime_error("partial sparse prefill test produced non-finite output: " + name);
        }
        float diff = std::fabs(dense_value - sparse_value);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
      }
      mean_diff /= double(o_count);
      std::cout << std::fixed << std::setprecision(6) << "[sage_sparse][prefill_partial] " << name
                << " max_diff=" << max_diff << " mean_diff=" << mean_diff << "\n";
      if (max_diff > 5e-3f || mean_diff > 5e-4f) {
        throw std::runtime_error("partial sparse prefill mismatch: " + name);
      }
    } catch (...) {
      ctx->deallocate(dev_q);
      ctx->deallocate(dev_k);
      ctx->deallocate(dev_v);
      ctx->deallocate(dev_qi8);
      ctx->deallocate(dev_ki8);
      ctx->deallocate(dev_qscale);
      ctx->deallocate(dev_kscale);
      ctx->deallocate(dev_dense_out);
      ctx->deallocate(dev_sparse_out);
      ctx->deallocate(dev_lut);
      ctx->deallocate(dev_valid);
      ctx->deallocate(dev_mask);
      throw;
    }

    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_qi8);
    ctx->deallocate(dev_ki8);
    ctx->deallocate(dev_qscale);
    ctx->deallocate(dev_kscale);
    ctx->deallocate(dev_dense_out);
    ctx->deallocate(dev_sparse_out);
    ctx->deallocate(dev_lut);
    ctx->deallocate(dev_valid);
    ctx->deallocate(dev_mask);
  }

  static void test_sparse_sage_partial_prefill_causal(const std::string& name, int batch, int num_heads_q,
                                                      int num_heads_kv, int seq_len, int head_dim,
                                                      int scale_block_size) {
    GETQ();
    LOG_LINE();
    size_t q_count = size_t(batch) * num_heads_q * seq_len * head_dim;
    size_t kv_count = size_t(batch) * num_heads_kv * seq_len * head_dim;
    size_t o_count = size_t(batch) * num_heads_q * seq_len * head_dim;
    int q_blocks = (seq_len + scale_block_size - 1) / scale_block_size;
    int kv_blocks = (seq_len + scale_block_size - 1) / scale_block_size;
    size_t q_scale_count = size_t(batch) * num_heads_q * q_blocks;
    size_t k_scale_count = size_t(batch) * num_heads_kv * kv_blocks;
    size_t lut_count = size_t(batch) * num_heads_q * q_blocks * kv_blocks;
    size_t valid_count = size_t(batch) * num_heads_q * q_blocks;
    size_t mask_count = size_t(batch) * seq_len * seq_len;
    float softmax_scale = 1.0f / std::sqrt(float(head_dim));

    auto host_q = to_fp16_vector(make_random_vector(q_count, -1.0f, 1.0f, 3201u + uint32_t(head_dim)));
    auto host_k = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 3301u + uint32_t(seq_len)));
    auto host_v = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 3401u + uint32_t(num_heads_q)));
    std::vector<int> host_lut;
    std::vector<int> host_valid;
    std::vector<float> host_mask;
    int query_tile_tokens = head_dim == 64 ? 128 : 256;
    std::vector<std::vector<int>> per_query_tile_selection =
        head_dim == 64 ? std::vector<std::vector<int>>{{0, 1, 2}, {1, 3}} : std::vector<std::vector<int>>{{0, 2, 3}};
    build_prefill_sparse_lut_and_mask(batch, num_heads_q, seq_len, seq_len, q_blocks, kv_blocks, scale_block_size,
                                      query_tile_tokens, per_query_tile_selection, host_lut, host_valid, host_mask,
                                      true);

    auto* dev_q = reinterpret_cast<sycl::half*>(ctx->allocate(q_count * sizeof(sycl::half)));
    auto* dev_k = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_v = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_qi8 = reinterpret_cast<int8_t*>(ctx->allocate(q_count * sizeof(int8_t)));
    auto* dev_ki8 = reinterpret_cast<int8_t*>(ctx->allocate(kv_count * sizeof(int8_t)));
    auto* dev_qscale = reinterpret_cast<float*>(ctx->allocate(q_scale_count * sizeof(float)));
    auto* dev_kscale = reinterpret_cast<float*>(ctx->allocate(k_scale_count * sizeof(float)));
    auto* dev_dense_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_sparse_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_lut = reinterpret_cast<int*>(ctx->allocate(lut_count * sizeof(int)));
    auto* dev_valid = reinterpret_cast<int*>(ctx->allocate(valid_count * sizeof(int)));
    auto* dev_mask = reinterpret_cast<float*>(ctx->allocate(mask_count * sizeof(float)));

    int q_stride_s = head_dim;
    int q_stride_d = 1;
    int q_stride_h = seq_len * head_dim;
    int q_stride_b = num_heads_q * seq_len * head_dim;
    int k_stride_s = head_dim;
    int k_stride_d = 1;
    int k_stride_h = seq_len * head_dim;
    int k_stride_b = num_heads_kv * seq_len * head_dim;
    int v_stride_d = 1;
    int v_stride_s = head_dim;
    int v_stride_h = seq_len * head_dim;
    int v_stride_b = num_heads_kv * seq_len * head_dim;
    int o_stride_s = head_dim;
    int o_stride_d = 1;
    int o_stride_h = seq_len * head_dim;
    int o_stride_b = num_heads_q * seq_len * head_dim;

    try {
      q->memcpy(dev_q, host_q.data(), q_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_k, host_k.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_v, host_v.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_lut, host_lut.data(), lut_count * sizeof(int)).wait();
      q->memcpy(dev_valid, host_valid.data(), valid_count * sizeof(int)).wait();
      q->memcpy(dev_mask, host_mask.data(), mask_count * sizeof(float)).wait();

      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_q, dev_qi8, dev_qscale, batch * num_heads_q, seq_len,
                                                      q_blocks, head_dim, scale_block_size);
      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_k, dev_ki8, dev_kscale, batch * num_heads_kv, seq_len,
                                                      kv_blocks, head_dim, scale_block_size);

      ark::sdpa_impl_qks8_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_dense_out, dev_mask, scale_block_size, dev_qscale,
                                 dev_kscale, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                                 k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                 o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len,
                                 seq_len, head_dim, softmax_scale, false);

      ark::sdpa_impl_qks8_sparse_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, scale_block_size,
                                        dev_qscale, dev_kscale, dev_lut, dev_valid, q_blocks, kv_blocks, q_stride_s,
                                        q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                                        k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                        o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len,
                                        seq_len, head_dim, softmax_scale, true);
      q->wait();

      std::vector<sycl::half> host_dense(o_count);
      std::vector<sycl::half> host_sparse(o_count);
      q->memcpy(host_dense.data(), dev_dense_out, o_count * sizeof(sycl::half)).wait();
      q->memcpy(host_sparse.data(), dev_sparse_out, o_count * sizeof(sycl::half)).wait();

      float max_diff = 0.0f;
      double mean_diff = 0.0;
      for (size_t index = 0; index < o_count; ++index) {
        float dense_value = float(host_dense[index]);
        float sparse_value = float(host_sparse[index]);
        if (!std::isfinite(dense_value) || !std::isfinite(sparse_value)) {
          throw std::runtime_error("causal partial sparse prefill test produced non-finite output: " + name);
        }
        float diff = std::fabs(dense_value - sparse_value);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
      }
      mean_diff /= double(o_count);
      std::cout << std::fixed << std::setprecision(6) << "[sage_sparse][prefill_partial_causal] " << name
                << " max_diff=" << max_diff << " mean_diff=" << mean_diff << "\n";
      if (max_diff > 5e-3f || mean_diff > 5e-4f) {
        throw std::runtime_error("causal partial sparse prefill mismatch: " + name);
      }
    } catch (...) {
      ctx->deallocate(dev_q);
      ctx->deallocate(dev_k);
      ctx->deallocate(dev_v);
      ctx->deallocate(dev_qi8);
      ctx->deallocate(dev_ki8);
      ctx->deallocate(dev_qscale);
      ctx->deallocate(dev_kscale);
      ctx->deallocate(dev_dense_out);
      ctx->deallocate(dev_sparse_out);
      ctx->deallocate(dev_lut);
      ctx->deallocate(dev_valid);
      ctx->deallocate(dev_mask);
      throw;
    }

    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_qi8);
    ctx->deallocate(dev_ki8);
    ctx->deallocate(dev_qscale);
    ctx->deallocate(dev_kscale);
    ctx->deallocate(dev_dense_out);
    ctx->deallocate(dev_sparse_out);
    ctx->deallocate(dev_lut);
    ctx->deallocate(dev_valid);
    ctx->deallocate(dev_mask);
  }

  static void test_sparse_sage_noncontiguous_prefill(const std::string& name, int batch, int num_heads_q,
                                                     int num_heads_kv, int seq_len, int head_dim,
                                                     int scale_block_size,
                                                     const std::vector<int>& selected_blocks, bool is_causal) {
    GETQ();
    LOG_LINE();
    size_t q_count = size_t(batch) * num_heads_q * seq_len * head_dim;
    size_t kv_count = size_t(batch) * num_heads_kv * seq_len * head_dim;
    size_t o_count = size_t(batch) * num_heads_q * seq_len * head_dim;
    int q_blocks = (seq_len + scale_block_size - 1) / scale_block_size;
    int kv_blocks = (seq_len + scale_block_size - 1) / scale_block_size;
    size_t q_scale_count = size_t(batch) * num_heads_q * q_blocks;
    size_t k_scale_count = size_t(batch) * num_heads_kv * kv_blocks;
    size_t lut_count = size_t(batch) * num_heads_q * q_blocks * kv_blocks;
    size_t valid_count = size_t(batch) * num_heads_q * q_blocks;
    size_t mask_count = size_t(batch) * seq_len * seq_len;
    float softmax_scale = 1.0f / std::sqrt(float(head_dim));

    auto host_q = to_fp16_vector(make_random_vector(q_count, -1.0f, 1.0f, 4201u + uint32_t(head_dim)));
    auto host_k = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 4301u + uint32_t(seq_len)));
    auto host_v = to_fp16_vector(make_random_vector(kv_count, -1.0f, 1.0f, 4401u + uint32_t(num_heads_q)));
    std::vector<int> host_lut;
    std::vector<int> host_valid;
    std::vector<float> host_mask;
    int query_tile_tokens = head_dim == 64 ? 128 : 256;
    int active_query_tiles = (seq_len + query_tile_tokens - 1) / query_tile_tokens;
    std::vector<std::vector<int>> per_query_tile_selection(size_t(active_query_tiles), selected_blocks);
    build_prefill_sparse_lut_and_mask(batch, num_heads_q, seq_len, seq_len, q_blocks, kv_blocks, scale_block_size,
                                      query_tile_tokens, per_query_tile_selection, host_lut, host_valid, host_mask,
                                      is_causal);

    auto* dev_q = reinterpret_cast<sycl::half*>(ctx->allocate(q_count * sizeof(sycl::half)));
    auto* dev_k = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_v = reinterpret_cast<sycl::half*>(ctx->allocate(kv_count * sizeof(sycl::half)));
    auto* dev_qi8 = reinterpret_cast<int8_t*>(ctx->allocate(q_count * sizeof(int8_t)));
    auto* dev_ki8 = reinterpret_cast<int8_t*>(ctx->allocate(kv_count * sizeof(int8_t)));
    auto* dev_qscale = reinterpret_cast<float*>(ctx->allocate(q_scale_count * sizeof(float)));
    auto* dev_kscale = reinterpret_cast<float*>(ctx->allocate(k_scale_count * sizeof(float)));
    auto* dev_dense_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_sparse_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_lut = reinterpret_cast<int*>(ctx->allocate(lut_count * sizeof(int)));
    auto* dev_valid = reinterpret_cast<int*>(ctx->allocate(valid_count * sizeof(int)));
    auto* dev_mask = reinterpret_cast<float*>(ctx->allocate(mask_count * sizeof(float)));

    int q_stride_s = head_dim;
    int q_stride_d = 1;
    int q_stride_h = seq_len * head_dim;
    int q_stride_b = num_heads_q * seq_len * head_dim;
    int k_stride_s = head_dim;
    int k_stride_d = 1;
    int k_stride_h = seq_len * head_dim;
    int k_stride_b = num_heads_kv * seq_len * head_dim;
    int v_stride_d = 1;
    int v_stride_s = head_dim;
    int v_stride_h = seq_len * head_dim;
    int v_stride_b = num_heads_kv * seq_len * head_dim;
    int o_stride_s = head_dim;
    int o_stride_d = 1;
    int o_stride_h = seq_len * head_dim;
    int o_stride_b = num_heads_q * seq_len * head_dim;

    try {
      q->memcpy(dev_q, host_q.data(), q_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_k, host_k.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_v, host_v.data(), kv_count * sizeof(sycl::half)).wait();
      q->memcpy(dev_lut, host_lut.data(), lut_count * sizeof(int)).wait();
      q->memcpy(dev_valid, host_valid.data(), valid_count * sizeof(int)).wait();
      q->memcpy(dev_mask, host_mask.data(), mask_count * sizeof(float)).wait();

      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_q, dev_qi8, dev_qscale, batch * num_heads_q, seq_len,
                                                      q_blocks, head_dim, scale_block_size);
      ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_k, dev_ki8, dev_kscale, batch * num_heads_kv, seq_len,
                                                      kv_blocks, head_dim, scale_block_size);

      ark::sdpa_impl_qks8_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_dense_out, dev_mask, scale_block_size, dev_qscale,
                                 dev_kscale, q_stride_s, q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d,
                                 k_stride_h, k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                 o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len,
                                 seq_len, head_dim, softmax_scale, false);

      ark::sdpa_impl_qks8_sparse_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_sparse_out, nullptr, scale_block_size,
                                        dev_qscale, dev_kscale, dev_lut, dev_valid, q_blocks, kv_blocks, q_stride_s,
                                        q_stride_d, q_stride_h, q_stride_b, k_stride_s, k_stride_d, k_stride_h,
                                        k_stride_b, v_stride_d, v_stride_s, v_stride_h, v_stride_b, o_stride_s,
                                        o_stride_d, o_stride_h, o_stride_b, batch, num_heads_q, num_heads_kv, seq_len,
                                        seq_len, head_dim, softmax_scale, is_causal);
      q->wait();

      std::vector<sycl::half> host_dense(o_count);
      std::vector<sycl::half> host_sparse(o_count);
      q->memcpy(host_dense.data(), dev_dense_out, o_count * sizeof(sycl::half)).wait();
      q->memcpy(host_sparse.data(), dev_sparse_out, o_count * sizeof(sycl::half)).wait();

      float max_diff = 0.0f;
      double mean_diff = 0.0;
      for (size_t index = 0; index < o_count; ++index) {
        float dense_value = float(host_dense[index]);
        float sparse_value = float(host_sparse[index]);
        if (!std::isfinite(dense_value) || !std::isfinite(sparse_value)) {
          throw std::runtime_error("non-contiguous sparse prefill test produced non-finite output: " + name);
        }
        float diff = std::fabs(dense_value - sparse_value);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
      }
      mean_diff /= double(o_count);
      std::cout << std::fixed << std::setprecision(6) << "[sage_sparse][prefill_noncontig] " << name
                << " max_diff=" << max_diff << " mean_diff=" << mean_diff << "\n";
      if (max_diff > 5e-3f || mean_diff > 5e-4f) {
        throw std::runtime_error("non-contiguous sparse prefill mismatch: " + name);
      }
    } catch (...) {
      ctx->deallocate(dev_q);
      ctx->deallocate(dev_k);
      ctx->deallocate(dev_v);
      ctx->deallocate(dev_qi8);
      ctx->deallocate(dev_ki8);
      ctx->deallocate(dev_qscale);
      ctx->deallocate(dev_kscale);
      ctx->deallocate(dev_dense_out);
      ctx->deallocate(dev_sparse_out);
      ctx->deallocate(dev_lut);
      ctx->deallocate(dev_valid);
      ctx->deallocate(dev_mask);
      throw;
    }

    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_qi8);
    ctx->deallocate(dev_ki8);
    ctx->deallocate(dev_qscale);
    ctx->deallocate(dev_kscale);
    ctx->deallocate(dev_dense_out);
    ctx->deallocate(dev_sparse_out);
    ctx->deallocate(dev_lut);
    ctx->deallocate(dev_valid);
    ctx->deallocate(dev_mask);
  }

#if 0  // legacy benchmarks rely on outdated sdpa_impl_qks8_* signatures
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
    size_t q_scale_count = size_t(batch) * num_heads_q * ((seq_len_q + scale_block_size - 1) / scale_block_size);
    size_t k_scale_count = size_t(batch) * num_heads_kv * ((seq_len_kv + scale_block_size - 1) / scale_block_size);
    size_t v_scale_count = k_scale_count * head_dim;
    float softmax_scale = 1.0f / std::sqrt(float(head_dim));

    auto host_q = to_fp16_vector(make_random_vector(q_count, -1.0f, 1.0f, 901u + uint32_t(seq_len_q)));
    auto host_k = to_fp16_vector(make_random_vector(k_count, -1.0f, 1.0f, 1001u + uint32_t(seq_len_kv)));
    auto host_v = to_fp16_vector(make_random_vector(k_count, -1.0f, 1.0f, 1101u + uint32_t(head_dim)));

    auto* dev_q = reinterpret_cast<sycl::half*>(ctx->allocate(q_count * sizeof(sycl::half)));
    auto* dev_k = reinterpret_cast<sycl::half*>(ctx->allocate(k_count * sizeof(sycl::half)));
    auto* dev_v = reinterpret_cast<sycl::half*>(ctx->allocate(k_count * sizeof(sycl::half)));
    auto* dev_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_kernel_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_pvi8_out = reinterpret_cast<sycl::half*>(ctx->allocate(o_count * sizeof(sycl::half)));
    auto* dev_qi8 = reinterpret_cast<int8_t*>(ctx->allocate(q_count * sizeof(int8_t)));
    auto* dev_ki8 = reinterpret_cast<int8_t*>(ctx->allocate(k_count * sizeof(int8_t)));
    auto* dev_vi8 = reinterpret_cast<int8_t*>(ctx->allocate(k_count * sizeof(int8_t)));
    auto* dev_qscale = reinterpret_cast<float*>(ctx->allocate(q_scale_count * sizeof(float)));
    auto* dev_kscale = reinterpret_cast<float*>(ctx->allocate(k_scale_count * sizeof(float)));
    auto* dev_vscale = reinterpret_cast<float*>(ctx->allocate(v_scale_count * sizeof(float)));
    q->memcpy(dev_q, host_q.data(), q_count * sizeof(sycl::half)).wait();
    q->memcpy(dev_k, host_k.data(), k_count * sizeof(sycl::half)).wait();
    q->memcpy(dev_v, host_v.data(), k_count * sizeof(sycl::half)).wait();

    ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_q, dev_qi8, dev_qscale, batch * num_heads_q, seq_len_q,
                                                    (seq_len_q + scale_block_size - 1) / scale_block_size, head_dim,
                                                    scale_block_size);
    ark::XpuWrapper::sage_dynamic_quant<sycl::half>(q, dev_k, dev_ki8, dev_kscale, batch * num_heads_kv, seq_len_kv,
                                                    (seq_len_kv + scale_block_size - 1) / scale_block_size, head_dim,
                                                    scale_block_size);
    ark::XpuWrapper::sage_dynamic_quant_v<sycl::half>(q, dev_v, dev_vi8, dev_vscale, batch * num_heads_kv,
                              seq_len_kv,
                              (seq_len_kv + scale_block_size - 1) / scale_block_size,
                              head_dim, scale_block_size);

    double sage_ms = run_bench(
        [&]() {
          ark::XpuWrapper::sagev1(q, dev_q, dev_k, dev_v, dev_out, nullptr, scale_block_size, batch, num_heads_q,
                                  num_heads_kv, seq_len_q, seq_len_kv, head_dim, softmax_scale, is_causal);
        },
        q, warmup, iters);

    double kernel_pvhalf_ms = run_bench(
        [&]() {
          ark::sdpa_impl_qks8_pvhalf(q, dev_qi8, dev_ki8, dev_v, dev_kernel_out, nullptr, scale_block_size,
                                      dev_qscale, dev_kscale, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv,
                                      head_dim, softmax_scale, is_causal);
        },
        q, warmup, iters);

    double kernel_pvi8_ms = run_bench(
        [&]() {
          ark::sdpa_impl_qks8_pvi8(q, dev_qi8, dev_ki8, dev_vi8, dev_pvi8_out, nullptr, scale_block_size, dev_qscale,
                                   dev_kscale, dev_vscale, batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv,
                                   head_dim, softmax_scale, is_causal);
        },
        q, warmup, iters);

    double flops = compute_sdpa_flops(batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim);
    double io_bytes = compute_sdpa_io_bytes(batch, num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, head_dim);
    double sage_tflops = flops / (sage_ms * 1e-3) / 1e12;
    double sage_gbps = io_bytes / (sage_ms * 1e-3) / 1e9;
    double kernel_pvhalf_tflops = flops / (kernel_pvhalf_ms * 1e-3) / 1e12;
    double kernel_pvhalf_gbps = io_bytes / (kernel_pvhalf_ms * 1e-3) / 1e9;
    double kernel_pvi8_tflops = flops / (kernel_pvi8_ms * 1e-3) / 1e12;
    double kernel_pvi8_gbps = io_bytes / (kernel_pvi8_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(3) << "[sagev1][bench] " << name << " sage_ms=" << sage_ms
              << " sage_TFLOPS=" << sage_tflops << " sage_GBps=" << sage_gbps << "\n";
    std::cout << std::fixed << std::setprecision(3) << "[sagev1][kernel_bench] " << name
              << " pvhalf_ms=" << kernel_pvhalf_ms << " pvhalf_TFLOPS=" << kernel_pvhalf_tflops
              << " pvhalf_GBps=" << kernel_pvhalf_gbps << "\n";
    std::cout << std::fixed << std::setprecision(3) << "[sagev1][kernel_bench] " << name
          << " pvi8_ms=" << kernel_pvi8_ms << " pvi8_TFLOPS=" << kernel_pvi8_tflops
          << " pvi8_GBps=" << kernel_pvi8_gbps << "\n";

    ctx->deallocate(dev_q);
    ctx->deallocate(dev_k);
    ctx->deallocate(dev_v);
    ctx->deallocate(dev_out);
    ctx->deallocate(dev_kernel_out);
    ctx->deallocate(dev_pvi8_out);
    ctx->deallocate(dev_qi8);
    ctx->deallocate(dev_ki8);
    ctx->deallocate(dev_vi8);
    ctx->deallocate(dev_qscale);
    ctx->deallocate(dev_kscale);
    ctx->deallocate(dev_vscale);
  }
#endif  // legacy benchmarks

  template <typename T>
  void run_dynamic_quant_bench(const char* tag, sycl::queue* q, int num_rows, int seq, int n_seq_blk,
                               int head_dim, int block_size, int warmup, int iters) {
    auto ctx = Context::Instance();
    size_t in_count = size_t(num_rows) * size_t(seq) * size_t(head_dim);
    size_t scale_count = size_t(num_rows) * size_t(n_seq_blk);
    auto* dev_in = reinterpret_cast<T*>(ctx->allocate(in_count * sizeof(T)));
    auto* dev_out = reinterpret_cast<int8_t*>(ctx->allocate(in_count * sizeof(int8_t)));
    auto* dev_scale = reinterpret_cast<float*>(ctx->allocate(scale_count * sizeof(float)));

    // initialize input with random fp values cast to T
    auto host_f = make_random_vector(in_count, -1.0f, 1.0f, 1234u + uint32_t(num_rows + seq));
    std::vector<T> host_t(in_count);
    for (size_t i = 0; i < in_count; ++i) host_t[i] = T(host_f[i]);
    q->memcpy(dev_in, host_t.data(), in_count * sizeof(T)).wait();

    double ms = run_bench(
        [&]() {
          ark::XpuWrapper::sage_dynamic_quant<T>(q, dev_in, dev_out, dev_scale, num_rows, seq, n_seq_blk,
                                                  head_dim, block_size);
        },
        q, warmup, iters);

    double bytes = double(in_count) * double(sizeof(T)) + double(in_count) * 1.0 +
                   double(scale_count) * double(sizeof(float));
    double gbps = bytes / (ms * 1e-3) / 1e9;
    std::cout << std::fixed << std::setprecision(4) << "[dq][" << tag << "] ms=" << ms
              << " bytes=" << bytes / 1e6 << "MB GBps=" << gbps << "\n";

    ctx->deallocate(dev_in);
    ctx->deallocate(dev_out);
    ctx->deallocate(dev_scale);
  }

  void benchmark_dynamic_quant(const std::string& name, int batch, int num_heads, int seq,
                               int head_dim, int block_size, int warmup, int iters) {
    GETQ();
    LOG_LINE();
    int num_rows = batch * num_heads;
    int n_seq_blk = (seq + block_size - 1) / block_size;
    std::cout << "[dq][cfg] " << name << " rows=" << num_rows << " seq=" << seq
              << " head_dim=" << head_dim << " block=" << block_size
              << " n_seq_blk=" << n_seq_blk << "\n";
    run_dynamic_quant_bench<sycl::half>("f16", q, num_rows, seq, n_seq_blk, head_dim, block_size, warmup,
                                         iters);
    run_dynamic_quant_bench<sycl::ext::oneapi::bfloat16>("bf16", q, num_rows, seq, n_seq_blk, head_dim,
                                                          block_size, warmup, iters);
  }
#endif
};
