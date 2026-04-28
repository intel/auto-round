//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#pragma once

#include "utils.hpp"

#define GETCTX()                                    \
  auto& eng = *DnnlContext::Instance()->get_eng(q); \
  auto& stream = *DnnlContext::Instance()->get_stream(q);

namespace ark {

class DnnlWrapper {
 public:
  using dt = dnnl::memory::data_type;
  using tag = dnnl::memory::format_tag;

  static void gemm(sycl::queue* q, int m, int n, int k, const void* a, dt at, const void* b, dt bt, bool BT, void* c,
                   dt ct, const void* bias) {
    GETCTX();
    dnnl::memory::dims a_dims = {m, k};
    const auto a_in_md = dnnl::memory::desc(a_dims, at, dnnl::memory::format_tag::ab);
    auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));

    dnnl::memory::dims b_dims = {k, n};

    const auto b_in_md =
        dnnl::memory::desc(b_dims, bt, BT ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
    auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));

    auto bias_md = dnnl::memory::desc({1, n}, ct, dnnl::memory::format_tag::ab);
    auto bias_mem = dnnl::memory(bias_md, eng, const_cast<void*>(bias));

    dnnl::memory::dims c_dims = {m, n};
    const auto c_md = dnnl::memory::desc(c_dims, ct, dnnl::memory::format_tag::ab);
    dnnl::primitive_attr primitive_attr;
    primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::strict);

    auto matmul_pd = bias ? dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, bias_md, c_md, primitive_attr)
                          : dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
    auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

    auto scratchpad_md = matmul_pd.scratchpad_desc();

    auto scratchpad_mem = DnnlContext::Instance()->get_scratch_mem(scratchpad_md, q);

    auto matmul_prim = dnnl::matmul(matmul_pd);

    std::unordered_map<int, dnnl::memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    if (bias) matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});
    matmul_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
    matmul_prim.execute(stream, matmul_args);
    if (q == nullptr) stream.wait();
  }

  static void dyn_quant_s8(sycl::queue* q, int m, int k, const void* a, dt adt, void* a_abs, int8_t* qa, void* maxa,
                           void* scalea) {
    GETCTX();
    using namespace dnnl;
    auto src_f32_md = memory::desc({m, k}, adt, memory::format_tag::ab);
    auto src_f32_mem = memory(src_f32_md, eng, const_cast<void*>(a));
    auto src_absf32_mem = memory(src_f32_md, eng, const_cast<void*>(a_abs));

    auto abs_attr = dnnl::eltwise_forward::primitive_desc(eng, prop_kind::forward_inference, algorithm::eltwise_abs,
                                                          src_f32_mem.get_desc(), src_f32_mem.get_desc(), 0.0f, 0.0f);

    auto abs_prim = dnnl::eltwise_forward(abs_attr);
    abs_prim.execute(stream, {{DNNL_ARG_SRC, src_f32_mem}, {DNNL_ARG_DST, src_absf32_mem}});
    // 1. 定义输出 memory (只有一个标量值)
    // --- 第二步：求最大值 (Reduction Max) ---
    auto max_abs_md = dnnl::memory::desc({1, 1}, adt, dnnl::memory::format_tag::ab);

    auto reduct_pd = dnnl::reduction::primitive_desc(eng, dnnl::algorithm::reduction_max, src_absf32_mem.get_desc(),
                                                     max_abs_md, 0.0f, 0.0f);

    auto max_abs_mem = dnnl::memory(max_abs_md, eng, maxa);
    dnnl::reduction(reduct_pd).execute(stream, {{DNNL_ARG_SRC, src_absf32_mem}, {DNNL_ARG_DST, max_abs_mem}});

    // 4. (可选) 将 max 转换为 127/max
    // 这里展示如何用 eltwise 算子在 Device 端直接完成转换：scale = 127 * (1/x)
    auto final_scale_mem = memory(max_abs_md, eng, scalea);
    auto eltwise_pd = eltwise_forward::primitive_desc(eng, prop_kind::forward_inference, algorithm::eltwise_linear,
                                                      max_abs_md, max_abs_md, 1 / 127.0f, .0f);

    eltwise_forward(eltwise_pd).execute(stream, {{DNNL_ARG_SRC, max_abs_mem}, {DNNL_ARG_DST, final_scale_mem}});

    auto src_s8_md = memory::desc({m, k}, memory::data_type::s8, memory::format_tag::ab);
    auto src_s8_mem = memory(src_s8_md, eng, qa);
    auto scale1_md = dnnl::memory::desc({1}, adt, dnnl::memory::format_tag::a);
    auto scale1_mem = memory(scale1_md, eng, scalea);

    // 使用 Reorder 进行量化
    primitive_attr q_attr;
    q_attr.set_scales_mask(DNNL_ARG_DST, 0);  // Per-tensor

    auto q_reorder_pd = reorder::primitive_desc(eng, src_f32_md, eng, src_s8_md, q_attr);
    reorder(q_reorder_pd)
        .execute(stream, {{DNNL_ARG_SRC, src_f32_mem},
                          {DNNL_ARG_DST, src_s8_mem},
                          {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, scale1_mem}});
  }

  static void sycl_dyn_quant_s8(sycl::queue* q, int m, int k, const void* a, dt adt, int8_t* qa, void* scalea,
                                int mask) {
#if ARK_XPU
    if (adt == dt::f32) {
      using T = float;
      using Pro = bestla::sycl_prologue_a::ActivationBase<T>;
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(T*)a, k}, qa, (T*)scalea, q);
    } else if (adt == dt::f16) {
      using T = sycl::half;
      using Pro = bestla::sycl_prologue_a::ActivationBase<T>;
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(T*)a, k}, qa, (T*)scalea, q);
    } else if (adt == dt::bf16) {
      using T = sycl::ext::oneapi::bfloat16;
      using Pro = bestla::sycl_prologue_a::ActivationBase<T>;
      Pro::template quant_s8<typename Pro::CfgQuantF32>(m, k, mask, {(T*)a, k}, qa, (T*)scalea, q);
    }
#endif
  }

  // a:mxk b:nxk c:mxn
  // scale_a:m  scale_b:n
  static void sycl_igemm_s8s8(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c,
                              dt ct, void* scale_a, void* scale_b, void* bias, int blocksize) {
#if ARK_XPU
    using namespace bestla::sycl_gemm;
    if (blocksize == k || blocksize == -1) {
      if (ct == dt::f32) {
        using T = float;
        Launcher<xmx::IGemmDQCfg<T>, xmx::IGemmDQCore>::run(
            q, {(void*)a, (void*)b, (void*)c, m, n, k, k, k, n, bias, scale_a, scale_b});
      } else if (ct == dt::f16) {
        using T = sycl::half;
        Launcher<xmx::IGemmDQCfg<T>, xmx::IGemmDQCore>::run(
            q, {(void*)a, (void*)b, (void*)c, m, n, k, k, k, n, bias, scale_a, scale_b});
      } else if (ct == dt::bf16) {
        using T = sycl::ext::oneapi::bfloat16;
        Launcher<xmx::IGemmDQCfg<T>, xmx::IGemmDQCore>::run(
            q, {(void*)a, (void*)b, (void*)c, m, n, k, k, k, n, bias, scale_a, scale_b});
      }
    } else {
      if (ct == dt::f32) {
        using T = float;
        Launcher<xmx::IKblockGemmDQCfg<T>, xmx::IKblockGemmDQCore>::run(
            q, {(void*)a, (void*)b, (void*)c, m, n, k, k, k, n, bias, scale_a, scale_b, blocksize});
      } else if (ct == dt::f16) {
        using T = sycl::half;
        Launcher<xmx::IKblockGemmDQCfg<T>, xmx::IKblockGemmDQCore>::run(
            q, {(void*)a, (void*)b, (void*)c, m, n, k, k, k, n, bias, scale_a, scale_b, blocksize});
      }
    }
#endif
  }

  // a:mxk b:nxk c:mxn
  // scale_a:m  scale_b:n
  static void igemm_s8s8(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c, dt ct,
                         void* scale_a, void* scale_b, void* bias) {
    GETCTX();

    dnnl::memory::dims a_dims = {m, k};
    const auto a_in_md = dnnl::memory::desc(a_dims, dt::s8, tag::ab);

    dnnl::memory::dims b_dims = {k, n};
    const auto b_in_md = dnnl::memory::desc(b_dims, dt::s8, BT ? tag::ba : tag::ab);

    auto bias_md = dnnl::memory::desc({1, n}, ct, dnnl::memory::format_tag::ab);
    auto bias_mem = dnnl::memory(bias_md, eng, bias);

    dnnl::memory::dims c_dims = {m, n};
    const auto c_md = dnnl::memory::desc(c_dims, ct, tag::ab);
    dnnl::primitive_attr primitive_attr;
    primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    const int src_mask = 0;  //  m, k   per m
    auto scalea_md = dnnl::memory::desc(dnnl::memory::dims{1}, ct, dnnl::memory::format_tag::a);
    const int wei_mask = (1 << 1);  // k,n   per n
    auto scaleb_md = dnnl::memory::desc(dnnl::memory::dims{n}, ct, dnnl::memory::format_tag::a);

    // primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::strict);
    primitive_attr.set_scales_mask(DNNL_ARG_SRC, src_mask);
    primitive_attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_mask);

    auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
    auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
    auto matmul_pd = bias ? dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, bias_md, c_md, primitive_attr)
                          : dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);

    auto scalea_mem = dnnl::memory(scalea_md, eng, const_cast<void*>(scale_a));
    auto scaleb_mem = dnnl::memory(scaleb_md, eng, const_cast<void*>(scale_b));
    auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

    auto scratchpad_md = matmul_pd.scratchpad_desc();

    auto scratchpad_mem = DnnlContext::Instance()->get_scratch_mem(scratchpad_md, q);

    auto matmul_prim = dnnl::matmul(matmul_pd);

    std::unordered_map<int, dnnl::memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, a_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
    if (bias) matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scalea_mem});
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scaleb_mem});
    matmul_args.insert({DNNL_ARG_DST, c_mem});
    matmul_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
    matmul_prim.execute(stream, matmul_args);
    if (q == nullptr) stream.wait();
  }

  // a:mxk:act b:nxk:s8 c:mxn:act
  // scale_b:n:act
  static void woq_s8(sycl::queue* q, int m, int n, int k, const void* a, const void* b, bool BT, void* c, dt act,
                     void* scale_b, void* bias, int blocksize) {
#if ARK_XPU
    size_t qa_size = (size_t)m * k;
    size_t tmp_size = qa_size + m * sizeof(float);
    auto tmp_ptr = DnnlContext::Instance()->get_scratch_mem(tmp_size, 1, q);
    auto qa_ptr = (int8_t*)tmp_ptr;
    auto scalea_ptr = qa_ptr + qa_size;
    sycl_dyn_quant_s8(q, m, k, a, act, qa_ptr, scalea_ptr, 0);
    sycl_igemm_s8s8(q, m, n, k, qa_ptr, b, BT, c, act, scalea_ptr, scale_b, bias, blocksize);
#else
    size_t abs_size = (size_t)m * k * dnnl::memory::data_type_size(act);
    size_t qa_size = (size_t)m * k;
    size_t tmp_size = qa_size + 64 * sizeof(float) + abs_size;
    auto tmp_ptr = DnnlContext::Instance()->get_scratch_mem(tmp_size, 1, q);
    auto abs_ptr = (int8_t*)tmp_ptr;
    auto qa_ptr = abs_ptr + abs_size;
    auto maxa_ptr = qa_ptr + qa_size;
    auto scalea_ptr = maxa_ptr + sizeof(float) * 32;
    dyn_quant_s8(q, m, k, a, act, abs_ptr, qa_ptr, maxa_ptr, scalea_ptr);
    igemm_s8s8(q, m, n, k, qa_ptr, b, BT, c, act, scalea_ptr, scale_b, bias);
#endif
  }
  struct WeightQuantAttr {
    bool enabled{false};
    int mask{0};
    std::vector<dnnl::memory::dim> groups;
    dnnl::memory::dims scale_dims;
    dnnl::memory::dims scale_strides;
    dt scale_dt{dt::undef};
    const void* scales{nullptr};
  };

#if 0
  template <typename T>
  static void reorder_scales_to_group_major(sycl::queue* q, int n, int groups_k, const T* src, T* dst) {
    if (groups_k == 1) {
      q->memcpy(dst, src, size_t(n) * sizeof(T)).wait();
      return;
    }
    q->parallel_for(sycl::range<2>(groups_k, n), [=](sycl::id<2> idx) {
       int g = idx[0];
       int col = idx[1];
       dst[g * n + col] = src[col * groups_k + g];
     }).wait();
  }

  static const void* prepare_scale_buffer(sycl::queue* q, const void* scales, dt st, int n, int groups_k,
                                          bool& used_scratch) {
    used_scratch = false;
    if (!scales) return nullptr;
    if (groups_k <= 1) return scales;
    size_t elem_cnt = size_t(groups_k) * size_t(n);
    size_t bytes = elem_cnt * dnnl::memory::data_type_size(st);
    if (!bytes) return nullptr;
    auto dst = DnnlContext::Instance()->get_scratch_mem(bytes, 5, q);
    used_scratch = true;
    switch (st) {
      case dt::f32:
        reorder_scales_to_group_major(q, n, groups_k, static_cast<const float*>(scales), static_cast<float*>(dst));
        break;
      case dt::f16:
        reorder_scales_to_group_major(q, n, groups_k, static_cast<const sycl::half*>(scales),
                                      static_cast<sycl::half*>(dst));
        break;
      case dt::bf16:
        reorder_scales_to_group_major(q, n, groups_k, static_cast<const sycl::ext::oneapi::bfloat16*>(scales),
                                      static_cast<sycl::ext::oneapi::bfloat16*>(dst));
        break;
      default:
        used_scratch = false;
        return nullptr;
    }
    return dst;
  }

  static void gemm(sycl::queue* q, int m, int n, int k, const void* a, dt at, dnnl_dim_t stra0, dnnl_dim_t stra1,
                   dnnl_dim_t stra2, const void* b, dt bt, dnnl_dim_t strb0, dnnl_dim_t strb1, dnnl_dim_t strb2,
                   void* c, dt ct, dnnl_dim_t batches_a, dnnl_dim_t batches_b, const WeightQuantAttr* wq_attr = nullptr,
                   bool sync = true) {
    auto& eng = *DnnlContext::Instance()->get_eng(q);
    auto& stream = *DnnlContext::Instance()->get_stream(q);
    dnnl::memory::dims a_dims = {batches_a, m, k};
    dnnl::memory::dims a_strides = {stra2, stra1, stra0};
    const auto a_in_md = dnnl::memory::desc(a_dims, at, a_strides);

    dnnl::memory::dims b_dims = {batches_b, k, n};
    dnnl::memory::dims b_strides = {strb2, strb1, strb0};
    const auto b_in_md = dnnl::memory::desc(b_dims, bt, b_strides);

    dnnl::memory::dims c_dims = {std::max(batches_a, batches_b), m, n};
    dnnl::memory::dims c_strides = {m * n, n, 1};
    const auto c_md = dnnl::memory::desc(c_dims, ct, c_strides);
    dnnl::primitive_attr primitive_attr;
    primitive_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (wq_attr && wq_attr->enabled) {
      primitive_attr.set_scales(DNNL_ARG_WEIGHTS, wq_attr->mask, wq_attr->groups, wq_attr->scale_dt);
      primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    }

    // primitive_attr.set_fpmath_mode(dnnl::fpmath_mode::strict);

    try {
      auto matmul_pd = dnnl::matmul::primitive_desc(eng, a_in_md, b_in_md, c_md, primitive_attr);
      auto a_mem = dnnl::memory(a_in_md, eng, const_cast<void*>(a));
      auto b_mem = dnnl::memory(b_in_md, eng, const_cast<void*>(b));
      auto c_mem = dnnl::memory(matmul_pd.dst_desc(), eng, c);

      auto scratchpad_md = matmul_pd.scratchpad_desc();

      auto scratchpad_mem = DnnlContext::Instance()->get_scratch_mem(scratchpad_md, q);

      auto matmul_prim = dnnl::matmul(matmul_pd);

      std::unordered_map<int, dnnl::memory> matmul_args;
      matmul_args.insert({DNNL_ARG_SRC, a_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});

      matmul_args.insert({DNNL_ARG_DST, c_mem});
      matmul_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
      dnnl::memory scale_mem;
      if (wq_attr && wq_attr->enabled) {
        auto scale_md = dnnl::memory::desc(wq_attr->scale_dims, wq_attr->scale_dt, wq_attr->scale_strides);
        scale_mem = dnnl::memory(scale_md, eng, const_cast<void*>(wq_attr->scales));
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem});
      }
      matmul_prim.execute(stream, matmul_args);
      // Avoid unconditional host-side synchronization: letting the caller decide
      // enables better overlap/pipelining across ops.
      if (sync) {
        stream.wait();
      }
    } catch (const dnnl::error& e) {
      std::cerr << "DNNL Error: " << e.what() << std::endl;
      throw;
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
      throw;
    }
  }

  static void row_gemm(sycl::queue& q, int8_t dev_idx, int m, int n, int k, const void* a, dt at, const void* b, dt bt,
                       bool b_T, void* c, dt ct, const WeightQuantAttr* wq_attr = nullptr, bool sync = true) {
    gemm(q, dev_idx, m, n, k, a, at, 1, k, k * m, b, bt, b_T ? k : 1, b_T ? 1 : n, n * k, c, ct, 1, 1, wq_attr, sync);
  }

  template <typename QueueT>
  static void gemv(QueueT& q, int8_t dev_idx, int m, int n, int k, const void* a, dt at, const void* x, dt xt, void* y,
                   dt yt, const WeightQuantAttr* wq_attr = nullptr, bool weights_transposed = true, bool sync = true) {
    // GEMV degenerates to GEMM with a single-row activation; default to transposed weights
    // to match the row-major NT layout used by weight-only quantization.
    row_gemm(q, dev_idx, m, n, k, a, at, x, xt, weights_transposed, y, yt, wq_attr, sync);
  }

  static float pow2_int(int power) {
    float result = 1.0f;
    if (power >= 0) {
      for (int i = 0; i < power; ++i) result *= 2.0f;
    } else {
      for (int i = 0; i < -power; ++i) result *= 0.5f;
    }
    return result;
  }

  static float decode_fp8_e4m3(uint8_t value) {
    constexpr int bias = 7;
    constexpr float max_val = 448.0f;
    int sign = (value & 0x80) ? -1 : 1;
    int exp = (value >> 3) & 0x0F;
    int mant = value & 0x07;
    if (exp == 0) {
      if (mant == 0) return 0.0f;
      float frac = mant / 8.0f;
      float val = frac * pow2_int(1 - bias);
      return sign * val;
    }
    if (exp == 0x0F) {
      return sign * max_val;
    }
    float frac = 1.0f + mant / 8.0f;
    float val = frac * pow2_int(exp - bias);
    return sign * val;
  }

  static float decode_fp8_e5m2(uint8_t value) {
    constexpr int bias = 15;
    constexpr float max_val = 57344.0f;
    int sign = (value & 0x80) ? -1 : 1;
    int exp = (value >> 2) & 0x1F;
    int mant = value & 0x03;
    if (exp == 0) {
      if (mant == 0) return 0.0f;
      float frac = mant / 4.0f;
      float val = frac * pow2_int(1 - bias);
      return sign * val;
    }
    if (exp == 0x1F) {
      return sign * max_val;
    }
    float frac = 1.0f + mant / 4.0f;
    float val = frac * pow2_int(exp - bias);
    return sign * val;
  }

  static float decode_fp8(uint8_t value, bool is_e4m3) {
    return is_e4m3 ? decode_fp8_e4m3(value) : decode_fp8_e5m2(value);
  }

  static void woq_gemm(sycl::queue* q, int m, int n, int k, const void* a, dt at, const void* b, dt bt, void* c, dt ct,
                       const void* scales, dt st, int blocksize) {
    bool weights_are_fp8 = (bt == dt::f8_e4m3 || bt == dt::f8_e5m2);
    if (weights_are_fp8 && blocksize > 0 && k > 0 && (k % blocksize) == 0) {
      WeightQuantAttr attr;
      attr.enabled = true;
      int groups_k = k / blocksize;
      bool single_group = groups_k == 1;
      // oneDNN matmul weights use dims {batch, k, n}, so per-column scales map to bit 2 and
      // additional k-group scales require bit 1 as well.
      attr.mask = single_group ? (1 << 2) : ((1 << 1) | (1 << 2));
      if (!single_group) {
        attr.groups = {static_cast<dnnl::memory::dim>(blocksize), 1};
      }
      bool used_scratch = false;
      const void* scale_ptr = prepare_scale_buffer(q, scales, st, n, groups_k, used_scratch);
      if (scale_ptr) {
        if (single_group) {
          attr.scale_dims = {static_cast<dnnl::memory::dim>(n)};
          attr.scale_strides = {1};
        } else {
          attr.scale_dims = {static_cast<dnnl::memory::dim>(groups_k), static_cast<dnnl::memory::dim>(n)};
          attr.scale_strides = {static_cast<dnnl::memory::dim>(n), 1};
        }
        attr.scale_dt = st;
        attr.scales = scale_ptr;

        const void* a_exec = a;
        dt a_exec_dt = at;
        if (at == dt::f32) {
          size_t elems = size_t(m) * k;
          size_t bytes = elems * sizeof(sycl::half);
          auto a_buf = DnnlContext::Instance()->get_scratch_mem(bytes, 6, q);
          auto a_src = static_cast<const float*>(a);
          auto a_dst = static_cast<sycl::half*>(a_buf);
          q->parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
             size_t i = idx[0];
             a_dst[i] = static_cast<sycl::half>(a_src[i]);
           }).wait();
          a_exec = a_buf;
          a_exec_dt = dt::f16;
        }

        void* c_exec = c;
        dt c_exec_dt = ct;
        bool need_c_cast = false;
        if (ct == dt::f32) {
          size_t elems = size_t(m) * n;
          size_t bytes = elems * sizeof(sycl::half);
          c_exec = DnnlContext::Instance()->get_scratch_ptr(bytes, 7, q, dev_idx);
          c_exec_dt = dt::f16;
          need_c_cast = true;
        }

        row_gemm(q, dev_idx, m, n, k, a_exec, a_exec_dt, b, bt, true, c_exec, c_exec_dt, &attr);

        if (need_c_cast) {
          size_t elems = size_t(m) * n;
          auto src_half = static_cast<sycl::half*>(c_exec);
          auto dst_f32 = static_cast<float*>(c);
          q.parallel_for(sycl::range<1>(elems), [=](sycl::id<1> idx) {
             size_t i = idx[0];
             dst_f32[i] = static_cast<float>(src_half[i]);
           }).wait();
        }
        return;
      }
    }

    bool need_fp32_compute = weights_are_fp8 || ct == dt::f16 || at != dt::f32;
    dt compute_dt = need_fp32_compute ? dt::f32 : ct;

    const void* a_compute = a;
    if (at != compute_dt) {
      size_t a_bytes = size_t(m) * k * sizeof(float);
      auto a_buf = DnnlContext::Instance()->get_scratch_ptr(a_bytes, 2, q, dev_idx);
      auto a_src = static_cast<const sycl::half*>(a);
      auto a_dst = static_cast<float*>(a_buf);
      q.parallel_for(sycl::range<1>(size_t(m) * k), [=](sycl::id<1> idx) {
         size_t i = idx[0];
         a_dst[i] = static_cast<float>(a_src[i]);
       }).wait();
      a_compute = a_buf;
    }

    const void* b_compute = b;
    if (weights_are_fp8) {
      size_t b_bytes = size_t(n) * k * sizeof(float);
      auto b_buf = DnnlContext::Instance()->get_scratch_ptr(b_bytes, 3, q, dev_idx);
      auto b_src = static_cast<const uint8_t*>(b);
      auto b_dst = static_cast<float*>(b_buf);
      bool is_e4m3 = (bt == dt::f8_e4m3);
      bool scale_is_f32 = (st == dt::f32);
      auto scale_f32 = static_cast<const float*>(scales);
      auto scale_f16 = static_cast<const sycl::half*>(scales);
      q.parallel_for(sycl::range<1>(size_t(n) * k), [=](sycl::id<1> idx) {
         size_t linear = idx[0];
         int col = linear % n;
         float scale = 1.0f;
         if (scale_is_f32) {
           scale = scale_f32[col];
         } else {
           scale = static_cast<float>(scale_f16[col]);
         }
         float dq = decode_fp8(b_src[linear], is_e4m3) * scale;
         b_dst[linear] = dq;
       }).wait();
      b_compute = b_buf;
      compute_dt = dt::f32;
    }

    void* c_compute = c;
    if (ct != compute_dt) {
      size_t c_bytes = size_t(m) * n * sizeof(float);
      c_compute = DnnlContext::Instance()->get_scratch_ptr(c_bytes, 4, q, dev_idx);
    }

    row_gemm(q, dev_idx, m, n, k, a_compute, compute_dt, b_compute, compute_dt, true, c_compute, compute_dt);

    if (ct != compute_dt) {
      auto src = static_cast<float*>(c_compute);
      auto dst = static_cast<sycl::half*>(c);
      q.parallel_for(sycl::range<1>(size_t(m) * n), [=](sycl::id<1> idx) {
         size_t i = idx[0];
         dst[i] = static_cast<sycl::half>(src[i]);
       }).wait();
    }
  }

  static void dequant_fp8(sycl::queue* q, int k, int n, const void* src, dt src_dt, void* dst, dt dst_dt,
                          const void* scales, dt st, int blocksize, bool src_col_major = true, bool sync = true) {
    bool weights_are_fp8 = (src_dt == dt::f8_e4m3 || src_dt == dt::f8_e5m2);
    if (!weights_are_fp8 || !src || !dst || !scales) return;

    int groups_k = (blocksize > 0 && (k % blocksize) == 0) ? (k / blocksize) : 1;
    bool single_group = groups_k == 1;
    dnnl::memory::dims mat_dims = {static_cast<dnnl::memory::dim>(k), static_cast<dnnl::memory::dim>(n)};
    dnnl::memory::dims src_strides = src_col_major ? dnnl::memory::dims{1, static_cast<dnnl::memory::dim>(k)}
                                                   : dnnl::memory::dims{static_cast<dnnl::memory::dim>(n), 1};
    dnnl::memory::dims dst_strides = {static_cast<dnnl::memory::dim>(n), 1};
    auto src_md = dnnl::memory::desc(mat_dims, src_dt, src_strides);
    auto dst_md = dnnl::memory::desc(mat_dims, dst_dt, dst_strides);

    dnnl::memory::dims scale_dims;
    dnnl::memory::dims scale_strides;
    if (single_group) {
      scale_dims = {static_cast<dnnl::memory::dim>(n)};
      scale_strides = {1};
    } else {
      scale_dims = {static_cast<dnnl::memory::dim>(groups_k), static_cast<dnnl::memory::dim>(n)};
      scale_strides = {static_cast<dnnl::memory::dim>(n), 1};
    }

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    int scale_mask = single_group ? (1 << 1) : ((1 << 0) | (1 << 1));
    attr.set_scales(DNNL_ARG_SRC, scale_mask, scale_dims, st);

    bool used_scratch = false;
    const void* scale_ptr = prepare_scale_buffer(q, scales, st, n, groups_k, used_scratch);
    if (!scale_ptr) return;

    GETCTX();
    auto reorder_pd = dnnl::reorder::primitive_desc(eng, src_md, eng, dst_md, attr);
    auto reorder_prim = dnnl::reorder(reorder_pd);

    auto src_mem = dnnl::memory(src_md, eng, const_cast<void*>(src));
    auto dst_mem = dnnl::memory(dst_md, eng, dst);
    auto scratchpad_mem = DnnlContext::Instance()->get_scratch_mem(reorder_pd.scratchpad_desc(), q);
    auto scale_md = dnnl::memory::desc(scale_dims, st, scale_strides);
    auto scale_mem = dnnl::memory(scale_md, eng, const_cast<void*>(scale_ptr));

    std::unordered_map<int, dnnl::memory> reorder_args;
    reorder_args.insert({DNNL_ARG_FROM, src_mem});
    reorder_args.insert({DNNL_ARG_TO, dst_mem});
    reorder_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_mem});
    reorder_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_mem});
    reorder_prim.execute(stream, reorder_args);
    if (sync) {
      stream.wait();
    }
  }
#endif
};

}  // namespace ark
