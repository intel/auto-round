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
#pragma once
#include "kernel_avx512f.h"

namespace bestla {
namespace kernel {
namespace avx512f {
namespace avx512_bf16 {
#if CompileBF16()
#if defined(__INTEL_LLVM_COMPILER)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bf16,avx512vl,avx512bw"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512bf16", "avx512vl", "avx512bw")
#endif

static inline __m256i zmm_cvt_fp32_bf16(__m512 vfp32) { return (__m256i)_mm512_cvtneps_pbh(vfp32); }

static inline __m512 load_bf16_fp32(const utils::bf16* srcptr) {
  auto tmp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcptr));
  auto vf32 = zmm_cvt_bf16_fp32(tmp);
  return vf32;
}

static inline BTLA_CODE bf16_cvt_fp32_2D_write_back(const utils::bf16* src_ptr, float* dst_ptr, int row, int col,
                                                    int src_step, int dst_step, bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(float);
  constexpr int simd_proc_elt = 16;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const auto tail_mask = _cvtu32_mask16((1U << col_tail) - 1);
  for (int i = 0; i < row; i++) {
    auto src = const_cast<utils::bf16*>(src_ptr + i * src_step);
    auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) _mm512_storeu_ps(dst + j, load_bf16_fp32(src + j));
    if (col_tail > 0) {
      __m256i tmp = _mm256_setzero_si256();
      tmp = _mm256_mask_loadu_epi16(tmp, tail_mask, src + j);
      _mm512_mask_storeu_ps(dst + j, tail_mask, zmm_cvt_bf16_fp32(tmp));
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE fp32_cvt_bf16_2D_write_back(const void* raw_srcptr, void* raw_dstptr, int row, int col,
                                                    int srcstride, int dststride, bool zeropadding) {
  auto srcptr = reinterpret_cast<const char*>(raw_srcptr);
  auto dstptr = reinterpret_cast<char*>(raw_dstptr);
  constexpr int simd_proc_elt = 32;
  auto col_body_loop = col / simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const uint32_t tail_mask = (1U << col_tail) - 1;
  int npadding = dststride - col * sizeof(utils::bf16);
  for (int i = 0; i < row; i++) {
    auto src = srcptr + i * srcstride;
    auto dst = dstptr + i * dststride;
    int j = 0;
    for (; j < col_body_loop; j++) {
      _mm512_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(utils::bf16)),
          (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
                                       _mm512_loadu_ps(src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
    if (col_tail > 0) {
      _mm512_mask_storeu_epi16(
          (dst + (j * simd_proc_elt) * sizeof(utils::bf16)), tail_mask,  //
          (__m512i)_mm512_cvtne2ps_pbh(
              _mm512_maskz_loadu_ps(tail_mask >> 16, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 16),
              _mm512_maskz_loadu_ps(tail_mask >> 0, src + sizeof(float) * simd_proc_elt * j + sizeof(float) * 0)));
    }
    if (zeropadding && npadding) {
      std::memset(dst + col * sizeof(utils::bf16), 0, npadding);
    }
  }
  return BTLA_CODE::Success;
}

template <typename T_DST>
static inline BTLA_CODE scale_exp_acc_sum_fp32(const float* src, const int src_step, T_DST* dst, int ld_dst,
                                               float* dst_sum, const int M_offset, const int N_offset, const int M,
                                               const int N, float scale, int causal_offset, void* /* tmpcache */,
                                               size_t /* cachesize */) {
  const auto v_scale = _mm512_set1_ps(scale);
  for (int i = 0; i < M; ++i) {
    const auto N_unmasked = std::min(N, causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + causal_offset + 1);

    const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
    int j = 0;
    auto v_sum = _mm512_setzero_ps();
    for (; j < N_unmasked - 15; j += 16) {
      const auto v_exp = kernel::avx512f::exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_loadu_ps(src + i * src_step + j)));
      v_sum = _mm512_add_ps(v_sum, v_exp);
      _mm256_storeu_epi16(dst + i * ld_dst + j, (__m256i)_mm512_cvtneps_pbh(v_exp));
    }
    if (j < N_unmasked) {
      const auto v_exp =
          kernel::avx512f::exp_ps_0_1(_mm512_mul_ps(v_scale, _mm512_maskz_loadu_ps(v_mask, src + i * src_step + j)));
      v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
      _mm256_storeu_epi16(dst + i * ld_dst + j, (__m256i)_mm512_maskz_cvtneps_pbh(v_mask, v_exp));
      j += 16;
    }
    dst_sum[i] += _mm512_reduce_add_ps(v_sum);

    if (j < utils::padto(N, 64)) std::memset(dst + i * ld_dst + j, 0, sizeof(*dst) * (utils::padto(N, 64) - j));
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE inplace_precompute_max_softmax_fp32_bf16(int m_size, int n_size, int n_pad_size, bool is_causal,
                                                                 float* src, utils::bf16* dst, const float* s_max,
                                                                 float* expsum, int ld_src, int ld_dst) {
  for (int ii = 0; ii < m_size; ++ii) {
    const auto i_src = src + ii * ld_src;
    const auto i_dst = dst + ii * ld_dst;
    const auto curr_n_size = n_size + (is_causal ? ii : 0);
    const auto v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
    const auto v_mask32 = _cvtu32_mask32((1U << (curr_n_size % 32)) - 1);
    {  // subtract max
      const auto row_max = _mm512_set1_ps(s_max[ii]);
      for (int jj = 0; jj < curr_n_size; jj += 16) {  // should be fine to do extra work on idx >= curr_n_size
        _mm512_storeu_ps(i_src + jj, _mm512_sub_ps(_mm512_loadu_ps(i_src + jj), row_max));
      }
    }
    auto v_sum = _mm512_setzero_ps();
    {  // exp & sum
      int jj = 0;
      for (; jj < curr_n_size / 16 * 16; jj += 16) {
        const auto v_exp = kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));
        v_sum = _mm512_add_ps(v_sum, v_exp);
        _mm512_storeu_ps(i_src + jj, v_exp);
      }
      if (jj < curr_n_size) {
        const auto v_exp =
            kernel::avx512f::exp_ps_0_1(_mm512_loadu_ps(i_src + jj));  // should be fine to load some extra
        v_sum = _mm512_mask_add_ps(v_sum, v_mask, v_sum, v_exp);
        _mm512_storeu_ps(i_src + jj, v_exp);  // should be fine to store some extra
      }
      expsum[ii] = _mm512_reduce_add_ps(v_sum);
      v_sum = _mm512_set1_ps(expsum[ii]);
    }
    {  // scale & bf16
      int jj = 0;
      for (; jj < curr_n_size / 32 * 32; jj += 32) {
        const auto v_softmax0 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
        const auto v_softmax1 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj + 16), v_sum);
        _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_cvtne2ps_pbh(v_softmax1, v_softmax0));
      }
      if (jj < curr_n_size) {
        const auto v_softmax0 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
        const auto v_softmax1 = _mm512_div_ps(_mm512_loadu_ps(i_src + jj + 16), v_sum);
#if defined(__GNUC__) && (__GNUC__ == 13) && (__GNUC_MINOR__ <= 2)
        // There is a bug on gcc 13.1/13.2 what reverse the parameter order;
        // A GUN team member said that it will befixed in GCC 13.3
        _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_maskz_cvtne2ps_pbh(v_mask32, v_softmax0, v_softmax1));
#else
        _mm512_storeu_epi16(i_dst + jj, (__m512i)_mm512_maskz_cvtne2ps_pbh(v_mask32, v_softmax1, v_softmax0));
#endif
        jj += 32;
      }
      if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(utils::bf16) * (n_pad_size - jj));
    }
  }
  return BTLA_CODE::Success;
}
#if defined(__INTEL_LLVM_COMPILER)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
}  // namespace avx512_bf16
}  // namespace avx512f
}  // namespace kernel
}  // namespace bestla
