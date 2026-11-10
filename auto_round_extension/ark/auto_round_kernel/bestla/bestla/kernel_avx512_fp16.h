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
#include "kernel_avx512_bf16.h"

namespace bestla {
namespace kernel {
namespace avx512f {
namespace avx512_fp16 {
#if CompileFP16()
#if defined(__INTEL_LLVM_COMPILER)
#pragma clang attribute push(__attribute__((target("avx512f,avx512bf16,avx512bw,avx512fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bf16", "avx512vl", "avx512bw", "avx512fp16")
#endif

inline __m512 zmm_cvt_fp16_fp32(__m256i vfp16) { return _mm512_cvtxph_ps((__m256h)vfp16); }

inline __m256i zmm_cvt_fp32_fp16(__m512 vfp32) { return (__m256i)_mm512_cvtxps_ph(vfp32); }

template <typename T>
static inline void store_fp32_T(__m512 src_y, T* dstptr) {
  if constexpr (std::is_same_v<T, utils::bf16>) {
    auto ymm = avx512_bf16::zmm_cvt_fp32_bf16(src_y);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr), ymm);
  } else if constexpr (std::is_same_v<T, float>) {
    _mm512_storeu_ps(dstptr, src_y);
  } else if constexpr (std::is_same_v<T, utils::fp16>) {
    auto ymm = zmm_cvt_fp32_fp16(src_y);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr), ymm);
  } else {
    assert(false);
  }
}

static inline BTLA_CODE fp32_cvt_fp16_2D_write_back(const float* src_ptr, utils::fp16* dst_ptr, int row, int col,
                                                    int src_step, int dst_step, bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(utils::fp16);
  constexpr int simd_proc_elt = 16;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const auto tail_mask = _cvtu32_mask16((1U << col_tail) - 1);
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      store_fp32_T(_mm512_loadu_ps(src + j), dst + j);
    }
    if (col_tail > 0) {
      auto vf32 = _mm512_maskz_loadu_ps(tail_mask, src + j);
      auto vf16 = zmm_cvt_fp32_fp16(vf32);
      _mm256_mask_storeu_epi16(dst + j, tail_mask, vf16);
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE fp16_cvt_fp32_2D_write_back(const utils::fp16* src_ptr, float* dst_ptr, int row, int col,
                                                    int src_step, int dst_step, bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(float);
  constexpr int simd_proc_elt = 16;
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  const auto tail_mask = _cvtu32_mask16((1U << col_tail) - 1);
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      auto vf32 = load_T_fp32(src + j);
      _mm512_storeu_ps(dst + j, vf32);
    }
    if (col_tail > 0) {
      auto vf16 = _mm256_maskz_loadu_epi16(tail_mask, src + j);
      auto v32 = zmm_cvt_fp16_fp32(vf16);
      _mm512_mask_storeu_ps(dst + j, tail_mask, v32);
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return BTLA_CODE::Success;
}
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"  // https://stackoverflow.com/a/49216021
#endif
// Load 2 fp16 vectors; convert them to bf16 and interleave them
template <int tail>
static inline std::array<__m512i, 2> load_fp16_bf16_interleave_word(const utils::fp16* a, size_t lda) {
  static_assert(tail > 0 && tail <= 2, "Unexpected tail value.");
  std::array<__m512i, 2> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(load_T_fp32(a + i * lda + 16), load_T_fp32(a + i * lda + 0)));
  }
  for (int i = tail; i < 2; ++i) dst[i] = _mm512_setzero_epi32();
  interleave_word(dst);
  return dst;
}

// load_fp16_bf16_interleave_word with maskz
template <int tail>
static inline std::array<__m512i, 2> load_maskz_fp16_bf16_interleave_word(const utils::fp16* a, size_t lda,
                                                                          uint32_t mask) {
  static_assert(tail > 0 && tail <= 2, "Unexpected tail value.");

  const auto mask_lo = mask;
  const auto mask_hi = mask >> 16;
  std::array<__m512i, 2> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                                      //
        zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),  //
        zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0))));
  }
  for (int i = tail; i < 2; ++i) dst[i] = _mm512_setzero_epi32();
  interleave_word(dst);
  return dst;
}

template <int tail>
static inline std::array<__m512i, 16> load_fp16_bf16_tr_x16_dword(const utils::fp16* a, size_t lda) {
  static_assert(tail > 0 && tail <= 16, "Unexpected tail value.");
  std::array<__m512i, 16> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                       //
        zmm_cvt_fp16_fp32(_mm256_loadu_epi16(a + i * lda + 16)),  //
        zmm_cvt_fp16_fp32(_mm256_loadu_epi16(a + i * lda + 0))));
  }
  for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();
  tr_x16_dword(dst);
  return dst;
}
static constexpr decltype(load_fp16_bf16_tr_x16_dword<1>)* load_fp16_bf16_tr_x16_dword_tbl[17]{
    load_fp16_bf16_tr_x16_dword<1>,  load_fp16_bf16_tr_x16_dword<1>,  load_fp16_bf16_tr_x16_dword<2>,
    load_fp16_bf16_tr_x16_dword<3>,  load_fp16_bf16_tr_x16_dword<4>,  load_fp16_bf16_tr_x16_dword<5>,
    load_fp16_bf16_tr_x16_dword<6>,  load_fp16_bf16_tr_x16_dword<7>,  load_fp16_bf16_tr_x16_dword<8>,
    load_fp16_bf16_tr_x16_dword<9>,  load_fp16_bf16_tr_x16_dword<10>, load_fp16_bf16_tr_x16_dword<11>,
    load_fp16_bf16_tr_x16_dword<12>, load_fp16_bf16_tr_x16_dword<13>, load_fp16_bf16_tr_x16_dword<14>,
    load_fp16_bf16_tr_x16_dword<15>, load_fp16_bf16_tr_x16_dword<16>,
};

template <int tail>
static inline std::array<__m512i, 16> load_maskz_fp16_bf16_tr_x16_dword(const utils::fp16* a, size_t lda,
                                                                        uint32_t mask) {
  static_assert(tail > 0 && tail <= 16, "Unexpected tail value.");
  std::array<__m512i, 16> dst;

  const auto mask_lo = mask;
  const auto mask_hi = mask >> 16;
  for (int i = 0; i < tail; ++i) {
    dst[i] = (__m512i)(_mm512_cvtne2ps_pbh(                                      //
        zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),  //
        zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0))));
  }
  for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();
  tr_x16_dword(dst);
  return dst;
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static constexpr decltype(load_maskz_fp16_bf16_tr_x16_dword<1>)* load_maskz_fp16_bf16_tr_x16_dword_tbl[17]{
    load_maskz_fp16_bf16_tr_x16_dword<1>,  load_maskz_fp16_bf16_tr_x16_dword<1>,  load_maskz_fp16_bf16_tr_x16_dword<2>,
    load_maskz_fp16_bf16_tr_x16_dword<3>,  load_maskz_fp16_bf16_tr_x16_dword<4>,  load_maskz_fp16_bf16_tr_x16_dword<5>,
    load_maskz_fp16_bf16_tr_x16_dword<6>,  load_maskz_fp16_bf16_tr_x16_dword<7>,  load_maskz_fp16_bf16_tr_x16_dword<8>,
    load_maskz_fp16_bf16_tr_x16_dword<9>,  load_maskz_fp16_bf16_tr_x16_dword<10>, load_maskz_fp16_bf16_tr_x16_dword<11>,
    load_maskz_fp16_bf16_tr_x16_dword<12>, load_maskz_fp16_bf16_tr_x16_dword<13>, load_maskz_fp16_bf16_tr_x16_dword<14>,
    load_maskz_fp16_bf16_tr_x16_dword<15>, load_maskz_fp16_bf16_tr_x16_dword<16>,
};

template <typename T_SRC, typename T_DST = T_SRC, int RowPack = 4 / sizeof(T_DST)>
struct padding_interleave_cvt {
  padding_interleave_cvt() = delete;
  static BTLA_CODE forward(const T_SRC* src, T_DST* dst, int NTile, int row, int col, int row_pad, int col_pad,
                           int src_step, int dst_step) {
    return BTLA_CODE::NotSupport;
  }
};

template <>
struct padding_interleave_cvt<utils::fp16, utils::bf16, 2> {
  static constexpr int RowPack = 2;
  padding_interleave_cvt() = delete;

  // M x N ===> N/NTile x M/RowPack x NTile x RowPack (leading dim stride = NTile * dststride)
  static BTLA_CODE forward(const utils::fp16* src, utils::bf16* dst, int NTile, int row, int col, int row_pad,
                           int col_pad, int src_step, int dst_step) {
    int i = 0;
    for (; i < row / RowPack * RowPack; i += RowPack) {
      int j = 0;
      for (; j < col / NTile * NTile; j += NTile) {
        assert(NTile % 32 == 0);
        for (int jj = 0; jj < NTile; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
      }
      if (j < col) {  // j: tail processing
        int jj = 0;
        for (; j + jj < col / 32 * 32; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
        if (j + jj < col) {  // jj: tail processing
          const uint32_t mask = (1U << (col - j - jj)) - 1;
          const auto xss = load_maskz_fp16_bf16_interleave_word<2>(src + i * src_step + j + jj, src_step, mask);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          jj += 32;
        }
        for (; jj < NTile; jj += 32) {  // jj: padding zero
          memset(dst + i * NTile + j * dst_step + jj * RowPack, 0, sizeof(utils::bf16) * 32 * RowPack);
        }
        j += NTile;
      }
      for (; j < col_pad; j += NTile) {  // j: padding zero
        memset(dst + i * NTile + j * dst_step, 0, sizeof(utils::bf16) * NTile * RowPack);
      }
    }
    if (i < row) {                      // i: tail processing
      static constexpr int tail_m = 1;  // must be 1
      int j = 0;
      for (; j < col / NTile * NTile; j += NTile) {
        assert(NTile % 32 == 0);
        for (int jj = 0; jj < NTile; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
      }
      if (j < col) {  // j: tail processing
        int jj = 0;
        for (; j + jj < col / 32 * 32; jj += 32) {
          const auto xss = load_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
        }
        if (j + jj < col) {  // jj: tail processing
          const uint32_t mask = (1U << (col - j - jj)) - 1;
          const auto xss = load_maskz_fp16_bf16_interleave_word<tail_m>(src + i * src_step + j + jj, src_step, mask);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 0) * RowPack, xss[0]);
          _mm512_storeu_si512(dst + i * NTile + j * dst_step + (jj + 16) * RowPack, xss[1]);
          jj += 32;
        }
        for (; jj < NTile; jj += 32) {  // jj: padding zero
          memset(dst + i * NTile + j * dst_step + jj * RowPack, 0, sizeof(utils::bf16) * 32 * RowPack);
        }
        j += NTile;
      }
      for (; j < col_pad; j += NTile) {  // j: padding zero
        memset(dst + i * NTile + j * dst_step, 0, sizeof(utils::bf16) * NTile * RowPack);
      }
      i += RowPack;
    }
    for (; i < row_pad; i += RowPack) {  // i: padding zero
      for (int j = 0; j < col_pad; j += NTile) {
        memset(dst + i * NTile + j * dst_step, 0, sizeof(utils::bf16) * NTile * RowPack);
      }
    }
    return BTLA_CODE::Success;
  }
};

template <typename T_SRC, typename T_DST = T_SRC, int ColPack = 4 / sizeof(T_DST)>
struct padding_trans_interleave_cvt {
  padding_trans_interleave_cvt() = delete;
  static BTLA_CODE forward(const T_SRC* src, T_DST* dst, int MTile, int row, int col, int row_pad, int col_pad,
                           int src_step, int dst_step) {
    return BTLA_CODE::NotSupport;
  }
};

template <>
struct padding_trans_interleave_cvt<utils::fp16, utils::bf16, 2> {
  static constexpr int ColPack = 2;
  padding_trans_interleave_cvt() = delete;

  static BTLA_CODE forward(const utils::fp16* src, utils::bf16* dst, int MTile, int row, int col, int row_pad,
                           int col_pad, int src_step, int dst_step) {
    assert(row_pad % 16 == 0 && col_pad % 32 == 0);
    int i = 0;
    for (; i < row / MTile * MTile; i += MTile) {
      assert(MTile % 16 == 0);
      int j = 0;
      for (; j < col / 32 * 32; j += 32) {
        for (int ii = 0; ii < MTile; ii += 16) {
          assert(MTile % 16 == 0);
          const auto xss = load_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
      }
      if (j < col) {  // j: tail processing
        for (int ii = 0; ii < MTile; ii += 16) {
          assert(MTile % 16 == 0);
          const uint32_t mask = (1U << (col - j)) - 1;
          const auto xss = load_maskz_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step, mask);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
        j += 32;
      }
      for (; j < col_pad; j += 2) {  // j: padding zero
        memset(dst + i * dst_step + j * MTile, 0, 2 * sizeof(utils::bf16) * MTile);
      }
    }
    if (i < row) {  // i: tail processing
      int ii = 0;
      for (; i + ii < row / 16 * 16; ii += 16) {
        int j = 0;
        for (; j < col / 32 * 32; j += 32) {
          assert(MTile % 16 == 0);
          const auto xss = load_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
        if (j < col) {  // j: tail processing
          assert(MTile % 16 == 0);
          const uint32_t mask = (1U << (col - j)) - 1;
          const auto xss = load_maskz_fp16_bf16_tr_x16_dword<16>(src + (i + ii) * src_step + j, src_step, mask);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
          j += 32;
        }
        for (; j < col_pad; j += 2) {  // j: padding zero
          memset(dst + i * dst_step + ii * ColPack + j * MTile, 0, 2 * sizeof(utils::bf16) * 16);
        }
      }
      if (i + ii < row) {  // ii: tail processing
        const int tbl_idx = row - i - ii;
        int j = 0;
        for (; j < col / 32 * 32; j += 32) {
          assert(MTile % 16 == 0);
          const auto xss = load_fp16_bf16_tr_x16_dword_tbl[tbl_idx](src + (i + ii) * src_step + j, src_step);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
        }
        if (j < col) {  // j: tail processing
          assert(MTile % 16 == 0);
          const uint32_t mask = (1U << (col - j)) - 1;
          const auto xss =
              load_maskz_fp16_bf16_tr_x16_dword_tbl[tbl_idx](src + (i + ii) * src_step + j, src_step, mask);
          for (int jj = 0; jj < 32; jj += 2) {
            _mm512_storeu_si512(dst + i * dst_step + ii * ColPack + (j + jj) * MTile, xss[jj / 2]);
          }
          j += 32;
        }
        for (; j < col_pad; j += 2) {  // j: padding zero
          memset(dst + i * dst_step + ii * ColPack + j * MTile, 0, 2 * sizeof(utils::bf16) * 16);
        }
        ii += 16;
      }
      for (; ii < MTile; ii += 16) {  // ii: padding zero
        for (int j = 0; j < col_pad; j += 2) {
          memset(dst + i * dst_step + ii * ColPack + j * MTile, 0, 2 * sizeof(utils::bf16) * 16);
        }
      }
      assert(ii == MTile);
      i += MTile;
    }
    assert(row_pad % MTile == 0);
    for (; i < row_pad; i += MTile) {  // i: padding zero
      for (int j = 0; j < col_pad; j += 2) {
        memset(dst + i * dst_step + j * MTile, 0, 2 * sizeof(utils::bf16) * MTile);
      }
    }
    return BTLA_CODE::Success;
  }
};

static inline BTLA_CODE scale_track_max_fp16_fp32(const utils::fp16* src, const int src_step, float* dst,
                                                  float* dst_max, int ld_dst, const int M_offset, const int N_offset,
                                                  const int M, const int N, float scale, int causal_offset,
                                                  float alibi_slope, float tanh_scale, void* tmpcache,
                                                  size_t cachesize) {
  const auto v_scale = _mm512_set1_ps(scale);
  for (int i = 0; i < M; ++i) {
    const auto N_unmasked = std::min(N, causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + causal_offset + 1);

    const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
    int j = 0;
    auto v_max = _mm512_set1_ps(-INFINITY);
    for (; j < N_unmasked - 15; j += 16) {
      const auto xs = _mm512_mul_ps(v_scale, _mm512_cvtxph_ps(_mm256_loadu_ph(src + i * src_step + j)));
      v_max = _mm512_max_ps(v_max, xs);
      _mm512_storeu_ps(dst + i * ld_dst + j, xs);
    }
    if (j < N_unmasked) {
      const auto xs = _mm512_mul_ps(
          v_scale, _mm512_cvtxph_ps(_mm256_castsi256_ph(_mm256_maskz_loadu_epi16(v_mask, src + i * src_step + j))));
      v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
      _mm512_storeu_ps(dst + i * ld_dst + j, xs);
      j += 16;
    }
    dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE inplace_precompute_max_softmax_fp32_fp16(int m_size, int n_size, int n_pad_size, bool is_causal,
                                                                 float* src, utils::fp16* dst, const float* s_max,
                                                                 float* expsum, int ld_src, int ld_dst) {
  for (int ii = 0; ii < m_size; ++ii) {
    const auto i_src = src + ii * ld_src;
    const auto i_dst = dst + ii * ld_dst;
    const auto curr_n_size = n_size + (is_causal ? ii : 0);
    const uint16_t v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
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
    {  // scale & fp16
      int jj = 0;
      for (; jj < curr_n_size / 16 * 16; jj += 16) {
        const auto v_softmax = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
        _mm256_storeu_ph(i_dst + jj, _mm512_cvtxps_ph(v_softmax));
      }
      if (jj < curr_n_size) {
        const auto v_softmax = _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum);
        _mm256_storeu_ph(i_dst + jj, _mm512_maskz_cvtxps_ph(v_mask, v_softmax));
        jj += 16;
      }
      if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(utils::fp16) * (n_pad_size - jj));
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
}  // namespace avx512_fp16
}  // namespace avx512f
}  // namespace kernel
}  // namespace bestla
