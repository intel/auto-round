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
#include "bestla.h"
#include "bestla_utils.h"
#include "kernel_jit.h"
#include "kernel_ref.h"

#include <array>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace bestla {
namespace kernel {
namespace avx512f {
#if CompileAVX512F()
#if defined(__INTEL_LLVM_COMPILER)
#pragma clang attribute push(__attribute__((target("avx512f,avx512vl,avx512bw,avx512dq"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx512f", "avx512bw", "avx512vl", "avx512dq")
#endif

inline __m512 zmm_cvt_fp16_fp32(__m256i vfp16) { return _mm512_cvtph_ps(vfp16); }

inline __m256i zmm_cvt_fp32_fp16(__m512 vfp32) {
  return _mm512_cvtps_ph(vfp32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

inline __m512 zmm_cvt_bf16_fp32(__m256i vbf16) {
  auto vf32 = _mm512_cvtepu16_epi32(vbf16);
  return _mm512_castsi512_ps(_mm512_slli_epi32(vf32, 16));
}

static inline __m256i cvt_fp32_to_bf16(const __m512 _src) {
  const auto bf16_and_helper = _mm512_set1_epi32(0x00000001);
  const auto bf16_add_helper = _mm512_set1_epi32(0X00007FFF);
  auto src = _mm512_castps_si512(_src);
  auto round_bias = _mm512_and_epi32(bf16_and_helper, _mm512_bsrli_epi128(src, 2));
  round_bias = _mm512_add_epi32(round_bias, bf16_add_helper);
  auto round_fp32_v = _mm512_add_epi32(round_bias, src);
  auto pack_bf16_value = _mm512_cvtepi32_epi16(_mm512_srli_epi32(round_fp32_v, 16));
  return pack_bf16_value;
}

inline __m256i zmm_cvt_fp32_bf16(__m512 vfp32) {
#if FP32_BF16_FAST
  return _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(vfp32), 2));
#else
  return cvt_fp32_to_bf16(vfp32);
#endif
}

inline __m512i zmm_cvt_fp32_bf16_x2(__m512 vfp32_hi, __m512 vfp32_lo) {
  auto lo = zmm_cvt_fp32_bf16(vfp32_lo);
  auto hi = zmm_cvt_fp32_bf16(vfp32_hi);
  __m512i result = _mm512_castsi256_si512(lo);
  result = _mm512_inserti64x4(result, hi, 1);
  return result;
}

static inline __m512 load_bf16_fp32(const utils::bf16* srcptr) {
  auto tmp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcptr));
  auto vf32 = zmm_cvt_bf16_fp32(tmp);
  return vf32;
}

static inline __m512 load_fp16_fp32(const utils::fp16* srcptr) {
  auto tmp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srcptr));
  auto vf32 = zmm_cvt_fp16_fp32(tmp);
  return vf32;
}

template <typename T>
static inline void convert_s8_fp_v16(T* dstptr, int8_t* srcptr) {
  auto xmm = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcptr));
  auto zmm = _mm512_cvtepi8_epi32(xmm);
  auto zmm1 = _mm512_cvtepi32_ps(zmm);
  if constexpr (std::is_same_v<T, utils::bf16>) {
    auto ymm = zmm_cvt_fp32_bf16(zmm1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr), ymm);
  } else if constexpr (std::is_same_v<T, utils::fp16>) {
    auto ymm = zmm_cvt_fp32_fp16(zmm1);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr), ymm);
  } else {
    _mm512_storeu_ps(dstptr, zmm1);
  }
}

static inline __m512i load_s8_s32(int8_t* srcptr) {
  auto xmm = _mm_loadu_si128(reinterpret_cast<__m128i*>(srcptr));
  auto ymm = _mm512_cvtepi8_epi32(xmm);
  return ymm;
}

static inline __m512 load_s8_fp32(int8_t* srcptr) {
  auto src_y = load_s8_s32(srcptr);
  auto dst_y = _mm512_cvtepi32_ps(src_y);
  return dst_y;
}

template <typename T>
static inline void store_fp32_T(__m512 src_y, T* dstptr) {
  if constexpr (std::is_same_v<T, utils::bf16>) {
    auto ymm = zmm_cvt_fp32_bf16(src_y);
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

template <typename T>
static inline __m512 load_T_fp32(const T* srcptr) {
  __m512 vtmp;
  if constexpr (std::is_same_v<T, float>) {
    vtmp = _mm512_loadu_ps(srcptr);
  } else if constexpr (std::is_same_v<T, utils::bf16>) {
    vtmp = load_bf16_fp32(srcptr);
  } else if constexpr (std::is_same_v<T, utils::fp16>) {
    vtmp = load_fp16_fp32(srcptr);
  } else {
    assert(0);
  }
  return vtmp;
}

static inline __m512i unpack_4bits(void* srcptr, __m512i mask) {
  auto raw_data = _mm256_loadu_si256(reinterpret_cast<__m256i*>(srcptr));
  auto ymm0 = _mm512_cvtepu8_epi16(raw_data);
  auto ymm1 = _mm512_slli_epi16(ymm0, 4);
  ymm0 = _mm512_or_si512(ymm0, ymm1);
  ymm0 = _mm512_and_si512(ymm0, mask);
  return ymm0;
}

static inline __m512i unpack_2bits(utils::bit2x4* ptr, const __m512i& vshift_y, const __m512i& vmask0_y,
                                   const __m512i& vsfhl_mask_y, const __m512i& vorder_y) {
  auto vraw_x = _mm_loadu_si128((const __m128i*)ptr);
  auto vsrc_y = _mm512_broadcast_i64x2(vraw_x);
  auto vordered_y = _mm512_permutex2var_epi32(vsrc_y, vorder_y, vsrc_y);
  auto vs_y = _mm512_srlv_epi32(vordered_y, vshift_y);
  auto v2_y = _mm512_and_si512(vs_y, vmask0_y);
  auto vout_y = _mm512_shuffle_epi8(v2_y, vsfhl_mask_y);
  return vout_y;
}

static inline __m512i unpack_1bits(utils::bit1x8* ptr, const __m512i& zmm_0x00, const __m512i& zmm_0x04) {
  auto bit1_mask1 = _cvtu64_mask64(*(uint64_t*)ptr);
  auto zmm1_ = _mm512_mask_mov_epi8(zmm_0x00, bit1_mask1, zmm_0x04);
  return zmm1_;
}

static inline __m512i unpack_4bits_high(__m256i v4bits, __m512i vmask) {
  auto ymm1 = _mm256_slli_epi32(v4bits, 4);
  auto zmm = _mm512_cvtepi8_epi16(v4bits);
  auto zmm1 = _mm512_cvtepi8_epi16(ymm1);
  zmm = _mm512_slli_epi16(zmm, 8);
  zmm1 = _mm512_mask_mov_epi8(zmm1, 0xaaaaaaaaaaaaaaaa, zmm);
  zmm1 = _mm512_and_epi32(zmm1, vmask);
  return zmm1;
}

static inline void convert_s4_s8_highbits(int8_t* dstptr, int8_t* srcptr, __m512i vmask, int LoadMask) {
  auto ymm = _mm256_maskz_loadu_epi32(__mmask8(LoadMask), reinterpret_cast<const __m256i*>(srcptr));
  auto zmm = unpack_4bits_high(ymm, vmask);
  _mm512_mask_storeu_epi64(dstptr, __mmask8(LoadMask), zmm);
}

static inline void convert_s4_s8_highbits_v32(int8_t* dstptr, int8_t* srcptr, __m512i vmask, int LoadMask) {
  auto xmm = _mm_maskz_loadu_epi32(__mmask8(LoadMask), reinterpret_cast<const __m256i*>(srcptr));
  auto ymm = _mm256_castsi128_si256(xmm);
  auto zmm = unpack_4bits_high(ymm, vmask);
  auto ymm_out = _mm512_castsi512_si256(zmm);
  _mm256_mask_storeu_epi64(dstptr, __mmask8(LoadMask), ymm_out);
}

constexpr void (*pad_fp4)(int8_t* dstptr, int8_t* srcptr, __m512i vmask, int) = &convert_s4_s8_highbits;

template <int N, typename _DST_T, bool _IS_SYM>
static inline void dequant_s8_N(_DST_T* dstptr, int8_t* srcptr, __m512* vscales, __m512i* vzps = nullptr) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto src_s8 = _mm_loadu_si128(reinterpret_cast<__m128i*>(srcptr + iv * 16));
    auto zmm = _mm512_cvtepi8_epi32(src_s8);
    if constexpr (!_IS_SYM) zmm = _mm512_sub_epi32(zmm, vzps[iv]);
    auto fzmm = _mm512_cvtepi32_ps(zmm);
    fzmm = _mm512_mul_ps(fzmm, vscales[iv]);
    if constexpr (std::is_same<_DST_T, float>::value) {
      _mm512_storeu_ps(dstptr + iv * 16, fzmm);
    } else if constexpr (std::is_same<_DST_T, utils::bf16>::value) {
      auto bf16_v = zmm_cvt_fp32_bf16(fzmm);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr + iv * 16), bf16_v);
    } else {
      assert(false);
    }
  }
}

template <bool IsAsym = false>
static inline __m512 dequant_s8_fp(int8_t* srcptr, __m512 vscales, __m512i vzps = __m512i()) {
  auto src_s32_y = load_s8_s32(srcptr);
  if constexpr (IsAsym) src_s32_y = _mm512_sub_epi32(src_s32_y, vzps);
  auto src_fp_y = _mm512_cvtepi32_ps(src_s32_y);
  src_fp_y = _mm512_mul_ps(src_fp_y, vscales);
  return src_fp_y;
}

template <int N, typename _DST_T, BTLA_DTYPE F4_T>
static inline void dequant_f4_N(_DST_T* dstptr, int8_t* srcptr, __m512* vscales, __m512i* vzps = nullptr) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
  float* LUT;
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  if constexpr (F4_T == BTLA_DTYPE::F4_BNB) {
    LUT = fp4_bnb_dequant_fp32_LUT;
  } else if constexpr (F4_T == BTLA_DTYPE::F4_NF4) {
    LUT = nf4_dequant_fp32_LUT;
  } else if constexpr (F4_T == BTLA_DTYPE::F4_E2M1) {
    LUT = fp4_e2m1_dequant_fp32_LUT;
  }
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto idx = _mm_loadu_si128(reinterpret_cast<__m128i*>(srcptr + iv * 16));
    idx = _mm_srli_epi32(idx, 4);
    auto pad_idx = _mm512_cvtepu8_epi32(idx);
    auto lut = _mm512_loadu_si512(LUT);
    auto fp32_dq_v = _mm512_permutexvar_epi32(pad_idx, lut);
    auto fzmm = _mm512_mul_ps(_mm512_castsi512_ps(fp32_dq_v), vscales[iv]);
    if constexpr (std::is_same<_DST_T, float>::value) {
      _mm512_storeu_ps(dstptr + iv * 16, fzmm);
    } else if constexpr (std::is_same<_DST_T, utils::bf16>::value) {
      auto bf16_v = zmm_cvt_fp32_bf16(fzmm);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr + iv * 16), bf16_v);
    } else {
      assert(false);
    }
  }
}

template <int N, typename _DST_T, BTLA_DTYPE F4_T>
static inline void unpack_f4_N(_DST_T* dstptr, int8_t* srcptr) {
  static_assert(N % 16 == 0);
  int constexpr VLoop = N / 16;
  float* LUT;
  static_assert(F4_T == BTLA_DTYPE::F4_BNB || F4_T == BTLA_DTYPE::F4_NF4 || F4_T == BTLA_DTYPE::F4_E2M1,
                "Unsupported F4 type");
  if constexpr (F4_T == BTLA_DTYPE::F4_BNB) {
    LUT = fp4_bnb_dequant_fp32_LUT;
  } else if constexpr (F4_T == BTLA_DTYPE::F4_NF4) {
    LUT = nf4_dequant_fp32_LUT;
  } else if constexpr (F4_T == BTLA_DTYPE::F4_E2M1) {
    LUT = fp4_e2m1_dequant_fp32_LUT;
  }
  for (int iv = 0; iv < VLoop; iv += 1) {
    auto idx = _mm_loadu_si128(reinterpret_cast<__m128i*>(srcptr + iv * 16));
    idx = _mm_srli_epi32(idx, 4);
    auto pad_idx = _mm512_cvtepu8_epi32(idx);
    auto lut = _mm512_loadu_si512(LUT);
    auto fp32_dq_v = _mm512_permutexvar_epi32(pad_idx, lut);
    auto fzmm = _mm512_castsi512_ps(fp32_dq_v);
    if constexpr (std::is_same<_DST_T, float>::value) {
      _mm512_storeu_ps(dstptr + iv * 16, fzmm);
    } else if constexpr (std::is_same<_DST_T, utils::bf16>::value) {
      auto bf16_v = zmm_cvt_fp32_bf16(fzmm);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr + iv * 16), bf16_v);
    } else {
      assert(false);
    }
  }
}

static inline void vec_broadcast_epi32_1_2(__m512i* dst2regs, __m512i* src1regs) {
  dst2regs[0] = _mm512_unpacklo_epi32(src1regs[0], src1regs[0]);
  dst2regs[1] = _mm512_unpackhi_epi32(src1regs[0], src1regs[0]);
}

static inline void vec_broadcast_ps_1_2(__m512* dst2regs, __m512* src1regs, __m512i idxreg) {
  auto tmpreg = _mm512_permutexvar_epi64(idxreg, _mm512_castps_si512(src1regs[0]));
  dst2regs[0] = _mm512_castsi512_ps(_mm512_unpacklo_epi32(tmpreg, tmpreg));
  dst2regs[1] = _mm512_castsi512_ps(_mm512_unpackhi_epi32(tmpreg, tmpreg));
}

template <bool LowBits>
static inline __m512 broadcast_ps_1_2(__m512 vsrc_y, const __m512i& vshuf_index_high, const __m512i& vshuf_index_low) {
  __m512 tmp;
  if constexpr (LowBits) {
    tmp = _mm512_permutex2var_ps(vsrc_y, vshuf_index_low, vsrc_y);
  } else {
    tmp = _mm512_permutex2var_ps(vsrc_y, vshuf_index_high, vsrc_y);
  }
  return tmp;
}

template <bool LowBits>
static inline __m512i broadcast_epi32_1_2(__m512i vsrc_y, const __m512i& vshuf_index_high,
                                          const __m512i& vshuf_index_low) {
  return _mm512_castps_si512(broadcast_ps_1_2<LowBits>(_mm512_castsi512_ps(vsrc_y), vshuf_index_high, vshuf_index_low));
}

static inline void vec_broadcast_epi32_1_2(__m512i* dst2regs, __m512i* src1regs, __m512i idxreg) {
  auto tmpreg = _mm512_permutexvar_epi64(idxreg, src1regs[0]);
  dst2regs[0] = _mm512_unpacklo_epi32(tmpreg, tmpreg);
  dst2regs[1] = _mm512_unpackhi_epi32(tmpreg, tmpreg);
}

static inline void vec_broadcast_pi8_1_2(__m128i* dst2regs, __m128i* src1regs, __m128i idxreg) {
  auto tmpreg = _mm_permutexvar_epi16(idxreg, src1regs[0]);
  dst2regs[0] = _mm_unpacklo_epi8(tmpreg, tmpreg);
  dst2regs[1] = _mm_unpackhi_epi8(tmpreg, tmpreg);
}

static inline void vec_broadcast_epi32_2_4(__m512i* dst4regs, __m512i* src2regs) {
  vec_broadcast_epi32_1_2(dst4regs, src2regs);
  vec_broadcast_epi32_1_2(dst4regs + 2, src2regs + 1);
}

static inline __m512i _mm512_sign_epi8(__m512i a, __m512i b) {
  __m512i zero = _mm512_setzero_si512();
  __mmask64 blt0 = _mm512_movepi8_mask(b);
  return _mm512_mask_sub_epi8(a, blt0, zero, a);
}

template <typename _ST, typename _DT, bool _IS_SYM>
static inline BTLA_CODE decompress_kblock_bit4_packrow1(utils::bit4x2* srcptr, _DT* dstptr, int row, int col,
                                                        int ld_src, int ld_dst, _ST* scales, int8_t* zero_points,
                                                        int k_offset, int kblock, int NPad,
                                                        void (*dequantize)(_DT*, int8_t*, __m512*, __m512i*),
                                                        void (*pad_bit4)(int8_t*, int8_t*, __m512i, int),
                                                        int8_t* tmpbuf, size_t tmpsize) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  if (col == 48) {
    constexpr int ColTile = 48;
    constexpr int NRegs = ColTile / 16;
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    constexpr int LoadMask48 = (1 << (48 / 8)) - 1;
    __m512 vscales[NRegs];
    __m512i vzps[NRegs];
    int constexpr UnrollRow = 4;
    int constexpr Loop64 = ColTile * UnrollRow / 64;
    assert(tmpsize >= (ColTile * UnrollRow));
    int row0 = kblock - k_offset % kblock;
    row0 = row0 == kblock ? 0 : row0;
    row0 = row0 > row ? row : row0;
    int row1 = row - row0;
    int irow = 0;
    if (row0) {
      int rowpad4 = utils::padto_le(row0, UnrollRow);
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16);
        if constexpr (!_IS_SYM) {
          auto tmp =
              _mm_loadu_si128(reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16));
          vzps[iv] = _mm512_cvtepi8_epi32(tmp);
        }
      }
      for (; irow < rowpad4; irow += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          pad_bit4(tmpbuf + iter64 * 64, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + 32 * iter64), zmm_mask,
                   LoadMask64);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          if constexpr (_IS_SYM) {
            dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * ColTile, vscales, nullptr);
          } else {
            dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * ColTile, vscales, vzps);
          }
        }
      }
      for (; irow < row0; irow++) {
        pad_bit4(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2), zmm_mask, LoadMask48);
        if constexpr (_IS_SYM) {
          dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, nullptr);
        } else {
          dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, vzps);
        }
      }
    }

    int row1_blk = utils::padto_le(row1, kblock) + row0;
    assert(kblock % UnrollRow == 0);
    assert(ld_src == 48);  // no padding for unroll process

    for (; irow < row1_blk; irow += kblock) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16);
        if constexpr (!_IS_SYM) {
          auto tmp =
              _mm_loadu_si128(reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16));
          vzps[iv] = _mm512_cvtepi8_epi32(tmp);
        }
      }

      for (int irr = 0; irr < kblock; irr += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          pad_bit4(tmpbuf + iter64 * 64, reinterpret_cast<int8_t*>(srcptr + (irow + irr) * ld_src / 2 + 32 * iter64),
                   zmm_mask, LoadMask64);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          if constexpr (_IS_SYM) {
            dequantize(dstptr + (irow + irr + iterr) * ld_dst, tmpbuf + iterr * ColTile, vscales, nullptr);
          } else {
            dequantize(dstptr + (irow + irr + iterr) * ld_dst, tmpbuf + iterr * ColTile, vscales, vzps);
          }
        }
      }
    }
    if (irow < row) {
      for (int iv = 0; iv < 3; iv++) {
        vscales[iv] = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16);
        if constexpr (!_IS_SYM) {
          auto tmp =
              _mm_loadu_si128(reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16));
          vzps[iv] = _mm512_cvtepi8_epi32(tmp);
        }
      }
      auto rowre = row - irow;
      int rowpad4 = utils::padto_le(rowre, UnrollRow) + irow;
      for (; irow < rowpad4; irow += UnrollRow) {
        for (int iter64 = 0; iter64 < Loop64; iter64++) {
          pad_bit4(tmpbuf + iter64 * 64, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + 32 * iter64), zmm_mask,
                   LoadMask64);
        }
        for (int iterr = 0; iterr < UnrollRow; iterr++) {
          if constexpr (_IS_SYM) {
            dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * ColTile, vscales, nullptr);
          } else {
            dequantize(dstptr + (irow + iterr) * ld_dst, tmpbuf + iterr * ColTile, vscales, vzps);
          }
        }
      }
      for (; irow < row; irow++) {
        pad_bit4(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2), zmm_mask, LoadMask48);
        if constexpr (_IS_SYM) {
          dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, nullptr);
        } else {
          dequantize(dstptr + irow * ld_dst, tmpbuf, vscales, vzps);
        }
      }
    }
    return BTLA_CODE::Success;
  }
  return BTLA_CODE::NotSupport;
}

template <BTLA_DTYPE _SRCT, typename _ST, typename _DT, bool _IS_SYM = true>
static inline BTLA_CODE decompress_kblock_bit4_packrow2(utils::bit4x2* srcptr, _DT* dstptr, int row, int col,
                                                        int ld_src, int ld_dst, _ST* scales, int8_t* zero_points,
                                                        int k_offset, int kblock, int NPad, int8_t* tmpbuf,
                                                        size_t tmpsize) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto broadcast_idx = _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7);
  auto broadcast_idx_128 = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
  auto constexpr SRC_TYPE =
      static_cast<BTLA_DTYPE>(utils::bestla_dtype_get_mask_val(_SRCT, BTLA_DTYPE::TypeMask, BTLA_DTYPE::TypeShift));
  if (col % 64 == 0) {
    constexpr int ColTile = 64;
    constexpr int NRegs = ColTile / 16;
    constexpr int ScaleRegCount = NRegs / 2;
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    for (int icol = 0; icol < col; icol += ColTile) {
      __m512 vscales[NRegs];
      __m512i vzps[NRegs];
      assert(tmpsize >= ColTile);
      int row0 = kblock - k_offset % kblock;
      row0 = row0 == kblock ? 0 : row0;
      row0 = row0 > row ? row : row0;
      int row1 = row - row0;
      int irow = 0;
      if (row0) {
        for (int iv = 0; iv < ScaleRegCount; iv++) {
          auto tmpscale = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp = _mm_loadu_si128(
                reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            auto vzp = _mm512_cvtepi8_epi32(tmpzp);
            vec_broadcast_epi32_1_2(vzps + iv * 2, &vzp, broadcast_idx);
          }
        }

        for (; irow < row0; irow++) {
          convert_s4_s8_highbits(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + icol / 2), zmm_mask,
                                 LoadMask64);
          if constexpr (SRC_TYPE == BTLA_DTYPE::TypeFloat) {
            dequant_f4_N<ColTile, _DT, _SRCT>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
          } else {
            dequant_s8_N<ColTile, _DT, _IS_SYM>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
          }
        }
      }

      int row1_blk = utils::padto_le(row1, kblock) + row0;
      for (; irow < row1_blk; irow += kblock) {
        for (int iv = 0; iv < ScaleRegCount; iv++) {
          auto tmpscale = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp = _mm_loadu_si128(
                reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            auto vzp = _mm512_cvtepi8_epi32(tmpzp);
            vec_broadcast_epi32_1_2(vzps + iv * 2, &vzp, broadcast_idx);
          }
        }

        for (int irr = 0; irr < kblock; irr += 1) {
          convert_s4_s8_highbits(tmpbuf, reinterpret_cast<int8_t*>(srcptr + (irow + irr) * ld_src / 2 + icol / 2),
                                 zmm_mask, LoadMask64);
          if constexpr (SRC_TYPE == BTLA_DTYPE::TypeFloat) {
            dequant_f4_N<ColTile, _DT, _SRCT>(dstptr + (irow + irr) * ld_dst + icol, tmpbuf, vscales, vzps);
          } else {
            dequant_s8_N<ColTile, _DT, _IS_SYM>(dstptr + (irow + irr) * ld_dst + icol, tmpbuf, vscales, vzps);
          }
        }
      }
      if (irow < row) {
        for (int iv = 0; iv < ScaleRegCount; iv++) {
          auto tmpscale = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp = _mm_loadu_si128(
                reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            auto vzp = _mm512_cvtepi8_epi32(tmpzp);
            vec_broadcast_epi32_1_2(vzps + iv * 2, &vzp, broadcast_idx);
          }
        }
      }
      for (; irow < row; irow++) {
        convert_s4_s8_highbits(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + icol / 2), zmm_mask,
                               LoadMask64);
        if constexpr (SRC_TYPE == BTLA_DTYPE::TypeFloat) {
          dequant_f4_N<ColTile, _DT, _SRCT>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
        } else {
          dequant_s8_N<ColTile, _DT, _IS_SYM>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
        }
      }
    }
    return BTLA_CODE::Success;
  } else if (col % 96 == 0) {
    constexpr int ColTile = 96;
    constexpr int NRegs = ColTile / 16;
    constexpr int ScaleRegCount = NRegs / 2;
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    for (int icol = 0; icol < col; icol += ColTile) {
      __m512 vscales[NRegs];
      __m512i vzps[NRegs];
      assert(tmpsize >= ColTile);
      int row0 = kblock - k_offset % kblock;
      row0 = row0 == kblock ? 0 : row0;
      row0 = row0 > row ? row : row0;
      int row1 = row - row0;
      int irow = 0;
      if (row0) {
        for (int iv = 0; iv < ScaleRegCount; iv++) {
          auto tmpscale = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp = _mm_loadu_si128(
                reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            auto vzp = _mm512_cvtepi8_epi32(tmpzp);
            vec_broadcast_epi32_1_2(vzps + iv * 2, &vzp, broadcast_idx);
          }
        }

        for (; irow < row0; irow++) {
          convert_s4_s8_highbits(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + icol / 2), zmm_mask,
                                 LoadMask64);
          convert_s4_s8_highbits_v32(tmpbuf + 64, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + icol / 2 + 32),
                                     zmm_mask, LoadMask64);
          if constexpr (SRC_TYPE == BTLA_DTYPE::TypeFloat) {
            dequant_f4_N<ColTile, _DT, _SRCT>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
          } else {
            dequant_s8_N<ColTile, _DT, _IS_SYM>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
          }
        }
      }

      int row1_blk = utils::padto_le(row1, kblock) + row0;
      for (; irow < row1_blk; irow += kblock) {
        for (int iv = 0; iv < ScaleRegCount; iv++) {
          auto tmpscale = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp = _mm_loadu_si128(
                reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            auto vzp = _mm512_cvtepi8_epi32(tmpzp);
            vec_broadcast_epi32_1_2(vzps + iv * 2, &vzp, broadcast_idx);
          }
        }

        for (int irr = 0; irr < kblock; irr += 1) {
          convert_s4_s8_highbits(tmpbuf, reinterpret_cast<int8_t*>(srcptr + (irow + irr) * ld_src / 2 + icol / 2),
                                 zmm_mask, LoadMask64);
          convert_s4_s8_highbits_v32(tmpbuf + 64,
                                     reinterpret_cast<int8_t*>(srcptr + (irow + irr) * ld_src / 2 + icol / 2 + 32),
                                     zmm_mask, LoadMask64);
          if constexpr (SRC_TYPE == BTLA_DTYPE::TypeFloat) {
            dequant_f4_N<ColTile, _DT, _SRCT>(dstptr + (irow + irr) * ld_dst + icol, tmpbuf, vscales, vzps);
          } else {
            dequant_s8_N<ColTile, _DT, _IS_SYM>(dstptr + (irow + irr) * ld_dst + icol, tmpbuf, vscales, vzps);
          }
        }
      }
      if (irow < row) {
        for (int iv = 0; iv < ScaleRegCount; iv++) {
          auto tmpscale = load_T_fp32(scales + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2);
          vec_broadcast_ps_1_2(vscales + iv * 2, &tmpscale, broadcast_idx);
          if constexpr (!_IS_SYM) {
            auto tmpzp = _mm_loadu_si128(
                reinterpret_cast<__m128i*>(zero_points + (k_offset + irow) / kblock * NPad + iv * 16 + icol / 2));
            auto vzp = _mm512_cvtepi8_epi32(tmpzp);
            vec_broadcast_epi32_1_2(vzps + iv * 2, &vzp, broadcast_idx);
          }
        }
      }
      for (; irow < row; irow++) {
        convert_s4_s8_highbits(tmpbuf, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + icol / 2), zmm_mask,
                               LoadMask64);
        convert_s4_s8_highbits_v32(tmpbuf + 64, reinterpret_cast<int8_t*>(srcptr + irow * ld_src / 2 + icol / 2 + 32),
                                   zmm_mask, LoadMask64);
        if constexpr (SRC_TYPE == BTLA_DTYPE::TypeFloat) {
          dequant_f4_N<ColTile, _DT, _SRCT>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
        } else {
          dequant_s8_N<ColTile, _DT, _IS_SYM>(dstptr + irow * ld_dst + icol, tmpbuf, vscales, vzps);
        }
      }
    }

    return BTLA_CODE::Success;
  }
  return BTLA_CODE::NotSupport;
}

template <bool WITH_SCALE, typename _DST_T, int _PACK_ROW, typename _S_T>
inline BTLA_CODE decompress_kblock_f8_fp(utils::f8* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
                                         _S_T* scales, int k_offset, int kblock, int NPad, BTLA_DTYPE src_f8_type) {
  int align_col = col / 16 * 16;
  int col_tail = col - align_col;
  auto ebits = utils::bestla_dtype_get_f8_ebits(src_f8_type);
  auto mantissabit = 7 - ebits;
  auto sign_revert_and_mask = _mm512_set1_epi32(0x80000000);
  auto e_revert_and_mask = _mm512_set1_epi32(0x0000007f);
  auto e_revert_shift = _mm512_set1_epi32(1);
  e_revert_shift = _mm512_slli_epi32(e_revert_shift, ebits - 1);
  e_revert_shift = _mm512_sub_epi32(e_revert_shift, _mm512_set1_epi32(128));
  auto mantissa_revert_and_mask = _mm512_set1_epi32(0x007fffff);
  auto packrow2_permute_idx = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    int j = 0;
    auto quant = [&](__mmask16 mask) {
      auto sign_revert =
          _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(mask, reinterpret_cast<__m128i*>(srcptr + i * ld_src + j)));
      auto e_revert = sign_revert;
      auto mantissa_revert = sign_revert;
      sign_revert = _mm512_slli_epi32(sign_revert, 24);
      sign_revert = _mm512_and_epi32(sign_revert, sign_revert_and_mask);
      e_revert = _mm512_and_epi32(e_revert, e_revert_and_mask);
      e_revert = _mm512_srli_epi32(e_revert, mantissabit);
      e_revert = _mm512_sub_epi32(e_revert, e_revert_shift);
      if constexpr (WITH_SCALE && std::is_same_v<_S_T, utils::f8>) {
        auto scale = _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(sptr + j / _PACK_ROW)));
        if constexpr (_PACK_ROW == 2) scale = _mm512_permutexvar_epi32(packrow2_permute_idx, scale);
        e_revert = _mm512_add_epi32(e_revert, scale);
      }
      e_revert = _mm512_slli_epi32(e_revert, 23);
      mantissa_revert = _mm512_slli_epi32(mantissa_revert, 23 - mantissabit);
      mantissa_revert = _mm512_and_epi32(mantissa_revert, mantissa_revert_and_mask);
      auto fp_v = _mm512_or_ps(_mm512_castsi512_ps(sign_revert), _mm512_castsi512_ps(e_revert));
      fp_v = _mm512_or_ps(fp_v, _mm512_castsi512_ps(mantissa_revert));
      if constexpr (WITH_SCALE && std::is_same_v<_S_T, float>) {
        auto scale = _mm512_loadu_ps(sptr + j / _PACK_ROW);
        if constexpr (_PACK_ROW == 2) scale = _mm512_permutexvar_ps(packrow2_permute_idx, scale);
        fp_v = _mm512_mul_ps(fp_v, scale);
      }
      if constexpr (std::is_same_v<_DST_T, float>) {
        _mm512_mask_storeu_ps(dstptr + i * ld_dst + j, mask, fp_v);
      } else if constexpr (std::is_same_v<_DST_T, utils::bf16>) {
        auto bf16_v = zmm_cvt_fp32_bf16(fp_v);
        _mm256_mask_storeu_epi16(reinterpret_cast<__m256i*>(dstptr + i * ld_dst + j), mask, bf16_v);
      } else {
        assert(0);
      }
    };
    for (; j < align_col; j += 16) quant(_cvtu32_mask16(0xffff));
    if (col_tail > 0) quant(_cvtu32_mask16(0xffff >> (16 - col_tail)));
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE S4_T, typename _DST_T, int _PACK_ROW, typename _ST>
static inline BTLA_CODE decompress_kblock_s4_fp_Dep(utils::int4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                    int ld_dst, _ST* scales, int8_t* zero_points, int k_offset,
                                                    int kblock, int NPad, int8_t* tmp, size_t tmpsize) {
  if constexpr (_PACK_ROW == 1) {
    if (zero_points == nullptr) {
      return decompress_kblock_bit4_packrow1<_ST, _DST_T, true>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad,
          &dequant_s8_N<48, _DST_T, true>, &convert_s4_s8_highbits, tmp, tmpsize);
    } else {
      return decompress_kblock_bit4_packrow1<_ST, _DST_T, false>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad,
          &dequant_s8_N<48, _DST_T, false>, &convert_s4_s8_highbits, tmp, tmpsize);
    }
  } else if constexpr (_PACK_ROW == 2) {
    if (zero_points == nullptr) {
      return decompress_kblock_bit4_packrow2<S4_T, _ST, _DST_T, true>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad, tmp, tmpsize);
    } else {
      return decompress_kblock_bit4_packrow2<S4_T, _ST, _DST_T, false>(
          srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points, k_offset, kblock, NPad, tmp, tmpsize);
    }
  }
  return BTLA_CODE::NotSupport;
}

template <BTLA_DTYPE S3_T, typename _DST_T>
inline BTLA_CODE decompress_kblock_s3_s8fp(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, _DST_T* dstptr,
                                           int interleave_n_offset, int unpack_elt, int8_t* tmp, size_t tmpsize) {
  auto head_ignore_num = interleave_n_offset % 128;
  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  auto zmm_shift = _mm512_set1_epi32(5);

  auto bit3_interleave_decompress_pack128 = [&](utils::bit2x4* src1, utils::bit1x8* src2, int8_t* dst) {
    const __m256i lowMask = _mm256_set1_epi8(0x03);
    const __m256i bit2_data = _mm256_loadu_si256((const __m256i*)src1);
    auto ymm0 = _mm256_and_si256(lowMask, bit2_data);                        // uop:1 p:015
    auto ymm1 = _mm256_and_si256(lowMask, _mm256_srli_epi16(bit2_data, 2));  // uop:1 p:01
    auto ymm2 = _mm256_and_si256(lowMask, _mm256_srli_epi16(bit2_data, 4));
    auto ymm3 = _mm256_and_si256(lowMask, _mm256_srli_epi16(bit2_data, 6));
    auto zmm1 = _mm512_inserti32x8(_mm512_castsi256_si512(ymm0), ymm1, 0x1);  // lat3, tp1 uop1 p:5
    auto zmm2 = _mm512_inserti32x8(_mm512_castsi256_si512(ymm2), ymm3, 0x1);

    unsigned long long* bit1_ptr = reinterpret_cast<unsigned long long*>(src2);
    auto bit1_mask1 = _cvtu64_mask64(*bit1_ptr);
    auto bit1_mask2 = _cvtu64_mask64(*(bit1_ptr + 1));
    auto zmm1_ = _mm512_mask_mov_epi8(zmm_0x00, bit1_mask1, zmm_0x04);
    auto zmm2_ = _mm512_mask_mov_epi8(zmm_0x00, bit1_mask2, zmm_0x04);
    zmm1 = _mm512_add_epi8(zmm1, zmm1_);
    zmm2 = _mm512_add_epi8(zmm2, zmm2_);
    zmm1 = _mm512_sllv_epi32(zmm1, zmm_shift);  // int3_clip => int8
    zmm2 = _mm512_sllv_epi32(zmm2, zmm_shift);  // int3_clip => int8

    _mm512_storeu_si512((__m512i*)dst, zmm1);
    _mm512_storeu_si512((__m512i*)(dst + 64), zmm2);
  };
  int compress_wei_ptr_offset = 0;
  if (head_ignore_num != 0) {
    assert(head_ignore_num % 8 == 0);

    auto base_bit2ptr = bit2ptr - head_ignore_num / 4;
    auto base_bit1ptr = bit1ptr - head_ignore_num / 8;
    auto head_write_num = 128 - head_ignore_num;
    bit3_interleave_decompress_pack128(base_bit2ptr, base_bit1ptr, tmp);
    for (int i = 0; i < head_write_num; i++) dstptr[i] = tmp[head_ignore_num + i];
    compress_wei_ptr_offset += head_write_num;
    unpack_elt -= head_write_num;
  }
  auto body_loop = unpack_elt / 128;
  auto tail_proc_num = unpack_elt % 128;

  bestla::kernel::jit::DecompressS3::forward_avx512f(bit2ptr + compress_wei_ptr_offset / 4,
                                                     bit1ptr + compress_wei_ptr_offset / 8,
                                                     dstptr + compress_wei_ptr_offset, tmp, body_loop * 128);
  compress_wei_ptr_offset += body_loop * 128;
  if (tail_proc_num > 0) {
    bit3_interleave_decompress_pack128(bit2ptr + compress_wei_ptr_offset / 4, bit1ptr + compress_wei_ptr_offset / 8,
                                       tmp);
    for (int i = 0; i < tail_proc_num; i++) dstptr[compress_wei_ptr_offset + i] = tmp[i];
  }
  return BTLA_CODE::Success;
}

template <BTLA_DTYPE _S3_T, typename _DST_T, int _PACK_ROW, typename _ST>
static inline BTLA_CODE decompress_kblock_bit3_packrow_fp(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                                          _DST_T* dstptr, int interleave_n_offset, int row, int col,
                                                          _ST* scales, int8_t* zero_points, int k_offset, int kblock,
                                                          int NPad, void* tmp, size_t tmpsize) {
  auto unpack_elt = row * col;
  decompress_kblock_s3_s8fp<_S3_T>(bit2ptr, bit1ptr, dstptr, interleave_n_offset, unpack_elt,
                                   reinterpret_cast<int8_t*>(tmp), tmpsize);
  // TODO(zhe): simd version
  for (int i = 0; i < row; i++) {
    int kpos = (k_offset + i) / kblock;
    auto sptr = scales + kpos * NPad;
    for (int j = 0; j < col; j++) {
      float tmp = static_cast<float>(dstptr[i * col + j]);
      if (zero_points != nullptr) tmp -= static_cast<float>(zero_points[kpos * NPad + j / _PACK_ROW]);
      dstptr[i * col + j] = static_cast<_DST_T>(tmp * sptr[j / _PACK_ROW]);
    }
  }

  return BTLA_CODE::Success;
}

template <BTLA_DTYPE _F4_T, typename _DST_T, int _PACK_ROW, typename _ST>
static inline BTLA_CODE decompress_kblock_f4_fp(utils::f4x2* srcptr, _DST_T* dstptr, int row, int col, int ld_src,
                                                int ld_dst, _ST* scales, int k_offset, int kblock, int NPad,
                                                int8_t* tmp, size_t tmpsize) {
  if constexpr (_PACK_ROW == 1) {
    return decompress_kblock_bit4_packrow1<_ST, _DST_T, true>(srcptr, dstptr, row, col, ld_src, ld_dst, scales, nullptr,
                                                              k_offset, kblock, NPad, &dequant_f4_N<48, _DST_T, _F4_T>,
                                                              pad_fp4, tmp, tmpsize);
  } else if constexpr (_PACK_ROW == 2) {
    return decompress_kblock_bit4_packrow2<_F4_T, _ST, _DST_T, true>(srcptr, dstptr, row, col, ld_src, ld_dst, scales,
                                                                     nullptr, k_offset, kblock, NPad, tmp, tmpsize);
  }
  return BTLA_CODE::NotSupport;
}

template <BTLA_DTYPE F4_T, typename DST_T>
inline BTLA_CODE decompress_kblock_f4_fp_noscale(utils::f4x2* srcptr, DST_T* dstptr, int row, int col, int ld_src,
                                                 int ld_dst, int8_t* tmp, size_t tmpsize) {
  uint32_t mask = 0xf0f0f0f0;
  auto zmm_mask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  if (col == ld_src) {
    size_t elesize = (size_t)row * col;
    size_t ele256 = utils::padto_le(elesize, 256);
    size_t ele64 = utils::padto_le(elesize, 64);
    assert(tmpsize >= 256);
    size_t i = 0;
    constexpr int LoadMask64 = (1 << (64 / 8)) - 1;
    for (; i < ele256; i += 256) {
      pad_fp4(tmp + 0, reinterpret_cast<int8_t*>(srcptr + i / 2 + 0), zmm_mask, LoadMask64);
      pad_fp4(tmp + 64, reinterpret_cast<int8_t*>(srcptr + i / 2 + 32), zmm_mask, LoadMask64);
      pad_fp4(tmp + 128, reinterpret_cast<int8_t*>(srcptr + i / 2 + 64), zmm_mask, LoadMask64);
      pad_fp4(tmp + 192, reinterpret_cast<int8_t*>(srcptr + i / 2 + 96), zmm_mask, LoadMask64);
      for (size_t j = 0; j < 256; j += 64) {
        unpack_f4_N<64, DST_T, F4_T>(dstptr + i + j, tmp + j);
      }
    }
    if (i + 64 <= ele64) {
      for (; i < ele64; i += 64) {
        pad_fp4(tmp, reinterpret_cast<int8_t*>(srcptr + i / 2), zmm_mask, LoadMask64);
        unpack_f4_N<64, DST_T, F4_T>(dstptr + i, tmp);
      }
    }
    for (; i < elesize; i += 2) {
      auto tmp = srcptr[i / 2];
      dstptr[i + 0] = static_cast<DST_T>(ref::f4_unpack<F4_T>(tmp.x));
      dstptr[i + 1] = static_cast<DST_T>(ref::f4_unpack<F4_T>(tmp.y));
    }
    return BTLA_CODE::Success;
  }
  return BTLA_CODE::NotSupport;
}

static inline BTLA_CODE quantize_f32_sign_int_rowblock_sym(const float* srcptr, int8_t* dstptr, int row, int col,
                                                           int ld_src, int ld_dst, float* scales, int blocksize) {
  int constexpr VLen = 16;
  auto v127 = _mm512_set1_ps(127.f);
  int col16 = utils::padto_le(col, 16);
  int i = 0;
  auto align_row = row / blocksize * blocksize;
  for (; i < col16; i += VLen) {
    int j = 0;
    auto simd_process_block = [&](int size) {
      __m512 vscale;
      __m512 vmaxval = _mm512_set1_ps(0.f);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_abs_ps(vsrc);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
      }
      vscale = _mm512_div_ps(vmaxval, v127);
      auto vrscale = _mm512_div_ps(v127, vmaxval);
      _mm512_storeu_ps(&scales[j / blocksize * ld_dst + i], vscale);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dstptr[(j + ij) * ld_dst + i]), vbsrc);
      }
    };
    for (; j < align_row; j += blocksize) simd_process_block(blocksize);
    if (j < row) simd_process_block(row - align_row);
  }
  for (; i < col; i++) {
    int j = 0;
    auto scalar_process_block = [&](int size) {
      float maxval = std::numeric_limits<float>::min();
      for (size_t ij = 0; ij < size; ij++) {
        maxval = std::max(maxval, std::abs(srcptr[(j + ij) * ld_src + i]));
      }
      float scale = maxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize * ld_dst + i] = scale;
      for (size_t ij = 0; ij < size; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>(srcptr[(j + ij) * ld_src + i] * rscale);
      }
    };
    for (; j < align_row; j += blocksize) scalar_process_block(blocksize);
    if (j < row) scalar_process_block(row - align_row);
  }
  return BTLA_CODE::Success;
}
static inline BTLA_CODE quantize_f32_sign_int_rowblock_sym_auto(const float* srcptr, int8_t* dstptr, int row, int col,
                                                                int ld_src, int ld_dst, float* scales, int blocksize,
                                                                BTLA_DTYPE qtype) {
  int constexpr VLen = 16;
  int col16 = utils::padto_le(col, VLen);
  int i = 0;
  auto NBits = utils::bestla_dtype_bits(qtype);
  int FullValue = 1 << (NBits - 1);
  int SymValue = FullValue - 1;
  auto align_row = row / blocksize * blocksize;
  for (; i < col16; i += VLen) {
    int j = 0;
    float tmp_min[VLen];
    float tmp_max[VLen];
    float tmp_abs[VLen];
    auto simd_process_block = [&](int size) {
      __m512 vscale;
      __m512 vmaxval = _mm512_set1_ps(std::numeric_limits<float>::min());
      __m512 vminval = _mm512_set1_ps(std::numeric_limits<float>::max());
      __m512 vabsval = _mm512_set1_ps(0.f);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
        vminval = _mm512_min_ps(vminval, vsrc);
        vsrc = _mm512_abs_ps(vsrc);
        vabsval = _mm512_max_ps(vabsval, vsrc);
      }
      _mm512_storeu_ps(tmp_min, vminval);
      _mm512_storeu_ps(tmp_max, vmaxval);
      _mm512_storeu_ps(tmp_abs, vabsval);

      for (int iv = 0; iv < VLen; iv++) {
        int NVal = SymValue;
        auto sum = tmp_max[iv] + tmp_min[iv];
        if (abs(sum) >= tmp_abs[iv] / FullValue) {
          NVal = sum > 0.f ? -FullValue : FullValue;
        }
        NVal = NVal << (8 - NBits);
        tmp_abs[iv] = static_cast<float>(NVal);
      }
      auto vmag = _mm512_loadu_ps(tmp_abs);
      vscale = _mm512_div_ps(vabsval, vmag);
      auto vrscale = _mm512_div_ps(vmag, vabsval);
      _mm512_storeu_ps(&scales[j / blocksize * ld_dst + i], vscale);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_mul_ps(vsrc, vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dstptr[(j + ij) * ld_dst + i]), vbsrc);
      }
    };
    for (; j < align_row; j += blocksize) simd_process_block(blocksize);
    if (j < row) simd_process_block(row - align_row);
  }
  kernel::ref::quantize_f32_sign_int_rowblock(srcptr + i, dstptr + i, row, col - i, ld_src, ld_dst, scales + i, nullptr,
                                              blocksize, qtype);
  return BTLA_CODE::Success;
}

static inline BTLA_CODE quantize_f32_sign_int_rowblock_asym(const float* srcptr, int8_t* dstptr, int row, int col,
                                                            int ld_src, int ld_dst, float* scales, int8_t* zero_points,
                                                            int blocksize) {
  int constexpr VLen = 16;
  auto v255 = _mm512_set1_ps(255.f);
  auto v2 = _mm512_set1_ps(2.f);
  auto v0 = _mm512_set1_ps(0.f);
  int col16 = utils::padto_le(col, 16);
  int i = 0;
  auto align_row = row / blocksize * blocksize;
  for (; i < col16; i += VLen) {
    int j = 0;
    auto simd_process_block = [&](int size) {
      __m512 vscale;
      __m512 vzp;
      __m512 vmaxval = v0;
      __m512 vminval = vmaxval;
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vmaxval = _mm512_max_ps(vmaxval, vsrc);
        vminval = _mm512_min_ps(vminval, vsrc);
      }
      auto vsub = _mm512_sub_ps(vmaxval, vminval);
      vscale = _mm512_div_ps(vsub, v255);
      auto vrscale = _mm512_div_ps(v255, vsub);
      _mm512_storeu_ps(&scales[j / blocksize * ld_dst + i], vscale);
      auto vsum = _mm512_add_ps(vmaxval, vminval);
      auto vmedium = _mm512_div_ps(vsum, v2);
      vzp = _mm512_mul_ps(_mm512_sub_ps(v0, vmedium), vrscale);
      auto vbzp = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(vzp));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(&zero_points[j / blocksize * ld_dst + i]), vbzp);
      for (size_t ij = 0; ij < size; ij++) {
        auto vsrc = _mm512_loadu_ps(&srcptr[(j + ij) * ld_src + i]);
        vsrc = _mm512_mul_ps(_mm512_sub_ps(vsrc, vmedium), vrscale);
        auto vdsrc = _mm512_cvtps_epi32(vsrc);
        auto vbsrc = _mm512_cvtsepi32_epi8(vdsrc);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&dstptr[(j + ij) * ld_dst + i]), vbsrc);
      }
    };
    for (; j < align_row; j += blocksize) simd_process_block(blocksize);
    if (j < row) simd_process_block(row - align_row);
  }
  for (; i < col; i++) {
    int j = 0;
    auto scalar_process_block = [&](int size) {
      float maxval = 0;
      float minval = 0;
      for (size_t ij = 0; ij < size; ij++) {
        maxval = std::max(maxval, srcptr[(j + ij) * ld_src + i]);
        minval = std::min(maxval, srcptr[(j + ij) * ld_src + i]);
      }
      float scale = (maxval - minval) / 255.f;
      float rscale = 1.f / scale;
      scales[j / blocksize * ld_dst + i] = scale;
      float fmedium = (maxval + minval) / 2.f;
      int8_t bzp = utils::cast<float, int8_t>((0 - fmedium) * rscale);
      zero_points[j / blocksize * ld_dst + i] = bzp;
      for (size_t ij = 0; ij < size; ij++) {
        dstptr[(j + ij) * ld_dst + i] = utils::cast<float, int8_t>((srcptr[(j + ij) * ld_src + i] - fmedium) * rscale);
      }
    };
    for (; j < align_row; j += blocksize) scalar_process_block(blocksize);
    if (j < row) scalar_process_block(row - align_row);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE quantize_f32_sign_int_rowblock(const float* srcptr, int8_t* dstptr, int row, int col,
                                                       int ld_src, int ld_dst, float* scales, int8_t* zero_points,
                                                       int blocksize, BTLA_DTYPE qtype) {
  if (zero_points == nullptr)
    if (qtype == BTLA_DTYPE::S4_CLIP || qtype == BTLA_DTYPE::S3_CLIP) {
      return quantize_f32_sign_int_rowblock_sym_auto(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize,
                                                     qtype);
    } else {
      return quantize_f32_sign_int_rowblock_sym(srcptr, dstptr, row, col, ld_src, ld_dst, scales, blocksize);
    }
  else
    return quantize_f32_sign_int_rowblock_asym(srcptr, dstptr, row, col, ld_src, ld_dst, scales, zero_points,
                                               blocksize);
}

static float F4_NF4_quant_sub_helper[] = {0.f,         0.23746347f, 0.38810113f, 0.50841697f, 0.61348899f, 0.71018467f,
                                          0.80257138f, 0.88788655f, 0.96835165f, 1.05161765f, 1.14011017f, 1.23740894f,
                                          1.34975982f, 1.49088332f, 1.70957482f, 2.0f};
static float F4_BNB_quant_sub_helper[] = {0.00260417f, 0.0859375f, 0.20833333f, 0.29166667f,
                                          0.4166667f,  0.583333f,  0.8333333f,  1.01f};
static float F4_E2M1_quant_sub_helper[] = {0.00520833f, 0.08854167f, 0.20833333f, 0.29166667f,
                                           0.41666667f, 0.58333333f, 0.83333333f, 1.01f};
constexpr static int8_t F4_NF4_simd_quant_v[] = {0b0111, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0000,
                                                 0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111};
constexpr static int8_t F4_BNB_simd_quant_v[] = {0b0000, 0b0001, 0b0110, 0b0111, 0b0100, 0b0101, 0b0010, 0b0011};
constexpr static int8_t F4_E2M1_simd_quant_v[] = {0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111};

template <std::size_t N, std::size_t... I>
constexpr auto broadcast_N_2_Nx16(const int8_t* arr, std::index_sequence<I...>) {
  return std::array<int8_t, N * 16>{(arr[I / 16])...};
}

template <std::size_t N>
constexpr auto broadcast_N_2_Nx16(const int8_t* arr) {
  return broadcast_N_2_Nx16<N>(arr, std::make_index_sequence<N * 16>{});
}

template <BTLA_DTYPE F4_T>
inline void f32_f4_quantize_4x16(const float* srcptr, int8_t* dstptr, int ld_src, int ld_dst,
                                 const int8_t* broadcast_f4_v, float* scales, __mmask16 ls_mask) {
  __m128i xmm0{}, xmm1{}, xmm2{}, xmm3{};
  __m512 zmm0{}, zmm1{}, zmm2{}, zmm3{}, zmm4, zmm5, zmm6, zmm7, zmm_scale{};
  __mmask16 mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask7;
  zmm_scale = _mm512_rcp14_ps(_mm512_mask_loadu_ps(zmm_scale, ls_mask, scales));
  auto avoid_double_cmp = _mm512_set1_ps(100.f);
  auto zmm_v0 = _mm512_set1_ps(0.f);
  zmm0 = _mm512_mask_loadu_ps(zmm0, ls_mask, srcptr);
  zmm1 = _mm512_mask_loadu_ps(zmm1, ls_mask, srcptr + 1 * ld_src);
  zmm2 = _mm512_mask_loadu_ps(zmm2, ls_mask, srcptr + 2 * ld_src);
  zmm3 = _mm512_mask_loadu_ps(zmm3, ls_mask, srcptr + 3 * ld_src);
  zmm0 = _mm512_mul_ps(zmm0, zmm_scale);
  zmm1 = _mm512_mul_ps(zmm1, zmm_scale);
  zmm2 = _mm512_mul_ps(zmm2, zmm_scale);
  zmm3 = _mm512_mul_ps(zmm3, zmm_scale);
  if constexpr (F4_T == BTLA_DTYPE::F4_NF4) {
    auto zmm_zp = _mm512_set1_ps(0.8480964004993439f);
    zmm0 = _mm512_add_ps(zmm0, zmm_zp);
    zmm1 = _mm512_add_ps(zmm1, zmm_zp);
    zmm2 = _mm512_add_ps(zmm2, zmm_zp);
    zmm3 = _mm512_add_ps(zmm3, zmm_zp);
  } else {
    mask4 = _mm512_cmp_ps_mask(zmm0, zmm_v0, 1);
    mask5 = _mm512_cmp_ps_mask(zmm1, zmm_v0, 1);
    mask6 = _mm512_cmp_ps_mask(zmm2, zmm_v0, 1);
    mask7 = _mm512_cmp_ps_mask(zmm3, zmm_v0, 1);

    zmm0 = _mm512_abs_ps(zmm0);
    zmm1 = _mm512_abs_ps(zmm1);
    zmm2 = _mm512_abs_ps(zmm2);
    zmm3 = _mm512_abs_ps(zmm3);
  }
  constexpr int loop_num = F4_T == BTLA_DTYPE::F4_NF4 ? 16 : 8;
  for (int i = 0; i < loop_num; i++) {
    __m512 sub_v;
    if constexpr (F4_T == BTLA_DTYPE::F4_NF4) sub_v = _mm512_set1_ps(F4_NF4_quant_sub_helper[i]);
    if constexpr (F4_T == BTLA_DTYPE::F4_BNB) sub_v = _mm512_set1_ps(F4_BNB_quant_sub_helper[i]);
    if constexpr (F4_T == BTLA_DTYPE::F4_E2M1) sub_v = _mm512_set1_ps(F4_E2M1_quant_sub_helper[i]);
    zmm4 = _mm512_sub_ps(zmm0, sub_v);
    zmm5 = _mm512_sub_ps(zmm1, sub_v);
    zmm6 = _mm512_sub_ps(zmm2, sub_v);
    zmm7 = _mm512_sub_ps(zmm3, sub_v);
    mask0 = _mm512_cmp_ps_mask(zmm4, zmm_v0, 2);
    mask1 = _mm512_cmp_ps_mask(zmm5, zmm_v0, 2);
    mask2 = _mm512_cmp_ps_mask(zmm6, zmm_v0, 2);
    mask3 = _mm512_cmp_ps_mask(zmm7, zmm_v0, 2);
    xmm0 = _mm_mask_blend_epi8(mask0, xmm0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(broadcast_f4_v + i * 16)));
    xmm1 = _mm_mask_blend_epi8(mask1, xmm1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(broadcast_f4_v + i * 16)));
    xmm2 = _mm_mask_blend_epi8(mask2, xmm2, _mm_loadu_si128(reinterpret_cast<const __m128i*>(broadcast_f4_v + i * 16)));
    xmm3 = _mm_mask_blend_epi8(mask3, xmm3, _mm_loadu_si128(reinterpret_cast<const __m128i*>(broadcast_f4_v + i * 16)));
    zmm0 = _mm512_mask_add_ps(zmm0, mask0, zmm0, avoid_double_cmp);
    zmm1 = _mm512_mask_add_ps(zmm1, mask1, zmm1, avoid_double_cmp);
    zmm2 = _mm512_mask_add_ps(zmm2, mask2, zmm2, avoid_double_cmp);
    zmm3 = _mm512_mask_add_ps(zmm3, mask3, zmm3, avoid_double_cmp);
  }
  if constexpr (F4_T != BTLA_DTYPE::F4_NF4) {
    auto xmm_bias = _mm_set1_epi8(0x08);
    xmm0 = _mm_mask_add_epi8(xmm0, mask4, xmm0, xmm_bias);
    xmm1 = _mm_mask_add_epi8(xmm1, mask5, xmm1, xmm_bias);
    xmm2 = _mm_mask_add_epi8(xmm2, mask6, xmm2, xmm_bias);
    xmm3 = _mm_mask_add_epi8(xmm3, mask7, xmm3, xmm_bias);
  }
  _mm_mask_storeu_epi8(dstptr, ls_mask, xmm0);
  _mm_mask_storeu_epi8(dstptr + 1 * ld_dst, ls_mask, xmm1);
  _mm_mask_storeu_epi8(dstptr + 2 * ld_dst, ls_mask, xmm2);
  _mm_mask_storeu_epi8(dstptr + 3 * ld_dst, ls_mask, xmm3);
}

template <BTLA_DTYPE F4_T>
inline void f32_f4_quantize_1x16(const float* srcptr, int8_t* dstptr, int ld_src, int ld_dst,
                                 const int8_t* broadcast_f4_v, float* scales, __mmask16 ls_mask) {
  __m512 zmm0{}, zmm1, zmm_scale{};
  zmm_scale = _mm512_rcp14_ps(_mm512_mask_loadu_ps(zmm_scale, ls_mask, scales));
  auto avoid_double_cmp = _mm512_set1_ps(100.f);
  auto zmm_v0 = _mm512_set1_ps(0.f);
  __m128i xmm0{};
  __mmask16 mask0, mask1;
  zmm0 = _mm512_mask_loadu_ps(zmm0, ls_mask, srcptr);
  zmm0 = _mm512_mul_ps(zmm0, zmm_scale);
  if constexpr (F4_T == BTLA_DTYPE::F4_NF4) {
    auto zp = _mm512_set1_ps(0.8480964004993439f);
    zmm0 = _mm512_add_ps(zmm0, zp);
  } else {
    mask1 = _mm512_cmp_ps_mask(zmm0, zmm_v0, 1);
    zmm0 = _mm512_abs_ps(zmm0);
  }
  constexpr int loop_num = F4_T == BTLA_DTYPE::F4_NF4 ? 16 : 8;
  for (int i = 0; i < loop_num; i++) {
    __m512 sub_v;
    if constexpr (F4_T == BTLA_DTYPE::F4_NF4) sub_v = _mm512_set1_ps(F4_NF4_quant_sub_helper[i]);
    if constexpr (F4_T == BTLA_DTYPE::F4_BNB) sub_v = _mm512_set1_ps(F4_BNB_quant_sub_helper[i]);
    if constexpr (F4_T == BTLA_DTYPE::F4_E2M1) sub_v = _mm512_set1_ps(F4_E2M1_quant_sub_helper[i]);
    zmm1 = _mm512_sub_ps(zmm0, sub_v);
    mask0 = _mm512_cmp_ps_mask(zmm1, zmm_v0, 2);
    xmm0 = _mm_mask_blend_epi8(mask0, xmm0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(broadcast_f4_v + i * 16)));
    zmm0 = _mm512_mask_add_ps(zmm0, mask0, zmm0, avoid_double_cmp);
  }
  if constexpr (F4_T != BTLA_DTYPE::F4_NF4) {
    auto xmm_bias = _mm_set1_epi8(0x08);
    xmm0 = _mm_mask_add_epi8(xmm0, mask1, xmm0, xmm_bias);
  }
  _mm_mask_storeu_epi8(dstptr, ls_mask, xmm0);
}

inline void calc_blkx16_scale(const float* srcptr, int blocksize, int ld_src, float* scales, __mmask16 ls_mask) {
  auto absmax = _mm512_set1_ps(0.f);
  __m512 tmp{};
  for (int i = 0; i < blocksize; i++) {
    absmax = _mm512_range_ps(absmax, _mm512_mask_loadu_ps(tmp, ls_mask, srcptr + i * ld_src), 7);
  }
  _mm512_mask_storeu_ps(scales, ls_mask, absmax);
}

constexpr auto broadcast_F4_NF4_quantv = broadcast_N_2_Nx16<16>(F4_NF4_simd_quant_v);
constexpr auto broadcast_F4_BNB_quantv = broadcast_N_2_Nx16<8>(F4_BNB_simd_quant_v);
constexpr auto broadcast_F4_E2M1_quantv = broadcast_N_2_Nx16<8>(F4_E2M1_simd_quant_v);

template <BTLA_DTYPE F4_T>
inline BTLA_CODE quantize_f32_f4_rowblock(const float* srcptr, int8_t* dstptr, int row, int col, int ld_src, int ld_dst,
                                          float* scales, int blocksize) {
  // assert(col % 16 == 0);
  auto align_row = row / blocksize * blocksize;
  auto align_blk = blocksize / 4 * 4;
  int8_t* broadcast_f4_quantv;
  if constexpr (F4_T == BTLA_DTYPE::F4_NF4) broadcast_f4_quantv = const_cast<int8_t*>(broadcast_F4_NF4_quantv.data());
  if constexpr (F4_T == BTLA_DTYPE::F4_BNB) broadcast_f4_quantv = const_cast<int8_t*>(broadcast_F4_BNB_quantv.data());
  if constexpr (F4_T == BTLA_DTYPE::F4_E2M1) broadcast_f4_quantv = const_cast<int8_t*>(broadcast_F4_E2M1_quantv.data());
  int i = 0;
  int align_col = col / 16 * 16;

  auto process_row_blk = [&](int i, int col_size) {
    int j = 0;
    __mmask16 ls_mask = _cvtu32_mask16(0xffff >> (16 - col_size));
    for (; j < align_row; j += blocksize) {
      calc_blkx16_scale(srcptr + j * ld_src + i, blocksize, ld_src, scales + j / blocksize * ld_dst + i, ls_mask);
      int k = 0;
      for (; k < align_blk; k += 4) {
        f32_f4_quantize_4x16<F4_T>(srcptr + (j + k) * ld_src + i, dstptr + (j + k) * ld_dst + i, ld_src, ld_dst,
                                   broadcast_f4_quantv, scales + j / blocksize * ld_dst + i, ls_mask);
      }
      for (; k < blocksize; k++) {
        f32_f4_quantize_1x16<F4_T>(srcptr + (j + k) * ld_src + i, dstptr + (j + k) * ld_dst + i, ld_src, ld_dst,
                                   broadcast_f4_quantv, scales + j / blocksize * ld_dst + i, ls_mask);
      }
    }
    if (j < row) {
      auto fin_row = row - align_row;
      calc_blkx16_scale(srcptr + j * ld_src + i, fin_row, ld_src, scales + j / blocksize * ld_dst + i, ls_mask);
      int k = 0;
      auto align_fin_blk = fin_row / 4 * 4;
      for (; k < align_fin_blk; k += 4) {
        f32_f4_quantize_4x16<F4_T>(srcptr + (j + k) * ld_src + i, dstptr + (j + k) * ld_dst + i, ld_src, ld_dst,
                                   broadcast_f4_quantv, scales + j / blocksize * ld_dst + i, ls_mask);
      }
      for (; k < fin_row; k++) {
        f32_f4_quantize_1x16<F4_T>(srcptr + (j + k) * ld_src + i, dstptr + (j + k) * ld_dst + i, ld_src, ld_dst,
                                   broadcast_f4_quantv, scales + j / blocksize * ld_dst + i, ls_mask);
      }
    }
  };

  for (; i < align_col; i += 16) process_row_blk(i, 16);
  if (i < col) process_row_blk(i, col - i);

  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE quantize_fp_u8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, uint8_t* dstptr,
                                                int ld_dst, float* scales, int ld_scale, uint8_t* zps, int blocksize,
                                                float* blkreduce) {
  int constexpr VLen = 16;
  int constexpr Unroll = 2;
  auto vff = _mm512_set1_epi32(255);
  auto v0 = _mm512_set1_epi32(0);
  int vblocksize = utils::padto_le(blocksize, VLen);
  int vblocksize_un = utils::padto_le(blocksize, VLen * Unroll);
  int colblk = utils::padto_le(col, blocksize);
  for (size_t i = 0; i < row; i += 1) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      __m512 vmaxval = _mm512_set1_ps(0.f);
      __m512 vminval = _mm512_set1_ps(0.f);
      size_t ij = 0;
      for (; ij < vblocksize_un; ij += VLen * Unroll) {
        for (size_t iu = 0; iu < Unroll; iu++) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src + iu * VLen);
          vmaxval = _mm512_max_ps(vmaxval, vsrc);
          vminval = _mm512_min_ps(vminval, vsrc);
        }
      }
      if (ij + VLen < vblocksize) {
        for (; ij < vblocksize; ij += VLen) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src);
          vmaxval = _mm512_max_ps(vmaxval, vsrc);
          vminval = _mm512_min_ps(vminval, vsrc);
        }
      }

      auto maxval = _mm512_reduce_max_ps(vmaxval);
      auto minval = _mm512_reduce_min_ps(vminval);
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
          maxval = std::max(maxval, srcval);
          minval = std::min(minval, srcval);
        }
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      float rscale = 1.f / scale;
      auto vrscale = _mm512_set1_ps(rscale);
      auto vdzp = _mm512_set1_epi32(zp);
      int sum = 0;
      ij = 0;
      for (; ij < vblocksize_un; ij += VLen * Unroll) {
        for (size_t iu = 0; iu < Unroll; iu++) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src + iu * VLen);
          vsrc = _mm512_mul_ps(vsrc, vrscale);
          auto vdsrc = _mm512_cvtps_epi32(vsrc);
          if (blkreduce) {
            sum += _mm512_reduce_add_epi32(vdsrc);
          }
          vdsrc = _mm512_add_epi32(vdsrc, vdzp);
          vdsrc = _mm512_min_epi32(vdsrc, vff);
          vdsrc = _mm512_max_epi32(vdsrc, v0);
          auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(&dstptr[(j + ij) + i * ld_dst + iu * VLen]), vbsrc);
        }
      }
      if (ij + VLen < vblocksize) {
        for (; ij < vblocksize; ij += VLen) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src);
          vsrc = _mm512_mul_ps(vsrc, vrscale);
          auto vdsrc = _mm512_cvtps_epi32(vsrc);
          if (blkreduce) {
            sum += _mm512_reduce_add_epi32(vdsrc);
          }
          vdsrc = _mm512_add_epi32(vdsrc, vdzp);
          vdsrc = _mm512_min_epi32(vdsrc, vff);
          vdsrc = _mm512_max_epi32(vdsrc, v0);
          auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(&dstptr[(j + ij) + i * ld_dst]), vbsrc);
        }
      }
      for (; ij < blocksize; ij++) {
        auto srcval = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
        srcval = srcval * rscale;
        auto srcint = utils::cast<float, int>(srcval);
        sum += srcint;
        srcint += zp;
        srcint = std::min(srcint, 0xff);
        srcint = std::max(srcint, 0);
        dstptr[(j + ij) + i * ld_dst] = static_cast<uint8_t>(srcint);
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }

    if (j < col) {
      float maxval = 0.f;
      float minval = 0.f;
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[ij + i * ld_src]);
        maxval = std::max(fsrc, maxval);
        minval = std::min(fsrc, minval);
      }
      float scale = (maxval - minval) / 255;
      uint8_t zp = utils::cast<float, uint8_t>((0 - minval) / scale);
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      zps[j / blocksize + i * ld_scale] = zp;
      int sum = 0;
      for (size_t ij = j; ij < col; ij++) {
        auto fsrc = static_cast<float>(srcptr[ij + i * ld_src]);
        auto srcint = utils::cast<float, int>(fsrc * rscale);
        sum += srcint;
        srcint += zp;
        srcint = srcint <= 255 ? srcint : 255;
        srcint = srcint >= 0 ? srcint : 0;
        dstptr[ij + i * ld_dst] = srcint;
      }
      if (blkreduce) {
        blkreduce[j / blocksize + i * ld_scale] = sum * scale;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE quantize_fp_s8_colblock(int row, int col, const SRC_T* srcptr, int ld_src, int8_t* dstptr,
                                                int ld_dst, float* scales, int ld_scale, int blocksize, float* reduce) {
  int constexpr VLen = 16;
  auto vpos = _mm512_set1_epi32(127);
  auto vneg = _mm512_set1_epi32(-128);
  int VBlockSizeU3 = utils::padto_le(blocksize, VLen * 3);
  int VBlockSize = utils::padto_le(blocksize, VLen);
  int colblk = utils::padto_le(col, blocksize);
  for (int i = 0; i < row; i += 1) {
    size_t j = 0;
    for (; j < colblk; j += blocksize) {
      __m512 vmaxval = _mm512_set1_ps(std::numeric_limits<float>::min());
      size_t ij = 0;
      for (; ij < VBlockSizeU3; ij += VLen * 3) {
        for (int iu = 0; iu < 3; iu++) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src + iu * VLen);
          vsrc = _mm512_abs_ps(vsrc);
          vmaxval = _mm512_max_ps(vmaxval, vsrc);
        }
      }
      if (ij + VLen < VBlockSize) {
        for (; ij < VBlockSize; ij += VLen) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src);
          vsrc = _mm512_abs_ps(vsrc);
          vmaxval = _mm512_max_ps(vmaxval, vsrc);
        }
      }

      auto maxval = _mm512_reduce_max_ps(vmaxval);
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = std::abs(static_cast<float>(srcptr[(j + ij) + i * ld_src]));
          maxval = std::max(maxval, srcval);
        }
      }
      float scale = maxval / 127;
      scales[j / blocksize + i * ld_scale] = scale;
      float rscale = 1.f / scale;
      auto vrscale = _mm512_set1_ps(rscale);
      ij = 0;
      int sum = 0;
      for (; ij < VBlockSizeU3; ij += VLen * 3) {
        for (int iu = 0; iu < 3; iu++) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src + iu * VLen);
          vsrc = _mm512_mul_ps(vsrc, vrscale);
          auto vdsrc = _mm512_cvtps_epi32(vsrc);
          sum += _mm512_reduce_add_epi32(vdsrc);
          vdsrc = _mm512_min_epi32(vdsrc, vpos);
          vdsrc = _mm512_max_epi32(vdsrc, vneg);
          auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(&dstptr[(j + ij) + i * ld_dst + iu * VLen]), vbsrc);
        }
      }
      if (ij + VLen < VBlockSize) {
        for (; ij < VBlockSize; ij += VLen) {
          __m512 vsrc = load_T_fp32(srcptr + j + ij + i * ld_src);
          vsrc = _mm512_mul_ps(vsrc, vrscale);
          auto vdsrc = _mm512_cvtps_epi32(vsrc);
          sum += _mm512_reduce_add_epi32(vdsrc);
          vdsrc = _mm512_min_epi32(vdsrc, vpos);
          vdsrc = _mm512_max_epi32(vdsrc, vneg);
          auto vbsrc = _mm512_cvtepi32_epi8(vdsrc);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(&dstptr[(j + ij) + i * ld_dst]), vbsrc);
        }
      }
      if (ij < blocksize) {
        for (; ij < blocksize; ij++) {
          auto srcval = static_cast<float>(srcptr[(j + ij) + i * ld_src]);
          srcval = srcval * rscale;
          auto srcint = int(roundf(srcval));
          sum += srcint;
          srcint = std::min(srcint, 127);
          srcint = std::max(srcint, -127);
          dstptr[(j + ij) + i * ld_dst] = static_cast<uint8_t>(srcint);
        }
      }
      if (reduce) reduce[j / blocksize + i * ld_scale] = sum * scale;
    }
    if (j < col) {
      float absmaxval = std::numeric_limits<float>::min();
      for (size_t ij = j; ij < col; ij++) {
        absmaxval = std::max(std::abs((float)srcptr[(ij) + i * ld_src]), absmaxval);
      }
      float scale = absmaxval / 127;
      float rscale = 1.f / scale;
      scales[j / blocksize + i * ld_scale] = scale;
      int sum = 0;
      for (size_t ij = j; ij < col; ij++) {
        dstptr[(ij) + i * ld_dst] = utils::cast<float, int8_t>((float)srcptr[(ij) + i * ld_src] * rscale);
        sum += dstptr[(ij) + i * ld_dst];
      }
      if (reduce) reduce[j / blocksize + i * ld_scale] = sum * scale;
    }
  }
  return BTLA_CODE::Success;
}

inline BTLA_CODE dq8_get_fp_scale(uint8_t* src, float* dst, int row, int col, int scale_offset, int dq_blk,
                                  int dq_offset_idx, float* dq_scale, int src_stride, int dst_stride, bool zeropadding,
                                  int mN) {
  auto head_proc_num = utils::updiv(scale_offset, 16) * 16 - scale_offset;
  auto zmm_dq_offset = _mm512_set1_ps(dq_scale[dq_offset_idx]);

  auto get_fp_scale = [&](int proc_src_num, __mmask16 mask, int scale_offset, uint8_t* src, float* dst) {
    auto dq_s_idx = scale_offset / dq_blk;
    auto zmm_dq_scale = _mm512_set1_ps(dq_scale[dq_s_idx]);
    float tmp[16];
    for (int i = 0; i < proc_src_num; i++) tmp[i] = dq8_bnb_LUT[src[i]];
    __m512 fp32_dq_zmm = _mm512_loadu_ps(tmp);
    auto fzmm = _mm512_mul_ps(fp32_dq_zmm, zmm_dq_scale);
    fzmm = _mm512_add_ps(fzmm, zmm_dq_offset);
    _mm512_mask_storeu_ps(dst, mask, fzmm);
  };

  for (int i = 0; i < row; i++) {
    if (head_proc_num > col) {
      auto mask = _cvtu32_mask16(0xffff >> (16 - col));
      get_fp_scale(col, mask, scale_offset + i * mN, src + i * src_stride, dst + i * dst_stride);
    } else {
      // TODO(zhe): consider head_proc_num==0 case.
      auto head_mask = _cvtu32_mask16(0xffff >> (16 - head_proc_num));
      auto body_mask = _cvtu32_mask16(0xffff);
      get_fp_scale(head_proc_num, head_mask, scale_offset + i * mN, src + i * src_stride, dst + i * dst_stride);
      auto scale_offset_iter = scale_offset + i * mN + head_proc_num;
      uint8_t* src_iter_ptr = src + head_proc_num;
      float* dst_iter_ptr = dst + head_proc_num;
      auto body_loop = (col - head_proc_num) / 16;
      auto tail_proc_num = (col - head_proc_num) % 16;
      int ii = 0;
      for (; ii < body_loop; ii++) {
        get_fp_scale(16, body_mask, scale_offset_iter + ii * 16, src_iter_ptr + i * src_stride + ii * 16,
                     dst_iter_ptr + i * dst_stride + ii * 16);
      }
      if (tail_proc_num > 0) {
        auto tail_mask = _cvtu32_mask16(0xffff >> (16 - tail_proc_num));
        get_fp_scale(tail_proc_num, tail_mask, scale_offset_iter + ii * 16, src_iter_ptr + i * src_stride + ii * 16,
                     dst_iter_ptr + i * dst_stride + ii * 16);
      }
    }
  }
  if (zeropadding) assert(0);
  return BTLA_CODE::Success;
}

static inline BTLA_CODE alphabeta_f32_f32(const float alpha, const float* srcptr, const int srcstep, const float beta,
                                          const float* src1ptr, const int src1step, float* dstptr, const int dststep,
                                          const int M, const int N) {
  int constexpr Vlen = 16;
  auto vN = utils::padto_le(N, Vlen);
  auto valpha = _mm512_set1_ps(alpha);
  auto vbeta = _mm512_set1_ps(beta);

  for (int i = 0; i < M; i++) {
    int j = 0;
    if (beta != 0.f) {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
        auto vsrc1 = _mm512_loadu_ps(src1ptr + i * src1step + j);
        auto vdst = _mm512_mul_ps(valpha, vsrc);
        vdst = _mm512_fmadd_ps(vbeta, vsrc1, vdst);
        _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j] + beta * src1ptr[i * src1step + j];
      }
    } else {
      for (; j < vN; j += Vlen) {
        auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
        auto vdst = _mm512_mul_ps(valpha, vsrc);
        _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
      }
      for (; j < N; j += 1) {
        dstptr[i * dststep + j] = alpha * srcptr[i * srcstep + j];
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCA_T>
static inline BTLA_CODE accum_alphaN_f32_f32(const SCA_T* alpha, const float* srcptr, const int srcstep, float* dstptr,
                                             const int dststep, const int M, const int N) {
  int constexpr Vlen = 16;
  auto vN = utils::padto_le(N, Vlen);
  int j = 0;
  for (; j < vN; j += Vlen) {
    __m512 valpha;
    if constexpr (std::is_same_v<SCA_T, float>) {
      valpha = _mm512_loadu_ps(alpha + j);
    } else if constexpr (std::is_same_v<SCA_T, utils::bf16>) {
      auto tmp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(alpha + j));
      valpha = zmm_cvt_bf16_fp32(tmp);
    } else if constexpr (std::is_same_v<SCA_T, utils::f8>) {
      valpha = _mm512_scalef_ps(
          _mm512_set1_ps(1),
          _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha + j)))));
    }
    for (size_t i = 0; i < M; i++) {
      auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
      auto vsrc1 = _mm512_loadu_ps(dstptr + i * dststep + j);
      auto vdst = _mm512_fmadd_ps(valpha, vsrc, vsrc1);
      _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
    }
  }
  for (; j < N; j += 1) {
    for (size_t i = 0; i < M; i++) {
      if constexpr (!std::is_same_v<SCA_T, utils::f8>) {
        dstptr[i * dststep + j] += static_cast<float>(alpha[j]) * srcptr[i * srcstep + j];
      } else {
        dstptr[i * dststep + j] += alpha[j].mul(srcptr[i * srcstep + j]);
      }
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE accum_f32_f32(const float* srcptr, const int srcstep, float* dstptr, const int dststep,
                                      const int M, const int N) {
  int constexpr Vlen = 16;
  auto vN = utils::padto_le(N, Vlen);
  int j = 0;
  for (; j < vN; j += Vlen) {
    for (size_t i = 0; i < M; i++) {
      auto vsrc = _mm512_loadu_ps(srcptr + i * srcstep + j);
      auto vsrc1 = _mm512_loadu_ps(dstptr + i * dststep + j);
      auto vdst = _mm512_add_ps(vsrc, vsrc1);
      _mm512_storeu_ps(dstptr + i * dststep + j, vdst);
    }
  }
  for (; j < N; j += 1) {
    for (size_t i = 0; i < M; i++) {
      dstptr[i * dststep + j] += srcptr[i * srcstep + j];
    }
  }
  return BTLA_CODE::Success;
}

static inline void vec_quanout_s32_u32_v16(const int32_t* srcptr, __m512& vfactor, __m512i& vzp, __m512i& vzeros,
                                           __m512i& v255, uint8_t* dstptr) {
  auto vsrcd = _mm512_loadu_si512(srcptr);
  auto vsrcf = _mm512_mul_ps(vfactor, _mm512_cvtepi32_ps(vsrcd));
  vsrcd = _mm512_cvtps_epi32(vsrcf);
  vsrcd = _mm512_add_epi32(vsrcd, vzp);
  vsrcd = _mm512_max_epi32(vsrcd, vzeros);
  vsrcd = _mm512_min_epi32(vsrcd, v255);
  auto vdstb = _mm512_cvtepi32_epi8(vsrcd);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dstptr), vdstb);
}

static inline BTLA_CODE quanout_s32_u32(const float alpha, const int32_t* srcptr, const int srcstep, uint8_t* dstptr,
                                        const int dststep, const int M, const int N, float scaleSrc, float scaleDst,
                                        int zpDst) {
  float factor = alpha * scaleSrc / scaleDst;
  auto vfactor = _mm512_set1_ps(factor);
  auto vzp = _mm512_set1_epi32(zpDst);
  auto vzeros = _mm512_set1_epi32(0);
  auto v255 = _mm512_set1_epi32(255);
  int N64 = utils::padto_le(N, 64);
  int N48 = utils::padto_le(N, 48);
  int N16 = utils::padto_le(N, 16);
  for (int i = 0; i < M; i++) {
    int j = 0;
    for (; j < N64; j += 64) {
      for (int iv = 0; iv < 4; iv++) {
        vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j + iv * 16], vfactor, vzp, vzeros, v255,
                                &dstptr[i * dststep + j + iv * 16]);
      }
    }
    if (N48 - j >= 48) {
      for (; j < N48; j += 48) {
        for (int iv = 0; iv < 3; iv++) {
          vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j + iv * 16], vfactor, vzp, vzeros, v255,
                                  &dstptr[i * dststep + j + iv * 16]);
        }
      }
    }
    if (N16 - j >= 16) {
      for (; j < N16; j += 16) {
        vec_quanout_s32_u32_v16(&srcptr[i * srcstep + j], vfactor, vzp, vzeros, v255, &dstptr[i * dststep + j]);
      }
    }
    for (; j < N; j++) {
      float fsrc = static_cast<float>(srcptr[i * srcstep + j]) * factor;
      dstptr[i * dststep + j] = utils::cast<float, uint8_t>(fsrc + static_cast<float>(zpDst));
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE accumulate_dequantize_s32_f32(const int32_t* srcptr, float* dstptr, float alpha, float beta,
                                                      int row, int col, int ld_src, int ld_dst, float* ascales,
                                                      int ldas, float* wscales) {
  auto vbeta = _mm512_set1_ps(beta);
  int col16 = utils::padto_le(col, 16);
  for (int irow = 0; irow < row; irow++) {
    auto scale = ascales[irow * ldas] * alpha;
    auto valpha = _mm512_set1_ps(scale);
    int icol = 0;
    for (; icol < col16; icol += 16) {
      auto vwscale = _mm512_loadu_ps(wscales + icol);
      auto vscale = _mm512_mul_ps(valpha, vwscale);
      auto vdst = _mm512_loadu_ps(dstptr + irow * ld_dst + icol);
      vdst = _mm512_mul_ps(vdst, vbeta);
      auto vsrcd = _mm512_loadu_si512(srcptr + irow * ld_src + icol);
      auto vsrc = _mm512_cvtepi32_ps(vsrcd);
      vsrc = _mm512_fmadd_ps(vsrc, vscale, vdst);
      _mm512_storeu_ps(dstptr + irow * ld_dst + icol, vsrc);
    }
    for (; icol < col; icol += 1) {
      dstptr[irow * ld_dst + icol] =
          scale * wscales[icol] * srcptr[irow * ld_src + icol] + beta * dstptr[irow * ld_dst + icol];
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCAB_T, int NTILE>
static inline BTLA_CODE dequant_s32_fp32_N(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                           const int row, const float* scaleA, const int ldsa, const SCAB_T* scaleB) {
  int constexpr VLen = 16;
  static_assert(NTILE % VLen == 0);
  int constexpr NRegs = NTILE / VLen;
  __m512 vwscale[NRegs];
  for (int ic = 0; ic < NRegs; ic++) {
    vwscale[ic] = load_T_fp32(scaleB + ic * VLen);
  }
  for (int irow = 0; irow < row; irow++) {
    auto scale = scaleA[irow * ldsa];
    auto valpha = _mm512_set1_ps(scale);
    for (int ic = 0; ic < NRegs; ic++) {
      auto vscale = _mm512_mul_ps(valpha, vwscale[ic]);
      auto vsrcd = _mm512_loadu_si512(srcptr + irow * srcstep + ic * VLen);
      auto vsrc = _mm512_cvtepi32_ps(vsrcd);
      vsrc = _mm512_mul_ps(vsrc, vscale);
      _mm512_storeu_ps(dstptr + irow * dststep + ic * VLen, vsrc);
    }
  }
  return BTLA_CODE::Success;
}

template <typename SCAB_T>
static inline BTLA_CODE dequant_s32_fp32(const int32_t* srcptr, const int srcstep, float* dstptr, const int dststep,
                                         const int row, const int col, const float* scaleA, const int ldsa,
                                         const SCAB_T* scaleB) {
  int col16 = utils::padto_le(col, 16);
  int col48 = utils::padto_le(col, 48);
  int col64 = utils::padto_le(col, 64);
  int icol = 0;
  for (; icol < col64; icol += 64) {
    dequant_s32_fp32_N<SCAB_T, 64>(srcptr + icol, srcstep, dstptr + icol, dststep, row, scaleA, ldsa, scaleB + icol);
  }
  if (icol + 48 <= col48) {
    for (; icol < col48; icol += 48) {
      dequant_s32_fp32_N<SCAB_T, 48>(srcptr + icol, srcstep, dstptr + icol, dststep, row, scaleA, ldsa, scaleB + icol);
    }
  }
  if (icol + 16 <= col16) {
    for (; icol < col16; icol += 16) {
      dequant_s32_fp32_N<SCAB_T, 16>(srcptr + icol, srcstep, dstptr + icol, dststep, row, scaleA, ldsa, scaleB + icol);
    }
  }
  for (; icol < col; icol += 1) {
    for (int irow = 0; irow < row; irow++) {
      auto scale = scaleA[irow * ldsa];
      dstptr[irow * dststep + icol] = scale * float(scaleB[icol]) * srcptr[irow * srcstep + icol];
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE broadcast_u8(int num, const uint8_t& srcval, uint8_t* dstptr) {
  int i = 0;
  int constexpr VN = 64 / sizeof(srcval);
  int numv = utils::padto_le(num, VN);
  auto vsrc = _mm512_set1_epi8(srcval);
  for (; i < numv; i += VN) {
    _mm512_storeu_si512(dstptr + i, vsrc);
  }
  int num32 = utils::padto_le(num, 32);
  if (i + 32 <= num32) {
    for (; i < num32; i += 32) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstptr + i), _mm512_castsi512_si256(vsrc));
    }
  }
  for (; i < num; i++) {
    dstptr[i] = srcval;
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE remove_act_zeropoint_bias(float* accptr, int ldacc, int row, int col, uint8_t* zps,
                                                  float* scales, int lds, const float* reduce) {
  int constexpr VLen = 16;
  auto col16 = utils::padto_le(col, VLen);
  for (int i = 0; i < row; i++) {
    auto zpf = static_cast<float>(zps[i * lds]) * scales[i * lds];
    int j = 0;
    auto vzp = _mm512_set1_ps(-zpf);
    for (; j < col16; j += VLen) {
      auto vreduce = _mm512_loadu_ps(reduce + j);
      auto vacc = _mm512_loadu_ps(&accptr[i * ldacc + j]);
      vacc = _mm512_fmadd_ps(vzp, vreduce, vacc);
      _mm512_storeu_ps(&accptr[i * ldacc + j], vacc);
    }
    if (j < col) {
      for (; j < col; j++) {
        accptr[i * ldacc + j] -= zpf * reduce[j];
      }
    }
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE remove_wei_zeropoint_bias(float* accptr, int ldacc, int row, int col, int8_t* zps,
                                                  float* scales, int lds, const float* reduce) {
  int constexpr VLen = 16;
  auto col16 = utils::padto_le(col, VLen);
  for (int i = 0; i < row; i++) {
    auto vreduce = _mm512_set1_ps(-reduce[i * lds]);
    int j = 0;
    for (; j < col16; j += VLen) {
      auto vzp_s32 = _mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(zps + j)));
      auto vzp_f32 = _mm512_cvtepi32_ps(vzp_s32);
      auto vzp = _mm512_mul_ps(vzp_f32, _mm512_loadu_ps(scales + j));
      auto vacc = _mm512_loadu_ps(&accptr[i * ldacc + j]);
      vacc = _mm512_fmadd_ps(vzp, vreduce, vacc);
      _mm512_storeu_ps(&accptr[i * ldacc + j], vacc);
    }
    if (j < col) {
      for (; j < col; j++) {
        accptr[i * ldacc + j] -= static_cast<float>(zps[j]) * scales[j] * reduce[i * lds];
      }
    }
  }
  return BTLA_CODE::Success;
}

template <bool ZPA, bool ZPB>
static inline BTLA_CODE remove_zeropoint_bias(int32_t* accptr, int ldacc, int row, int col, uint8_t* zpa, int8_t* zpb,
                                              int lda, int k, const int32_t* reducea, const int32_t* reduceb) {
  int constexpr VLen = 16;
  auto col16 = utils::padto_le(col, VLen);
  auto vk = _mm512_set1_epi32(k);
  for (int i = 0; i < row; i++) {
    auto vreducea = ZPB ? _mm512_set1_epi32(reducea[i * lda]) : _mm512_set1_epi32(0);
    auto vzpa = ZPA ? _mm512_set1_epi32(int(zpa[i * lda])) : _mm512_set1_epi32(0);
    int j = 0;
    for (; j < col16; j += VLen) {
      auto vzpb = ZPB ? load_s8_s32(zpb + j) : _mm512_set1_epi32(0);
      auto vreduceb = ZPA ? _mm512_loadu_epi32(reduceb + j) : _mm512_set1_epi32(0);
      auto vacc = _mm512_loadu_epi32(&accptr[i * ldacc + j]);
      __m512i tmp;
      if constexpr (ZPA) {
        tmp = _mm512_mullo_epi32(vzpa, vreduceb);
        vacc = _mm512_sub_epi32(vacc, tmp);
      }
      if constexpr (ZPB) {
        tmp = _mm512_mullo_epi32(vzpb, vreducea);
        vacc = _mm512_sub_epi32(vacc, tmp);
      }
      if constexpr (ZPA && ZPB) {
        tmp = _mm512_mullo_epi32(vzpb, vk);
        tmp = _mm512_mullo_epi32(tmp, vzpa);
        vacc = _mm512_sub_epi32(vacc, tmp);
      }
      _mm512_storeu_epi32(&accptr[i * ldacc + j], vacc);
    }
    if (j < col) {
      int32_t vreducea = ZPB ? reducea[i * lda] : 0;
      uint8_t vzpa = ZPA ? zpa[i * lda] : 0;
      for (; j < col; j++) {
        auto vzpb = ZPB ? zpb[j] : 0;
        accptr[i * ldacc + j] -= ZPB ? vzpb * vreducea : 0;
        accptr[i * ldacc + j] -= ZPA ? vzpa * reduceb[j] : 0;
        accptr[i * ldacc + j] -= ZPB && ZPA ? vzpa * vzpb * k : 0;
      }
    }
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE cvt_fp32_T_2D(const float* src_ptr, T* dst_ptr, int row, int col, int src_step, int dst_step,
                                      bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(T);
  constexpr int simd_proc_elt = 16;
  float tmpbuf[simd_proc_elt];
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      store_fp32_T(_mm512_loadu_ps(src + j), dst + j);
    }
    if (col_tail > 0) {
      memcpy(tmpbuf, src + j, col_tail * sizeof(src_ptr[0]));
      auto vf32 = _mm512_loadu_ps(tmpbuf);
      store_fp32_T(vf32, (T*)tmpbuf);
      memcpy(dst + j, tmpbuf, col_tail * sizeof(dst[0]));
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE cvt_T_fp32_2D(const T* src_ptr, float* dst_ptr, int row, int col, int src_step, int dst_step,
                                      bool zeropadding) {
  const int npadding = (dst_step - col) * sizeof(float);
  constexpr int simd_proc_elt = 16;
  float* tmpbuf = (float*)alloca(simd_proc_elt * sizeof(float));
  auto col_body = col / simd_proc_elt * simd_proc_elt;
  auto col_tail = col % simd_proc_elt;
  for (int i = 0; i < row; i++) {
    const auto src = src_ptr + i * src_step;
    const auto dst = dst_ptr + i * dst_step;
    int j = 0;
    for (; j < col_body; j += simd_proc_elt) {
      auto vf32 = load_T_fp32(src + j);
      _mm512_storeu_ps(dst + j, vf32);
    }
    if (col_tail > 0) {
      memcpy(tmpbuf, src + j, col_tail * sizeof(src_ptr[0]));
      auto vf32 = load_T_fp32((T*)tmpbuf);
      _mm512_storeu_ps(tmpbuf, vf32);
      memcpy(dst + j, tmpbuf, col_tail * sizeof(dst[0]));
    }
    if (zeropadding && npadding) std::memset(dst + col, 0, npadding);
  }
  return BTLA_CODE::Success;
}

template <typename SRC_T>
static inline BTLA_CODE col_block_reduce_sum(const SRC_T* srcptr, int ldsrc, int row, int col, int blocksize,
                                             float* reduce, int ldr) {
  int constexpr VLen = 16;
  auto vblock2_ = utils::padto_le(blocksize, VLen * 2);
  auto vblock_ = utils::padto_le(blocksize, VLen);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j += blocksize) {
      auto tmp = 0.f;
      auto vsum = _mm512_set1_ps(0.f);
      int jj = 0;
      auto vblock2 = j + vblock2_ <= col ? vblock2_ : 0;
      auto vblock = j + vblock_ <= col ? vblock_ : 0;
      for (; jj < vblock2; jj += VLen * 2) {
        auto vtmp = _mm512_loadu_ps(srcptr + i * ldsrc + j + jj);
        auto vtmp1 = _mm512_loadu_ps(srcptr + i * ldsrc + j + jj + VLen);
        auto s0 = _mm512_reduce_add_ps(vtmp);
        auto s1 = _mm512_reduce_add_ps(vtmp1);
        tmp += s0;
        tmp += s1;
      }
      if (jj + VLen <= vblock) {
        for (; jj < vblock; jj += VLen) {
          auto vtmp = _mm512_loadu_ps(srcptr + i * ldsrc + j + jj);
          auto s0 = _mm512_reduce_add_ps(vtmp);
          tmp += s0;
        }
      }
      for (; jj < blocksize; jj++) {
        tmp += *(srcptr + i * ldsrc + j + jj);
      }
      reduce[i * ldr + j / blocksize] = tmp;
    }
  }
  return BTLA_CODE::Success;
}
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"  // https://stackoverflow.com/a/49216021
#endif
// Interleave 2 bf16 zmm vectors inplace
static inline void interleave_word(std::array<__m512i, 2>& dst) {  // NOLINT [runtime/references]
  static constexpr uint32_t perm_idx_a[16]{
      0 | 0,  1 | 0,  2 | 0,  3 | 0,   //
      0 | 16, 1 | 16, 2 | 16, 3 | 16,  //
      4 | 0,  5 | 0,  6 | 0,  7 | 0,   //
      4 | 16, 5 | 16, 6 | 16, 7 | 16,  //
  };
  static constexpr uint32_t perm_idx_b[16]{
      8 | 0,   9 | 0,   10 | 0,  11 | 0,   //
      8 | 16,  9 | 16,  10 | 16, 11 | 16,  //
      12 | 0,  13 | 0,  14 | 0,  15 | 0,   //
      12 | 16, 13 | 16, 14 | 16, 15 | 16,  //
  };
  static const auto v_perm_idx_a = _mm512_loadu_si512(perm_idx_a);
  static const auto v_perm_idx_b = _mm512_loadu_si512(perm_idx_b);

  __m512i tmp[2];
  tmp[0] = _mm512_unpacklo_epi16(dst[0], dst[1]);
  tmp[1] = _mm512_unpackhi_epi16(dst[0], dst[1]);
  dst[0] = _mm512_permutex2var_epi32(tmp[0], v_perm_idx_a, tmp[1]);
  dst[1] = _mm512_permutex2var_epi32(tmp[0], v_perm_idx_b, tmp[1]);
}

// Interleave 16 zmm vectors of dwords inplace
static inline void tr_x16_dword(std::array<__m512i, 16>& dst) {  // NOLINT [runtime/references]
  __m512i tmp[16];

  for (int i = 0; i < 8; ++i) {
    tmp[2 * i] = _mm512_unpacklo_epi32(dst[2 * i], dst[2 * i + 1]);
    tmp[2 * i + 1] = _mm512_unpackhi_epi32(dst[2 * i], dst[2 * i + 1]);
  }

  for (int i = 0; i < 4; ++i) {
    dst[4 * i] = _mm512_unpacklo_epi64(tmp[4 * i], tmp[4 * i + 2]);
    dst[4 * i + 1] = _mm512_unpackhi_epi64(tmp[4 * i], tmp[4 * i + 2]);
    dst[4 * i + 2] = _mm512_unpacklo_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
    dst[4 * i + 3] = _mm512_unpackhi_epi64(tmp[4 * i + 1], tmp[4 * i + 3]);
  }

  for (int i = 0; i < 2; ++i) {
    tmp[8 * i + 0] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0x88);
    tmp[8 * i + 1] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0x88);
    tmp[8 * i + 2] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0x88);
    tmp[8 * i + 3] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0x88);
    tmp[8 * i + 4] = _mm512_shuffle_i32x4(dst[8 * i + 0], dst[8 * i + 4], 0xdd);
    tmp[8 * i + 5] = _mm512_shuffle_i32x4(dst[8 * i + 1], dst[8 * i + 5], 0xdd);
    tmp[8 * i + 6] = _mm512_shuffle_i32x4(dst[8 * i + 2], dst[8 * i + 6], 0xdd);
    tmp[8 * i + 7] = _mm512_shuffle_i32x4(dst[8 * i + 3], dst[8 * i + 7], 0xdd);
  }

  dst[0] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0x88);
  dst[1] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0x88);
  dst[2] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0x88);
  dst[3] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0x88);
  dst[4] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0x88);
  dst[5] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0x88);
  dst[6] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0x88);
  dst[7] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0x88);
  dst[8] = _mm512_shuffle_i32x4(tmp[0], tmp[8], 0xdd);
  dst[9] = _mm512_shuffle_i32x4(tmp[1], tmp[9], 0xdd);
  dst[10] = _mm512_shuffle_i32x4(tmp[2], tmp[10], 0xdd);
  dst[11] = _mm512_shuffle_i32x4(tmp[3], tmp[11], 0xdd);
  dst[12] = _mm512_shuffle_i32x4(tmp[4], tmp[12], 0xdd);
  dst[13] = _mm512_shuffle_i32x4(tmp[5], tmp[13], 0xdd);
  dst[14] = _mm512_shuffle_i32x4(tmp[6], tmp[14], 0xdd);
  dst[15] = _mm512_shuffle_i32x4(tmp[7], tmp[15], 0xdd);
}

// Load 2 fp16 vectors; convert them to bf16 and interleave them
template <int tail>
static inline std::array<__m512i, 2> load_fp16_bf16_interleave_word(const utils::fp16* a, size_t lda) {
  static_assert(tail > 0 && tail <= 2, "Unexpected tail value.");
  std::array<__m512i, 2> dst;
  for (int i = 0; i < tail; ++i) {
    dst[i] = zmm_cvt_fp32_bf16_x2(load_T_fp32(a + i * lda + 16), load_T_fp32(a + i * lda + 0));
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
    dst[i] = zmm_cvt_fp32_bf16_x2(                                               //
        zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),  //
        zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0)));
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
    dst[i] = zmm_cvt_fp32_bf16_x2(zmm_cvt_fp16_fp32(_mm256_loadu_epi16(a + i * lda + 16)),
                                  zmm_cvt_fp16_fp32(_mm256_loadu_epi16(a + i * lda + 0)));
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
    dst[i] = zmm_cvt_fp32_bf16_x2(zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_hi, a + i * lda + 16)),
                                  zmm_cvt_fp16_fp32(_mm256_maskz_loadu_epi16(mask_lo, a + i * lda + 0)));
  }
  for (int i = tail; i < 16; ++i) dst[i] = _mm512_setzero_epi32();
  tr_x16_dword(dst);
  return dst;
}
static constexpr decltype(load_maskz_fp16_bf16_tr_x16_dword<1>)* load_maskz_fp16_bf16_tr_x16_dword_tbl[17]{
    load_maskz_fp16_bf16_tr_x16_dword<1>,  load_maskz_fp16_bf16_tr_x16_dword<1>,  load_maskz_fp16_bf16_tr_x16_dword<2>,
    load_maskz_fp16_bf16_tr_x16_dword<3>,  load_maskz_fp16_bf16_tr_x16_dword<4>,  load_maskz_fp16_bf16_tr_x16_dword<5>,
    load_maskz_fp16_bf16_tr_x16_dword<6>,  load_maskz_fp16_bf16_tr_x16_dword<7>,  load_maskz_fp16_bf16_tr_x16_dword<8>,
    load_maskz_fp16_bf16_tr_x16_dword<9>,  load_maskz_fp16_bf16_tr_x16_dword<10>, load_maskz_fp16_bf16_tr_x16_dword<11>,
    load_maskz_fp16_bf16_tr_x16_dword<12>, load_maskz_fp16_bf16_tr_x16_dword<13>, load_maskz_fp16_bf16_tr_x16_dword<14>,
    load_maskz_fp16_bf16_tr_x16_dword<15>, load_maskz_fp16_bf16_tr_x16_dword<16>,
};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

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

static inline BTLA_CODE layernorm(const float* srcptr, const float* scaleptr, const float* biasptr, float epsilon,
                                  int norm_size, float* dstptr, float* mean_out, float* mean_square_out,
                                  bool simplified) {
  int constexpr VLen = 16;
  int norm_size16 = utils::padto_le(norm_size, VLen);
  int h = 0;
  __m512 vmean = _mm512_setzero_ps(), vmeansq = _mm512_setzero_ps();
  for (; h < norm_size16; h += VLen) {
    auto tmp = _mm512_loadu_ps(srcptr + h);
    vmean = _mm512_add_ps(vmean, tmp);
    tmp = _mm512_mul_ps(tmp, tmp);
    vmeansq = _mm512_add_ps(vmeansq, tmp);
  }
  float mean = _mm512_reduce_add_ps(vmean);
  float mean_square = _mm512_reduce_add_ps(vmeansq);
  for (; h < norm_size; h++) {
    mean += srcptr[h];
    mean_square += srcptr[h] * srcptr[h];
  }
  mean = mean / norm_size;
  if (simplified) {
    mean_square = std::sqrt(mean_square / norm_size + epsilon);
  } else {
    mean_square = std::sqrt(mean_square / norm_size - mean * mean + epsilon);
  }
  auto vm = _mm512_set1_ps(mean);
  float inv_meansq = 1.f / mean_square;
  auto vms = _mm512_set1_ps(inv_meansq);
  h = 0;
  if (simplified) {
    if (scaleptr) {
      for (; h < norm_size16; h += VLen) {
        auto inp = _mm512_loadu_ps(srcptr + h);
        auto scale = _mm512_loadu_ps(scaleptr + h);
        inp = _mm512_mul_ps(inp, vms);
        inp = _mm512_mul_ps(inp, scale);
        _mm512_storeu_ps(dstptr + h, inp);
      }
      for (; h < norm_size; h++) {
        dstptr[h] = srcptr[h] * inv_meansq * scaleptr[h];
      }
    } else {
      for (; h < norm_size16; h += VLen) {
        auto inp = _mm512_loadu_ps(srcptr + h);
        inp = _mm512_mul_ps(inp, vms);
        _mm512_storeu_ps(dstptr + h, inp);
      }
      for (; h < norm_size; h++) {
        dstptr[h] = srcptr[h] * inv_meansq;
      }
    }

  } else {
    if (scaleptr) {
      if (biasptr == nullptr) {
        for (; h < norm_size16; h += VLen) {
          auto inp = _mm512_loadu_ps(srcptr + h);
          auto scale = _mm512_loadu_ps(scaleptr + h);
          inp = _mm512_sub_ps(inp, vm);
          inp = _mm512_mul_ps(inp, vms);
          inp = _mm512_mul_ps(inp, scale);
          _mm512_storeu_ps(dstptr + h, inp);
        }
        for (; h < norm_size; h++) {
          dstptr[h] = (srcptr[h] - mean) * inv_meansq * scaleptr[h];
        }
      } else {
        for (; h < norm_size16; h += VLen) {
          auto inp = _mm512_loadu_ps(srcptr + h);
          auto scale = _mm512_loadu_ps(scaleptr + h);
          inp = _mm512_sub_ps(inp, vm);
          inp = _mm512_mul_ps(inp, vms);
          inp = _mm512_mul_ps(inp, scale);
          auto bias = _mm512_loadu_ps(biasptr + h);
          inp = _mm512_add_ps(inp, bias);
          _mm512_storeu_ps(dstptr + h, inp);
        }
        for (; h < norm_size; h++) {
          dstptr[h] = (srcptr[h] - mean) * inv_meansq * scaleptr[h] + biasptr[h];
        }
      }
    } else {
      for (; h < norm_size16; h += VLen) {
        auto inp = _mm512_loadu_ps(srcptr + h);
        inp = _mm512_sub_ps(inp, vm);
        inp = _mm512_mul_ps(inp, vms);
        _mm512_storeu_ps(dstptr + h, inp);
      }
      for (; h < norm_size; h++) {
        dstptr[h] = (srcptr[h] - mean) * inv_meansq;
      }
    }
  }

  if (mean_out) {
    *mean_out = mean;
  }
  if (mean_square_out) {
    *mean_square_out = mean_square;
  }
  return BTLA_CODE::Success;
}

inline __m512 poly_scale_2nd_ps(const __m512 z, const __m512 f, const __m512 c0, const __m512 c1, const __m512 c2) {
  const auto y = _mm512_fmadd_ps(_mm512_fmadd_ps(f, c0, c1), f, c2);  // auto y = (f * c0 + c1) * f + c2;
  const auto exp = _mm512_scalef_ps(y, z);
  return exp;
}

inline __m512 exp_ps_0_1(const __m512 x) {
  static const auto c0 = _mm512_set1_ps(0.240226507f);
  static const auto c1 = _mm512_set1_ps(0.452920674f);
  static const auto c2 = _mm512_set1_ps(0.713483036f);
  static const float v_log2e = std::log2(std::exp(1.f));
  static const auto log2e = _mm512_set1_ps(v_log2e);
  static const auto half = _mm512_set1_ps(.5f);

  const auto x1 = _mm512_fmadd_ps(x, log2e, half);  // auto x1 = x * log2e + _mm512_set1_ps(.5f);
  const auto z = _mm512_floor_ps(x1);
  const auto f = _mm512_sub_ps(x1, z);  // auto f = x1 - z;

  return poly_scale_2nd_ps(z, f, c0, c1, c2);
}

static inline __m512i load_zp_epi8_broadcast_epi16(int8_t* zpptr, const __m512i& vindex) {
  auto v_zp_x = _mm256_loadu_si256((const __m256i*)zpptr);
  auto v_zp_y = _mm512_cvtepi8_epi16(v_zp_x);
  auto v_zp_y_cast = _mm512_shuffle_epi8(v_zp_y, vindex);  // TODO(Yu) AVX512F only
  return v_zp_y_cast;
}

static inline __m512i load_zp_epi8_broadcast_epi32(int8_t* zpptr, const __m512i& vindex) {
  auto v_zp_x = _mm_loadu_si128((const __m128i*)zpptr);
  auto v_zp_y = _mm512_cvtepi8_epi32(v_zp_x);
  auto v_zp_y_cast = _mm512_shuffle_epi8(v_zp_y, vindex);  // TODO(Yu) AVX512F only
  return v_zp_y_cast;
}

static inline BTLA_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, size_t elesize, int8_t* tmp,
                                         size_t tmpsize) {
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  size_t velt = utils::padto_le(elesize, 64);
  size_t i = 0;
  auto vbias = _mm512_set1_epi8(8);
  for (; i < velt; i += 64) {
    auto vout_y = unpack_4bits(reinterpret_cast<int8_t*>(srcptr + i / 2), vmask);
    vout_y = _mm512_sub_epi8(vout_y, vbias);
    _mm512_storeu_si512((__m512i*)(dstptr + i), vout_y);
  }
  if (velt < elesize) {
    if (elesize >= 64) {
      i = elesize - 64;
      auto vout_y = unpack_4bits(reinterpret_cast<int8_t*>(srcptr + i / 2), vmask);
      vout_y = _mm512_sub_epi8(vout_y, vbias);
      _mm512_storeu_si512((__m512i*)(dstptr + i), vout_y);
    } else {
      ref::decompress_kblock_s4_s8<1, 1>(srcptr + i / 2, nullptr, dstptr + i, 0, 0, 0, 0, 1,
                                         static_cast<int>(elesize - i), nullptr, 0);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8_pack1_row(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 16;
  static_assert((NTILE % 16) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  int constexpr UnpackLoop = Unroll * NTILE / 64;
  __m512i v_zp_y[UnpackLoop];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(8);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < UnpackLoop; i++) {
      v_zp_y[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpout = tmp + Unroll * NTILE / 2;
      for (int i = 0; i < UnpackLoop; i++) {
        auto v_s8_y = unpack_4bits(tmp + i * 32, vmask);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8_pack2_row(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 16;
  static_assert((NTILE % 16) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m512i v_zp_y[NReg];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(8);
  const auto vindex = _mm512_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0,
                                      14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16(tmp + i * 32, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpout = tmp + Unroll * PackRow * NTILE / 2;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(tmp + i * 32, vmask);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s4_s8_pack4_row(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 16;
  static_assert((NTILE % 16) == 0);
  int constexpr PackRow = 4;
  __m512i v_zp_y[NReg];
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(8);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 16, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
inline BTLA_CODE decompress_kblock_s4_s8(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                         int n_offset, int k_offset, int row, int col, int8_t* tmp, size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::int4x2* srcptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                    int n_offset, int k_offset, int row, int8_t* tmp, size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s4_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s4_s8_pack2_row<NTILE>;
      }
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s4_s8_pack4_row<NTILE>;
      }
      if (func) {
        int head_end = utils::padto(k_offset, blocksize);
        head_end = std::min(head_end, k_offset + row);
        int head_size = head_end - k_offset;
        if (head_size > 0) {
          (*func)(srcptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, head_size, tmp, tmpsize);
        }
        int body_size = row - head_size;
        if (body_size > 0) {
          (*func)(srcptr + head_size * NTILE / 2, zpptr, dstptr + head_size * NTILE, blocksize, ldzp, n_offset,
                  head_end, body_size, tmp, tmpsize);
        }
        return BTLA_CODE::Success;
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  } else {
    size_t elesize = static_cast<size_t>(row) * col;
    return decompress_s4_s8(srcptr, dstptr, elesize, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s2_s8(utils::bit2x4* bit2ptr, int8_t* dstptr, size_t unpack_elt, int8_t* tmp,
                                         size_t tmpsize) {
  int constexpr VBits = 512;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(2);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    vout = _mm512_sub_epi8(vout, vbias);
    _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= VElt) {
      i = unpack_elt - VElt;
      auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      vout = _mm512_sub_epi8(vout, vbias);
      _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
    } else {
      ref::decompress_s2_s8(bit2ptr + i / 4, dstptr + i, unpack_elt - i, tmp, tmpsize);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s2_s8_pack4_row(utils::bit2x4* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 4;
  __m512i v_zp_y[NReg];
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(2);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 16, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s2_s8_pack2_row(utils::bit2x4* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 2;
  int constexpr Unroll = 2;
  __m512i v_zp_y[NReg];
  const auto vindex = _mm512_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0,
                                      14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(2);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16(tmp + i * 32, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * PackRow * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmp + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s2_s8_pack1_row(utils::bit2x4* srcptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(2);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      memcpy(tmp, srcptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmp + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s2_s8(utils::bit2x4* bit2ptr, int8_t* zpptr, int8_t* dstptr, int blocksize,
                                                int ldzp, int n_offset, int k_offset, int row, int col, int8_t* tmp,
                                                size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::bit2x4* srcptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                    int n_offset, int k_offset, int row, int8_t* tmp, size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s2_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s2_s8_pack2_row<NTILE>;
      }
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s2_s8_pack4_row<NTILE>;
      }
      if (func) {
        int head_end = utils::padto(k_offset, blocksize);
        head_end = std::min(head_end, k_offset + row);
        int head_size = head_end - k_offset;
        if (head_size > 0) {
          (*func)(bit2ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, head_size, tmp, tmpsize);
        }
        int body_size = row - head_size;
        if (body_size > 0) {
          (*func)(bit2ptr + head_size * NTILE / 4, zpptr, dstptr + head_size * NTILE, blocksize, ldzp, n_offset,
                  head_end, body_size, tmp, tmpsize);
        }
        return BTLA_CODE::Success;
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  } else {
    size_t elesize = static_cast<size_t>(row) * col;
    return decompress_s2_s8(bit2ptr, dstptr, elesize, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s3_s8(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 512;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(4);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
    vout = _mm512_or_si512(vout, vb1);
    vout = _mm512_sub_epi8(vout, vbias);
    _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= VElt) {
      i = unpack_elt - VElt;
      auto vout = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
      vout = _mm512_or_si512(vout, vb1);
      vout = _mm512_sub_epi8(vout, vbias);
      _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
    } else {
      ref::decompress_s3_s8(bit2ptr + i / 4, bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp, tmpsize);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8_pack1_row(utils::bit2x4* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(4);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb2ptr = tmp;
      memcpy(tmpb2ptr, srcptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8_pack2_row(utils::bit2x4* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(4);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);

  const auto vindex = _mm512_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0,
                                      14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16(tmp + i * 32, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb2ptr = tmp;
      memcpy(tmpb2ptr, srcptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8_pack4_row(utils::bit2x4* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 4;
  __m512i v_zp_y[NReg];
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vbias = _mm512_set1_epi8(4);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 16, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b2ptr = srcptr + (ir + ib) * NTILE / 4;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s3_s8(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset,
                                                int row, int col, int8_t* tmp, size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                    int blocksize, int ldzp, int n_offset, int k_offset, int row, int8_t* tmp,
                                    size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s3_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s3_s8_pack2_row<NTILE>;
      }
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s3_s8_pack4_row<NTILE>;
      }

      if (func) {
        int head_end = utils::padto(k_offset, blocksize);
        head_end = std::min(head_end, k_offset + row);
        int head_size = head_end - k_offset;
        if (head_size > 0) {
          (*func)(bit2ptr, bit1ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, head_size, tmp, tmpsize);
        }
        int body_size = row - head_size;
        if (body_size > 0) {
          (*func)(bit2ptr + head_size * NTILE / 4, bit1ptr + head_size * NTILE / 8, zpptr, dstptr + head_size * NTILE,
                  blocksize, ldzp, n_offset, head_end, body_size, tmp, tmpsize);
        }
        return BTLA_CODE::Success;
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  } else {
    size_t elesize = static_cast<size_t>(row) * col;
    return decompress_s3_s8(bit2ptr, bit1ptr, dstptr, elesize, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s1_s8(utils::bit1x8* bit1ptr, int8_t* dstptr, size_t unpack_elt, int8_t* tmp,
                                         size_t tmpsize) {
  int constexpr VBits = 512;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
    vb1 = _mm512_srli_epi32(vb1, 2);
    vb1 = _mm512_sub_epi8(vb1, vbias);
    _mm512_storeu_si512((__m512i*)(dstptr + i), vb1);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= VElt) {
      i = unpack_elt - VElt;
      auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
      vb1 = _mm512_srli_epi32(vb1, 2);
      vb1 = _mm512_sub_epi8(vb1, vbias);
      _mm512_storeu_si512((__m512i*)(dstptr + i), vb1);
    } else {
      ref::decompress_s1_s8(bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp, tmpsize);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8_pack1_row(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        vb1 = _mm512_srli_epi32(vb1, 2);
        vb1 = _mm512_sub_epi8(vb1, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), vb1);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb1ptr = tmp;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        vb1 = _mm512_srli_epi32(vb1, 2);
        vb1 = _mm512_sub_epi8(vb1, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), vb1);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8_pack2_row(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);

  const auto vindex = _mm512_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0,
                                      14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16(tmp + i * 32, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        vb1 = _mm512_srli_epi32(vb1, 2);
        vb1 = _mm512_sub_epi8(vb1, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), vb1);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb1ptr = tmp;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        vb1 = _mm512_srli_epi32(vb1, 2);
        vb1 = _mm512_sub_epi8(vb1, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), vb1);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8_pack4_row(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 16, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        vb1 = _mm512_srli_epi32(vb1, 2);
        vb1 = _mm512_sub_epi8(vb1, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), vb1);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s1_s8(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr, int blocksize,
                                                int ldzp, int n_offset, int k_offset, int row, int col, int8_t* tmp,
                                                size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp,
                                    int n_offset, int k_offset, int row, int8_t* tmp, size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s1_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s1_s8_pack2_row<NTILE>;
      }
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s1_s8_pack4_row<NTILE>;
      }
      if (func) {
        int head_end = utils::padto(k_offset, blocksize);
        head_end = std::min(head_end, k_offset + row);
        int head_size = head_end - k_offset;
        if (head_size > 0) {
          (*func)(bit1ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, head_size, tmp, tmpsize);
        }
        int body_size = row - head_size;
        if (body_size > 0) {
          (*func)(bit1ptr + head_size * NTILE / 8, zpptr, dstptr + head_size * NTILE, blocksize, ldzp, n_offset,
                  head_end, body_size, tmp, tmpsize);
        }
        return BTLA_CODE::Success;
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  } else {
    size_t elesize = static_cast<size_t>(row) * col;
    return decompress_s1_s8(bit1ptr, dstptr, elesize, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s5_s8(utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 512;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
    auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
    vb1 = _mm512_slli_epi32(vb1, 2);
    vout = _mm512_or_si512(vout, vb1);
    vout = _mm512_sub_epi8(vout, vbias);
    _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= VElt) {
      i = unpack_elt - VElt;
      auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
      auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
      vb1 = _mm512_slli_epi32(vb1, 2);
      vout = _mm512_or_si512(vout, vb1);
      vout = _mm512_sub_epi8(vout, vbias);
      _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
    } else {
      ref::decompress_s5_s8(bit4ptr + i / 2, bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp, tmpsize);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8_pack1_row(utils::bit4x2* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        vb1 = _mm512_slli_epi32(vb1, 2);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits((utils::bit2x4*)(tmpb4ptr + i * 32), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        vb1 = _mm512_slli_epi32(vb1, 2);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8_pack2_row(utils::bit4x2* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);

  const auto vindex = _mm512_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0,
                                      14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16(tmp + i * 32, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        vb1 = _mm512_slli_epi32(vb1, 2);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits((utils::bit2x4*)(tmpb4ptr + i * 32), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        vb1 = _mm512_slli_epi32(vb1, 2);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8_pack4_row(utils::bit4x2* srcptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 16, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        vb1 = _mm512_slli_epi32(vb1, 2);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s5_s8(utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr, int8_t* zpptr,
                                                int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset,
                                                int row, int col, int8_t* tmp, size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::bit4x2* bit4ptr, utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                    int blocksize, int ldzp, int n_offset, int k_offset, int row, int8_t* tmp,
                                    size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s5_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s5_s8_pack2_row<NTILE>;
      }
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s5_s8_pack4_row<NTILE>;
      }
      if (func) {
        int head_end = utils::padto(k_offset, blocksize);
        head_end = std::min(head_end, k_offset + row);
        int head_size = head_end - k_offset;
        if (head_size > 0) {
          (*func)(bit4ptr, bit1ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, head_size, tmp, tmpsize);
        }
        int body_size = row - head_size;
        if (body_size > 0) {
          (*func)(bit4ptr + head_size * NTILE / 2, bit1ptr + head_size * NTILE / 8, zpptr, dstptr + head_size * NTILE,
                  blocksize, ldzp, n_offset, head_end, body_size, tmp, tmpsize);
        }
        return BTLA_CODE::Success;
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  } else {
    size_t elesize = static_cast<size_t>(row) * col;
    return decompress_s5_s8(bit4ptr, bit1ptr, dstptr, elesize, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s7_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                         int8_t* dstptr, size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 512;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
    auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
    auto vb2 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    vb1 = _mm512_slli_epi32(vb1, 4);
    vb2 = _mm512_slli_epi32(vb2, 4);
    vout = _mm512_or_si512(vout, vb1);
    vout = _mm512_or_si512(vout, vb2);
    vout = _mm512_sub_epi8(vout, vbias);
    _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= VElt) {
      i = unpack_elt - VElt;
      auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
      auto vb1 = unpack_1bits(bit1ptr + i / 8, zmm_0x00, zmm_0x04);
      auto vb2 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      vb1 = _mm512_slli_epi32(vb1, 4);
      vb2 = _mm512_slli_epi32(vb2, 4);
      vout = _mm512_or_si512(vout, vb1);
      vout = _mm512_or_si512(vout, vb2);
      vout = _mm512_sub_epi8(vout, vbias);
      _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
    } else {
      ref::decompress_s7_s8(bit4ptr + i / 2, bit2ptr + i / 4, bit1ptr + i / 8, dstptr + i, unpack_elt - i, tmp,
                            tmpsize);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8_pack1_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr,
                                                          utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        auto vb2 = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        vb2 = _mm512_slli_epi32(vb2, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_or_si512(v_s8_y, vb2);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpb2ptr = tmp + Unroll * NTILE * 3 / 4;
      memcpy(tmpb2ptr, bit2ptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits((utils::bit2x4*)(tmpb4ptr + i * 32), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        auto vb2 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        vb2 = _mm512_slli_epi32(vb2, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_or_si512(v_s8_y, vb2);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8_pack2_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr,
                                                          utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);

  const auto vindex = _mm512_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0,
                                      14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16(tmp + i * 32, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        auto vb2 = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        vb2 = _mm512_slli_epi32(vb2, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_or_si512(v_s8_y, vb2);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb1ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb1ptr, bit1ptr + (ir + ib) * NTILE / 8, k_tail * NTILE / 8);
      auto tmpb2ptr = tmp + Unroll * NTILE * 3 / 4;
      memcpy(tmpb2ptr, bit2ptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits((utils::bit2x4*)(tmpb4ptr + i * 32), vmask);
        auto vb1 = unpack_1bits((utils::bit1x8*)(tmpb1ptr + i * 8), zmm_0x00, zmm_0x04);
        auto vb2 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        vb2 = _mm512_slli_epi32(vb2, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_or_si512(v_s8_y, vb2);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8_pack4_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr,
                                                          utils::bit1x8* bit1ptr, int8_t* zpptr, int8_t* dstptr,
                                                          int blocksize, int ldzp, int n_offset, int k_offset, int row,
                                                          int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 16, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b1ptr = bit1ptr + (ir + ib) * NTILE / 8;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_1bits(b1ptr + i * 8, zmm_0x00, zmm_0x04);
        auto vb2 = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        vb2 = _mm512_slli_epi32(vb2, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_or_si512(v_s8_y, vb2);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s7_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                                int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                int k_offset, int row, int col, int8_t* tmp, size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, utils::bit1x8* bit1ptr,
                                    int8_t* zpptr, int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset,
                                    int row, int8_t* tmp, size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s7_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s7_s8_pack2_row<NTILE>;
      }
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s7_s8_pack4_row<NTILE>;
      }
      if (func) {
        int head_end = utils::padto(k_offset, blocksize);
        head_end = std::min(head_end, k_offset + row);
        int head_size = head_end - k_offset;
        if (head_size > 0) {
          (*func)(bit4ptr, bit2ptr, bit1ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, head_size, tmp,
                  tmpsize);
        }
        int body_size = row - head_size;
        if (body_size > 0) {
          (*func)(bit4ptr + head_size * NTILE / 2, bit2ptr + head_size * NTILE / 4, bit1ptr + head_size * NTILE / 8,
                  zpptr, dstptr + head_size * NTILE, blocksize, ldzp, n_offset, head_end, body_size, tmp, tmpsize);
        }
        return BTLA_CODE::Success;
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  } else {
    size_t elesize = static_cast<size_t>(row) * col;
    return decompress_s7_s8(bit4ptr, bit2ptr, bit1ptr, dstptr, elesize, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE decompress_s6_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, int8_t* dstptr,
                                         size_t unpack_elt, int8_t* tmp, size_t tmpsize) {
  int constexpr VBits = 512;
  int constexpr VElt = VBits / 8;
  size_t i = 0;
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  size_t elt_pad = utils::padto_le(unpack_elt, VElt);
  for (; i < elt_pad; i += VElt) {
    auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
    auto vb1 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
    vb1 = _mm512_slli_epi32(vb1, 4);
    vout = _mm512_or_si512(vout, vb1);
    vout = _mm512_sub_epi8(vout, vbias);
    _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
  }
  if (elt_pad < unpack_elt) {
    if (unpack_elt >= VElt) {
      i = unpack_elt - VElt;
      auto vout = unpack_4bits(bit4ptr + i / 2, vmask);
      auto vb1 = unpack_2bits(bit2ptr + i / 4, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
      vb1 = _mm512_slli_epi32(vb1, 4);
      vout = _mm512_or_si512(vout, vb1);
      vout = _mm512_sub_epi8(vout, vbias);
      _mm512_storeu_si512((__m512i*)(dstptr + i), vout);
    } else {
      ref::decompress_s6_s8(bit4ptr + i / 2, bit2ptr + i / 4, dstptr + i, unpack_elt - i, tmp, tmpsize);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8_pack1_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < Unroll; i++) {
      memcpy(tmp + i * NTILE, zptr, NTILE * sizeof(int8_t));
    }
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }

    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb2ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb2ptr, bit2ptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits((utils::bit2x4*)(tmpb4ptr + i * 32), vmask);
        auto vb1 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8_pack2_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 1;
  int constexpr Unroll = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  const auto vindex = _mm512_set_epi8(14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8,
                                      8, 6, 6, 4, 4, 2, 2, 0, 0, 14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0,
                                      14, 14, 12, 12, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    memcpy(tmp, zptr, NTILE * sizeof(int8_t));
    memcpy(tmp + NTILE, zptr, NTILE * sizeof(int8_t));
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi16(tmp + i * 32, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    int k_remain_unrll = utils::padto_le(k_remain, PackRow * Unroll);
    int ib = 0;
    for (; ib < k_remain_unrll; ib += PackRow * Unroll) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
    int k_tail = k_remain - k_remain_unrll;
    if (k_tail > 0) {
      auto tmpb4ptr = tmp;
      memcpy(tmpb4ptr, srcptr + (ir + ib) * NTILE / 2, k_tail * NTILE / 2);
      auto tmpb2ptr = tmp + Unroll * NTILE / 2;
      memcpy(tmpb2ptr, bit2ptr + (ir + ib) * NTILE / 4, k_tail * NTILE / 4);
      auto tmpout = tmp + Unroll * NTILE;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits((utils::bit2x4*)(tmpb4ptr + i * 32), vmask);
        auto vb1 = unpack_2bits((utils::bit2x4*)(tmpb2ptr + i * 16), vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(tmpout + i * 64), v_s8_y);
      }
      memcpy(dstptr + (ir + ib) * NTILE, tmpout, k_tail * NTILE);
    }
  }
  return BTLA_CODE::Success;
}

template <int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8_pack4_row(utils::bit4x2* srcptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                          int8_t* dstptr, int blocksize, int ldzp, int n_offset,
                                                          int k_offset, int row, int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  static_assert((NTILE % VLen) == 0);
  int constexpr PackRow = 4;
  __m512i v_zp_y[NReg];
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  for (int ir = 0; ir < row; ir += blocksize) {
    auto zptr = zpptr + (k_offset + ir) / blocksize * ldzp + n_offset;
    for (int i = 0; i < NReg; i++) {
      v_zp_y[i] = load_zp_epi8_broadcast_epi32(zptr + i * 16, vindex);
      v_zp_y[i] = _mm512_add_epi8(v_zp_y[i], vbias);
    }
    int k_remain = utils::remainsize(ir, row, blocksize);
    for (int ib = 0; ib < k_remain; ib += PackRow) {
      auto b4ptr = srcptr + (ir + ib) * NTILE / 2;
      auto b2ptr = bit2ptr + (ir + ib) * NTILE / 4;
      for (int i = 0; i < NReg; i++) {
        auto v_s8_y = unpack_4bits(b4ptr + i * 32, vmask);
        auto vb1 = unpack_2bits(b2ptr + i * 16, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
        vb1 = _mm512_slli_epi32(vb1, 4);
        v_s8_y = _mm512_or_si512(v_s8_y, vb1);
        v_s8_y = _mm512_sub_epi8(v_s8_y, v_zp_y[i]);
        _mm512_storeu_si512((__m512i*)(dstptr + i * 64 + (ir + ib) * NTILE), v_s8_y);
      }
    }
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE>
static inline BTLA_CODE decompress_kblock_s6_s8(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, int8_t* zpptr,
                                                int8_t* dstptr, int blocksize, int ldzp, int n_offset, int k_offset,
                                                int row, int col, int8_t* tmp, size_t tmpsize) {
  if (zpptr) {
    typedef BTLA_CODE (*decompfunc)(utils::bit4x2* bit4ptr, utils::bit2x4* bit2ptr, int8_t* zpptr, int8_t* dstptr,
                                    int blocksize, int ldzp, int n_offset, int k_offset, int row, int8_t* tmp,
                                    size_t tmpsize);
    decompfunc func = nullptr;
    if (col == NTILE) {
      if constexpr (PackRow == 1) {
        func = &decompress_kblock_s6_s8_pack1_row<NTILE>;
      }
      if constexpr (PackRow == 2) {
        func = &decompress_kblock_s6_s8_pack2_row<NTILE>;
      }
      if constexpr (PackRow == 4) {
        func = &decompress_kblock_s6_s8_pack4_row<NTILE>;
      }
      if (func) {
        int head_end = utils::padto(k_offset, blocksize);
        head_end = std::min(head_end, k_offset + row);
        int head_size = head_end - k_offset;
        if (head_size > 0) {
          (*func)(bit4ptr, bit2ptr, zpptr, dstptr, blocksize, ldzp, n_offset, k_offset, head_size, tmp, tmpsize);
        }
        int body_size = row - head_size;
        if (body_size > 0) {
          (*func)(bit4ptr + head_size * NTILE / 2, bit2ptr + head_size * NTILE / 4, zpptr, dstptr + head_size * NTILE,
                  blocksize, ldzp, n_offset, head_end, body_size, tmp, tmpsize);
        }
        return BTLA_CODE::Success;
      }
    }
    assert(0);
    return BTLA_CODE::NotSupport;
  } else {
    size_t elesize = static_cast<size_t>(row) * col;
    return decompress_s6_s8(bit4ptr, bit2ptr, dstptr, elesize, tmp, tmpsize);
  }
  return BTLA_CODE::Success;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s8_fp_row(int8_t* srcptr, DST_T* dstptr, int row, void* scales_, BTLA_DTYPE sdtype,
                                             int8_t* zero_points, int k_offset, int n_offset, int blocksize, int ldzp,
                                             int8_t* tmp, size_t tmpsize) {
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  const auto vshuf_index_low = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const auto vshuf_index_high = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);
  if (zero_points == nullptr) {
    for (int ir = 0; ir < row; ir += blocksize) {
      int k_remain = utils::remainsize(ir, row, blocksize);
      int ele_off = (k_offset + ir) / blocksize * ldzp + n_offset;
      if constexpr (PackRow == 1) {
        __m512 vscale_y[NReg];
        if (sdtype == BTLA_DTYPE::F32) {
          auto sptr = (float*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = _mm512_loadu_ps(sptr + i * VLen);
        } else if (sdtype == BTLA_DTYPE::BF16) {
          auto sptr = (utils::bf16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_bf16_fp32(sptr + i * VLen);
        } else if (sdtype == BTLA_DTYPE::F16) {
          auto sptr = (utils::fp16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_fp16_fp32(sptr + i * VLen);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            auto vdeq_y = dequant_s8_fp(b8ptr + i * VLen, vscale_y[i]);
            store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * VLen);
          }
        }
      } else if constexpr (PackRow == 4) {
        __m512 vscale_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m512 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm512_loadu_ps(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::F16) {
            auto sptr = (utils::fp16*)scales_ + ele_off;
            vraw = load_fp16_fp32(sptr + i * VLen);
          } else {
            assert(0);
          }
          auto vcast_y = broadcast_ps_1_2<true>(vraw, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_high, vshuf_index_low);
          vcast_y = broadcast_ps_1_2<false>(vraw, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 2] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 3] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_high, vshuf_index_low);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp(b8ptr + i * VLen * PackRow + ip * VLen, vscale_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * VLen * PackRow + ip * VLen);
            }
          }
        }
      } else if constexpr (PackRow == 2) {
        __m512 vscale_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m512 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm512_loadu_ps(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::F16) {
            auto sptr = (utils::fp16*)scales_ + ele_off;
            vraw = load_fp16_fp32(sptr + i * VLen);
          }
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vraw, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vraw, vshuf_index_high, vshuf_index_low);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp(b8ptr + i * VLen * PackRow + ip * VLen, vscale_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * VLen * PackRow + ip * VLen);
            }
          }
        }
      } else {
        assert(0);
      }
    }
    return BTLA_CODE::Success;
  } else {
    for (int ir = 0; ir < row; ir += blocksize) {
      int k_remain = utils::remainsize(ir, row, blocksize);
      int ele_off = (k_offset + ir) / blocksize * ldzp + n_offset;
      if constexpr (PackRow == 1) {
        __m512 vscale_y[NReg];
        if (sdtype == BTLA_DTYPE::F32) {
          auto sptr = (float*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = _mm512_loadu_ps(sptr + i * VLen);
        } else if (sdtype == BTLA_DTYPE::BF16) {
          auto sptr = (utils::bf16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_bf16_fp32(sptr + i * VLen);
        } else if (sdtype == BTLA_DTYPE::F16) {
          auto sptr = (utils::fp16*)scales_ + ele_off;
          for (int i = 0; i < NReg; i++) vscale_y[i] = load_fp16_fp32(sptr + i * VLen);
        }
        __m512i vzp_y[NReg];
        for (int i = 0; i < NReg; i++) vzp_y[i] = load_s8_s32(zero_points + ele_off + i * VLen);
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            auto vdeq_y = dequant_s8_fp<true>(b8ptr + i * VLen, vscale_y[i], vzp_y[i]);
            store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * VLen);
          }
        }
      } else if constexpr (PackRow == 4) {
        __m512 vscale_y[PackRow * NReg];
        __m512i vzp_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m512 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm512_loadu_ps(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::F16) {
            auto sptr = (utils::fp16*)scales_ + ele_off;
            vraw = load_fp16_fp32(sptr + i * VLen);
          } else {
            assert(0);
          }
          auto vcast_y = broadcast_ps_1_2<true>(vraw, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_high, vshuf_index_low);
          vcast_y = broadcast_ps_1_2<false>(vraw, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 2] = broadcast_ps_1_2<true>(vcast_y, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 3] = broadcast_ps_1_2<false>(vcast_y, vshuf_index_high, vshuf_index_low);

          auto tmp = load_s8_s32(zero_points + ele_off + i * VLen);
          auto vcasti_y = broadcast_epi32_1_2<true>(tmp, vshuf_index_high, vshuf_index_low);
          vzp_y[i * PackRow + 0] = broadcast_epi32_1_2<true>(vcasti_y, vshuf_index_high, vshuf_index_low);
          vzp_y[i * PackRow + 1] = broadcast_epi32_1_2<false>(vcasti_y, vshuf_index_high, vshuf_index_low);
          vcasti_y = broadcast_epi32_1_2<false>(tmp, vshuf_index_high, vshuf_index_low);
          vzp_y[i * PackRow + 2] = broadcast_epi32_1_2<true>(vcasti_y, vshuf_index_high, vshuf_index_low);
          vzp_y[i * PackRow + 3] = broadcast_epi32_1_2<false>(vcasti_y, vshuf_index_high, vshuf_index_low);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp<true>(b8ptr + i * VLen * PackRow + ip * VLen, vscale_y[i * PackRow + ip],
                                                vzp_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * VLen * PackRow + ip * VLen);
            }
          }
        }
      } else if constexpr (PackRow == 2) {
        __m512 vscale_y[PackRow * NReg];
        __m512i vzp_y[PackRow * NReg];
        for (int i = 0; i < NReg; i++) {
          __m512 vraw;
          if (sdtype == BTLA_DTYPE::F32) {
            auto sptr = (float*)scales_ + ele_off;
            vraw = _mm512_loadu_ps(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::BF16) {
            auto sptr = (utils::bf16*)scales_ + ele_off;
            vraw = load_bf16_fp32(sptr + i * VLen);
          } else if (sdtype == BTLA_DTYPE::F16) {
            auto sptr = (utils::fp16*)scales_ + ele_off;
            vraw = load_fp16_fp32(sptr + i * VLen);
          }
          vscale_y[i * PackRow + 0] = broadcast_ps_1_2<true>(vraw, vshuf_index_high, vshuf_index_low);
          vscale_y[i * PackRow + 1] = broadcast_ps_1_2<false>(vraw, vshuf_index_high, vshuf_index_low);
          auto tmp = load_s8_s32(zero_points + ele_off + i * VLen);
          vzp_y[i * PackRow + 0] = broadcast_epi32_1_2<true>(tmp, vshuf_index_high, vshuf_index_low);
          vzp_y[i * PackRow + 1] = broadcast_epi32_1_2<false>(tmp, vshuf_index_high, vshuf_index_low);
        }
        for (int ib = 0; ib < k_remain; ib += PackRow) {
          auto b8ptr = srcptr + (ir + ib) * NTILE;
          for (int i = 0; i < NReg; i++) {
            for (int ip = 0; ip < PackRow; ip++) {
              auto vdeq_y = dequant_s8_fp<true>(b8ptr + i * VLen * PackRow + ip * VLen, vscale_y[i * PackRow + ip],
                                                vzp_y[i * PackRow + ip]);
              store_fp32_T(vdeq_y, dstptr + (ir + ib) * NTILE + i * VLen * PackRow + ip * VLen);
            }
          }
        }
      } else {
        assert(0);
      }
    }
    return BTLA_CODE::Success;
  }
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s8_fp(int8_t* srcptr, DST_T* dstptr, int row, int col, void* scales_,
                                         BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                         int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(srcptr, dstptr, head_size, scales_, sdtype, zero_points,
                                                         k_offset, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(srcptr + head_size * NTILE, dstptr + head_size * NTILE,
                                                         body_size, scales_, sdtype, zero_points, head_end, n_offset,
                                                         blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}
template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s4_fp_row(utils::int4x2* srcptr, DST_T* dstptr, int row, void* scales_,
                                             BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                             int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  auto ret = decompress_kblock_s4_s8<PackRow, NTILE>(srcptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset,
                                                     row, NTILE, tmp, tmpsize);
  assert(ret == BTLA_CODE::Success);
  return decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(tmps8ptr, dstptr, row, scales_, sdtype, nullptr, k_offset,
                                                            n_offset, blocksize, ldzp, tmp, tmpsize);
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s4_fp(utils::int4x2* srcptr, DST_T* dstptr, int row, int col, void* scales_,
                                         BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                         int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s4_fp_row<PackRow, NTILE, DST_T>(srcptr, dstptr, head_size, scales_, sdtype, zero_points,
                                                         k_offset, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s4_fp_row<PackRow, NTILE, DST_T>(srcptr + head_size * NTILE / 2, dstptr + head_size * NTILE,
                                                         body_size, scales_, sdtype, zero_points, head_end, n_offset,
                                                         blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s2_fp_row(utils::bit2x4* b2ptr, DST_T* dstptr, int row, void* scales_,
                                             BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                             int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  auto ret = decompress_kblock_s2_s8<PackRow, NTILE>(b2ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset,
                                                     row, NTILE, tmp, tmpsize);
  assert(ret == BTLA_CODE::Success);
  return decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(tmps8ptr, dstptr, row, scales_, sdtype, nullptr, k_offset,
                                                            n_offset, blocksize, ldzp, tmp, tmpsize);
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s2_fp(utils::bit2x4* b2ptr, DST_T* dstptr, int row, int col, void* scales_,
                                         BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                         int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s2_fp_row<PackRow, NTILE, DST_T>(b2ptr, dstptr, head_size, scales_, sdtype, zero_points,
                                                         k_offset, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s2_fp_row<PackRow, NTILE, DST_T>(b2ptr + head_size * NTILE / 4, dstptr + head_size * NTILE,
                                                         body_size, scales_, sdtype, zero_points, head_end, n_offset,
                                                         blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s3_fp_row(utils::bit2x4* b2ptr, utils::bit1x8* b1ptr, DST_T* dstptr, int row,
                                             void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset,
                                             int n_offset, int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  auto ret = decompress_kblock_s3_s8<PackRow, NTILE>(b2ptr, b1ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset,
                                                     k_offset, row, NTILE, tmp, tmpsize);
  assert(ret == BTLA_CODE::Success);
  return decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(tmps8ptr, dstptr, row, scales_, sdtype, nullptr, k_offset,
                                                            n_offset, blocksize, ldzp, tmp, tmpsize);
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s3_fp(utils::bit2x4* b2ptr, utils::bit1x8* b1ptr, DST_T* dstptr, int row, int col,
                                         void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset,
                                         int n_offset, int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s3_fp_row<PackRow, NTILE, DST_T>(b2ptr, b1ptr, dstptr, head_size, scales_, sdtype, zero_points,
                                                         k_offset, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s3_fp_row<PackRow, NTILE, DST_T>(
          b2ptr + head_size * NTILE / 4, b1ptr + head_size * NTILE / 8, dstptr + head_size * NTILE, body_size, scales_,
          sdtype, zero_points, head_end, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s1_fp_row(utils::bit1x8* b1ptr, DST_T* dstptr, int row, void* scales_,
                                             BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                             int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  auto ret = decompress_kblock_s1_s8<PackRow, NTILE>(b1ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset, k_offset,
                                                     row, NTILE, tmp, tmpsize);
  assert(ret == BTLA_CODE::Success);
  return decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(tmps8ptr, dstptr, row, scales_, sdtype, nullptr, k_offset,
                                                            n_offset, blocksize, ldzp, tmp, tmpsize);
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s1_fp(utils::bit1x8* b1ptr, DST_T* dstptr, int row, int col, void* scales_,
                                         BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset, int n_offset,
                                         int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s1_fp_row<PackRow, NTILE, DST_T>(b1ptr, dstptr, head_size, scales_, sdtype, zero_points,
                                                         k_offset, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s1_fp_row<PackRow, NTILE, DST_T>(b1ptr + head_size * NTILE / 8, dstptr + head_size * NTILE,
                                                         body_size, scales_, sdtype, zero_points, head_end, n_offset,
                                                         blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s5_fp_row(utils::bit4x2* b4ptr, utils::bit1x8* b1ptr, DST_T* dstptr, int row,
                                             void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset,
                                             int n_offset, int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  auto ret = decompress_kblock_s5_s8<PackRow, NTILE>(b4ptr, b1ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset,
                                                     k_offset, row, NTILE, tmp, tmpsize);
  assert(ret == BTLA_CODE::Success);
  return decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(tmps8ptr, dstptr, row, scales_, sdtype, nullptr, k_offset,
                                                            n_offset, blocksize, ldzp, tmp, tmpsize);
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s5_fp(utils::bit4x2* b4ptr, utils::bit1x8* b1ptr, DST_T* dstptr, int row, int col,
                                         void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset,
                                         int n_offset, int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s5_fp_row<PackRow, NTILE, DST_T>(b4ptr, b1ptr, dstptr, head_size, scales_, sdtype, zero_points,
                                                         k_offset, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s5_fp_row<PackRow, NTILE, DST_T>(
          b4ptr + head_size * NTILE / 2, b1ptr + head_size * NTILE / 8, dstptr + head_size * NTILE, body_size, scales_,
          sdtype, zero_points, head_end, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s6_fp_row(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, DST_T* dstptr, int row,
                                             void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset,
                                             int n_offset, int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  auto ret = decompress_kblock_s6_s8<PackRow, NTILE>(b4ptr, b2ptr, zero_points, tmps8ptr, blocksize, ldzp, n_offset,
                                                     k_offset, row, NTILE, tmp, tmpsize);
  assert(ret == BTLA_CODE::Success);
  return decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(tmps8ptr, dstptr, row, scales_, sdtype, nullptr, k_offset,
                                                            n_offset, blocksize, ldzp, tmp, tmpsize);
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s6_fp(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, DST_T* dstptr, int row, int col,
                                         void* scales_, BTLA_DTYPE sdtype, int8_t* zero_points, int k_offset,
                                         int n_offset, int blocksize, int ldzp, int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s6_fp_row<PackRow, NTILE, DST_T>(b4ptr, b2ptr, dstptr, head_size, scales_, sdtype, zero_points,
                                                         k_offset, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s6_fp_row<PackRow, NTILE, DST_T>(
          b4ptr + head_size * NTILE / 2, b2ptr + head_size * NTILE / 4, dstptr + head_size * NTILE, body_size, scales_,
          sdtype, zero_points, head_end, n_offset, blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s7_fp_row(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, utils::bit1x8* b1ptr,
                                             DST_T* dstptr, int row, void* scales_, BTLA_DTYPE sdtype,
                                             int8_t* zero_points, int k_offset, int n_offset, int blocksize, int ldzp,
                                             int8_t* tmp, size_t tmpsize) {
  int constexpr NReg = NTILE / 8;
  const auto DstSize = row * NTILE * sizeof(DST_T);
  const auto S8Size = row * NTILE * sizeof(int8_t);
  auto tmps8ptr = (int8_t*)dstptr;
  tmps8ptr += DstSize - S8Size;
  auto ret = decompress_kblock_s7_s8<PackRow, NTILE>(b4ptr, b2ptr, b1ptr, zero_points, tmps8ptr, blocksize, ldzp,
                                                     n_offset, k_offset, row, NTILE, tmp, tmpsize);
  assert(ret == BTLA_CODE::Success);
  return decompress_kblock_s8_fp_row<PackRow, NTILE, DST_T>(tmps8ptr, dstptr, row, scales_, sdtype, nullptr, k_offset,
                                                            n_offset, blocksize, ldzp, tmp, tmpsize);
}

template <int PackRow, int NTILE, typename DST_T>
inline BTLA_CODE decompress_kblock_s7_fp(utils::bit4x2* b4ptr, utils::bit2x4* b2ptr, utils::bit1x8* b1ptr,
                                         DST_T* dstptr, int row, int col, void* scales_, BTLA_DTYPE sdtype,
                                         int8_t* zero_points, int k_offset, int n_offset, int blocksize, int ldzp,
                                         int8_t* tmp, size_t tmpsize) {
  auto ret = BTLA_CODE::NotSupport;
  if (col == NTILE) {
    int head_end = utils::padto(k_offset, blocksize);
    head_end = std::min(head_end, k_offset + row);
    int head_size = head_end - k_offset;
    if (head_size > 0) {
      decompress_kblock_s7_fp_row<PackRow, NTILE, DST_T>(b4ptr, b2ptr, b1ptr, dstptr, head_size, scales_, sdtype,
                                                         zero_points, k_offset, n_offset, blocksize, ldzp, tmp,
                                                         tmpsize);
    }
    int body_size = row - head_size;
    if (body_size > 0) {
      decompress_kblock_s7_fp_row<PackRow, NTILE, DST_T>(b4ptr + head_size * NTILE / 2, b2ptr + head_size * NTILE / 4,
                                                         b1ptr + head_size * NTILE / 8, dstptr + head_size * NTILE,
                                                         body_size, scales_, sdtype, zero_points, head_end, n_offset,
                                                         blocksize, ldzp, tmp, tmpsize);
    }
    return BTLA_CODE::Success;
  }
  return ret;
}

template <typename ScaleT, int NReg, int MTILE>
static inline void gemv_dequant_s32fp32(const float* asptr, int ldzp, const ScaleT* bsptr, __m512i* iacc,
                                        __m512* facc) {
  __m512 v_a_scale[MTILE];
  for (int im = 0; im < MTILE; im++) {
    v_a_scale[im] = _mm512_set1_ps(*(asptr + im * ldzp));
  }

  for (int i = 0; i < NReg; i++) {
    __m512 v_b_scale = load_T_fp32(bsptr + i * 16);
    for (int im = 0; im < MTILE; im++) {
      auto vtmp = _mm512_mul_ps(v_a_scale[im], v_b_scale);
      auto tmp = _mm512_cvtepi32_ps(iacc[im * NReg + i]);
      facc[im * NReg + i] = _mm512_fmadd_ps(tmp, vtmp, facc[im * NReg + i]);
    }
  }
}

template <int NReg, int MReg>
static inline void gemv_remove_zp(const uint8_t* azptr, int ldzp, __m512i* iacc, __m512i* bacc) {
  if constexpr (MReg == 1) {
    auto zp = int(azptr[0]);
    __m512i v_a_zp = _mm512_set1_epi32(zp);
    for (int in = 0; in < NReg; in++) {
      auto vtmp = _mm512_mullo_epi32(v_a_zp, bacc[in]);
      iacc[in] = _mm512_sub_epi32(iacc[in], vtmp);
    }
  } else {
    __m512i v_a_zp[MReg];
    for (int im = 0; im < MReg; im++) {
      auto zp = int(azptr[im * ldzp]);
      v_a_zp[im] = _mm512_set1_epi32(zp);
      for (int in = 0; in < NReg; in++) {
        auto vtmp = _mm512_mullo_epi32(v_a_zp[im], bacc[in]);
        iacc[im * NReg + in] = _mm512_sub_epi32(iacc[im * NReg + in], vtmp);
      }
    }
  }
}

template <int MTILE, int NReg, int Unroll>
static inline void accumulate_fp32_s8_fp32(const float* Aptr, int lda, int8_t* Bptr, __m512* vacc, __m512* vsca) {
  if constexpr (MTILE == 1) {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m512 va = _mm512_set1_ps(*(Aptr + ikk));
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 16 + ikk * NReg * 16);
        ftmp = _mm512_mul_ps(ftmp, vsca[i]);
        vacc[i] = _mm512_fmadd_ps(va, ftmp, vacc[i]);
      }
    }
  } else {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m512 va[MTILE];
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 16 + ikk * NReg * 16);
        ftmp = _mm512_mul_ps(ftmp, vsca[i]);
        for (int im = 0; im < MTILE; im++) {
          if (i == 0) {
            va[im] = _mm512_set1_ps(*(Aptr + ikk + im * lda));
          }
          vacc[im * NReg + i] = _mm512_fmadd_ps(va[im], ftmp, vacc[im * NReg + i]);
        }
      }
    }
  }
}

template <int MTILE, int NReg, int Unroll>
static inline void accumulate_fp32_s8_fp32(const float* Aptr, int lda, int8_t* Bptr, __m512* vacc_loc) {
  if constexpr (MTILE == 1) {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m512 va = _mm512_set1_ps(*(Aptr + ikk));
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 16 + ikk * NReg * 16);
        vacc_loc[i] = _mm512_fmadd_ps(va, ftmp, vacc_loc[i]);
      }
    }
  } else {
    for (int ikk = 0; ikk < Unroll; ikk++) {
      __m512 va[MTILE];
      for (int i = 0; i < NReg; i++) {
        auto ftmp = load_s8_fp32(Bptr + i * 16 + ikk * NReg * 16);
        for (int im = 0; im < MTILE; im++) {
          if (i == 0) {
            va[im] = _mm512_set1_ps(*(Aptr + ikk + im * lda));
          }
          vacc_loc[im * NReg + i] = _mm512_fmadd_ps(va[im], ftmp, vacc_loc[im * NReg + i]);
        }
      }
    }
  }
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_4bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto& b4ptr = B.b4ptr;
  int blks = k / blocksize;
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (4 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;
    __m512 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * VLen);
    }

    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;

      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc, v_b_scale);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
          vb = _mm512_sub_epi8(vb, vbias);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc, v_b_scale);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = (utils::bit2x4*)B.b2ptr;
  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (2 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m512 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm512_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b2ptr += VLen * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb = _mm512_sub_epi8(vb, vbias);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b2ptr += VLen * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m512 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * VLen);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm512_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_3bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = (utils::bit2x4*)B.b2ptr;
  auto b1ptr = (utils::bit1x8*)B.b1ptr;

  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (3 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m512 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm512_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b2ptr += VLen * Unroll / 4;
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, vbias);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b2ptr += VLen * Unroll / 4;
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m512 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * VLen);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm512_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_1bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b1ptr = (utils::bit1x8*)B.b1ptr;

  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m512 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm512_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb1 = _mm512_srli_epi32(vb1, 2);
          vb1 = _mm512_sub_epi8(vb1, bzp[i]);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb1);
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb1 = _mm512_srli_epi32(vb1, 2);
          vb1 = _mm512_sub_epi8(vb1, vbias);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb1);
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m512 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * VLen);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm512_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_5bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = (utils::bit4x2*)B.b4ptr;
  auto b1ptr = (utils::bit1x8*)B.b1ptr;

  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m512 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm512_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb1 = _mm512_slli_epi32(vb1, 2);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b4ptr += VLen * Unroll / 2;
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          vb1 = _mm512_slli_epi32(vb1, 2);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, vbias);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b4ptr += VLen * Unroll / 2;
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m512 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * VLen);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm512_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_6bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = (utils::bit4x2*)B.b4ptr;
  auto b2ptr = (utils::bit2x4*)B.b2ptr;

  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m512 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm512_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm512_slli_epi32(vb1, 4);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b4ptr += VLen * Unroll / 2;
          b2ptr += VLen * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm512_slli_epi32(vb1, 4);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_sub_epi8(vb, vbias);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b4ptr += VLen * Unroll / 2;
          b2ptr += VLen * Unroll / 4;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m512 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * VLen);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm512_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_7bit_fp32_fp32(const float* A, int lda, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = (utils::bit4x2*)B.b4ptr;
  auto b2ptr = (utils::bit2x4*)B.b2ptr;
  auto b1ptr = (utils::bit1x8*)B.b1ptr;

  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  int constexpr KTILE = 1;
  for (int ib = 0; ib < blks; ib += 1) {
    auto bsptr = B.sptr + ib * B.ldzp;

    __m512 acc_loc[NReg * MReg];
    for (int i = 0; i < NReg * MReg; i++) {
      acc_loc[i] = _mm512_setzero_ps();
    }
    int constexpr Unroll = 4;
    assert((blocksize % 4) == 0);
    assert(tmpsize >= NTILE * Unroll);

    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < Unroll; i++) {
        memcpy(tmp + i * NTILE, bzptr, NTILE);
      }
      for (int i = 0; i < NReg; i++) {
        bzp[i] = _mm512_loadu_si512((const __m512i*)(tmp + i * 64));
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm512_slli_epi32(vb1, 4);
          vb2 = _mm512_slli_epi32(vb2, 4);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_or_si512(vb, vb2);
          vb = _mm512_sub_epi8(vb, bzp[i]);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b4ptr += VLen * Unroll / 2;
          b2ptr += VLen * Unroll / 4;
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }

    } else {
      for (int ik = 0; ik < blocksize; ik += Unroll) {
        for (int i = 0; i < NReg; i++) {
          auto vb = unpack_4bits(b4ptr, vmask);
          auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
          auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
          vb1 = _mm512_slli_epi32(vb1, 4);
          vb2 = _mm512_slli_epi32(vb2, 4);
          vb = _mm512_or_si512(vb, vb1);
          vb = _mm512_or_si512(vb, vb2);
          vb = _mm512_sub_epi8(vb, vbias);
          _mm512_storeu_si512((__m512i*)(tmp + 64 * i), vb);
          b4ptr += VLen * Unroll / 2;
          b2ptr += VLen * Unroll / 4;
          b1ptr += VLen * Unroll / 8;
        }
        accumulate_fp32_s8_fp32<MTILE, NReg, Unroll>(A + ib * blocksize + ik, lda, tmp, acc_loc);
      }
    }

    __m512 v_b_scale[NReg];
    for (int i = 0; i < NReg; i++) {
      v_b_scale[i] = load_T_fp32(bsptr + i * VLen);
    }
    for (int im = 0; im < MTILE; im++) {
      for (int in = 0; in < NReg; in++) {
        acc[im * NReg + in] = _mm512_fmadd_ps(acc_loc[im * NReg + in], v_b_scale[in], acc[im * NReg + in]);
      }
    }
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

static inline __m512i _mm512_dpbusd_avx512bw_epi32(__m512i& c, const __m512i& a, const __m512i& b) {
  const __m512i dot2 = _mm512_maddubs_epi16(a, b);
  const __m512i ones = _mm512_set1_epi16(1);
  const __m512i sum4 = _mm512_madd_epi16(ones, dot2);
  return _mm512_add_epi32(c, sum4);
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_4bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto& a8ptr = A.aptr;
  auto& b4ptr = B.b4ptr;
  auto& asptr = A.sptr;
  auto& azptr = A.zpptr;
  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  // Initialize accumulator with zeros
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (4 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  const __m512i onesu8 = _mm512_set1_epi8(1);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    __m512i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * VLen, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += 4) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += 4) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(a8ptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits((void*)(b4ptr + i * 32 + (ib * blocksize + ik) * NTILE / 2), vmask);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
          }
        }
      }
    }
    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_2bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  int constexpr VLen = 16;
  int blks = k / blocksize;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (2 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);
  const auto onesu8 = _mm512_set1_epi8(1);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    __m512i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b2ptr += VLen * KTILE / 4;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += VLen * KTILE / 4;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b2ptr += VLen * KTILE / 4;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += VLen * KTILE / 4;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_3bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);

  int blks = k / blocksize;
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (3 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);
  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  const auto onesu8 = _mm512_set1_epi8(1);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    __m512i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_1bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);

  int blks = k / blocksize;
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (1 - 1);
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  const auto onesu8 = _mm512_set1_epi8(1);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    __m512i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_srli_epi32(vb1, 2);
            vb1 = _mm512_sub_epi8(vb1, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb1);
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_srli_epi32(vb1, 2);
            vb1 = _mm512_sub_epi8(vb1, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb1);
            }
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_srli_epi32(vb1, 2);
            vb1 = _mm512_sub_epi8(vb1, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb1);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb1);
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_srli_epi32(vb1, 2);
            vb1 = _mm512_sub_epi8(vb1, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb1);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb1);
            }
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_5bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);

  int blks = k / blocksize;
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (5 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  const auto onesu8 = _mm512_set1_epi8(1);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    __m512i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_slli_epi32(vb1, 2);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b4ptr += VLen * KTILE / 2;
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_slli_epi32(vb1, 2);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += VLen * KTILE / 2;
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_slli_epi32(vb1, 2);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b4ptr += VLen * KTILE / 2;
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            vb1 = _mm512_slli_epi32(vb1, 2);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += VLen * KTILE / 2;
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_6bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);

  int blks = k / blocksize;
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (6 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);
  const auto onesu8 = _mm512_set1_epi8(1);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    __m512i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename ScaleT, int NTILE, int MTILE>
static inline BTLA_CODE gemv_7bit_u8s8_fp32(const utils::GemvParamA& A, const utils::GemvParamB<ScaleT>& B, float* C,
                                            int ldc, int k, int blocksize, int8_t* tmp, size_t tmpsize) {
  auto b4ptr = reinterpret_cast<utils::bit4x2*>(B.b4ptr);
  auto b2ptr = reinterpret_cast<utils::bit2x4*>(B.b2ptr);
  auto b1ptr = reinterpret_cast<utils::bit1x8*>(B.b1ptr);

  int blks = k / blocksize;
  int constexpr VLen = 16;
  int constexpr NReg = NTILE / VLen;
  int constexpr MReg = MTILE;
  __m512 acc[NReg * MReg];
  for (int i = 0; i < NReg * MReg; i++) {
    acc[i] = _mm512_setzero_ps();
  }
  int constexpr FullRange = 1 << (7 - 1);
  uint32_t mask = 0x0f0f0f0f;
  auto vmask = _mm512_set1_epi32(*reinterpret_cast<int*>(&mask));
  auto vbias = _mm512_set1_epi8(FullRange);

  auto zmm_0x04 = _mm512_set1_epi8(0x04);
  auto zmm_0x00 = _mm512_set1_epi8(0x00);
  const auto vindex = _mm512_set_epi8(12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4,
                                      4, 4, 4, 0, 0, 0, 0, 12, 12, 12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0, 12, 12,
                                      12, 12, 8, 8, 8, 8, 4, 4, 4, 4, 0, 0, 0, 0);

  uint64_t mask0 = 0x0303030303030303;
  auto vmask0 = _mm512_set1_epi64(*(int64_t*)&mask0);
  auto vshift_y = _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto vsfhl_mask_y = _mm512_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2,
                                      13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto vorder_y = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
  const auto onesu8 = _mm512_set1_epi8(1);
  int constexpr KTILE = 4;
  for (int ib = 0; ib < blks; ib += 1) {
    __m512i iacc[NReg * MReg];
    __m512i bacc[NReg];
    for (int i = 0; i < NReg * MReg; i++) {
      iacc[i] = _mm512_setzero_si512();
    }
    for (int i = 0; i < NReg; i++) {
      bacc[i] = _mm512_setzero_si512();
    }
    if (B.zpptr) {
      __m512i bzp[NReg];
      auto bzptr = B.zpptr + ib * B.ldzp;
      for (int i = 0; i < NReg; i++) {
        bzp[i] = load_zp_epi8_broadcast_epi32(bzptr + i * 16, vindex);
        bzp[i] = _mm512_add_epi8(bzp[i], vbias);
      }
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb2 = _mm512_slli_epi32(vb2, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_or_si512(vb, vb2);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb2 = _mm512_slli_epi32(vb2, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_or_si512(vb, vb2);
            vb = _mm512_sub_epi8(vb, bzp[i]);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    } else {
      for (int ik = 0; ik < blocksize; ik += KTILE) {
        if constexpr (MTILE == 1) {
          __m512i va = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik));
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb2 = _mm512_slli_epi32(vb2, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_or_si512(vb, vb2);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            iacc[i] = _mm512_dpbusd_avx512bw_epi32(iacc[i], va, vb);
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        } else {
          __m512i va[MReg];
          for (int i = 0; i < MReg; i++) {
            va[i] = _mm512_set1_epi32(*(int*)(A.aptr + ib * blocksize + ik + i * A.lda));
          }
          for (int i = 0; i < NReg; i++) {
            auto vb = unpack_4bits(b4ptr, vmask);
            auto vb1 = unpack_1bits(b1ptr, zmm_0x00, zmm_0x04);
            auto vb2 = unpack_2bits(b2ptr, vshift_y, vmask0, vsfhl_mask_y, vorder_y);
            vb1 = _mm512_slli_epi32(vb1, 4);
            vb2 = _mm512_slli_epi32(vb2, 4);
            vb = _mm512_or_si512(vb, vb1);
            vb = _mm512_or_si512(vb, vb2);
            vb = _mm512_sub_epi8(vb, vbias);
            bacc[i] = _mm512_dpbusd_avx512bw_epi32(bacc[i], onesu8, vb);
            for (int j = 0; j < MReg; j++) {
              iacc[j * NReg + i] = _mm512_dpbusd_avx512bw_epi32(iacc[j * NReg + i], va[j], vb);
            }
            b4ptr += VLen * KTILE / 2;
            b2ptr += VLen * KTILE / 4;
            b1ptr += VLen * KTILE / 8;
          }
        }
      }
    }

    gemv_remove_zp<NReg, MReg>(A.zpptr + ib, A.ldzp, iacc, bacc);
    gemv_dequant_s32fp32<ScaleT, NReg, MTILE>(A.sptr + ib, A.ldzp, B.sptr + ib * B.ldzp, iacc, acc);
  }

  for (int j = 0; j < MReg; j++) {
    for (int i = 0; i < NReg; i++) {
      _mm512_storeu_ps(C + i * VLen + j * ldc, acc[j * NReg + i]);
    }
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE mul(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
  int constexpr VLen = 16;
  size_t velt = utils::padto_le(size, VLen);
  size_t i = 0;
  auto vfunc = [&]() {
    auto v0 = load_T_fp32(src0ptr + i);
    auto v1 = load_T_fp32(src1ptr + i);
    auto out = _mm512_mul_ps(v0, v1);
    store_fp32_T(out, dstptr + i);
  };
  for (; i < velt; i += VLen) vfunc();
  if (i < size) {
    if (size >= VLen) {
      i = size - VLen;
      vfunc();
    } else {
      ref::mul(src0ptr + i, src1ptr + i, dstptr + i, size - i);
    }
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE inv(const T* src0ptr, T* dstptr, size_t size) {
  int constexpr VLen = 16;
  size_t velt = utils::padto_le(size, VLen);
  size_t i = 0;
  auto vfunc = [&]() {
    auto v0 = load_T_fp32(src0ptr + i);
    auto out = _mm512_rcp14_ps(v0);
    store_fp32_T(out, dstptr + i);
  };
  for (; i < velt; i += VLen) vfunc();
  if (i < size) {
    if (size >= VLen) {
      i = size - VLen;
      vfunc();
    } else {
      ref::inv(src0ptr + i, dstptr + i, size - i);
    }
  }
  return BTLA_CODE::Success;
}

template <typename T>
static inline BTLA_CODE add(const T* src0ptr, const T* src1ptr, T* dstptr, size_t size) {
  int constexpr VLen = 16;
  size_t velt = utils::padto_le(size, VLen);
  size_t i = 0;
  auto vfunc = [&]() {
    auto v0 = load_T_fp32(src0ptr + i);
    auto v1 = load_T_fp32(src1ptr + i);
    auto out = _mm512_add_ps(v0, v1);
    store_fp32_T(out, dstptr + i);
  };
  for (; i < velt; i += VLen) vfunc();
  if (i < size) {
    if (size >= VLen) {
      i = size - VLen;
      vfunc();
    } else {
      ref::add(src0ptr + i, src1ptr + i, dstptr + i, size - i);
    }
  }
  return BTLA_CODE::Success;
}

template <bool HAS_ALIBI, bool HAS_TANH>
static inline BTLA_CODE scale_track_max_fp32_fp32(const float* src, const int src_step, float* dst, float* dst_max,
                                                  int ld_dst, const int M_offset, const int N_offset, const int M,
                                                  const int N, float scale, int causal_offset, float _alibi_slope,
                                                  float tanh_scale, int padding_type, void* tmpcache,
                                                  size_t cachesize) {
  static constexpr float seq15[16]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  alignas(32) const uint32_t mask8[9][8]{
      {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
      {0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
      {0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
      {0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
      {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
      {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000, 0x00000000},
      {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000},
      {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000},
      {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
  };
  const auto v_scale = _mm512_set1_ps(scale);
  const auto v_seq15 = _mm512_loadu_ps(seq15);
  const auto alibi_slope = _mm512_set1_ps(_alibi_slope);
  const auto alibi_base = _mm512_mul_ps(alibi_slope, _mm512_add_ps(v_seq15, _mm512_set1_ps(N_offset)));
  const auto alibi_step = _mm512_set1_ps(_alibi_slope * 16);

  for (int i = 0; i < M; ++i) {
    auto alibi_curr = alibi_base;
    const auto N_unmasked = std::min(N, padding_type == 0   ? INT32_MAX
                                        : padding_type == 1 ? i + M_offset - N_offset + causal_offset + 1
                                                            : causal_offset - N_offset);

    const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
    int j = 0;
    auto v_max = _mm512_set1_ps(-INFINITY);
    for (; j < N_unmasked - 15; j += 16) {
      const auto xs = _mm512_fmadd_ps(v_scale, _mm512_loadu_ps(src + i * src_step + j), alibi_curr);
      v_max = _mm512_max_ps(v_max, xs);
      _mm512_storeu_ps(dst + i * ld_dst + j, xs);
      if constexpr (HAS_ALIBI)
        alibi_curr = _mm512_add_ps(alibi_curr, alibi_step);
      else if constexpr (HAS_TANH) {
        float* dst_ = dst + i * ld_dst + j;
        for (int jj = 0; jj < 16; jj++) dst_[jj] = tanh_scale * tanh(dst_[jj]);
      }
    }
    if (j < N_unmasked) {
      const auto xs = _mm512_fmadd_ps(v_scale, _mm512_maskz_loadu_ps(v_mask, src + i * src_step + j), alibi_curr);
      v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
      _mm512_storeu_ps(dst + i * ld_dst + j, xs);
      if constexpr (HAS_ALIBI)
        alibi_curr = _mm512_add_ps(alibi_curr, alibi_step);
      else if constexpr (HAS_TANH) {
        float* dst_ = dst + i * ld_dst + j;
        for (int jj = 0; jj < N_unmasked - j; jj++) dst_[jj] = tanh_scale * tanh(dst_[jj]);
      }
      j += 16;
    }
    dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));
    if constexpr (HAS_TANH) dst_max[i] = tanh_scale * tanh(dst_max[i]);
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE scale_track_max_int32_fp32(const int32_t* src, const int src_step, float* dst, float* dst_max,
                                                   int ld_dst, const int M_offset, const int N_offset, const int M,
                                                   const int N, float scale, int causal_offset, float alibi_slope,
                                                   float tanh_scale, void* tmpcache, size_t cachesize) {
  const auto v_scale = _mm512_set1_ps(scale);
  for (int i = 0; i < M; ++i) {
    const auto N_unmasked = std::min(N, causal_offset < 0 ? INT32_MAX : i + M_offset - N_offset + causal_offset + 1);
    const auto v_mask = _cvtu32_mask16((1U << (N_unmasked % 16)) - 1);
    int j = 0;
    auto v_max = _mm512_set1_ps(-INFINITY);
    for (; j < N_unmasked - 15; j += 16) {
      const auto xs = _mm512_mul_ps(v_scale, _mm512_cvtepi32_ps(_mm512_loadu_si512(src + i * src_step + j)));
      v_max = _mm512_max_ps(v_max, xs);
      _mm512_storeu_ps(dst + i * ld_dst + j, xs);
    }
    if (j < N_unmasked) {
      const auto xs =
          _mm512_mul_ps(v_scale, _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(v_mask, src + i * src_step + j)));
      v_max = _mm512_mask_max_ps(v_max, v_mask, v_max, xs);
      _mm512_storeu_ps(dst + i * ld_dst + j, xs);
      j += 16;
    }
    dst_max[i] = std::max(dst_max[i], _mm512_reduce_max_ps(v_max));
  }
  return BTLA_CODE::Success;
}

static inline BTLA_CODE weight_cvt_bf16_fp32_n48(const utils::bf16* B, int ldb, bool is_padded, float* dst_ptr,
                                                 int dst_step, int k_size, int n_size, int k_offset, int n_offset,
                                                 void* tmpcache, size_t cachesize) {
  assert(is_padded);
  const auto src = B + k_offset * 48 + n_offset * ldb;
  assert(n_size <= 48);
  assert(n_offset % 48 == 0 && k_offset % 2 == 0);
  static const auto mask_hi = _cvtu32_mask32(0xaaaaaaaaU);
  for (int i = 0; i < k_size; i += 2) {
    for (int j = 0; j < n_size; j += 16) {
      const auto cur_src = src + i * 48 + j * 2;
      const auto cur_dst = dst_ptr + i * 48 + j;
      const auto src_lo = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_loadu_si512(cur_src), 16U));
      const auto src_hi = _mm512_castsi512_ps(_mm512_maskz_loadu_epi16(mask_hi, cur_src));
      _mm512_store_ps(cur_dst + 0, src_lo);
      _mm512_store_ps(cur_dst + 48, src_hi);
    }
  }

  return BTLA_CODE::Success;
}

static inline BTLA_CODE inplace_precompute_max_softmax_fp32_fp32(int m_size, int n_size, int n_pad_size, bool is_causal,
                                                                 float* src, float* dst, const float* s_max,
                                                                 float* expsum, int ld_src, int ld_dst) {
  for (int ii = 0; ii < m_size; ++ii) {
    const auto i_src = src + ii * ld_src;
    const auto i_dst = dst + ii * ld_dst;
    const auto curr_n_size = n_size + (is_causal ? ii : 0);
    const auto v_mask = _cvtu32_mask16((1U << (curr_n_size % 16)) - 1);
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
    {  // scale & store
      int jj = 0;
      for (; jj < utils::padto_le(curr_n_size, 16); jj += 16) {
        _mm512_store_ps(i_dst + jj, _mm512_div_ps(_mm512_loadu_ps(i_src + jj), v_sum));
      }
      if (jj < curr_n_size) {
        _mm512_store_ps(i_dst + jj, _mm512_maskz_div_ps(v_mask, _mm512_loadu_ps(i_src + jj), v_sum));
        jj += 16;
      }
      if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(utils::bf16) * (n_pad_size - jj));
    }
  }

  return BTLA_CODE::Success;
}

static inline BTLA_CODE inplace_precompute_max_softmax_fp32_u8(int m_size, int n_size, int n_pad_size, bool is_causal,
                                                               float* src, uint8_t* dst, const float* s_max,
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
    {  // exp & sum
      auto v_sum = _mm512_setzero_ps();
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
    }
    {  // scale & int8
      const auto v_scale = _mm512_set1_ps(UINT8_MAX);
      int jj = 0;
      for (; jj < curr_n_size / 16 * 16; jj += 16) {
        const auto v_softmax = _mm512_mul_ps(_mm512_loadu_ps(i_src + jj), v_scale);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(i_dst + jj), _mm512_cvtusepi32_epi8(_mm512_cvtps_epu32(v_softmax)));
      }
      if (jj < curr_n_size) {
        const auto v_softmax = _mm512_mul_ps(_mm512_loadu_ps(i_src + jj), v_scale);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(i_dst + jj),
                         _mm512_maskz_cvtusepi32_epi8(v_mask, _mm512_cvtps_epu32(v_softmax)));
        jj += 16;
      }
      if (jj < n_pad_size) memset(i_dst + jj, 0, sizeof(uint8_t) * (n_pad_size - jj));
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
}  // namespace avx512f
}  // namespace kernel
}  // namespace bestla
